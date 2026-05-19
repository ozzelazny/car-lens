"""Console script for the Phase 5.2 fine-tune harness.

Invoke via the ``phase5-train`` entry point declared in
``pyproject.toml``::

    phase5-train [--db PATH] [--source stanford_cars[,vmmrdb,...]]
                 [--train-split train] [--val-split test]
                 [--model MobileCLIP-S2] [--pretrained datacompdr]
                 [--device cpu|cuda|mps] [--batch-size N] [--num-workers N]
                 [--epochs N]
                 [--lr-backbone F] [--lr-head F]
                 [--weight-decay F] [--warmup-epochs N]
                 [--label-smoothing F]
                 [--hard-neg-weight F] [--hard-neg-confusion-path PATH]
                 [--resume-checkpoint PATH]
                 [--checkpoint-dir PATH] [--output PATH]
                 [--seed N] [-v]

Trains a 196-way (or whatever the source dictates) classification head
over the MobileCLIP-S2 image encoder with hard-negative-aware
cross-entropy weighting (derived from the Phase 5.1 confusion-pair
JSON). Saves the best checkpoint (by val top-1) to
``models/checkpoints/`` and writes a JSON report.

The ``--source`` flag accepts one or more comma-separated source names
(e.g. ``compcars,vmmrdb,stanford_cars``) so a single training run can
span every dataset the user wants to include.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from car_lense_engine.db import open_db

from .train_classifier import (
    TrainConfig,
    run_training,
    write_report,
)


def _parse_sources(raw: str) -> list[str]:
    """Parse a comma-separated ``--source`` argument into a non-empty list.

    Empty / whitespace-only entries (e.g. from a stray trailing comma)
    are dropped. Raises :class:`ValueError` if the final list is empty.
    """
    items = [s.strip() for s in raw.split(",") if s.strip()]
    if not items:
        raise ValueError("at least one non-empty source is required")
    return items


DEFAULT_DB = Path("db/crawl.sqlite")
DEFAULT_SOURCE = "stanford_cars"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_VAL_SPLIT = "test"
DEFAULT_MODEL = "MobileCLIP-S2"
DEFAULT_PRETRAINED = "datacompdr"
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 2
DEFAULT_EPOCHS = 20
DEFAULT_LR_BACKBONE = 1e-5
DEFAULT_LR_HEAD = 1e-3
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_EPOCHS = 1
DEFAULT_LABEL_SMOOTHING = 0.1
DEFAULT_HARD_NEG_WEIGHT = 2.0
DEFAULT_HARD_NEG_CONFUSION_PATH = Path("reports/phase5_baseline.json")
DEFAULT_CHECKPOINT_DIR = Path("models/checkpoints")
DEFAULT_OUTPUT = Path("reports/phase5_train.json")
DEFAULT_SEED = 42
DEVICE_CHOICES: tuple[str, ...] = ("cpu", "cuda", "mps")

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser for the ``phase5-train`` command."""
    parser = argparse.ArgumentParser(
        prog="phase5-train",
        description=(
            "Phase 5.2 fine-tune: full backbone fine-tune of MobileCLIP-S2 "
            "with a linear classification head and hard-negative-aware CE "
            "weighting. Writes the best checkpoint (by val top-1) to the "
            "configured checkpoint directory and a JSON report to disk."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"path to the crawler SQLite DB (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE,
        help=(
            "one or more comma-separated ``listings.source`` values to "
            f"train on (default: {DEFAULT_SOURCE}). Examples: "
            "`stanford_cars`, `compcars,vmmrdb`, "
            "`compcars,vmmrdb,stanford_cars`."
        ),
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default=DEFAULT_TRAIN_SPLIT,
        help=f"train split (default: {DEFAULT_TRAIN_SPLIT})",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default=DEFAULT_VAL_SPLIT,
        help=f"val split (default: {DEFAULT_VAL_SPLIT})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenCLIP model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=DEFAULT_PRETRAINED,
        help=f"OpenCLIP pretrained tag (default: {DEFAULT_PRETRAINED})",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=DEVICE_CHOICES,
        default=DEFAULT_DEVICE,
        help=f"torch device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"images per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=(
            "DataLoader worker processes (default: "
            f"{DEFAULT_NUM_WORKERS}; use 0 on Windows/WSL if you hit "
            "multiprocessing issues)"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=DEFAULT_LR_BACKBONE,
        help=f"LR for the pre-trained backbone (default: {DEFAULT_LR_BACKBONE})",
    )
    parser.add_argument(
        "--lr-head",
        type=float,
        default=DEFAULT_LR_HEAD,
        help=f"LR for the new linear head (default: {DEFAULT_LR_HEAD})",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help=f"AdamW weight decay (default: {DEFAULT_WEIGHT_DECAY})",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=DEFAULT_WARMUP_EPOCHS,
        help=f"linear-warmup epochs (default: {DEFAULT_WARMUP_EPOCHS})",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=DEFAULT_LABEL_SMOOTHING,
        help=f"label smoothing (default: {DEFAULT_LABEL_SMOOTHING})",
    )
    parser.add_argument(
        "--hard-neg-weight",
        type=float,
        default=DEFAULT_HARD_NEG_WEIGHT,
        help=(
            "multiplicative CE weight for classes appearing in Phase 5.1 "
            f"confusion pairs (default: {DEFAULT_HARD_NEG_WEIGHT}; pass 1.0 "
            "to disable)"
        ),
    )
    parser.add_argument(
        "--hard-neg-confusion-path",
        type=Path,
        default=DEFAULT_HARD_NEG_CONFUSION_PATH,
        help=(
            "path to the Phase 5.1 baseline JSON whose confusion_top_pairs "
            f"drives hard-negative weighting (default: "
            f"{DEFAULT_HARD_NEG_CONFUSION_PATH})"
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "optional path to a Phase 5.2 fine-tuned checkpoint .pt file; "
            "when set, the image-encoder weights from the checkpoint are "
            "overlaid on top of the pretrained backbone before training "
            "starts (the classification head is always reinitialised "
            "fresh because the class-id space may differ). Resumed runs "
            "tag the saved checkpoint filename with ``_resumed`` so they "
            "do not clobber the source checkpoint (default: none)"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"directory for checkpoint .pt files (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"path for the JSON training report (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``phase5-train`` console script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.batch_size <= 0:
        parser.error(f"--batch-size must be > 0, got {args.batch_size}")
    if args.num_workers < 0:
        parser.error(f"--num-workers must be >= 0, got {args.num_workers}")
    if args.epochs < 0:
        parser.error(f"--epochs must be >= 0, got {args.epochs}")
    if args.hard_neg_weight <= 0:
        parser.error(f"--hard-neg-weight must be > 0, got {args.hard_neg_weight}")
    if args.warmup_epochs < 0:
        parser.error(f"--warmup-epochs must be >= 0, got {args.warmup_epochs}")

    db_path: Path = args.db
    if not db_path.exists():
        parser.error(f"DB path does not exist: {db_path}")

    resume_checkpoint: Path | None = args.resume_checkpoint
    if resume_checkpoint is not None and not resume_checkpoint.exists():
        parser.error(f"--resume-checkpoint path does not exist: {resume_checkpoint}")

    try:
        sources = _parse_sources(args.source)
    except ValueError as exc:
        parser.error(f"--source: {exc}")

    # Resolve the confusion-pair file: if a path was passed but doesn't
    # exist, log a warning and disable hard-neg weighting (the underlying
    # builder already tolerates None; we surface the situation here so
    # the user notices before training starts).
    hn_path: Path | None = args.hard_neg_confusion_path
    if hn_path is not None and not Path(hn_path).exists():
        logger.warning(
            "phase5-train: hard-negative confusion file %s not found -- "
            "training will proceed with uniform class weights",
            hn_path,
        )
        hn_path = None

    config = TrainConfig(
        model_name=args.model,
        pretrained=args.pretrained,
        source=sources,
        train_split=args.train_split,
        val_split=args.val_split,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        hard_neg_weight=args.hard_neg_weight,
        hard_neg_confusion_path=hn_path,
        resume_checkpoint=resume_checkpoint,
        seed=args.seed,
    )

    conn = open_db(db_path)
    try:
        report = run_training(
            conn=conn,
            config=config,
            checkpoint_dir=args.checkpoint_dir,
        )
    finally:
        conn.close()

    write_report(report, args.output)
    print(
        f"phase5-train: best_top1={report.best_val_top1:.4f} "
        f"best_top5={report.best_val_top5:.4f} "
        f"best_epoch={report.best_epoch} "
        f"n_classes={report.n_classes} "
        f"n_train={report.n_train} "
        f"n_val={report.n_val} "
        f"elapsed={report.total_elapsed_s:.1f}s"
    )
    if report.checkpoint_path:
        print(f"phase5-train: best checkpoint -> {report.checkpoint_path}")
    print(f"phase5-train: report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
