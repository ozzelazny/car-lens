"""Console script for the Phase 5.3 view-classifier head trainer.

Invoke via the ``train-view-classifier`` entry point declared in
``pyproject.toml``::

    train-view-classifier [--db PATH]
                          [--backbone-checkpoint PATH|none]
                          [--epochs N] [--lr F]
                          [--batch-size N] [--backbone-batch-size N]
                          [--device cpu|cuda|mps]
                          [--min-view-score F]
                          [--head-arch linear|mlp]
                          [--seed N]
                          [--output PATH]
                          [--report PATH] [-v]

Trains a small classification head (6 classes: ``front``, ``rear``,
``side``, ``three-quarter-front``, ``three-quarter-rear``,
``non-exterior``) on top of a frozen MobileCLIP-S2 image encoder. The
backbone is loaded from the optional fine-tuned checkpoint (default:
the production Phase 5.2 checkpoint); passing ``none`` falls back to
the raw ``datacompdr`` pretrained weights.

The output ``.pt`` file is self-contained: it carries both the head's
state dict AND a copy of the backbone state dict so the
:mod:`recognize_api` service can load one file and run inference end
to end without separately resolving the backbone path.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from car_lense_engine.db import open_db

from .view_classifier import (
    ViewClassifierConfig,
    train_view_classifier,
    write_report,
)

DEFAULT_DB = Path("db/crawl.sqlite")
DEFAULT_BACKBONE_CHECKPOINT = Path("models/checkpoints/mobileclip_s2_compcars_epoch09_top1_91.9.pt")
DEFAULT_MODEL = "MobileCLIP-S2"
DEFAULT_PRETRAINED = "datacompdr"
DEFAULT_DEVICE = "cpu"
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 1024
DEFAULT_BACKBONE_BATCH_SIZE = 128
DEFAULT_MIN_VIEW_SCORE = 0.6
DEFAULT_HEAD_ARCH = "linear"
DEFAULT_SEED = 42
DEFAULT_OUTPUT = Path("models/checkpoints/view_classifier_v1.pt")
DEFAULT_REPORT = Path("reports/phase5_3_view_classifier.json")
DEFAULT_BINARY_OUTPUT = Path("models/checkpoints/exterior_classifier_v1.pt")
DEFAULT_BINARY_REPORT = Path("reports/phase5_3_exterior_classifier.json")
DEVICE_CHOICES: tuple[str, ...] = ("cpu", "cuda", "mps")
HEAD_ARCH_CHOICES: tuple[str, ...] = ("linear", "mlp")

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser for the ``train-view-classifier`` command."""
    parser = argparse.ArgumentParser(
        prog="train-view-classifier",
        description=(
            "Phase 5.3 view-classifier trainer: train a 6-way head "
            "(front / rear / side / three-quarter-front / "
            "three-quarter-rear / non-exterior) on top of a frozen "
            "MobileCLIP-S2 image encoder. Writes a self-contained .pt "
            "checkpoint and a JSON training report."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"path to the crawler SQLite DB (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        default=str(DEFAULT_BACKBONE_CHECKPOINT),
        help=(
            "path to a Phase 5.2 fine-tuned MobileCLIP checkpoint to use "
            "as the frozen backbone; pass 'none' to use the raw "
            f"{DEFAULT_PRETRAINED!r} weights (default: "
            f"{DEFAULT_BACKBONE_CHECKPOINT})"
        ),
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
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"AdamW learning rate for the head (default: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(f"head-training batch size over cached features (default: {DEFAULT_BATCH_SIZE})"),
    )
    parser.add_argument(
        "--backbone-batch-size",
        type=int,
        default=DEFAULT_BACKBONE_BATCH_SIZE,
        help=(
            "feature-cache batch size for the backbone forward pass "
            f"(default: {DEFAULT_BACKBONE_BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=DEVICE_CHOICES,
        default=DEFAULT_DEVICE,
        help=f"torch device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--min-view-score",
        type=float,
        default=DEFAULT_MIN_VIEW_SCORE,
        help=(
            "drop rows whose view_score is below this threshold (default: "
            f"{DEFAULT_MIN_VIEW_SCORE})"
        ),
    )
    parser.add_argument(
        "--head-arch",
        type=str,
        choices=HEAD_ARCH_CHOICES,
        default=DEFAULT_HEAD_ARCH,
        help=f"head architecture (default: {DEFAULT_HEAD_ARCH})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "path for the view-classifier .pt checkpoint "
            f"(default: {DEFAULT_OUTPUT}, or {DEFAULT_BINARY_OUTPUT} when --binary is set)"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "path for the JSON training report "
            f"(default: {DEFAULT_REPORT}, or {DEFAULT_BINARY_REPORT} when --binary is set)"
        ),
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help=(
            "train a 2-class exterior vs non-exterior head instead of the 6-way "
            "view head. Non-exterior rows (interior/detail/non-car) are pulled "
            "regardless of images.split and assigned to train/val/test "
            "deterministically by SHA-1(image_id)."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def _resolve_backbone_checkpoint(value: str, parser: argparse.ArgumentParser) -> Path | None:
    """Parse the ``--backbone-checkpoint`` argument.

    ``"none"`` (case-insensitive) returns ``None``; anything else is
    treated as a path that MUST exist on disk. We don't auto-fallback
    to the pretrained weights when a path is wrong -- that's a recipe
    for silently training the wrong model.
    """
    if value.lower() == "none":
        return None
    path = Path(value)
    if not path.exists():
        parser.error(f"--backbone-checkpoint path does not exist: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``train-view-classifier`` console script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.batch_size <= 0:
        parser.error(f"--batch-size must be > 0, got {args.batch_size}")
    if args.backbone_batch_size <= 0:
        parser.error(f"--backbone-batch-size must be > 0, got {args.backbone_batch_size}")
    if args.epochs < 0:
        parser.error(f"--epochs must be >= 0, got {args.epochs}")
    if args.min_view_score < 0.0 or args.min_view_score > 1.0:
        parser.error(f"--min-view-score must be in [0, 1], got {args.min_view_score}")

    db_path: Path = args.db
    if not db_path.exists():
        parser.error(f"DB path does not exist: {db_path}")

    backbone_checkpoint = _resolve_backbone_checkpoint(args.backbone_checkpoint, parser)

    # Resolve output / report paths: when --binary is set we route to a
    # different default file so the 6-way checkpoint isn't clobbered.
    output_path: Path = (
        args.output
        if args.output is not None
        else (DEFAULT_BINARY_OUTPUT if args.binary else DEFAULT_OUTPUT)
    )
    report_path: Path = (
        args.report
        if args.report is not None
        else (DEFAULT_BINARY_REPORT if args.binary else DEFAULT_REPORT)
    )

    config = ViewClassifierConfig(
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        backbone_batch_size=args.backbone_batch_size,
        min_view_score=args.min_view_score,
        head_arch=args.head_arch,
        backbone_checkpoint=backbone_checkpoint,
        seed=args.seed,
        binary=args.binary,
    )

    conn = open_db(db_path)
    try:
        payload = train_view_classifier(conn=conn, config=config)
    finally:
        conn.close()

    # Write the .pt checkpoint via torch.save -- import lazily so the
    # CLI module remains import-cheap.
    import torch  # noqa: PLC0415

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload: dict[str, Any] = {
        "head_state_dict": payload.head_state_dict,
        "image_encoder_state_dict": payload.image_encoder_state_dict,
        "class_names": payload.class_names,
        "config": payload.config,
        "val_confusion_matrix": payload.val_confusion_matrix,
    }
    torch.save(out_payload, output_path)
    payload.report.checkpoint_path = str(output_path)
    write_report(payload.report, report_path)

    print(
        f"train-view-classifier: best_top1={payload.report.best_val_top1:.4f} "
        f"best_epoch={payload.report.best_epoch} "
        f"n_train={payload.report.n_train} "
        f"n_val={payload.report.n_val} "
        f"head_arch={config.head_arch} "
        f"binary={config.binary} "
        f"elapsed={payload.report.total_elapsed_s:.1f}s"
    )
    print(f"train-view-classifier: checkpoint -> {output_path}")
    print(f"train-view-classifier: report -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
