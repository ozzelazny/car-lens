"""Console script for the Phase 5.1 zero-shot baseline harness.

Invoke via the ``phase5-baseline`` entry point declared in
``pyproject.toml``::

    phase5-baseline [--db PATH] [--source stanford_cars]
                    [--train-split train] [--test-split test]
                    [--model MobileCLIP-S2] [--pretrained datacompdr]
                    [--device cpu|cuda|mps] [--batch-size N]
                    [--output PATH] [--per-class-top N] [-v]

Builds one mean-embedding prototype per class from the train split, then
reports top-K nearest-prototype accuracy on the test split. Writes the
full :class:`~car_lense_engine.eval.baseline.BaselineReport` as JSON.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from car_lense_engine.db import open_db

from .baseline import (
    BaselineConfig,
    build_prototypes,
    evaluate,
    summary_line,
    write_report,
)

DEFAULT_DB = Path("db/crawl.sqlite")
DEFAULT_SOURCE = "stanford_cars"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_TEST_SPLIT = "test"
DEFAULT_MODEL = "MobileCLIP-S2"
DEFAULT_PRETRAINED = "datacompdr"
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 16
DEFAULT_OUTPUT = Path("reports/phase5_baseline.json")
DEFAULT_PER_CLASS_TOP = 20
DEVICE_CHOICES: tuple[str, ...] = ("cpu", "cuda", "mps")

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser for the ``phase5-baseline`` command."""
    parser = argparse.ArgumentParser(
        prog="phase5-baseline",
        description=(
            "Phase 5.1 zero-shot baseline: build prototype embeddings from the "
            "train split and report top-K nearest-prototype accuracy on the "
            "test split. Outputs a JSON report."
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
        help=f"listings.source to evaluate against (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default=DEFAULT_TRAIN_SPLIT,
        help=f"split used to build prototypes (default: {DEFAULT_TRAIN_SPLIT})",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default=DEFAULT_TEST_SPLIT,
        help=f"split used for evaluation (default: {DEFAULT_TEST_SPLIT})",
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
        help=f"images per forward pass (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"path for the JSON report (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--per-class-top",
        type=int,
        default=DEFAULT_PER_CLASS_TOP,
        help=(
            "number of best + worst classes to include in the per-class "
            f"breakdown (default: {DEFAULT_PER_CLASS_TOP}); also caps the "
            "confusion-pair list"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``phase5-baseline`` console script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.batch_size <= 0:
        parser.error(f"--batch-size must be > 0, got {args.batch_size}")
    if args.per_class_top < 0:
        parser.error(f"--per-class-top must be >= 0, got {args.per_class_top}")

    db_path: Path = args.db
    if not db_path.exists():
        parser.error(f"DB path does not exist: {db_path}")

    config = BaselineConfig(
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
    )

    conn = open_db(db_path)
    try:
        prototypes = build_prototypes(
            conn=conn,
            config=config,
            source=args.source,
            split=args.train_split,
        )
        report = evaluate(
            conn=conn,
            config=config,
            prototypes=prototypes,
            source=args.source,
            split=args.test_split,
            per_class_top=args.per_class_top,
        )
    finally:
        conn.close()

    write_report(report, args.output)
    print(
        f"phase5-baseline: {summary_line(report)} "
        f"n_classes={report.n_classes} "
        f"n_train_images={report.n_train_images} "
        f"n_test_images={report.n_test_images}"
    )
    print(f"phase5-baseline: report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
