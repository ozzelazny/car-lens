"""Console script for the per-image stratified train/val/test split (Phase 3.5).

Invoke via the ``make-splits`` entry point declared in ``pyproject.toml``::

    make-splits --source compcars [--db PATH] [--seed N]
                [--train-frac 0.8] [--val-frac 0.1]
                [--min-group-size 10] [--rebuild]
                [--listing-coherent | --no-listing-coherent]
                [--dry-run] [--report PATH] [-v]

Reads ``images`` joined with ``listings``, groups eligible rows by
``(canonical_make, canonical_model, generation_year, view)``, and assigns
each image to one of ``train`` / ``val`` / ``test``. Writes the assignment
back to ``images.split``.

Idempotent: rows already carrying a split are left alone unless
``--rebuild`` is passed. Pass ``--dry-run`` to compute (and optionally
report) the split without touching the DB.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from car_lense_engine.db import open_db

from .make_splits import SplitSummary, make_splits

DEFAULT_DB = Path("db/crawl.sqlite")
DEFAULT_SEED = 42
DEFAULT_TRAIN_FRAC = 0.8
DEFAULT_VAL_FRAC = 0.1
DEFAULT_MIN_GROUP_SIZE = 10

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser for the ``make-splits`` command."""
    parser = argparse.ArgumentParser(
        prog="make-splits",
        description=(
            "Assign each image in the crawler DB to a train/val/test split, "
            "stratified by (canonical_make, canonical_model, generation_year, "
            "view) and (by default) coherent at the listing level so two "
            "photos of the same physical car never appear in different "
            "splits. Idempotent — re-runs leave existing assignments alone "
            "unless --rebuild is passed."
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
        required=True,
        help="restrict the split to images whose listing has this source (e.g. compcars)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"deterministic shuffle seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=DEFAULT_TRAIN_FRAC,
        help=f"target train fraction (default: {DEFAULT_TRAIN_FRAC})",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=DEFAULT_VAL_FRAC,
        help=(
            f"target val fraction (default: {DEFAULT_VAL_FRAC}); "
            "test_frac = 1 - train_frac - val_frac"
        ),
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=DEFAULT_MIN_GROUP_SIZE,
        help=(
            "cells with fewer images than this get all-train assignment "
            f"(no eval signal for that cell). Default: {DEFAULT_MIN_GROUP_SIZE}"
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="recompute every eligible row, overwriting prior split assignments",
    )
    # --listing-coherent / --no-listing-coherent, default True.
    coherent = parser.add_mutually_exclusive_group()
    coherent.add_argument(
        "--listing-coherent",
        dest="listing_coherent",
        action="store_true",
        default=True,
        help="assign whole listings to the same split (default ON)",
    )
    coherent.add_argument(
        "--no-listing-coherent",
        dest="listing_coherent",
        action="store_false",
        help="shuffle images independently (allows same-vehicle cross-split leakage)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute the assignment but skip the DB write",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional path to write the JSON summary to",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Cross-field validation: split fractions, min-group-size, DB existence."""
    if args.train_frac < 0 or args.val_frac < 0:
        parser.error(
            f"--train-frac and --val-frac must be non-negative; got "
            f"{args.train_frac} and {args.val_frac}"
        )
    total = args.train_frac + args.val_frac
    if not 0.0 < total <= 1.0:
        parser.error(f"--train-frac + --val-frac must be in (0, 1]; got {total}")
    if args.min_group_size <= 0:
        parser.error(f"--min-group-size must be > 0, got {args.min_group_size}")
    db_path: Path = args.db
    if not db_path.exists():
        parser.error(f"DB path does not exist: {db_path}")


def _write_report(path: Path, summary: SplitSummary) -> None:
    """Serialize the summary to a JSON file (pretty-printed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _print_summary(summary: SplitSummary, *, dry_run: bool) -> None:
    """Emit a one-line stdout summary so the operator sees totals immediately."""
    prefix = "make-splits (dry-run)" if dry_run else "make-splits"
    print(
        f"{prefix}: eligible={summary.total_eligible} "
        f"train={summary.total_train} val={summary.total_val} test={summary.total_test} "
        f"small_group={summary.total_skipped_small_group} "
        f"excluded_non_exterior={summary.total_excluded_non_exterior} "
        f"per_class_p10={summary.per_class_coverage_p10}"
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``make-splits`` console script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _validate_args(args, parser)

    conn = open_db(args.db)
    try:
        summary = make_splits(
            conn,
            source=args.source,
            seed=args.seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            min_group_size=args.min_group_size,
            rebuild=args.rebuild,
            listing_coherent=args.listing_coherent,
            dry_run=args.dry_run,
        )
    finally:
        conn.close()

    if args.report is not None:
        _write_report(args.report, summary)
        logger.info("make-splits: wrote report -> %s", args.report)

    _print_summary(summary, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
