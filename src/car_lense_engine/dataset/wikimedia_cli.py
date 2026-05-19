"""Console script for the Wikimedia Commons vintage-car ingest (Phase 4.4).

Invoke via the ``import-wikimedia`` entry point declared in ``pyproject.toml``::

    import-wikimedia [--db PATH] [--api-url URL]
                     [--year-min Y] [--year-max Y]
                     [--max-per-category N]
                     [--output-dir PATH]
                     [--seed CATEGORY ...]
                     [--limit N] [--dry-run]
                     [--report PATH] [-v]

Walks the configured Wikimedia category seeds, extracts ``(year, make,
model)`` from each file's parent categories, downloads the image bytes
at <= 1 req/sec, and writes one synthetic listing + one image row per
file into the crawler DB. Idempotent — re-runs skip rows already present.

The default seed list covers the pre-2000 vintage gap (1900s through
1990s "automobiles" rollups + every "Cars introduced in <year>" page in
the configured range). Override with ``--seed`` to customize the crawl.

``--dry-run`` exercises the API walk + label extraction without
touching the DB or downloading bytes. Use this to validate connectivity
and label coverage before kicking off a real ingest.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from car_lense_engine.db import open_db

from .wikimedia import (
    WIKIMEDIA_API_URL,
    WIKIMEDIA_USER_AGENT,
    WikimediaIngestConfig,
    WikimediaIngestSummary,
    ingest_wikimedia,
)

DEFAULT_DB = Path("db/crawl.sqlite")
DEFAULT_OUTPUT_DIR = Path("data/public/wikimedia")
DEFAULT_REPORT = Path("reports/phase4_4_wikimedia.json")
DEFAULT_YEAR_MIN = 1900
DEFAULT_YEAR_MAX = 1999
DEFAULT_MAX_PER_CATEGORY = 500
DEFAULT_SPLIT = "train"
SPLIT_CHOICES: tuple[str, ...] = ("train", "val", "test")

logger = logging.getLogger(__name__)


def _default_seed_categories(year_min: int, year_max: int) -> list[str]:
    """Build the default seed list: per-year + per-decade Commons categories.

    For each year in ``[year_min, year_max]``:

    * ``"Cars introduced in <year>"`` (specific-year category — the primary
      source of year-tagged files).

    For each decade in the range:

    * ``"<decade>s automobiles"`` (rollup; year is inferred from individual
      file categories or via the decade-midpoint fallback).
    """
    cats: list[str] = []
    for year in range(year_min, year_max + 1):
        cats.append(f"Category:Cars introduced in {year}")
    decade_start = (year_min // 10) * 10
    decade_end = (year_max // 10) * 10
    for d in range(decade_start, decade_end + 1, 10):
        cats.append(f"Category:{d}s automobiles")
    return cats


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser for the ``import-wikimedia`` command."""
    parser = argparse.ArgumentParser(
        prog="import-wikimedia",
        description=(
            "Ingest pre-2000 vintage cars from Wikimedia Commons via the "
            "MediaWiki API. Walks seed categories, extracts (year, make, "
            "model) from each file's parent categories, and inserts one "
            "synthetic listing + one image row per file into the crawler "
            "DB. Polite crawl: 1 req/sec by default, with descriptive UA. "
            "Idempotent."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"path to the crawler SQLite DB (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=WIKIMEDIA_API_URL,
        help=f"MediaWiki API endpoint (default: {WIKIMEDIA_API_URL})",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=WIKIMEDIA_USER_AGENT,
        help="HTTP User-Agent string (Wikimedia requires identifiable UAs)",
    )
    parser.add_argument(
        "--year-min",
        type=int,
        default=DEFAULT_YEAR_MIN,
        help=f"earliest year to ingest (default: {DEFAULT_YEAR_MIN})",
    )
    parser.add_argument(
        "--year-max",
        type=int,
        default=DEFAULT_YEAR_MAX,
        help=f"latest year to ingest (default: {DEFAULT_YEAR_MAX} -- pre-2000 vintage)",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=DEFAULT_MAX_PER_CATEGORY,
        help=(
            f"per-category file cap (default: {DEFAULT_MAX_PER_CATEGORY}). "
            "Keeps the run bounded when a single category has thousands of files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for written images (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        action="append",
        default=None,
        help=(
            "category to walk (repeatable). If omitted, the default seed list "
            "is derived from --year-min / --year-max: each year contributes "
            "'Cars introduced in <year>' and each decade contributes "
            "'<decade>s automobiles'."
        ),
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help=(
            "minimum delay in seconds between Wikimedia API requests "
            "(default: 1.0 -- Wikimedia's published polite-crawl rate)"
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=SPLIT_CHOICES,
        default=DEFAULT_SPLIT,
        help=(f"semantic split tag recorded in listings.split (default: {DEFAULT_SPLIT})"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the total number of files processed this run (smoke-test hatch)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "walk the API and extract labels but skip ALL DB writes and image "
            "downloads. Useful for connectivity / coverage previews."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(f"optional path to write the JSON summary to (suggested: {DEFAULT_REPORT})"),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Cross-field validation past argparse's basic type-checks."""
    if args.year_min > args.year_max:
        parser.error(f"--year-min ({args.year_min}) must be <= --year-max ({args.year_max})")
    if args.max_per_category <= 0:
        parser.error(f"--max-per-category must be > 0, got {args.max_per_category}")
    if args.limit is not None and args.limit <= 0:
        parser.error(f"--limit must be > 0, got {args.limit}")
    if args.rate_limit < 0:
        parser.error(f"--rate-limit must be >= 0, got {args.rate_limit}")


def _write_report(path: Path, summary: WikimediaIngestSummary) -> None:
    """Serialize the summary to a JSON file (pretty-printed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.model_dump(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _print_summary(summary: WikimediaIngestSummary, *, dry_run: bool) -> None:
    """Emit a one-line stdout summary so the operator sees totals immediately."""
    prefix = "import-wikimedia (dry-run)" if dry_run else "import-wikimedia"
    print(
        f"{prefix}: processed={summary.processed} "
        f"listings_inserted={summary.listings_inserted} "
        f"images_inserted={summary.images_inserted} "
        f"skipped_existing={summary.skipped_existing} "
        f"skipped_no_label={summary.skipped_no_label} "
        f"skipped_out_of_year_range={summary.skipped_out_of_year_range} "
        f"skipped_unsupported_type={summary.skipped_unsupported_type} "
        f"skipped_download_failures={summary.skipped_download_failures} "
        f"api_errors={summary.api_errors}"
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``import-wikimedia`` console script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _validate_args(args, parser)

    seed_categories: tuple[str, ...] = tuple(
        args.seed
        if args.seed is not None
        else _default_seed_categories(args.year_min, args.year_max)
    )

    config = WikimediaIngestConfig(
        api_url=args.api_url,
        user_agent=args.user_agent,
        output_dir=args.output_dir,
        seed_categories=seed_categories,
        year_min=args.year_min,
        year_max=args.year_max,
        max_images_per_category=args.max_per_category,
        rate_limit_seconds=args.rate_limit,
        split=args.split,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    db_path: Path = args.db
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = open_db(db_path)
    try:
        summary = ingest_wikimedia(conn=conn, config=config)
    finally:
        conn.close()

    if args.report is not None:
        _write_report(args.report, summary)
        logger.info("import-wikimedia: wrote report -> %s", args.report)

    _print_summary(summary, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
