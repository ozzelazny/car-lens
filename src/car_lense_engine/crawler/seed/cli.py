"""Console script for the search-query seeder.

Invoke via the ``seed-queue`` entry point declared in ``pyproject.toml``::

    seed-queue [--catalog PATH] [--db PATH] [--top-n N] [--sites s1,s2,...]
               [--cities c1,c2,...] [--dry-run] [-v]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from car_lense_engine.catalog.schema import Catalog
from car_lense_engine.db import open_db

from .ranker import rank_models
from .seed import build_urls_for, seed_queue
from .urls import DEFAULT_CRAIGSLIST_CITIES, SITE_BUILDERS

DEFAULT_CATALOG = Path("catalog/classes.json")
DEFAULT_DB = Path("db/crawl.sqlite")
DEFAULT_TOP_N = 2000


def _parse_csv(value: str) -> list[str]:
    """Split a comma-separated CLI argument into a stripped list of tokens."""
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser for the ``seed-queue`` command."""
    all_sites = ",".join(SITE_BUILDERS.keys())
    parser = argparse.ArgumentParser(
        prog="seed-queue",
        description="Generate per-site search URLs and enqueue them into the crawl queue.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help=f"path to classes.json (default: {DEFAULT_CATALOG})",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"path to SQLite crawl DB (default: {DEFAULT_DB} unless --dry-run)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"top-N (make, model) combos by popularity (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--sites",
        type=_parse_csv,
        default=list(SITE_BUILDERS.keys()),
        help=f"comma-separated site IDs (default: all = {all_sites})",
    )
    parser.add_argument(
        "--cities",
        type=_parse_csv,
        default=DEFAULT_CRAIGSLIST_CITIES,
        help="comma-separated Craigslist city codes (default: built-in 10-city list)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print '<source>\\t<url>' lines to stdout instead of enqueuing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``seed-queue`` console script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    unknown = [s for s in args.sites if s not in SITE_BUILDERS]
    if unknown:
        parser.error(f"unknown site IDs: {unknown}. Known: {sorted(SITE_BUILDERS)}")

    catalog_path: Path = args.catalog
    if not catalog_path.exists():
        parser.error(f"catalog file not found: {catalog_path}")
    catalog = Catalog.model_validate_json(catalog_path.read_text(encoding="utf-8"))
    log.info(
        "loaded catalog: %d makes, %d models",
        catalog.meta.total_makes,
        catalog.meta.total_models,
    )

    ranked = rank_models(catalog, top_n=args.top_n)
    log.info("ranked %d classes", len(ranked))

    if args.dry_run:
        count = 0
        for seed in build_urls_for(ranked, args.sites, cities=args.cities):
            sys.stdout.write(f"{seed.source}\t{seed.url}\n")
            count += 1
        sys.stdout.flush()
        print(f"dry-run: yielded {count} URLs", file=sys.stderr)
        return 0

    db_path: Path = args.db if args.db is not None else DEFAULT_DB
    conn = open_db(db_path)
    try:
        stats = seed_queue(conn, ranked, list(args.sites), cities=args.cities)
    finally:
        conn.close()

    print(
        f"seed-queue: yielded={stats.total_yielded} inserted={stats.inserted} "
        f"duplicates={stats.duplicates} per_site={stats.per_site}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
