"""Console script for the search-query seeder.

Invoke via the ``seed-queue`` entry point declared in ``pyproject.toml``::

    seed-queue [--catalog PATH] [--db PATH] [--top-n N] [--sites s1,s2,...]
               [--cities c1,c2,...] [--dry-run] [-v]
               [--via-sitemap s1,s2 [--sitemap-max-listings N]]

``--via-sitemap SOURCES`` switches to sitemap-walking mode for the named
sources (AutoTrader, Cars & Bids). Their search surfaces are blocked behind
Akamai / Cloudflare interstitials; the sitemap XML is the only realistic
discovery path. Sources listed here are NOT also seeded via search URLs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from car_lense_engine.catalog.schema import Catalog
from car_lense_engine.crawler.core.curlcffi_fetcher import CurlCffiFetcher
from car_lense_engine.crawler.core.sitemap import SitemapWalker
from car_lense_engine.db import open_db

from .ranker import rank_models
from .seed import build_urls_for, seed_queue
from .sitemap_seed import LISTING_FILTERS, SITEMAP_ROOTS, seed_queue_from_sitemap
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
    sitemap_sources = ",".join(sorted(SITEMAP_ROOTS))
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
        help=f"comma-separated site IDs for search-URL seeding (default: all = {all_sites})",
    )
    parser.add_argument(
        "--cities",
        type=_parse_csv,
        default=DEFAULT_CRAIGSLIST_CITIES,
        help="comma-separated Craigslist city codes (default: built-in 10-city list)",
    )
    parser.add_argument(
        "--via-sitemap",
        type=_parse_csv,
        default=[],
        help=(
            "comma-separated sources to seed via sitemap walking instead of "
            f"search URLs (known: {sitemap_sources}). Sources listed here are "
            "skipped by the regular search-URL seeder."
        ),
    )
    parser.add_argument(
        "--sitemap-max-listings",
        type=int,
        default=None,
        help="cap on listings enqueued per --via-sitemap source (default: no cap)",
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


def _run_sitemap_seeding(
    *,
    conn: object,
    sources: list[str],
    max_listings: int | None,
    dry_run: bool,
    log: logging.Logger,
) -> None:
    """Walk each requested sitemap source and either enqueue or dump to stdout.

    Builds one shared :class:`CurlCffiFetcher` for the duration of the walk
    so the underlying session is reused across sources.
    """
    fetcher = CurlCffiFetcher()
    try:
        walker = SitemapWalker(fetcher=fetcher)
        for source in sources:
            if dry_run:
                # Walk + filter only; do NOT touch the DB. Print one
                # '<source>\t<url>' line per matched listing URL so the
                # output mirrors the search-URL dry-run format.
                count = 0
                filt = LISTING_FILTERS[source]
                for url in walker.walk(SITEMAP_ROOTS[source]):
                    if not filt(url):
                        continue
                    sys.stdout.write(f"{source}\t{url}\n")
                    count += 1
                    if max_listings is not None and count >= max_listings:
                        break
                sys.stdout.flush()
                print(
                    f"dry-run sitemap[{source}]: yielded {count} listing URLs",
                    file=sys.stderr,
                )
            else:
                import sqlite3

                assert isinstance(conn, sqlite3.Connection)
                stats = seed_queue_from_sitemap(
                    conn,
                    source=source,
                    walker=walker,
                    max_listings=max_listings,
                )
                log.info(
                    "sitemap[%s]: walked=%d matched=%d inserted=%d duplicates=%d",
                    source,
                    stats.walked,
                    stats.matched,
                    stats.inserted,
                    stats.duplicates,
                )
                print(
                    f"sitemap[{stats.source}]: walked={stats.walked} "
                    f"matched={stats.matched} inserted={stats.inserted} "
                    f"duplicates={stats.duplicates}",
                    file=sys.stderr,
                )
    finally:
        fetcher.close()


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

    sitemap_sources: list[str] = list(args.via_sitemap)
    unknown_sm = [s for s in sitemap_sources if s not in SITEMAP_ROOTS]
    if unknown_sm:
        parser.error(
            f"unknown --via-sitemap source IDs: {unknown_sm}. Known: {sorted(SITEMAP_ROOTS)}"
        )

    # Sources listed in --via-sitemap are NOT also seeded via search URLs.
    search_sites: list[str] = [s for s in args.sites if s not in sitemap_sources]

    # Catalog is only required when we'll actually run the search-URL seeder.
    catalog: Catalog | None = None
    ranked: list[object] = []
    if search_sites:
        catalog_path: Path = args.catalog
        if not catalog_path.exists():
            parser.error(f"catalog file not found: {catalog_path}")
        catalog = Catalog.model_validate_json(catalog_path.read_text(encoding="utf-8"))
        log.info(
            "loaded catalog: %d makes, %d models",
            catalog.meta.total_makes,
            catalog.meta.total_models,
        )
        ranked = list(rank_models(catalog, top_n=args.top_n))
        log.info("ranked %d classes", len(ranked))

    if args.dry_run:
        # Search-URL dry-run output (unchanged contract).
        count = 0
        if search_sites:
            for seed in build_urls_for(ranked, search_sites, cities=args.cities):  # type: ignore[arg-type]
                sys.stdout.write(f"{seed.source}\t{seed.url}\n")
                count += 1
            sys.stdout.flush()
            print(f"dry-run: yielded {count} URLs", file=sys.stderr)
        # Sitemap dry-run output (one TSV line per listing URL).
        if sitemap_sources:
            _run_sitemap_seeding(
                conn=None,
                sources=sitemap_sources,
                max_listings=args.sitemap_max_listings,
                dry_run=True,
                log=log,
            )
        return 0

    db_path: Path = args.db if args.db is not None else DEFAULT_DB
    conn = open_db(db_path)
    try:
        if search_sites:
            stats = seed_queue(conn, ranked, list(search_sites), cities=args.cities)  # type: ignore[arg-type]
            print(
                f"seed-queue: yielded={stats.total_yielded} inserted={stats.inserted} "
                f"duplicates={stats.duplicates} per_site={stats.per_site}",
                file=sys.stderr,
            )
        if sitemap_sources:
            _run_sitemap_seeding(
                conn=conn,
                sources=sitemap_sources,
                max_listings=args.sitemap_max_listings,
                dry_run=False,
                log=log,
            )
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
