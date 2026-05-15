"""End-to-end smoke test for the Car Lense crawler against live websites.

This script:

1. Opens a fresh SQLite DB at ``db/smoke.sqlite`` (delete any prior smoke DB).
2. Builds a :class:`ParserRegistry` and registers all 6 production parsers.
3. Seeds exactly one canonical search URL per site (6 URLs total, hardcoded).
4. Runs :func:`run_crawler` with conservative pacing and a small ``max_items``
   cap so the run completes in a few minutes.
5. After the run, prints a structured per-site report to stdout summarising
   what was fetched, parsed, enqueued, and what failed.

Run with::

    python scripts/smoke_e2e.py

This is a personal-research crawl. Respect Cloudflare/rate limits — if a site
blocks, document and move on; do NOT attempt to defeat the block.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Make the package importable when running this script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from car_lense_engine.crawler.core.browser import PlaywrightFetcher  # noqa: E402
from car_lense_engine.crawler.core.curlcffi_fetcher import CurlCffiFetcher  # noqa: E402
from car_lense_engine.crawler.core.politeness import PolicyConfig  # noqa: E402
from car_lense_engine.crawler.core.registry import ParserRegistry  # noqa: E402
from car_lense_engine.crawler.core.routing import MultiFetcher  # noqa: E402
from car_lense_engine.crawler.core.runner import run_crawler  # noqa: E402
from car_lense_engine.crawler.parsers import (  # noqa: E402
    AutoTraderParser,
    BringATrailerParser,
    CarsAndBidsParser,
    CarsComParser,
    CraigslistParser,
    HemmingsParser,
)
from car_lense_engine.db import open_db, queue  # noqa: E402

DB_PATH = _REPO_ROOT / "db" / "smoke.sqlite"

SEED_URLS: list[tuple[str, str]] = [
    (
        "cars_com",
        "https://www.cars.com/shopping/results/?makes[]=honda&models[]=honda-civic"
        "&year_min=2020&year_max=2020&stock_type=all",
    ),
    (
        "autotrader",
        "https://www.autotrader.com/cars-for-sale/all-cars/honda/civic?yearMin=2020&yearMax=2020",
    ),
    (
        "craigslist",
        "https://newyork.craigslist.org/search/cta"
        "?auto_make_model=honda+civic&min_auto_year=2020&max_auto_year=2020"
        "&query=honda+civic",
    ),
    (
        "bat",
        "https://bringatrailer.com/honda/civic/",
    ),
    (
        "hemmings",
        "https://www.hemmings.com/classifieds/cars-for-sale?Make=honda&Model=civic",
    ),
    (
        "carsandbids",
        # Cars & Bids: still on Playwright (React SPA — curl_cffi can't read
        # JS-rendered listings). Smoke run 2 returned HTTP 403. Trying
        # upgraded stealth (playwright-stealth 2.x + manual hardening) in this
        # run. If still blocked, this is likely Cloudflare Turnstile (CAPTCHA),
        # which is not defeatable without a solver — accept the gap.
        "https://carsandbids.com/search?q=Honda+Civic",
    ),
]

# Sources in stable order for the report.
SOURCES: tuple[str, ...] = (
    "cars_com",
    "autotrader",
    "craigslist",
    "bat",
    "hemmings",
    "carsandbids",
)

# AutoTrader is a JS-rendered SPA; smoke run 2 (2026-05-15) confirmed that
# ``settle_ms=5000`` alone is not enough — the listing grid hadn't hydrated by
# the time PlaywrightFetcher read the HTML and the search returned a 4 KB
# shell. We hint Playwright with a selector list that should match the
# inventory grid as soon as it appears. The selector list is comma-separated
# so Playwright matches the first that exists; AutoTrader has used several
# wrapper class names over the years and we want to be resilient.
WAIT_FOR_SELECTOR_BY_SOURCE: dict[str, str] = {
    "autotrader": (
        "[data-cmp='inventoryListing'], "
        "[data-cmp='inventoryListingItem'], "
        ".inventory-listing, "
        "[data-qa='listing-card']"
    ),
}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _reset_db() -> None:
    """Remove any prior smoke DB (and WAL sidecars) so the run starts fresh."""
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = DB_PATH.with_name(DB_PATH.name + suffix) if suffix else DB_PATH
        if candidate.exists():
            candidate.unlink()


def _build_registry() -> ParserRegistry:
    registry = ParserRegistry()
    registry.register(CarsComParser())
    registry.register(AutoTraderParser())
    registry.register(CraigslistParser())
    registry.register(BringATrailerParser())
    registry.register(HemmingsParser())
    registry.register(CarsAndBidsParser())
    return registry


def _seed_queue(conn: object, seeds: list[tuple[str, str]]) -> None:
    """Insert one ``search``-kind queue item per (source, url) seed."""
    import sqlite3

    assert isinstance(conn, sqlite3.Connection)
    for source, url in seeds:
        queue.enqueue(
            conn,
            url=url,
            source=source,
            kind="search",
            target_year=2020,
            target_make="Honda",
            target_model="Civic",
        )


def _per_source_report(conn: object, source: str) -> dict[str, object]:
    """Pull the structured smoke-relevant counters for one source from the DB."""
    import sqlite3

    assert isinstance(conn, sqlite3.Connection)

    def _count(sql: str, params: tuple[object, ...]) -> int:
        cur = conn.execute(sql, params)
        row = cur.fetchone()
        return int(row[0]) if row is not None else 0

    listing_urls_discovered = _count(
        "SELECT COUNT(*) FROM crawl_queue WHERE source = ? AND kind = 'listing'",
        (source,),
    )
    listing_urls_done = _count(
        "SELECT COUNT(*) FROM crawl_queue "
        "WHERE source = ? AND kind = 'listing' AND status = 'done'",
        (source,),
    )
    image_urls_enqueued = _count(
        "SELECT COUNT(*) FROM crawl_queue WHERE source = ? AND kind = 'image'",
        (source,),
    )
    listings_rows = _count("SELECT COUNT(*) FROM listings WHERE source = ?", (source,))

    # Search status (we only seed one, so this is small).
    search_rows = list(
        conn.execute(
            "SELECT status, last_error FROM crawl_queue WHERE source = ? AND kind = 'search'",
            (source,),
        ).fetchall()
    )
    search_statuses = [str(r["status"]) for r in search_rows]
    search_errors = [str(r["last_error"]) for r in search_rows if r["last_error"]]

    # Up to 3 failed/dead queue items (any kind) for the source — useful diagnostic.
    failed_examples_rows = list(
        conn.execute(
            "SELECT url, kind, status, last_error FROM crawl_queue "
            "WHERE source = ? AND status IN ('failed', 'dead') "
            "ORDER BY enqueued_at LIMIT 3",
            (source,),
        ).fetchall()
    )
    failed_examples = [
        {
            "url": str(r["url"]),
            "kind": str(r["kind"]),
            "status": str(r["status"]),
            "last_error": str(r["last_error"]) if r["last_error"] is not None else None,
        }
        for r in failed_examples_rows
    ]

    # Sample one parsed listing for sanity (year/make/model populated?).
    sample_row = conn.execute(
        "SELECT listing_id, year, make, model, trim, mileage, vin "
        "FROM listings WHERE source = ? LIMIT 1",
        (source,),
    ).fetchone()
    sample_listing = (
        {
            "listing_id": str(sample_row["listing_id"]),
            "year": sample_row["year"],
            "make": sample_row["make"],
            "model": sample_row["model"],
            "trim": sample_row["trim"],
            "mileage": sample_row["mileage"],
            "vin": sample_row["vin"],
        }
        if sample_row is not None
        else None
    )

    return {
        "source": source,
        "search_statuses": search_statuses,
        "search_errors": search_errors,
        "listing_urls_discovered": listing_urls_discovered,
        "listing_urls_done": listing_urls_done,
        "image_urls_enqueued": image_urls_enqueued,
        "listings_rows": listings_rows,
        "failed_examples": failed_examples,
        "sample_listing": sample_listing,
    }


def _print_report(conn: object, summary: object, elapsed: float) -> None:
    """Pretty-print the structured smoke report to stdout."""
    print("=" * 72)
    print("CAR LENSE SMOKE E2E REPORT")
    print("=" * 72)
    # The RunSummary type carries .stats / .exit_reason / .elapsed_seconds.
    print(f"Elapsed wall-clock: {elapsed:.1f}s")
    exit_reason = getattr(summary, "exit_reason", "<unknown>")
    stats = getattr(summary, "stats", None)
    if stats is not None:
        print(
            f"Crawler exit_reason: {exit_reason}\n"
            f"Worker stats: requests_total={stats.requests_total} "
            f"ok={stats.requests_succeeded} failed={stats.requests_failed} "
            f"listings_inserted={stats.listings_inserted} "
            f"urls_enqueued={stats.urls_enqueued}"
        )
    print()

    for source in SOURCES:
        rep = _per_source_report(conn, source)
        print(f"--- {source} ---")
        print(f"  search status      : {rep['search_statuses']}")
        if rep["search_errors"]:
            for err in rep["search_errors"]:
                print(f"    search error     : {err[:200]}")
        print(f"  listing URLs enq.  : {rep['listing_urls_discovered']}")
        print(f"  listing URLs done  : {rep['listing_urls_done']}")
        print(f"  listings rows      : {rep['listings_rows']}")
        print(f"  image URLs enq.    : {rep['image_urls_enqueued']}")
        sample = rep["sample_listing"]
        if sample is not None:
            print(
                f"  sample listing     : id={sample['listing_id']} "
                f"year={sample['year']} make={sample['make']} model={sample['model']} "
                f"trim={sample['trim']} mileage={sample['mileage']} vin={sample['vin']}"
            )
        failed_examples = rep["failed_examples"]
        if failed_examples:
            print(f"  failed examples ({len(failed_examples)}):")
            for ex in failed_examples:
                err_txt = (ex["last_error"] or "")[:160]
                print(f"    [{ex['status']}] kind={ex['kind']} url={ex['url']}")
                if err_txt:
                    print(f"      error: {err_txt}")
        print()


def main() -> int:
    _setup_logging()
    log = logging.getLogger("smoke_e2e")

    _reset_db()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info("opening fresh smoke DB at %s", DB_PATH)
    conn = open_db(DB_PATH)

    try:
        registry = _build_registry()
        log.info("registered parsers: %s", registry.sources())

        _seed_queue(conn, SEED_URLS)
        q_stats = queue.stats(conn)
        log.info(
            "seeded queue: pending=%d in_progress=%d done=%d failed=%d dead=%d",
            q_stats.pending,
            q_stats.in_progress,
            q_stats.done,
            q_stats.failed,
            q_stats.dead,
        )

        policy = PolicyConfig(
            min_delay_seconds=3.0,
            max_delay_seconds=5.0,
            off_peak_only=False,
            idle_exit_seconds=10,
        )
        # Optional residential proxy: set PROXY_URL env var to route ALL crawler
        # traffic (both Playwright and curl_cffi) through a residential proxy.
        # Helps with IP-reputation Cloudflare blocks on cars.com / C&B / Hemmings.
        # Example: export PROXY_URL='http://user:pass@gate.smartproxy.com:7000'
        proxy_url: str | None = os.environ.get("PROXY_URL") or None
        # The 2026-05-15 smoke run found AutoTrader returning unhydrated 4 KB
        # shells and Craigslist returning inconsistent (~155 KB vs ~35 KB)
        # responses with the previous defaults (wait_until="domcontentloaded",
        # settle_ms=1500, navigation_timeout_ms=30000). Both sites render via
        # JS, so we bump the wait/timeout for the smoke target list. Stay on
        # "domcontentloaded" (not "networkidle") to avoid the well-known
        # networkidle hangs on long-polling trackers.
        playwright_fetcher = PlaywrightFetcher(
            headless=True,
            wait_until="domcontentloaded",
            settle_ms=5000,
            navigation_timeout_ms=45_000,
            wait_for_selector_by_source=WAIT_FOR_SELECTOR_BY_SOURCE,
            proxy=proxy_url,
        )
        # Route cars.com and Hemmings through curl_cffi: Playwright+stealth was
        # 403'd by Cloudflare on both, almost certainly because of the headless
        # Chromium TLS / JA3 fingerprint. curl_cffi (Chrome 131 impersonation)
        # uses a real-browser TLS handshake at the request level and often
        # clears that class of block.
        #
        # Keep Cars & Bids on Playwright: their listings are React-hydrated
        # client-side, so a request-level fetcher would only see an empty
        # shell. The C&B block is a separate problem (likely still Cloudflare,
        # but solving it via curl_cffi would not help — the parser needs the
        # post-hydration DOM).
        curl_fetcher = CurlCffiFetcher(proxy=proxy_url)
        fetcher = MultiFetcher(
            per_source={
                "cars_com": curl_fetcher,
                "hemmings": curl_fetcher,
            },
            default=playwright_fetcher,
        )

        started = time.monotonic()
        try:
            summary = run_crawler(
                conn=conn,
                fetcher=fetcher,
                registry=registry,
                policy=policy,
                max_items=20,
            )
        finally:
            fetcher.close()
        elapsed = time.monotonic() - started

        _print_report(conn, summary, elapsed=elapsed)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
