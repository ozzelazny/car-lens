"""One-off diagnostic: fetch a real cars.com listing-detail page.

Smoke run #5 enqueued 19 cars.com listing URLs and fetched HTTP 200 for all
of them, but the parser found no Vehicle JSON-LD on any. This script grabs
the first queued listing URL from the smoke DB and saves the response HTML
to ``tests/crawler/parsers/fixtures/real_world/`` so we can inspect it and
update the parser accordingly.

Usage::

    python scripts/fetch_carscom_detail.py

Tries ``impersonate="chrome131"`` first, falls back to ``firefox133`` if
that 403s. Exits non-zero if both profiles fail.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from car_lense_engine.crawler.core.curlcffi_fetcher import CurlCffiFetcher
from car_lense_engine.crawler.core.fetcher import FetchError

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "db" / "smoke.sqlite"
FIXTURE_DIR = REPO_ROOT / "tests" / "crawler" / "parsers" / "fixtures" / "real_world"
# Hardcoded fallback in case smoke.sqlite was already cleared or never had
# any cars.com listing URLs enqueued. Picked from a smoke #5 sample.
HARDCODED_FALLBACK_URL = "https://www.cars.com/vehicledetail/3715142b-250e-4689-a303-e5924eb2ceaa/"


def _pick_listing_url() -> str:
    if not DB_PATH.exists():
        print(f"smoke DB missing at {DB_PATH}; using hardcoded fallback URL")
        return HARDCODED_FALLBACK_URL
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT url FROM crawl_queue WHERE source='cars_com' AND kind='listing' LIMIT 1"
        )
        row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        print("no cars.com listing rows in smoke.sqlite; using hardcoded fallback")
        return HARDCODED_FALLBACK_URL
    return str(row[0])


def _try_fetch(url: str, impersonate: str) -> tuple[int, str] | None:
    print(f"[{impersonate}] GET {url}")
    fetcher = CurlCffiFetcher(impersonate=impersonate)
    try:
        page = fetcher.fetch(url)
    except FetchError as exc:
        print(f"[{impersonate}] FetchError: {exc}")
        return None
    finally:
        fetcher.close()
    return page.status, page.html


def main() -> int:
    url = _pick_listing_url()
    print(f"target URL: {url}")

    for impersonate in ("chrome131", "firefox133"):
        outcome = _try_fetch(url, impersonate)
        if outcome is None:
            continue
        status, html = outcome
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = FIXTURE_DIR / f"cars_com_detail_curlcffi_{impersonate}_{timestamp}.html"
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        print(f"[{impersonate}] status={status} size={len(html)} saved={out_path}")
        print("--- first 300 chars ---")
        print(html[:300])
        print("--- end preview ---")
        return 0

    print("ALL impersonation profiles failed; cannot fetch a real detail page.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
