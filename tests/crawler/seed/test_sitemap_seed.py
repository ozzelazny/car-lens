"""Tests for sitemap-driven seeding."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from car_lense_engine.crawler.core.fetcher import FetchedPage
from car_lense_engine.crawler.core.sitemap import SitemapWalker
from car_lense_engine.crawler.seed.cli import main as cli_main
from car_lense_engine.crawler.seed.sitemap_seed import (
    SITEMAP_ROOTS,
    is_autotrader_listing,
    is_carsandbids_listing,
    seed_queue_from_sitemap,
)


class _CannedWalker:
    """Walker stand-in: walk() returns a hardcoded list regardless of root URL."""

    def __init__(self, urls: list[str]) -> None:
        self._urls = urls

    def walk(self, root_url: str) -> Iterator[str]:
        yield from self._urls


# ----------------------------------------------------------- filter unit tests


def test_is_autotrader_listing_accepts_vehicledetails_with_6plus_digit_id() -> None:
    assert is_autotrader_listing("https://www.autotrader.com/cars-for-sale/vehicledetails/123456/")
    assert is_autotrader_listing(
        "https://www.autotrader.com/cars-for-sale/vehicledetails/used-2020-honda-civic/789012345"
    )


def test_is_autotrader_listing_slug_embedded_id() -> None:
    """AutoTrader's hyphen-separated slug-embedded id shape must match.

    The parser docs both ``/vehicledetails/{slug}/{id}`` and
    ``/vehicledetails/{slug}-{id}`` as canonical listing URL shapes. The
    seeder filter previously rejected the hyphen form because it required
    a literal ``/`` before the trailing digit run.
    """
    assert (
        is_autotrader_listing(
            "https://www.autotrader.com/cars-for-sale/vehicledetails/2020-honda-civic-12345678"
        )
        is True
    )


def test_is_autotrader_listing_slash_separated_id() -> None:
    """AutoTrader's slash-separated id shape must continue to match."""
    assert (
        is_autotrader_listing(
            "https://www.autotrader.com/cars-for-sale/vehicledetails/2020-honda-civic/12345678"
        )
        is True
    )


def test_is_autotrader_listing_rejects_non_listing_paths() -> None:
    # Category page (no numeric id).
    assert not is_autotrader_listing("https://www.autotrader.com/cars-for-sale/honda/civic")
    # Wrong path prefix.
    assert not is_autotrader_listing("https://www.autotrader.com/research/honda/civic/")
    # Path has fewer than 6 digits.
    assert not is_autotrader_listing(
        "https://www.autotrader.com/cars-for-sale/vehicledetails/12345/"
    )
    # Empty / nonsense URLs.
    assert not is_autotrader_listing("https://www.autotrader.com/")
    assert not is_autotrader_listing("not-a-url")


def test_is_autotrader_listing_vehicle_path_shape() -> None:
    """AT's marketplace sitemap uses /cars-for-sale/vehicle/<id> form."""
    assert (
        is_autotrader_listing("https://www.autotrader.com/cars-for-sale/vehicle/12345678") is True
    )
    assert (
        is_autotrader_listing("https://www.autotrader.com/cars-for-sale/vehicle/12345678/") is True
    )


def test_is_autotrader_listing_vehicledetails_path_shape_still_accepted() -> None:
    """Existing slug-embedded and slash-separated forms continue to match."""
    assert (
        is_autotrader_listing(
            "https://www.autotrader.com/cars-for-sale/vehicledetails/2020-honda-civic-12345678"
        )
        is True
    )
    assert (
        is_autotrader_listing(
            "https://www.autotrader.com/cars-for-sale/vehicledetails/2020-honda-civic/12345678"
        )
        is True
    )


def test_is_autotrader_listing_rejects_static_pages() -> None:
    assert is_autotrader_listing("https://www.autotrader.com/luxury") is False
    assert is_autotrader_listing("https://www.autotrader.com/coupe") is False
    assert is_autotrader_listing("https://www.autotrader.com/cars-for-sale/all-cars/honda") is False
    # Only 3 digits — must fail the >=6-digit id requirement.
    assert is_autotrader_listing("https://www.autotrader.com/cars-for-sale/vehicle/123") is False


def test_is_carsandbids_listing_accepts_two_segment_auctions_path() -> None:
    # Short / shareable form: /auctions/<slug>
    assert is_carsandbids_listing("https://carsandbids.com/auctions/2020-honda-civic")
    assert is_carsandbids_listing("https://carsandbids.com/auctions/some-slug/")


def test_is_carsandbids_listing_accepts_canonical_sitemap_shape() -> None:
    """``cab-sitemap/auctions.xml`` emits ``/auctions/<short-id>/<title-slug>``."""
    assert is_carsandbids_listing(
        "https://carsandbids.com/auctions/9aQM0NwG/2017-jeep-wrangler-unlimited-sahara-4x4"
    )
    assert is_carsandbids_listing(
        "https://carsandbids.com/auctions/rMealxLL/2015-bentley-continental-gt"
    )
    # Trailing slash tolerated.
    assert is_carsandbids_listing(
        "https://carsandbids.com/auctions/KdxPO0Rp/2012-mercedes-benz-gl350-bluetec/"
    )


def test_is_carsandbids_listing_rejects_non_listing_paths() -> None:
    # Bare /auctions/ index.
    assert not is_carsandbids_listing("https://carsandbids.com/auctions/")
    assert not is_carsandbids_listing("https://carsandbids.com/auctions")
    # Wrong path prefix.
    assert not is_carsandbids_listing("https://carsandbids.com/past-auctions/foo")
    # Sub-page of an auction (one segment too deep).
    assert not is_carsandbids_listing(
        "https://carsandbids.com/auctions/9aQM0NwG/2017-jeep-wrangler/bids"
    )


# --------------------------------------------------- seed_queue_from_sitemap


def test_seed_queue_from_sitemap_autotrader_filter(db: sqlite3.Connection) -> None:
    """Mix of listing + non-listing URLs: only listing URLs enqueue."""
    urls = [
        "https://www.autotrader.com/cars-for-sale/vehicledetails/12345678/",
        "https://www.autotrader.com/cars-for-sale/honda/civic",  # filtered out
        "https://www.autotrader.com/cars-for-sale/vehicledetails/87654321/",
        "https://www.autotrader.com/research/honda/civic/",  # filtered out
    ]
    walker = _CannedWalker(urls)
    stats = seed_queue_from_sitemap(db, source="autotrader", walker=walker)  # type: ignore[arg-type]
    assert stats.walked == 4
    assert stats.matched == 2
    assert stats.inserted == 2
    assert stats.duplicates == 0

    rows = db.execute("SELECT url, source, kind FROM crawl_queue ORDER BY url").fetchall()
    assert len(rows) == 2
    for row in rows:
        assert row["source"] == "autotrader"
        assert row["kind"] == "listing"
        assert "vehicledetails" in row["url"]


def test_seed_queue_from_sitemap_carsandbids_filter(db: sqlite3.Connection) -> None:
    """Mix of short-form, canonical-form, and non-listing URLs.

    URL shapes mirror what ``cab-sitemap/auctions.xml`` actually emits as
    of 2026-05-15 (canonical ``/auctions/<short-id>/<title-slug>``) plus
    the short shareable form humans use directly.
    """
    urls = [
        # Short / shareable form (one trailing segment).
        "https://carsandbids.com/auctions/1991-honda-crx-si",
        # Canonical sitemap form (two trailing segments).
        "https://carsandbids.com/auctions/9aQM0NwG/2017-jeep-wrangler-unlimited-sahara-4x4",
        "https://carsandbids.com/auctions/",  # filtered out
        "https://carsandbids.com/past-auctions/foo",  # filtered out
        "https://carsandbids.com/auctions/2020-honda-civic-type-r",
    ]
    walker = _CannedWalker(urls)
    stats = seed_queue_from_sitemap(db, source="carsandbids", walker=walker)  # type: ignore[arg-type]
    assert stats.walked == 5
    assert stats.matched == 3
    assert stats.inserted == 3

    urls_inserted = sorted(
        row["url"] for row in db.execute("SELECT url FROM crawl_queue").fetchall()
    )
    assert urls_inserted == [
        "https://carsandbids.com/auctions/1991-honda-crx-si",
        "https://carsandbids.com/auctions/2020-honda-civic-type-r",
        "https://carsandbids.com/auctions/9aQM0NwG/2017-jeep-wrangler-unlimited-sahara-4x4",
    ]


def test_seed_queue_from_sitemap_max_listings_caps_yield(
    db: sqlite3.Connection,
) -> None:
    """``max_listings`` bounds the count even when more URLs are walked."""
    urls = [
        f"https://www.autotrader.com/cars-for-sale/vehicledetails/{1_000_000 + i}/"
        for i in range(20)
    ]
    walker = _CannedWalker(urls)
    stats = seed_queue_from_sitemap(
        db,
        source="autotrader",
        walker=walker,  # type: ignore[arg-type]
        max_listings=5,
    )
    assert stats.matched == 5
    assert stats.inserted == 5
    count = db.execute("SELECT COUNT(*) AS n FROM crawl_queue").fetchone()["n"]
    assert int(count) == 5


def test_seed_queue_from_sitemap_unknown_source_raises(
    db: sqlite3.Connection,
) -> None:
    walker = _CannedWalker([])
    with pytest.raises(ValueError, match="unknown sitemap source"):
        seed_queue_from_sitemap(
            db,
            source="not_a_site",
            walker=walker,  # type: ignore[arg-type]
        )


def test_seed_queue_from_sitemap_duplicates_counted(
    db: sqlite3.Connection,
) -> None:
    """Calling the seeder twice with the same URLs counts the second pass as duplicates."""
    urls = [
        "https://www.autotrader.com/cars-for-sale/vehicledetails/11111111/",
        "https://www.autotrader.com/cars-for-sale/vehicledetails/22222222/",
    ]
    first = seed_queue_from_sitemap(
        db,
        source="autotrader",
        walker=_CannedWalker(urls),  # type: ignore[arg-type]
    )
    assert first.inserted == 2
    second = seed_queue_from_sitemap(
        db,
        source="autotrader",
        walker=_CannedWalker(urls),  # type: ignore[arg-type]
    )
    assert second.inserted == 0
    assert second.duplicates == 2


# ----------------------------------------------------------------- CLI tests


class _FakeFetcher:
    """Fake fetcher that maps URL -> body string."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def fetch(self, url: str) -> FetchedPage:
        self.calls.append(url)
        body = self._responses.get(url, "")
        return FetchedPage(
            url=url,
            status=200,
            html=body,
            fetched_at=datetime(2026, 5, 15, 12, 0, 0),
        )

    def close(self) -> None:
        pass


def _urlset(locs: list[str]) -> str:
    inner = "".join(f"<url><loc>{loc}</loc></url>" for loc in locs)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{inner}</urlset>"
    )


def test_seed_queue_from_sitemap_via_cli_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--via-sitemap autotrader --dry-run`` prints listing URLs to stdout.

    We monkey-patch ``CurlCffiFetcher`` inside the CLI module to return our
    fake fetcher, and patch the standard AutoTrader sitemap root to a
    canned urlset of listing-shaped URLs.
    """
    listing_urls = [
        "https://www.autotrader.com/cars-for-sale/vehicledetails/111111/",
        "https://www.autotrader.com/cars-for-sale/honda/civic",  # non-listing, filtered
        "https://www.autotrader.com/cars-for-sale/vehicledetails/222222/",
    ]
    fake_responses = {
        SITEMAP_ROOTS["autotrader"]: _urlset(listing_urls),
    }
    fake = _FakeFetcher(fake_responses)

    # Patch the CurlCffiFetcher class inside the CLI module to return our
    # fake. The CLI builds it with no args, so we use a factory that
    # ignores kwargs and yields ``fake``.
    def _factory(*args: Any, **kwargs: Any) -> _FakeFetcher:
        return fake

    monkeypatch.setattr(
        "car_lense_engine.crawler.seed.cli.CurlCffiFetcher",
        _factory,
    )

    # Run the CLI in dry-run mode against a non-existent catalog (which is
    # fine because no search-sites are listed in --sites, but the parser
    # default is "all" -- we'd hit the catalog path. To avoid that, pass
    # --sites with an empty list semantically: we pass a single site that
    # IS in --via-sitemap so it's stripped from the search-URL pass and
    # the search-URL pass becomes a no-op.
    rc = cli_main(
        [
            "--sites",
            "autotrader",
            "--via-sitemap",
            "autotrader",
            "--dry-run",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    # 2 listing URLs (the third is filtered out as a category page).
    assert len(lines) == 2
    for line in lines:
        source, url = line.split("\t")
        assert source == "autotrader"
        assert "/vehicledetails/" in url
    assert "yielded 2 listing URLs" in captured.err


def test_cli_rejects_unknown_via_sitemap_source(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--via-sitemap`` with an unknown source must error out."""
    catalog_path = tmp_path / "classes.json"
    catalog_path.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli_main(
            [
                "--catalog",
                str(catalog_path),
                "--via-sitemap",
                "not_a_site",
                "--dry-run",
            ]
        )


# --------------------------------------------- end-to-end walker + seeder


def test_walker_and_seeder_compose_against_autotrader_shape(
    db: sqlite3.Connection,
) -> None:
    """A SitemapWalker fed an AutoTrader-style index + child sitemap routes
    correctly through the seeder filter and lands in the queue."""
    root = SITEMAP_ROOTS["autotrader"]
    child = "https://www.autotrader.com/sitemap_main.xml"
    responses = {
        root: (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            f"<sitemap><loc>{child}</loc></sitemap>"
            "</sitemapindex>"
        ),
        child: _urlset(
            [
                "https://www.autotrader.com/cars-for-sale/vehicledetails/700001/",
                "https://www.autotrader.com/cars-for-sale/honda/civic",
                "https://www.autotrader.com/cars-for-sale/vehicledetails/700002/",
            ]
        ),
    }
    fetcher = _FakeFetcher(responses)
    walker = SitemapWalker(fetcher=fetcher, min_delay_seconds=0)
    stats = seed_queue_from_sitemap(db, source="autotrader", walker=walker)
    # 3 URLs walked from the child urlset, 2 matched the listing filter.
    assert stats.walked == 3
    assert stats.matched == 2
    assert stats.inserted == 2
    rows = db.execute("SELECT url FROM crawl_queue ORDER BY url").fetchall()
    urls = [r["url"] for r in rows]
    assert urls == [
        "https://www.autotrader.com/cars-for-sale/vehicledetails/700001/",
        "https://www.autotrader.com/cars-for-sale/vehicledetails/700002/",
    ]
