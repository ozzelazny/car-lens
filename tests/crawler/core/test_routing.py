"""Tests for :mod:`car_lense_engine.crawler.core.routing`."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from car_lense_engine.crawler.core.fetcher import FetchedPage
from car_lense_engine.crawler.core.routing import (
    HOSTNAME_TO_SOURCE,
    MultiFetcher,
    known_sources,
    source_for_url,
)


@dataclass
class _RecordingFetcher:
    """Minimal :class:`Fetcher` that records every ``fetch()`` call."""

    name: str = "fake"
    calls: list[str] = field(default_factory=list)
    close_count: int = 0

    def fetch(self, url: str) -> FetchedPage:
        self.calls.append(url)
        return FetchedPage(
            url=url,
            status=200,
            html=f"<html>{self.name}</html>",
            fetched_at=datetime.now(UTC).replace(tzinfo=None),
        )

    def close(self) -> None:
        self.close_count += 1


# ----------------------------------------------------------- source_for_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.cars.com/shopping/results/", "cars_com"),
        ("https://cars.com/path", "cars_com"),
        ("https://www.autotrader.com/cars-for-sale/all-cars/honda/civic", "autotrader"),
        ("https://autotrader.com/x", "autotrader"),
        ("https://bringatrailer.com/honda/civic/", "bat"),
        ("https://www.bringatrailer.com/abc", "bat"),
        ("https://www.hemmings.com/classifieds/", "hemmings"),
        ("https://hemmings.com/abc", "hemmings"),
        ("https://carsandbids.com/search?q=Honda+Civic", "carsandbids"),
        ("https://www.carsandbids.com/x", "carsandbids"),
    ],
)
def test_source_for_url_known_hosts(url: str, expected: str) -> None:
    assert source_for_url(url) == expected


def test_source_for_url_www_prefix_handled() -> None:
    assert source_for_url("https://www.cars.com/shopping") == "cars_com"
    assert source_for_url("https://cars.com/shopping") == "cars_com"


@pytest.mark.parametrize(
    "url",
    [
        "https://newyork.craigslist.org/search/cta?query=honda",
        "https://losangeles.craigslist.org/d/cars-trucks/search/cta",
        "https://sfbay.craigslist.org/eby/cto/d/honda/123.html",
        "https://chicago.craigslist.org/abc",
    ],
)
def test_source_for_url_craigslist_subdomain(url: str) -> None:
    assert source_for_url(url) == "craigslist"


def test_source_for_url_unknown() -> None:
    assert source_for_url("https://example.com/") is None
    assert source_for_url("https://google.com/search?q=cars") is None


def test_source_for_url_malformed() -> None:
    # Should not raise, just return None.
    assert source_for_url("not a url") is None
    assert source_for_url("") is None
    assert source_for_url("://broken") is None


def test_source_for_url_is_case_insensitive_on_host() -> None:
    assert source_for_url("https://WWW.CARS.COM/x") == "cars_com"
    assert source_for_url("https://Newyork.Craigslist.Org/x") == "craigslist"


def test_known_sources_contains_all_expected() -> None:
    sources = known_sources()
    for expected in ("cars_com", "autotrader", "bat", "hemmings", "carsandbids", "craigslist"):
        assert expected in sources, f"expected {expected!r} in known_sources()"


def test_hostname_to_source_has_www_and_apex_variants() -> None:
    """Every site listed must have both apex and www. entries."""
    for source in {"cars_com", "autotrader", "bat", "hemmings", "carsandbids"}:
        hosts_for_source = [h for h, s in HOSTNAME_TO_SOURCE.items() if s == source]
        assert any(h.startswith("www.") for h in hosts_for_source), (
            f"missing www. variant for source={source}: {hosts_for_source}"
        )
        assert any(not h.startswith("www.") for h in hosts_for_source), (
            f"missing apex variant for source={source}: {hosts_for_source}"
        )


# ------------------------------------------------------------- MultiFetcher


def test_multi_fetcher_routes_by_source() -> None:
    curl = _RecordingFetcher(name="curl")
    playwright = _RecordingFetcher(name="playwright")
    multi = MultiFetcher(per_source={"cars_com": curl}, default=playwright)

    multi.fetch("https://www.cars.com/shopping/results/")
    multi.fetch("https://www.autotrader.com/cars-for-sale")

    assert curl.calls == ["https://www.cars.com/shopping/results/"]
    assert playwright.calls == ["https://www.autotrader.com/cars-for-sale"]


def test_multi_fetcher_unknown_source_uses_default() -> None:
    curl = _RecordingFetcher(name="curl")
    playwright = _RecordingFetcher(name="playwright")
    multi = MultiFetcher(per_source={"cars_com": curl}, default=playwright)

    multi.fetch("https://example.com/anything")

    assert curl.calls == []
    assert playwright.calls == ["https://example.com/anything"]


def test_multi_fetcher_routes_craigslist_subdomain() -> None:
    curl = _RecordingFetcher(name="curl")
    playwright = _RecordingFetcher(name="playwright")
    multi = MultiFetcher(per_source={"craigslist": curl}, default=playwright)

    multi.fetch("https://newyork.craigslist.org/search/cta?q=honda")

    assert curl.calls == ["https://newyork.craigslist.org/search/cta?q=honda"]
    assert playwright.calls == []


def test_multi_fetcher_close_dedupes() -> None:
    """If the same fetcher appears multiple times, close it exactly once."""
    shared = _RecordingFetcher(name="shared")
    other = _RecordingFetcher(name="other")
    multi = MultiFetcher(
        per_source={"cars_com": shared, "hemmings": shared},
        default=shared,
    )
    multi.close()
    assert shared.close_count == 1
    assert other.close_count == 0  # unrelated fetcher untouched


def test_multi_fetcher_close_closes_each_distinct_fetcher_once() -> None:
    curl = _RecordingFetcher(name="curl")
    playwright = _RecordingFetcher(name="playwright")
    multi = MultiFetcher(per_source={"cars_com": curl}, default=playwright)

    multi.close()
    assert curl.close_count == 1
    assert playwright.close_count == 1


def test_multi_fetcher_close_idempotent() -> None:
    curl = _RecordingFetcher(name="curl")
    playwright = _RecordingFetcher(name="playwright")
    multi = MultiFetcher(per_source={"cars_com": curl}, default=playwright)

    multi.close()
    multi.close()
    multi.close()
    assert curl.close_count == 1
    assert playwright.close_count == 1


def test_multi_fetcher_context_manager_closes() -> None:
    curl = _RecordingFetcher(name="curl")
    playwright = _RecordingFetcher(name="playwright")
    with MultiFetcher(per_source={"cars_com": curl}, default=playwright) as multi:
        multi.fetch("https://www.cars.com/x")
    assert curl.close_count == 1
    assert playwright.close_count == 1


def test_multi_fetcher_returns_inner_fetched_page_unchanged() -> None:
    curl = _RecordingFetcher(name="curl")
    playwright = _RecordingFetcher(name="playwright")
    multi = MultiFetcher(per_source={"cars_com": curl}, default=playwright)

    page = multi.fetch("https://www.cars.com/x")
    assert page.html == "<html>curl</html>"
    assert page.status == 200
