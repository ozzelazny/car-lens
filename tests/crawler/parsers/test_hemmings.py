"""Tests for the Hemmings parser."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

from car_lense_engine.crawler.parsers import HemmingsParser
from car_lense_engine.crawler.parsers.base import (
    DiscoveredUrl,
    ParsedListing,
    ParseResult,
)

SEARCH_FIXTURE = "hemmings_search.html"
LISTING_FIXTURE = "hemmings_listing.html"
SEARCH_URL = "https://www.hemmings.com/classifieds/cars-for-sale?Make=Chevrolet&Model=Camaro"
LISTING_URL = "https://www.hemmings.com/classifieds/dealer/chevrolet/camaro/2845391/"

# Real-world fixture saved by the block diagnostic (commit 12f8cc6) — the
# actual 319 KB Livewire-rendered Hemmings page returned by
# ``CurlCffiFetcher(impersonate="chrome131")`` against the
# Honda Civic classifieds search slug.
REAL_SEARCH_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "real_world"
    / "hemmings_search_curlcffi_chrome131_20260515T205423Z.html"
)
REAL_SEARCH_URL = "https://www.hemmings.com/classifieds/cars-for-sale/honda/civic"
# Manual count: 7 unique ``/classifieds/listing/<slug>-<id>`` ids in the
# saved fixture (each listing appears twice — once as the card anchor,
# once as a sub-element link — so the deduped count is 7).
REAL_SEARCH_MIN_LISTINGS = 7


class _LoadFixture(Protocol):
    def __call__(self, name: str) -> str: ...


# ---------- protocol / shape -------------------------------------------------


def test_parser_source_attribute() -> None:
    assert HemmingsParser().source == "hemmings"


def test_parser_implements_protocol() -> None:
    parser: Any = HemmingsParser()
    assert hasattr(parser, "parse") and callable(parser.parse)
    assert hasattr(parser, "source") and isinstance(parser.source, str)


# ---------- search ----------------------------------------------------------


def test_parse_search_extracts_listing_urls(load_fixture: _LoadFixture) -> None:
    html = load_fixture(SEARCH_FIXTURE)
    parser = HemmingsParser()
    result = parser.parse(html=html, url=SEARCH_URL, kind="search", hints={})

    assert isinstance(result, ParseResult)
    listings = [u for u in result.new_urls if u.kind == "listing"]
    searches = [u for u in result.new_urls if u.kind == "search"]

    assert len(listings) == 3
    listing_urls = [u.url for u in listings]
    assert listing_urls == [
        "https://www.hemmings.com/classifieds/dealer/chevrolet/camaro/2845391/",
        "https://www.hemmings.com/classifieds/dealer/chevrolet/camaro/2845392/",
        "https://www.hemmings.com/auctions/1969-chevrolet-camaro-rs/2845393",
    ]
    for du in listings:
        assert isinstance(du, DiscoveredUrl)
        assert du.source == "hemmings"

    assert len(searches) == 1
    assert searches[0].url.endswith("?page=3")
    assert searches[0].source == "hemmings"


def test_parse_search_propagates_hints(load_fixture: _LoadFixture) -> None:
    html = load_fixture(SEARCH_FIXTURE)
    parser = HemmingsParser()
    hints: dict[str, str | int | None] = {
        "target_year": 1969,
        "target_make": "Chevrolet",
        "target_model": "Camaro",
    }
    result = parser.parse(html=html, url=SEARCH_URL, kind="search", hints=hints)

    for du in result.new_urls:
        assert du.target_year == 1969
        assert du.target_make == "Chevrolet"
        assert du.target_model == "Camaro"


def test_parse_search_empty_html_returns_notes() -> None:
    parser = HemmingsParser()
    result = parser.parse(html="", url=SEARCH_URL, kind="search", hints={})

    assert result.new_urls == []
    assert result.new_listing is None
    assert any("no listing cards" in n for n in result.notes)


# ---------- listing ---------------------------------------------------------


def test_parse_listing_extracts_canonical_fields(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = HemmingsParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    listing = result.new_listing
    assert isinstance(listing, ParsedListing)
    assert listing.year == 1969
    assert listing.make == "Chevrolet"
    assert listing.model == "Camaro"
    assert listing.trim == "SS"
    assert listing.body_style == "Coupe"
    assert listing.mileage == 65000
    assert listing.vin == "124379N654321"
    assert listing.source == "hemmings"
    assert listing.url == LISTING_URL


def test_parse_listing_image_urls_extracted(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = HemmingsParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    images = result.new_listing.image_urls
    assert len(images) == 3
    assert all(u.startswith("https://") for u in images)


def test_parse_listing_id_format(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = HemmingsParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    assert result.new_listing.listing_id == "hemmings:2845391"


def test_parse_listing_id_auctions_shape(load_fixture: _LoadFixture) -> None:
    """Auction-shaped URLs also yield the trailing numeric native id."""
    html = load_fixture(LISTING_FIXTURE)
    parser = HemmingsParser()
    url = "https://www.hemmings.com/auctions/1969-chevrolet-camaro-rs/2845393"
    result = parser.parse(html=html, url=url, kind="listing", hints={})

    assert result.new_listing is not None
    assert result.new_listing.listing_id == "hemmings:2845393"


def test_parse_listing_id_extraction_fails_gracefully() -> None:
    # JSON-LD present but URL has no trailing numeric ID.
    html = """
    <html><head>
      <script type="application/ld+json">
      {"@type": "Vehicle", "manufacturer": {"name": "Chevrolet"}, "model": {"name": "Camaro"}}
      </script>
    </head><body></body></html>
    """
    parser = HemmingsParser()
    result = parser.parse(
        html=html,
        url="https://www.hemmings.com/classifieds/dealer/no-id-here/",
        kind="listing",
        hints={},
    )

    assert result.new_listing is None
    assert any("could not extract listing_id" in n for n in result.notes)


def test_parse_listing_missing_jsonld_returns_notes() -> None:
    html = "<html><head></head><body><h1>No JSON-LD here</h1></body></html>"
    parser = HemmingsParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is None
    assert any("no Vehicle JSON-LD" in n for n in result.notes)


def test_raw_html_sha256_populated(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = HemmingsParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    digest = result.new_listing.raw_html_sha256
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# ---------- image kind ------------------------------------------------------


def test_parse_image_kind_returns_noop() -> None:
    parser = HemmingsParser()
    result = parser.parse(
        html="<binary>",
        url="https://www.hemmings.com/photo.jpg",
        kind="image",
        hints={},
    )
    assert result.new_urls == []
    assert result.new_listing is None
    assert any("image kind is a no-op" in n for n in result.notes)


def test_parse_unknown_kind_returns_note() -> None:
    parser = HemmingsParser()
    result = parser.parse(html="", url=SEARCH_URL, kind="weird", hints={})
    assert result.new_urls == []
    assert result.new_listing is None
    assert any("unknown kind" in n for n in result.notes)


# ---------- real-world fixture ----------------------------------------------


def test_parse_search_real_html_extracts_listings() -> None:
    """Smoke test against the saved 319 KB real Hemmings page.

    The fixture is the actual response produced by
    ``CurlCffiFetcher(impersonate="chrome131")`` against the Honda Civic
    classifieds search slug on 2026-05-15. Real cards use the
    ``/classifieds/listing/<year>-<make>-<model>-<city>-<state>-<id>``
    URL shape — the trailing numeric segment is captured as the native id.
    """
    html = REAL_SEARCH_FIXTURE.read_text(encoding="utf-8")
    parser = HemmingsParser()
    result = parser.parse(html=html, url=REAL_SEARCH_URL, kind="search", hints={})

    assert isinstance(result, ParseResult)
    listings = [u for u in result.new_urls if u.kind == "listing"]
    assert len(listings) >= REAL_SEARCH_MIN_LISTINGS, (
        f"expected >= {REAL_SEARCH_MIN_LISTINGS} listings, got {len(listings)}"
    )
    seen_urls: set[str] = set()
    for du in listings:
        assert du.source == "hemmings"
        assert du.url.startswith("https://www.hemmings.com/")
        assert "/classifieds/" in du.url or "/auctions/" in du.url
        assert du.url not in seen_urls, f"duplicate listing url: {du.url}"
        seen_urls.add(du.url)


def test_hemmings_search_parses_new_listing_url_shape() -> None:
    """Real-world fixture exercising the new ``/listing/<slug>-<id>`` shape.

    The Camaro search fixture (2026-05-16) contains 20+ cards using the
    bare ``/listing/`` prefix (no ``/classifieds/``). The regex must
    accept this shape; the id extractor pulls the trailing numeric.
    """
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "real_world"
        / "hemmings_search_camaro_curlcffi_chrome131_20260516T031700Z.html"
    )
    html = fixture.read_text(encoding="utf-8")
    result = HemmingsParser().parse(
        html=html,
        url="https://www.hemmings.com/classifieds/cars-for-sale/chevrolet/camaro?YearFrom=1981&YearTo=2026",
        kind="search",
        hints={"target_year": None, "target_make": "Chevrolet", "target_model": "Camaro"},
    )
    listing_urls = [u.url for u in result.new_urls if u.kind == "listing"]
    assert len(listing_urls) >= 15, (
        f"expected >=15 listings, got {len(listing_urls)}; notes={result.notes}"
    )
    # Sanity-check shape: every URL should end in 6+ digits and contain /listing/
    for u in listing_urls:
        assert "/listing/" in u
        assert re.search(r"\d{6,}/?$", u)
    # Discovery should NOT emit a 'selectors may need updating' note.
    assert not any("selectors may need updating" in n for n in result.notes), result.notes
