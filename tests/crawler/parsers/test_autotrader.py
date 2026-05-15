"""Tests for the AutoTrader parser."""

from __future__ import annotations

from typing import Any, Protocol

from car_lense_engine.crawler.parsers import AutoTraderParser
from car_lense_engine.crawler.parsers.base import (
    DiscoveredUrl,
    ParsedListing,
    ParseResult,
)

SEARCH_FIXTURE = "autotrader_search.html"
LISTING_FIXTURE = "autotrader_listing.html"
SEARCH_URL = (
    "https://www.autotrader.com/cars-for-sale/all-cars/honda/civic?yearMin=2020&yearMax=2020"
)
LISTING_URL = (
    "https://www.autotrader.com/cars-for-sale/vehicledetails/"
    "2019-toyota-camry-le-houston-tx-99887766"
)


class _LoadFixture(Protocol):
    def __call__(self, name: str) -> str: ...


# ---------- protocol / shape -------------------------------------------------


def test_parser_source_attribute() -> None:
    assert AutoTraderParser().source == "autotrader"


def test_parser_implements_protocol() -> None:
    """Duck-type the Parser protocol surface."""
    parser: Any = AutoTraderParser()
    assert hasattr(parser, "parse") and callable(parser.parse)
    assert hasattr(parser, "source") and isinstance(parser.source, str)


# ---------- search ----------------------------------------------------------


def test_parse_search_extracts_listing_urls(load_fixture: _LoadFixture) -> None:
    html = load_fixture(SEARCH_FIXTURE)
    parser = AutoTraderParser()
    result = parser.parse(html=html, url=SEARCH_URL, kind="search", hints={})

    assert isinstance(result, ParseResult)
    listings = [u for u in result.new_urls if u.kind == "listing"]
    searches = [u for u in result.new_urls if u.kind == "search"]

    assert len(listings) == 3
    listing_urls = [u.url for u in listings]
    assert listing_urls == [
        (
            "https://www.autotrader.com/cars-for-sale/vehicledetails/"
            "2020-honda-civic-si-philadelphia-pa-12345678"
        ),
        "https://www.autotrader.com/cars-for-sale/vehicle/87654321",
        (
            "https://www.autotrader.com/cars-for-sale/vehicledetails/"
            "2019-toyota-camry-le-houston-tx-99887766"
        ),
    ]
    for du in listings:
        assert isinstance(du, DiscoveredUrl)
        assert du.source == "autotrader"

    # One pagination URL — the "Next page" anchor.
    assert len(searches) == 1
    assert searches[0].url.endswith("?page=3")
    assert searches[0].source == "autotrader"


def test_parse_search_propagates_hints(load_fixture: _LoadFixture) -> None:
    html = load_fixture(SEARCH_FIXTURE)
    parser = AutoTraderParser()
    hints: dict[str, str | int | None] = {
        "target_year": 2020,
        "target_make": "Honda",
        "target_model": "Civic",
    }
    result = parser.parse(html=html, url=SEARCH_URL, kind="search", hints=hints)

    for du in result.new_urls:
        assert du.target_year == 2020
        assert du.target_make == "Honda"
        assert du.target_model == "Civic"


def test_parse_search_ignores_previous_page_link(load_fixture: _LoadFixture) -> None:
    """The 'Previous page' anchor must NOT be picked as a next-page link."""
    html = load_fixture(SEARCH_FIXTURE)
    parser = AutoTraderParser()
    result = parser.parse(html=html, url=SEARCH_URL, kind="search", hints={})

    searches = [u for u in result.new_urls if u.kind == "search"]
    assert len(searches) == 1
    # The previous-page href is ?page=1, the next is ?page=3.
    assert "page=1" not in searches[0].url
    assert searches[0].url.endswith("?page=3")


def test_parse_search_picks_up_next_page_text_variant(load_fixture: _LoadFixture) -> None:
    """The fixture uses 'Next page' text (not exactly 'Next') — must still match."""
    html = load_fixture(SEARCH_FIXTURE)
    parser = AutoTraderParser()
    result = parser.parse(html=html, url=SEARCH_URL, kind="search", hints={})

    searches = [u for u in result.new_urls if u.kind == "search"]
    assert len(searches) == 1
    assert searches[0].url.endswith("?page=3")


def test_parse_search_empty_html_returns_notes() -> None:
    parser = AutoTraderParser()
    result = parser.parse(html="", url=SEARCH_URL, kind="search", hints={})

    assert result.new_urls == []
    assert result.new_listing is None
    assert any("no listing cards" in n for n in result.notes)


# ---------- listing ---------------------------------------------------------


def test_parse_listing_extracts_vehicle_jsonld(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = AutoTraderParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    listing = result.new_listing
    assert isinstance(listing, ParsedListing)
    assert listing.year == 2019
    assert listing.make == "Toyota"
    assert listing.model == "Camry"
    assert listing.trim == "LE"
    assert listing.body_style == "Sedan"
    assert listing.mileage == 22000
    assert listing.vin == "4T1B11HK5KU123456"
    assert listing.source == "autotrader"
    assert listing.url == LISTING_URL


def test_parse_listing_image_urls_extracted(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = AutoTraderParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    images = result.new_listing.image_urls
    assert len(images) == 4
    assert all(u.startswith("https://") for u in images)


def test_parse_listing_id_format(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = AutoTraderParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    assert result.new_listing.listing_id == "autotrader:99887766"


def test_parse_listing_id_extracts_from_long_slug(load_fixture: _LoadFixture) -> None:
    """Given a long-slug URL, the trailing numeric segment is the listing_id."""
    html = load_fixture(LISTING_FIXTURE)
    parser = AutoTraderParser()
    url = (
        "https://www.autotrader.com/cars-for-sale/vehicledetails/"
        "2020-honda-civic-si-philadelphia-pa-12345678"
    )
    result = parser.parse(html=html, url=url, kind="listing", hints={})

    assert result.new_listing is not None
    assert result.new_listing.listing_id == "autotrader:12345678"


def test_parse_listing_id_extraction_fails_gracefully() -> None:
    # JSON-LD present but URL has no trailing numeric ID.
    html = """
    <html><head>
      <script type="application/ld+json">
      {"@type": "Vehicle", "manufacturer": {"name": "Toyota"}, "model": {"name": "Camry"}}
      </script>
    </head><body></body></html>
    """
    parser = AutoTraderParser()
    result = parser.parse(
        html=html,
        url="https://www.autotrader.com/cars-for-sale/vehicledetails/no-id-here/",
        kind="listing",
        hints={},
    )

    assert result.new_listing is None
    assert any("could not extract listing_id" in n for n in result.notes)


def test_parse_listing_missing_jsonld_returns_notes() -> None:
    html = "<html><head></head><body><h1>No JSON-LD here</h1></body></html>"
    parser = AutoTraderParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is None
    assert any("no Vehicle JSON-LD" in n for n in result.notes)


def test_raw_html_sha256_populated(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = AutoTraderParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    digest = result.new_listing.raw_html_sha256
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# ---------- image kind ------------------------------------------------------


def test_parse_image_kind_returns_noop() -> None:
    parser = AutoTraderParser()
    result = parser.parse(
        html="<binary>",
        url="https://images.autotrader.com/photo.jpg",
        kind="image",
        hints={},
    )
    assert result.new_urls == []
    assert result.new_listing is None
    assert any("image kind is a no-op" in n for n in result.notes)


def test_parse_unknown_kind_returns_note() -> None:
    parser = AutoTraderParser()
    result = parser.parse(html="", url=SEARCH_URL, kind="weird", hints={})
    assert result.new_urls == []
    assert result.new_listing is None
    assert any("unknown kind" in n for n in result.notes)
