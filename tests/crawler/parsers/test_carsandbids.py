"""Tests for the Cars & Bids parser."""

from __future__ import annotations

from typing import Any, Protocol

from car_lense_engine.crawler.parsers import CarsAndBidsParser
from car_lense_engine.crawler.parsers.base import (
    DiscoveredUrl,
    ParsedListing,
    ParseResult,
)

SEARCH_FIXTURE = "carsandbids_search.html"
LISTING_FIXTURE = "carsandbids_listing.html"
SEARCH_URL = "https://carsandbids.com/search?q=BMW+M3"
LISTING_URL = "https://carsandbids.com/auctions/2011-bmw-m3-sedan-silver/"


class _LoadFixture(Protocol):
    def __call__(self, name: str) -> str: ...


# ---------- protocol / shape -------------------------------------------------


def test_parser_source_attribute() -> None:
    assert CarsAndBidsParser().source == "carsandbids"


def test_parser_implements_protocol() -> None:
    parser: Any = CarsAndBidsParser()
    assert hasattr(parser, "parse") and callable(parser.parse)
    assert hasattr(parser, "source") and isinstance(parser.source, str)


# ---------- search ----------------------------------------------------------


def test_parse_search_extracts_listing_urls(load_fixture: _LoadFixture) -> None:
    html = load_fixture(SEARCH_FIXTURE)
    parser = CarsAndBidsParser()
    result = parser.parse(html=html, url=SEARCH_URL, kind="search", hints={})

    assert isinstance(result, ParseResult)
    listings = [u for u in result.new_urls if u.kind == "listing"]
    searches = [u for u in result.new_urls if u.kind == "search"]

    assert len(listings) == 3
    listing_urls = [u.url for u in listings]
    assert listing_urls == [
        "https://carsandbids.com/auctions/2008-bmw-m3-coupe-blue/",
        "https://carsandbids.com/auctions/2011-bmw-m3-sedan-silver/",
        "https://carsandbids.com/auctions/2015-bmw-m3-zcp-blackout/",
    ]
    for du in listings:
        assert isinstance(du, DiscoveredUrl)
        assert du.source == "carsandbids"

    assert len(searches) == 1
    assert searches[0].url.endswith("?page=3")
    assert searches[0].source == "carsandbids"


def test_parse_search_propagates_hints(load_fixture: _LoadFixture) -> None:
    html = load_fixture(SEARCH_FIXTURE)
    parser = CarsAndBidsParser()
    hints: dict[str, str | int | None] = {
        "target_year": 2011,
        "target_make": "BMW",
        "target_model": "M3",
    }
    result = parser.parse(html=html, url=SEARCH_URL, kind="search", hints=hints)

    for du in result.new_urls:
        assert du.target_year == 2011
        assert du.target_make == "BMW"
        assert du.target_model == "M3"


def test_parse_search_empty_html_returns_notes() -> None:
    parser = CarsAndBidsParser()
    result = parser.parse(html="", url=SEARCH_URL, kind="search", hints={})

    assert result.new_urls == []
    assert result.new_listing is None
    assert any("no listing cards" in n for n in result.notes)


# ---------- listing ---------------------------------------------------------


def test_parse_listing_extracts_canonical_fields(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = CarsAndBidsParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    listing = result.new_listing
    assert isinstance(listing, ParsedListing)
    assert listing.year == 2011
    assert listing.make == "BMW"
    assert listing.model == "M3"
    assert listing.body_style == "Sedan"
    assert listing.mileage == 48000
    assert listing.vin == "WBSWD9C57BP123456"
    assert listing.source == "carsandbids"
    assert listing.url == LISTING_URL


def test_parse_listing_image_urls_extracted(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = CarsAndBidsParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    images = result.new_listing.image_urls
    # 2 string entries + 1 ImageObject with url + 1 ImageObject with contentUrl.
    assert len(images) == 4
    assert all(u.startswith("https://") for u in images)


def test_parse_listing_id_format(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = CarsAndBidsParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    assert result.new_listing.listing_id == "carsandbids:2011-bmw-m3-sedan-silver"


def test_parse_listing_id_extraction_fails_gracefully() -> None:
    # JSON-LD present but URL does not match the /auctions/<slug>/ shape.
    html = """
    <html><head>
      <script type="application/ld+json">
      {"@type": "Vehicle", "manufacturer": {"name": "BMW"}, "model": {"name": "M3"}}
      </script>
    </head><body></body></html>
    """
    parser = CarsAndBidsParser()
    result = parser.parse(
        html=html,
        url="https://carsandbids.com/listings/no-auctions-prefix/",
        kind="listing",
        hints={},
    )

    assert result.new_listing is None
    assert any("could not extract listing_id" in n for n in result.notes)


def test_parse_listing_missing_jsonld_returns_notes() -> None:
    html = "<html><head></head><body><h1>No JSON-LD here</h1></body></html>"
    parser = CarsAndBidsParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is None
    assert any("no Vehicle JSON-LD" in n for n in result.notes)


def test_raw_html_sha256_populated(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = CarsAndBidsParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    digest = result.new_listing.raw_html_sha256
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# ---------- image kind ------------------------------------------------------


def test_parse_image_kind_returns_noop() -> None:
    parser = CarsAndBidsParser()
    result = parser.parse(
        html="<binary>",
        url="https://media.carsandbids.com/photo.jpg",
        kind="image",
        hints={},
    )
    assert result.new_urls == []
    assert result.new_listing is None
    assert any("image kind is a no-op" in n for n in result.notes)


def test_parse_unknown_kind_returns_note() -> None:
    parser = CarsAndBidsParser()
    result = parser.parse(html="", url=SEARCH_URL, kind="weird", hints={})
    assert result.new_urls == []
    assert result.new_listing is None
    assert any("unknown kind" in n for n in result.notes)
