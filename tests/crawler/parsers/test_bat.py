"""Tests for the Bring a Trailer parser."""

from __future__ import annotations

from typing import Any, Protocol

from car_lense_engine.crawler.parsers import BringATrailerParser
from car_lense_engine.crawler.parsers.base import (
    DiscoveredUrl,
    ParsedListing,
    ParseResult,
)

SEARCH_FIXTURE = "bat_search.html"
LISTING_FIXTURE = "bat_listing.html"
SEARCH_URL = "https://bringatrailer.com/porsche/911/?year_min=1988&year_max=1990"
LISTING_URL = "https://bringatrailer.com/listing/1989-porsche-911-coupe-77/"


class _LoadFixture(Protocol):
    def __call__(self, name: str) -> str: ...


# ---------- protocol / shape -------------------------------------------------


def test_parser_source_attribute() -> None:
    assert BringATrailerParser().source == "bat"


def test_parser_implements_protocol() -> None:
    parser: Any = BringATrailerParser()
    assert hasattr(parser, "parse") and callable(parser.parse)
    assert hasattr(parser, "source") and isinstance(parser.source, str)


# ---------- search ----------------------------------------------------------


def test_parse_search_extracts_listing_urls(load_fixture: _LoadFixture) -> None:
    html = load_fixture(SEARCH_FIXTURE)
    parser = BringATrailerParser()
    result = parser.parse(html=html, url=SEARCH_URL, kind="search", hints={})

    assert isinstance(result, ParseResult)
    listings = [u for u in result.new_urls if u.kind == "listing"]
    searches = [u for u in result.new_urls if u.kind == "search"]

    assert len(listings) == 3
    listing_urls = [u.url for u in listings]
    assert listing_urls == [
        "https://bringatrailer.com/listing/1988-porsche-911-carrera-targa-22/",
        "https://bringatrailer.com/listing/1989-porsche-911-coupe-77/",
        "https://bringatrailer.com/listing/1990-porsche-911-carrera-4-31/",
    ]
    for du in listings:
        assert isinstance(du, DiscoveredUrl)
        assert du.source == "bat"

    assert len(searches) == 1
    assert searches[0].url.endswith("?page=3")
    assert searches[0].source == "bat"


def test_parse_search_propagates_hints(load_fixture: _LoadFixture) -> None:
    html = load_fixture(SEARCH_FIXTURE)
    parser = BringATrailerParser()
    hints: dict[str, str | int | None] = {
        "target_year": 1989,
        "target_make": "Porsche",
        "target_model": "911",
    }
    result = parser.parse(html=html, url=SEARCH_URL, kind="search", hints=hints)

    for du in result.new_urls:
        assert du.target_year == 1989
        assert du.target_make == "Porsche"
        assert du.target_model == "911"


def test_parse_search_empty_html_returns_notes() -> None:
    parser = BringATrailerParser()
    result = parser.parse(html="", url=SEARCH_URL, kind="search", hints={})

    assert result.new_urls == []
    assert result.new_listing is None
    assert any("no listing cards" in n for n in result.notes)


# ---------- listing ---------------------------------------------------------


def test_parse_listing_extracts_canonical_fields(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = BringATrailerParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    listing = result.new_listing
    assert isinstance(listing, ParsedListing)
    assert listing.year == 1989
    assert listing.make == "Porsche"
    assert listing.model == "911"
    assert listing.body_style == "Coupe"
    assert listing.mileage == 78000
    assert listing.vin == "WP0AB0915KS120123"
    assert listing.source == "bat"
    assert listing.url == LISTING_URL


def test_parse_listing_image_urls_extracted(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = BringATrailerParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    images = result.new_listing.image_urls
    # 2 string entries + 1 ImageObject dict entry.
    assert len(images) == 3
    assert all(u.startswith("https://") for u in images)


def test_parse_listing_id_format(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = BringATrailerParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    assert result.new_listing.listing_id == "bat:1989-porsche-911-coupe-77"


def test_parse_listing_id_extraction_fails_gracefully() -> None:
    # JSON-LD present but URL does not match the /listing/<slug>/ shape.
    html = """
    <html><head>
      <script type="application/ld+json">
      {"@type": "Vehicle", "manufacturer": {"name": "Porsche"}, "model": {"name": "911"}}
      </script>
    </head><body></body></html>
    """
    parser = BringATrailerParser()
    result = parser.parse(
        html=html,
        url="https://bringatrailer.com/auctions/no-listing-prefix/",
        kind="listing",
        hints={},
    )

    assert result.new_listing is None
    assert any("could not extract listing_id" in n for n in result.notes)


def test_parse_listing_missing_jsonld_returns_notes() -> None:
    html = "<html><head></head><body><h1>No JSON-LD here</h1></body></html>"
    parser = BringATrailerParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is None
    assert any("no Vehicle JSON-LD" in n for n in result.notes)


def test_parse_listing_falls_back_to_hints_for_missing_make_model() -> None:
    """When JSON-LD lacks make/model, the queue hints should fill them in."""
    html = """
    <html><head>
      <script type="application/ld+json">
      {"@type": "Product", "name": "Auction lot 42", "image": []}
      </script>
    </head><body></body></html>
    """
    parser = BringATrailerParser()
    hints: dict[str, str | int | None] = {
        "target_year": 1965,
        "target_make": "Shelby",
        "target_model": "Cobra",
    }
    result = parser.parse(
        html=html,
        url="https://bringatrailer.com/listing/1965-shelby-cobra-csx-replica/",
        kind="listing",
        hints=hints,
    )

    assert result.new_listing is not None
    assert result.new_listing.year == 1965
    assert result.new_listing.make == "Shelby"
    assert result.new_listing.model == "Cobra"


def test_raw_html_sha256_populated(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = BringATrailerParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    digest = result.new_listing.raw_html_sha256
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# ---------- image kind ------------------------------------------------------


def test_parse_image_kind_returns_noop() -> None:
    parser = BringATrailerParser()
    result = parser.parse(
        html="<binary>",
        url="https://bringatrailer.com/photo.jpg",
        kind="image",
        hints={},
    )
    assert result.new_urls == []
    assert result.new_listing is None
    assert any("image kind is a no-op" in n for n in result.notes)


def test_parse_unknown_kind_returns_note() -> None:
    parser = BringATrailerParser()
    result = parser.parse(html="", url=SEARCH_URL, kind="weird", hints={})
    assert result.new_urls == []
    assert result.new_listing is None
    assert any("unknown kind" in n for n in result.notes)
