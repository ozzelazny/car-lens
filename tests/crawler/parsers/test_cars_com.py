"""Tests for the cars.com parser."""

from __future__ import annotations

from typing import Any, Protocol

from car_lense_engine.crawler.parsers import CarsComParser
from car_lense_engine.crawler.parsers.base import (
    DiscoveredUrl,
    ParsedListing,
    ParseResult,
)

SEARCH_FIXTURE = "cars_com_search.html"
LISTING_FIXTURE = "cars_com_listing.html"
SEARCH_URL = "https://www.cars.com/shopping/results/?makes[]=honda&models[]=honda-civic"
LISTING_URL = "https://www.cars.com/vehicledetail/744993211/"


class _LoadFixture(Protocol):
    def __call__(self, name: str) -> str: ...


# ---------- protocol / shape -------------------------------------------------


def test_parser_source_attribute() -> None:
    assert CarsComParser().source == "cars_com"


def test_parser_implements_protocol() -> None:
    """Duck-type the Parser protocol surface."""
    parser: Any = CarsComParser()
    assert hasattr(parser, "parse") and callable(parser.parse)
    assert hasattr(parser, "source") and isinstance(parser.source, str)


# ---------- search ----------------------------------------------------------


def test_parse_search_extracts_listing_urls(load_fixture: _LoadFixture) -> None:
    html = load_fixture(SEARCH_FIXTURE)
    parser = CarsComParser()
    result = parser.parse(html=html, url=SEARCH_URL, kind="search", hints={})

    assert isinstance(result, ParseResult)
    listings = [u for u in result.new_urls if u.kind == "listing"]
    searches = [u for u in result.new_urls if u.kind == "search"]

    assert len(listings) == 3
    listing_urls = [u.url for u in listings]
    assert listing_urls == [
        "https://www.cars.com/vehicledetail/744993211/",
        "https://www.cars.com/vehicledetail/812005317/",
        "https://www.cars.com/vehicledetail/900112233/",
    ]
    for du in listings:
        assert isinstance(du, DiscoveredUrl)
        assert du.source == "cars_com"

    # One pagination URL — the rel="next" anchor.
    assert len(searches) == 1
    assert searches[0].url.endswith("?page=2")
    assert searches[0].source == "cars_com"


def test_parse_search_propagates_hints(load_fixture: _LoadFixture) -> None:
    html = load_fixture(SEARCH_FIXTURE)
    parser = CarsComParser()
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


def test_parse_search_empty_html_returns_notes() -> None:
    parser = CarsComParser()
    result = parser.parse(html="", url=SEARCH_URL, kind="search", hints={})

    assert result.new_urls == []
    assert result.new_listing is None
    assert any("no listing cards" in n for n in result.notes)


def test_parse_search_extracts_hrefs_from_spark_link_button() -> None:
    """cars.com newer listing cards use ``<spark-link-button>`` custom
    elements instead of plain ``<a>`` tags. The parser must pick up the
    href from either tag type."""
    html = """
    <html><body>
      <ul class="vehicle-cards">
        <li class="vehicle-card">
          <spark-link-button href="/vehicledetail/12345/">View listing</spark-link-button>
        </li>
      </ul>
    </body></html>
    """
    parser = CarsComParser()
    result = parser.parse(html=html, url=SEARCH_URL, kind="search", hints={})

    listings = [u for u in result.new_urls if u.kind == "listing"]
    assert len(listings) == 1
    assert listings[0].url == "https://www.cars.com/vehicledetail/12345/"
    assert listings[0].source == "cars_com"


def test_parse_search_accepts_overview_suffix() -> None:
    """The href regex accepts ``/overview/``, ``/photos/``, and
    ``/features/`` sub-routes on a listing-detail URL."""
    html = """
    <html><body>
      <a class="vehicle-card-link" href="/vehicledetail/12345/overview/">View</a>
      <a class="image-link" href="/vehicledetail/67890/photos/">Photos</a>
      <a class="features-link" href="/vehicledetail/24680/features/">Features</a>
    </body></html>
    """
    parser = CarsComParser()
    result = parser.parse(html=html, url=SEARCH_URL, kind="search", hints={})

    listings = [u for u in result.new_urls if u.kind == "listing"]
    listing_urls = sorted(u.url for u in listings)
    assert listing_urls == [
        "https://www.cars.com/vehicledetail/12345/overview/",
        "https://www.cars.com/vehicledetail/24680/features/",
        "https://www.cars.com/vehicledetail/67890/photos/",
    ]


# ---------- listing ---------------------------------------------------------


def test_parse_listing_extracts_vehicle_jsonld(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = CarsComParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    listing = result.new_listing
    assert isinstance(listing, ParsedListing)
    assert listing.year == 2020
    assert listing.make == "Honda"
    assert listing.model == "Civic"
    assert listing.trim == "Si"
    assert listing.body_style == "Sedan"
    assert listing.mileage == 45000
    assert listing.vin == "2HGFC3B36LH700123"
    assert listing.source == "cars_com"
    assert listing.url == LISTING_URL


def test_parse_listing_image_urls_extracted(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = CarsComParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    images = result.new_listing.image_urls
    assert len(images) == 3
    assert all(u.startswith("https://") for u in images)


def test_parse_listing_id_format(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = CarsComParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    assert result.new_listing.listing_id == "cars_com:744993211"


def test_parse_listing_missing_jsonld_returns_notes() -> None:
    html = "<html><head></head><body><h1>No JSON-LD here</h1></body></html>"
    parser = CarsComParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is None
    assert any("no Vehicle JSON-LD" in n for n in result.notes)


def test_parse_listing_id_extraction_fails_gracefully() -> None:
    # JSON-LD present but URL does not match the /vehicledetail/<id>/ shape.
    html = """
    <html><head>
      <script type="application/ld+json">
      {"@type": "Vehicle", "manufacturer": {"name": "Honda"}, "model": {"name": "Civic"}}
      </script>
    </head><body></body></html>
    """
    parser = CarsComParser()
    result = parser.parse(
        html=html,
        url="https://example.com/notavehicle/",
        kind="listing",
        hints={},
    )

    assert result.new_listing is None
    assert any("could not extract listing_id" in n for n in result.notes)


def test_raw_html_sha256_populated(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = CarsComParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    digest = result.new_listing.raw_html_sha256
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# ---------- image kind ------------------------------------------------------


def test_parse_image_kind_returns_noop() -> None:
    parser = CarsComParser()
    result = parser.parse(
        html="<binary>",
        url="https://images.cars.com/photo.jpg",
        kind="image",
        hints={},
    )
    assert result.new_urls == []
    assert result.new_listing is None
    assert any("image kind is a no-op" in n for n in result.notes)


def test_parse_unknown_kind_returns_note() -> None:
    parser = CarsComParser()
    result = parser.parse(html="", url=SEARCH_URL, kind="weird", hints={})
    assert result.new_urls == []
    assert result.new_listing is None
    assert any("unknown kind" in n for n in result.notes)
