"""Tests for the cars.com parser."""

from __future__ import annotations

from pathlib import Path
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

# Real-world fixture saved by the block diagnostic (commit 12f8cc6) —
# the actual 1.22 MB SSR'd Honda Civic results page returned by
# ``CurlCffiFetcher(impersonate="firefox133")`` against cars.com. The
# fixture is the canonical proof that the parser still works against
# production HTML; any selector / regex change that lowers the extracted
# count below the lower bound asserted here breaks the contract.
REAL_SEARCH_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "real_world"
    / "cars_com_search_curlcffi_firefox_20260515T205423Z.html"
)
# Manual count of distinct ``/vehicledetail/<uuid>/`` ids in the fixture
# (27 cards × multiple anchor variants each). The parser dedupes by
# native_id, so the lower bound is the distinct-listing count.
REAL_SEARCH_MIN_LISTINGS = 27

# Real-world DETAIL-page fixture saved by
# ``scripts/fetch_carscom_detail.py`` against the first cars.com listing
# URL enqueued by smoke run #5. The page is the 613 KB SSR'd VDP for a
# used 2020 Honda Civic LX. Production HTML carries the canonical fields
# in an inline ``CarsWeb.VehicleDetailController.show`` JSON state blob,
# NOT in a schema.org JSON-LD ``Vehicle`` block; this fixture locks in
# the parser's CarsWeb fallback against that real shape.
REAL_DETAIL_FIXTURES_GLOB = "cars_com_detail_*.html"
# Native id parsed from the URL of the fixture above.
REAL_DETAIL_LISTING_ID = "3715142b-250e-4689-a303-e5924eb2ceaa"
REAL_DETAIL_URL = f"https://www.cars.com/vehicledetail/{REAL_DETAIL_LISTING_ID}/"


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
    ``/features/`` sub-routes on a listing-detail URL and canonicalises
    them to the base ``/vehicledetail/<id>/`` form so the worker does not
    re-crawl photo / feature variants as separate listings."""
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
        "https://www.cars.com/vehicledetail/12345/",
        "https://www.cars.com/vehicledetail/24680/",
        "https://www.cars.com/vehicledetail/67890/",
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


# ---------- real-world fixture ----------------------------------------------


def test_parse_search_real_html_extracts_listings() -> None:
    """Smoke test against the saved 1.22 MB real cars.com SSR'd page.

    The fixture is the actual response produced by
    ``CurlCffiFetcher(impersonate="firefox133")`` against
    ``https://www.cars.com/shopping/results/?makes[]=honda&models[]=honda-civic``
    on 2026-05-15. Real cards use ``<a href="/vehicledetail/<uuid>/?attribution_type=...">``,
    ``<fuse-button href=...>``, and ``<card-gallery card-href=...>`` —
    all three must be picked up and deduped by native id.
    """
    html = REAL_SEARCH_FIXTURE.read_text(encoding="utf-8")
    parser = CarsComParser()
    result = parser.parse(html=html, url=SEARCH_URL, kind="search", hints={})

    assert isinstance(result, ParseResult)
    listings = [u for u in result.new_urls if u.kind == "listing"]
    assert len(listings) >= REAL_SEARCH_MIN_LISTINGS, (
        f"expected >= {REAL_SEARCH_MIN_LISTINGS} listings, got {len(listings)}"
    )
    # The parser canonicalises every variant down to ``/vehicledetail/<id>/``
    # so the URL set has no duplicate native ids and no query strings.
    seen_ids: set[str] = set()
    for du in listings:
        assert du.source == "cars_com"
        assert du.url.startswith("https://www.cars.com/vehicledetail/")
        assert du.url.endswith("/")
        assert "?" not in du.url, f"query string leaked into canonical URL: {du.url}"
        native_id = du.url.rstrip("/").rsplit("/", 1)[-1]
        assert native_id not in seen_ids, f"duplicate native id: {native_id}"
        seen_ids.add(native_id)


def test_parse_listing_real_html_extracts_canonical_fields() -> None:
    """Lock in detail-page parsing against the real saved fixture.

    Real cars.com VDPs do NOT embed a schema.org Vehicle JSON-LD block;
    the canonical fields live in the inline
    ``<script id="CarsWeb.VehicleDetailController.show">`` JSON state.
    This test guards the CarsWeb fallback so we don't silently regress
    back to "no Vehicle JSON-LD found" for every production listing.
    """
    fixture_dir = Path(__file__).parent / "fixtures" / "real_world"
    matches = sorted(fixture_dir.glob(REAL_DETAIL_FIXTURES_GLOB))
    assert matches, (
        f"no real cars.com detail fixtures matching {REAL_DETAIL_FIXTURES_GLOB!r} "
        f"in {fixture_dir}; run scripts/fetch_carscom_detail.py to create one"
    )
    html_path = matches[0]
    html = html_path.read_text(encoding="utf-8")
    parser = CarsComParser()
    result = parser.parse(html=html, url=REAL_DETAIL_URL, kind="listing", hints={})

    assert result.new_listing is not None, result.notes
    pl = result.new_listing
    # At minimum: year, make, model must be populated for a real cars.com page.
    assert pl.year is not None
    assert pl.make is not None
    assert pl.model is not None
    # The known shape of the fixture (used 2020 Honda Civic LX). These hard
    # asserts catch silent regressions if the parser starts picking up the
    # wrong sub-tree.
    assert pl.year == 2020
    assert pl.make == "Honda"
    assert pl.model == "Civic"
    assert pl.trim == "LX"
    assert pl.body_style == "Sedan"
    assert pl.vin == "19XFC2F69LE000132"
    assert pl.mileage == 76939
    # Image URLs come from the ``<img slot="image">`` fallback on real HTML.
    assert len(pl.image_urls) >= 1
    assert all("platform.cstatic-images.com" in u for u in pl.image_urls)
    # listing_id derived from URL.
    assert pl.listing_id == f"cars_com:{REAL_DETAIL_LISTING_ID}"
