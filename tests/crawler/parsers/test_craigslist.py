"""Tests for the Craigslist parser."""

from __future__ import annotations

from typing import Any, Protocol

from car_lense_engine.crawler.parsers import CraigslistParser
from car_lense_engine.crawler.parsers.base import (
    DiscoveredUrl,
    ParsedListing,
    ParseResult,
)

SEARCH_FIXTURE = "craigslist_search.html"
LISTING_FIXTURE = "craigslist_listing.html"
SEARCH_URL = (
    "https://newyork.craigslist.org/search/cta"
    "?auto_make_model=honda+civic&min_auto_year=2020&max_auto_year=2020"
)
LISTING_URL = "https://newyork.craigslist.org/que/cto/d/2020-honda-civic-si/7123456789.html"


class _LoadFixture(Protocol):
    def __call__(self, name: str) -> str: ...


# ---------- protocol / shape -------------------------------------------------


def test_parser_source_attribute() -> None:
    assert CraigslistParser().source == "craigslist"


def test_parser_implements_protocol() -> None:
    """Duck-type the Parser protocol surface."""
    parser: Any = CraigslistParser()
    assert hasattr(parser, "parse") and callable(parser.parse)
    assert hasattr(parser, "source") and isinstance(parser.source, str)


# ---------- search ----------------------------------------------------------


def test_parse_search_extracts_listing_urls(load_fixture: _LoadFixture) -> None:
    html = load_fixture(SEARCH_FIXTURE)
    parser = CraigslistParser()
    result = parser.parse(html=html, url=SEARCH_URL, kind="search", hints={})

    assert isinstance(result, ParseResult)
    listings = [u for u in result.new_urls if u.kind == "listing"]
    searches = [u for u in result.new_urls if u.kind == "search"]

    assert len(listings) == 3
    listing_urls = [u.url for u in listings]
    assert listing_urls == [
        "https://newyork.craigslist.org/que/cto/d/2020-honda-civic-si/7123456789.html",
        "https://losangeles.craigslist.org/lac/ctd/d/2018-ford-f-150-xlt/7234567890.html",
        "https://sfbay.craigslist.org/sby/cto/d/2019-toyota-camry-le/7345678901.html",
    ]
    for du in listings:
        assert isinstance(du, DiscoveredUrl)
        assert du.source == "craigslist"

    # One pagination URL — the "Next ›" anchor.
    assert len(searches) == 1
    assert searches[0].url.endswith("?s=120")
    assert searches[0].source == "craigslist"


def test_parse_search_propagates_hints(load_fixture: _LoadFixture) -> None:
    html = load_fixture(SEARCH_FIXTURE)
    parser = CraigslistParser()
    hints: dict[str, str | int | None] = {
        "target_year": 2020,
        "target_make": "Honda",
        "target_model": "Civic",
    }
    result = parser.parse(html=html, url=SEARCH_URL, kind="search", hints=hints)

    assert result.new_urls
    for du in result.new_urls:
        assert du.target_year == 2020
        assert du.target_make == "Honda"
        assert du.target_model == "Civic"


def test_parse_search_empty_html_returns_notes() -> None:
    parser = CraigslistParser()
    result = parser.parse(html="", url=SEARCH_URL, kind="search", hints={})

    assert result.new_urls == []
    assert result.new_listing is None
    assert any("no listing cards" in n for n in result.notes)


# ---------- listing ---------------------------------------------------------


def test_parse_listing_extracts_canonical_fields_from_hints(
    load_fixture: _LoadFixture,
) -> None:
    """Hints win over title parsing when both are available."""
    html = load_fixture(LISTING_FIXTURE)
    parser = CraigslistParser()
    hints: dict[str, str | int | None] = {
        "target_year": 2020,
        "target_make": "Honda",
        "target_model": "Civic",
    }
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints=hints)

    assert result.new_listing is not None
    listing = result.new_listing
    assert isinstance(listing, ParsedListing)
    assert listing.year == 2020
    assert listing.make == "Honda"
    assert listing.model == "Civic"
    assert listing.source == "craigslist"
    assert listing.url == LISTING_URL


def test_parse_listing_falls_back_to_title_heuristic(
    load_fixture: _LoadFixture,
) -> None:
    """With no hints, parser extracts year/make/model from the title text."""
    html = load_fixture(LISTING_FIXTURE)
    parser = CraigslistParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    listing = result.new_listing
    assert listing.year == 2020
    assert listing.make == "Honda"
    # Two-token model: "civic si" → "Civic Si"
    assert listing.model == "Civic Si"


def test_title_heuristic_two_word_make() -> None:
    parser = CraigslistParser()
    year, make, model = parser._parse_title_heuristic("2018 Land Rover Defender")
    assert year == 2018
    assert make == "Land Rover"
    assert model == "Defender"


def test_title_heuristic_year_at_end() -> None:
    parser = CraigslistParser()
    year, make, model = parser._parse_title_heuristic("ford f-150 supercrew 2019")
    assert year == 2019
    assert make == "Ford"
    assert model == "F-150 Supercrew"


def test_title_heuristic_lowercase() -> None:
    parser = CraigslistParser()
    year, make, model = parser._parse_title_heuristic("honda civic 2020")
    assert year == 2020
    assert make == "Honda"
    assert model == "Civic"


def test_title_heuristic_unknown_make() -> None:
    parser = CraigslistParser()
    year, make, model = parser._parse_title_heuristic("2020 Foo Bar")
    assert year == 2020
    assert make is None
    assert model is None


def test_parse_listing_id_format(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = CraigslistParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    assert result.new_listing.listing_id == "craigslist:7123456789"


def test_parse_listing_id_extraction_fails_gracefully() -> None:
    # URL with no trailing \d+\.html — and no fallback id in the HTML.
    html = '<html><body><span id="titletextonly">2020 Honda Civic</span></body></html>'
    parser = CraigslistParser()
    result = parser.parse(
        html=html,
        url="https://newyork.craigslist.org/que/cto/d/no-id-here/",
        kind="listing",
        hints={},
    )
    assert result.new_listing is None
    assert any("could not extract listing_id" in n for n in result.notes)


def test_parse_listing_extracts_mileage_from_attrgroup(
    load_fixture: _LoadFixture,
) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = CraigslistParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    assert result.new_listing.mileage == 45000


def test_parse_listing_extracts_body_style_from_attrgroup(
    load_fixture: _LoadFixture,
) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = CraigslistParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    assert result.new_listing.body_style == "Sedan"


def test_parse_listing_extracts_vin_from_attrgroup(
    load_fixture: _LoadFixture,
) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = CraigslistParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    assert result.new_listing.vin == "2HGFC3B36LH700123"


def test_parse_listing_extracts_vin_from_body_text() -> None:
    """When no attrgroup VIN, parser falls back to a regex over body text."""
    html = """
    <html><body>
      <span id="titletextonly">2020 Honda Civic Si</span>
      <p class="attrgroup">
        <span>odometer: 45000</span>
        <span>type: sedan</span>
      </p>
      <section id="postingbody">
        Selling my Civic. VIN is 2HGFC3B36LH700123 if you want to look it up.
      </section>
    </body></html>
    """
    parser = CraigslistParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    assert result.new_listing.vin == "2HGFC3B36LH700123"


def test_parse_listing_handles_h2_postingtitle_without_inner_span() -> None:
    """Regression: <h2 class="postingtitle"> with the title text directly inside
    (no inner <span id="titletextonly">) must still yield a parsed listing."""
    html = """
    <html><body>
      <h2 class="postingtitle">2020 Honda Civic Si</h2>
      <p class="attrgroup">
        <span>odometer: 45000</span>
      </p>
    </body></html>
    """
    parser = CraigslistParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    listing = result.new_listing
    assert listing.year == 2020
    assert listing.make == "Honda"
    assert listing.model == "Civic Si"
    assert listing.mileage == 45000
    assert listing.listing_id == "craigslist:7123456789"


def test_parse_listing_image_urls_extracted(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = CraigslistParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    images = result.new_listing.image_urls
    # Three full-size gallery thumbs; the inline <img> 300x300 versions are
    # distinct URLs so they ALSO get added — but the test contract says we
    # should have "3 images, all https". The thumb hrefs are the canonical
    # full-size set; the embedded <img> tags are smaller previews.
    https_only = [u for u in images if u.startswith("https://")]
    assert https_only == images
    assert len(https_only) >= 3
    # The three primary 600x450 hrefs must be present.
    expected = {
        "https://images.craigslist.org/00X0X_abc123_600x450.jpg",
        "https://images.craigslist.org/00Y0Y_def456_600x450.jpg",
        "https://images.craigslist.org/00Z0Z_ghi789_600x450.jpg",
    }
    assert expected.issubset(set(images))


def test_parse_image_kind_returns_noop() -> None:
    parser = CraigslistParser()
    result = parser.parse(
        html="<binary>",
        url="https://images.craigslist.org/photo.jpg",
        kind="image",
        hints={},
    )
    assert result.new_urls == []
    assert result.new_listing is None
    assert any("image kind is a no-op" in n for n in result.notes)


def test_parse_unknown_kind_returns_note() -> None:
    parser = CraigslistParser()
    result = parser.parse(html="", url=SEARCH_URL, kind="weird", hints={})
    assert result.new_urls == []
    assert result.new_listing is None
    assert any("unknown kind" in n for n in result.notes)


def test_raw_html_sha256_populated(load_fixture: _LoadFixture) -> None:
    html = load_fixture(LISTING_FIXTURE)
    parser = CraigslistParser()
    result = parser.parse(html=html, url=LISTING_URL, kind="listing", hints={})

    assert result.new_listing is not None
    digest = result.new_listing.raw_html_sha256
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
