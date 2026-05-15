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
    """When JSON-LD lacks make/model AND the ``name`` field doesn't yield a
    known make (Shelby is not in ``MAKE_POPULARITY``), the queue hints
    should fill them in."""
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


# ---------- name-field make/model heuristic --------------------------------


def test_parse_listing_extracts_make_model_from_name_when_jsonld_null() -> None:
    """JSON-LD has ``name`` but no ``brand``/``manufacturer``/``model``; the
    parser should derive make+model from the name string and ignore the
    fact that hints are empty."""
    html = """
    <html><head>
      <script type="application/ld+json">
      {"@type": "Product",
       "name": "1991 Honda CRX Si",
       "brand": null,
       "manufacturer": null,
       "model": null,
       "image": []}
      </script>
    </head><body></body></html>
    """
    parser = BringATrailerParser()
    result = parser.parse(
        html=html,
        url="https://bringatrailer.com/listing/1991-honda-crx-si-9/",
        kind="listing",
        hints={},
    )

    assert result.new_listing is not None
    listing = result.new_listing
    assert listing.year == 1991
    assert listing.make == "Honda"
    # Model is capped at the FIRST token after the make; the tail
    # ("Si") goes into ``trim``.
    assert listing.model == "Crx"
    assert listing.trim == "Si"


def test_parse_listing_explicit_brand_wins_over_name_parse() -> None:
    """If JSON-LD has explicit ``brand`` AND a parseable ``name``, the
    explicit field wins — name-parsing must not override it."""
    html = """
    <html><head>
      <script type="application/ld+json">
      {"@type": "Product",
       "name": "1991 Honda CRX",
       "brand": {"name": "Porsche"},
       "model": {"name": "911"},
       "image": []}
      </script>
    </head><body></body></html>
    """
    parser = BringATrailerParser()
    result = parser.parse(
        html=html,
        url="https://bringatrailer.com/listing/1991-honda-crx-confusing/",
        kind="listing",
        hints={},
    )

    assert result.new_listing is not None
    # Explicit JSON-LD wins; we do NOT fall through to the name-parser.
    assert result.new_listing.make == "Porsche"
    assert result.new_listing.model == "911"


def test_parse_listing_hints_used_only_when_name_parse_fails() -> None:
    """``name`` references an unknown make → name-parse returns (None,None)
    → parser falls back to queue hints for make/model."""
    html = """
    <html><head>
      <script type="application/ld+json">
      {"@type": "Product",
       "name": "1991 Someunknownmake CRX",
       "brand": null,
       "manufacturer": null,
       "model": null,
       "image": []}
      </script>
    </head><body></body></html>
    """
    parser = BringATrailerParser()
    hints: dict[str, str | int | None] = {
        "target_year": 1991,
        "target_make": "Honda",
        "target_model": "Civic",
    }
    result = parser.parse(
        html=html,
        url="https://bringatrailer.com/listing/1991-mystery-crx/",
        kind="listing",
        hints=hints,
    )

    assert result.new_listing is not None
    assert result.new_listing.year == 1991
    assert result.new_listing.make == "Honda"
    assert result.new_listing.model == "Civic"


def test_parse_listing_prefix_words_before_year_are_skipped() -> None:
    """BaT names sometimes lead with editorial copy ("No Reserve:",
    "Original-Owner") before the year. The parser must walk past those
    to find the year, then look for the make."""
    html = """
    <html><head>
      <script type="application/ld+json">
      {"@type": "Product",
       "name": "No Reserve: Original-Owner 1986 Honda Civic Si",
       "brand": null,
       "manufacturer": null,
       "model": null,
       "image": []}
      </script>
    </head><body></body></html>
    """
    parser = BringATrailerParser()
    result = parser.parse(
        html=html,
        url="https://bringatrailer.com/listing/1986-honda-civic-si-22/",
        kind="listing",
        hints={},
    )

    assert result.new_listing is not None
    listing = result.new_listing
    assert listing.year == 1986
    assert listing.make == "Honda"
    assert listing.model == "Civic"
    assert listing.trim == "Si"


def test_parse_listing_two_word_make_in_name() -> None:
    """Two-word makes ("Land Rover") must be matched as a single unit
    before the parser starts taking model tokens."""
    html = """
    <html><head>
      <script type="application/ld+json">
      {"@type": "Product",
       "name": "2018 Land Rover Defender",
       "brand": null,
       "manufacturer": null,
       "model": null,
       "image": []}
      </script>
    </head><body></body></html>
    """
    parser = BringATrailerParser()
    result = parser.parse(
        html=html,
        url="https://bringatrailer.com/listing/2018-land-rover-defender-1/",
        kind="listing",
        hints={},
    )

    assert result.new_listing is not None
    listing = result.new_listing
    assert listing.year == 2018
    assert listing.make == "Land Rover"
    assert listing.model == "Defender"


def test_parse_name_make_model_strips_trim_into_separate_field() -> None:
    """The model walk caps at the first token; the rest goes to ``trim``.

    Real-world examples from smoke run 2: BaT names like
    ``"1991 Honda CRX Si"`` and ``"1989 Honda Civic Cx Hatchback 5-Speed"``
    used to collapse the whole tail into ``model``. Post-fix, the model
    is the first token after the make and ``trim`` holds the rest.
    """
    parser = BringATrailerParser()

    make, model, trim = parser._parse_name_make_model("1991 Honda CRX Si")
    assert make == "Honda"
    assert model == "Crx"
    assert trim == "Si"

    make, model, trim = parser._parse_name_make_model("1989 Honda Civic Cx Hatchback 5-Speed")
    assert make == "Honda"
    assert model == "Civic"
    assert trim == "Cx Hatchback 5-Speed"

    # Two-word make: the make consumes two tokens, so ``Defender`` is
    # still the first model token. Tail ("90") becomes the trim.
    make, model, trim = parser._parse_name_make_model("2018 Land Rover Defender 90")
    assert make == "Land Rover"
    assert model == "Defender"
    assert trim == "90"


def test_parse_name_make_model_unit_skips_prefix_and_finds_year() -> None:
    """Direct unit test on the helper — confirms the (make, model, trim)
    tuple shape without going through the full ``_parse_listing`` flow."""
    parser = BringATrailerParser()
    make, model, trim = parser._parse_name_make_model(
        "No Reserve: Original-Owner 1986 Honda Civic Si"
    )
    assert make == "Honda"
    assert model == "Civic"
    assert trim == "Si"


def test_parse_name_make_model_unit_empty_or_unknown() -> None:
    """Empty / non-matching names return (None, None, None)."""
    parser = BringATrailerParser()
    assert parser._parse_name_make_model(None) == (None, None, None)
    assert parser._parse_name_make_model("") == (None, None, None)
    # No year, no known make → nothing to anchor on.
    assert parser._parse_name_make_model("Some random text 4242") == (None, None, None)


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
