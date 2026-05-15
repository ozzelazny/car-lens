"""Tests for the shared parser utilities in ``parsers/common``."""

from __future__ import annotations

from bs4 import BeautifulSoup
from bs4.element import Tag

from car_lense_engine.crawler.parsers.common import (
    extract_jsonld,
    find_jsonld_by_type,
    find_links,
    find_next_page,
    is_next_link,
    normalize_url,
    parse_int_safe,
    parse_year_safe,
    sha256_text,
)

# ---------- extract_jsonld ---------------------------------------------------


def test_extract_jsonld_finds_all_blocks() -> None:
    """Plain blocks, @graph-wrapped blocks, and malformed blocks are handled."""
    html = """
    <html><head>
      <script type="application/ld+json">
      {"@context": "https://schema.org/", "@type": "Product", "name": "alpha"}
      </script>
      <script type="application/ld+json">
      {
        "@context": "https://schema.org/",
        "@graph": [
          {"@type": "Vehicle", "name": "beta"},
          {"@type": "BreadcrumbList", "name": "gamma"}
        ]
      }
      </script>
      <script type="application/ld+json">
      { not valid json here
      </script>
      <script>console.log('regular js skipped')</script>
    </head><body></body></html>
    """
    blocks = extract_jsonld(html)
    names = [b.get("name") for b in blocks]
    # malformed block is skipped, @graph is flattened, plain block is kept.
    assert names == ["alpha", "beta", "gamma"]


def test_extract_jsonld_handles_empty_html() -> None:
    assert extract_jsonld("") == []
    assert extract_jsonld("<html></html>") == []


# ---------- find_jsonld_by_type ---------------------------------------------


def test_find_jsonld_by_type_case_insensitive() -> None:
    blocks = [
        {"@type": "breadcrumbList", "name": "bc"},
        {"@type": "VEHICLE", "name": "car"},
    ]
    found = find_jsonld_by_type(blocks, "Vehicle")
    assert found is not None
    assert found["name"] == "car"


def test_find_jsonld_by_type_handles_list_types() -> None:
    """``@type`` may be a list per JSON-LD spec."""
    blocks = [
        {"@type": ["Thing", "Product", "Vehicle"], "name": "multi"},
    ]
    found = find_jsonld_by_type(blocks, "Vehicle")
    assert found is not None
    assert found["name"] == "multi"


def test_find_jsonld_by_type_returns_none_when_absent() -> None:
    blocks = [{"@type": "BreadcrumbList"}]
    assert find_jsonld_by_type(blocks, "Vehicle") is None


def test_find_jsonld_by_type_returns_first_match() -> None:
    blocks = [
        {"@type": "Vehicle", "name": "first"},
        {"@type": "Vehicle", "name": "second"},
    ]
    found = find_jsonld_by_type(blocks, "Vehicle")
    assert found is not None
    assert found["name"] == "first"


# ---------- normalize_url ----------------------------------------------------


def test_normalize_url_strips_utm() -> None:
    base = "https://www.cars.com/"
    result = normalize_url(
        base,
        "/vehicledetail/123/?utm_source=email&utm_medium=ad&color=red",
    )
    assert result == "https://www.cars.com/vehicledetail/123/?color=red"


def test_normalize_url_strips_fragment() -> None:
    base = "https://www.cars.com/"
    result = normalize_url(base, "/page/?x=1#photos")
    assert result == "https://www.cars.com/page/?x=1"


def test_normalize_url_resolves_relative() -> None:
    base = "https://www.cars.com/shopping/results/"
    assert normalize_url(base, "/vehicledetail/123/") == "https://www.cars.com/vehicledetail/123/"
    assert normalize_url(base, "?page=2") == "https://www.cars.com/shopping/results/?page=2"


def test_normalize_url_keeps_non_utm_params() -> None:
    base = "https://www.cars.com/"
    result = normalize_url(base, "/x?a=1&utm_source=foo&b=2")
    assert result == "https://www.cars.com/x?a=1&b=2"


# ---------- parse_int_safe ---------------------------------------------------


def test_parse_int_safe_extracts_digits() -> None:
    assert parse_int_safe("12,345 mi") == 12345
    assert parse_int_safe("$45,000") == 45000
    assert parse_int_safe("100") == 100


def test_parse_int_safe_handles_none_and_empty() -> None:
    assert parse_int_safe(None) is None
    assert parse_int_safe("") is None
    assert parse_int_safe("no digits here") is None


# ---------- parse_year_safe -------------------------------------------------


def test_parse_year_safe_at_start_and_middle() -> None:
    assert parse_year_safe("2020 Honda Civic") == 2020
    assert parse_year_safe("Civic, 2020 model") == 2020


def test_parse_year_safe_absent_or_out_of_range() -> None:
    assert parse_year_safe("not a year here") is None
    assert parse_year_safe("1899 too old") is None
    assert parse_year_safe("2099 too future") is None


def test_parse_year_safe_handles_none() -> None:
    assert parse_year_safe(None) is None


# ---------- sha256_text -----------------------------------------------------


def test_sha256_text_deterministic() -> None:
    a = sha256_text("hello world")
    b = sha256_text("hello world")
    assert a == b
    assert len(a) == 64
    assert all(c in "0123456789abcdef" for c in a)


def test_sha256_text_changes_with_input() -> None:
    assert sha256_text("a") != sha256_text("b")


# ---------- find_links ------------------------------------------------------


def test_find_links_dedupes_preserve_order() -> None:
    html = """
    <html><body>
      <a class="card" href="/x/1/">x1</a>
      <a class="card" href="/x/2/">x2</a>
      <a class="card" href="/x/1/">x1 again</a>
      <a class="other" href="/y/1/">y1</a>
      <a class="card" href="/x/3/">x3</a>
    </body></html>
    """
    found = find_links(html, css_selector="a.card", base_url="https://example.com/")
    assert found == [
        "https://example.com/x/1/",
        "https://example.com/x/2/",
        "https://example.com/x/3/",
    ]


def test_find_links_skips_missing_href() -> None:
    html = '<a class="card">no href</a><a class="card" href="">empty</a>'
    found = find_links(html, css_selector="a.card", base_url="https://x.com/")
    assert found == []


# ---------- is_next_link / find_next_page -----------------------------------


def _single_anchor(html: str) -> Tag:
    """Helper — parse and return the first <a> tag."""
    soup = BeautifulSoup(html, features="lxml")
    anchor = soup.find("a")
    assert isinstance(anchor, Tag)
    return anchor


def test_is_next_link_rel_next() -> None:
    anchor = _single_anchor('<a rel="next" href="/p2">Continue</a>')
    assert is_next_link(anchor) is True


def test_is_next_link_aria_label_contains_next() -> None:
    anchor = _single_anchor('<a aria-label="Go to next page" href="/p2">›</a>')
    assert is_next_link(anchor) is True


def test_is_next_link_text_contains_next() -> None:
    anchor = _single_anchor('<a href="/p2">Next page</a>')
    assert is_next_link(anchor) is True


def test_is_next_link_excludes_previous() -> None:
    anchor = _single_anchor('<a href="/p1">Previous page</a>')
    assert is_next_link(anchor) is False


def test_is_next_link_excludes_prev() -> None:
    anchor = _single_anchor('<a href="/p1">Prev</a>')
    assert is_next_link(anchor) is False


def test_is_next_link_unrelated_text_is_false() -> None:
    anchor = _single_anchor('<a href="/">Home</a>')
    assert is_next_link(anchor) is False


def test_find_next_page_returns_first_match() -> None:
    html = """
    <html><body>
      <a href="/prev">Previous</a>
      <a href="/p3" rel="next">Next page</a>
      <a href="/p4">Next</a>
    </body></html>
    """
    soup = BeautifulSoup(html, features="lxml")
    assert find_next_page(soup, base_url="https://example.com/") == "https://example.com/p3"


def test_find_next_page_returns_none_when_absent() -> None:
    html = "<html><body><a href='/x'>Previous</a></body></html>"
    soup = BeautifulSoup(html, features="lxml")
    assert find_next_page(soup, base_url="https://example.com/") is None
