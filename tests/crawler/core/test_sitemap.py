"""Tests for :class:`SitemapWalker`.

The walker is exercised against a fake fetcher that returns canned
:class:`FetchedPage` objects for known URLs. No network access.
"""

from __future__ import annotations

import gzip
from datetime import datetime
from typing import Any

import pytest

from car_lense_engine.crawler.core.fetcher import FetchedPage, FetchError
from car_lense_engine.crawler.core.sitemap import SitemapWalker


class _FakeFetcher:
    """Maps URL -> response body (str). Records every fetch() call."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def fetch(self, url: str) -> FetchedPage:
        self.calls.append(url)
        if url not in self._responses:
            raise FetchError(f"unmocked URL: {url}")
        body = self._responses[url]
        return FetchedPage(
            url=url,
            status=200,
            html=body,
            fetched_at=datetime(2026, 5, 15, 12, 0, 0),
        )

    def close(self) -> None:
        pass


def _urlset(locs: list[str]) -> str:
    """Build a sitemap-protocol ``<urlset>`` XML body with the given locs."""
    inner = "".join(f"<url><loc>{loc}</loc></url>" for loc in locs)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{inner}</urlset>"
    )


def _sitemapindex(locs: list[str]) -> str:
    """Build a ``<sitemapindex>`` XML body referencing child sitemap URLs."""
    inner = "".join(f"<sitemap><loc>{loc}</loc></sitemap>" for loc in locs)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{inner}</sitemapindex>"
    )


# --------------------------------------------------------------------- tests


def test_walker_yields_locs_from_urlset() -> None:
    """A minimal urlset with one URL yields that URL."""
    fetcher = _FakeFetcher(
        {
            "https://example.com/sitemap.xml": _urlset(
                ["https://example.com/cars-for-sale/vehicledetails/123456/"]
            ),
        }
    )
    walker = SitemapWalker(fetcher=fetcher)
    result = list(walker.walk("https://example.com/sitemap.xml"))
    assert result == ["https://example.com/cars-for-sale/vehicledetails/123456/"]


def test_walker_recurses_sitemapindex() -> None:
    """A sitemapindex referencing 2 child sitemaps (each 2 locs) yields all 4 locs."""
    fetcher = _FakeFetcher(
        {
            "https://example.com/sitemap.xml": _sitemapindex(
                [
                    "https://example.com/sitemap_a.xml",
                    "https://example.com/sitemap_b.xml",
                ]
            ),
            "https://example.com/sitemap_a.xml": _urlset(
                [
                    "https://example.com/a1",
                    "https://example.com/a2",
                ]
            ),
            "https://example.com/sitemap_b.xml": _urlset(
                [
                    "https://example.com/b1",
                    "https://example.com/b2",
                ]
            ),
        }
    )
    walker = SitemapWalker(fetcher=fetcher)
    result = list(walker.walk("https://example.com/sitemap.xml"))
    # Depth-first, in-document order: a1, a2 (from sitemap_a), then b1, b2.
    assert result == [
        "https://example.com/a1",
        "https://example.com/a2",
        "https://example.com/b1",
        "https://example.com/b2",
    ]


def test_walker_respects_max_depth() -> None:
    """A 3-deep index tree with max_depth=2 yields only level-1 + level-2 URLs.

    Layout:
      root.xml (level 0, index) -> level1.xml
      level1.xml (level 1, index) -> level2.xml
      level2.xml (level 2, index) -> level3.xml
      level3.xml (level 3, urlset with /deep)

    max_depth=2 lets us recurse from level 0 -> 1 -> 2 (recursing into
    level3.xml would be depth=3, which is > max_depth=2, so it's refused).
    The result is empty because level2.xml is an index (no <loc>s at its
    own level) and we refuse to follow into level3.xml.
    """
    fetcher = _FakeFetcher(
        {
            "https://e.test/root.xml": _sitemapindex(["https://e.test/level1.xml"]),
            "https://e.test/level1.xml": _sitemapindex(["https://e.test/level2.xml"]),
            "https://e.test/level2.xml": _sitemapindex(["https://e.test/level3.xml"]),
            "https://e.test/level3.xml": _urlset(["https://e.test/deep"]),
        }
    )
    walker = SitemapWalker(fetcher=fetcher, max_depth=2)
    result = list(walker.walk("https://e.test/root.xml"))
    # The walker refuses to recurse into level3 (which would be depth=3).
    # Confirms the leaf at level 3 is never reached.
    assert "https://e.test/deep" not in result
    assert result == []
    # And critically, level3.xml was never fetched.
    assert "https://e.test/level3.xml" not in fetcher.calls

    # Sanity: with max_depth=3, the deep URL DOES surface (showing the cap
    # is what's blocking it, not a different bug).
    fetcher2 = _FakeFetcher(
        {
            "https://e.test/root.xml": _sitemapindex(["https://e.test/level1.xml"]),
            "https://e.test/level1.xml": _sitemapindex(["https://e.test/level2.xml"]),
            "https://e.test/level2.xml": _sitemapindex(["https://e.test/level3.xml"]),
            "https://e.test/level3.xml": _urlset(["https://e.test/deep"]),
        }
    )
    walker2 = SitemapWalker(fetcher=fetcher2, max_depth=3)
    result2 = list(walker2.walk("https://e.test/root.xml"))
    assert result2 == ["https://e.test/deep"]


def test_walker_respects_max_urls() -> None:
    """A single urlset with 100 locs yields exactly the first 10 with max_urls=10."""
    locs = [f"https://example.com/listing/{i}" for i in range(100)]
    fetcher = _FakeFetcher({"https://example.com/sm.xml": _urlset(locs)})
    walker = SitemapWalker(fetcher=fetcher, max_urls=10)
    result = list(walker.walk("https://example.com/sm.xml"))
    assert len(result) == 10
    assert result == locs[:10]


def test_walker_handles_gzipped_sitemap() -> None:
    """Body returned as gzip-encoded bytes (decoded via latin-1 into html str)
    is transparently decompressed before parsing."""
    xml = _urlset(["https://example.com/gz/1", "https://example.com/gz/2"])
    gz_bytes = gzip.compress(xml.encode("utf-8"))
    # Mirror what curl_cffi does when no charset is announced: decode bytes
    # via latin-1 so every byte round-trips. This is the realistic shape the
    # FetchedPage.html field arrives in for binary content.
    gz_as_str = gz_bytes.decode("latin-1")

    fetcher = _FakeFetcher({"https://example.com/sitemap.xml.gz": gz_as_str})
    walker = SitemapWalker(fetcher=fetcher)
    result = list(walker.walk("https://example.com/sitemap.xml.gz"))
    assert result == ["https://example.com/gz/1", "https://example.com/gz/2"]


def test_walker_handles_namespaces() -> None:
    """XML using the standard sitemap namespace is parsed correctly.

    ``_urlset`` already emits the standard namespace; verify nothing else
    breaks even when a vendor-style prefix is mixed in. We re-confirm the
    basic urlset case here for the namespace contract.
    """
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
        '        xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">'
        "<url>"
        "  <loc>https://example.com/listing/1</loc>"
        "  <image:image><image:loc>https://example.com/img/1.jpg</image:loc></image:image>"
        "</url>"
        "</urlset>"
    )
    fetcher = _FakeFetcher({"https://example.com/sm.xml": body})
    walker = SitemapWalker(fetcher=fetcher)
    result = list(walker.walk("https://example.com/sm.xml"))
    # Only the canonical <loc> at the <url> level should surface — the
    # <image:loc> child of <image:image> should not be confused for it.
    assert result == ["https://example.com/listing/1"]


def test_walker_skips_malformed_xml() -> None:
    """Malformed XML at the root yields nothing (logged, no exception)."""
    fetcher = _FakeFetcher({"https://example.com/bad.xml": "<not really xml <<>"})
    walker = SitemapWalker(fetcher=fetcher)
    result = list(walker.walk("https://example.com/bad.xml"))
    assert result == []


def test_walker_skips_malformed_child_sitemap_but_continues_siblings() -> None:
    """A malformed child sitemap inside an index doesn't kill the whole walk."""
    fetcher = _FakeFetcher(
        {
            "https://e.test/root.xml": _sitemapindex(
                [
                    "https://e.test/broken.xml",
                    "https://e.test/good.xml",
                ]
            ),
            "https://e.test/broken.xml": "<not really xml <<>",
            "https://e.test/good.xml": _urlset(["https://e.test/ok"]),
        }
    )
    walker = SitemapWalker(fetcher=fetcher)
    result = list(walker.walk("https://e.test/root.xml"))
    assert result == ["https://e.test/ok"]


def test_walker_handles_fetch_failure_gracefully() -> None:
    """A FetchError on the root URL yields no URLs (no exception)."""

    class _BoomFetcher:
        def fetch(self, url: str) -> FetchedPage:
            raise FetchError(f"HTTP 503 for {url}")

        def close(self) -> None:
            pass

    walker = SitemapWalker(fetcher=_BoomFetcher())
    result = list(walker.walk("https://example.com/sm.xml"))
    assert result == []


def test_walker_rejects_invalid_max_depth() -> None:
    """Negative max_depth must raise ValueError at construction time."""

    class _Stub:
        def fetch(self, url: str) -> Any:  # pragma: no cover - unused
            raise NotImplementedError

        def close(self) -> None:  # pragma: no cover - unused
            pass

    with pytest.raises(ValueError, match="max_depth"):
        SitemapWalker(fetcher=_Stub(), max_depth=-1)


def test_walker_rejects_invalid_max_urls() -> None:
    """Zero/negative max_urls must raise ValueError at construction time."""

    class _Stub:
        def fetch(self, url: str) -> Any:  # pragma: no cover - unused
            raise NotImplementedError

        def close(self) -> None:  # pragma: no cover - unused
            pass

    with pytest.raises(ValueError, match="max_urls"):
        SitemapWalker(fetcher=_Stub(), max_urls=0)


def test_walker_dedupes_cycle_in_sitemap_tree() -> None:
    """If a sitemapindex points (transitively) back to itself, we don't loop forever."""
    fetcher = _FakeFetcher(
        {
            "https://e.test/root.xml": _sitemapindex(["https://e.test/child.xml"]),
            "https://e.test/child.xml": _sitemapindex(["https://e.test/root.xml"]),
        }
    )
    walker = SitemapWalker(fetcher=fetcher)
    result = list(walker.walk("https://e.test/root.xml"))
    assert result == []
    # Each URL fetched exactly once.
    assert sorted(fetcher.calls) == [
        "https://e.test/child.xml",
        "https://e.test/root.xml",
    ]


def test_walker_max_urls_cap_across_sitemapindex() -> None:
    """When the cap is hit mid-recursion, the walker stops yielding promptly."""
    fetcher = _FakeFetcher(
        {
            "https://e.test/root.xml": _sitemapindex(
                [
                    "https://e.test/sm1.xml",
                    "https://e.test/sm2.xml",
                ]
            ),
            "https://e.test/sm1.xml": _urlset([f"https://e.test/a/{i}" for i in range(5)]),
            "https://e.test/sm2.xml": _urlset([f"https://e.test/b/{i}" for i in range(5)]),
        }
    )
    walker = SitemapWalker(fetcher=fetcher, max_urls=3)
    result = list(walker.walk("https://e.test/root.xml"))
    assert len(result) == 3
    # First three from sm1 (depth-first).
    assert result == [
        "https://e.test/a/0",
        "https://e.test/a/1",
        "https://e.test/a/2",
    ]
