"""Shared parser utilities used by every per-site parser.

These helpers are intentionally small and conservative — each accepts arbitrary
HTML and returns ``None`` / an empty result rather than raising on unexpected
shapes. The worker treats any uncaught exception as a hard failure, so parsers
must swallow content quirks here and surface them via :class:`ParseResult.notes`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
from bs4.element import Tag

logger = logging.getLogger(__name__)


# ---------- JSON-LD ----------------------------------------------------------


def extract_jsonld(html: str) -> list[dict[str, Any]]:
    """Return every JSON-LD block as a flat list of dicts.

    Walks ``<script type="application/ld+json">`` blocks, parses each, and
    flattens any top-level ``@graph`` wrappers so callers can search by
    ``@type`` without worrying about how the site nests its structured data.

    Blocks that fail JSON parsing are skipped (and logged at debug).
    """
    soup = BeautifulSoup(html, features="lxml")
    blocks: list[dict[str, Any]] = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.debug("skipping malformed JSON-LD block: %s", exc)
            continue
        # JSON-LD may be a single object, a list of objects, or a wrapper
        # containing an @graph list. Flatten all three into a single list.
        for entry in _flatten_jsonld(parsed):
            if isinstance(entry, dict):
                blocks.append(entry)
    return blocks


def _flatten_jsonld(node: Any) -> list[Any]:
    """Flatten JSON-LD nesting: lists, @graph wrappers, single objects."""
    if isinstance(node, list):
        out: list[Any] = []
        for item in node:
            out.extend(_flatten_jsonld(item))
        return out
    if isinstance(node, dict):
        graph = node.get("@graph")
        if isinstance(graph, list):
            return _flatten_jsonld(graph)
        return [node]
    return []


def find_jsonld_by_type(
    blocks: list[dict[str, Any]],
    type_name: str,
) -> dict[str, Any] | None:
    """Return the first block whose ``@type`` matches ``type_name``.

    ``@type`` may be a string or a list of strings (the JSON-LD spec allows
    both). Matching is case-insensitive.
    """
    needle = type_name.lower()
    for block in blocks:
        block_type = block.get("@type")
        if isinstance(block_type, str):
            if block_type.lower() == needle:
                return block
        elif isinstance(block_type, list):
            for entry in block_type:
                if isinstance(entry, str) and entry.lower() == needle:
                    return block
    return None


# ---------- URL handling -----------------------------------------------------


def normalize_url(base: str, href: str) -> str:
    """Resolve ``href`` against ``base`` and drop noise.

    Specifically:

    * resolves relative URLs against ``base`` via :func:`urljoin`
    * strips any URL fragment (``#section``)
    * strips tracking query parameters whose name starts with ``utm_``
    """
    absolute = urljoin(base, href)
    parts = urlparse(absolute)

    cleaned_query = _strip_utm_params(parts.query)
    cleaned = parts._replace(fragment="", query=cleaned_query)
    return urlunparse(cleaned)


def _strip_utm_params(query: str) -> str:
    """Remove every ``utm_*`` param from a raw query string.

    We re-build the query manually rather than using ``parse_qsl`` so that
    parameters without values (``?foo``) and duplicate keys round-trip exactly.
    """
    if not query:
        return ""
    kept: list[str] = []
    for pair in query.split("&"):
        if not pair:
            continue
        name = pair.split("=", 1)[0]
        if name.startswith("utm_"):
            continue
        kept.append(pair)
    return "&".join(kept)


def is_next_link(anchor: Tag) -> bool:
    """True if the anchor looks like a 'next page' link (case-insensitive).

    Recognises three signals, in order of strength:

    * ``rel="next"``
    * ``aria-label`` containing ``"next"``
    * inner text containing ``"next"`` while excluding ``"previous"``/``"prev"``

    Shared by every site whose pagination is plain-HTML rather than JS-driven
    (cars.com / AutoTrader / Craigslist / BaT / Hemmings / Cars & Bids all
    qualify). The check is intentionally lenient so it survives small markup
    drift (``"Next"`` vs ``"Next page"`` vs ``"Next ›"``).
    """
    rel = anchor.get("rel")
    if isinstance(rel, list) and "next" in {str(r).lower() for r in rel}:
        return True

    aria = anchor.get("aria-label")
    if isinstance(aria, str) and "next" in aria.lower():
        return True

    text = anchor.get_text(strip=True).lower()
    return "next" in text and "previous" not in text and "prev" not in text


def find_next_page(soup: BeautifulSoup, *, base_url: str) -> str | None:
    """Locate a pagination "next" link via :func:`is_next_link`.

    Returns the first matching anchor's resolved absolute URL, or ``None`` if
    no anchor on the page looks like a next-page link.
    """
    for anchor in soup.find_all("a", href=True):
        if not is_next_link(anchor):
            continue
        href = anchor.get("href")
        if isinstance(href, str) and href.strip():
            return normalize_url(base_url, href.strip())
    return None


def find_links(html: str, *, css_selector: str, base_url: str) -> list[str]:
    """Return absolute URLs for every ``<a>`` matching ``css_selector``.

    Results are deduplicated while preserving first-seen order. Anchors
    without an ``href`` attribute are skipped.
    """
    soup = BeautifulSoup(html, features="lxml")
    seen: set[str] = set()
    urls: list[str] = []
    for tag in soup.select(css_selector):
        href = tag.get("href")
        if not isinstance(href, str) or not href.strip():
            continue
        absolute = normalize_url(base_url, href.strip())
        if absolute in seen:
            continue
        seen.add(absolute)
        urls.append(absolute)
    return urls


# ---------- Hashing ----------------------------------------------------------


def sha256_text(s: str) -> str:
    """Return the hex SHA-256 of ``s`` encoded as UTF-8."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------- Cheap value parsers ----------------------------------------------


_DIGITS_RE = re.compile(r"\d+")
_YEAR_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def parse_int_safe(s: str | None) -> int | None:
    """Strip non-digit characters and return the result as int, else ``None``.

    Useful for shapes like ``"12,345 mi"`` or ``"$45,000"``.
    """
    if s is None:
        return None
    digits = "".join(_DIGITS_RE.findall(s))
    if not digits:
        return None
    return int(digits)


def parse_year_safe(s: str | None) -> int | None:
    """Find a 4-digit year in ``s`` and return it if it's plausible.

    Plausible = ``1900 <= year <= current_year + 1``.
    """
    if s is None:
        return None
    upper = datetime.now().year + 1
    for match in _YEAR_RE.finditer(s):
        candidate = int(match.group(1))
        if 1900 <= candidate <= upper:
            return candidate
    return None
