"""Craigslist parser — search pages and individual listing pages.

Craigslist is fundamentally different from cars.com and AutoTrader:

* **No JSON-LD** — old-school classifieds; everything must be scraped from HTML.
* **Free-text titles** — year/make/model must be inferred heuristically when
  the queue doesn't already carry the canonical values via ``hints``.
* **Many city subdomains** — ``newyork.craigslist.org``,
  ``losangeles.craigslist.org``, etc. — so the listing-href regex matches any
  ``<city>.craigslist.org`` host.
* **Two coexisting HTML layouts** — Craigslist has been refreshing their UI;
  old (`<span id="titletextonly">`, `result-title` anchors) and new
  (`<h1 class="postingtitle">`, `posting-title` anchors) shapes both appear in
  the wild.

The parser handles both shapes and leans on the queue's ``hints`` whenever
they're populated, falling back to title-heuristic extraction only as a last
resort. Every error path returns a :class:`ParseResult` with notes rather
than raising, since the worker treats exceptions as hard failures.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from bs4.element import Tag

from .base import DiscoveredUrl, ParsedListing, ParseResult
from .common import normalize_url, parse_int_safe, parse_year_safe, sha256_text

logger = logging.getLogger(__name__)


# Listing-href shape: ``https://<city>.craigslist.org/<area>/cto|ctd/d/<slug>/<id>.html``
# - ``cto`` = cars/trucks by owner; ``ctd`` = cars/trucks by dealer. Match both.
# - ``<city>`` is any lowercase-alnum subdomain (no enumeration — Craigslist
#   has hundreds and adds new ones).
# - The trailing ``<id>`` is a 6+ digit run; that's our native listing id.
_LISTING_HREF_RE = re.compile(
    r"^https?://[a-z0-9-]+\.craigslist\.org/[^/]+/ct[od]/d/[^/]+/\d{6,}\.html$"
)

# Pulls the trailing digit run out of a listing URL path.
_LISTING_ID_FROM_URL_RE = re.compile(r"/(\d{6,})\.html$")

# Defensive VIN regex — 17 chars, excludes I, O, Q per the VIN spec.
_VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")

# Token boundary characters for the title heuristic. We split on whitespace,
# punctuation, and a few delimiters that commonly precede price / location
# qualifiers in Craigslist titles.
_TITLE_SPLIT_RE = re.compile(r"[\s,/]+")

# Token "boundary" markers — when scanning forward for the model after the
# make, we stop at any of these. ``-`` is intentionally NOT in this set
# because models like ``F-150`` and ``CX-5`` contain a hyphen mid-name.
_MODEL_BOUNDARY_TOKENS: frozenset[str] = frozenset({"$", "(", ")", "|", "·"})


class CraigslistParser:
    """Per-site parser for Craigslist search and listing pages."""

    source: str = "craigslist"

    def __init__(self, *, known_makes: frozenset[str] | None = None) -> None:
        """Build a parser.

        Parameters
        ----------
        known_makes:
            Pool of make names (lowercased) used by the title heuristic when
            no queue hints are available. Defaults to a snapshot of
            :data:`~car_lense_engine.crawler.seed.ranker.MAKE_POPULARITY`
            keys, lowercased. Override for tests or to widen the pool.
        """
        self._known_makes: frozenset[str] = (
            known_makes if known_makes is not None else self._default_known_makes()
        )

    @staticmethod
    def _default_known_makes() -> frozenset[str]:
        """Return the lowercased set of well-known US makes."""
        # Imported inside the method to avoid module-load circular risk; the
        # ranker imports the catalog layer, which we don't want to drag in at
        # parser-package import time.
        from car_lense_engine.crawler.seed.ranker import MAKE_POPULARITY

        return frozenset(name.lower() for name in MAKE_POPULARITY)

    def parse(
        self,
        *,
        html: str,
        url: str,
        kind: str,
        hints: dict[str, str | int | None],
    ) -> ParseResult:
        if kind == "search":
            return self._parse_search(html=html, url=url, hints=hints)
        if kind == "listing":
            return self._parse_listing(html=html, url=url, hints=hints)
        if kind == "image":
            return ParseResult(
                notes=["image kind is a no-op for craigslist (downloads handled by image pipeline)"]
            )
        return ParseResult(notes=[f"unknown kind: {kind}"])

    # ------------------------------------------------------------------ search

    def _parse_search(
        self,
        *,
        html: str,
        url: str,
        hints: dict[str, str | int | None],
    ) -> ParseResult:
        soup = BeautifulSoup(html, features="lxml")
        target_year = _as_int(hints.get("target_year"))
        target_make = _as_str(hints.get("target_make"))
        target_model = _as_str(hints.get("target_model"))

        listing_urls: list[str] = []
        seen: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href")
            if not isinstance(href, str):
                continue
            stripped = href.strip()
            if not _LISTING_HREF_RE.match(stripped):
                continue
            absolute = normalize_url(url, stripped)
            if absolute in seen:
                continue
            seen.add(absolute)
            listing_urls.append(absolute)

        new_urls: list[DiscoveredUrl] = [
            DiscoveredUrl(
                url=listing_url,
                source=self.source,
                kind="listing",
                target_year=target_year,
                target_make=target_make,
                target_model=target_model,
            )
            for listing_url in listing_urls
        ]

        # Pagination — accept rel=next / aria-label containing "next" / inner
        # text containing "next" (excluding "previous"/"prev"). Craigslist
        # most often uses ``<a class="button next">Next ›</a>`` or
        # ``<a class="cl-next-page">``; both are picked up by text matching.
        next_url = _find_next_page(soup, base_url=url)
        if next_url is not None:
            new_urls.append(
                DiscoveredUrl(
                    url=next_url,
                    source=self.source,
                    kind="search",
                    target_year=target_year,
                    target_make=target_make,
                    target_model=target_model,
                )
            )

        if not listing_urls:
            return ParseResult(
                new_urls=new_urls,
                notes=["no listing cards found on search page; selectors may need updating"],
            )
        return ParseResult(new_urls=new_urls)

    # ------------------------------------------------------------------ listing

    def _parse_listing(
        self,
        *,
        html: str,
        url: str,
        hints: dict[str, str | int | None],
    ) -> ParseResult:
        soup = BeautifulSoup(html, features="lxml")

        native_id = _extract_native_id(url) or _extract_native_id_from_html(soup)
        if native_id is None:
            return ParseResult(notes=[f"could not extract listing_id from URL: {url}"])

        title = _extract_title(soup)

        # Hints-first: prefer canonical values the seeder routed this URL for.
        # Only fall back to title parsing for fields the hints leave blank.
        hint_year = _as_int(hints.get("target_year"))
        hint_make = _as_str(hints.get("target_make"))
        hint_model = _as_str(hints.get("target_model"))

        title_year, title_make, title_model = self._parse_title_heuristic(title)

        year = hint_year if hint_year is not None else title_year
        make = hint_make if hint_make is not None else title_make
        model = hint_model if hint_model is not None else title_model

        attr_text = _collect_attrgroup_text(soup)
        body_text = _extract_body_text(soup)

        mileage = _extract_attr_int(attr_text, "odometer")
        body_style = _extract_attr_str(attr_text, "type")
        if body_style is not None:
            body_style = body_style.title()
        vin = _extract_vin(attr_text, body_text)
        image_urls = _extract_image_urls(soup, base_url=url)

        listing = ParsedListing(
            listing_id=f"{self.source}:{native_id}",
            source=self.source,
            url=url,
            year=year,
            make=make,
            model=model,
            trim=None,  # v1: skip trim heuristic (messy on free-text titles)
            body_style=body_style,
            mileage=mileage,
            vin=vin,
            raw_html_sha256=sha256_text(html),
            image_urls=image_urls,
        )

        notes: list[str] = []
        if title is None:
            notes.append("no title found on listing page")
        if make is None and model is None and title is not None:
            notes.append(f"could not infer make/model from title: {title!r}")
        return ParseResult(new_listing=listing, notes=notes)

    # ---------------------------------------------------------- title heuristic

    def _parse_title_heuristic(
        self, title: str | None
    ) -> tuple[int | None, str | None, str | None]:
        """Best-effort year/make/model extraction from a free-text title.

        Strategy:
          1. Find a plausible 4-digit year anywhere in the title.
          2. Lowercase + tokenise; look for the **first** token sequence
             matching a known make. Two-word makes (``land rover``,
             ``alfa romeo``, ``mercedes-benz``) are tried before single-word.
          3. Take the next 1–2 non-empty tokens as the model name, stopping
             at price/parenthesis boundaries.

        Returns
        -------
        ``(year, make, model)`` with any field set to ``None`` when it
        couldn't be inferred. Make/model are returned in Title Case.
        """
        if not title:
            return (None, None, None)

        year = parse_year_safe(title)

        tokens = [t for t in _TITLE_SPLIT_RE.split(title.strip()) if t]
        # Lowercase-but-original-cased pair lists for matching + reconstruction.
        lower_tokens = [t.lower() for t in tokens]

        make_str: str | None = None
        make_end_idx = -1  # exclusive — first token AFTER the make

        for i in range(len(lower_tokens)):
            # Try two-word make first (e.g. "land rover", "alfa romeo").
            if i + 1 < len(lower_tokens):
                two_word = f"{lower_tokens[i]} {lower_tokens[i + 1]}"
                if two_word in self._known_makes:
                    make_str = two_word.title()
                    make_end_idx = i + 2
                    break
            # Single-word make.
            if lower_tokens[i] in self._known_makes:
                # Preserve canonical-ish casing: "BMW" / "GMC" stays uppercase,
                # everything else gets Title Case.
                token = lower_tokens[i]
                make_str = token.upper() if len(token) <= 3 else token.title()
                make_end_idx = i + 1
                break

        if make_str is None or make_end_idx < 0:
            return (year, None, None)

        # Collect the next 1–2 tokens as the model, stopping at boundary chars.
        model_tokens: list[str] = []
        for j in range(make_end_idx, len(lower_tokens)):
            tok = tokens[j]
            # Stop at price-style ($) / parenthesis / pipe tokens.
            if tok in _MODEL_BOUNDARY_TOKENS:
                break
            if tok.startswith("$") or tok.startswith("(") or tok.startswith("|"):
                break
            # Skip a bare year token — happens when the title shape is
            # "make model year" (e.g. "ford f-150 supercrew 2019").
            if year is not None and tok == str(year):
                continue
            model_tokens.append(tok)
            if len(model_tokens) >= 2:
                break

        if not model_tokens:
            return (year, make_str, None)

        # Preserve the original casing for things like "F-150" by Title-casing
        # only the alpha portions and leaving hyphen/digit segments intact.
        model_str = " ".join(_titlecase_model_token(t) for t in model_tokens)
        return (year, make_str, model_str)


# ---------- helpers ----------------------------------------------------------


def _is_next_link(anchor: Tag) -> bool:
    """True if the anchor looks like a 'next page' link (case-insensitive).

    Recognises three signals, in order of strength:

    * ``rel="next"``
    * ``aria-label`` containing ``"next"``
    * inner text containing ``"next"`` while excluding ``"previous"``/``"prev"``
    """
    rel = anchor.get("rel")
    if isinstance(rel, list) and "next" in {str(r).lower() for r in rel}:
        return True

    aria = anchor.get("aria-label")
    if isinstance(aria, str) and "next" in aria.lower():
        return True

    text = anchor.get_text(strip=True).lower()
    return "next" in text and "previous" not in text and "prev" not in text


def _find_next_page(soup: BeautifulSoup, *, base_url: str) -> str | None:
    """Locate a pagination "next" link via :func:`_is_next_link`."""
    for anchor in soup.find_all("a", href=True):
        if not _is_next_link(anchor):
            continue
        href = anchor.get("href")
        if isinstance(href, str) and href.strip():
            return normalize_url(base_url, href.strip())
    return None


def _extract_native_id(url: str) -> str | None:
    """Pull the trailing digit run from a Craigslist listing URL path."""
    path = urlparse(url).path
    match = _LISTING_ID_FROM_URL_RE.search(path)
    if match is None:
        return None
    return match.group(1)


def _extract_native_id_from_html(soup: BeautifulSoup) -> str | None:
    """Fallback: look for ``post id: <digits>`` in ``<p class="postinginfo">``.

    Older Craigslist layouts surface the post id explicitly inside a
    ``<p class="postinginfos">`` container.
    """
    for p in soup.find_all("p", class_="postinginfo"):
        text = p.get_text(" ", strip=True).lower()
        if "post id" not in text:
            continue
        match = re.search(r"post id:\s*(\d{6,})", text)
        if match is not None:
            return match.group(1)
    return None


def _extract_title(soup: BeautifulSoup) -> str | None:
    """Locate the listing title across legacy and modern Craigslist layouts."""
    # Legacy layout.
    legacy = soup.find("span", id="titletextonly")
    if isinstance(legacy, Tag):
        text = legacy.get_text(strip=True)
        if text:
            return text

    # Newer layout — full <h1 class="postingtitle"> contains a price prefix and
    # the title text as a child <span>. Prefer the inner title span if present,
    # otherwise fall back to the h1's stripped text.
    h1 = soup.find("h1", class_="postingtitle")
    if isinstance(h1, Tag):
        inner = h1.find("span", id="titletextonly")
        if isinstance(inner, Tag):
            text = inner.get_text(strip=True)
            if text:
                return text
        text = h1.get_text(" ", strip=True)
        if text:
            return text

    return None


def _collect_attrgroup_text(soup: BeautifulSoup) -> str:
    """Concatenate every ``<p class="attrgroup">`` span's text, lowercase."""
    parts: list[str] = []
    for group in soup.find_all("p", class_="attrgroup"):
        for span in group.find_all("span"):
            parts.append(span.get_text(" ", strip=True))
    return " | ".join(parts).lower()


def _extract_body_text(soup: BeautifulSoup) -> str:
    """Text content of ``<section id="postingbody">`` (whitespace-collapsed)."""
    section = soup.find("section", id="postingbody")
    if isinstance(section, Tag):
        return section.get_text(" ", strip=True)
    return ""


def _extract_attr_str(attr_text: str, key: str) -> str | None:
    """Pull ``<key>: <value>`` out of the attribute-group blob.

    ``attr_text`` is pre-lowercased; ``key`` should be lowercase too. Returns
    the raw value up to the next pipe / end-of-text, with whitespace
    collapsed and no trailing punctuation.
    """
    pattern = re.compile(rf"{re.escape(key)}:\s*([^|]+?)(?:\s*\||$)")
    match = pattern.search(attr_text)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _extract_attr_int(attr_text: str, key: str) -> int | None:
    """Like :func:`_extract_attr_str` but parses the value as an int."""
    value = _extract_attr_str(attr_text, key)
    if value is None:
        return None
    return parse_int_safe(value)


def _extract_vin(attr_text: str, body_text: str) -> str | None:
    """VIN from attrgroup first (``VIN: <17chars>``), else regex over body."""
    # Attrgroup path — case-insensitive because attr_text is already lowered.
    match = re.search(r"vin:\s*([a-hj-npr-z0-9]{17})\b", attr_text)
    if match is not None:
        return match.group(1).upper()

    # Body text fallback — search uppercase since the regex is uppercase-only.
    body_match = _VIN_RE.search(body_text.upper())
    if body_match is not None:
        return body_match.group(0)

    return None


def _extract_image_urls(soup: BeautifulSoup, *, base_url: str) -> list[str]:
    """Pull image URLs from gallery anchors and inline ``<img>`` tags.

    Looks at ``<a class="thumb" href="...">`` first (full-size links) and
    falls back to ``<img src="...">`` within ``<div class="gallery">`` or
    ``<div class="swipe">``. Keeps only http(s) URLs; dedups order-preserving.
    """
    seen: set[str] = set()
    urls: list[str] = []

    def _push(candidate: str | None) -> None:
        if not candidate:
            return
        absolute = normalize_url(base_url, candidate.strip())
        if not (absolute.startswith("http://") or absolute.startswith("https://")):
            return
        if absolute in seen:
            return
        seen.add(absolute)
        urls.append(absolute)

    # Full-size gallery anchors.
    for anchor in soup.find_all("a", class_="thumb"):
        href = anchor.get("href") if isinstance(anchor, Tag) else None
        if isinstance(href, str):
            _push(href)

    # Inline <img> tags within gallery containers.
    for container in soup.find_all("div", class_=re.compile(r"\b(gallery|swipe)\b")):
        if not isinstance(container, Tag):
            continue
        for img in container.find_all("img"):
            src = img.get("src") if isinstance(img, Tag) else None
            if isinstance(src, str):
                _push(src)

    return urls


def _titlecase_model_token(token: str) -> str:
    """Title-case alpha runs in a token while preserving digits and hyphens.

    Examples:
        ``"f-150"`` → ``"F-150"``
        ``"civic"`` → ``"Civic"``
        ``"cx-5"``  → ``"Cx-5"``  (acceptable; trims/sub-models stay lossy)
    """
    return re.sub(r"[a-zA-Z]+", lambda m: m.group(0).capitalize(), token)


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None
