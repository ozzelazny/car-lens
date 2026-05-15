"""cars.com parser — search pages and individual listing pages.

Listing pages embed a JSON-LD ``Vehicle`` (sometimes ``Car`` or ``Product``)
block we read directly. Search pages we scrape via CSS selectors plus a
regex on listing-card hrefs of the form ``/vehicledetail/<native_id>/``.

Both flows are defensive: any unexpected shape is logged via
:class:`ParseResult.notes` rather than raising, since the worker treats
exceptions as hard failures.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bs4 import BeautifulSoup

from .base import DiscoveredUrl, ParsedListing, ParseResult
from .common import (
    extract_jsonld,
    find_jsonld_by_type,
    normalize_url,
    parse_int_safe,
    parse_year_safe,
    sha256_text,
)

logger = logging.getLogger(__name__)


# Listing-card href shape: /vehicledetail/{native_id}/ (trailing slash optional).
_LISTING_HREF_RE = re.compile(r"^/vehicledetail/[^/]+/?$")

# Native ID extractor from a full listing URL path.
_LISTING_ID_FROM_URL_RE = re.compile(r"/vehicledetail/([^/?#]+)/?")


class CarsComParser:
    """Per-site parser for cars.com search and listing pages."""

    source: str = "cars_com"

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
                notes=["image kind is a no-op for cars_com (downloads handled by image pipeline)"]
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
            if not _LISTING_HREF_RE.match(href.strip()):
                continue
            absolute = normalize_url(url, href.strip())
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

        # Pagination — accept rel=next / aria-label="Next" / text "Next".
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
        del hints  # listing extraction does not need target_* hints

        blocks = extract_jsonld(html)
        vehicle = (
            find_jsonld_by_type(blocks, "Vehicle")
            or find_jsonld_by_type(blocks, "Car")
            or find_jsonld_by_type(blocks, "Product")
        )
        if vehicle is None:
            return ParseResult(notes=["no Vehicle JSON-LD found"])

        native_id = _extract_native_id(url)
        if native_id is None:
            return ParseResult(notes=[f"could not extract listing_id from URL: {url}"])

        year = parse_year_safe(
            _as_str(
                vehicle.get("vehicleModelDate")
                or vehicle.get("modelDate")
                or vehicle.get("productionDate")
            )
        )
        make = _name_or_string(vehicle.get("manufacturer"))
        model = _name_or_string(vehicle.get("model"))
        trim = _as_str(vehicle.get("vehicleConfiguration") or vehicle.get("trim"))
        mileage = _parse_mileage(vehicle.get("mileageFromOdometer"))
        vin = _as_str(vehicle.get("vehicleIdentificationNumber"))
        body_style = _as_str(vehicle.get("bodyType"))
        image_urls = _extract_image_urls(vehicle.get("image"))

        listing = ParsedListing(
            listing_id=f"{self.source}:{native_id}",
            source=self.source,
            url=url,
            year=year,
            make=make,
            model=model,
            trim=trim,
            body_style=body_style,
            mileage=mileage,
            vin=vin,
            raw_html_sha256=sha256_text(html),
            image_urls=image_urls,
        )
        return ParseResult(new_listing=listing)


# ---------- helpers ----------------------------------------------------------


def _find_next_page(soup: BeautifulSoup, *, base_url: str) -> str | None:
    """Locate a pagination "next" link via rel=next / aria-label / inner text."""
    # rel="next" is the strongest signal.
    for anchor in soup.find_all("a", href=True):
        rel = anchor.get("rel")
        if isinstance(rel, list) and "next" in {r.lower() for r in rel}:
            href = anchor.get("href")
            if isinstance(href, str) and href.strip():
                return normalize_url(base_url, href.strip())

    # aria-label="Next".
    for anchor in soup.find_all("a", attrs={"aria-label": True}, href=True):
        aria = anchor.get("aria-label")
        if isinstance(aria, str) and aria.strip().lower() == "next":
            href = anchor.get("href")
            if isinstance(href, str) and href.strip():
                return normalize_url(base_url, href.strip())

    # Text content "Next".
    for anchor in soup.find_all("a", href=True):
        text = anchor.get_text(strip=True)
        if text.lower() == "next":
            href = anchor.get("href")
            if isinstance(href, str) and href.strip():
                return normalize_url(base_url, href.strip())

    return None


def _extract_native_id(url: str) -> str | None:
    match = _LISTING_ID_FROM_URL_RE.search(url)
    if match is None:
        return None
    return match.group(1)


def _name_or_string(value: Any) -> str | None:
    """Accept either a string or a JSON-LD ``{'name': ...}`` dict."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str):
            return name.strip() or None
    return None


def _parse_mileage(value: Any) -> int | None:
    """JSON-LD ``mileageFromOdometer`` is usually ``{'value': 12345}`` but may be a string."""
    if value is None:
        return None
    if isinstance(value, dict):
        inner = value.get("value")
        if isinstance(inner, int):
            return inner
        if isinstance(inner, str):
            return parse_int_safe(inner)
        if isinstance(inner, float):
            return int(inner)
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return parse_int_safe(value)
    return None


def _extract_image_urls(value: Any) -> list[str]:
    """``image`` may be a string, a list of strings, or absent. Keep only http(s)."""
    if value is None:
        return []
    candidates: list[Any] = list(value) if isinstance(value, list) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if isinstance(item, str):
            url = item.strip()
        elif isinstance(item, dict):
            inner = item.get("url") or item.get("contentUrl")
            url = inner.strip() if isinstance(inner, str) else ""
        else:
            url = ""
        if not url:
            continue
        if not (url.startswith("http://") or url.startswith("https://")):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


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
