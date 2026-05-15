"""Hemmings parser — search pages and individual listing pages.

Hemmings ships two coexisting listing URL shapes:

* classifieds: ``https://www.hemmings.com/classifieds/dealer/<make>/<model>/<id>``
* auctions:    ``https://www.hemmings.com/auctions/<slug>/<id>``

In both shapes the trailing path segment is a 6+ digit numeric native id we
expose as ``listing_id = "hemmings:<id>"``. Listing pages embed JSON-LD —
classifieds usually as ``Vehicle``, auctions as ``Product`` (sometimes
``Car``). We try Vehicle, then Car, then Product.

Both flows are defensive: any unexpected shape is logged via
:class:`ParseResult.notes` rather than raising.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from .base import DiscoveredUrl, ParsedListing, ParseResult
from .common import (
    extract_jsonld,
    find_jsonld_by_type,
    find_next_page,
    normalize_url,
    parse_int_safe,
    parse_year_safe,
    sha256_text,
)

logger = logging.getLogger(__name__)


# Listing href shape. Matches both classifieds and auctions URL shapes,
# relative or absolute, with the trailing numeric (``\d{6,}``) capturing
# the native listing id.
_LISTING_HREF_RE = re.compile(
    r"^(?:https?://(?:www\.)?hemmings\.com)?"
    r"/(?:classifieds|auctions)/[^?#]*?\d{6,}/?$"
)

# Trailing numeric id (>= 6 digits, to dodge zip codes / route fragments).
_LISTING_ID_FROM_PATH_RE = re.compile(r"(\d{6,})/?$")


class HemmingsParser:
    """Per-site parser for Hemmings search and listing pages."""

    source: str = "hemmings"

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
                notes=["image kind is a no-op for hemmings (downloads handled by image pipeline)"]
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

        next_url = find_next_page(soup, base_url=url)
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

        # Hints serve as a fallback for fields the JSON-LD doesn't carry
        # (Product-shaped auction blocks frequently lack a model field).
        hint_year = _as_int(hints.get("target_year"))
        hint_make = _as_str(hints.get("target_make"))
        hint_model = _as_str(hints.get("target_model"))

        jsonld_year = parse_year_safe(
            _as_str(
                vehicle.get("vehicleModelDate")
                or vehicle.get("modelDate")
                or vehicle.get("productionDate")
            )
        )
        if jsonld_year is None:
            jsonld_year = parse_year_safe(_as_str(vehicle.get("name")))

        jsonld_make = _name_or_string(vehicle.get("manufacturer") or vehicle.get("brand"))
        jsonld_model = _name_or_string(vehicle.get("model"))

        year = jsonld_year if jsonld_year is not None else hint_year
        make = jsonld_make if jsonld_make is not None else hint_make
        model = jsonld_model if jsonld_model is not None else hint_model

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


def _extract_native_id(url: str) -> str | None:
    """Pull the trailing numeric id from a Hemmings listing URL path."""
    path = urlparse(url).path
    match = _LISTING_ID_FROM_PATH_RE.search(path)
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
    """``image`` may be a string, a list of strings, or ImageObject dicts."""
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
