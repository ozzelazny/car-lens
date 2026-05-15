"""cars.com parser — search pages and individual listing pages.

Listing pages in 2026 are server-rendered with an inline JSON state blob
in ``<script type="application/json" id="CarsWeb.VehicleDetailController.show">``
that holds every canonical field (year, make, model, trim, vin, mileage,
body_style) on a real production page. Hand-crafted test fixtures embed
the same fields in a JSON-LD ``Vehicle`` block; both shapes are supported.

Extraction order on a listing page:

1. JSON-LD ``Vehicle`` / ``Car`` / ``Product`` block (legacy / fixture path).
2. Inline ``CarsWeb.VehicleDetailController.show`` JSON blob —
   ``call_source_dni_metadata.dimensions`` and ``dealer_chat_metadata.carsData``
   (production path).
3. ``<title>`` / ``<h1 id="vehicle-title">`` fallback for year/make/model
   if neither of the above produced them.

Images are pulled from the JSON-LD ``image`` array first, falling back to
``<img slot="image">`` tags pointing at ``platform.cstatic-images.com``
(the gallery wrapper cars.com uses for vehicle photos).

Search pages we scrape via CSS selectors plus a regex on listing-card
hrefs of the form ``/vehicledetail/<native_id>/``.

Both flows are defensive: any unexpected shape is logged via
:class:`ParseResult.notes` rather than raising, since the worker treats
exceptions as hard failures.
"""

from __future__ import annotations

import json
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


# Listing-card href shape: ``/vehicledetail/{native_id}/`` (trailing slash
# optional). Real cars.com pages also append a query string —
# ``?attribution_type=p_one``, ``?openLeadForm=true``, etc. — on the same
# anchor; we accept (and later strip) those. cars.com also exposes
# sub-routes on the same listing — ``/overview/``, ``/photos/``,
# ``/features/`` — that we accept as equivalent entry points. The host
# prefix is optional so both relative (``/vehicledetail/...``) and absolute
# (``https://www.cars.com/vehicledetail/...``) anchors match.
_LISTING_HREF_RE = re.compile(
    r"^(?:https?://(?:www\.)?cars\.com)?"
    r"/vehicledetail/[^/?#]+"
    r"(?:/(?:overview|photos|features))?"
    r"/?(?:\?[^#]*)?$"
)

# Native ID extractor from a full listing URL path.
_LISTING_ID_FROM_URL_RE = re.compile(r"/vehicledetail/([^/?#]+)")


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

        # Pull listing hrefs from every tag shape cars.com currently uses:
        #
        # * plain ``<a href="/vehicledetail/...">`` — the canonical anchor.
        # * ``<fuse-button href="...">`` — the "View Listing" CTA on each
        #   card; one per card, duplicated alongside the plain anchor.
        # * ``<card-gallery card-href="...">`` — the photo-carousel wrapper
        #   on each card; the link is on a non-standard attribute.
        # * ``<spark-link-button href="...">`` — legacy/transitional shape
        #   retained for safety.
        #
        # Real cars.com pages emit several hrefs per listing (varying query
        # strings such as ``?attribution_type=p_one``) so we dedupe by the
        # extracted ``native_id`` and keep the first canonical form we see
        # for that id.
        listing_urls: list[str] = []
        seen_ids: set[str] = set()
        tag_attr_candidates: list[tuple[list[str], str]] = [
            (["a", "fuse-button", "spark-link-button"], "href"),
            (["card-gallery"], "card-href"),
        ]
        for tag_names, attr in tag_attr_candidates:
            for tag in soup.find_all(tag_names):
                href = tag.get(attr)
                if not isinstance(href, str):
                    continue
                stripped = href.strip()
                if not stripped or not _LISTING_HREF_RE.match(stripped):
                    continue
                native_id = _extract_native_id(stripped)
                if native_id is None or native_id in seen_ids:
                    continue
                seen_ids.add(native_id)
                # Drop the query string entirely — cars.com listing pages
                # do not need ``?attribution_type=...`` etc., and stripping
                # gives us a stable canonical URL for downstream dedup.
                canonical = f"/vehicledetail/{native_id}/"
                listing_urls.append(normalize_url(url, canonical))

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

        native_id = _extract_native_id(url)
        if native_id is None:
            return ParseResult(notes=[f"could not extract listing_id from URL: {url}"])

        notes: list[str] = []
        soup = BeautifulSoup(html, features="lxml")

        # --- primary path: JSON-LD Vehicle/Car/Product ----------------------
        blocks = extract_jsonld(html)
        vehicle = (
            find_jsonld_by_type(blocks, "Vehicle")
            or find_jsonld_by_type(blocks, "Car")
            or find_jsonld_by_type(blocks, "Product")
        )

        year: int | None = None
        make: str | None = None
        model: str | None = None
        trim: str | None = None
        mileage: int | None = None
        vin: str | None = None
        body_style: str | None = None
        image_urls: list[str] = []

        if vehicle is not None:
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

        # --- secondary path: CarsWeb.VehicleDetailController.show JSON blob --
        # Real cars.com production pages embed structured vehicle state in
        # an inline ``<script type="application/json" id="CarsWeb...">`` blob
        # rather than schema.org JSON-LD. We probe two sub-trees that both
        # carry the canonical fields and merge anything the JSON-LD path
        # missed.
        cars_web = _extract_cars_web_state(soup)
        if cars_web is not None:
            cw_year, cw_make, cw_model, cw_trim, cw_mileage, cw_vin, cw_body = (
                _fields_from_cars_web_state(cars_web)
            )
            year = year if year is not None else cw_year
            make = make if make is not None else cw_make
            model = model if model is not None else cw_model
            trim = trim if trim is not None else cw_trim
            mileage = mileage if mileage is not None else cw_mileage
            vin = vin if vin is not None else cw_vin
            body_style = body_style if body_style is not None else cw_body

        # --- last-resort heuristic: <title> / <h1 id="vehicle-title"> --------
        # The title looks like "Used 2020 Honda Civic LX For Sale ... | Cars.com"
        # and the H1 like "Used 2020 Honda Civic LX". Both reliably contain
        # year/make/model when other sources have failed.
        if year is None or make is None or model is None:
            t_year, t_make, t_model, t_trim = _heuristic_from_title(soup)
            year = year if year is not None else t_year
            make = make if make is not None else t_make
            model = model if model is not None else t_model
            trim = trim if trim is not None else t_trim

        # --- image fallback: <img slot="image" src=...> ----------------------
        # Real cars.com listings render the photo gallery as ``<img slot="image">``
        # pointing at ``platform.cstatic-images.com``. The JSON-LD ``image``
        # array is absent on production HTML, so we fall back here.
        if not image_urls:
            image_urls = _extract_gallery_image_urls(soup)

        if vehicle is None and cars_web is None and (year is None or make is None):
            return ParseResult(notes=["no Vehicle JSON-LD found and no CarsWeb state on page"])
        if vehicle is None:
            notes.append("JSON-LD Vehicle/Car/Product absent; used CarsWeb state fallback")

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
        return ParseResult(new_listing=listing, notes=notes)


# ---------- helpers ----------------------------------------------------------


def _extract_cars_web_state(soup: BeautifulSoup) -> dict[str, Any] | None:
    """Return parsed JSON from ``<script id="CarsWeb.VehicleDetailController.show">``.

    Real cars.com listing pages embed the vehicle's structured state as an
    inline JSON blob in this script tag. Returns ``None`` if the tag is
    absent or fails to parse — callers fall back to other paths.
    """
    tag = soup.find("script", attrs={"id": "CarsWeb.VehicleDetailController.show"})
    if tag is None:
        return None
    raw = tag.string or tag.get_text() or ""
    raw = raw.strip()
    if not raw:
        return None
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.debug("failed to parse CarsWeb.VehicleDetailController.show JSON: %s", exc)
        return None
    if not isinstance(data, dict):
        return None
    return data


def _fields_from_cars_web_state(
    state: dict[str, Any],
) -> tuple[int | None, str | None, str | None, str | None, int | None, str | None, str | None]:
    """Pull year/make/model/trim/mileage/vin/body_style out of the inline state.

    Two sub-trees are probed in order: ``call_source_dni_metadata.dimensions``
    (the richer one — list-of-strings for most fields) and
    ``dealer_chat_metadata.carsData`` (the flatter one — scalars). We merge
    them by always preferring the first non-empty value.
    """
    dimensions = _as_dict(_get_path(state, ["call_source_dni_metadata", "dimensions"]))
    cars_data = _as_dict(_get_path(state, ["dealer_chat_metadata", "carsData"]))

    year = _first_int(
        [
            _scalar_or_first(dimensions.get("year")),
            cars_data.get("year"),
        ]
    )
    make = _first_nonempty_str(
        [
            _scalar_or_first(dimensions.get("make")),
            cars_data.get("make"),
        ]
    )
    model = _first_nonempty_str(
        [
            _scalar_or_first(dimensions.get("model")),
            cars_data.get("model"),
        ]
    )
    # ``dimensions.trim`` is properly cased ("LX"); ``carsData.trim`` is
    # often lowercase ("lx"). Prefer dimensions when present.
    trim = _first_nonempty_str(
        [
            _scalar_or_first(dimensions.get("trim")),
            cars_data.get("trim"),
        ]
    )
    mileage = _first_int(
        [
            _scalar_or_first(dimensions.get("mileage")),
            cars_data.get("mileage"),
        ]
    )
    vin = _first_nonempty_str(
        [
            _scalar_or_first(dimensions.get("vin")),
            cars_data.get("vin"),
        ]
    )
    body_style = _first_nonempty_str(
        [
            _scalar_or_first(dimensions.get("bodyStyle")),
            cars_data.get("bodystyle"),
        ]
    )
    # ``bodystyle`` from carsData is lowercased ("sedan"); title-case it for
    # consistency with the JSON-LD path (which yields "Sedan").
    if body_style is not None:
        body_style = body_style.title()
    return year, make, model, trim, mileage, vin, body_style


def _heuristic_from_title(
    soup: BeautifulSoup,
) -> tuple[int | None, str | None, str | None, str | None]:
    """Year / make / model / trim from ``<title>`` or ``<h1 id="vehicle-title">``.

    Used only when JSON-LD and the CarsWeb state both fail to produce a
    field. Best-effort: extracts the first 4-digit year and treats the
    next token as the make. We deliberately do NOT load a static make list
    here — keeping it heuristic avoids drift between the parser and the
    catalog.
    """
    text = ""
    h1 = soup.find("h1", attrs={"id": "vehicle-title"})
    if h1 is not None:
        text = h1.get_text(strip=True)
    if not text:
        title = soup.find("title")
        if title is not None:
            text = title.get_text(strip=True)
    if not text:
        return None, None, None, None

    year = parse_year_safe(text)
    if year is None:
        return None, None, None, None
    # Tokenise the text after the year. Strip the trailing " | Cars.com"
    # suffix and any leading "Used " / "New " / "Certified " marker.
    stripped = re.sub(r"\s*\|\s*Cars\.com\s*$", "", text).strip()
    tokens = stripped.split()
    try:
        year_idx = tokens.index(str(year))
    except ValueError:
        return year, None, None, None
    after = tokens[year_idx + 1 :]
    # The next token is the make, then the model. Trim is whatever follows
    # before "For Sale" / "$..." / etc.
    make = after[0] if len(after) >= 1 else None
    model = after[1] if len(after) >= 2 else None
    trim_tokens: list[str] = []
    for token in after[2:]:
        if token.lower() in {"for", "sale"} or token.startswith("$"):
            break
        trim_tokens.append(token)
    trim = " ".join(trim_tokens).strip() or None
    return year, make, model, trim


def _extract_gallery_image_urls(soup: BeautifulSoup) -> list[str]:
    """Collect ``<img slot="image" src=...>`` URLs from the photo gallery.

    cars.com renders the vehicle photo gallery with custom elements that
    hold the actual ``<img>`` tags via the ``slot="image"`` attribute.
    Sources point at ``platform.cstatic-images.com`` for real photos
    (versus stock article thumbnails on ``images.cars.com``). We keep
    only ``platform.cstatic-images.com`` images to avoid pulling stock art.
    """
    out: list[str] = []
    seen: set[str] = set()
    for tag in soup.find_all("img", attrs={"slot": "image"}):
        src = tag.get("src")
        if not isinstance(src, str):
            continue
        url = src.strip()
        if not url or not url.startswith("https://"):
            continue
        if "platform.cstatic-images.com" not in url:
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


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


def _as_dict(value: Any) -> dict[str, Any]:
    """Return ``value`` if it's a dict, else an empty dict."""
    return value if isinstance(value, dict) else {}


def _get_path(node: Any, path: list[str]) -> Any:
    """Walk ``path`` through nested dicts. Return ``None`` on any miss."""
    cur: Any = node
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _scalar_or_first(value: Any) -> Any:
    """If ``value`` is a list, return its first element; else return as-is.

    cars.com's ``dimensions`` sub-tree wraps most fields in a single-element
    list (``"make": ["Honda"]``). Unwrap it before further coercion.
    """
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _first_nonempty_str(values: list[Any]) -> str | None:
    """Return the first value in ``values`` that's a non-empty string."""
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _first_int(values: list[Any]) -> int | None:
    """Coerce values to int in order; return the first non-None coercion."""
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            parsed = parse_int_safe(value)
            if parsed is not None:
                return parsed
    return None
