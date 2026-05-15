"""Bring a Trailer parser — search pages and individual auction listing pages.

BaT is an auction-style marketplace, structurally similar to cars.com but with
slug-keyed (not numeric-id-keyed) listing URLs. Listing pages typically embed
a JSON-LD ``Product`` block (auction, not fixed-sale) and sometimes ``Vehicle``
or ``Car``; we try all three in that order.

Listing URLs look like ``https://bringatrailer.com/listing/<slug>/`` — the
slug is our native id. We surface it as ``listing_id = "bat:<slug>"``. When
JSON-LD lacks year/make/model, we first try to parse them out of the
``name`` string (BaT consistently encodes ``"<year> <make> <model> ..."``
there, even though ``brand``/``manufacturer``/``model`` are usually null),
and only fall back to the worker's queue ``hints`` (the original seed
target) when name-parsing also fails. Without the name-parsing step,
every BaT listing collapses to whatever (make, model) the seed routed
the URL for, regardless of the actual car.

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


# Listing href shape — accepts both relative (``/listing/<slug>/``) and
# absolute (``https://bringatrailer.com/listing/<slug>/``) forms. Trailing
# slash is optional.
_LISTING_HREF_RE = re.compile(r"^(?:https?://(?:www\.)?bringatrailer\.com)?/listing/[a-z0-9-]+/?$")

# Extract the slug from a listing URL path.
_LISTING_SLUG_FROM_PATH_RE = re.compile(r"^/listing/([a-z0-9-]+)/?$")

# Same tokenizer shape as the Craigslist title heuristic — split on
# whitespace, commas, and slashes. Hyphens are kept inside tokens so
# names like ``Mercedes-Benz`` and ``F-150`` survive intact.
_NAME_SPLIT_RE = re.compile(r"[\s,/]+")

# Year pattern for the prefix-skip loop. BaT names often start with
# editorial-style preamble (``"No Reserve: Original-Owner 1986 Honda
# Civic Si"``) so we scan forward token-by-token until we land on a
# plausible 4-digit year.
_YEAR_TOKEN_RE = re.compile(r"^\d{4}$")

# Tokens that mark the end of the model run after the make. ``-`` is a
# free-standing separator BaT uses before trim/price qualifiers (e.g.
# ``"... Civic Si - 5-Speed Manual"``); hyphens *inside* a token survive
# the tokenizer above so this only fires on a bare dash.
_BAT_MODEL_BOUNDARY_TOKENS: frozenset[str] = frozenset({"-", "$", "(", ")", "|", "·", ":"})

# Common trim / transmission / body-style tokens that should not appear
# inside the ``model`` field. With ``_BAT_MAX_MODEL_TOKENS=1`` this set
# is documentation rather than active filtering — the single-token cap
# stops the walk before any of these can creep into the model — but it
# records the rule explicitly for future maintainers and tests.
_BAT_TRIM_STOPWORDS: frozenset[str] = frozenset(
    {
        # Honda / Toyota / domestic trim badges.
        "si",
        "ex",
        "lx",
        "dx",
        "cx",
        "ls",
        "lt",
        "ltz",
        "se",
        "sel",
        "sl",
        "slt",
        # Transmission descriptors.
        "5-speed",
        "6-speed",
        "4-speed",
        "manual",
        "automatic",
        "awd",
        "4wd",
        "fwd",
        "rwd",
        # Body styles.
        "hatchback",
        "sedan",
        "coupe",
        "convertible",
        "wagon",
        "suv",
        "truck",
        # Sport / performance suffixes.
        "touring",
        "sport",
        "type",
        "r",
        "s",
        "x",
        "gt",
        "rs",
        "trd",
        "limited",
    }
)

# Maximum number of tokens we accept as the model name. We cap at 1:
# most BaT model names are single words (``Civic``, ``CRX``, ``Defender``,
# ``911``, ``F-150``). Anything after the first token (trim, transmission,
# body-style) is routed into ``trim`` instead. Multi-word makes are still
# handled — ``Land Rover Defender`` works because the make consumes two
# tokens before the model walk starts, so ``Defender`` is still the first
# model token.
_BAT_MAX_MODEL_TOKENS: int = 1


class BringATrailerParser:
    """Per-site parser for Bring a Trailer search and listing pages."""

    source: str = "bat"

    def __init__(self, *, known_makes: frozenset[str] | None = None) -> None:
        """Build a parser.

        Parameters
        ----------
        known_makes:
            Pool of make names (lowercased) used by the ``name``-field
            heuristic when JSON-LD ``brand``/``manufacturer`` are absent.
            Defaults to a snapshot of
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
                notes=["image kind is a no-op for bat (downloads handled by image pipeline)"]
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

        slug = _extract_slug(url)
        if slug is None:
            return ParseResult(notes=[f"could not extract listing_id from URL: {url}"])

        # Layered fallbacks for year/make/model. BaT auction ``Product``
        # blocks usually have only ``name`` populated; ``brand``,
        # ``manufacturer``, and ``model`` are typically ``null``.
        #   1. Explicit JSON-LD fields (``manufacturer``/``brand``/``model``).
        #   2. Heuristic parse of the JSON-LD ``name`` string.
        #   3. Queue hints (the original seed target) as a last resort.
        hint_year = _as_int(hints.get("target_year"))
        hint_make = _as_str(hints.get("target_make"))
        hint_model = _as_str(hints.get("target_model"))

        name_str = _as_str(vehicle.get("name"))

        jsonld_year = parse_year_safe(
            _as_str(
                vehicle.get("vehicleModelDate")
                or vehicle.get("modelDate")
                or vehicle.get("productionDate")
            )
        )
        # Fallback: try to read a year out of the ``name`` field, which BaT
        # almost always sets to something like "1989 Porsche 911 Carrera".
        if jsonld_year is None:
            jsonld_year = parse_year_safe(name_str)

        jsonld_make = _name_or_string(vehicle.get("manufacturer") or vehicle.get("brand"))
        jsonld_model = _name_or_string(vehicle.get("model"))

        # Step 2: derive make/model/trim from ``name`` only for fields that
        # the explicit JSON-LD didn't already supply. This means explicit
        # ``brand``/``manufacturer`` wins (the existing contract); we only
        # name-parse when those are null.
        name_make: str | None = None
        name_model: str | None = None
        name_trim: str | None = None
        if jsonld_make is None or jsonld_model is None:
            name_make, name_model, name_trim = self._parse_name_make_model(name_str)

        year = jsonld_year if jsonld_year is not None else hint_year
        make = jsonld_make if jsonld_make is not None else (name_make or hint_make)
        model = jsonld_model if jsonld_model is not None else (name_model or hint_model)

        # ``trim`` only falls back to the name-parsed tail when JSON-LD
        # doesn't supply its own ``vehicleConfiguration`` / ``trim``.
        trim = _as_str(vehicle.get("vehicleConfiguration") or vehicle.get("trim"))
        if trim is None:
            trim = name_trim
        mileage = _parse_mileage(vehicle.get("mileageFromOdometer"))
        vin = _as_str(vehicle.get("vehicleIdentificationNumber"))
        body_style = _as_str(vehicle.get("bodyType"))
        image_urls = _extract_image_urls(vehicle.get("image"))

        listing = ParsedListing(
            listing_id=f"{self.source}:{slug}",
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

    # ------------------------------------------------------------- name parser

    def _parse_name_make_model(self, name: str | None) -> tuple[str | None, str | None, str | None]:
        """Extract ``(make, model, trim)`` from a BaT ``name`` field.

        BaT names follow ``"<year> <make> <model> [<trim>] ..."`` but
        often carry editorial prefixes like ``"No Reserve:"`` or
        ``"Original-Owner"`` before the year. Strategy:

          1. Tokenise on whitespace/commas/slashes (hyphens stay inside
             tokens so ``Mercedes-Benz`` survives).
          2. Skip leading tokens until a 4-digit year is found, then
             advance past it. (Year extraction itself is handled by
             :func:`parse_year_safe`; we just need the cursor position.)
          3. Match a two-word make first (``Land Rover``, ``Alfa Romeo``),
             then a single-word make against the known-makes pool.
          4. Take the FIRST token after the make as the model (cap of
             :data:`_BAT_MAX_MODEL_TOKENS` = 1). Most BaT model names are
             single-word: ``Civic``, ``CRX``, ``Defender``, ``911``,
             ``F-150``. Multi-word makes already consumed two tokens, so
             ``Land Rover Defender`` correctly yields model=``Defender``.
          5. Collect any remaining tokens (up to the first boundary
             separator such as ``-``, ``$``, ``(``, ``|``) into ``trim``.
             This is what catches sub-trim and transmission descriptors
             like ``Si``, ``Cx Hatchback 5-Speed``, ``Type R``.

        Returns
        -------
        ``(make, model, trim)`` with each field set to ``None`` when the
        respective extraction failed. Make is returned in the canonical
        case used by ``MAKE_POPULARITY`` (e.g. ``"Honda"``, ``"Bmw"``);
        model and trim are title-cased per-token, preserving digits and
        hyphens.
        """
        if not name:
            return (None, None, None)

        tokens = [t for t in _NAME_SPLIT_RE.split(name.strip()) if t]
        if not tokens:
            return (None, None, None)
        lower_tokens = [t.lower() for t in tokens]

        # Step 2: find the year token's index so we can resume scanning
        # for the make immediately after it. If there's no year token at
        # all, fall back to scanning from the start (some BaT names omit
        # the year, e.g. event lots).
        start_idx = 0
        for i, tok in enumerate(lower_tokens):
            if _YEAR_TOKEN_RE.match(tok):
                start_idx = i + 1
                break

        # Step 3: locate the make. Two-word match first, then single-word.
        make_canonical: str | None = None
        make_end_idx = -1
        for i in range(start_idx, len(lower_tokens)):
            if i + 1 < len(lower_tokens):
                two_word = f"{lower_tokens[i]} {lower_tokens[i + 1]}"
                if two_word in self._known_makes:
                    make_canonical = _canonical_make_case(two_word)
                    make_end_idx = i + 2
                    break
            if lower_tokens[i] in self._known_makes:
                make_canonical = _canonical_make_case(lower_tokens[i])
                make_end_idx = i + 1
                break

        if make_canonical is None or make_end_idx < 0:
            return (None, None, None)

        # Step 4: collect model tokens (capped at _BAT_MAX_MODEL_TOKENS=1),
        # stopping at boundary chars. Step 5 below sweeps the rest into
        # ``trim``.
        model_tokens: list[str] = []
        trim_start_idx = make_end_idx
        for j in range(make_end_idx, len(lower_tokens)):
            tok = tokens[j]
            if tok in _BAT_MODEL_BOUNDARY_TOKENS:
                trim_start_idx = j
                break
            if tok.startswith("$") or tok.startswith("(") or tok.startswith("|"):
                trim_start_idx = j
                break
            model_tokens.append(tok)
            trim_start_idx = j + 1
            if len(model_tokens) >= _BAT_MAX_MODEL_TOKENS:
                break

        if not model_tokens:
            return (make_canonical, None, None)

        model_str = " ".join(_titlecase_model_token(t) for t in model_tokens)

        # Step 5: tail tokens become the trim. Stop on the same boundary
        # separators we use for the model walk.
        trim_tokens: list[str] = []
        for k in range(trim_start_idx, len(lower_tokens)):
            tok = tokens[k]
            if tok in _BAT_MODEL_BOUNDARY_TOKENS:
                break
            if tok.startswith("$") or tok.startswith("(") or tok.startswith("|"):
                break
            trim_tokens.append(tok)

        trim_str = " ".join(_titlecase_model_token(t) for t in trim_tokens) if trim_tokens else None
        return (make_canonical, model_str, trim_str)


# ---------- helpers ----------------------------------------------------------


def _extract_slug(url: str) -> str | None:
    """Pull the slug from a BaT listing URL path: ``/listing/<slug>/``."""
    path = urlparse(url).path
    match = _LISTING_SLUG_FROM_PATH_RE.match(path)
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


def _canonical_make_case(lower_make: str) -> str:
    """Return ``lower_make`` in the canonical case used by ``MAKE_POPULARITY``.

    The catalog (NHTSA-derived) Title-cases everything — even acronyms —
    so ``"bmw"`` becomes ``"Bmw"`` and ``"mclaren"`` becomes ``"Mclaren"``.
    Two-word makes (``"land rover"``) Title-case each word independently
    (``"Land Rover"``); hyphenated makes (``"mercedes-benz"``) likewise
    capitalise each segment (``"Mercedes-Benz"``). This is intentional
    for v1 to match what the seeder/ranker emits; downstream display
    layers can pretty it up if they want.
    """
    return " ".join(
        "-".join(seg.capitalize() for seg in word.split("-")) for word in lower_make.split(" ")
    )


def _titlecase_model_token(token: str) -> str:
    """Title-case alpha runs in a token while preserving digits and hyphens.

    Examples:
        ``"crx"``   → ``"Crx"``
        ``"f-150"`` → ``"F-150"``
        ``"si"``    → ``"Si"``
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
