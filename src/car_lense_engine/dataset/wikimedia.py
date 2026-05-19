"""Wikimedia Commons vintage-car ingest (Phase 4.4).

Pulls pre-2000 vehicle images from Wikimedia Commons via the MediaWiki API,
extracts ``(year, make, model)`` labels from each file's parent categories,
and writes one synthetic listing + one image row per file into the crawler
DB.

Design choices
--------------

* **API-only.** No HTML scraping. The MediaWiki ``action=query`` API exposes
  category traversal (``list=categorymembers``) and per-file metadata
  (``prop=imageinfo|categories``), which is everything we need.
* **Polite crawl.** A descriptive User-Agent (Wikimedia requires identifiable
  UAs) and a configurable minimum delay between API calls (default 1.0 s)
  applied via :func:`_RateLimiter`. On HTTP 429 / 5xx we back off
  exponentially and retry up to ``WIKIMEDIA_MAX_RETRIES`` times.
* **Category seeds → leaf category traversal.** Callers pass an iterable of
  seed categories; the ingest walks ``list=categorymembers&cmtype=file`` on
  each (with ``cmcontinue`` pagination) and yields files individually. We
  do NOT recurse into subcategories automatically — the caller controls
  which categories to pull from, so the breadth of the crawl is predictable.
* **Label heuristic — categories only.** Each file's parent categories are
  scanned for year / make / model tokens via
  :func:`extract_label_triple`. Files without a confident
  ``(year, make, model)`` triple are dropped (counted as ``skipped_no_label``).
* **Content-addressed storage.** ``image_id = sha256(bytes)``; the on-disk
  filename and the DB ``image_id`` are both the hash. Files live under
  ``data/public/wikimedia/<sha256>.<ext>`` (a flat layout — no class
  subdirs, because Wikimedia's labels are noisy and one-per-class subdirs
  would proliferate).
* **Dry-run.** ``config.dry_run`` skips both the DB writes and the byte
  downloads; everything else (the API traversal + label extraction) still
  runs so the operator can preview what would be ingested.
* **No pHash dedup yet.** Phase 3.2 territory — see TODO.md.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import re
import sqlite3
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict

from car_lense_engine.db import images, listings
from car_lense_engine.db.models import Image, Listing

from .canonical_labels import normalize_make, normalize_model, year_to_generation

logger = logging.getLogger(__name__)

_SOURCE: str = "wikimedia_commons"

WIKIMEDIA_API_URL: str = "https://commons.wikimedia.org/w/api.php"
WIKIMEDIA_USER_AGENT: str = (
    "car-lense-engine/0.1 (https://github.com/ashzelazny/car_lense; contact: ashzelazny@gmail.com)"
)
WIKIMEDIA_MAX_RETRIES: int = 5
WIKIMEDIA_BACKOFF_BASE: float = 1.0

# Map content-types served by Wikimedia thumbnails / originals to file
# extensions. JPEG and PNG cover the vast majority; WebP is occasionally
# emitted by their thumbnailing layer. Anything else triggers a skip.
_CONTENT_TYPE_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

# Category name -> year. Two canonical phrasings appear on Commons:
#   - "Cars introduced in 1965" / "1965 automobiles" -- bare year-only
#   - "1965 Ford Mustang"                            -- year-prefixed model
# We accept both. The year token must be a 4-digit number in the valid
# historical range so we don't mis-parse e.g. "Category:Top 10 cars".
_YEAR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^Cars introduced in (\d{4})$", re.IGNORECASE),
    re.compile(r"^Vehicles introduced in (\d{4})$", re.IGNORECASE),
    re.compile(r"^Automobiles introduced in (\d{4})$", re.IGNORECASE),
    re.compile(r"^(\d{4}) automobiles$", re.IGNORECASE),
    re.compile(r"^(\d{4}) cars$", re.IGNORECASE),
    re.compile(r"^(\d{4}) vehicles$", re.IGNORECASE),
)

# Catch-all year prefix on a category like "1965 Ford Mustang" or
# "1972 Chevrolet Chevelle". Used as a *secondary* signal: the trailing
# tokens are scanned for a known make + model separately.
_YEAR_PREFIX_RE: re.Pattern[str] = re.compile(r"^(\d{4})\s+(\S.*)$")

# Decade-only categories (no specific year). Used as a *fallback* when no
# year-specific category was found. We treat the decade midpoint as the
# year (e.g. "1960s automobiles" -> 1965) but mark it as "decade-only" so
# the extractor can decide whether to keep the row.
_DECADE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(\d{4})s automobiles$", re.IGNORECASE),
    re.compile(r"^(\d{4})s cars$", re.IGNORECASE),
    re.compile(r"^(\d{4})s vehicles$", re.IGNORECASE),
)

# Disambiguation / qualifier suffixes that we strip from a make category
# before alias lookup. "Ford (cars)" -> "Ford"; "BMW vehicles" -> "BMW".
_MAKE_SUFFIX_RE = re.compile(
    r"\s*(?:\(.*?\)|vehicles|automobiles|cars|motor company|motors)$",
    re.IGNORECASE,
)

# Additional makes commonly seen on Wikimedia categories that don't need
# alias-map entries because their Title Case form is already canonical.
# Used as a recognition allow-list in :func:`_find_make` — we only accept a
# category as "a make" if its lowercase form matches either the alias map
# OR this set, otherwise generic words like "Cars" or "Sedans" would pass.
_KNOWN_MAKES_EXTRA: frozenset[str] = frozenset(
    {
        "acura",
        "audi",
        "bentley",
        "bugatti",
        "cadillac",
        "chrysler",
        "citroen",
        "citroën",
        "daewoo",
        "daihatsu",
        "datsun",
        "dodge",
        "ferrari",
        "fisker",
        "ford",
        "genesis",
        "honda",
        "hummer",
        "hyundai",
        "infiniti",
        "isuzu",
        "jaguar",
        "jeep",
        "lancia",
        "lexus",
        "lincoln",
        "lotus",
        "maserati",
        "maybach",
        "mclaren",
        "mercury",
        "mitsubishi",
        "morgan",
        "nissan",
        "oldsmobile",
        "opel",
        "packard",
        "peugeot",
        "plymouth",
        "pontiac",
        "porsche",
        "ram",
        "renault",
        "saturn",
        "scion",
        "seat",
        "skoda",
        "studebaker",
        "subaru",
        "suzuki",
        "toyota",
        "triumph",
        "volkswagen",
        "volvo",
    }
)

_VALID_YEAR_MIN: int = 1800
_VALID_YEAR_MAX: int = 2099


class WikimediaIngestConfig(BaseModel):
    """Run-time configuration for :func:`ingest_wikimedia`."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    api_url: str = WIKIMEDIA_API_URL
    user_agent: str = WIKIMEDIA_USER_AGENT
    output_dir: pathlib.Path
    seed_categories: tuple[str, ...]
    year_min: int = 1900
    year_max: int = 1999
    max_images_per_category: int = 500
    rate_limit_seconds: float = 1.0
    split: str = "train"
    limit: int | None = None
    dry_run: bool = False
    request_timeout_seconds: float = 30.0

    def validated(self) -> WikimediaIngestConfig:
        """Defensive validation past Pydantic's basic type-checks."""
        if self.year_min > self.year_max:
            raise ValueError(f"year_min ({self.year_min}) must be <= year_max ({self.year_max})")
        if self.year_min < _VALID_YEAR_MIN or self.year_max > _VALID_YEAR_MAX:
            raise ValueError(
                f"year range must be within [{_VALID_YEAR_MIN}, {_VALID_YEAR_MAX}]; "
                f"got [{self.year_min}, {self.year_max}]"
            )
        if self.max_images_per_category <= 0:
            raise ValueError(
                f"max_images_per_category must be > 0, got {self.max_images_per_category}"
            )
        if self.rate_limit_seconds < 0:
            raise ValueError(f"rate_limit_seconds must be >= 0, got {self.rate_limit_seconds}")
        if self.limit is not None and self.limit <= 0:
            raise ValueError(f"limit must be > 0 or None, got {self.limit}")
        if self.request_timeout_seconds <= 0:
            raise ValueError(
                f"request_timeout_seconds must be > 0, got {self.request_timeout_seconds}"
            )
        if not self.seed_categories:
            raise ValueError("at least one seed_category must be provided")
        return self


class WikimediaIngestSummary(BaseModel):
    """Per-run ingest counters."""

    model_config = ConfigDict(extra="forbid")

    processed: int = 0
    listings_inserted: int = 0
    images_inserted: int = 0
    skipped_existing: int = 0
    skipped_no_label: int = 0
    skipped_out_of_year_range: int = 0
    skipped_unsupported_type: int = 0
    skipped_download_failures: int = 0
    api_errors: int = 0


@dataclass
class _RateLimiter:
    """Minimum-delay-between-calls helper.

    ``min_delay`` of 0 disables rate limiting entirely (handy for tests).
    Uses an injectable clock + sleep so tests can substitute deterministic
    versions.
    """

    min_delay: float
    clock: Callable[[], float] = field(default_factory=lambda: time.monotonic)
    sleep: Callable[[float], None] = field(default_factory=lambda: time.sleep)
    _last_call_at: float | None = field(default=None, init=False)

    def wait(self) -> None:
        """Sleep just long enough that ``min_delay`` has elapsed since the prior call."""
        if self.min_delay <= 0:
            return
        now = self.clock()
        if self._last_call_at is not None:
            elapsed = now - self._last_call_at
            remaining = self.min_delay - elapsed
            if remaining > 0:
                self.sleep(remaining)
        self._last_call_at = self.clock()


def extract_label_triple(
    file_categories: Iterable[str],
    *,
    year_min: int = _VALID_YEAR_MIN,
    year_max: int = _VALID_YEAR_MAX,
) -> tuple[int, str, str] | None:
    """Extract ``(year, make, model)`` from a file's Wikipedia parent categories.

    Heuristic, in order:

    1. **Year.** Find the first category matching one of the explicit year
       patterns (``"Cars introduced in 1965"`` / ``"1965 automobiles"`` /
       etc.). If none match, fall back to the decade midpoint (``"1960s
       automobiles"`` -> 1965). If still nothing, drop.
    2. **Make.** Find a category whose stripped name (suffixes like
       ``"(cars)"``, ``"vehicles"`` removed) routes through the canonical
       make alias map. Take the FIRST canonical make found; cars rarely
       have multiple makes in their parent-category list.
    3. **Model.** Look for a category whose first token equals the canonical
       make and that has at least one trailing token. The trailing tokens
       become the model. Example: with ``make = "Ford"``, the category
       ``"Ford Mustang"`` yields ``model = "Mustang"``;
       ``"Ford Mustang first generation"`` yields ``model = "Mustang First
       Generation"`` (we don't try to strip "first generation" — Phase 4.5
       canonicalization does case-only normalization and that's enough).

    All three components must be present and the year must fall in
    ``[year_min, year_max]`` for the function to succeed. Returns ``None``
    otherwise.

    Examples
    --------

    >>> extract_label_triple(["Category:1965 Ford Mustang", "Category:Ford Mustang"])
    (1965, 'Ford', 'Mustang')

    >>> extract_label_triple(["Category:1960s automobiles"])  # decade-only, no make
    >>> # returns None

    >>> extract_label_triple(["Category:Cars introduced in 1972", "Category:Chevrolet Chevelle"])
    (1972, 'Chevrolet', 'Chevelle')
    """
    cats = [_strip_category_prefix(c) for c in file_categories]
    cats = [c for c in cats if c]

    year = _find_year(cats, year_min=year_min, year_max=year_max)
    if year is None:
        return None

    make = _find_make(cats)
    if make is None:
        return None

    model = _find_model(cats, make=make)
    if model is None:
        return None

    return year, make, model


def iter_category_files(
    api_url: str,
    category: str,
    *,
    session: Any,
    rate_limit: _RateLimiter,
    user_agent: str = WIKIMEDIA_USER_AGENT,
    cm_limit: int = 500,
    max_files: int | None = None,
    request_timeout: float = 30.0,
) -> Iterator[dict[str, Any]]:
    """Yield File-namespace members of ``category`` via the MediaWiki API.

    Uses ``action=query&list=categorymembers&cmtype=file&cmnamespace=6`` with
    ``cmcontinue`` pagination. The category title must include the
    ``Category:`` prefix; if missing, it's added automatically.

    Each yielded dict has the shape::

        {"pageid": int, "ns": 6, "title": "File:Foo.jpg"}

    Stops cleanly at ``max_files`` if set. Honors ``rate_limit`` between
    HTTP calls. Network errors are not caught — caller decides retry policy
    (see :func:`_fetch_with_retry`).
    """
    if not category.lower().startswith("category:"):
        category = f"Category:{category}"
    params: dict[str, Any] = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "file",
        "cmnamespace": 6,
        "cmlimit": min(int(cm_limit), 500),
        "format": "json",
        "formatversion": 2,
    }
    yielded = 0
    while True:
        rate_limit.wait()
        data = _fetch_with_retry(
            session,
            api_url,
            params=params,
            user_agent=user_agent,
            timeout=request_timeout,
        )
        members = (data.get("query") or {}).get("categorymembers") or []
        for member in members:
            yield member
            yielded += 1
            if max_files is not None and yielded >= max_files:
                return
        cont = data.get("continue") or {}
        cmcontinue = cont.get("cmcontinue")
        if not cmcontinue:
            return
        params = dict(params)
        params["cmcontinue"] = cmcontinue


def fetch_file_metadata(
    api_url: str,
    titles: list[str],
    *,
    session: Any,
    rate_limit: _RateLimiter,
    user_agent: str = WIKIMEDIA_USER_AGENT,
    request_timeout: float = 30.0,
) -> dict[str, dict[str, Any]]:
    """Look up parent categories + image URL for the given File-namespace titles.

    ``titles`` may include up to 50 entries per call (the MediaWiki API
    cap). For larger batches, the caller is responsible for chunking. The
    returned dict maps each input title to a metadata blob shaped::

        {
            "categories": [str, ...],   # parent-category titles
            "url": str | None,           # direct image URL
            "size": int | None,          # bytes
            "mime": str | None,          # content-type as reported by MW
        }

    Titles that aren't found (or that the API returns as ``missing``) get
    an empty entry. The caller decides how to treat missing rows.
    """
    if not titles:
        return {}
    rate_limit.wait()
    params: dict[str, Any] = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "categories|imageinfo",
        "cllimit": "max",
        "iiprop": "url|size|mime",
        "format": "json",
        "formatversion": 2,
    }
    data = _fetch_with_retry(
        session,
        api_url,
        params=params,
        user_agent=user_agent,
        timeout=request_timeout,
    )
    pages = (data.get("query") or {}).get("pages") or []
    out: dict[str, dict[str, Any]] = {t: {"categories": [], "url": None} for t in titles}
    for page in pages:
        title = page.get("title")
        if not isinstance(title, str):
            continue
        cats = [
            c.get("title")
            for c in (page.get("categories") or [])
            if isinstance(c.get("title"), str)
        ]
        ii = page.get("imageinfo") or []
        url: str | None = None
        size: int | None = None
        mime: str | None = None
        if ii:
            first = ii[0]
            url = first.get("url") if isinstance(first.get("url"), str) else None
            size_raw = first.get("size")
            size = int(size_raw) if isinstance(size_raw, int) else None
            mime = first.get("mime") if isinstance(first.get("mime"), str) else None
        out[title] = {
            "categories": cats,
            "url": url,
            "size": size,
            "mime": mime,
        }
    return out


def ingest_wikimedia(
    *,
    conn: sqlite3.Connection,
    config: WikimediaIngestConfig,
    session: Any | None = None,
    image_fetcher: Callable[[str, Any, float], tuple[bytes, str]] | None = None,
) -> WikimediaIngestSummary:
    """Walk seed categories, extract labels, persist listings + images.

    Parameters
    ----------
    conn:
        Open SQLite connection (migrations applied).
    config:
        :class:`WikimediaIngestConfig`. See its docstring for tuning knobs.
    session:
        Optional HTTP session-like object exposing ``.get(url, params=...,
        headers=...)`` returning an object with ``.status_code``, ``.json()``,
        ``.headers``, and ``.content``. Defaults to a fresh
        ``httpx.Client``. Injectable for tests.
    image_fetcher:
        Optional ``(url, session, timeout) -> (bytes, content_type)`` callable.
        Defaults to a thin wrapper around the same HTTP session. Injectable
        for tests.

    Returns
    -------
    WikimediaIngestSummary
        Per-run counters. ``processed`` counts every file pulled from the
        API; the rest are decomposed sub-counts.
    """
    config = config.validated()

    output_dir = pathlib.Path(config.output_dir)
    if not config.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    own_session = False
    if session is None:
        session = _open_session(user_agent=config.user_agent)
        own_session = True

    if image_fetcher is None:
        image_fetcher = _default_image_fetcher

    rate_limit = _RateLimiter(min_delay=config.rate_limit_seconds)

    summary = WikimediaIngestSummary()
    try:
        for category in config.seed_categories:
            logger.info("wikimedia: walking category=%s", category)
            cat_member_iter = iter_category_files(
                config.api_url,
                category,
                session=session,
                rate_limit=rate_limit,
                user_agent=config.user_agent,
                max_files=config.max_images_per_category,
                request_timeout=config.request_timeout_seconds,
            )
            try:
                summary = _ingest_category(
                    conn=conn,
                    config=config,
                    session=session,
                    image_fetcher=image_fetcher,
                    rate_limit=rate_limit,
                    output_dir=output_dir,
                    cat_members=cat_member_iter,
                    summary=summary,
                )
            except _IngestLimitReached as stop:
                logger.info("wikimedia: reached limit=%d, stopping", config.limit)
                return stop.summary
    finally:
        if own_session:
            with suppress(Exception):
                close = getattr(session, "close", None)
                if callable(close):
                    close()

    logger.info(
        "wikimedia: done: processed=%d listings_inserted=%d images_inserted=%d "
        "skipped_existing=%d skipped_no_label=%d skipped_out_of_year_range=%d "
        "skipped_unsupported_type=%d skipped_download_failures=%d api_errors=%d",
        summary.processed,
        summary.listings_inserted,
        summary.images_inserted,
        summary.skipped_existing,
        summary.skipped_no_label,
        summary.skipped_out_of_year_range,
        summary.skipped_unsupported_type,
        summary.skipped_download_failures,
        summary.api_errors,
    )
    return summary


# --------------------------------------------------------------- internals


class _IngestLimitReached(Exception):
    """Raised internally to short-circuit the seed-category loop on ``--limit``.

    Carries the current :class:`WikimediaIngestSummary` so the outer caller
    can propagate the in-flight counts back to the user.
    """

    def __init__(self, summary: WikimediaIngestSummary) -> None:
        super().__init__("ingest limit reached")
        self.summary = summary


def _ingest_category(
    *,
    conn: sqlite3.Connection,
    config: WikimediaIngestConfig,
    session: Any,
    image_fetcher: Callable[[str, Any, float], tuple[bytes, str]],
    rate_limit: _RateLimiter,
    output_dir: pathlib.Path,
    cat_members: Iterator[dict[str, Any]],
    summary: WikimediaIngestSummary,
) -> WikimediaIngestSummary:
    """Process every file yielded by ``cat_members`` and mutate ``summary``.

    Batched in groups of 50 (the MediaWiki API cap) so we hit the metadata
    endpoint as few times as possible.
    """
    batch: list[dict[str, Any]] = []
    for member in cat_members:
        batch.append(member)
        if len(batch) >= 50:
            summary = _process_batch(
                conn=conn,
                config=config,
                session=session,
                image_fetcher=image_fetcher,
                rate_limit=rate_limit,
                output_dir=output_dir,
                batch=batch,
                summary=summary,
            )
            batch = []
            if config.limit is not None and summary.processed >= config.limit:
                raise _IngestLimitReached(summary)
    if batch:
        summary = _process_batch(
            conn=conn,
            config=config,
            session=session,
            image_fetcher=image_fetcher,
            rate_limit=rate_limit,
            output_dir=output_dir,
            batch=batch,
            summary=summary,
        )
        if config.limit is not None and summary.processed >= config.limit:
            raise _IngestLimitReached(summary)
    return summary


def _process_batch(
    *,
    conn: sqlite3.Connection,
    config: WikimediaIngestConfig,
    session: Any,
    image_fetcher: Callable[[str, Any, float], tuple[bytes, str]],
    rate_limit: _RateLimiter,
    output_dir: pathlib.Path,
    batch: list[dict[str, Any]],
    summary: WikimediaIngestSummary,
) -> WikimediaIngestSummary:
    titles = [m["title"] for m in batch if isinstance(m.get("title"), str)]
    try:
        meta = fetch_file_metadata(
            config.api_url,
            titles,
            session=session,
            rate_limit=rate_limit,
            user_agent=config.user_agent,
            request_timeout=config.request_timeout_seconds,
        )
    except _WikimediaAPIError as exc:
        logger.warning("wikimedia: metadata fetch failed for batch of %d: %s", len(titles), exc)
        summary = summary.model_copy(update={"api_errors": summary.api_errors + 1})
        return summary

    for title in titles:
        if config.limit is not None and summary.processed >= config.limit:
            return summary
        summary = summary.model_copy(update={"processed": summary.processed + 1})

        entry = meta.get(title) or {}
        cats: list[str] = entry.get("categories") or []
        triple = extract_label_triple(cats, year_min=config.year_min, year_max=config.year_max)
        if triple is None:
            summary = summary.model_copy(update={"skipped_no_label": summary.skipped_no_label + 1})
            continue
        year, make, model = triple
        if year < config.year_min or year > config.year_max:
            summary = summary.model_copy(
                update={
                    "skipped_out_of_year_range": summary.skipped_out_of_year_range + 1,
                }
            )
            continue

        url: str | None = entry.get("url")
        if not isinstance(url, str) or not url:
            summary = summary.model_copy(
                update={"skipped_download_failures": summary.skipped_download_failures + 1}
            )
            logger.debug("wikimedia: no direct url for %s, skipping", title)
            continue

        # Dry-run short-circuit: don't fetch bytes, don't write to DB.
        if config.dry_run:
            logger.info(
                "wikimedia (dry-run): would ingest title=%r year=%d make=%r model=%r url=%s",
                title,
                year,
                make,
                model,
                url,
            )
            continue

        # Fetch bytes (with rate limit + retry).
        rate_limit.wait()
        try:
            body, content_type = image_fetcher(url, session, config.request_timeout_seconds)
        except _WikimediaAPIError as exc:
            logger.warning("wikimedia: image download failed for %s: %s", url, exc)
            summary = summary.model_copy(
                update={"skipped_download_failures": summary.skipped_download_failures + 1}
            )
            continue

        ext = _CONTENT_TYPE_EXT.get(content_type.lower())
        if ext is None:
            logger.debug(
                "wikimedia: unsupported content-type %r for %s, skipping", content_type, url
            )
            summary = summary.model_copy(
                update={"skipped_unsupported_type": summary.skipped_unsupported_type + 1}
            )
            continue

        image_id = hashlib.sha256(body).hexdigest()

        if images.get_image_by_sha(conn, image_id) is not None:
            summary = summary.model_copy(update={"skipped_existing": summary.skipped_existing + 1})
            continue

        target_path = output_dir / f"{image_id}{ext}"
        _atomic_write_bytes(target_path, body)

        listing_id = f"{_SOURCE}:{image_id}"
        canonical_make = normalize_make(make)
        canonical_model = normalize_model(model)
        generation_year = year_to_generation(year)

        if _insert_listing_if_new(
            conn,
            listing_id=listing_id,
            url=url,
            year=year,
            make=make,
            model=model,
            canonical_make=canonical_make,
            canonical_model=canonical_model,
            generation_year=generation_year,
            split=config.split,
        ):
            summary = summary.model_copy(
                update={"listings_inserted": summary.listings_inserted + 1}
            )
        if _insert_image_if_new(
            conn,
            image_id=image_id,
            listing_id=listing_id,
            source_url=url,
            local_path=target_path,
            byte_count=len(body),
        ):
            summary = summary.model_copy(update={"images_inserted": summary.images_inserted + 1})

    return summary


def _strip_category_prefix(title: str) -> str:
    """Strip a ``Category:`` namespace prefix if present, returning the bare name.

    Whitespace is collapsed to single spaces and the result is stripped.
    ``""`` is returned for empty / non-string input.
    """
    if not isinstance(title, str):
        return ""
    s = title.strip()
    if s.lower().startswith("category:"):
        s = s[len("category:") :].strip()
    # Some categories use underscores instead of spaces.
    s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def _find_year(
    categories: list[str], *, year_min: int = _VALID_YEAR_MIN, year_max: int = _VALID_YEAR_MAX
) -> int | None:
    """Pick the most-specific year from a list of category names.

    Strategy, in order of specificity:

    1. Explicit ``"Cars introduced in 1965"`` / ``"1965 automobiles"``.
    2. Year-prefixed make-bearing category: a four-digit year followed by
       something the make detector would recognize (``"1965 Ford Mustang"``).
       Required to share a known make token so we don't accidentally turn
       ``"1965 Top Album Covers"`` into a car year.
    3. Decade midpoint (``"1960s automobiles"`` -> 1965).

    Returns ``None`` if no acceptable year is found within
    ``[year_min, year_max]``.
    """
    # Pass 1: explicit year.
    for cat in categories:
        for pat in _YEAR_PATTERNS:
            m = pat.match(cat)
            if m is not None:
                year = int(m.group(1))
                if year_min <= year <= year_max:
                    return year
    # Pass 2: year-prefix on a known-make category.
    for cat in categories:
        m = _YEAR_PREFIX_RE.match(cat)
        if m is None:
            continue
        year = int(m.group(1))
        if not (year_min <= year <= year_max):
            continue
        rest = m.group(2)
        first_token = rest.split(" ", 1)[0].lower()
        if _is_known_make_token(first_token):
            return year
    # Pass 3: decade midpoint.
    for cat in categories:
        for pat in _DECADE_PATTERNS:
            m = pat.match(cat)
            if m is not None:
                decade_start = int(m.group(1))
                # Decade midpoint: e.g. 1960s -> 1965.
                midpoint = decade_start + 5
                if year_min <= midpoint <= year_max:
                    return midpoint
    return None


def _is_known_make_token(lower_key: str) -> bool:
    """Return True if ``lower_key`` is a recognized make (alias or known-list).

    Used as a recognition gate. The canonical translation still lives in
    :func:`canonical_labels.normalize_make`; this helper only answers
    membership questions.
    """
    from .canonical_labels import _MAKE_ALIAS_MAP  # noqa: PLC0415

    return lower_key in _MAKE_ALIAS_MAP or lower_key in _KNOWN_MAKES_EXTRA


def _find_make(categories: list[str]) -> str | None:
    """Return the canonical make for the first category that maps to one.

    A category is considered a "make-bearing" category when, after stripping
    common suffixes (``"(cars)"``, ``"vehicles"``, ``"automobiles"``,
    ``"motor company"``, ``"motors"``):

    1. The full stripped string lowercased is a recognized make
       (alias map or known-make set), OR
    2. The first whitespace token lowercased is a recognized make.

    The returned form is the canonical make per
    :func:`canonical_labels.normalize_make` (Title Case for the long
    tail; explicit override for aliases like ``"chevy"`` -> ``"Chevrolet"``
    and brand-cased forms like ``"bmw"`` -> ``"BMW"``).

    Returns ``None`` if no category yields a known make. This conservatism
    is intentional — generic Wikipedia categories like ``"Sedans"`` or
    ``"Concept cars"`` MUST NOT be misread as makes.
    """
    for cat in categories:
        candidate = _MAKE_SUFFIX_RE.sub("", cat).strip()
        if not candidate:
            continue
        key = candidate.lower()
        if _is_known_make_token(key):
            canonical = normalize_make(candidate)
            if canonical is not None:
                return canonical
        # Also try the very first token (e.g. "Ferrari 250 GT" -> first
        # token "Ferrari"). Some Wikipedia categories embed the model right
        # in the make line and lack a separate make category.
        first_token = candidate.split(" ", 1)[0]
        if _is_known_make_token(first_token.lower()):
            canonical = normalize_make(first_token)
            if canonical is not None:
                return canonical
    return None


def _find_model(categories: list[str], *, make: str) -> str | None:
    """Return the model string for the first category that starts with ``make``.

    Match logic: case-insensitive compare of ``make`` against the first
    word(s) of the category. The remainder (with the make stripped) is the
    model. Returns ``None`` when no category matches.

    Multi-word makes (``"Land Rover"``, ``"Alfa Romeo"``,
    ``"Mercedes-Benz"``) are handled because we check the *exact* prefix
    before the first space — not just the first token.
    """
    make_lower = make.lower()
    # Build candidate prefixes: the canonical form, plus the form returned
    # by ``_MAKE_SUFFIX_RE.sub`` would yield (i.e. raw category prefix).
    for cat in categories:
        cat_lower = cat.lower()
        if not cat_lower.startswith(make_lower):
            continue
        if len(cat) == len(make):
            # Category IS just the make name; no model token.
            continue
        next_char = cat[len(make)]
        if next_char != " " and next_char != "-":
            # ``Forde`` shouldn't match ``Ford``.
            continue
        remainder = cat[len(make) :].strip(" -").strip()
        if remainder:
            return remainder
    return None


def _open_session(*, user_agent: str) -> Any:
    """Open an ``httpx.Client`` session with the descriptive UA pre-applied."""
    try:
        import httpx  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dep listed in pyproject
        raise RuntimeError("httpx is required for the Wikimedia ingest") from exc
    return httpx.Client(headers={"User-Agent": user_agent}, follow_redirects=True)


class _WikimediaAPIError(RuntimeError):
    """Raised when a Wikimedia API or image-download call fails after retries."""


def _fetch_with_retry(
    session: Any,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    user_agent: str,
    timeout: float,
    max_retries: int = WIKIMEDIA_MAX_RETRIES,
    backoff_base: float = WIKIMEDIA_BACKOFF_BASE,
) -> dict[str, Any]:
    """GET ``url`` with the API params and retry on 429 / 5xx.

    Returns the JSON-decoded body on success. Raises
    :class:`_WikimediaAPIError` after ``max_retries`` attempts.
    """
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(
                url,
                params=params,
                headers={"User-Agent": user_agent, "Accept": "application/json"},
                timeout=timeout,
            )
            status = int(getattr(response, "status_code", 0))
            if status == 200:
                json_method = getattr(response, "json", None)
                if callable(json_method):
                    result = json_method()
                else:  # pragma: no cover - defensive
                    raise _WikimediaAPIError(f"response has no .json(): {response!r}")
                if not isinstance(result, dict):
                    raise _WikimediaAPIError(f"non-dict JSON body from {url}")
                return result
            if status in (429, 500, 502, 503, 504):
                sleep_for = backoff_base * (2 ** (attempt - 1))
                logger.warning(
                    "wikimedia: HTTP %d on %s (attempt %d/%d), sleeping %.1fs",
                    status,
                    url,
                    attempt,
                    max_retries,
                    sleep_for,
                )
                time.sleep(sleep_for)
                continue
            raise _WikimediaAPIError(f"HTTP {status} from {url}")
        except _WikimediaAPIError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            last_err = exc
            sleep_for = backoff_base * (2 ** (attempt - 1))
            logger.warning(
                "wikimedia: transport error on %s (attempt %d/%d): %r; sleeping %.1fs",
                url,
                attempt,
                max_retries,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
    raise _WikimediaAPIError(
        f"giving up on {url} after {max_retries} attempts; last_err={last_err!r}"
    )


def _default_image_fetcher(url: str, session: Any, timeout: float) -> tuple[bytes, str]:
    """Fetch ``url`` and return ``(body_bytes, content_type)``.

    Uses the same session as the metadata API (single HTTP/2 keep-alive
    connection where supported). Raises :class:`_WikimediaAPIError` on any
    transport / status-code failure.
    """
    try:
        response = session.get(url, timeout=timeout)
    except Exception as exc:  # pragma: no cover - defensive
        raise _WikimediaAPIError(f"transport error fetching image {url}: {exc!r}") from exc
    status = int(getattr(response, "status_code", 0))
    if status != 200:
        raise _WikimediaAPIError(f"HTTP {status} fetching image {url}")
    body = getattr(response, "content", b"")
    if not isinstance(body, (bytes, bytearray)):
        raise _WikimediaAPIError(f"expected bytes from image response, got {type(body).__name__}")
    headers = getattr(response, "headers", {}) or {}
    raw_ct: str | None = None
    if hasattr(headers, "get"):
        raw_ct = headers.get("content-type") or headers.get("Content-Type")
    content_type = ""
    if isinstance(raw_ct, str):
        content_type = raw_ct.split(";", 1)[0].strip().lower()
    return bytes(body), content_type


def _atomic_write_bytes(path: pathlib.Path, body: bytes) -> None:
    """Write ``body`` to ``path`` via a ``.tmp`` rename. Skip if file exists."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError:
        if tmp.exists():
            with suppress(OSError):
                tmp.unlink()
        raise


def _insert_listing_if_new(
    conn: sqlite3.Connection,
    *,
    listing_id: str,
    url: str,
    year: int,
    make: str,
    model: str,
    canonical_make: str | None,
    canonical_model: str | None,
    generation_year: int | None,
    split: str,
) -> bool:
    """Insert a listing row, returning True if a new row was created."""
    from car_lense_engine.db.listings import get_listing  # noqa: PLC0415

    if get_listing(conn, listing_id) is not None:
        return False

    listing = Listing(
        listing_id=listing_id,
        source="wikimedia_commons",
        url=url,
        year=year,
        make=make,
        model=model,
        split=split,
        canonical_make=canonical_make,
        canonical_model=canonical_model,
        generation_year=generation_year,
    )
    try:
        listings.insert_listing(conn, listing)
    except sqlite3.IntegrityError as exc:
        logger.debug("wikimedia: listing insert race for %s: %r", listing_id, exc)
        return False
    return True


def _insert_image_if_new(
    conn: sqlite3.Connection,
    *,
    image_id: str,
    listing_id: str,
    source_url: str,
    local_path: pathlib.Path,
    byte_count: int,
) -> bool:
    """Insert an image row, returning True if a new row was created."""
    if images.get_image_by_sha(conn, image_id) is not None:
        return False

    image = Image(
        image_id=image_id,
        listing_id=listing_id,
        source_url=source_url,
        local_path=str(local_path),
        bytes=byte_count,
        position=1,
    )
    try:
        images.insert_image(conn, image)
    except sqlite3.IntegrityError as exc:
        logger.debug("wikimedia: image insert race for %s: %r", image_id[:12], exc)
        return False
    return True


__all__ = [
    "WIKIMEDIA_API_URL",
    "WIKIMEDIA_USER_AGENT",
    "WikimediaIngestConfig",
    "WikimediaIngestSummary",
    "extract_label_triple",
    "fetch_file_metadata",
    "ingest_wikimedia",
    "iter_category_files",
]
