"""Parser protocol and result types — the contract Phase 2 parsers implement.

A Parser turns rendered HTML into one of two things (or both):

* a list of :class:`DiscoveredUrl` to enqueue (search-page case, image links, ...)
* a :class:`ParsedListing` describing the canonical vehicle on a listing page

The crawler core knows nothing about specific sites; per-site parsers register
themselves in a :class:`~car_lense_engine.crawler.core.registry.ParserRegistry`
and the worker dispatches by ``QueueItem.source``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class DiscoveredUrl(BaseModel):
    """A URL the parser wants the crawler to enqueue.

    ``kind`` mirrors the queue's ``kind`` column:

    * ``listing`` — an individual vehicle listing page
    * ``image``   — a direct image asset
    * ``search``  — a paginated next-page (next-of-search) URL
    """

    model_config = ConfigDict(extra="forbid")

    url: str
    source: str
    kind: str
    parent_listing_id: str | None = None
    target_year: int | None = None
    target_make: str | None = None
    target_model: str | None = None


class ParsedListing(BaseModel):
    """A complete listing extracted from a listing page."""

    model_config = ConfigDict(extra="forbid")

    listing_id: str
    source: str
    url: str
    year: int | None = None
    make: str | None = None
    model: str | None = None
    trim: str | None = None
    body_style: str | None = None
    mileage: int | None = None
    vin: str | None = None
    raw_html_sha256: str | None = None
    image_urls: list[str] = Field(default_factory=list)


class ParseResult(BaseModel):
    """What every parser call returns.

    All fields default to empty / ``None`` so a parser can return only the
    pieces it produced (e.g., a search-page parser sets only ``new_urls``).
    """

    model_config = ConfigDict(extra="forbid")

    new_urls: list[DiscoveredUrl] = Field(default_factory=list)
    new_listing: ParsedListing | None = None
    notes: list[str] = Field(default_factory=list)


@runtime_checkable
class Parser(Protocol):
    """Per-site parser. Phase 2 implements one per source."""

    source: str

    def parse(
        self,
        *,
        html: str,
        url: str,
        kind: str,
        hints: dict[str, str | int | None],
    ) -> ParseResult:
        """Parse a page and return discovered URLs / parsed listing.

        Parameters
        ----------
        html:
            Rendered HTML returned by the fetcher.
        url:
            Final URL of the fetched page (post-redirects).
        kind:
            Matches the queue item's kind: ``'search' | 'listing' | 'image'``.
        hints:
            Context from the queue: ``target_year``, ``target_make``,
            ``target_model`` (any may be ``None``).
        """
        ...
