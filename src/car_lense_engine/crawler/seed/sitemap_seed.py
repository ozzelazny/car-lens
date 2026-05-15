"""Sitemap-driven seeding for sites whose search surfaces are unreachable.

AutoTrader and Cars & Bids both block their search / listing endpoints
behind Akamai or Cloudflare interstitials yet expose machine-readable
sitemap XML at well-known paths (see ``BLOCKS_DIAGNOSTIC.md``). This module
plugs the :class:`~car_lense_engine.crawler.core.sitemap.SitemapWalker` into
the existing crawl-queue contract:

#. ``SITEMAP_ROOTS`` declares the root sitemap URL per source.
#. Per-source ``listing_url_filter`` callables decide which URLs out of the
   walk are individual vehicle / auction pages versus category / static
   index pages that should be skipped.
#. :func:`seed_queue_from_sitemap` walks, filters, and enqueues to the
   durable crawl queue as ``kind='listing'`` rows so the worker fetches
   each immediately (no intermediate search-page step).

The listing-URL filters intentionally live next to the seeder rather than
inside :mod:`urls.py` — :mod:`urls.py` is for *outbound* search-URL
construction, this module is for *inbound* sitemap-URL classification.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Callable
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from car_lense_engine.crawler.core.sitemap import SitemapWalker
from car_lense_engine.db import queue

logger = logging.getLogger(__name__)


SITEMAP_ROOTS: dict[str, str] = {
    # AutoTrader's top-level ``sitemap.xml`` is an index whose first six child
    # sitemaps are static-taxonomy / marketing / dealer content (the 2026-05-15
    # diagnostic confirmed 0 vehicle URLs in the first 50 walked from there).
    # The real vehicle URLs live under ``marketplace/sitemaps/inventory.xml``
    # — point the seeder directly at that 39 MB urlset to skip the wasted
    # 6-branch BFS.
    "autotrader": "https://www.autotrader.com/marketplace/sitemaps/inventory.xml",
    # Cars & Bids' robots.txt advertises a sitemap *index* at
    # ``/cab-sitemap/xml_sitemap.xml`` which points at three child sub-sitemaps
    # (``auctions.xml``, ``auction-videos.xml``, ``makes.xml``). Smoke run #6
    # showed the walker's BFS burning its 10K-URL budget on sibling sub-sitemaps
    # before reaching ``auctions.xml`` and producing 0 matches. Point the seeder
    # directly at the auctions child urlset (~2 MB, ~9.9K auction URLs in the
    # expected ``/auctions/<slug>/<title>`` shape) to skip the index BFS — same
    # pattern used for AutoTrader's inventory.xml above.
    "carsandbids": "https://carsandbids.com/cab-sitemap/auctions.xml",
}
"""Root sitemap URL per source identifier.

AutoTrader's inventory sitemap is a urlset (~39 MB) of vehicle-detail
URLs. Cars & Bids' ``cab-sitemap/xml_sitemap.xml`` is a sitemap index;
the walker recurses indexes transparently.
"""


# AutoTrader listings live under one of two ``/cars-for-sale/...`` prefixes
# and end in a 6+ digit numeric id at the tail of the path. AutoTrader uses
# two distinct URL shape families (see ``crawler/parsers/autotrader.py``):
#   * ``/cars-for-sale/vehicledetails/{slug}/{id}``   — slash-separated id
#   * ``/cars-for-sale/vehicledetails/{slug}-{id}``   — slug-embedded id
#   * ``/cars-for-sale/vehicle/{id}``                 — marketplace inventory
#     sitemap form (no slug; id is the final segment).
# The parser handles all of these via ``re.search`` on ``(\d{6,})/?$``; mirror
# that here by accepting either separator before the digit run. An optional
# trailing slash is tolerated.
_AUTOTRADER_PATH_RE = re.compile(r"(?:[/-])\d{6,}/?$")
_AUTOTRADER_LISTING_PREFIXES: tuple[str, ...] = (
    "/cars-for-sale/vehicledetails/",
    "/cars-for-sale/vehicle/",
)


def is_autotrader_listing(url: str) -> bool:
    """Return True if ``url`` is an AutoTrader vehicle-detail listing."""
    path = urlparse(url).path
    if not any(prefix in path for prefix in _AUTOTRADER_LISTING_PREFIXES):
        return False
    return bool(_AUTOTRADER_PATH_RE.search(path))


def is_carsandbids_listing(url: str) -> bool:
    """Return True if ``url`` is a Cars & Bids individual auction page.

    Cars & Bids serves two valid URL shapes for an auction:

    * ``/auctions/<slug>`` — a single segment under ``/auctions/`` (the
      short / shareable form a human types or links to).
    * ``/auctions/<short-id>/<title-slug>`` — the canonical form emitted
      by ``cab-sitemap/auctions.xml`` (e.g.
      ``/auctions/9aQM0NwG/2017-jeep-wrangler-unlimited-sahara-4x4``).
      Both segments are listing identifiers, not sub-pages.

    Sub-pages of an auction (``.../bids``, ``.../comments``, etc.) add a
    third trailing segment and are rejected. The bare ``/auctions/`` and
    ``/auctions`` indexes are also rejected.
    """
    parsed = urlparse(url)
    path = parsed.path
    if not path.startswith("/auctions/"):
        return False
    if path == "/auctions/" or path == "/auctions":
        return False
    segments = [seg for seg in path.strip("/").split("/") if seg]
    # segments[0] is always "auctions" given the prefix check above. Accept
    # either one or two trailing segments (1991-honda-crx-si OR
    # 9aQM0NwG/2017-jeep-wrangler...); reject deeper paths like
    # /auctions/<slug>/bids.
    return len(segments) in (2, 3)


ListingUrlFilter = Callable[[str], bool]

LISTING_FILTERS: dict[str, ListingUrlFilter] = {
    "autotrader": is_autotrader_listing,
    "carsandbids": is_carsandbids_listing,
}
"""Per-source filters that recognise individual listing URLs.

Keys must intersect with :data:`SITEMAP_ROOTS` so :func:`seed_queue_from_sitemap`
can dispatch by source identifier.
"""


class SitemapSeedStats(BaseModel):
    """Result summary from :func:`seed_queue_from_sitemap`."""

    model_config = ConfigDict(extra="forbid")

    source: str
    walked: int = 0
    matched: int = 0
    inserted: int = 0
    duplicates: int = 0
    examples: list[str] = Field(default_factory=list)


def seed_queue_from_sitemap(
    conn: sqlite3.Connection,
    source: str,
    walker: SitemapWalker,
    *,
    listing_url_filter: ListingUrlFilter | None = None,
    max_listings: int | None = None,
) -> SitemapSeedStats:
    """Walk ``source``'s sitemap, filter to listing URLs, and enqueue them.

    Parameters
    ----------
    conn:
        Open SQLite connection produced by :func:`car_lense_engine.db.open_db`.
    source:
        Source identifier (must appear in :data:`SITEMAP_ROOTS`).
    walker:
        A constructed :class:`SitemapWalker`. The caller owns the underlying
        fetcher and is responsible for closing it; the walker is reusable.
    listing_url_filter:
        Optional override for the listing-URL filter. Defaults to the
        per-source filter from :data:`LISTING_FILTERS`.
    max_listings:
        Hard cap on how many listings to enqueue. ``None`` (default) means
        "no cap beyond the walker's own ``max_urls``".
    """
    if source not in SITEMAP_ROOTS:
        raise ValueError(f"unknown sitemap source: {source!r}; known: {sorted(SITEMAP_ROOTS)}")
    if listing_url_filter is None:
        if source not in LISTING_FILTERS:
            raise ValueError(
                f"no default listing-URL filter for source {source!r}; "
                f"pass listing_url_filter=<callable>"
            )
        listing_url_filter = LISTING_FILTERS[source]

    root_url = SITEMAP_ROOTS[source]
    stats = SitemapSeedStats(source=source)
    for url in walker.walk(root_url):
        stats.walked += 1
        if not listing_url_filter(url):
            continue
        stats.matched += 1
        if len(stats.examples) < 5:
            stats.examples.append(url)
        inserted = queue.enqueue(
            conn,
            url=url,
            source=source,
            kind="listing",
            target_year=None,
            target_make=None,
            target_model=None,
        )
        if inserted:
            stats.inserted += 1
        else:
            stats.duplicates += 1
        if max_listings is not None and stats.matched >= max_listings:
            break
    logger.info(
        "seed_queue_from_sitemap[%s]: walked=%d matched=%d inserted=%d duplicates=%d",
        source,
        stats.walked,
        stats.matched,
        stats.inserted,
        stats.duplicates,
    )
    return stats
