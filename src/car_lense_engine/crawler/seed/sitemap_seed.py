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
    # Cars & Bids' robots.txt advertises the sitemap at
    # ``/cab-sitemap/xml_sitemap.xml`` (with ``_sitemap.xml`` suffix). The
    # previously used ``/cab-sitemap/xml`` returned an HTML SPA shell rather
    # than XML.
    "carsandbids": "https://carsandbids.com/cab-sitemap/xml_sitemap.xml",
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

    Auction pages live at ``/auctions/<slug>`` (exactly two path segments,
    not the bare ``/auctions/`` index).
    """
    parsed = urlparse(url)
    path = parsed.path
    if not path.startswith("/auctions/"):
        return False
    if path == "/auctions/" or path == "/auctions":
        return False
    segments = [seg for seg in path.strip("/").split("/") if seg]
    return len(segments) == 2


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
