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
    "autotrader": "https://www.autotrader.com/sitemap.xml",
    "carsandbids": "https://carsandbids.com/cab-sitemap/xml",
}
"""Root sitemap URL per source identifier.

AutoTrader's ``sitemap.xml`` is a sitemap *index* (~696 bytes pointing to
sub-sitemaps); Cars & Bids' ``cab-sitemap/xml`` similarly. The walker
recurses indexes transparently.
"""


# AutoTrader listings live under ``/cars-for-sale/vehicledetails/.../<6+ digit id>``
# per ``urls.autotrader`` and the existing parser. The ID lives at the tail of
# the path; we accept an optional trailing slash.
_AUTOTRADER_PATH_RE = re.compile(r"/\d{6,}/?$")


def is_autotrader_listing(url: str) -> bool:
    """Return True if ``url`` is an AutoTrader vehicle-detail listing."""
    if "/cars-for-sale/vehicledetails/" not in url:
        return False
    return bool(_AUTOTRADER_PATH_RE.search(urlparse(url).path))


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
