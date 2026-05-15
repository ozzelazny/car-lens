"""Search-query generator: catalog → ranked classes → per-site URLs → crawl queue."""

from __future__ import annotations

from .ranker import MAKE_POPULARITY, RankedClass, rank_models
from .seed import SeedStats, SeedUrl, build_urls_for, seed_queue
from .sitemap_seed import (
    LISTING_FILTERS,
    SITEMAP_ROOTS,
    SitemapSeedStats,
    is_autotrader_listing,
    is_carsandbids_listing,
    seed_queue_from_sitemap,
)
from .urls import DEFAULT_CRAIGSLIST_CITIES, SITE_BUILDERS

__all__ = [
    "DEFAULT_CRAIGSLIST_CITIES",
    "LISTING_FILTERS",
    "MAKE_POPULARITY",
    "RankedClass",
    "SITE_BUILDERS",
    "SITEMAP_ROOTS",
    "SeedStats",
    "SeedUrl",
    "SitemapSeedStats",
    "build_urls_for",
    "is_autotrader_listing",
    "is_carsandbids_listing",
    "rank_models",
    "seed_queue",
    "seed_queue_from_sitemap",
]
