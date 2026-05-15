"""Search-query generator: catalog → ranked classes → per-site URLs → crawl queue."""

from __future__ import annotations

from .ranker import MAKE_POPULARITY, RankedClass, rank_models
from .seed import SeedStats, SeedUrl, build_urls_for, seed_queue
from .urls import DEFAULT_CRAIGSLIST_CITIES, SITE_BUILDERS

__all__ = [
    "DEFAULT_CRAIGSLIST_CITIES",
    "MAKE_POPULARITY",
    "RankedClass",
    "SITE_BUILDERS",
    "SeedStats",
    "SeedUrl",
    "build_urls_for",
    "rank_models",
    "seed_queue",
]
