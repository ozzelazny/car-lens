"""Orchestrator: ranked-class list → per-site URLs → SQLite crawl queue."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Iterator

from pydantic import BaseModel, ConfigDict, Field

from car_lense_engine.db import queue

from .ranker import RankedClass
from .urls import SITE_BUILDERS

logger = logging.getLogger(__name__)


class SeedUrl(BaseModel):
    """One generated search URL tagged with the class it represents."""

    model_config = ConfigDict(extra="forbid")

    url: str
    source: str
    target_year: int | None
    target_make: str
    target_model: str


class SeedStats(BaseModel):
    """Result summary from :func:`seed_queue`."""

    model_config = ConfigDict(extra="forbid")

    total_yielded: int = 0
    inserted: int = 0
    duplicates: int = 0
    per_site: dict[str, int] = Field(default_factory=dict)


def build_urls_for(
    ranked: list[RankedClass],
    sites: Iterable[str],
    cities: list[str] | None = None,
) -> Iterator[SeedUrl]:
    """Yield one :class:`SeedUrl` per (class, site, generated-URL) combination."""
    site_list = list(sites)
    for site in site_list:
        if site not in SITE_BUILDERS:
            raise ValueError(f"unknown site identifier: {site!r}")

    for rc in ranked:
        for site in site_list:
            builder = SITE_BUILDERS[site]
            if site == "craigslist":
                urls = builder(rc.make, rc.model, rc.year_min, rc.year_max, cities=cities)
            else:
                urls = builder(rc.make, rc.model, rc.year_min, rc.year_max)
            for url in urls:
                yield SeedUrl(
                    url=url,
                    source=site,
                    target_year=rc.year_min,
                    target_make=rc.make,
                    target_model=rc.model,
                )


def seed_queue(
    conn: sqlite3.Connection,
    ranked: list[RankedClass],
    sites: list[str],
    cities: list[str] | None = None,
) -> SeedStats:
    """Generate URLs and enqueue them into the durable crawl queue."""
    stats = SeedStats()
    for seed in build_urls_for(ranked, sites, cities=cities):
        stats.total_yielded += 1
        stats.per_site[seed.source] = stats.per_site.get(seed.source, 0) + 1
        inserted = queue.enqueue(
            conn,
            url=seed.url,
            source=seed.source,
            kind="search",
            target_year=seed.target_year,
            target_make=seed.target_make,
            target_model=seed.target_model,
        )
        if inserted:
            stats.inserted += 1
        else:
            stats.duplicates += 1
    logger.info(
        "seed_queue: yielded=%d inserted=%d duplicates=%d per_site=%s",
        stats.total_yielded,
        stats.inserted,
        stats.duplicates,
        stats.per_site,
    )
    return stats
