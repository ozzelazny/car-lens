"""Crawler module — Playwright + stealth listing fetchers.

The site-agnostic runtime lives in :mod:`car_lense_engine.crawler.core` and
the per-site parsers in :mod:`car_lense_engine.crawler.parsers`. This package
re-exports the most commonly used names for convenience.
"""

from __future__ import annotations

from .core import (
    FetchedPage,
    Fetcher,
    FetchError,
    ParserRegistry,
    PolicyConfig,
    RunSummary,
    Worker,
    WorkerStats,
    run_crawler,
)
from .parsers import DiscoveredUrl, ParsedListing, Parser, ParseResult

__all__ = [
    "DiscoveredUrl",
    "FetchError",
    "FetchedPage",
    "Fetcher",
    "ParseResult",
    "ParsedListing",
    "Parser",
    "ParserRegistry",
    "PolicyConfig",
    "RunSummary",
    "Worker",
    "WorkerStats",
    "run_crawler",
]
