"""Site-agnostic crawler runtime.

Public surface:

* :class:`Fetcher`, :class:`FetchedPage`, :class:`FetchError` — fetcher protocol
* :class:`PlaywrightFetcher` — default implementation (Playwright + stealth)
* :class:`PolicyConfig` — politeness / pacing knobs
* :class:`ParserRegistry` — Phase 2 plug-in point
* :class:`Worker`, :class:`WorkerStats` — single in-flight URL loop
* :func:`run_crawler` and :class:`RunSummary` — top-level orchestrator
"""

from __future__ import annotations

from .fetcher import FetchedPage, Fetcher, FetchError
from .politeness import PolicyConfig, is_off_peak, jittered_delay, sleep_until_off_peak
from .registry import ParserRegistry
from .runner import RunSummary, run_crawler
from .worker import Worker, WorkerStats

__all__ = [
    "FetchError",
    "FetchedPage",
    "Fetcher",
    "ParserRegistry",
    "PolicyConfig",
    "RunSummary",
    "Worker",
    "WorkerStats",
    "is_off_peak",
    "jittered_delay",
    "run_crawler",
    "sleep_until_off_peak",
]
