"""Tests for run_crawler — orchestration loop and exit reasons."""

from __future__ import annotations

import os
import signal
import sqlite3
import threading
import time

import pytest

from car_lense_engine.crawler.core.politeness import PolicyConfig
from car_lense_engine.crawler.core.registry import ParserRegistry
from car_lense_engine.crawler.core.runner import run_crawler
from car_lense_engine.crawler.parsers.base import (
    DiscoveredUrl,
    ParsedListing,
    ParseResult,
)
from car_lense_engine.db import queue

from .conftest import FakeFetcher, FakeParser


def _quick_policy(idle: int = 1) -> PolicyConfig:
    """Fast policy for tests: tiny delays, tiny idle-exit window."""
    return PolicyConfig(
        min_delay_seconds=0.0,
        max_delay_seconds=0.0,
        idle_exit_seconds=idle,
    )


def _registry_with(parser: FakeParser) -> ParserRegistry:
    reg = ParserRegistry()
    reg.register(parser)
    return reg


def test_run_crawler_processes_until_empty(
    db: sqlite3.Connection, fake_fetcher: FakeFetcher
) -> None:
    for i in range(3):
        queue.enqueue(db, url=f"https://cars.com/l/{i}", source="cars_com", kind="listing")

    parser = FakeParser(source="cars_com", result_factory=ParseResult())
    summary = run_crawler(
        conn=db,
        fetcher=fake_fetcher,
        registry=_registry_with(parser),
        policy=_quick_policy(idle=1),
        sleep_fn=lambda _s: None,
    )
    assert summary.exit_reason == "queue_empty"
    assert summary.stats.requests_total == 3
    assert summary.stats.requests_succeeded == 3
    assert len(parser.calls) == 3


def test_run_crawler_respects_max_items(db: sqlite3.Connection, fake_fetcher: FakeFetcher) -> None:
    for i in range(10):
        queue.enqueue(db, url=f"https://cars.com/l/{i}", source="cars_com", kind="listing")
    parser = FakeParser(source="cars_com", result_factory=ParseResult())
    summary = run_crawler(
        conn=db,
        fetcher=fake_fetcher,
        registry=_registry_with(parser),
        policy=_quick_policy(idle=60),
        max_items=3,
        sleep_fn=lambda _s: None,
    )
    assert summary.exit_reason == "max_items_reached"
    assert summary.stats.requests_total == 3


def test_run_crawler_enqueues_discovered_then_drains(
    db: sqlite3.Connection, fake_fetcher: FakeFetcher
) -> None:
    """The runner should keep working as new URLs are discovered during a run."""
    queue.enqueue(db, url="https://cars.com/search", source="cars_com", kind="search")

    def _factory(*, url: str, kind: str, **_kw: object) -> ParseResult:
        if kind == "search":
            return ParseResult(
                new_urls=[
                    DiscoveredUrl(
                        url=f"https://cars.com/listing/{i}",
                        source="cars_com",
                        kind="listing",
                    )
                    for i in range(2)
                ]
            )
        return ParseResult(
            new_listing=ParsedListing(
                listing_id=f"cars_com:{url.rsplit('/', 1)[-1]}",
                source="cars_com",
                url=url,
            )
        )

    parser = FakeParser(source="cars_com", result_factory=_factory)
    summary = run_crawler(
        conn=db,
        fetcher=fake_fetcher,
        registry=_registry_with(parser),
        policy=_quick_policy(idle=1),
        sleep_fn=lambda _s: None,
    )
    # 1 search + 2 discovered listings.
    assert summary.stats.requests_succeeded == 3
    assert summary.stats.listings_inserted == 2
    assert summary.stats.urls_enqueued == 2
    assert summary.exit_reason == "queue_empty"


def test_run_crawler_idle_exit(db: sqlite3.Connection, fake_fetcher: FakeFetcher) -> None:
    """Empty queue should idle-exit roughly within the configured window."""
    parser = FakeParser(source="cars_com", result_factory=ParseResult())
    started = time.monotonic()
    summary = run_crawler(
        conn=db,
        fetcher=fake_fetcher,
        registry=_registry_with(parser),
        policy=_quick_policy(idle=1),
    )
    elapsed = time.monotonic() - started
    assert summary.exit_reason == "queue_empty"
    assert summary.stats.requests_total == 0
    # Allow a generous upper bound to keep the test stable on slow CI hosts.
    assert elapsed < 5.0


@pytest.mark.skipif(os.name != "posix", reason="signal install only tested on POSIX")
def test_run_crawler_signal_handler_set_and_restored(
    db: sqlite3.Connection, fake_fetcher: FakeFetcher
) -> None:
    sentinel = signal.getsignal(signal.SIGINT)

    captured: dict[str, object] = {}

    def _factory(*, url: str, **_kw: object) -> ParseResult:
        captured["sigint_during_run"] = signal.getsignal(signal.SIGINT)
        return ParseResult()

    queue.enqueue(db, url="https://cars.com/x", source="cars_com", kind="listing")
    parser = FakeParser(source="cars_com", result_factory=_factory)
    run_crawler(
        conn=db,
        fetcher=fake_fetcher,
        registry=_registry_with(parser),
        policy=_quick_policy(idle=1),
        sleep_fn=lambda _s: None,
    )

    # The handler in place during the run must have been our installed one (not the sentinel).
    assert captured["sigint_during_run"] is not sentinel
    # And it must be restored on exit.
    assert signal.getsignal(signal.SIGINT) is sentinel


@pytest.mark.skipif(os.name != "posix", reason="signal delivery test is POSIX-only")
def test_run_crawler_exits_on_signal(db: sqlite3.Connection, fake_fetcher: FakeFetcher) -> None:
    """Deliver SIGTERM mid-run and confirm the runner exits with exit_reason='signal'."""
    # Lots of pending items so the loop doesn't drain before we signal.
    for i in range(50):
        queue.enqueue(db, url=f"https://cars.com/l/{i}", source="cars_com", kind="listing")

    parser = FakeParser(source="cars_com", result_factory=ParseResult())
    pid = os.getpid()

    def _send() -> None:
        # Small delay to let the runner install its handler and start working.
        time.sleep(0.1)
        os.kill(pid, signal.SIGTERM)

    threading.Thread(target=_send, daemon=True).start()
    summary = run_crawler(
        conn=db,
        fetcher=fake_fetcher,
        registry=ParserRegistry() if False else _registry_with(parser),
        policy=_quick_policy(idle=60),
        sleep_fn=lambda _s: None,
    )
    assert summary.exit_reason == "signal"


def test_run_crawler_progress_logging_smoke(
    db: sqlite3.Connection, fake_fetcher: FakeFetcher, caplog: pytest.LogCaptureFixture
) -> None:
    """Confirm a progress line is emitted at the configured cadence."""
    for i in range(5):
        queue.enqueue(db, url=f"https://cars.com/l/{i}", source="cars_com", kind="listing")
    parser = FakeParser(source="cars_com", result_factory=ParseResult())
    caplog.set_level("INFO")
    summary = run_crawler(
        conn=db,
        fetcher=fake_fetcher,
        registry=_registry_with(parser),
        policy=_quick_policy(idle=1),
        progress_every=2,
        sleep_fn=lambda _s: None,
    )
    assert summary.exit_reason == "queue_empty"
    progress_lines = [r for r in caplog.records if "progress:" in r.getMessage()]
    assert progress_lines, "expected at least one progress log line"
