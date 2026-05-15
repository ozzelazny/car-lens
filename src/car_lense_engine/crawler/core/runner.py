"""Top-level run loop: orchestrate one :class:`Worker`, signal-safe, idle-exit.

The runner owns the lifecycle:

* installs SIGINT/SIGTERM handlers that flip an internal stop flag,
* lets the in-flight item finish before exiting,
* honours an optional ``max_items`` cap,
* exits when the queue stays empty for ``policy.idle_exit_seconds``,
* restores the previous signal handlers on exit.
"""

from __future__ import annotations

import contextlib
import logging
import random
import signal
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from car_lense_engine.db import queue

from .fetcher import Fetcher
from .politeness import PolicyConfig, sleep_until_off_peak
from .registry import ParserRegistry
from .worker import Worker, WorkerStats

logger = logging.getLogger(__name__)


ExitReason = str  # 'queue_empty' | 'max_items_reached' | 'signal' | 'fatal'


class RunSummary(BaseModel):
    """Final report from :func:`run_crawler`."""

    model_config = ConfigDict(extra="forbid")

    stats: WorkerStats
    exit_reason: ExitReason
    elapsed_seconds: float


def run_crawler(
    *,
    conn: sqlite3.Connection,
    fetcher: Fetcher,
    registry: ParserRegistry,
    policy: PolicyConfig,
    source: str | None = None,
    max_items: int | None = None,
    progress_every: int = 10,
    rng: random.Random | None = None,
    clock: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    idle_poll_seconds: float = 1.0,
) -> RunSummary:
    """Run a single worker until the queue drains, ``max_items`` is reached, or a signal arrives.

    Signal handlers for SIGINT and SIGTERM are installed only while the loop
    runs and restored on exit, so the function is safe to call from libraries.
    On non-main threads (or platforms that disallow signal install) the loop
    still runs — just without graceful-shutdown via signals.
    """
    stop_event = threading.Event()

    def _handler(signum: int, _frame: Any) -> None:  # noqa: ANN401 - signal API
        logger.info("received signal %d; stopping after in-flight item", signum)
        stop_event.set()

    previous_handlers: list[tuple[int, Any]] = []

    def _install_signal_handlers() -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous = signal.signal(sig, _handler)
            except (ValueError, OSError):
                # Not on the main thread, or platform doesn't support it.
                continue
            previous_handlers.append((sig, previous))

    def _restore_signal_handlers() -> None:
        for sig, prev in previous_handlers:
            with contextlib.suppress(ValueError, OSError):  # pragma: no cover - defensive
                signal.signal(sig, prev)

    worker = Worker(
        conn=conn,
        fetcher=fetcher,
        registry=registry,
        policy=policy,
        rng=rng,
        clock=clock,
        sleep_fn=sleep_fn,
    )

    started = time.monotonic()
    items_processed = 0
    last_work_at = time.monotonic()
    exit_reason: ExitReason = "queue_empty"
    sleeper: Callable[[float], None] = sleep_fn if sleep_fn is not None else time.sleep

    _install_signal_handlers()
    try:
        # Optional off-peak gate. Honours the same stop flag — though sleep_until_off_peak
        # blocks for a single long sleep; we accept that cost rather than slicing it up.
        sleep_until_off_peak(policy, sleep_fn=sleeper)

        while True:
            if stop_event.is_set():
                exit_reason = "signal"
                break
            if max_items is not None and items_processed >= max_items:
                exit_reason = "max_items_reached"
                break

            did_work = worker.run_one(source=source)

            if did_work:
                items_processed += 1
                last_work_at = time.monotonic()
                if progress_every > 0 and items_processed % progress_every == 0:
                    _log_progress(conn, worker.stats, items_processed, source=source)
            else:
                idle_for = time.monotonic() - last_work_at
                if idle_for >= policy.idle_exit_seconds:
                    exit_reason = "queue_empty"
                    break
                # Wait a bit before polling again, but check stop_event periodically.
                sleeper(min(idle_poll_seconds, max(0.0, policy.idle_exit_seconds - idle_for)))
    except Exception:
        logger.exception("fatal error in run_crawler")
        exit_reason = "fatal"
    finally:
        _restore_signal_handlers()

    elapsed = time.monotonic() - started
    summary = RunSummary(stats=worker.stats, exit_reason=exit_reason, elapsed_seconds=elapsed)
    logger.info(
        "run_crawler done: exit_reason=%s items=%d listings=%d urls_enqueued=%d "
        "succeeded=%d failed=%d elapsed=%.1fs",
        summary.exit_reason,
        items_processed,
        summary.stats.listings_inserted,
        summary.stats.urls_enqueued,
        summary.stats.requests_succeeded,
        summary.stats.requests_failed,
        summary.elapsed_seconds,
    )
    return summary


def _log_progress(
    conn: sqlite3.Connection,
    stats: WorkerStats,
    items_processed: int,
    source: str | None,
) -> None:
    """Emit one progress line with current queue + worker stats."""
    q = queue.stats(conn, source=source)
    logger.info(
        "progress: processed=%d listings=%d urls_enqueued=%d ok=%d fail=%d "
        "queue[pending=%d in_progress=%d done=%d failed=%d dead=%d]",
        items_processed,
        stats.listings_inserted,
        stats.urls_enqueued,
        stats.requests_succeeded,
        stats.requests_failed,
        q.pending,
        q.in_progress,
        q.done,
        q.failed,
        q.dead,
    )
