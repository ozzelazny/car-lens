"""Tests for the durable crawl_queue accessors and worker semantics."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from car_lense_engine.db import open_db, queue


def test_enqueue_and_claim(db: sqlite3.Connection) -> None:
    urls = [f"https://cars.com/listing/{i}" for i in range(3)]
    for u in urls:
        assert queue.enqueue(db, u, source="cars_com", kind="listing") is True

    claimed: list[str] = []
    for _ in range(3):
        item = queue.claim_next(db)
        assert item is not None
        assert item.status == "in_progress"
        assert item.claimed_at is not None
        claimed.append(item.url)

    assert set(claimed) == set(urls)
    assert queue.claim_next(db) is None


def test_enqueue_duplicate_returns_false(db: sqlite3.Connection) -> None:
    url = "https://cars.com/listing/1"
    assert queue.enqueue(db, url, source="cars_com", kind="listing") is True
    assert queue.enqueue(db, url, source="cars_com", kind="listing") is False


def test_claim_respects_source_filter(db: sqlite3.Connection) -> None:
    queue.enqueue(db, "https://cars.com/a", source="cars_com", kind="listing")
    queue.enqueue(db, "https://autotrader.com/a", source="autotrader", kind="listing")

    item = queue.claim_next(db, source="autotrader")
    assert item is not None
    assert item.source == "autotrader"
    assert item.url == "https://autotrader.com/a"

    # No more 'autotrader' available; cars_com is still pending.
    assert queue.claim_next(db, source="autotrader") is None
    item2 = queue.claim_next(db, source="cars_com")
    assert item2 is not None
    assert item2.source == "cars_com"


def test_mark_done_clears_error(db: sqlite3.Connection) -> None:
    queue.enqueue(db, "https://cars.com/a", source="cars_com", kind="listing")
    item = queue.claim_next(db)
    assert item is not None
    queue.mark_done(db, item.url)
    row = db.execute(
        "SELECT status, last_error FROM crawl_queue WHERE url = ?", (item.url,)
    ).fetchone()
    assert row["status"] == "done"
    assert row["last_error"] is None


def test_mark_failed_backoff(db: sqlite3.Connection) -> None:
    url = "https://cars.com/a"
    queue.enqueue(db, url, source="cars_com", kind="listing")
    item = queue.claim_next(db)
    assert item is not None

    queue.mark_failed(db, url, "boom")
    row = db.execute(
        "SELECT status, attempts, last_error, next_try_at FROM crawl_queue WHERE url = ?",
        (url,),
    ).fetchone()
    assert row["status"] == "failed"
    assert int(row["attempts"]) == 1
    assert row["last_error"] == "boom"

    # next_try_at is in the future, so claim_next should return None for this row.
    assert queue.claim_next(db) is None

    # Simulate time passing: rewind next_try_at to the past.
    past = (datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=10)).isoformat(
        sep=" ", timespec="seconds"
    )
    with db:
        db.execute(
            "UPDATE crawl_queue SET status = 'pending', next_try_at = ? WHERE url = ?",
            (past, url),
        )
    reclaimed = queue.claim_next(db)
    assert reclaimed is not None
    assert reclaimed.url == url
    assert reclaimed.attempts == 1  # preserved across retries


def test_mark_failed_dead_after_5_attempts(db: sqlite3.Connection) -> None:
    url = "https://cars.com/dead"
    queue.enqueue(db, url, source="cars_com", kind="listing")

    for _ in range(5):
        queue.mark_failed(db, url, "still broken")

    row = db.execute("SELECT status, attempts FROM crawl_queue WHERE url = ?", (url,)).fetchone()
    assert row["status"] == "dead"
    assert int(row["attempts"]) == 5


def test_stats_counts(db: sqlite3.Connection) -> None:
    queue.enqueue(db, "https://cars.com/1", source="cars_com", kind="listing")
    queue.enqueue(db, "https://cars.com/2", source="cars_com", kind="listing")
    item = queue.claim_next(db)
    assert item is not None
    queue.mark_done(db, item.url)

    stats = queue.stats(db)
    assert stats.pending == 1
    assert stats.done == 1
    assert stats.in_progress == 0
    assert stats.failed == 0
    assert stats.dead == 0


def test_concurrent_claim_no_double_pickup(db_path: Path) -> None:
    """Two connections race for the single pending row; exactly one wins."""
    bootstrap = open_db(db_path)
    try:
        queue.enqueue(bootstrap, "https://cars.com/only", source="cars_com", kind="listing")
    finally:
        bootstrap.close()

    barrier = threading.Barrier(2)
    results: list[object] = [None, None]

    def worker(idx: int) -> None:
        conn = open_db(db_path)
        # Use a generous busy_timeout so BEGIN IMMEDIATE waits instead of erroring.
        conn.execute("PRAGMA busy_timeout = 5000")
        try:
            barrier.wait()
            results[idx] = queue.claim_next(conn)
        finally:
            conn.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    winners = [r for r in results if r is not None]
    losers = [r for r in results if r is None]
    assert len(winners) == 1, f"expected exactly one winner, got {results}"
    assert len(losers) == 1


def test_requeue_resets_state(db: sqlite3.Connection) -> None:
    url = "https://cars.com/a"
    queue.enqueue(db, url, source="cars_com", kind="listing")
    for _ in range(2):
        queue.mark_failed(db, url, "oops")
    queue.requeue(db, url)

    row = db.execute(
        "SELECT status, attempts, last_error FROM crawl_queue WHERE url = ?", (url,)
    ).fetchone()
    assert row["status"] == "pending"
    assert int(row["attempts"]) == 0
    assert row["last_error"] is None


def test_invalid_kind_rejected(db: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError), db:
        db.execute(
            "INSERT INTO crawl_queue (url, source, kind) VALUES (?, ?, ?)",
            ("https://x", "cars_com", "bogus"),
        )


def test_claim_next_prefers_image_over_listing(db: sqlite3.Connection) -> None:
    queue.enqueue(db, "https://cars.com/listing/1", source="cars_com", kind="listing")
    queue.enqueue(
        db,
        "https://cars.com/image/1.jpg",
        source="cars_com",
        kind="image",
        parent_listing_id="listing-1",
    )

    first = queue.claim_next(db)
    assert first is not None
    assert first.kind == "image"
    assert first.url == "https://cars.com/image/1.jpg"

    second = queue.claim_next(db)
    assert second is not None
    assert second.kind == "listing"
    assert second.url == "https://cars.com/listing/1"


def test_claim_next_prefers_listing_over_search(db: sqlite3.Connection) -> None:
    queue.enqueue(db, "https://cars.com/search?q=foo", source="cars_com", kind="search")
    queue.enqueue(db, "https://cars.com/listing/1", source="cars_com", kind="listing")

    first = queue.claim_next(db)
    assert first is not None
    assert first.kind == "listing"
    assert first.url == "https://cars.com/listing/1"

    second = queue.claim_next(db)
    assert second is not None
    assert second.kind == "search"
    assert second.url == "https://cars.com/search?q=foo"


def test_claim_next_kind_priority_then_next_try_at(db: sqlite3.Connection) -> None:
    earlier_url = "https://cars.com/image/early.jpg"
    later_url = "https://cars.com/image/late.jpg"
    queue.enqueue(db, earlier_url, source="cars_com", kind="image")
    queue.enqueue(db, later_url, source="cars_com", kind="image")

    now = datetime.now(UTC).replace(tzinfo=None)
    earlier_iso = (now - timedelta(seconds=60)).isoformat(sep=" ", timespec="seconds")
    later_iso = (now - timedelta(seconds=10)).isoformat(sep=" ", timespec="seconds")
    with db:
        db.execute(
            "UPDATE crawl_queue SET next_try_at = ? WHERE url = ?",
            (earlier_iso, earlier_url),
        )
        db.execute(
            "UPDATE crawl_queue SET next_try_at = ? WHERE url = ?",
            (later_iso, later_url),
        )

    first = queue.claim_next(db)
    assert first is not None
    assert first.url == earlier_url

    second = queue.claim_next(db)
    assert second is not None
    assert second.url == later_url


def test_claim_next_kind_priority_respects_source_filter(db: sqlite3.Connection) -> None:
    queue.enqueue(
        db,
        "https://bat.example/image/1.jpg",
        source="bat",
        kind="image",
        parent_listing_id="bat-1",
    )
    queue.enqueue(db, "https://cars.com/listing/1", source="cars_com", kind="listing")

    # Filtering by cars_com must not pull in the bat image even though it has
    # higher kind priority globally.
    item = queue.claim_next(db, source="cars_com")
    assert item is not None
    assert item.source == "cars_com"
    assert item.kind == "listing"

    # No more pending cars_com items.
    assert queue.claim_next(db, source="cars_com") is None

    # The bat image is still available under its own source.
    bat_item = queue.claim_next(db, source="bat")
    assert bat_item is not None
    assert bat_item.source == "bat"
    assert bat_item.kind == "image"
