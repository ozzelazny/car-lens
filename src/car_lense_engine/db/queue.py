"""Typed accessors for the durable ``crawl_queue`` table.

The queue is the resumability point for the crawler: state survives crashes,
workers can be restarted, and ``claim_next`` is safe under concurrent access
because it uses ``BEGIN IMMEDIATE`` to serialise selection-then-update.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from .models import QueueItem, QueueStats

# Backoff cap: 1 hour, regardless of attempts.
_BACKOFF_CAP_SECONDS: int = 3600

# After this many failures a queue item is parked as 'dead'.
_DEAD_AFTER_ATTEMPTS: int = 5


def _now() -> datetime:
    """Single source of truth for "now" — overridden in tests via monkeypatch.

    Returns a *naive* UTC datetime. SQLite's default TIMESTAMP converter
    parses ``YYYY-MM-DD HH:MM:SS[.ffffff]`` and rejects strings with
    timezone offsets, so we store wall-clock UTC without an offset.
    """
    return datetime.now(UTC).replace(tzinfo=None)


def _iso(dt: datetime) -> str:
    """SQLite-friendly ISO-8601 timestamp string (naive, space separator)."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt.isoformat(sep=" ", timespec="seconds")


def enqueue(
    conn: sqlite3.Connection,
    url: str,
    source: str,
    kind: str,
    target_year: int | None = None,
    target_make: str | None = None,
    target_model: str | None = None,
    parent_listing_id: str | None = None,
) -> bool:
    """Add a URL to the queue. Returns ``True`` if inserted, ``False`` on duplicate."""
    sql = (
        "INSERT OR IGNORE INTO crawl_queue ("
        "  url, source, kind, target_year, target_make, target_model, parent_listing_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    with conn:
        cur = conn.execute(
            sql,
            (url, source, kind, target_year, target_make, target_model, parent_listing_id),
        )
        return cur.rowcount > 0


def claim_next(
    conn: sqlite3.Connection,
    source: str | None = None,
) -> QueueItem | None:
    """Atomically claim one pending queue item whose ``next_try_at`` has passed.

    Uses ``BEGIN IMMEDIATE`` so two competing workers cannot pick the same row:
    the second one will see the row's status flipped to ``in_progress`` and
    will fall through to the next eligible row (or return ``None``).
    """
    now = _now()
    now_iso = _iso(now)

    # Manage the transaction explicitly — we need IMMEDIATE locking.
    conn.execute("BEGIN IMMEDIATE")
    try:
        params: tuple[object, ...]
        if source is None:
            select_sql = (
                "SELECT * FROM crawl_queue "
                "WHERE status = 'pending' AND next_try_at <= ? "
                "ORDER BY next_try_at, enqueued_at LIMIT 1"
            )
            params = (now_iso,)
        else:
            select_sql = (
                "SELECT * FROM crawl_queue "
                "WHERE status = 'pending' AND source = ? AND next_try_at <= ? "
                "ORDER BY next_try_at, enqueued_at LIMIT 1"
            )
            params = (source, now_iso)
        row = conn.execute(select_sql, params).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None

        conn.execute(
            "UPDATE crawl_queue SET status = 'in_progress', claimed_at = ? WHERE url = ?",
            (now_iso, row["url"]),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    # Re-read after update so the returned model reflects the new status/claimed_at.
    fresh = conn.execute("SELECT * FROM crawl_queue WHERE url = ?", (row["url"],)).fetchone()
    return QueueItem.model_validate(dict(fresh))


def mark_done(conn: sqlite3.Connection, url: str) -> None:
    """Mark a previously-claimed URL as successfully processed."""
    with conn:
        conn.execute(
            "UPDATE crawl_queue SET status = 'done', last_error = NULL WHERE url = ?",
            (url,),
        )


def mark_failed(conn: sqlite3.Connection, url: str, error: str) -> None:
    """Mark a URL as failed.

    Increments ``attempts``, records the error message, schedules the next
    retry using exponential backoff (capped at one hour), and parks the row
    as ``dead`` once attempts cross the ``_DEAD_AFTER_ATTEMPTS`` threshold.
    """
    row = conn.execute("SELECT attempts FROM crawl_queue WHERE url = ?", (url,)).fetchone()
    if row is None:
        return
    new_attempts = int(row["attempts"]) + 1
    backoff = min(2**new_attempts, _BACKOFF_CAP_SECONDS)
    next_try_dt = _now() + timedelta(seconds=backoff)
    next_try = _iso(next_try_dt)
    new_status = "dead" if new_attempts >= _DEAD_AFTER_ATTEMPTS else "failed"
    with conn:
        conn.execute(
            "UPDATE crawl_queue SET "
            "  status = ?, attempts = ?, last_error = ?, next_try_at = ? "
            "WHERE url = ?",
            (new_status, new_attempts, error, next_try, url),
        )


def mark_dead(conn: sqlite3.Connection, url: str, error: str | None = None) -> None:
    """Force a URL into the terminal ``dead`` state."""
    with conn:
        conn.execute(
            "UPDATE crawl_queue SET status = 'dead', last_error = COALESCE(?, last_error) "
            "WHERE url = ?",
            (error, url),
        )


def requeue(conn: sqlite3.Connection, url: str) -> None:
    """Reset a failed/dead row back to ``pending`` and clear retry scheduling."""
    with conn:
        conn.execute(
            "UPDATE crawl_queue SET "
            "  status = 'pending', attempts = 0, last_error = NULL, "
            "  next_try_at = CURRENT_TIMESTAMP, claimed_at = NULL "
            "WHERE url = ?",
            (url,),
        )


def stats(conn: sqlite3.Connection, source: str | None = None) -> QueueStats:
    """Return per-status counts, optionally filtered to one source."""
    if source is None:
        sql = "SELECT status, COUNT(*) AS n FROM crawl_queue GROUP BY status"
        rows = conn.execute(sql).fetchall()
    else:
        sql = "SELECT status, COUNT(*) AS n FROM crawl_queue WHERE source = ? GROUP BY status"
        rows = conn.execute(sql, (source,)).fetchall()
    counts: dict[str, int] = {r["status"]: int(r["n"]) for r in rows}
    return QueueStats(
        pending=counts.get("pending", 0),
        in_progress=counts.get("in_progress", 0),
        done=counts.get("done", 0),
        failed=counts.get("failed", 0),
        dead=counts.get("dead", 0),
    )
