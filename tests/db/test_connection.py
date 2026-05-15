"""Tests for the connection helper and migration runner."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from car_lense_engine.db import open_db

_EXPECTED_TABLES = {
    "listings",
    "images",
    "crawl_queue",
    "dedupe_phash",
    "schema_version",
}


def _table_names(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {row["name"] for row in cur.fetchall()}


def test_open_db_creates_schema(db_path: Path) -> None:
    conn = open_db(db_path)
    try:
        names = _table_names(conn)
        missing = _EXPECTED_TABLES - names
        assert not missing, f"missing tables: {missing}"
    finally:
        conn.close()


def test_open_db_idempotent(db_path: Path) -> None:
    conn1 = open_db(db_path)
    tables_before = _table_names(conn1)
    version_before = conn1.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()["v"]
    conn1.close()

    conn2 = open_db(db_path)
    try:
        tables_after = _table_names(conn2)
        version_after = conn2.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()[
            "v"
        ]
        assert tables_before == tables_after
        assert version_before == version_after
    finally:
        conn2.close()


def test_pragmas_set(db: sqlite3.Connection) -> None:
    mode = db.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"

    fk = db.execute("PRAGMA foreign_keys").fetchone()[0]
    assert int(fk) == 1
