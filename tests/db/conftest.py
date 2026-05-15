"""Shared fixtures for the DB test suite."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from car_lense_engine.db import open_db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Return a fresh on-disk SQLite path for each test."""
    return tmp_path / "crawl.sqlite"


@pytest.fixture
def db(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a fresh DB with migrations applied; close it on teardown."""
    conn = open_db(db_path)
    try:
        yield conn
    finally:
        conn.close()
