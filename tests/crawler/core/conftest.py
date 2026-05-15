"""Shared fixtures for the crawler-core test suite."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from car_lense_engine.crawler.core.fetcher import FetchedPage, FetchError
from car_lense_engine.crawler.parsers.base import ParseResult
from car_lense_engine.db import open_db, queue

# ---------- DB fixtures ------------------------------------------------------


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


# ---------- Fetcher fixtures -------------------------------------------------


@dataclass
class FakeFetcher:
    """In-memory fetcher. Returns canned :class:`FetchedPage` per URL or raises."""

    pages: dict[str, FetchedPage] = field(default_factory=dict)
    error_for: set[str] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)
    closed: bool = False

    def set_page(self, url: str, html: str = "<html></html>", status: int = 200) -> None:
        self.pages[url] = FetchedPage(
            url=url,
            status=status,
            html=html,
            fetched_at=datetime.now(UTC).replace(tzinfo=None),
        )

    def fail(self, url: str) -> None:
        self.error_for.add(url)

    def fetch(self, url: str) -> FetchedPage:
        self.calls.append(url)
        if url in self.error_for:
            raise FetchError(f"fake fetcher configured to fail for {url}")
        if url in self.pages:
            return self.pages[url]
        # Default: synthesize a minimal HTML page with a 200 status.
        return FetchedPage(
            url=url,
            status=200,
            html="<html><body></body></html>",
            fetched_at=datetime.now(UTC).replace(tzinfo=None),
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_fetcher() -> FakeFetcher:
    return FakeFetcher()


# ---------- Parser fixtures --------------------------------------------------


@dataclass
class FakeParser:
    """Records every parse() call and returns a canned :class:`ParseResult`."""

    source: str = "cars_com"
    result_factory: object = None
    calls: list[dict[str, object]] = field(default_factory=list)

    def parse(
        self,
        *,
        html: str,
        url: str,
        kind: str,
        hints: dict[str, str | int | None],
    ) -> ParseResult:
        self.calls.append({"html": html, "url": url, "kind": kind, "hints": dict(hints)})
        if self.result_factory is None:
            return ParseResult()
        if callable(self.result_factory):
            res = self.result_factory(url=url, kind=kind, hints=hints)
            if not isinstance(res, ParseResult):  # pragma: no cover - defensive
                raise TypeError(
                    f"FakeParser.result_factory must return ParseResult, got {type(res).__name__}"
                )
            return res
        if isinstance(self.result_factory, ParseResult):
            return self.result_factory
        raise TypeError(  # pragma: no cover - defensive
            "FakeParser.result_factory must be None, callable, or ParseResult"
        )


@pytest.fixture
def fake_parser() -> FakeParser:
    return FakeParser()


# ---------- Queue fixtures ---------------------------------------------------


@pytest.fixture
def populated_queue(db: sqlite3.Connection) -> sqlite3.Connection:
    """Enqueue a handful of URLs against a single source ('cars_com')."""
    for i in range(3):
        queue.enqueue(
            db,
            url=f"https://cars.com/listing/{i}",
            source="cars_com",
            kind="listing",
            target_year=2020,
            target_make="Honda",
            target_model="Civic",
        )
    return db
