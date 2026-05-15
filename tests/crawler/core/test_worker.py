"""Tests for the Worker — claim → fetch → parse → persist → mark_done|failed."""

from __future__ import annotations

import random
import sqlite3
from datetime import datetime

from car_lense_engine.crawler.core.politeness import PolicyConfig
from car_lense_engine.crawler.core.registry import ParserRegistry
from car_lense_engine.crawler.core.worker import Worker
from car_lense_engine.crawler.parsers.base import (
    DiscoveredUrl,
    ParsedListing,
    ParseResult,
)
from car_lense_engine.db import Listing, listings, queue

from .conftest import FakeFetcher, FakeParser


def _make_worker(
    db: sqlite3.Connection,
    *,
    fetcher: FakeFetcher,
    parser: FakeParser | None = None,
    policy: PolicyConfig | None = None,
    rng: random.Random | None = None,
    sleep_fn: object = None,
    clock: object = None,
) -> tuple[Worker, list[float]]:
    """Build a Worker with a fixed-seed RNG and a captured sleep list."""
    sleeps: list[float] = []
    if sleep_fn is None:
        sleep_fn = sleeps.append
    if policy is None:
        policy = PolicyConfig(min_delay_seconds=0.1, max_delay_seconds=0.2)
    registry = ParserRegistry()
    if parser is not None:
        registry.register(parser)
    worker = Worker(
        conn=db,
        fetcher=fetcher,
        registry=registry,
        policy=policy,
        rng=rng if rng is not None else random.Random(0),
        clock=clock if clock is not None else None,
        sleep_fn=sleep_fn,  # type: ignore[arg-type]
    )
    return worker, sleeps


def test_run_one_empty_queue_returns_false(
    db: sqlite3.Connection, fake_fetcher: FakeFetcher
) -> None:
    worker, sleeps = _make_worker(db, fetcher=fake_fetcher)
    assert worker.run_one() is False
    assert worker.stats.requests_total == 0
    assert sleeps == []


def test_run_one_success_path(db: sqlite3.Connection, fake_fetcher: FakeFetcher) -> None:
    url = "https://cars.com/listing/abc"
    queue.enqueue(
        db,
        url=url,
        source="cars_com",
        kind="listing",
        target_year=2021,
        target_make="Honda",
        target_model="Civic",
    )
    fake_fetcher.set_page(url, html="<html>listing</html>")

    parsed = ParsedListing(
        listing_id="cars_com:abc",
        source="cars_com",
        url=url,
        year=2021,
        make="Honda",
        model="Civic",
        image_urls=[
            "https://cars.com/img/abc/1.jpg",
            "https://cars.com/img/abc/2.jpg",
        ],
    )
    result = ParseResult(new_listing=parsed)
    parser = FakeParser(source="cars_com", result_factory=result)
    worker, sleeps = _make_worker(db, fetcher=fake_fetcher, parser=parser)

    assert worker.run_one() is True

    # Listing inserted.
    row = db.execute("SELECT listing_id, year, make, model FROM listings").fetchone()
    assert row is not None
    assert row["listing_id"] == "cars_com:abc"
    assert row["year"] == 2021
    assert row["make"] == "Honda"
    assert row["model"] == "Civic"

    # 2 image URLs enqueued.
    img_rows = db.execute(
        "SELECT url FROM crawl_queue WHERE kind = 'image' ORDER BY url"
    ).fetchall()
    assert [r["url"] for r in img_rows] == sorted(parsed.image_urls)

    # Item marked done.
    q = db.execute("SELECT status FROM crawl_queue WHERE url = ?", (url,)).fetchone()
    assert q["status"] == "done"

    # Stats and sleep.
    assert worker.stats.listings_inserted == 1
    assert worker.stats.urls_enqueued == 2
    assert worker.stats.requests_total == 1
    assert worker.stats.requests_succeeded == 1
    assert worker.stats.requests_failed == 0
    assert len(sleeps) == 1
    assert 0.1 <= sleeps[0] <= 0.2

    # Parser received the hints from the queue.
    assert parser.calls == [
        {
            "html": "<html>listing</html>",
            "url": url,
            "kind": "listing",
            "hints": {
                "target_year": 2021,
                "target_make": "Honda",
                "target_model": "Civic",
            },
        }
    ]


def test_run_one_fetch_error_marks_failed(
    db: sqlite3.Connection, fake_fetcher: FakeFetcher
) -> None:
    url = "https://cars.com/listing/dead"
    queue.enqueue(db, url=url, source="cars_com", kind="listing")
    fake_fetcher.fail(url)

    parser = FakeParser(source="cars_com")
    worker, sleeps = _make_worker(db, fetcher=fake_fetcher, parser=parser)

    assert worker.run_one() is True

    row = db.execute(
        "SELECT status, attempts, last_error FROM crawl_queue WHERE url = ?", (url,)
    ).fetchone()
    assert row["status"] == "failed"
    assert int(row["attempts"]) == 1
    assert row["last_error"] is not None and "fetch failed" in row["last_error"]

    assert worker.stats.requests_total == 1
    assert worker.stats.requests_failed == 1
    assert worker.stats.requests_succeeded == 0
    assert len(sleeps) == 1
    # Parser must NOT have been called since fetch failed.
    assert parser.calls == []


def test_run_one_parser_missing_marks_failed(
    db: sqlite3.Connection, fake_fetcher: FakeFetcher
) -> None:
    url = "https://cars.com/listing/no-parser"
    queue.enqueue(db, url=url, source="cars_com", kind="listing")

    worker, sleeps = _make_worker(db, fetcher=fake_fetcher, parser=None)
    assert worker.run_one() is True

    row = db.execute("SELECT status, last_error FROM crawl_queue WHERE url = ?", (url,)).fetchone()
    assert row["status"] == "failed"
    assert row["last_error"] is not None
    assert "no parser registered" in row["last_error"]
    assert "cars_com" in row["last_error"]

    # Fetcher must NOT have been called since registry lookup failed first.
    assert fake_fetcher.calls == []
    assert worker.stats.requests_failed == 1
    assert len(sleeps) == 1


def test_run_one_parser_returns_new_urls_only(
    db: sqlite3.Connection, fake_fetcher: FakeFetcher
) -> None:
    search_url = "https://cars.com/search?p=1"
    queue.enqueue(
        db,
        url=search_url,
        source="cars_com",
        kind="search",
        target_year=2020,
        target_make="Toyota",
        target_model="Camry",
    )
    fake_fetcher.set_page(search_url, html="<html>search</html>")

    discovered = [
        DiscoveredUrl(
            url=f"https://cars.com/listing/{i}",
            source="cars_com",
            kind="listing",
            target_year=2020,
            target_make="Toyota",
            target_model="Camry",
        )
        for i in range(5)
    ]
    parser = FakeParser(
        source="cars_com",
        result_factory=ParseResult(new_urls=discovered),
    )
    worker, _ = _make_worker(db, fetcher=fake_fetcher, parser=parser)

    assert worker.run_one() is True

    # No listings row inserted.
    n = db.execute("SELECT COUNT(*) AS n FROM listings").fetchone()
    assert int(n["n"]) == 0

    # 5 new listing URLs enqueued in addition to the original search row.
    listing_rows = db.execute(
        "SELECT url FROM crawl_queue WHERE kind = 'listing' ORDER BY url"
    ).fetchall()
    assert [r["url"] for r in listing_rows] == [du.url for du in discovered]

    assert worker.stats.listings_inserted == 0
    assert worker.stats.urls_enqueued == 5
    assert worker.stats.requests_succeeded == 1


def test_run_one_parser_exception_marks_failed(
    db: sqlite3.Connection, fake_fetcher: FakeFetcher
) -> None:
    url = "https://cars.com/listing/boom"
    queue.enqueue(db, url=url, source="cars_com", kind="listing")

    def _boom(**_kw: object) -> ParseResult:
        raise RuntimeError("parser exploded")

    parser = FakeParser(source="cars_com", result_factory=_boom)
    worker, sleeps = _make_worker(db, fetcher=fake_fetcher, parser=parser)

    assert worker.run_one() is True
    row = db.execute("SELECT status, last_error FROM crawl_queue WHERE url = ?", (url,)).fetchone()
    assert row["status"] == "failed"
    assert "parse failed" in row["last_error"]
    assert worker.stats.requests_failed == 1
    assert len(sleeps) == 1


def test_sleep_uses_jittered_delay(db: sqlite3.Connection, fake_fetcher: FakeFetcher) -> None:
    queue.enqueue(db, url="https://cars.com/x", source="cars_com", kind="listing")
    parser = FakeParser(source="cars_com")
    policy = PolicyConfig(min_delay_seconds=2.5, max_delay_seconds=4.5)
    worker, sleeps = _make_worker(
        db,
        fetcher=fake_fetcher,
        parser=parser,
        policy=policy,
        rng=random.Random(123),
    )
    worker.run_one()
    assert len(sleeps) == 1
    assert 2.5 <= sleeps[0] <= 4.5


def test_run_one_clock_injected(db: sqlite3.Connection, fake_fetcher: FakeFetcher) -> None:
    """A fixed clock can be injected; the worker stores it without touching wall-clock."""
    queue.enqueue(db, url="https://cars.com/c", source="cars_com", kind="listing")
    parser = FakeParser(source="cars_com")
    fixed = datetime(2026, 1, 1, 0, 0, 0)
    worker, _ = _make_worker(
        db,
        fetcher=fake_fetcher,
        parser=parser,
        clock=lambda: fixed,
    )
    assert worker.clock() == fixed
    worker.run_one()
    # Calling the clock still returns the fixed time after a run.
    assert worker.clock() == fixed


def test_run_one_duplicate_listing_treated_as_success(
    db: sqlite3.Connection, fake_fetcher: FakeFetcher
) -> None:
    """Re-inserting a listing with the same listing_id should not blow up the worker."""
    url1 = "https://cars.com/listing/dup1"
    url2 = "https://cars.com/listing/dup2"
    queue.enqueue(db, url=url1, source="cars_com", kind="listing")
    queue.enqueue(db, url=url2, source="cars_com", kind="listing")

    def _factory(*, url: str, **_kw: object) -> ParseResult:
        return ParseResult(
            new_listing=ParsedListing(
                listing_id="cars_com:same",
                source="cars_com",
                url=url,
            )
        )

    parser = FakeParser(source="cars_com", result_factory=_factory)
    worker, _ = _make_worker(db, fetcher=fake_fetcher, parser=parser)

    assert worker.run_one() is True
    assert worker.run_one() is True
    # Only one listing row, both queue items marked done.
    n = db.execute("SELECT COUNT(*) AS n FROM listings").fetchone()
    assert int(n["n"]) == 1
    done = db.execute("SELECT COUNT(*) AS n FROM crawl_queue WHERE status = 'done'").fetchone()
    assert int(done["n"]) == 2
    assert worker.stats.requests_succeeded == 2
    # listings_inserted only counts the first one.
    assert worker.stats.listings_inserted == 1


def test_run_one_images_enqueued_even_when_listing_already_exists(
    db: sqlite3.Connection, fake_fetcher: FakeFetcher
) -> None:
    """Regression: a retry that hits IntegrityError on the listings insert must
    still enqueue the parser's image URLs.

    Scenario: a prior worker run inserted the ``listings`` row for
    ``cars_com:200`` and then crashed before enqueuing the listing's images.
    On the next run, we claim the same listing URL again; the parser returns
    the same listing_id (so the insert raises IntegrityError) plus two
    image_urls. Those two URLs MUST land in the queue — anything else is
    silent data loss.
    """
    url = "https://cars.com/listing/200"
    # Pre-insert the listings row as if a prior run had written it.
    listings.insert_listing(
        db,
        Listing(
            listing_id="cars_com:200",
            source="cars_com",
            url=url,
        ),
    )
    # Enqueue the same listing URL — this is the retry.
    queue.enqueue(db, url=url, source="cars_com", kind="listing")

    image_urls = [
        "https://img1.example.com/a.jpg",
        "https://img2.example.com/b.jpg",
    ]
    parsed = ParsedListing(
        listing_id="cars_com:200",
        source="cars_com",
        url=url,
        image_urls=image_urls,
    )
    parser = FakeParser(source="cars_com", result_factory=ParseResult(new_listing=parsed))
    worker, _ = _make_worker(db, fetcher=fake_fetcher, parser=parser)

    assert worker.run_one() is True

    # The single pre-existing listings row is unchanged — no new insert happened.
    n_listings = db.execute("SELECT COUNT(*) AS n FROM listings").fetchone()
    assert int(n_listings["n"]) == 1
    assert worker.stats.listings_inserted == 0

    # Load-bearing assertion: BOTH image URLs are now in the queue.
    img_rows = db.execute(
        "SELECT url FROM crawl_queue WHERE kind = 'image' ORDER BY url"
    ).fetchall()
    assert [r["url"] for r in img_rows] == sorted(image_urls)
    assert worker.stats.urls_enqueued == 2

    # Listing queue item marked done (not failed).
    q = db.execute("SELECT status FROM crawl_queue WHERE url = ?", (url,)).fetchone()
    assert q["status"] == "done"
