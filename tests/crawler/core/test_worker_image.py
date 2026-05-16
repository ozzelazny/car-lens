"""Worker integration tests for ``kind='image'`` queue items.

The Worker routes image-kind items to an injected ImageDownloader. These
tests assert that routing happens, that the parser registry is bypassed,
and that a missing downloader fails cleanly.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from car_lense_engine.crawler.core.image_downloader import ImageDownloadError
from car_lense_engine.crawler.core.politeness import PolicyConfig
from car_lense_engine.crawler.core.registry import ParserRegistry
from car_lense_engine.crawler.core.worker import Worker
from car_lense_engine.db import Image, Listing, listings, queue

from .conftest import FakeFetcher, FakeParser


@dataclass
class FakeImageDownloader:
    """Records every download() call; returns canned Image or raises."""

    calls: list[dict[str, object]] = field(default_factory=list)
    return_image: Image | None = None
    raise_exc: Exception | None = None

    def download(
        self,
        conn: sqlite3.Connection,
        url: str,
        *,
        source: str,
        listing_id: str,
        position: int | None = None,
    ) -> Image | None:
        self.calls.append(
            {
                "url": url,
                "source": source,
                "listing_id": listing_id,
                "position": position,
            }
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.return_image

    def close(self) -> None:
        pass


def _make_worker(
    db: sqlite3.Connection,
    fetcher: FakeFetcher,
    *,
    image_downloader: FakeImageDownloader | None,
) -> tuple[Worker, list[float]]:
    sleeps: list[float] = []
    policy = PolicyConfig(min_delay_seconds=0.0, max_delay_seconds=0.0)
    registry = ParserRegistry()
    registry.register(FakeParser(source="cars_com"))
    worker = Worker(
        conn=db,
        fetcher=fetcher,
        registry=registry,
        policy=policy,
        image_downloader=image_downloader,  # type: ignore[arg-type]
        rng=random.Random(0),
        sleep_fn=sleeps.append,
    )
    return worker, sleeps


@pytest.fixture
def listing_in_db(db: sqlite3.Connection) -> str:
    lid = "cars_com:99"
    listings.insert_listing(
        db,
        Listing(listing_id=lid, source="cars_com", url="https://cars.com/listing/99"),
    )
    return lid


def test_worker_dispatches_image_to_downloader(
    db: sqlite3.Connection,
    fake_fetcher: FakeFetcher,
    listing_in_db: str,
    tmp_path: Path,
) -> None:
    img_url = "https://cdn.example.com/x.jpg"
    queue.enqueue(
        db,
        url=img_url,
        source="cars_com",
        kind="image",
        parent_listing_id=listing_in_db,
    )

    fake_image = Image(
        image_id="a" * 64,
        listing_id=listing_in_db,
        source_url=img_url,
        local_path=str(tmp_path / "x.jpg"),
        phash="abc",
        width=10,
        height=10,
        bytes=123,
    )
    downloader = FakeImageDownloader(return_image=fake_image)
    worker, sleeps = _make_worker(db, fake_fetcher, image_downloader=downloader)

    assert worker.run_one() is True

    # Downloader was called with the right plumbing.
    assert len(downloader.calls) == 1
    assert downloader.calls[0]["url"] == img_url
    assert downloader.calls[0]["source"] == "cars_com"
    assert downloader.calls[0]["listing_id"] == listing_in_db

    # The page fetcher must NOT have been touched — image items bypass it.
    assert fake_fetcher.calls == []

    # Queue item marked done.
    row = db.execute("SELECT status FROM crawl_queue WHERE url = ?", (img_url,)).fetchone()
    assert row["status"] == "done"
    assert worker.stats.requests_succeeded == 1
    assert worker.stats.requests_failed == 0
    assert len(sleeps) == 1


def test_worker_marks_failed_when_no_downloader_configured(
    db: sqlite3.Connection,
    fake_fetcher: FakeFetcher,
    listing_in_db: str,
) -> None:
    img_url = "https://cdn.example.com/y.jpg"
    queue.enqueue(
        db,
        url=img_url,
        source="cars_com",
        kind="image",
        parent_listing_id=listing_in_db,
    )

    worker, sleeps = _make_worker(db, fake_fetcher, image_downloader=None)
    assert worker.run_one() is True

    row = db.execute(
        "SELECT status, last_error FROM crawl_queue WHERE url = ?", (img_url,)
    ).fetchone()
    assert row["status"] == "failed"
    assert "no image downloader configured" in row["last_error"]
    assert fake_fetcher.calls == []
    assert worker.stats.requests_failed == 1
    assert len(sleeps) == 1


def test_worker_marks_failed_when_download_raises(
    db: sqlite3.Connection,
    fake_fetcher: FakeFetcher,
    listing_in_db: str,
) -> None:
    img_url = "https://cdn.example.com/boom.jpg"
    queue.enqueue(
        db,
        url=img_url,
        source="cars_com",
        kind="image",
        parent_listing_id=listing_in_db,
    )

    downloader = FakeImageDownloader(raise_exc=ImageDownloadError("HTTP 503"))
    worker, _ = _make_worker(db, fake_fetcher, image_downloader=downloader)

    assert worker.run_one() is True

    row = db.execute(
        "SELECT status, last_error FROM crawl_queue WHERE url = ?", (img_url,)
    ).fetchone()
    assert row["status"] == "failed"
    assert "image download failed" in row["last_error"]
    assert "HTTP 503" in row["last_error"]
    assert worker.stats.requests_failed == 1


def test_worker_marks_failed_when_parent_listing_id_missing(
    db: sqlite3.Connection,
    fake_fetcher: FakeFetcher,
) -> None:
    """An image queue item with no parent_listing_id can't be routed to a path."""
    img_url = "https://cdn.example.com/orphan.jpg"
    queue.enqueue(db, url=img_url, source="cars_com", kind="image")

    downloader = FakeImageDownloader()
    worker, _ = _make_worker(db, fake_fetcher, image_downloader=downloader)
    assert worker.run_one() is True

    row = db.execute(
        "SELECT status, last_error FROM crawl_queue WHERE url = ?", (img_url,)
    ).fetchone()
    assert row["status"] == "failed"
    assert "parent_listing_id" in row["last_error"]
    # Downloader was NEVER invoked.
    assert downloader.calls == []


def test_worker_returns_idempotent_none_as_success(
    db: sqlite3.Connection,
    fake_fetcher: FakeFetcher,
    listing_in_db: str,
) -> None:
    """Downloader returning None (already-have-bytes) is still a queue success."""
    img_url = "https://cdn.example.com/dup.jpg"
    queue.enqueue(
        db,
        url=img_url,
        source="cars_com",
        kind="image",
        parent_listing_id=listing_in_db,
    )

    downloader = FakeImageDownloader(return_image=None)
    worker, _ = _make_worker(db, fake_fetcher, image_downloader=downloader)

    assert worker.run_one() is True

    row = db.execute("SELECT status FROM crawl_queue WHERE url = ?", (img_url,)).fetchone()
    assert row["status"] == "done"
    assert worker.stats.requests_succeeded == 1
