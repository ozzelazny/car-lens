"""Pydantic models mirroring the SQLite schema."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

Source = Literal[
    "cars_com",
    "autotrader",
    "craigslist",
    "bat",
    "hemmings",
    "carsandbids",
    "stanford_cars",
]

QueueKind = Literal["search", "listing", "image"]

QueueStatus = Literal["pending", "in_progress", "done", "failed", "dead"]


class _Base(BaseModel):
    """Base class — allow constructing from sqlite3.Row mappings."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class Listing(_Base):
    """One vehicle listing scraped from a source site."""

    listing_id: str
    source: Source
    url: str
    year: int | None = None
    make: str | None = None
    model: str | None = None
    trim: str | None = None
    body_style: str | None = None
    mileage: int | None = None
    vin: str | None = None
    raw_html_sha256: str | None = None
    scraped_at: datetime | None = None


class Image(_Base):
    """One downloaded image file linked to a listing."""

    image_id: str
    listing_id: str
    source_url: str
    local_path: str
    phash: str | None = None
    width: int | None = None
    height: int | None = None
    bytes: int | None = None
    position: int | None = None
    downloaded_at: datetime | None = None
    view: str | None = None
    view_score: float | None = None
    view_labeled_at: datetime | None = None


class QueueItem(_Base):
    """One URL in the durable crawl queue."""

    url: str
    source: str
    kind: QueueKind
    target_year: int | None = None
    target_make: str | None = None
    target_model: str | None = None
    parent_listing_id: str | None = None
    status: QueueStatus = "pending"
    attempts: int = 0
    last_error: str | None = None
    next_try_at: datetime | None = None
    enqueued_at: datetime | None = None
    claimed_at: datetime | None = None


class QueueStats(_Base):
    """Aggregate counts by status, returned by ``queue.stats``."""

    pending: int = 0
    in_progress: int = 0
    done: int = 0
    failed: int = 0
    dead: int = 0
