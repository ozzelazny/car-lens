"""Database module — SQLite schema and access layer.

Public API:

* :func:`open_db` — open (or create) the crawler SQLite DB with migrations applied.
* Pydantic models: :class:`Listing`, :class:`Image`, :class:`QueueItem`, :class:`QueueStats`.
* Per-table accessor sub-modules: :mod:`listings`, :mod:`images`, :mod:`queue`.
"""

from __future__ import annotations

from . import images, listings, queue
from .connection import open_db
from .models import (
    Image,
    Listing,
    QueueItem,
    QueueKind,
    QueueStats,
    QueueStatus,
    Source,
)

__all__ = [
    "Image",
    "Listing",
    "QueueItem",
    "QueueKind",
    "QueueStats",
    "QueueStatus",
    "Source",
    "images",
    "listings",
    "open_db",
    "queue",
]
