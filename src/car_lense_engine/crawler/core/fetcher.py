"""Fetcher protocol — abstracts page fetching so the worker can be tested without a browser."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class FetchedPage(BaseModel):
    """The successful result of a :meth:`Fetcher.fetch` call."""

    model_config = ConfigDict(extra="forbid")

    url: str
    """Final URL after redirects."""

    status: int
    """HTTP status returned by the final response."""

    html: str
    """Rendered HTML of the page."""

    fetched_at: datetime
    """Naive UTC timestamp captured at fetch completion, matching the DB layer's convention."""


class FetchError(Exception):
    """Raised by a :class:`Fetcher` when a URL cannot be retrieved successfully.

    The worker treats any :class:`FetchError` as a transient failure and calls
    :func:`queue.mark_failed` so the URL is retried later with backoff.
    """


@runtime_checkable
class Fetcher(Protocol):
    """Synchronous fetcher contract."""

    def fetch(self, url: str) -> FetchedPage:
        """Fetch ``url`` and return rendered HTML. Raise :class:`FetchError` on failure."""
        ...

    def close(self) -> None:
        """Release any underlying resources (browsers, sockets, ...)."""
        ...
