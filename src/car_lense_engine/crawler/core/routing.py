"""Per-source fetcher routing.

The crawler treats the :class:`Fetcher` Protocol as opaque: it calls
``fetch(url)`` and receives a :class:`FetchedPage`. Different sites need
different transports (Playwright for JS-heavy SPAs, ``curl_cffi`` for
Cloudflare-fingerprinted requests). This module provides:

* :func:`source_for_url` — map a URL to one of the known source identifiers.
* :class:`MultiFetcher` — wrap N inner fetchers and dispatch based on source.

URL parsing happens inside the router so the :class:`Fetcher` Protocol stays
unchanged (still just ``fetch(url)``).
"""

from __future__ import annotations

import logging
from types import TracebackType
from urllib.parse import urlparse

from .fetcher import FetchedPage, Fetcher

logger = logging.getLogger(__name__)


HOSTNAME_TO_SOURCE: dict[str, str] = {
    # cars.com
    "cars.com": "cars_com",
    "www.cars.com": "cars_com",
    # autotrader
    "autotrader.com": "autotrader",
    "www.autotrader.com": "autotrader",
    # bat (bring-a-trailer)
    "bringatrailer.com": "bat",
    "www.bringatrailer.com": "bat",
    # hemmings
    "hemmings.com": "hemmings",
    "www.hemmings.com": "hemmings",
    # cars and bids
    "carsandbids.com": "carsandbids",
    "www.carsandbids.com": "carsandbids",
}
"""Direct hostname → source-identifier lookup for well-known sites."""

CRAIGSLIST_HOST_SUFFIX: str = ".craigslist.org"
"""Craigslist uses per-city subdomains (e.g. ``newyork.craigslist.org``);
match by suffix instead of enumerating every city."""


def source_for_url(url: str) -> str | None:
    """Return the source identifier for ``url``, or None if unknown.

    Parses ``url`` with :func:`urllib.parse.urlparse`. Hostname matching is
    case-insensitive. Craigslist city subdomains (``<city>.craigslist.org``)
    map to ``"craigslist"`` via the suffix check.

    Returns ``None`` for malformed URLs (no exception is raised).
    """
    try:
        parsed = urlparse(url)
    except (ValueError, AttributeError):  # pragma: no cover - urlparse is permissive
        return None
    host = (parsed.hostname or "").lower()
    if not host:
        return None
    if host in HOSTNAME_TO_SOURCE:
        return HOSTNAME_TO_SOURCE[host]
    if host.endswith(CRAIGSLIST_HOST_SUFFIX) or host == CRAIGSLIST_HOST_SUFFIX.lstrip("."):
        return "craigslist"
    return None


def known_sources() -> set[str]:
    """Return the set of source identifiers this router recognises."""
    return set(HOSTNAME_TO_SOURCE.values()) | {"craigslist"}


class MultiFetcher:
    """Route :meth:`fetch` calls to one of N inner fetchers based on URL source.

    Construct with a per-source mapping and a default fallback fetcher::

        MultiFetcher(
            per_source={"cars_com": curl_fetcher, "hemmings": curl_fetcher},
            default=playwright_fetcher,
        )

    ``fetch(url)`` looks up the source via :func:`source_for_url`. If a
    per-source fetcher is registered, it handles the request; otherwise the
    default is used. ``close()`` closes every distinct underlying fetcher
    (deduped by ``id()``) exactly once.
    """

    def __init__(
        self,
        *,
        per_source: dict[str, Fetcher],
        default: Fetcher,
    ) -> None:
        self._per_source: dict[str, Fetcher] = dict(per_source)
        self._default: Fetcher = default
        self._closed = False
        logger.info(
            "MultiFetcher configured: per_source=%s default=%s",
            {s: type(f).__name__ for s, f in self._per_source.items()},
            type(self._default).__name__,
        )

    # ------------------------------------------------------------ public API

    def fetch(self, url: str) -> FetchedPage:
        """Dispatch ``url`` to the correct inner fetcher and return its result."""
        source = source_for_url(url)
        fetcher: Fetcher
        if source is not None and source in self._per_source:
            fetcher = self._per_source[source]
            logger.debug(
                "MultiFetcher dispatch: url=%s source=%s -> %s",
                url,
                source,
                type(fetcher).__name__,
            )
        else:
            fetcher = self._default
            logger.debug(
                "MultiFetcher dispatch: url=%s source=%s -> default(%s)",
                url,
                source,
                type(fetcher).__name__,
            )
        return fetcher.fetch(url)

    def close(self) -> None:
        """Close every distinct underlying fetcher once. Idempotent."""
        if self._closed:
            return
        self._closed = True
        seen: set[int] = set()
        fetchers: list[Fetcher] = [*self._per_source.values(), self._default]
        for fetcher in fetchers:
            if id(fetcher) in seen:
                continue
            seen.add(id(fetcher))
            try:
                fetcher.close()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("MultiFetcher inner close raised: %r", exc)

    def __enter__(self) -> MultiFetcher:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
