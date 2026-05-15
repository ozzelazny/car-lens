"""Thin httpx wrapper around the NHTSA vPIC API.

Two endpoints are exposed:

* :meth:`NHTSAClient.get_car_makes` —
  ``GetMakesForVehicleType/car``
* :meth:`NHTSAClient.get_models_for_make_year` —
  ``GetModelsForMakeIdYear/makeId/{make_id}/modelyear/{year}``

The client adds:

* a token-bucket rate limiter (default 5 req/sec)
* exponential-backoff retry on transient errors via :mod:`tenacity`
* an optional :class:`JSONFileCache` so repeat runs skip HTTP entirely
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .cache import JSONFileCache

logger = logging.getLogger(__name__)

BASE_URL = "https://vpic.nhtsa.dot.gov/api/vehicles"


def _title_case(name: str) -> str:
    """Convert an ``ALL CAPS`` NHTSA name to ``Title Case`` for our output."""
    return " ".join(part.capitalize() for part in name.strip().split())


@dataclass
class Make:
    """Lightweight in-memory record returned by :meth:`NHTSAClient.get_car_makes`."""

    make_id: int
    make_name: str


@dataclass
class ModelRecord:
    """Lightweight in-memory record returned by
    :meth:`NHTSAClient.get_models_for_make_year`."""

    model_id: int
    model_name: str
    make_id: int
    make_name: str


class TokenBucket:
    """Simple thread-safe token bucket for steady-rate request pacing."""

    def __init__(self, rate_per_sec: float) -> None:
        """Build a bucket that refills ``rate_per_sec`` tokens per second."""
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        self._rate = rate_per_sec
        self._capacity = max(1.0, rate_per_sec)
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available, then consume it."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = 1.0 - self._tokens
                wait = deficit / self._rate
            time.sleep(wait)


class NHTSAError(RuntimeError):
    """Raised when the NHTSA API cannot be reached after retries."""


def _is_retryable_status(exc: BaseException) -> bool:
    """Return ``True`` for the transient HTTP/transport errors we retry on."""
    if isinstance(exc, httpx.TimeoutException | httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    return False


class NHTSAClient:
    """Client for the NHTSA vPIC API with caching, retry, and rate limiting."""

    def __init__(
        self,
        client: httpx.Client,
        cache: JSONFileCache | None = None,
        rate_per_sec: float = 5.0,
        max_attempts: int = 5,
    ) -> None:
        """Wrap an :class:`httpx.Client` with retry, rate limiting, and optional cache."""
        self._client = client
        self._cache = cache
        self._bucket = TokenBucket(rate_per_sec)
        self._max_attempts = max_attempts

    # ----- public API ---------------------------------------------------

    def get_car_makes(self) -> list[Make]:
        """Return every passenger-car make known to NHTSA vPIC."""
        url = f"{BASE_URL}/GetMakesForVehicleType/car?format=json"
        payload = self._fetch(url)
        results = payload.get("Results", [])
        makes: list[Make] = []
        seen: set[int] = set()
        for row in results:
            make_id_raw = row.get("MakeId")
            make_name_raw = row.get("MakeName")
            if make_id_raw is None or not make_name_raw:
                continue
            make_id = int(make_id_raw)
            if make_id in seen:
                continue
            seen.add(make_id)
            makes.append(Make(make_id=make_id, make_name=_title_case(str(make_name_raw))))
        makes.sort(key=lambda m: m.make_name)
        return makes

    def get_models_for_make_year(self, make_id: int, year: int) -> list[ModelRecord]:
        """Return all models produced by ``make_id`` in ``year`` (may be empty)."""
        url = f"{BASE_URL}/GetModelsForMakeIdYear/makeId/{make_id}/modelyear/{year}?format=json"
        payload = self._fetch(url)
        results = payload.get("Results", [])
        models: list[ModelRecord] = []
        seen: set[int] = set()
        for row in results:
            model_id_raw = row.get("Model_ID")
            model_name_raw = row.get("Model_Name")
            make_id_raw = row.get("Make_ID")
            make_name_raw = row.get("Make_Name")
            if model_id_raw is None or not model_name_raw:
                continue
            model_id = int(model_id_raw)
            if model_id in seen:
                continue
            seen.add(model_id)
            models.append(
                ModelRecord(
                    model_id=model_id,
                    model_name=_title_case(str(model_name_raw)),
                    make_id=int(make_id_raw) if make_id_raw is not None else make_id,
                    make_name=(_title_case(str(make_name_raw)) if make_name_raw else ""),
                )
            )
        return models

    # ----- internals ----------------------------------------------------

    def _fetch(self, url: str) -> dict[str, Any]:
        """Fetch ``url`` honoring cache, rate limit, and retry policies."""
        if self._cache is not None:
            cached = self._cache.get(url)
            if cached is not None:
                logger.debug("cache hit: %s", url)
                return cached
        try:
            payload = self._fetch_with_retry(url)
        except RetryError as exc:
            raise NHTSAError(f"NHTSA request failed after retries: {url}") from exc
        if self._cache is not None:
            self._cache.set(url, payload)
        return payload

    def _fetch_with_retry(self, url: str) -> dict[str, Any]:
        """Wrap :meth:`_fetch_once` in tenacity-driven exponential backoff."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=16),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError)
            ),
        )
        def _do() -> dict[str, Any]:
            try:
                return self._fetch_once(url)
            except httpx.HTTPStatusError as exc:
                if _is_retryable_status(exc):
                    logger.warning("retrying %s after status %s", url, exc.response.status_code)
                    raise
                raise NHTSAError(
                    f"non-retryable status {exc.response.status_code} for {url}"
                ) from exc

        return _do()

    def _fetch_once(self, url: str) -> dict[str, Any]:
        """Perform a single rate-limited GET; raise for non-2xx responses."""
        self._bucket.acquire()
        logger.debug("GET %s", url)
        response = self._client.get(url)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise NHTSAError(f"unexpected non-object JSON from {url}")
        return data
