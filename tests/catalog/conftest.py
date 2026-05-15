"""Shared fixtures for the catalog test suite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx

from car_lense_engine.catalog.cache import JSONFileCache
from car_lense_engine.catalog.nhtsa_client import BASE_URL, NHTSAClient


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Provide a fresh cache directory for one test."""
    d = tmp_path / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def cache(tmp_cache_dir: Path) -> JSONFileCache:
    """Provide a :class:`JSONFileCache` bound to a clean dir."""
    return JSONFileCache(tmp_cache_dir)


@pytest.fixture
def respx_mock() -> Iterator[respx.MockRouter]:
    """Activate respx to intercept all HTTP traffic for one test."""
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def http_client() -> Iterator[httpx.Client]:
    """Yield a short-timeout :class:`httpx.Client`."""
    with httpx.Client(timeout=5) as client:
        yield client


@pytest.fixture
def nhtsa_client(
    http_client: httpx.Client,
    cache: JSONFileCache,
) -> NHTSAClient:
    """Provide a fast :class:`NHTSAClient` (very high rate cap to keep tests fast)."""
    return NHTSAClient(http_client, cache=cache, rate_per_sec=1000.0, max_attempts=4)


@pytest.fixture
def base_url() -> str:
    """Expose the API base URL for tests building mock routes."""
    return BASE_URL
