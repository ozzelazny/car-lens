"""Tests for :class:`car_lense_engine.catalog.cache.JSONFileCache`."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx

from car_lense_engine.catalog.cache import JSONFileCache, _sanitize
from car_lense_engine.catalog.nhtsa_client import NHTSAClient


def _stub_makes() -> dict[str, object]:
    return {
        "Count": 1,
        "Message": "ok",
        "Results": [{"MakeId": 1, "MakeName": "ACME"}],
    }


def test_cache_hit_skips_http(
    tmp_cache_dir: Path,
    http_client: httpx.Client,
    respx_mock: respx.MockRouter,
    base_url: str,
) -> None:
    """A pre-seeded cache file should suppress the HTTP call entirely."""
    cache = JSONFileCache(tmp_cache_dir)
    url = f"{base_url}/GetMakesForVehicleType/car?format=json"
    cache.set(url, _stub_makes())

    route = respx_mock.get(
        f"{base_url}/GetMakesForVehicleType/car",
        params={"format": "json"},
    ).mock(return_value=httpx.Response(500, text="should never be called"))

    client = NHTSAClient(http_client, cache=cache, rate_per_sec=1000.0)
    makes = client.get_car_makes()
    assert makes[0].make_name == "Acme"
    assert route.call_count == 0


def test_cache_miss_writes_and_returns(
    tmp_cache_dir: Path,
    http_client: httpx.Client,
    respx_mock: respx.MockRouter,
    base_url: str,
) -> None:
    """A miss should hit the network, then the second call should be a hit."""
    cache = JSONFileCache(tmp_cache_dir)
    route = respx_mock.get(
        f"{base_url}/GetMakesForVehicleType/car",
        params={"format": "json"},
    ).mock(return_value=httpx.Response(200, json=_stub_makes()))

    client = NHTSAClient(http_client, cache=cache, rate_per_sec=1000.0)
    client.get_car_makes()
    files = list(tmp_cache_dir.glob("*.json"))
    assert len(files) == 1
    with files[0].open() as fh:
        on_disk = json.load(fh)
    assert on_disk["Results"][0]["MakeName"] == "ACME"

    # second call: still 1 network hit
    client.get_car_makes()
    assert route.call_count == 1


def test_rebuild_invalidates_cache(tmp_cache_dir: Path) -> None:
    """:meth:`JSONFileCache.clear` should wipe every cached file."""
    cache = JSONFileCache(tmp_cache_dir)
    cache.set("https://example.com/a", {"x": 1})
    cache.set("https://example.com/b", {"y": 2})
    assert len(list(tmp_cache_dir.glob("*.json"))) == 2

    cache.clear()
    assert list(tmp_cache_dir.glob("*.json")) == []
    assert tmp_cache_dir.exists()

    # The cache is still usable after a clear.
    cache.set("https://example.com/c", {"z": 3})
    assert cache.get("https://example.com/c") == {"z": 3}


def test_sanitize_handles_special_chars() -> None:
    """The sanitizer must reject URL characters that aren't filesystem-safe."""
    name = _sanitize("https://vpic.nhtsa.dot.gov/api/vehicles/Foo?bar=1&baz=2")
    assert "/" not in name
    assert "?" not in name
    assert "&" not in name
    # Bytes should be reversible enough to remain unique-per-URL.
    assert len(name) > 0
