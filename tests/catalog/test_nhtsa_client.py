"""Tests for the :mod:`car_lense_engine.catalog.nhtsa_client` module."""

from __future__ import annotations

import time

import httpx
import pytest
import respx

from car_lense_engine.catalog.cache import JSONFileCache
from car_lense_engine.catalog.nhtsa_client import (
    NHTSAClient,
    NHTSAError,
    TokenBucket,
)


def _makes_payload() -> dict[str, object]:
    return {
        "Count": 2,
        "Message": "Response returned successfully",
        "Results": [
            {"MakeId": 474, "MakeName": "HONDA"},
            {"MakeId": 448, "MakeName": "TOYOTA"},
        ],
    }


def _models_payload(make_id: int, make_name: str) -> dict[str, object]:
    return {
        "Count": 2,
        "Message": "Response returned successfully",
        "Results": [
            {
                "Model_ID": 1861,
                "Model_Name": "CIVIC",
                "Make_ID": make_id,
                "Make_Name": make_name,
            },
            {
                "Model_ID": 1862,
                "Model_Name": "ACCORD",
                "Make_ID": make_id,
                "Make_Name": make_name,
            },
        ],
    }


def test_get_car_makes_normalizes_casing(
    respx_mock: respx.MockRouter,
    nhtsa_client: NHTSAClient,
    base_url: str,
) -> None:
    """Make names returned by NHTSA in ALL CAPS should be Title Cased."""
    respx_mock.get(
        f"{base_url}/GetMakesForVehicleType/car",
        params={"format": "json"},
    ).mock(return_value=httpx.Response(200, json=_makes_payload()))

    makes = nhtsa_client.get_car_makes()

    names = [m.make_name for m in makes]
    assert names == ["Honda", "Toyota"]
    # IDs preserved unchanged
    by_name = {m.make_name: m.make_id for m in makes}
    assert by_name["Honda"] == 474
    assert by_name["Toyota"] == 448


def test_get_models_for_make_year_handles_empty_results(
    respx_mock: respx.MockRouter,
    nhtsa_client: NHTSAClient,
    base_url: str,
) -> None:
    """An empty NHTSA response should produce an empty model list, not raise."""
    respx_mock.get(
        f"{base_url}/GetModelsForMakeIdYear/makeId/474/modelyear/1980",
        params={"format": "json"},
    ).mock(
        return_value=httpx.Response(
            200,
            json={"Count": 0, "Message": "Results not found", "Results": []},
        )
    )

    models = nhtsa_client.get_models_for_make_year(474, 1980)
    assert models == []


def test_get_models_normalizes_casing(
    respx_mock: respx.MockRouter,
    nhtsa_client: NHTSAClient,
    base_url: str,
) -> None:
    """Model names should also be Title Cased."""
    respx_mock.get(
        f"{base_url}/GetModelsForMakeIdYear/makeId/474/modelyear/2020",
        params={"format": "json"},
    ).mock(return_value=httpx.Response(200, json=_models_payload(474, "HONDA")))

    models = nhtsa_client.get_models_for_make_year(474, 2020)
    names = [m.model_name for m in models]
    assert "Civic" in names
    assert "Accord" in names
    assert all(m.make_name == "Honda" for m in models)


def test_retry_on_5xx(
    respx_mock: respx.MockRouter,
    nhtsa_client: NHTSAClient,
    base_url: str,
) -> None:
    """Two 500s followed by a 200 should succeed via retry."""
    route = respx_mock.get(
        f"{base_url}/GetMakesForVehicleType/car",
        params={"format": "json"},
    ).mock(
        side_effect=[
            httpx.Response(500, text="boom"),
            httpx.Response(503, text="boom again"),
            httpx.Response(200, json=_makes_payload()),
        ]
    )

    makes = nhtsa_client.get_car_makes()
    assert len(makes) == 2
    assert route.call_count == 3


def test_retry_exhausted_raises_nhtsa_error(
    respx_mock: respx.MockRouter,
    http_client: httpx.Client,
    cache: JSONFileCache,
    base_url: str,
) -> None:
    """After exhausting retries the client should raise :class:`NHTSAError`."""
    respx_mock.get(
        f"{base_url}/GetMakesForVehicleType/car",
        params={"format": "json"},
    ).mock(return_value=httpx.Response(500, text="permanent failure"))

    client = NHTSAClient(http_client, cache=cache, rate_per_sec=1000.0, max_attempts=2)
    with pytest.raises((NHTSAError, httpx.HTTPStatusError)):
        client.get_car_makes()


def test_rate_limit_enforced() -> None:
    """After the initial burst, the 5 req/s bucket paces acquires ~200ms apart."""
    bucket = TokenBucket(rate_per_sec=5.0)
    # Drain the bucket's initial burst capacity so we measure steady-state pacing.
    for _ in range(5):
        bucket.acquire()
    start = time.monotonic()
    for _ in range(4):
        bucket.acquire()
    elapsed = time.monotonic() - start
    # 4 paced tokens at 5/s = ~0.8s minimum (allow some slack on the low end).
    assert elapsed >= 0.6, f"expected >=0.6s, got {elapsed:.3f}s"
    # And we should not be wildly over what 5 req/s allows.
    assert elapsed < 2.0, f"unexpectedly slow: {elapsed:.3f}s"


def test_cache_short_circuits_http(
    respx_mock: respx.MockRouter,
    nhtsa_client: NHTSAClient,
    base_url: str,
) -> None:
    """A second call to the same endpoint should not hit the network."""
    route = respx_mock.get(
        f"{base_url}/GetMakesForVehicleType/car",
        params={"format": "json"},
    ).mock(return_value=httpx.Response(200, json=_makes_payload()))

    nhtsa_client.get_car_makes()
    nhtsa_client.get_car_makes()
    assert route.call_count == 1
