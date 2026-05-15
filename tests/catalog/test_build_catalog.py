"""End-to-end-ish tests for the catalog orchestrator and CLI plumbing."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx

from car_lense_engine.catalog.build_catalog import build_catalog, write_catalog
from car_lense_engine.catalog.nhtsa_client import BASE_URL, NHTSAClient


def _mock_makes(router: respx.MockRouter, makes: list[tuple[int, str]]) -> None:
    router.get(
        f"{BASE_URL}/GetMakesForVehicleType/car",
        params={"format": "json"},
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "Count": len(makes),
                "Message": "ok",
                "Results": [{"MakeId": mid, "MakeName": name} for mid, name in makes],
            },
        )
    )


def _mock_models(
    router: respx.MockRouter,
    make_id: int,
    make_name: str,
    year: int,
    models: list[tuple[int, str]],
) -> None:
    router.get(
        f"{BASE_URL}/GetModelsForMakeIdYear/makeId/{make_id}/modelyear/{year}",
        params={"format": "json"},
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "Count": len(models),
                "Message": "ok",
                "Results": [
                    {
                        "Model_ID": mid,
                        "Model_Name": name,
                        "Make_ID": make_id,
                        "Make_Name": make_name,
                    }
                    for mid, name in models
                ],
            },
        )
    )


def test_end_to_end_small(
    respx_mock: respx.MockRouter,
    nhtsa_client: NHTSAClient,
    tmp_path: Path,
) -> None:
    """Two makes x three years: verify shape, merged years, Title Case."""
    _mock_makes(respx_mock, [(1, "HONDA"), (2, "TOYOTA")])

    # Honda: civic in all three years, accord in only the last two.
    _mock_models(respx_mock, 1, "HONDA", 2020, [(100, "CIVIC")])
    _mock_models(respx_mock, 1, "HONDA", 2021, [(100, "CIVIC"), (101, "ACCORD")])
    _mock_models(respx_mock, 1, "HONDA", 2022, [(100, "CIVIC"), (101, "ACCORD")])

    # Toyota: corolla only in 2022.
    _mock_models(respx_mock, 2, "TOYOTA", 2020, [])
    _mock_models(respx_mock, 2, "TOYOTA", 2021, [])
    _mock_models(respx_mock, 2, "TOYOTA", 2022, [(200, "COROLLA")])

    catalog = build_catalog(nhtsa_client, year_range=(2020, 2022), progress=False)

    # Names Title Cased, makes sorted by name.
    assert [m.make_name for m in catalog.makes] == ["Honda", "Toyota"]
    honda = catalog.makes[0]
    # Models within a make sorted by name: Accord then Civic.
    assert [m.model_name for m in honda.models] == ["Accord", "Civic"]
    accord = honda.models[0]
    civic = honda.models[1]
    assert accord.years == [2021, 2022]
    assert civic.years == [2020, 2021, 2022]
    assert civic.model_id == 100

    toyota = catalog.makes[1]
    assert [m.model_name for m in toyota.models] == ["Corolla"]
    assert toyota.models[0].years == [2022]

    # Round-trip through JSON to confirm pydantic serialization.
    out = tmp_path / "classes.json"
    write_catalog(catalog, out)
    on_disk = json.loads(out.read_text())
    assert on_disk["makes"][0]["make_name"] == "Honda"
    assert on_disk["makes"][0]["models"][1]["years"] == [2020, 2021, 2022]
    assert on_disk["meta"]["source"] == "NHTSA vPIC"
    assert tuple(on_disk["meta"]["year_range"]) == (2020, 2022)


def test_meta_counts_correct(
    respx_mock: respx.MockRouter,
    nhtsa_client: NHTSAClient,
) -> None:
    """``meta.total_*`` counts should match the rolled-up output."""
    _mock_makes(respx_mock, [(1, "HONDA")])
    _mock_models(respx_mock, 1, "HONDA", 2020, [(100, "CIVIC")])
    _mock_models(respx_mock, 1, "HONDA", 2021, [(100, "CIVIC"), (101, "ACCORD")])

    catalog = build_catalog(nhtsa_client, year_range=(2020, 2021), progress=False)

    assert catalog.meta.total_makes == 1
    assert catalog.meta.total_models == 2  # Civic + Accord
    # Civic in 2 years + Accord in 1 year = 3 (make, model, year) entries
    assert catalog.meta.total_class_entries == 3


def test_make_with_no_models_in_any_year_omitted(
    respx_mock: respx.MockRouter,
    nhtsa_client: NHTSAClient,
) -> None:
    """A make that returned zero models every year shouldn't appear in output."""
    _mock_makes(respx_mock, [(1, "HONDA"), (2, "GHOSTMAKE")])
    _mock_models(respx_mock, 1, "HONDA", 2020, [(100, "CIVIC")])
    _mock_models(respx_mock, 1, "HONDA", 2021, [(100, "CIVIC")])
    _mock_models(respx_mock, 2, "GHOSTMAKE", 2020, [])
    _mock_models(respx_mock, 2, "GHOSTMAKE", 2021, [])

    catalog = build_catalog(nhtsa_client, year_range=(2020, 2021), progress=False)
    names = [m.make_name for m in catalog.makes]
    assert names == ["Honda"]
    assert catalog.meta.total_makes == 1


def test_max_makes_limits_iteration(
    respx_mock: respx.MockRouter,
    nhtsa_client: NHTSAClient,
) -> None:
    """``max_makes=1`` should slice to the first make (alphabetical)."""
    _mock_makes(respx_mock, [(1, "HONDA"), (2, "ACURA")])
    # Acura is alphabetically first after Title Case.
    _mock_models(respx_mock, 2, "ACURA", 2020, [(300, "TLX")])

    catalog = build_catalog(nhtsa_client, year_range=(2020, 2020), max_makes=1, progress=False)
    assert [m.make_name for m in catalog.makes] == ["Acura"]


def test_cli_writes_output(
    respx_mock: respx.MockRouter,
    tmp_path: Path,
) -> None:
    """The CLI entry point should write a valid JSON file to ``--output``."""
    from car_lense_engine.catalog.cli import main

    _mock_makes(respx_mock, [(1, "HONDA")])
    _mock_models(respx_mock, 1, "HONDA", 2020, [(100, "CIVIC")])

    output = tmp_path / "out.json"
    cache_dir = tmp_path / "cache"
    rc = main(
        [
            "--output",
            str(output),
            "--years",
            "2020:2020",
            "--cache-dir",
            str(cache_dir),
            "--max-makes",
            "5",
            "-v",
        ]
    )
    assert rc == 0
    data = json.loads(output.read_text())
    assert data["makes"][0]["make_name"] == "Honda"
    assert data["makes"][0]["models"][0]["model_name"] == "Civic"


def test_cli_rebuild_clears_cache(
    respx_mock: respx.MockRouter,
    tmp_path: Path,
) -> None:
    """``--rebuild`` should wipe stale cache entries before fetching."""
    from car_lense_engine.catalog.cache import JSONFileCache
    from car_lense_engine.catalog.cli import main

    cache_dir = tmp_path / "cache"
    cache = JSONFileCache(cache_dir)
    # Seed with a poisoned entry the CLI would otherwise pick up.
    cache.set(
        f"{BASE_URL}/GetMakesForVehicleType/car?format=json",
        {"Count": 0, "Message": "stale", "Results": []},
    )

    _mock_makes(respx_mock, [(1, "HONDA")])
    _mock_models(respx_mock, 1, "HONDA", 2020, [(100, "CIVIC")])

    out = tmp_path / "classes.json"
    rc = main(
        [
            "--output",
            str(out),
            "--years",
            "2020:2020",
            "--cache-dir",
            str(cache_dir),
            "--rebuild",
            "-v",
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    # If --rebuild didn't wipe, the makes list would be empty.
    assert [m["make_name"] for m in data["makes"]] == ["Honda"]
