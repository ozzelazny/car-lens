"""Shared fixtures for the crawler-seed test suite."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from car_lense_engine.catalog.schema import Catalog, Make, Meta, Model
from car_lense_engine.db import open_db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Return a fresh on-disk SQLite path for each test."""
    return tmp_path / "crawl.sqlite"


@pytest.fixture
def db(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a fresh DB with migrations applied; close it on teardown."""
    conn = open_db(db_path)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def tiny_catalog() -> Catalog:
    """Provide a small Catalog with 4 makes of varying year_max and popularity.

    - Honda Civic: very recent + very popular make (weight 1.0)
    - Toyota Camry: very recent + very popular make (weight 1.0)
    - Ferrari F40: vintage + unknown-to-table make (default 0.3)
    - GhostMake Phantom: recent + unknown-to-table make (default 0.3)
    """
    makes = [
        Make(
            make_id=1,
            make_name="Honda",
            models=[
                Model(model_id=11, model_name="Civic", years=[2020, 2021, 2022]),
                Model(model_id=12, model_name="Accord", years=[2019, 2020, 2021]),
            ],
        ),
        Make(
            make_id=2,
            make_name="Toyota",
            models=[
                Model(model_id=21, model_name="Camry", years=[2020, 2021, 2022]),
            ],
        ),
        Make(
            make_id=3,
            make_name="Ferrari",
            models=[
                Model(model_id=31, model_name="F40", years=[1987, 1988, 1989, 1990, 1991, 1992]),
            ],
        ),
        Make(
            make_id=4,
            make_name="GhostMake",
            models=[
                Model(model_id=41, model_name="Phantom", years=[2021, 2022]),
            ],
        ),
    ]
    return Catalog(
        meta=Meta(
            generated_at="2026-01-01T00:00:00Z",
            source="test-fixture",
            year_range=(1981, 2026),
            total_makes=len(makes),
            total_models=sum(len(m.models) for m in makes),
            total_class_entries=sum(len(m.years) for mk in makes for m in mk.models),
        ),
        makes=makes,
    )
