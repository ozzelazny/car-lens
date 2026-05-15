"""Tests for the seed orchestrator and CLI."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from car_lense_engine.catalog.schema import Catalog
from car_lense_engine.crawler.seed.cli import main as cli_main
from car_lense_engine.crawler.seed.ranker import RankedClass, rank_models
from car_lense_engine.crawler.seed.seed import build_urls_for, seed_queue


def _tiny_ranked() -> list[RankedClass]:
    """Two ranked classes for use in DB-write tests."""
    return [
        RankedClass(
            make="Honda",
            model="Civic",
            make_id=1,
            model_id=11,
            year_min=2020,
            year_max=2022,
            score=1.0,
        ),
        RankedClass(
            make="Toyota",
            model="Camry",
            make_id=2,
            model_id=21,
            year_min=2020,
            year_max=2022,
            score=0.9,
        ),
    ]


def test_seed_queue_inserts_rows(db: sqlite3.Connection) -> None:
    """2 ranked classes × 2 single-URL sites = 4 queue rows."""
    stats = seed_queue(db, _tiny_ranked(), sites=["cars_com", "autotrader"])
    assert stats.total_yielded == 4
    assert stats.inserted == 4
    assert stats.duplicates == 0
    assert stats.per_site == {"cars_com": 2, "autotrader": 2}

    cnt = db.execute("SELECT COUNT(*) AS n FROM crawl_queue").fetchone()["n"]
    assert int(cnt) == 4

    # Every inserted row is kind='search' and carries a target_make/model.
    rows = db.execute(
        "SELECT source, kind, target_make, target_model, target_year FROM crawl_queue"
    ).fetchall()
    for row in rows:
        assert row["kind"] == "search"
        assert row["source"] in {"cars_com", "autotrader"}
        assert row["target_make"] in {"Honda", "Toyota"}
        assert row["target_model"] in {"Civic", "Camry"}
        assert row["target_year"] == 2020


def test_seed_queue_duplicates_counted(db: sqlite3.Connection) -> None:
    """Calling the seeder twice with the same inputs marks the second pass as duplicates."""
    first = seed_queue(db, _tiny_ranked(), sites=["cars_com"])
    assert first.inserted == 2
    assert first.duplicates == 0

    second = seed_queue(db, _tiny_ranked(), sites=["cars_com"])
    assert second.total_yielded == 2
    assert second.inserted == 0
    assert second.duplicates == 2


def test_per_site_counts_with_craigslist(db: sqlite3.Connection) -> None:
    """Craigslist yields N URLs per class (one per city); per-site math should reflect that."""
    cities = ["newyork", "losangeles", "sfbay"]
    stats = seed_queue(
        db,
        _tiny_ranked(),
        sites=["cars_com", "craigslist"],
        cities=cities,
    )
    # 2 ranked × 1 cars_com URL = 2; 2 ranked × 3 craigslist URLs = 6; total = 8.
    assert stats.per_site == {"cars_com": 2, "craigslist": 6}
    assert stats.total_yielded == 8
    assert stats.inserted == 8


def test_build_urls_for_yields_seedurls_lazily(tiny_catalog: Catalog) -> None:
    """build_urls_for returns an iterator — exhaust it and check shape."""
    ranked = rank_models(tiny_catalog, top_n=10)
    seeds = list(build_urls_for(ranked, sites=["cars_com"]))
    # One cars_com URL per (make, model).
    expected = sum(len(mk.models) for mk in tiny_catalog.makes)
    assert len(seeds) == expected
    for s in seeds:
        assert s.source == "cars_com"
        assert s.url.startswith("https://www.cars.com/")
        assert s.target_make
        assert s.target_model


def test_build_urls_for_rejects_unknown_site() -> None:
    """Passing a site not in SITE_BUILDERS raises ValueError."""
    with pytest.raises(ValueError, match="unknown site identifier"):
        list(build_urls_for(_tiny_ranked(), sites=["bogus"]))


def test_dry_run_via_cli(
    tmp_path: Path,
    tiny_catalog: Catalog,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--dry-run prints '<source>\\t<url>' lines and never touches a DB."""
    catalog_path = tmp_path / "classes.json"
    catalog_path.write_text(tiny_catalog.model_dump_json(), encoding="utf-8")

    rc = cli_main(
        [
            "--catalog",
            str(catalog_path),
            "--top-n",
            "3",
            "--sites",
            "cars_com,autotrader",
            "--dry-run",
        ]
    )
    assert rc == 0

    captured = capsys.readouterr()
    lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    # 3 classes × 2 sites × 1 URL each = 6 lines
    assert len(lines) == 6
    for line in lines:
        parts = line.split("\t")
        assert len(parts) == 2, f"expected TSV, got: {line!r}"
        source, url = parts
        assert source in {"cars_com", "autotrader"}
        assert url.startswith("https://")
    # The yield-count summary lands on stderr.
    assert "dry-run: yielded 6 URLs" in captured.err


def test_cli_writes_to_db_when_no_dry_run(
    tmp_path: Path,
    tiny_catalog: Catalog,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without --dry-run the CLI opens the DB and enqueues rows."""
    catalog_path = tmp_path / "classes.json"
    catalog_path.write_text(tiny_catalog.model_dump_json(), encoding="utf-8")
    db_path = tmp_path / "crawl.sqlite"

    rc = cli_main(
        [
            "--catalog",
            str(catalog_path),
            "--db",
            str(db_path),
            "--top-n",
            "2",
            "--sites",
            "cars_com",
        ]
    )
    assert rc == 0

    # Verify rows landed in the DB.
    import sqlite3 as _sqlite3

    conn = _sqlite3.connect(db_path)
    try:
        n = conn.execute("SELECT COUNT(*) FROM crawl_queue").fetchone()[0]
    finally:
        conn.close()
    assert int(n) == 2

    captured = capsys.readouterr()
    assert "inserted=2" in captured.err


def test_cli_rejects_unknown_site(
    tmp_path: Path,
    tiny_catalog: Catalog,
) -> None:
    """An invalid --sites value should cause argparse to exit non-zero."""
    catalog_path = tmp_path / "classes.json"
    catalog_path.write_text(tiny_catalog.model_dump_json(), encoding="utf-8")

    with pytest.raises(SystemExit):
        cli_main(
            [
                "--catalog",
                str(catalog_path),
                "--sites",
                "not_a_site",
                "--dry-run",
            ]
        )
