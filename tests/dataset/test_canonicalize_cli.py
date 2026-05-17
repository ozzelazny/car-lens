"""End-to-end tests for the ``canonicalize-labels`` CLI.

Seeds a tmp SQLite DB with rows from each source pattern (crawled Title
Case, Stanford lowercase, CompCars typo / alias / all-caps), runs the
CLI, and verifies the canonical_make / canonical_model columns are
populated correctly. Also covers ``--source``, ``--limit``, and
``--rebuild``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from car_lense_engine.dataset import canonicalize_cli
from car_lense_engine.db import Listing, listings, open_db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "crawl.sqlite"


@pytest.fixture
def db(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = open_db(db_path)
    try:
        yield conn
    finally:
        conn.close()


def _insert(
    conn: sqlite3.Connection,
    *,
    listing_id: str,
    source: str,
    make: str | None,
    model: str | None,
    year: int | None = 2020,
) -> None:
    listings.insert_listing(
        conn,
        Listing(
            listing_id=listing_id,
            source=source,  # type: ignore[arg-type]
            url=f"x://{listing_id}",
            year=year,
            make=make,
            model=model,
            split="train",
        ),
    )


def _canonical(conn: sqlite3.Connection, listing_id: str) -> tuple[str | None, str | None]:
    row = conn.execute(
        "SELECT canonical_make, canonical_model FROM listings WHERE listing_id = ?",
        (listing_id,),
    ).fetchone()
    return (row["canonical_make"], row["canonical_model"])


# --------------------------------------------------------------- happy path


def test_cli_populates_canonical_across_sources(db_path: Path) -> None:
    """One row per source pattern; the CLI canonicalizes every row."""
    conn = open_db(db_path)
    try:
        _insert(conn, listing_id="crawled-1", source="cars_com", make="Chevrolet", model="Tahoe")
        _insert(
            conn, listing_id="stanford-1", source="stanford_cars", make="chevrolet", model="tahoe"
        )
        _insert(conn, listing_id="compcars-1", source="compcars", make="Chevy", model="Tahoe")
        _insert(conn, listing_id="compcars-2", source="compcars", make="BWM", model="3 Series")
        _insert(conn, listing_id="compcars-3", source="compcars", make="MAZDA", model="cx-5")
        _insert(conn, listing_id="compcars-4", source="compcars", make="Benz", model="A-Class")
    finally:
        conn.close()

    rc = canonicalize_cli.main(["--db", str(db_path)])
    assert rc == 0

    conn = open_db(db_path)
    try:
        assert _canonical(conn, "crawled-1") == ("Chevrolet", "Tahoe")
        assert _canonical(conn, "stanford-1") == ("Chevrolet", "Tahoe")
        assert _canonical(conn, "compcars-1") == ("Chevrolet", "Tahoe")
        assert _canonical(conn, "compcars-2") == ("BMW", "3 Series")
        assert _canonical(conn, "compcars-3") == ("Mazda", "Cx-5")
        assert _canonical(conn, "compcars-4") == ("Mercedes-Benz", "A-Class")
    finally:
        conn.close()


def test_cli_is_idempotent_without_rebuild(db_path: Path) -> None:
    """Re-running without --rebuild skips rows already populated."""
    conn = open_db(db_path)
    try:
        _insert(conn, listing_id="a-1", source="cars_com", make="Acura", model="rl")
    finally:
        conn.close()

    assert canonicalize_cli.main(["--db", str(db_path)]) == 0

    # Manually corrupt the canonical_make to verify the second run
    # does NOT overwrite it (default behaviour: skip non-NULL canonical).
    conn = open_db(db_path)
    try:
        conn.execute(
            "UPDATE listings SET canonical_make = ? WHERE listing_id = ?",
            ("CORRUPTED", "a-1"),
        )
        conn.commit()
    finally:
        conn.close()

    assert canonicalize_cli.main(["--db", str(db_path)]) == 0
    conn = open_db(db_path)
    try:
        cm, _ = _canonical(conn, "a-1")
        assert cm == "CORRUPTED"
    finally:
        conn.close()


def test_cli_rebuild_overwrites_existing(db_path: Path) -> None:
    """--rebuild re-runs the normalizer against every row."""
    conn = open_db(db_path)
    try:
        _insert(conn, listing_id="a-1", source="cars_com", make="Acura", model="rl")
    finally:
        conn.close()

    assert canonicalize_cli.main(["--db", str(db_path)]) == 0

    # Corrupt the canonical_make and verify --rebuild fixes it.
    conn = open_db(db_path)
    try:
        conn.execute(
            "UPDATE listings SET canonical_make = ? WHERE listing_id = ?",
            ("CORRUPTED", "a-1"),
        )
        conn.commit()
    finally:
        conn.close()

    assert canonicalize_cli.main(["--db", str(db_path), "--rebuild"]) == 0
    conn = open_db(db_path)
    try:
        cm, cmodel = _canonical(conn, "a-1")
        assert cm == "Acura"
        assert cmodel == "Rl"
    finally:
        conn.close()


def test_cli_source_filter(db_path: Path) -> None:
    """--source restricts the pass to rows with that source value."""
    conn = open_db(db_path)
    try:
        _insert(conn, listing_id="crawled-1", source="cars_com", make="Chevrolet", model="Tahoe")
        _insert(conn, listing_id="compcars-1", source="compcars", make="Chevy", model="Tahoe")
    finally:
        conn.close()

    assert canonicalize_cli.main(["--db", str(db_path), "--source", "compcars"]) == 0

    conn = open_db(db_path)
    try:
        # compcars row canonicalized.
        assert _canonical(conn, "compcars-1") == ("Chevrolet", "Tahoe")
        # cars_com row untouched (still NULL).
        assert _canonical(conn, "crawled-1") == (None, None)
    finally:
        conn.close()


def test_cli_limit_caps_rows_processed(db_path: Path) -> None:
    """--limit N processes at most N candidate rows in listing_id order."""
    conn = open_db(db_path)
    try:
        for i in range(5):
            _insert(
                conn,
                listing_id=f"row-{i:02d}",
                source="cars_com",
                make="Acura",
                model="rl",
            )
    finally:
        conn.close()

    assert canonicalize_cli.main(["--db", str(db_path), "--limit", "3"]) == 0

    conn = open_db(db_path)
    try:
        n_populated = conn.execute(
            "SELECT COUNT(*) AS n FROM listings WHERE canonical_make IS NOT NULL"
        ).fetchone()["n"]
        assert n_populated == 3
    finally:
        conn.close()


def test_cli_missing_db_rejects(tmp_path: Path) -> None:
    """Pointing --db at a non-existent file errors out cleanly."""
    missing = tmp_path / "nope.sqlite"
    with pytest.raises(SystemExit) as excinfo:
        canonicalize_cli.main(["--db", str(missing)])
    assert excinfo.value.code == 2


def test_cli_invalid_limit_rejects(db_path: Path) -> None:
    """--limit must be > 0."""
    open_db(db_path).close()
    with pytest.raises(SystemExit) as excinfo:
        canonicalize_cli.main(["--db", str(db_path), "--limit", "0"])
    assert excinfo.value.code == 2


def test_cli_handles_null_make_gracefully(db_path: Path) -> None:
    """Rows with NULL make produce NULL canonical_make (no crash)."""
    conn = open_db(db_path)
    try:
        _insert(conn, listing_id="null-1", source="cars_com", make=None, model=None)
    finally:
        conn.close()

    assert canonicalize_cli.main(["--db", str(db_path)]) == 0
    conn = open_db(db_path)
    try:
        # Because canonical_make stays NULL the row remains a candidate
        # for the next pass, but the UPDATE still ran -- verify the row
        # exists and the canonical fields are NULL.
        cm, cmodel = _canonical(conn, "null-1")
        assert cm is None
        assert cmodel is None
    finally:
        conn.close()


def test_cli_prints_final_summary(db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI prints a one-line summary with the final counts."""
    conn = open_db(db_path)
    try:
        _insert(conn, listing_id="a", source="cars_com", make="Chevy", model="Tahoe")
        _insert(conn, listing_id="b", source="cars_com", make="bmw", model="m3")
    finally:
        conn.close()

    assert canonicalize_cli.main(["--db", str(db_path)]) == 0
    out = capsys.readouterr().out
    assert "canonicalize-labels:" in out
    assert "total_rows=2" in out
    assert "updated=2" in out
    assert "distinct_canonical_makes=" in out


def test_migration_8_adds_canonical_columns_and_index(db: sqlite3.Connection) -> None:
    """Migration 8 adds canonical_make + canonical_model + the partial index."""
    cur = db.execute("PRAGMA table_info(listings)")
    cols = {str(row["name"]) for row in cur.fetchall()}
    assert "canonical_make" in cols
    assert "canonical_model" in cols
    cur = db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_listings_canonical_class'"
    )
    assert cur.fetchone() is not None
