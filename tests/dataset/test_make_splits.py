"""Tests for the per-image stratified train/val/test split (Phase 3.5).

Each test primes an in-memory SQLite DB via :func:`open_db` so the full
migration stack (including 010, which adds ``images.split``) is applied,
then exercises :func:`make_splits` directly.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from car_lense_engine.dataset.make_splits import make_splits
from car_lense_engine.db import Image, Listing, images, listings, open_db

_TEST_SOURCE = "compcars"
_TEST_VIEW = "front"
_TEST_MAKE = "Toyota"
_TEST_MODEL = "Camry"
_TEST_GENERATION = 2012


# --------------------------------------------------------- fixtures


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = open_db(tmp_path / "crawl.sqlite")
    try:
        yield conn
    finally:
        conn.close()


# --------------------------------------------------------- seed helpers


def _insert_listing(
    conn: sqlite3.Connection,
    *,
    listing_id: str,
    canonical_make: str | None = _TEST_MAKE,
    canonical_model: str | None = _TEST_MODEL,
    generation_year: int | None = _TEST_GENERATION,
    source: str = _TEST_SOURCE,
) -> None:
    listings.insert_listing(
        conn,
        Listing(
            listing_id=listing_id,
            source=source,  # type: ignore[arg-type]
            url=f"x://{listing_id}",
            canonical_make=canonical_make,
            canonical_model=canonical_model,
            generation_year=generation_year,
        ),
    )


def _insert_image(
    conn: sqlite3.Connection,
    *,
    image_id: str,
    listing_id: str,
    view: str | None = _TEST_VIEW,
    view_score: float | None = 0.9,
) -> None:
    images.insert_image(
        conn,
        Image(
            image_id=image_id,
            listing_id=listing_id,
            source_url=f"x://{listing_id}/{image_id}",
            local_path=f"/tmp/{image_id}.jpg",
            position=1,
        ),
    )
    if view is not None:
        conn.execute(
            "UPDATE images SET view = ?, view_score = ? WHERE image_id = ?",
            (view, view_score, image_id),
        )
        conn.commit()


def _seed_balanced_pool(
    conn: sqlite3.Connection,
    *,
    n_listings: int,
    images_per_listing: int = 1,
    view: str = _TEST_VIEW,
    listing_prefix: str = "L",
) -> list[str]:
    """Seed ``n_listings`` listings, each with ``images_per_listing`` images.

    Returns the list of image_ids inserted.
    """
    image_ids: list[str] = []
    for i in range(n_listings):
        listing_id = f"{listing_prefix}{i:04d}"
        _insert_listing(conn, listing_id=listing_id)
        for j in range(images_per_listing):
            image_id = f"img_{listing_prefix}{i:04d}_{j:02d}"
            _insert_image(conn, image_id=image_id, listing_id=listing_id, view=view)
            image_ids.append(image_id)
    return image_ids


def _get_split(conn: sqlite3.Connection, image_id: str) -> str | None:
    row = conn.execute("SELECT split FROM images WHERE image_id = ?", (image_id,)).fetchone()
    return None if row is None else row["split"]


def _split_counts(conn: sqlite3.Connection) -> dict[str | None, int]:
    counts: dict[str | None, int] = {}
    for row in conn.execute("SELECT split, COUNT(*) AS n FROM images GROUP BY split").fetchall():
        counts[row["split"]] = int(row["n"])
    return counts


# --------------------------------------------------------- tests


def test_basic_80_10_10_split(db: sqlite3.Connection) -> None:
    """100 eligible images across 100 listings -> 80/10/10 ± 1."""
    _seed_balanced_pool(db, n_listings=100, images_per_listing=1)

    summary = make_splits(db, source=_TEST_SOURCE)

    assert summary.total_eligible == 100
    assert summary.total_train == 80
    assert summary.total_val == 10
    assert summary.total_test == 10
    assert summary.total_skipped_small_group == 0
    assert summary.total_excluded_non_exterior == 0

    counts = _split_counts(db)
    # 100 images all eligible, none should remain NULL after a clean run.
    assert counts.get(None, 0) == 0
    assert counts.get("train", 0) == 80
    assert counts.get("val", 0) == 10
    assert counts.get("test", 0) == 10


def test_non_exterior_rows_left_null(db: sqlite3.Connection) -> None:
    """Interior / detail / non-car / NULL view rows are NOT assigned a split."""
    # 20 eligible exterior images (enough to clear min_group_size).
    _seed_balanced_pool(db, n_listings=20, images_per_listing=1, view="front")

    # Plus a mix of non-exterior rows.
    for i, view in enumerate(["interior", "detail", "non-car", None]):
        listing_id = f"nonex_{i}"
        _insert_listing(db, listing_id=listing_id)
        _insert_image(
            db,
            image_id=f"img_nonex_{i}",
            listing_id=listing_id,
            view=view,
        )

    summary = make_splits(db, source=_TEST_SOURCE)

    # Only the 20 exterior images are eligible.
    assert summary.total_eligible == 20
    assert summary.total_excluded_non_exterior == 4

    # Each non-exterior row still carries NULL.
    for i in range(4):
        assert _get_split(db, f"img_nonex_{i}") is None
    # And each exterior row has SOMETHING assigned.
    counts = _split_counts(db)
    # 4 non-exterior NULLs + 20 eligible assigned.
    assert counts.get(None, 0) == 4
    assert counts.get("train", 0) + counts.get("val", 0) + counts.get("test", 0) == 20


def test_small_group_all_train(db: sqlite3.Connection) -> None:
    """Cells with < min_group_size images go entirely into train."""
    # 5 images, default min_group_size=10 -> all train.
    image_ids = _seed_balanced_pool(db, n_listings=5, images_per_listing=1)

    summary = make_splits(db, source=_TEST_SOURCE)

    assert summary.total_eligible == 5
    assert summary.total_train == 5
    assert summary.total_val == 0
    assert summary.total_test == 0
    assert summary.total_skipped_small_group == 5

    for image_id in image_ids:
        assert _get_split(db, image_id) == "train"


def test_deterministic_under_seed(db: sqlite3.Connection, tmp_path: Path) -> None:
    """Same seed -> identical assignment; different seed -> different (likely)."""
    _seed_balanced_pool(db, n_listings=100, images_per_listing=1)

    make_splits(db, source=_TEST_SOURCE, seed=42, rebuild=True)
    snapshot_a = {
        row["image_id"]: row["split"]
        for row in db.execute("SELECT image_id, split FROM images").fetchall()
    }

    make_splits(db, source=_TEST_SOURCE, seed=42, rebuild=True)
    snapshot_b = {
        row["image_id"]: row["split"]
        for row in db.execute("SELECT image_id, split FROM images").fetchall()
    }
    assert snapshot_a == snapshot_b

    make_splits(db, source=_TEST_SOURCE, seed=123, rebuild=True)
    snapshot_c = {
        row["image_id"]: row["split"]
        for row in db.execute("SELECT image_id, split FROM images").fetchall()
    }
    # With 100 distinct listings + a fresh shuffle, the val/test selection
    # should differ between seeds with overwhelming probability.
    assert snapshot_a != snapshot_c


def test_listing_coherent_split(db: sqlite3.Connection) -> None:
    """Listing-coherent mode keeps every photo of one listing in the same split.

    image-level mode is allowed to mix them across splits — we don't *force*
    a mixed assignment (a particular seed could happen to coalesce by
    chance), but we do force the coherent case to satisfy the invariant.
    """
    # 12 listings, 4 images each (48 images total) -> well above min_group_size.
    # That gives the coherent splitter enough head-room that the train/val/test
    # quotas are filled by complete listings rather than via cross-listing splits.
    for i in range(12):
        listing_id = f"L{i:02d}"
        _insert_listing(db, listing_id=listing_id)
        for j in range(4):
            _insert_image(
                db,
                image_id=f"img_{i:02d}_{j}",
                listing_id=listing_id,
                view=_TEST_VIEW,
            )

    # ---- coherent mode (default): per-listing coherence invariant ----
    make_splits(db, source=_TEST_SOURCE, listing_coherent=True, rebuild=True)
    by_listing: dict[str, set[str | None]] = {}
    for row in db.execute(
        "SELECT listing_id, split FROM images WHERE listing_id LIKE 'L%'"
    ).fetchall():
        by_listing.setdefault(str(row["listing_id"]), set()).add(row["split"])
    for listing_id, splits in by_listing.items():
        assert len(splits) == 1, f"listing {listing_id} crosses splits in coherent mode: {splits}"

    # ---- image-level mode: invariant is allowed to break ----
    # Reset the splits, then re-run image-level. We use enough listings (12)
    # with 4 images each that some listing almost certainly straddles two
    # splits. If not, the test still passes provided the split sizes match.
    db.execute("UPDATE images SET split = NULL")
    db.commit()
    make_splits(db, source=_TEST_SOURCE, listing_coherent=False, rebuild=True)
    counts = _split_counts(db)
    assert counts.get("train", 0) + counts.get("val", 0) + counts.get("test", 0) == 48


def test_idempotent_without_rebuild(db: sqlite3.Connection) -> None:
    """Re-running without --rebuild leaves existing splits unchanged."""
    _seed_balanced_pool(db, n_listings=100, images_per_listing=1)

    make_splits(db, source=_TEST_SOURCE, seed=42)
    snapshot_a = {
        row["image_id"]: row["split"]
        for row in db.execute("SELECT image_id, split FROM images").fetchall()
    }

    # Re-run with a different seed but no --rebuild: the assignment should
    # NOT change because every row already has a non-NULL split.
    make_splits(db, source=_TEST_SOURCE, seed=999)
    snapshot_b = {
        row["image_id"]: row["split"]
        for row in db.execute("SELECT image_id, split FROM images").fetchall()
    }

    assert snapshot_a == snapshot_b


def test_rebuild_overwrites(db: sqlite3.Connection) -> None:
    """--rebuild recomputes every split, overwriting any prior assignment."""
    _seed_balanced_pool(db, n_listings=100, images_per_listing=1)

    # Manually set everyone to 'test' so we can detect overwrite.
    db.execute("UPDATE images SET split = 'test'")
    db.commit()

    make_splits(db, source=_TEST_SOURCE, seed=42, rebuild=True)
    counts = _split_counts(db)
    # After rebuild we expect roughly 80/10/10 — not the all-'test' snapshot.
    assert counts.get("train", 0) == 80
    assert counts.get("val", 0) == 10
    assert counts.get("test", 0) == 10


def test_missing_canonical_excluded(db: sqlite3.Connection) -> None:
    """Listings with NULL canonical_make are excluded from the eligible pool."""
    # 20 listings with canonical labels.
    _seed_balanced_pool(db, n_listings=20, images_per_listing=1)
    # 3 listings missing canonical_make: ineligible regardless of view.
    for i in range(3):
        listing_id = f"nocls_{i}"
        _insert_listing(db, listing_id=listing_id, canonical_make=None, canonical_model=None)
        _insert_image(
            db,
            image_id=f"img_nocls_{i}",
            listing_id=listing_id,
            view=_TEST_VIEW,
        )

    summary = make_splits(db, source=_TEST_SOURCE)

    assert summary.total_eligible == 20
    # The 3 missing-canonical rows count as "excluded non-exterior" in the
    # ineligibility bucket (the function treats canonical-missing the same
    # as non-exterior).
    assert summary.total_excluded_non_exterior == 3

    for i in range(3):
        assert _get_split(db, f"img_nocls_{i}") is None
