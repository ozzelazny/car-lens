"""Tests for the CompCars ingest module.

Each test builds a small in-memory ZIP archive containing:

* ``misc/make_model_name.mat`` and ``misc/car_type.mat`` (written via
  ``scipy.io.savemat``),
* a handful of ``image/<make_id>/<model_id>/<year>/<sha>.jpg`` entries
  carrying tiny PIL-encoded JPEG bytes.

The ingest is then invoked against the synthetic archive; no network or HF
access happens.
"""

from __future__ import annotations

import io
import sqlite3
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pytest

scipy_io = pytest.importorskip("scipy.io")
np = pytest.importorskip("numpy")
PIL_Image = pytest.importorskip("PIL.Image")

from car_lense_engine.dataset.compcars import (  # noqa: E402
    ImportStats,
    import_compcars,
)
from car_lense_engine.db import images, listings, open_db  # noqa: E402

# --------------------------------------------------------- helpers


def _jpeg_bytes(color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> bytes:
    """Build a tiny solid-color JPEG byte blob."""
    img = PIL_Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_name_mat_bytes(makes: list[str], models: list[str]) -> bytes:
    """Build a make_model_name.mat-shaped byte blob."""
    buf = io.BytesIO()
    make_arr = np.array(makes, dtype=object).reshape(-1, 1)
    model_arr = np.array(models, dtype=object).reshape(-1, 1)
    scipy_io.savemat(buf, {"make_names": make_arr, "model_names": model_arr})
    return buf.getvalue()


def _make_body_mat_bytes(car_types: list[str], model_to_type: list[int]) -> bytes:
    """Build a car_type.mat-shaped byte blob."""
    buf = io.BytesIO()
    car_arr = np.array(car_types, dtype=object).reshape(1, -1)
    mt_arr = np.array(model_to_type, dtype=np.int32).reshape(-1, 1)
    scipy_io.savemat(buf, {"car_type": car_arr, "model_type": mt_arr})
    return buf.getvalue()


def _build_zip(
    path: Path,
    *,
    entries: list[tuple[str, bytes]],
    name_mat: bytes,
    body_mat: bytes,
    prefix: str = "",
) -> None:
    """Write a synthetic CompCars-shaped ZIP to ``path``.

    ``entries`` is a list of ``(in_zip_path, body_bytes)`` tuples. The
    ``misc/`` .mat files are written under the same optional ``prefix`` so
    we can exercise the leading-folder-stripping path on demand.
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(f"{prefix}misc/make_model_name.mat", name_mat)
        zf.writestr(f"{prefix}misc/car_type.mat", body_mat)
        for name, body in entries:
            zf.writestr(f"{prefix}{name}", body)


# --------------------------------------------------------- fixtures


@pytest.fixture
def db(tmp_path: Path) -> Iterable[sqlite3.Connection]:
    conn = open_db(tmp_path / "crawl.sqlite")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "compcars_out"


@pytest.fixture
def name_mat() -> bytes:
    return _make_name_mat_bytes(
        makes=["Audi", "BMW", "Mercedes-Benz"],
        models=["A4", "3 Series", "C-Class"],
    )


@pytest.fixture
def body_mat() -> bytes:
    return _make_body_mat_bytes(
        car_types=["MPV", "SUV", "sedan", "hatchback"],
        model_to_type=[3, 2, 4],  # A4->sedan, 3 Series->SUV, C-Class->hatchback
    )


# --------------------------------------------------------- happy-path


def test_ingest_writes_images_and_inserts_rows(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
    name_mat: bytes,
    body_mat: bytes,
) -> None:
    zip_path = tmp_path / "compcars.zip"
    _build_zip(
        zip_path,
        entries=[
            ("image/1/1/2012/aaa.jpg", _jpeg_bytes((255, 0, 0))),
            ("image/2/2/2013/bbb.jpg", _jpeg_bytes((0, 255, 0))),
            ("image/3/3/2014/ccc.jpg", _jpeg_bytes((0, 0, 255))),
        ],
        name_mat=name_mat,
        body_mat=body_mat,
    )

    stats = import_compcars(
        conn=db,
        out_dir=out_dir,
        zip_path=zip_path,
    )

    assert stats == ImportStats(
        processed=3,
        inserted_listings=3,
        inserted_images=3,
        skipped_existing=0,
        skipped_parse_failures=0,
        skipped_no_year=0,
        skipped_unknown_class=0,
    )

    written = list(out_dir.rglob("*.jpg"))
    assert len(written) == 3
    for p in written:
        assert len(p.stem) == 64  # sha256 hex

    rl = listings.list_by_class(db, source="compcars")
    assert len(rl) == 3
    by_make = {row.make: row for row in rl}
    assert by_make["Audi"].year == 2012
    assert by_make["Audi"].model == "A4"
    assert by_make["Audi"].body_style == "sedan"
    assert by_make["BMW"].year == 2013
    assert by_make["BMW"].model == "3 Series"
    assert by_make["BMW"].body_style == "SUV"
    assert by_make["Mercedes-Benz"].year == 2014
    assert by_make["Mercedes-Benz"].model == "C-Class"
    assert by_make["Mercedes-Benz"].body_style == "hatchback"

    for row in rl:
        assert row.split == "train"
        img_rows = images.list_for_listing(db, row.listing_id)
        assert len(img_rows) == 1
        assert Path(img_rows[0].local_path).exists()
        assert img_rows[0].position == 1
        assert img_rows[0].phash is None


def test_ingest_records_source_compcars(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
    name_mat: bytes,
    body_mat: bytes,
) -> None:
    zip_path = tmp_path / "compcars.zip"
    _build_zip(
        zip_path,
        entries=[("image/1/1/2012/aaa.jpg", _jpeg_bytes((255, 0, 0)))],
        name_mat=name_mat,
        body_mat=body_mat,
    )
    import_compcars(conn=db, out_dir=out_dir, zip_path=zip_path)
    cur = db.execute("SELECT source FROM listings")
    assert {row["source"] for row in cur.fetchall()} == {"compcars"}


def test_ingest_synthetic_url_and_listing_id(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
    name_mat: bytes,
    body_mat: bytes,
) -> None:
    zip_path = tmp_path / "compcars.zip"
    _build_zip(
        zip_path,
        entries=[("image/1/1/2012/aaa.jpg", _jpeg_bytes((255, 0, 0)))],
        name_mat=name_mat,
        body_mat=body_mat,
    )
    import_compcars(conn=db, out_dir=out_dir, zip_path=zip_path)
    rl = listings.list_by_class(db, source="compcars")
    assert len(rl) == 1
    only = rl[0]
    assert only.listing_id.startswith("compcars:")
    assert only.url.startswith("compcars://1/1/2012/")


# --------------------------------------------------------- idempotency / skips


def test_ingest_is_idempotent(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
    name_mat: bytes,
    body_mat: bytes,
) -> None:
    zip_path = tmp_path / "compcars.zip"
    _build_zip(
        zip_path,
        entries=[
            ("image/1/1/2012/aaa.jpg", _jpeg_bytes((255, 0, 0))),
            ("image/2/2/2013/bbb.jpg", _jpeg_bytes((0, 255, 0))),
        ],
        name_mat=name_mat,
        body_mat=body_mat,
    )

    first = import_compcars(conn=db, out_dir=out_dir, zip_path=zip_path)
    assert first.inserted_images == 2

    second = import_compcars(conn=db, out_dir=out_dir, zip_path=zip_path)
    assert second.processed == 2
    assert second.inserted_listings == 0
    assert second.inserted_images == 0
    assert second.skipped_existing == 2


def test_ingest_skips_nan_year(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
    name_mat: bytes,
    body_mat: bytes,
) -> None:
    zip_path = tmp_path / "compcars.zip"
    _build_zip(
        zip_path,
        entries=[
            ("image/1/1/2012/aaa.jpg", _jpeg_bytes((255, 0, 0))),
            ("image/2/2/nan/bbb.jpg", _jpeg_bytes((0, 255, 0))),
            ("image/3/3/5008/ccc.jpg", _jpeg_bytes((0, 0, 255))),
        ],
        name_mat=name_mat,
        body_mat=body_mat,
    )

    stats = import_compcars(conn=db, out_dir=out_dir, zip_path=zip_path)
    assert stats.processed == 3
    assert stats.inserted_images == 1
    assert stats.skipped_no_year == 2
    assert stats.skipped_parse_failures == 0


def test_ingest_skips_unknown_class(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
    name_mat: bytes,
    body_mat: bytes,
) -> None:
    zip_path = tmp_path / "compcars.zip"
    _build_zip(
        zip_path,
        entries=[
            ("image/1/1/2012/aaa.jpg", _jpeg_bytes((255, 0, 0))),
            # make_id=99 isn't in the 3-entry name table -> unknown_class.
            ("image/99/1/2013/bbb.jpg", _jpeg_bytes((0, 255, 0))),
        ],
        name_mat=name_mat,
        body_mat=body_mat,
    )

    stats = import_compcars(conn=db, out_dir=out_dir, zip_path=zip_path)
    assert stats.processed == 2
    assert stats.inserted_images == 1
    assert stats.skipped_unknown_class == 1


def test_ingest_respects_limit(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
    name_mat: bytes,
    body_mat: bytes,
) -> None:
    zip_path = tmp_path / "compcars.zip"
    _build_zip(
        zip_path,
        entries=[(f"image/1/1/2012/{i:02d}.jpg", _jpeg_bytes((i * 10, 0, 0))) for i in range(5)],
        name_mat=name_mat,
        body_mat=body_mat,
    )

    stats = import_compcars(
        conn=db,
        out_dir=out_dir,
        zip_path=zip_path,
        limit=2,
    )
    assert stats.processed == 2
    assert stats.inserted_images == 2


# --------------------------------------------------------- archive shape


def test_ingest_handles_top_level_prefix(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
    name_mat: bytes,
    body_mat: bytes,
) -> None:
    """Some packagings wrap the dataset under a top-level folder; we tolerate that."""
    zip_path = tmp_path / "compcars.zip"
    _build_zip(
        zip_path,
        entries=[("image/1/1/2012/aaa.jpg", _jpeg_bytes((255, 0, 0)))],
        name_mat=name_mat,
        body_mat=body_mat,
        prefix="Compcars_Data/",
    )

    stats = import_compcars(conn=db, out_dir=out_dir, zip_path=zip_path)
    assert stats.processed == 1
    assert stats.inserted_images == 1


def test_ingest_records_provided_split(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
    name_mat: bytes,
    body_mat: bytes,
) -> None:
    zip_path = tmp_path / "compcars.zip"
    _build_zip(
        zip_path,
        entries=[("image/1/1/2012/aaa.jpg", _jpeg_bytes((255, 0, 0)))],
        name_mat=name_mat,
        body_mat=body_mat,
    )
    import_compcars(conn=db, out_dir=out_dir, zip_path=zip_path, split="val")
    rl = listings.list_by_class(db, source="compcars")
    assert len(rl) == 1
    assert rl[0].split == "val"


# --------------------------------------------------------- validation


def test_ingest_invalid_log_every_rejected(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        import_compcars(
            conn=db,
            out_dir=out_dir,
            zip_path=tmp_path / "does_not_matter.zip",
            log_every=0,
        )


def test_ingest_invalid_limit_rejected(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        import_compcars(
            conn=db,
            out_dir=out_dir,
            zip_path=tmp_path / "does_not_matter.zip",
            limit=0,
        )


def test_ingest_missing_zip_rejected(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        import_compcars(
            conn=db,
            out_dir=out_dir,
            zip_path=tmp_path / "missing.zip",
        )
