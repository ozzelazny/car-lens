"""Tests for the VMMRdb ingest module.

The HF ``datasets`` library is stubbed via ``monkeypatch`` so tests never
hit the network. Each fake row is a ``{"image": PIL.Image, "label": int}``
dict — matching the schema of the ``venetis/VMMRdb_make_model_*`` mirror.

The local-ZIP path is exercised against synthetic mini-ZIPs that mirror
the on-disk layout of the full VMMRdb GitHub release
(``<prefix>/<class_dir>/<image_id>.jpg``).
"""

from __future__ import annotations

import io
import json
import sqlite3
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest

PIL_Image = pytest.importorskip("PIL.Image")

from car_lense_engine.dataset.vmmrdb import (  # noqa: E402
    ImportStats,
    import_vmmrdb,
    import_vmmrdb_from_zip,
)
from car_lense_engine.db import images, listings, open_db  # noqa: E402

# --------------------------------------------------------- helpers


def _make_image(color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> Any:
    """Build a tiny solid-color PIL image."""
    return PIL_Image.new("RGB", size, color)


def _write_catalog(tmp_path: Path) -> Path:
    """Write a minimal classes.json — the VMMRdb ingest only uses it for existence."""
    catalog = {
        "meta": {
            "generated_at": "2026-01-01T00:00:00Z",
            "source": "test",
            "year_range": [1981, 2026],
            "total_makes": 0,
            "total_models": 0,
            "total_class_entries": 0,
        },
        "makes": [],
    }
    path = tmp_path / "classes.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return path


class _FakeClassLabel:
    """Stub for ``datasets.ClassLabel`` — exposes ``names`` and ``int2str``."""

    def __init__(self, names: list[str]) -> None:
        self.names = names

    def int2str(self, i: int) -> str:
        return self.names[i]


class _StubDataset:
    """Iterable matching the surface of ``datasets.load_dataset(streaming=True)``.

    Optionally exposes a ``features`` mapping so the production label
    decoder can resolve integer ClassLabel rows via ``features['label'].int2str``.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        features: dict[str, Any] | None = None,
    ) -> None:
        self._rows = rows
        self.features = features

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._rows)


def _install_stub(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, Any]],
    features: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    """Patch ``datasets.load_dataset`` to return ``_StubDataset(rows, features)``.

    Returns the list of (dataset_id, split) call-args for assertions.
    """
    calls: list[tuple[str, str]] = []

    def fake_load_dataset(
        dataset_id: str, *, split: str, streaming: bool, **_: Any
    ) -> _StubDataset:
        assert streaming is True, "ingest must use streaming=True"
        calls.append((dataset_id, split))
        return _StubDataset(rows, features=features)

    # Stub it inside the ingest module's lazy-import boundary.
    import sys
    import types

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    return calls


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
    return tmp_path / "vmmrdb"


@pytest.fixture
def catalog_path(tmp_path: Path) -> Path:
    return _write_catalog(tmp_path)


# --------------------------------------------------------- tests


def test_ingest_writes_images_and_inserts_rows(
    db: sqlite3.Connection,
    out_dir: Path,
    catalog_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "image": _make_image((255, 0, 0)),
            "label": 0,
            "class": "honda_civic_2005",
        },
        {
            "image": _make_image((0, 255, 0)),
            "label": 1,
            "class": "ford_f-150_2010",
        },
        {
            "image": _make_image((0, 0, 255)),
            "label": 2,
            "class": "chevrolet_silverado_1500_2012",
        },
    ]
    _install_stub(monkeypatch, rows)

    stats = import_vmmrdb(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
        hf_dataset="some/mirror",
    )

    assert stats == ImportStats(
        processed=3,
        inserted_listings=3,
        inserted_images=3,
        skipped_existing=0,
        skipped_parse_failures=0,
    )

    # All three image files exist on disk under a per-class subdir.
    written = list(out_dir.rglob("*.jpg"))
    assert len(written) == 3
    for p in written:
        # SHA-256 hex digest is 64 chars; the filename should be that.
        assert len(p.stem) == 64

    # Listings row sanity-check.
    rl = listings.list_by_class(db, source="vmmrdb")
    assert len(rl) == 3
    by_model = {row.model: row for row in rl}
    assert by_model["civic"].year == 2005
    assert by_model["civic"].make == "honda"
    assert by_model["f-150"].year == 2010
    assert by_model["f-150"].make == "ford"
    assert by_model["silverado_1500"].year == 2012
    assert by_model["silverado_1500"].make == "chevrolet"

    # Image row local_path points at the actual on-disk file. split was 'train'.
    for row in rl:
        assert row.split == "train"
        img_rows = images.list_for_listing(db, row.listing_id)
        assert len(img_rows) == 1
        assert Path(img_rows[0].local_path).exists()
        assert img_rows[0].position == 1
        assert img_rows[0].phash is None


def test_ingest_records_source_vmmrdb(
    db: sqlite3.Connection,
    out_dir: Path,
    catalog_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every inserted listing must have ``source='vmmrdb'``."""
    rows = [
        {"image": _make_image((255, 0, 0)), "label": 0, "class": "honda_civic_2005"},
    ]
    _install_stub(monkeypatch, rows)
    import_vmmrdb(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
        hf_dataset="some/mirror",
    )
    cur = db.execute("SELECT source FROM listings")
    sources = {row["source"] for row in cur.fetchall()}
    assert sources == {"vmmrdb"}


def test_ingest_is_idempotent(
    db: sqlite3.Connection,
    out_dir: Path,
    catalog_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"image": _make_image((255, 0, 0)), "label": 0, "class": "honda_civic_2005"},
        {"image": _make_image((0, 255, 0)), "label": 1, "class": "ford_f-150_2010"},
    ]
    _install_stub(monkeypatch, rows)

    first = import_vmmrdb(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
        hf_dataset="some/mirror",
    )
    assert first.inserted_images == 2

    second = import_vmmrdb(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
        hf_dataset="some/mirror",
    )
    # Same byte stream -> same SHA-256 -> rows already in DB -> all skipped.
    assert second.processed == 2
    assert second.inserted_listings == 0
    assert second.inserted_images == 0
    assert second.skipped_existing == 2


def test_ingest_skips_parse_failures(
    db: sqlite3.Connection,
    out_dir: Path,
    catalog_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"image": _make_image((255, 0, 0)), "label": 0, "class": "honda_civic_2005"},
        {
            # No underscore -> parse failure.
            "image": _make_image((128, 128, 128)),
            "label": 99,
            "class": "banana",
        },
        {"image": _make_image((0, 255, 0)), "label": 1, "class": "ford_f-150_2010"},
    ]
    _install_stub(monkeypatch, rows)

    stats = import_vmmrdb(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
        hf_dataset="some/mirror",
    )

    assert stats.processed == 3
    assert stats.inserted_listings == 2
    assert stats.inserted_images == 2
    assert stats.skipped_parse_failures == 1
    # The bad row should not have written a file or inserted any DB row.
    written = list(out_dir.rglob("*.jpg"))
    assert len(written) == 2


def test_ingest_respects_limit(
    db: sqlite3.Connection,
    out_dir: Path,
    catalog_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "image": _make_image((i * 10, 0, 0)),
            "label": i,
            "class": "honda_civic_2005",
        }
        for i in range(5)
    ]
    # All same class string but distinct pixel colors -> distinct SHA-256s.
    _install_stub(monkeypatch, rows)

    stats = import_vmmrdb(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
        hf_dataset="some/mirror",
        limit=2,
    )
    assert stats.processed == 2
    assert stats.inserted_images == 2


def test_ingest_synthetic_url_and_listing_id(
    db: sqlite3.Connection,
    out_dir: Path,
    catalog_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The synthetic URL/listing_id convention is part of the contract."""
    rows = [
        {"image": _make_image((255, 0, 0)), "label": 0, "class": "honda_civic_2005"},
    ]
    _install_stub(monkeypatch, rows)

    import_vmmrdb(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
        hf_dataset="some/mirror",
    )

    rl = listings.list_by_class(db, source="vmmrdb")
    assert len(rl) == 1
    only = rl[0]
    assert only.listing_id.startswith("vmmrdb:")
    assert only.url.startswith("vmmrdb://")
    # The URL embeds the class_id (year_make_model) and the image_id.
    image_id_part = only.listing_id.split(":", 1)[1]
    assert image_id_part in only.url


def test_ingest_invalid_log_every_rejected(
    db: sqlite3.Connection,
    out_dir: Path,
    catalog_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stub(monkeypatch, [])
    with pytest.raises(ValueError):
        import_vmmrdb(
            conn=db,
            out_dir=out_dir,
            catalog_path=catalog_path,
            hf_dataset="some/mirror",
            log_every=0,
        )


def test_ingest_invalid_limit_rejected(
    db: sqlite3.Connection,
    out_dir: Path,
    catalog_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stub(monkeypatch, [])
    with pytest.raises(ValueError):
        import_vmmrdb(
            conn=db,
            out_dir=out_dir,
            catalog_path=catalog_path,
            hf_dataset="some/mirror",
            limit=0,
        )


def test_ingest_missing_catalog_rejected(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stub(monkeypatch, [])
    missing = tmp_path / "no_such_catalog.json"
    with pytest.raises(FileNotFoundError):
        import_vmmrdb(
            conn=db,
            out_dir=out_dir,
            catalog_path=missing,
            hf_dataset="some/mirror",
        )


def test_class_id_slugify_handles_special_chars() -> None:
    from car_lense_engine.dataset.vmmrdb import _slugify

    assert _slugify("2005_honda_civic") == "2005_honda_civic"
    assert _slugify("mercedes benz_s550") == "mercedes_benz_s550"
    assert _slugify("ford_f-150_2010") == "ford_f-150_2010"
    assert _slugify("Foo/Bar Baz") == "Foo_Bar_Baz"


def test_ingest_resolves_int_classlabel_from_hf_features(
    db: sqlite3.Connection,
    out_dir: Path,
    catalog_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The venetis mirror emits an int ClassLabel — verify the decoder resolves it.

    Rows here carry ``label`` as an int (no pre-resolved ``class`` string),
    which matches ``venetis/VMMRdb_make_model_*``. The decoder must consult
    ``ds.features['label'].int2str`` to recover the class name. Note that the
    venetis mirror omits the year — labels are ``"<make>_<model>"`` only.
    """
    class_names = ["acura_cl", "mercedes benz_s550"]
    fake_features = {"label": _FakeClassLabel(class_names)}
    rows: list[dict[str, Any]] = [
        {"image": _make_image((255, 0, 0)), "label": 0},
        {"image": _make_image((0, 0, 255)), "label": 1},
    ]
    _install_stub(monkeypatch, rows, features=fake_features)

    stats = import_vmmrdb(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
        hf_dataset="venetis/VMMRdb_make_model_train",
    )

    assert stats.processed == 2
    assert stats.inserted_listings == 2
    assert stats.inserted_images == 2
    assert stats.skipped_parse_failures == 0

    rl = listings.list_by_class(db, source="vmmrdb")
    assert {row.make for row in rl} == {"acura", "mercedes benz"}
    by_make = {row.make: row for row in rl}
    assert by_make["acura"].model == "cl"
    assert by_make["acura"].year is None
    assert by_make["mercedes benz"].model == "s550"
    assert by_make["mercedes benz"].year is None


def test_ingest_raises_parse_failure_when_int_label_has_no_features(
    db: sqlite3.Connection,
    out_dir: Path,
    catalog_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Int-only ``label`` with no ``features`` decoder ⇒ parse failure (skipped)."""
    rows: list[dict[str, Any]] = [
        {"image": _make_image((255, 0, 0)), "label": 0},
    ]
    # No features kwarg => stub dataset's .features is None.
    _install_stub(monkeypatch, rows)

    stats = import_vmmrdb(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
        hf_dataset="some/mirror",
    )

    assert stats.processed == 1
    assert stats.inserted_listings == 0
    assert stats.inserted_images == 0
    assert stats.skipped_parse_failures == 1


def test_ingest_uses_load_dataset_with_streaming(
    db: sqlite3.Connection,
    out_dir: Path,
    catalog_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"image": _make_image((255, 0, 0)), "label": 0, "class": "honda_civic_2005"},
    ]
    calls = _install_stub(monkeypatch, rows)
    import_vmmrdb(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
        hf_dataset="some/mirror",
        split="test",
    )
    # Non-venetis mirror -> the semantic split is passed straight through.
    assert calls == [("some/mirror", "test")]


def test_ingest_venetis_mirror_forces_train_split(
    db: sqlite3.Connection,
    out_dir: Path,
    catalog_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """venetis mirrors expose a single 'train' HF split per dataset id.

    Even when the user passes --split val (to record val in listings.split),
    the HF stream must be opened with split='train'.
    """
    rows = [
        {"image": _make_image((255, 0, 0)), "label": 0, "class": "honda_civic_2005"},
    ]
    calls = _install_stub(monkeypatch, rows)
    import_vmmrdb(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
        hf_dataset="venetis/VMMRdb_make_model_val",
        split="val",
    )
    assert calls == [("venetis/VMMRdb_make_model_val", "train")]
    # And the row was tagged with the semantic split 'val'.
    rl = listings.list_by_class(db, source="vmmrdb")
    assert len(rl) == 1
    assert rl[0].split == "val"


# --------------------------------------------------------- full-release ZIP


def _jpeg_bytes(color: tuple[int, int, int], size: tuple[int, int] = (16, 16)) -> bytes:
    """Encode a tiny solid-color PIL image as JPEG bytes."""
    img = PIL_Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _build_vmmrdb_zip(
    path: Path,
    *,
    entries: list[tuple[str, bytes]],
) -> None:
    """Write a synthetic VMMRdb-shaped ZIP to ``path``.

    ``entries`` is a list of ``(in_zip_path, body_bytes)`` tuples. The
    in-zip path should already include whatever top-level prefix the test
    is exercising (e.g. ``VMMRdb/honda_civic_2005/img1.jpg``).
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name, body in entries:
            zf.writestr(name, body)


def test_ingest_from_zip_happy_path(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    """Two entries: one year-suffix class, one no-year class — both ingested."""
    zip_path = tmp_path / "vmmrdb.zip"
    _build_vmmrdb_zip(
        zip_path,
        entries=[
            ("VMMRdb/honda_civic_2005/img1.jpg", _jpeg_bytes((255, 0, 0))),
            ("VMMRdb/acura_cl/img2.jpg", _jpeg_bytes((0, 255, 0))),
        ],
    )

    stats = import_vmmrdb_from_zip(
        conn=db,
        zip_path=zip_path,
        out_dir=out_dir,
    )

    assert stats == ImportStats(
        processed=2,
        inserted_listings=2,
        inserted_images=2,
        skipped_existing=0,
        skipped_parse_failures=0,
    )

    rl = listings.list_by_class(db, source="vmmrdb")
    assert len(rl) == 2
    by_model = {row.model: row for row in rl}
    assert by_model["civic"].make == "honda"
    assert by_model["civic"].year == 2005
    assert by_model["cl"].make == "acura"
    assert by_model["cl"].year is None

    # Files are on disk under per-class subdirs with sha256 stems.
    written = list(out_dir.rglob("*.jpg"))
    assert len(written) == 2
    for p in written:
        assert len(p.stem) == 64

    # Image rows point at the written files.
    for row in rl:
        assert row.split == "train"
        img_rows = images.list_for_listing(db, row.listing_id)
        assert len(img_rows) == 1
        assert Path(img_rows[0].local_path).exists()
        assert img_rows[0].position == 1


def test_ingest_from_zip_skips_non_image_entries(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    """Non-image entries (txt files, READMEs, directories) are silently skipped."""
    zip_path = tmp_path / "vmmrdb.zip"
    _build_vmmrdb_zip(
        zip_path,
        entries=[
            ("VMMRdb/honda_civic_2005/img1.jpg", _jpeg_bytes((255, 0, 0))),
            ("VMMRdb/README.txt", b"this is not an image"),
            ("VMMRdb/honda_civic_2005/notes.txt", b"side note"),
            ("VMMRdb/acura_cl/img2.jpg", _jpeg_bytes((0, 255, 0))),
        ],
    )

    stats = import_vmmrdb_from_zip(
        conn=db,
        zip_path=zip_path,
        out_dir=out_dir,
    )

    # Only the two JPEGs are counted as processed.
    assert stats.processed == 2
    assert stats.inserted_images == 2
    assert stats.skipped_parse_failures == 0


def test_ingest_from_zip_handles_unknown_prefix(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    """Entries under VMMRdb_master/, no prefix, or VMMRdb/ all parse correctly."""
    zip_path = tmp_path / "vmmrdb.zip"
    _build_vmmrdb_zip(
        zip_path,
        entries=[
            ("VMMRdb_master/honda_civic_2005/img1.jpg", _jpeg_bytes((255, 0, 0))),
            ("acura_cl/img2.jpg", _jpeg_bytes((0, 255, 0))),
            ("VMMRdb/ford_f-150_2010/img3.jpg", _jpeg_bytes((0, 0, 255))),
        ],
    )

    stats = import_vmmrdb_from_zip(
        conn=db,
        zip_path=zip_path,
        out_dir=out_dir,
    )

    assert stats.processed == 3
    assert stats.inserted_images == 3
    assert stats.skipped_parse_failures == 0

    rl = listings.list_by_class(db, source="vmmrdb")
    assert {row.model for row in rl} == {"civic", "cl", "f-150"}
    by_model = {row.model: row for row in rl}
    assert by_model["civic"].year == 2005
    assert by_model["cl"].year is None
    assert by_model["f-150"].year == 2010


def test_ingest_from_zip_idempotent(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    """Re-running over the same ZIP inserts nothing new."""
    zip_path = tmp_path / "vmmrdb.zip"
    _build_vmmrdb_zip(
        zip_path,
        entries=[
            ("VMMRdb/honda_civic_2005/img1.jpg", _jpeg_bytes((255, 0, 0))),
            ("VMMRdb/acura_cl/img2.jpg", _jpeg_bytes((0, 255, 0))),
        ],
    )

    first = import_vmmrdb_from_zip(conn=db, zip_path=zip_path, out_dir=out_dir)
    assert first.inserted_images == 2

    second = import_vmmrdb_from_zip(conn=db, zip_path=zip_path, out_dir=out_dir)
    assert second.processed == 2
    assert second.inserted_listings == 0
    assert second.inserted_images == 0
    assert second.skipped_existing == 2


def test_ingest_from_zip_dry_run_writes_nothing(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    """``dry_run=True`` counts rows but writes no files and inserts no rows."""
    zip_path = tmp_path / "vmmrdb.zip"
    _build_vmmrdb_zip(
        zip_path,
        entries=[
            ("VMMRdb/honda_civic_2005/img1.jpg", _jpeg_bytes((255, 0, 0))),
            ("VMMRdb/acura_cl/img2.jpg", _jpeg_bytes((0, 255, 0))),
        ],
    )

    stats = import_vmmrdb_from_zip(
        conn=db,
        zip_path=zip_path,
        out_dir=out_dir,
        dry_run=True,
    )

    assert stats.processed == 2
    assert stats.inserted_listings == 0
    assert stats.inserted_images == 0

    # No files were extracted.
    if out_dir.exists():
        assert list(out_dir.rglob("*.jpg")) == []

    # No DB rows were inserted.
    rl = listings.list_by_class(db, source="vmmrdb")
    assert rl == []


def test_ingest_from_zip_limit_caps_processing(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    """``limit=2`` stops the iterator after processing 2 image entries."""
    zip_path = tmp_path / "vmmrdb.zip"
    _build_vmmrdb_zip(
        zip_path,
        entries=[
            (f"VMMRdb/honda_civic_2005/img{i}.jpg", _jpeg_bytes((i * 30, 0, 0))) for i in range(5)
        ],
    )

    stats = import_vmmrdb_from_zip(
        conn=db,
        zip_path=zip_path,
        out_dir=out_dir,
        limit=2,
    )

    assert stats.processed == 2
    assert stats.inserted_images == 2


def test_ingest_from_zip_skips_parse_failures(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    """Entries whose class-dir token can't be parsed are counted but skipped."""
    zip_path = tmp_path / "vmmrdb.zip"
    _build_vmmrdb_zip(
        zip_path,
        entries=[
            ("VMMRdb/honda_civic_2005/img1.jpg", _jpeg_bytes((255, 0, 0))),
            # Class dir with no underscore -> parse failure.
            ("VMMRdb/banana/img2.jpg", _jpeg_bytes((128, 128, 128))),
            ("VMMRdb/acura_cl/img3.jpg", _jpeg_bytes((0, 0, 255))),
        ],
    )

    stats = import_vmmrdb_from_zip(
        conn=db,
        zip_path=zip_path,
        out_dir=out_dir,
    )

    assert stats.processed == 3
    assert stats.inserted_images == 2
    assert stats.skipped_parse_failures == 1


def test_ingest_from_zip_missing_file_rejected(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        import_vmmrdb_from_zip(
            conn=db,
            zip_path=tmp_path / "no_such.zip",
            out_dir=out_dir,
        )


def test_ingest_from_zip_invalid_limit_rejected(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "vmmrdb.zip"
    _build_vmmrdb_zip(zip_path, entries=[])
    with pytest.raises(ValueError):
        import_vmmrdb_from_zip(
            conn=db,
            zip_path=zip_path,
            out_dir=out_dir,
            limit=0,
        )


def test_ingest_from_zip_invalid_log_every_rejected(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "vmmrdb.zip"
    _build_vmmrdb_zip(zip_path, entries=[])
    with pytest.raises(ValueError):
        import_vmmrdb_from_zip(
            conn=db,
            zip_path=zip_path,
            out_dir=out_dir,
            log_every=0,
        )


def test_ingest_from_zip_records_provided_split(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    """The semantic ``split`` kwarg is recorded in ``listings.split``."""
    zip_path = tmp_path / "vmmrdb.zip"
    _build_vmmrdb_zip(
        zip_path,
        entries=[("VMMRdb/honda_civic_2005/img1.jpg", _jpeg_bytes((255, 0, 0)))],
    )
    import_vmmrdb_from_zip(
        conn=db,
        zip_path=zip_path,
        out_dir=out_dir,
        split="val",
    )
    rl = listings.list_by_class(db, source="vmmrdb")
    assert len(rl) == 1
    assert rl[0].split == "val"


def test_ingest_from_zip_skips_hidden_and_macosx(
    db: sqlite3.Connection,
    out_dir: Path,
    tmp_path: Path,
) -> None:
    """``__MACOSX/`` resource-forks and hidden ``.foo`` entries are skipped."""
    zip_path = tmp_path / "vmmrdb.zip"
    _build_vmmrdb_zip(
        zip_path,
        entries=[
            ("VMMRdb/honda_civic_2005/img1.jpg", _jpeg_bytes((255, 0, 0))),
            ("__MACOSX/VMMRdb/honda_civic_2005/._img1.jpg", b"macos junk"),
            ("VMMRdb/.DS_Store", b"finder junk"),
        ],
    )

    stats = import_vmmrdb_from_zip(
        conn=db,
        zip_path=zip_path,
        out_dir=out_dir,
    )

    assert stats.processed == 1
    assert stats.inserted_images == 1
    assert stats.skipped_parse_failures == 0
