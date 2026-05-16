"""Tests for the Stanford Cars ingest module.

The HF ``datasets`` library is stubbed via ``monkeypatch`` so tests never
hit the network. Each fake row is a ``{"image": PIL.Image, "label": int,
"class": str}`` dict — matching the schema of the real HF mirror.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest

PIL_Image = pytest.importorskip("PIL.Image")

from car_lense_engine.dataset.stanford_cars import (  # noqa: E402
    ImportStats,
    import_stanford_cars,
)
from car_lense_engine.db import images, listings, open_db  # noqa: E402

# --------------------------------------------------------- helpers


def _make_image(color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> Any:
    """Build a tiny solid-color PIL image."""
    return PIL_Image.new("RGB", size, color)


def _write_catalog(tmp_path: Path) -> Path:
    """Write a minimal classes.json with the makes we use in test fixtures."""
    catalog = {
        "meta": {
            "generated_at": "2026-01-01T00:00:00Z",
            "source": "test",
            "year_range": [1981, 2026],
            "total_makes": 0,
            "total_models": 0,
            "total_class_entries": 0,
        },
        "makes": [
            {"make_id": 1, "make_name": "Acura", "models": []},
            {"make_id": 2, "make_name": "Hyundai", "models": []},
            {"make_id": 3, "make_name": "Tesla", "models": []},
            {"make_id": 4, "make_name": "Chevrolet", "models": []},
            {"make_id": 5, "make_name": "AM General", "models": []},
        ],
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
    return tmp_path / "stanford_cars"


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
            "class": "Acura RL Sedan 2012",
        },
        {
            "image": _make_image((0, 255, 0)),
            "label": 1,
            "class": "Hyundai Sonata 2007",
        },
        {
            "image": _make_image((0, 0, 255)),
            "label": 2,
            "class": "Tesla Model S Sedan 2012",
        },
    ]
    _install_stub(monkeypatch, rows)

    stats = import_stanford_cars(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
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
    rl = listings.list_by_class(db, source="stanford_cars")
    assert len(rl) == 3
    by_model = {row.model: row for row in rl}
    assert by_model["RL"].year == 2012
    assert by_model["RL"].make == "Acura"
    assert by_model["RL"].body_style == "Sedan"
    assert by_model["Sonata"].body_style is None
    assert by_model["Model S"].make == "Tesla"

    # Image row local_path points at the actual on-disk file.
    for row in rl:
        img_rows = images.list_for_listing(db, row.listing_id)
        assert len(img_rows) == 1
        assert Path(img_rows[0].local_path).exists()
        assert img_rows[0].position == 1
        assert img_rows[0].phash is None


def test_ingest_is_idempotent(
    db: sqlite3.Connection,
    out_dir: Path,
    catalog_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "image": _make_image((255, 0, 0)),
            "label": 0,
            "class": "Acura RL Sedan 2012",
        },
        {
            "image": _make_image((0, 255, 0)),
            "label": 1,
            "class": "Hyundai Sonata 2007",
        },
    ]
    _install_stub(monkeypatch, rows)

    first = import_stanford_cars(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
    )
    assert first.inserted_images == 2

    second = import_stanford_cars(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
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
        {
            "image": _make_image((255, 0, 0)),
            "label": 0,
            "class": "Acura RL Sedan 2012",
        },
        {
            # No trailing year -> parse failure.
            "image": _make_image((128, 128, 128)),
            "label": 99,
            "class": "Banana",
        },
        {
            "image": _make_image((0, 255, 0)),
            "label": 1,
            "class": "Hyundai Sonata 2007",
        },
    ]
    _install_stub(monkeypatch, rows)

    stats = import_stanford_cars(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
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
            "class": "Acura RL Sedan 2012",
        }
        for i in range(5)
    ]
    # All same class string but distinct pixel colors -> distinct SHA-256s.
    _install_stub(monkeypatch, rows)

    stats = import_stanford_cars(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
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
        {
            "image": _make_image((255, 0, 0)),
            "label": 0,
            "class": "Acura RL Sedan 2012",
        }
    ]
    _install_stub(monkeypatch, rows)

    import_stanford_cars(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
    )

    rl = listings.list_by_class(db, source="stanford_cars")
    assert len(rl) == 1
    only = rl[0]
    assert only.listing_id.startswith("stanford_cars:")
    assert only.url.startswith("stanford_cars://")
    # The URL embeds the class_id (derived from year/make/model/body) and
    # the image_id. The exact slug doesn't matter for the test; just check
    # that the URL contains the trailing sha.
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
        import_stanford_cars(
            conn=db,
            out_dir=out_dir,
            catalog_path=catalog_path,
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
        import_stanford_cars(
            conn=db,
            out_dir=out_dir,
            catalog_path=catalog_path,
            limit=0,
        )


def test_class_id_slugify_handles_special_chars() -> None:
    from car_lense_engine.dataset.stanford_cars import _slugify

    assert _slugify("2012_Mercedes-Benz_Sprinter_Van") == "2012_Mercedes-Benz_Sprinter_Van"
    assert _slugify("Foo/Bar Baz") == "Foo_Bar_Baz"
    assert _slugify("Foo   Bar") == "Foo_Bar"


def test_ingest_resolves_int_classlabel_from_hf_features(
    db: sqlite3.Connection,
    out_dir: Path,
    catalog_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default HF mirror emits an int ClassLabel — verify the decoder resolves it.

    Rows here carry ``label`` as an int (no pre-resolved ``class`` string),
    which matches ``Multimodal-Fatima/StanfordCars_train``. The decoder must
    consult ``ds.features['label'].int2str`` to recover the class name.
    """
    class_names = ["Acura RL Sedan 2012", "Tesla Model S Sedan 2012"]
    fake_features = {"label": _FakeClassLabel(class_names)}
    rows: list[dict[str, Any]] = [
        {"image": _make_image((255, 0, 0)), "label": 0},
        {"image": _make_image((0, 0, 255)), "label": 1},
    ]
    _install_stub(monkeypatch, rows, features=fake_features)

    stats = import_stanford_cars(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
    )

    assert stats.processed == 2
    assert stats.inserted_listings == 2
    assert stats.inserted_images == 2
    assert stats.skipped_parse_failures == 0

    rl = listings.list_by_class(db, source="stanford_cars")
    assert {row.make for row in rl} == {"Acura", "Tesla"}
    by_make = {row.make: row for row in rl}
    assert by_make["Acura"].model == "RL"
    assert by_make["Acura"].year == 2012
    assert by_make["Tesla"].model == "Model S"
    assert by_make["Tesla"].year == 2012


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

    stats = import_stanford_cars(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
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
        {
            "image": _make_image((255, 0, 0)),
            "label": 0,
            "class": "Acura RL Sedan 2012",
        }
    ]
    calls = _install_stub(monkeypatch, rows)
    import_stanford_cars(
        conn=db,
        out_dir=out_dir,
        catalog_path=catalog_path,
        hf_dataset="some/mirror",
        split="test",
    )
    assert calls == [("some/mirror", "test")]
