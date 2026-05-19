"""Tests for the ``build-prototypes`` CLI (Phase 6.1 per-view payload).

We reuse the same OpenCLIP stubbing strategy as ``test_baseline.py`` so
the test never downloads real weights. The CLI's job is to:

1. Open the DB.
2. Call :func:`car_lense_engine.eval.baseline.build_prototypes` (v1) or
   :func:`build_prototypes_by_view` (v2 with ``--per-view``).
3. Serialize the result as a ``.pt`` payload with the documented
   schema.

The view-conditional payload (``--per-view``) is the new contract; the
default-mode test guards backwards compat.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from car_lense_engine.dataset.canonical_labels import year_to_generation
from car_lense_engine.db import Image, Listing, images, listings, open_db
from car_lense_engine.eval import build_prototypes_cli
from car_lense_engine.eval.baseline import EXTERIOR_VIEWS, class_id_for

torch = pytest.importorskip("torch")


# --------------------------------------------------------------- stub model


class _StubModel:
    """Mirror of the stub in ``test_baseline.py``.

    ``encode_image`` consults a ``path -> embedding`` table that the
    test populates; ``preprocess`` smuggles the path's row index through
    the last slot of the preprocessed tensor.
    """

    def __init__(self, *, path_index: dict[str, int], embeddings: torch.Tensor) -> None:
        self._path_index = path_index
        self._embeddings = embeddings

    def eval(self) -> _StubModel:
        return self

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        indices = batch[:, -1].long().tolist()
        return self._embeddings[indices]


class _StubOpenClip:
    def __init__(self, model: _StubModel, preprocess: Any) -> None:
        self._model = model
        self._preprocess = preprocess

    def create_model_and_transforms(
        self,
        model_name: str,
        *,
        pretrained: str,
        device: str,
    ) -> tuple[_StubModel, None, Any]:
        return self._model, None, self._preprocess


def _install_stub_open_clip(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embeddings_by_path: dict[Path, torch.Tensor],
) -> None:
    paths = list(embeddings_by_path.keys())
    path_index = {str(p): i for i, p in enumerate(paths)}
    if paths:
        embeds = torch.stack([embeddings_by_path[p] for p in paths], dim=0)
        embed_dim = int(embeds.shape[1])
    else:
        embed_dim = 8
        embeds = torch.zeros((0, embed_dim))

    def stub_preprocess(img: Any) -> torch.Tensor:
        path_str: str = img._test_path  # type: ignore[attr-defined]
        idx = path_index[path_str]
        out = torch.zeros(embed_dim + 1)
        out[-1] = float(idx)
        return out

    n, d = embeds.shape
    padded = torch.zeros((n, d + 1))
    padded[:, :d] = embeds
    model = _StubModel(path_index=path_index, embeddings=padded)
    stub = _StubOpenClip(model, stub_preprocess)
    fake_mod: Any = stub
    monkeypatch.setitem(sys.modules, "open_clip", fake_mod)


@pytest.fixture
def patch_pil_to_carry_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image as PILImage

    original_open = PILImage.open

    def open_with_path(fp: Any, *args: Any, **kwargs: Any) -> Any:
        img = original_open(fp, *args, **kwargs)
        img._test_path = str(fp)  # type: ignore[attr-defined]
        original_convert = img.convert

        def convert_with_path(mode: str, *cargs: Any, **ckwargs: Any) -> Any:
            converted = original_convert(mode, *cargs, **ckwargs)
            converted._test_path = img._test_path
            return converted

        img.convert = convert_with_path  # type: ignore[method-assign]
        return img

    monkeypatch.setattr(PILImage, "open", open_with_path)


# --------------------------------------------------------------- fixtures


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


def _make_image_file(path: Path) -> Path:
    from PIL import Image as PILImage

    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (4, 4), color=(255, 255, 255)).save(path, format="JPEG")
    return path


def _basis(idx: int, dim: int = 8) -> torch.Tensor:
    v = torch.full((dim,), 0.1)
    v[idx % dim] = 5.0
    return v


def _seed_view_rows(
    conn: sqlite3.Connection,
    *,
    tmp_path: Path,
    rows_spec: list[tuple[int, str, str, str | None]],
) -> dict[Path, tuple[str, str | None]]:
    """Seed the DB and return a ``path -> (class_id, view)`` map."""
    out: dict[Path, tuple[str, str | None]] = {}
    for offset, (year, make, model, view) in enumerate(rows_spec):
        gen_year = year_to_generation(year)
        cid = class_id_for(gen_year, make, model)
        assert cid is not None
        counter = offset + 1
        listing_id = f"stanford_cars:cli_{counter:04d}"
        url = f"stanford_cars://{cid}/{counter:04d}"
        view_token = view if view is not None else "noview"
        img_path = tmp_path / "imgs" / f"{cid}_{view_token}_{counter}.jpg"
        _make_image_file(img_path)
        listings.insert_listing(
            conn,
            Listing(
                listing_id=listing_id,
                source="stanford_cars",
                url=url,
                year=year,
                make=make,
                model=model,
                split="train",
                canonical_make=make,
                canonical_model=model,
                generation_year=gen_year,
            ),
        )
        image_id = f"{counter:064d}"
        images.insert_image(
            conn,
            Image(
                image_id=image_id,
                listing_id=listing_id,
                source_url=url,
                local_path=str(img_path),
                position=1,
            ),
        )
        with conn:
            conn.execute(
                "UPDATE images SET split = ?, view = ? WHERE image_id = ?",
                ("train", view, image_id),
            )
        out[img_path] = (cid, view)
    return out


def _seed_view_rows_for_source(
    conn: sqlite3.Connection,
    *,
    tmp_path: Path,
    rows_spec: list[tuple[int, str, str, str | None, str]],
    counter_offset: int = 0,
) -> dict[Path, tuple[str, str | None]]:
    """Like :func:`_seed_view_rows` but each row carries an explicit source.

    Each spec is ``(year, make, model, view, source)``. Used by the
    multi-source CLI tests so a single DB can contain rows from
    ``compcars``, ``vmmrdb``, etc. without collisions on listing_id /
    image_id (the caller passes ``counter_offset`` to keep ids unique
    across multiple invocations).
    """
    out: dict[Path, tuple[str, str | None]] = {}
    for offset, (year, make, model, view, source) in enumerate(rows_spec):
        gen_year = year_to_generation(year)
        cid = class_id_for(gen_year, make, model)
        assert cid is not None
        counter = offset + 1 + counter_offset
        listing_id = f"{source}:cli_{counter:04d}"
        url = f"{source}://{cid}/{counter:04d}"
        view_token = view if view is not None else "noview"
        img_path = tmp_path / "imgs" / f"{source}_{cid}_{view_token}_{counter}.jpg"
        _make_image_file(img_path)
        listings.insert_listing(
            conn,
            Listing(
                listing_id=listing_id,
                source=source,
                url=url,
                year=year,
                make=make,
                model=model,
                split="train",
                canonical_make=make,
                canonical_model=model,
                generation_year=gen_year,
            ),
        )
        image_id = f"{counter:064d}"
        images.insert_image(
            conn,
            Image(
                image_id=image_id,
                listing_id=listing_id,
                source_url=url,
                local_path=str(img_path),
                position=1,
            ),
        )
        with conn:
            conn.execute(
                "UPDATE images SET split = ?, view = ? WHERE image_id = ?",
                ("train", view, image_id),
            )
        out[img_path] = (cid, view)
    return out


# --------------------------------------------------------------- tests


def test_cli_default_writes_v1_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """Without ``--per-view``, the CLI writes the legacy v1 payload."""
    open_db(db_path).close()
    conn = open_db(db_path)
    try:
        path_map = _seed_view_rows(
            conn,
            tmp_path=tmp_path,
            rows_spec=[
                (2012, "honda", "civic", "front"),
                (2012, "honda", "civic", "rear"),
                (2012, "toyota", "camry", "side"),
            ],
        )
    finally:
        conn.close()

    embeddings_by_path: dict[Path, torch.Tensor] = {p: _basis(i) for i, p in enumerate(path_map)}
    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    output = tmp_path / "cache" / "prototypes.pt"
    rc = build_prototypes_cli.main(
        [
            "--db",
            str(db_path),
            "--source",
            "stanford_cars",
            "--train-split",
            "train",
            "--model",
            "MobileCLIP-S2",
            "--pretrained",
            "datacompdr",
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert output.exists()

    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert "schema_version" not in payload
    assert "prototypes" in payload
    assert "class_ids" in payload
    assert int(payload["prototypes"].shape[0]) == len(payload["class_ids"])


def test_cli_per_view_writes_v2_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """``--per-view`` writes a schema_version=2 payload with view-keyed tensors."""
    open_db(db_path).close()
    conn = open_db(db_path)
    try:
        path_map = _seed_view_rows(
            conn,
            tmp_path=tmp_path,
            rows_spec=[
                (2012, "honda", "civic", "front"),
                (2012, "honda", "civic", "rear"),
                (2012, "toyota", "camry", "side"),
                (2012, "toyota", "camry", "side"),
                # Non-exterior: must be filtered out.
                (2012, "honda", "civic", "interior"),
            ],
        )
    finally:
        conn.close()

    embeddings_by_path: dict[Path, torch.Tensor] = {p: _basis(i) for i, p in enumerate(path_map)}
    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    output = tmp_path / "cache" / "prototypes_by_view.pt"
    rc = build_prototypes_cli.main(
        [
            "--db",
            str(db_path),
            "--source",
            "stanford_cars",
            "--train-split",
            "train",
            "--model",
            "MobileCLIP-S2",
            "--pretrained",
            "datacompdr",
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--output",
            str(output),
            "--per-view",
        ]
    )
    assert rc == 0
    assert output.exists()

    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert int(payload["schema_version"]) == 2
    assert set(payload["prototypes_by_view"].keys()) == set(EXTERIOR_VIEWS)
    assert list(payload["view_names"]) == list(EXTERIOR_VIEWS)
    n_classes = len(payload["class_ids"])
    assert n_classes == 2  # honda civic + toyota camry
    embed_dim = int(payload["config"]["embed_dim"])
    assert embed_dim > 0
    for view in EXTERIOR_VIEWS:
        tensor = payload["prototypes_by_view"][view]
        assert tensor.shape == (n_classes, embed_dim)
    # Config block reports the per-view metadata.
    cfg = payload["config"]
    assert cfg["source"] == "stanford_cars"
    assert cfg["split"] == "train"
    assert cfg["checkpoint_path_used"] is None
    assert cfg["model"] == "MobileCLIP-S2"


def test_build_prototypes_cli_parses_comma_separated_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """``--source compcars,vmmrdb`` propagates as a list into the cache config.

    Seed listings under two distinct sources, run the CLI with both
    names, and verify the resulting ``.pt`` payload's ``config.source``
    field is the joined string (cache stays a flat string for back-compat
    with existing readers) AND that the prototype index includes the
    classes from both sources.
    """
    open_db(db_path).close()
    conn = open_db(db_path)
    try:
        # One class in compcars, one in vmmrdb.
        path_map_a = _seed_view_rows_for_source(
            conn,
            tmp_path=tmp_path,
            rows_spec=[
                (2012, "honda", "civic", "front", "compcars"),
            ],
        )
        path_map_b = _seed_view_rows_for_source(
            conn,
            tmp_path=tmp_path,
            rows_spec=[
                (2012, "toyota", "camry", "front", "vmmrdb"),
            ],
            counter_offset=10,
        )
    finally:
        conn.close()

    embeddings_by_path: dict[Path, torch.Tensor] = {
        p: _basis(i) for i, p in enumerate([*path_map_a, *path_map_b])
    }
    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    output = tmp_path / "cache" / "multi_prototypes.pt"
    rc = build_prototypes_cli.main(
        [
            "--db",
            str(db_path),
            "--source",
            "compcars,vmmrdb",
            "--train-split",
            "train",
            "--model",
            "MobileCLIP-S2",
            "--pretrained",
            "datacompdr",
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    payload = torch.load(output, map_location="cpu", weights_only=False)
    # The cache stores the joined string form for backward compatibility
    # with existing readers that ``str(cfg["source"])`` it.
    assert payload["config"]["source"] == "compcars,vmmrdb"
    # Both classes (one per source) are represented.
    assert len(payload["class_ids"]) == 2


def test_build_prototypes_cli_rejects_empty_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
) -> None:
    """An all-whitespace ``--source`` exits with code 2."""
    open_db(db_path).close()
    _install_stub_open_clip(monkeypatch, embeddings_by_path={})

    with pytest.raises(SystemExit) as excinfo:
        build_prototypes_cli.main(
            [
                "--db",
                str(db_path),
                "--source",
                " , ,",
                "--device",
                "cpu",
                "--batch-size",
                "2",
            ]
        )
    assert excinfo.value.code == 2


def test_cli_per_view_default_output_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """``--per-view`` without ``--output`` writes to ``prototypes_by_view.pt``."""
    open_db(db_path).close()
    conn = open_db(db_path)
    try:
        path_map = _seed_view_rows(
            conn,
            tmp_path=tmp_path,
            rows_spec=[(2012, "honda", "civic", "front")],
        )
    finally:
        conn.close()

    embeddings_by_path: dict[Path, torch.Tensor] = {p: _basis(0) for p in path_map}
    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    # Run from tmp_path so the relative default lands inside the tmp tree.
    monkeypatch.chdir(tmp_path)
    rc = build_prototypes_cli.main(
        [
            "--db",
            str(db_path),
            "--source",
            "stanford_cars",
            "--train-split",
            "train",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--per-view",
        ]
    )
    assert rc == 0
    # Default per-view filename, not prototypes.pt.
    assert (tmp_path / "cache" / "prototypes_by_view.pt").exists()
    assert not (tmp_path / "cache" / "prototypes.pt").exists()
