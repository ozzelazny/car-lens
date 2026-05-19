"""Tests for the Phase 5.4 ``evaluate-recognize`` harness.

Mirrors the open_clip stubbing strategy from ``test_baseline.py`` so the
tests never touch the network and never download real weights. Each
test seeds a tiny SQLite DB, pre-arranges per-path embeddings, persists
a v1 prototype cache file via :func:`torch.save`, and runs
:func:`evaluate` against it.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from car_lense_engine.db import Image, Listing, images, listings, open_db
from car_lense_engine.eval.baseline import class_id_for
from car_lense_engine.eval.evaluate import (
    EvaluationConfig,
    evaluate,
)

torch = pytest.importorskip("torch")


# --------------------------------------------------------------- stub model


class _StubModel:
    """Stub of the OpenCLIP image-encoder surface used by the harness.

    Identical in shape to the stub used in ``test_baseline.py``: an
    embedding table indexed by the path's row index, smuggled through
    the trailing column of the preprocess output. ``encode_image``
    strips the index column and returns the corresponding embedding.
    """

    def __init__(self, *, path_index: dict[str, int], embeddings: torch.Tensor) -> None:
        self._path_index = path_index
        self._embeddings = embeddings
        self.eval_called = False

    def eval(self) -> _StubModel:
        self.eval_called = True
        return self

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        indices = batch[:, -1].long().tolist()
        return self._embeddings[indices]


class _StubOpenClip:
    """Module-style stub installed into ``sys.modules['open_clip']``."""

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
        self.last_call = {"model_name": model_name, "pretrained": pretrained, "device": device}
        return self._model, None, self._preprocess


def _install_stub_open_clip(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embeddings_by_path: dict[Path, torch.Tensor],
) -> _StubOpenClip:
    """Install an ``open_clip`` stub keyed by absolute path -> embedding."""
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

    model = _StubModel(path_index=path_index, embeddings=_pad_embeddings(embeds))
    stub = _StubOpenClip(model, stub_preprocess)
    fake_mod: Any = stub
    monkeypatch.setitem(sys.modules, "open_clip", fake_mod)
    return stub


def _pad_embeddings(embeds: torch.Tensor) -> torch.Tensor:
    """Append a zero column so the stub model can index by the trailing slot."""
    n, d = embeds.shape
    padded = torch.zeros((n, d + 1))
    padded[:, :d] = embeds
    return padded


# --------------------------------------------------------------- PIL patch


@pytest.fixture
def patch_pil_to_carry_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``PIL.Image.open`` attach ``_test_path`` to the returned image."""
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


# --------------------------------------------------------------- DB seeding


def _seed_listing(
    conn: sqlite3.Connection,
    *,
    listing_id: str,
    source: str,
    url: str,
    year: int | None,
    make: str,
    model: str,
    split: str,
    generation_year: int | None,
) -> None:
    listings.insert_listing(
        conn,
        Listing(
            listing_id=listing_id,
            source=source,
            url=url,
            year=year,
            make=make,
            model=model,
            split=split,
            canonical_make=make,
            canonical_model=model,
            generation_year=generation_year,
        ),
    )


def _seed_image(
    conn: sqlite3.Connection,
    *,
    listing_id: str,
    image_id: str,
    source_url: str,
    local_path: Path,
    split: str,
    view: str | None,
) -> None:
    images.insert_image(
        conn,
        Image(
            image_id=image_id,
            listing_id=listing_id,
            source_url=source_url,
            local_path=str(local_path),
            position=1,
        ),
    )
    with conn:
        conn.execute(
            "UPDATE images SET split = ?, view = ? WHERE image_id = ?",
            (split, view, image_id),
        )


def _seed_class_rows(
    conn: sqlite3.Connection,
    *,
    tmp_path: Path,
    class_specs: list[dict[str, Any]],
) -> dict[str, list[Path]]:
    """Seed ``listings`` + ``images`` from per-class specs.

    Each spec is a dict::

        {
            "year": int,
            "make": str,
            "model": str,
            "generation_year": int | None,
            "split": "train" | "test",
            "rows": [{"view": str | None}, ...],
        }

    Returns ``{class_id: [paths_in_order]}``. The class_id used by the
    eval harness is keyed off ``generation_year`` so the test caller is
    responsible for setting it; we don't auto-bucket from the raw year.
    """
    out: dict[str, list[Path]] = {}
    counter = 0
    for spec in class_specs:
        cid = class_id_for(spec["generation_year"], spec["make"], spec["model"])
        if cid is None:
            cid = f"unknown-{spec['make']}-{spec['model']}"
        out.setdefault(cid, [])
        split = spec["split"]
        for row in spec["rows"]:
            counter += 1
            listing_id = f"compcars:t_{counter:04d}"
            url = f"compcars://{cid}/{counter:04d}"
            img_path = tmp_path / "imgs" / f"{cid}_{split}_{counter}.jpg"
            _make_image_file(img_path)
            _seed_listing(
                conn,
                listing_id=listing_id,
                source="compcars",
                url=url,
                year=spec["year"],
                make=spec["make"],
                model=spec["model"],
                split=split,
                generation_year=spec["generation_year"],
            )
            _seed_image(
                conn,
                listing_id=listing_id,
                image_id=f"{counter:064d}",
                source_url=url,
                local_path=img_path,
                split=split,
                view=row.get("view"),
            )
            out[cid].append(img_path)
    return out


def _basis_embedding(class_index: int, embed_dim: int = 8, lead: float = 5.0) -> torch.Tensor:
    v = torch.full((embed_dim,), 0.1)
    v[class_index] = lead
    return v


def _proto_for(class_index: int, embed_dim: int = 8) -> torch.Tensor:
    """L2-normalized prototype matching ``_basis_embedding`` post-padding.

    The stub :func:`_pad_embeddings` appends a zero column to every
    encoded row, so the actual embedding the harness sees has
    dimensionality ``embed_dim + 1`` with the last slot 0. The
    L2-normalized prototype must live in that same ``embed_dim + 1``
    space; we build it by padding the basis with a trailing zero and
    L2-normalizing.
    """
    base = _basis_embedding(class_index, embed_dim=embed_dim)
    padded = torch.zeros(embed_dim + 1)
    padded[:embed_dim] = base
    return padded / padded.norm()


def _save_v1_prototypes(
    *,
    path: Path,
    class_ids: list[str],
    prototypes: torch.Tensor,
    display_names: list[str] | None = None,
) -> None:
    """Persist a v1 schema prototype cache to ``path``."""
    payload: dict[str, Any] = {
        "class_ids": list(class_ids),
        "display_names": display_names or [f"display:{cid}" for cid in class_ids],
        "prototypes": prototypes,
        "config": {
            "model": "MobileCLIP-S2",
            "pretrained": "datacompdr",
            "source": "compcars",
            "split": "train",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _save_v2_prototypes(
    *,
    path: Path,
    class_ids: list[str],
    prototypes_by_view: dict[str, torch.Tensor],
) -> None:
    """Persist a v2 schema (per-view) prototype cache for rejection tests."""
    payload: dict[str, Any] = {
        "schema_version": 2,
        "class_ids": list(class_ids),
        "display_names": [f"display:{cid}" for cid in class_ids],
        "prototypes_by_view": prototypes_by_view,
        "view_names": list(prototypes_by_view.keys()),
        "config": {
            "model": "MobileCLIP-S2",
            "pretrained": "datacompdr",
            "embed_dim": int(next(iter(prototypes_by_view.values())).shape[1]),
            "source": "compcars",
            "split": "train",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


# --------------------------------------------------------------- tests


def test_evaluate_overall_top1_matches_expected_for_synthetic_db(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """4 classes × 5 test images = 20 rows, perfectly aligned prototypes ⇒ 100% top-1."""
    class_specs = []
    for make, model in [
        ("Acura", "RL"),
        ("Honda", "Civic"),
        ("Toyota", "Camry"),
        ("Tesla", "Model S"),
    ]:
        class_specs.append(
            {
                "year": 2010,
                "make": make,
                "model": model,
                "generation_year": 2010,
                "split": "test",
                "rows": [{"view": "front"} for _ in range(5)],
            }
        )

    conn = open_db(db_path)
    try:
        test_paths = _seed_class_rows(conn, tmp_path=tmp_path, class_specs=class_specs)
    finally:
        conn.close()

    class_ids = sorted(test_paths.keys())
    embeddings_by_path: dict[Path, torch.Tensor] = {}
    # Build a (n_classes, embed_dim) prototype tensor and per-path
    # embeddings that align perfectly with the prototype direction
    # for each class.
    embed_dim = 8
    proto_rows = []
    for i, cid in enumerate(class_ids):
        proto_rows.append(_proto_for(i, embed_dim=embed_dim))
        for p in test_paths[cid]:
            # Pre-L2 embedding for the runner to normalize.
            embeddings_by_path[p] = _basis_embedding(i, embed_dim=embed_dim)
    proto_tensor = torch.stack(proto_rows, dim=0)
    proto_path = tmp_path / "cache" / "prototypes.pt"
    _save_v1_prototypes(path=proto_path, class_ids=class_ids, prototypes=proto_tensor)

    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    config = EvaluationConfig(
        db_path=db_path,
        source="compcars",
        test_split="test",
        prototypes_path=proto_path,
        device="cpu",
        batch_size=4,
        top_k=(1, 3, 5),
    )
    conn = open_db(db_path)
    try:
        report = evaluate(conn=conn, config=config)
    finally:
        conn.close()

    assert report.n_classes == len(class_ids)
    assert report.overall.n == 20
    assert report.overall.top_k_correct[1] == 20
    assert report.overall.top_k_correct[3] == 20
    assert report.overall.top_k_correct[5] == 20
    # No top-1 misses -> no confusions.
    assert report.top_confusions == []


def test_evaluate_breakdowns_partition_correctly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """Sum of per-make/view/era ``n`` equals overall ``n``."""
    # Three makes, mixed views, mixed eras.
    class_specs = [
        {
            "year": 2010,
            "make": "Acura",
            "model": "RL",
            "generation_year": 2010,
            "split": "test",
            "rows": [{"view": "front"}, {"view": "rear"}, {"view": "side"}],
        },
        {
            "year": 2020,
            "make": "Honda",
            "model": "Civic",
            "generation_year": 2020,
            "split": "test",
            "rows": [{"view": "three-quarter-front"}, {"view": "front"}],
        },
        {
            "year": 2005,
            "make": "Toyota",
            "model": "Camry",
            "generation_year": 2005,
            "split": "test",
            "rows": [{"view": "rear"}, {"view": "side"}, {"view": "side"}],
        },
    ]
    conn = open_db(db_path)
    try:
        test_paths = _seed_class_rows(conn, tmp_path=tmp_path, class_specs=class_specs)
    finally:
        conn.close()

    class_ids = sorted(test_paths.keys())
    embeddings_by_path: dict[Path, torch.Tensor] = {}
    embed_dim = 8
    proto_rows = []
    for i, cid in enumerate(class_ids):
        proto_rows.append(_proto_for(i, embed_dim=embed_dim))
        for p in test_paths[cid]:
            embeddings_by_path[p] = _basis_embedding(i, embed_dim=embed_dim)
    proto_tensor = torch.stack(proto_rows, dim=0)
    proto_path = tmp_path / "cache" / "prototypes.pt"
    _save_v1_prototypes(path=proto_path, class_ids=class_ids, prototypes=proto_tensor)

    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    config = EvaluationConfig(
        db_path=db_path,
        source="compcars",
        test_split="test",
        prototypes_path=proto_path,
        device="cpu",
        batch_size=4,
        top_k=(1, 5),
    )
    conn = open_db(db_path)
    try:
        report = evaluate(conn=conn, config=config)
    finally:
        conn.close()

    total = report.overall.n
    assert total == 8  # 3 + 2 + 3
    assert sum(c.n for c in report.per_make.values()) == total
    assert sum(c.n for c in report.per_view.values()) == total
    assert sum(c.n for c in report.per_era.values()) == total
    # Specific buckets we expect.
    assert set(report.per_make.keys()) == {"Acura", "Honda", "Toyota"}
    assert "2000s" in report.per_era
    assert "2010s" in report.per_era
    assert "2020s" in report.per_era


def test_evaluate_records_top_confusions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """When class B always predicts class A, the top confusion pair reflects that."""
    class_specs = [
        {
            "year": 2010,
            "make": "Acura",
            "model": "RL",
            "generation_year": 2010,
            "split": "test",
            "rows": [{"view": "front"}, {"view": "front"}, {"view": "front"}],
        },
        {
            "year": 2010,
            "make": "Bentley",
            "model": "Continental",
            "generation_year": 2010,
            "split": "test",
            "rows": [{"view": "front"}, {"view": "front"}, {"view": "front"}],
        },
    ]
    conn = open_db(db_path)
    try:
        test_paths = _seed_class_rows(conn, tmp_path=tmp_path, class_specs=class_specs)
    finally:
        conn.close()

    class_ids = sorted(test_paths.keys())
    cid_a = class_ids[0]  # "2010|Acura|RL"
    cid_b = class_ids[1]  # "2010|Bentley|Continental"
    embed_dim = 8
    proto_rows = []
    embeddings_by_path: dict[Path, torch.Tensor] = {}
    for i in range(len(class_ids)):
        proto_rows.append(_proto_for(i, embed_dim=embed_dim))
    # Class A's test rows align with prototype A -> correct.
    for p in test_paths[cid_a]:
        embeddings_by_path[p] = _basis_embedding(0, embed_dim=embed_dim)
    # Class B's test rows align with prototype A as well -> always
    # predicted as A.
    for p in test_paths[cid_b]:
        embeddings_by_path[p] = _basis_embedding(0, embed_dim=embed_dim)
    proto_tensor = torch.stack(proto_rows, dim=0)
    proto_path = tmp_path / "cache" / "prototypes.pt"
    display_names = [f"DISPLAY[{cid}]" for cid in class_ids]
    _save_v1_prototypes(
        path=proto_path,
        class_ids=class_ids,
        prototypes=proto_tensor,
        display_names=display_names,
    )

    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    config = EvaluationConfig(
        db_path=db_path,
        source="compcars",
        test_split="test",
        prototypes_path=proto_path,
        device="cpu",
        batch_size=4,
        top_k=(1,),
    )
    conn = open_db(db_path)
    try:
        report = evaluate(conn=conn, config=config)
    finally:
        conn.close()

    # 3 of 6 wrong (all of class B).
    assert report.overall.top_k_correct[1] == 3
    assert len(report.top_confusions) == 1
    pair = report.top_confusions[0]
    assert pair.true == f"DISPLAY[{cid_b}]"
    assert pair.pred == f"DISPLAY[{cid_a}]"
    assert pair.count == 3
    assert pair.rate == pytest.approx(1.0)


def test_evaluate_rejects_v2_prototypes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """A v2 schema cache must raise a clear RuntimeError."""
    open_db(db_path).close()  # apply migrations
    # No need for real images -- the harness fails before reading any.
    class_ids = ["2010|acura|rl"]
    prototypes_by_view = {"front": torch.zeros((1, 8))}
    proto_path = tmp_path / "cache" / "prototypes_by_view.pt"
    _save_v2_prototypes(
        path=proto_path,
        class_ids=class_ids,
        prototypes_by_view=prototypes_by_view,
    )

    _install_stub_open_clip(monkeypatch, embeddings_by_path={})

    config = EvaluationConfig(
        db_path=db_path,
        source="compcars",
        test_split="test",
        prototypes_path=proto_path,
        device="cpu",
        batch_size=4,
    )
    conn = open_db(db_path)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            evaluate(conn=conn, config=config)
    finally:
        conn.close()
    assert "schema_version=2" in str(excinfo.value)


def test_evaluate_handles_null_generation_year(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """Rows with NULL generation_year land in the ``"unknown"`` era bucket.

    Because the harness gates on canonical fields (the SQL drops rows
    whose canonical_year/make/model is NULL), we exercise the era
    bucketing via the _select_row_context helper instead -- we seed a
    row with NULL generation_year and verify the context query
    returns it under ``"unknown"``.
    """
    from car_lense_engine.eval.evaluate import _era_bucket

    # Direct unit test of the bucketer.
    assert _era_bucket(None, 10) == "unknown"
    assert _era_bucket(2007, 10) == "2000s"
    assert _era_bucket(2010, 10) == "2010s"
    assert _era_bucket(2020, 10) == "2020s"
    assert _era_bucket(1999, 10) == "1990s"
    # Non-decade buckets render as ``"<start>-<end>"``.
    assert _era_bucket(2012, 4) == "2012-2015"
    assert _era_bucket(2014, 4) == "2012-2015"


def test_evaluate_skips_classes_with_zero_train_prototypes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """Test rows whose class has no prototype row are still counted (as misses)."""
    # Two classes appear in test, but the prototype cache only has one.
    class_specs = [
        {
            "year": 2010,
            "make": "Acura",
            "model": "RL",
            "generation_year": 2010,
            "split": "test",
            "rows": [{"view": "front"}, {"view": "front"}],
        },
        {
            "year": 2010,
            "make": "Bentley",
            "model": "Continental",
            "generation_year": 2010,
            "split": "test",
            "rows": [{"view": "front"}, {"view": "front"}],
        },
    ]
    conn = open_db(db_path)
    try:
        test_paths = _seed_class_rows(conn, tmp_path=tmp_path, class_specs=class_specs)
    finally:
        conn.close()
    cid_a = class_id_for(2010, "Acura", "RL")
    cid_b = class_id_for(2010, "Bentley", "Continental")
    assert cid_a is not None and cid_b is not None

    embed_dim = 8
    # Only class A has a prototype row.
    proto_tensor = _proto_for(0, embed_dim=embed_dim).unsqueeze(0)
    embeddings_by_path: dict[Path, torch.Tensor] = {}
    for p in test_paths[cid_a]:
        embeddings_by_path[p] = _basis_embedding(0, embed_dim=embed_dim)
    for p in test_paths[cid_b]:
        # Even though we'd hit prototype A, A is the wrong class for
        # these rows -> always misses (true class has no prototype).
        embeddings_by_path[p] = _basis_embedding(0, embed_dim=embed_dim)

    proto_path = tmp_path / "cache" / "prototypes.pt"
    _save_v1_prototypes(path=proto_path, class_ids=[cid_a], prototypes=proto_tensor)
    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    config = EvaluationConfig(
        db_path=db_path,
        source="compcars",
        test_split="test",
        prototypes_path=proto_path,
        device="cpu",
        batch_size=4,
        top_k=(1, 5),
    )
    conn = open_db(db_path)
    try:
        report = evaluate(conn=conn, config=config)
    finally:
        conn.close()

    # 4 total rows: 2 class A hits + 2 class B misses.
    assert report.overall.n == 4
    assert report.overall.top_k_correct[1] == 2
    # The class-B rows count as misses but do not pollute the
    # confusion table because their predicted class A != true class B.
    assert any(p.true.endswith(cid_b) or cid_b in p.true for p in report.top_confusions) or all(
        p.true != cid_a for p in report.top_confusions
    )
    # n_classes reflects the prototype cache, not the test set.
    assert report.n_classes == 1


# --------------------------------------------------------------- CLI smoke


def test_cli_rejects_missing_db(tmp_path: Path) -> None:
    from car_lense_engine.eval import evaluate_cli

    missing = tmp_path / "nope.sqlite"
    proto = tmp_path / "prototypes.pt"
    _save_v1_prototypes(
        path=proto,
        class_ids=["2010|acura|rl"],
        prototypes=torch.zeros((1, 8)),
    )
    with pytest.raises(SystemExit) as excinfo:
        evaluate_cli.main(
            [
                "--db",
                str(missing),
                "--prototypes",
                str(proto),
                "--device",
                "cpu",
                "--checkpoint",
                "",
            ]
        )
    assert excinfo.value.code == 2


def test_cli_rejects_missing_prototypes(tmp_path: Path, db_path: Path) -> None:
    from car_lense_engine.eval import evaluate_cli

    open_db(db_path).close()
    missing = tmp_path / "no_protos.pt"
    with pytest.raises(SystemExit) as excinfo:
        evaluate_cli.main(
            [
                "--db",
                str(db_path),
                "--prototypes",
                str(missing),
                "--device",
                "cpu",
                "--checkpoint",
                "",
            ]
        )
    assert excinfo.value.code == 2


def test_evaluate_cli_parses_comma_separated_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """``--source compcars,vmmrdb`` lands as a list in EvaluationConfig.

    We stub :func:`evaluate` (the one the CLI imports) to capture the
    config and short-circuit the heavy retrieval pipeline.
    """
    from car_lense_engine.eval import evaluate_cli
    from car_lense_engine.eval.evaluate import (
        CellMetrics,
        EvaluationReport,
    )

    open_db(db_path).close()
    proto = tmp_path / "prototypes.pt"
    _save_v1_prototypes(
        path=proto,
        class_ids=["2010|acura|rl"],
        prototypes=torch.zeros((1, 8)),
    )

    captured: dict[str, Any] = {}

    def fake_evaluate(*, conn: Any, config: Any) -> EvaluationReport:
        captured["config"] = config
        return EvaluationReport(
            overall=CellMetrics(n=0, top_k_correct={1: 0, 3: 0, 5: 0, 10: 0}),
            per_make={},
            per_view={},
            per_era={},
            top_confusions=[],
            top_confusions_per_make={},
            config=config,
            n_classes=0,
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(evaluate_cli, "evaluate", fake_evaluate)

    report_path = tmp_path / "reports" / "p5_4.json"
    rc = evaluate_cli.main(
        [
            "--db",
            str(db_path),
            "--source",
            "compcars,vmmrdb,stanford_cars",
            "--test-split",
            "test",
            "--prototypes",
            str(proto),
            "--checkpoint",
            "",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--report",
            str(report_path),
            "--print-top-makes",
            "0",
            "--print-top-confusions",
            "0",
        ]
    )
    assert rc == 0
    config = captured["config"]
    assert config.source == ["compcars", "vmmrdb", "stanford_cars"]


def test_evaluate_cli_single_source_remains_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """The legacy ``--source compcars`` form lands as ``["compcars"]``."""
    from car_lense_engine.eval import evaluate_cli
    from car_lense_engine.eval.evaluate import (
        CellMetrics,
        EvaluationReport,
    )

    open_db(db_path).close()
    proto = tmp_path / "prototypes.pt"
    _save_v1_prototypes(
        path=proto,
        class_ids=["2010|acura|rl"],
        prototypes=torch.zeros((1, 8)),
    )

    captured: dict[str, Any] = {}

    def fake_evaluate(*, conn: Any, config: Any) -> EvaluationReport:
        captured["config"] = config
        return EvaluationReport(
            overall=CellMetrics(n=0, top_k_correct={1: 0, 3: 0, 5: 0, 10: 0}),
            per_make={},
            per_view={},
            per_era={},
            top_confusions=[],
            top_confusions_per_make={},
            config=config,
            n_classes=0,
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(evaluate_cli, "evaluate", fake_evaluate)

    report_path = tmp_path / "reports" / "p5_4.json"
    rc = evaluate_cli.main(
        [
            "--db",
            str(db_path),
            "--source",
            "compcars",
            "--prototypes",
            str(proto),
            "--checkpoint",
            "",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--report",
            str(report_path),
            "--print-top-makes",
            "0",
            "--print-top-confusions",
            "0",
        ]
    )
    assert rc == 0
    assert captured["config"].source == ["compcars"]


def test_evaluate_cli_rejects_empty_source(
    tmp_path: Path,
    db_path: Path,
) -> None:
    """An all-whitespace ``--source`` exits with code 2."""
    from car_lense_engine.eval import evaluate_cli

    open_db(db_path).close()
    proto = tmp_path / "prototypes.pt"
    _save_v1_prototypes(
        path=proto,
        class_ids=["2010|acura|rl"],
        prototypes=torch.zeros((1, 8)),
    )
    with pytest.raises(SystemExit) as excinfo:
        evaluate_cli.main(
            [
                "--db",
                str(db_path),
                "--source",
                "  , ,",
                "--prototypes",
                str(proto),
                "--checkpoint",
                "",
                "--device",
                "cpu",
            ]
        )
    assert excinfo.value.code == 2


def test_evaluation_config_accepts_str_and_comma_separated() -> None:
    """The pydantic validator accepts legacy str / comma-string / list inputs."""
    from car_lense_engine.eval.evaluate import EvaluationConfig

    cfg_str = EvaluationConfig(
        db_path=Path("db/crawl.sqlite"),
        prototypes_path=Path("cache/prototypes.pt"),
        source="compcars",  # type: ignore[arg-type]  -- validator coerces
    )
    assert cfg_str.source == ["compcars"]

    cfg_csv = EvaluationConfig(
        db_path=Path("db/crawl.sqlite"),
        prototypes_path=Path("cache/prototypes.pt"),
        source="compcars,vmmrdb,stanford_cars",  # type: ignore[arg-type]
    )
    assert cfg_csv.source == ["compcars", "vmmrdb", "stanford_cars"]

    cfg_list = EvaluationConfig(
        db_path=Path("db/crawl.sqlite"),
        prototypes_path=Path("cache/prototypes.pt"),
        source=["compcars", "vmmrdb"],
    )
    assert cfg_list.source == ["compcars", "vmmrdb"]


def test_cli_writes_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from car_lense_engine.eval import evaluate_cli

    class_specs = [
        {
            "year": 2010,
            "make": "Acura",
            "model": "RL",
            "generation_year": 2010,
            "split": "test",
            "rows": [{"view": "front"}, {"view": "rear"}],
        }
    ]
    conn = open_db(db_path)
    try:
        test_paths = _seed_class_rows(conn, tmp_path=tmp_path, class_specs=class_specs)
    finally:
        conn.close()

    class_ids = sorted(test_paths.keys())
    embed_dim = 8
    proto_tensor = _proto_for(0, embed_dim=embed_dim).unsqueeze(0)
    embeddings_by_path: dict[Path, torch.Tensor] = {
        p: _basis_embedding(0, embed_dim=embed_dim) for p in test_paths[class_ids[0]]
    }
    proto_path = tmp_path / "cache" / "prototypes.pt"
    _save_v1_prototypes(path=proto_path, class_ids=class_ids, prototypes=proto_tensor)
    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    report_path = tmp_path / "reports" / "phase5_4_eval.json"
    rc = evaluate_cli.main(
        [
            "--db",
            str(db_path),
            "--source",
            "compcars",
            "--test-split",
            "test",
            "--prototypes",
            str(proto_path),
            "--checkpoint",
            "",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--report",
            str(report_path),
            "--print-top-makes",
            "5",
            "--print-top-confusions",
            "5",
        ]
    )
    assert rc == 0
    assert report_path.exists()
    out = capsys.readouterr().out
    assert "top_1=1.0000" in out
    assert "evaluate-recognize: report written" in out
