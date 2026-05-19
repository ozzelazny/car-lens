"""Tests for the Phase 5.3 view-classifier head trainer.

The OpenCLIP backbone is replaced with a torch-based stub via
``sys.modules`` so the tests never download real weights and never
touch the network. ``torch`` itself is a runtime dependency so we
import it normally (skip-if-missing).

Strategy
--------

* Seed a tiny SQLite DB whose ``images`` table has rows with the full
  set of view labels (front / rear / side / three-quarter-front /
  three-quarter-rear / interior / detail / non-car).
* Install a stub ``open_clip`` module whose
  ``create_model_and_transforms`` returns a trainable ``StubBackbone``
  + a simple resize + ToTensor preprocess so the cache-features pass
  produces real tensors with non-trivial structure.
* Drive :func:`train_view_classifier` for 2 epochs and check the
  returned :class:`CheckpointPayload` shape + report counts.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from car_lense_engine.db import Image, Listing, images, listings, open_db

torch = pytest.importorskip("torch")
pytest.importorskip("torch.nn")
pytest.importorskip("torchvision")


# --------------------------------------------------------------- stub backbone


class _StubVisual(torch.nn.Module):  # type: ignore[misc, name-defined]
    """A tiny trainable image encoder; pools spatial dims to RGB then projects."""

    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(3, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3))
        return self.proj(pooled)


class _StubModel(torch.nn.Module):  # type: ignore[misc, name-defined]
    """OpenCLIP-shaped wrapper around :class:`_StubVisual`."""

    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        self.visual = _StubVisual(embed_dim=embed_dim)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


def _stub_preprocess(img: Any) -> torch.Tensor:
    """Convert a PIL image to a (3, 8, 8) float tensor in [0, 1]."""
    from torchvision import transforms as T  # noqa: PLC0415

    pipeline = T.Compose([T.Resize((8, 8)), T.ToTensor()])
    return pipeline(img)  # type: ignore[no-any-return]


class _StubOpenClip:
    """Drop-in replacement for the ``open_clip`` module."""

    def __init__(self) -> None:
        self.last_call: dict[str, Any] | None = None

    def create_model_and_transforms(
        self,
        model_name: str,
        *,
        pretrained: str,
        device: str,
    ) -> tuple[_StubModel, None, Any]:
        self.last_call = {
            "model_name": model_name,
            "pretrained": pretrained,
            "device": device,
        }
        return _StubModel(embed_dim=16), None, _stub_preprocess


@pytest.fixture
def stub_open_clip(monkeypatch: pytest.MonkeyPatch) -> _StubOpenClip:
    stub = _StubOpenClip()
    monkeypatch.setitem(sys.modules, "open_clip", stub)
    return stub


# --------------------------------------------------------------- DB fixtures


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


def _make_image_file(path: Path, color: tuple[int, int, int] = (128, 128, 128)) -> Path:
    from PIL import Image as PILImage  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (8, 8), color=color).save(path, format="JPEG")
    return path


def _seed_view_row(
    conn: sqlite3.Connection,
    *,
    tmp_path: Path,
    listing_id: str,
    image_id: str,
    view: str | None,
    view_score: float | None,
    split: str | None,
    source: str = "compcars",
    color: tuple[int, int, int] = (50, 50, 50),
) -> Path:
    """Insert one listing + one image with the requested view metadata."""
    url = f"{source}://{listing_id}"
    listings.insert_listing(
        conn,
        Listing(
            listing_id=listing_id,
            source=source,  # type: ignore[arg-type]
            url=url,
            year=2015,
            make="Honda",
            model="Civic",
            canonical_make="Honda",
            canonical_model="Civic",
            generation_year=2012,
        ),
    )
    img_path = tmp_path / "imgs" / f"{image_id}.jpg"
    _make_image_file(img_path, color=color)
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
            "UPDATE images SET view = ?, view_score = ?, split = ? WHERE image_id = ?",
            (view, view_score, split, image_id),
        )
    return img_path


# --------------------------------------------------------------- collapse_view


def test_collapse_view_handles_all_view_names() -> None:
    """Exhaustive map check for every supported view label."""
    from car_lense_engine.training.view_classifier import collapse_view

    expected = {
        "front": 0,
        "rear": 1,
        "side": 2,
        "three-quarter-front": 3,
        "three-quarter-rear": 4,
        "interior": 5,
        "detail": 5,
        "non-car": 5,
    }
    for view, idx in expected.items():
        assert collapse_view(view) == idx, f"{view!r} -> {idx}"


def test_collapse_view_unknown_raises() -> None:
    from car_lense_engine.training.view_classifier import collapse_view

    with pytest.raises(KeyError):
        collapse_view("engine-bay")


def test_collapse_view_to_binary_handles_all_view_names() -> None:
    """Exhaustive map check for every supported view label in binary mode."""
    from car_lense_engine.training.view_classifier import collapse_view_to_binary

    expected = {
        "front": 0,
        "rear": 0,
        "side": 0,
        "three-quarter-front": 0,
        "three-quarter-rear": 0,
        "interior": 1,
        "detail": 1,
        "non-car": 1,
    }
    for view, idx in expected.items():
        assert collapse_view_to_binary(view) == idx, f"{view!r} -> {idx}"


def test_collapse_view_to_binary_raises_on_unknown() -> None:
    from car_lense_engine.training.view_classifier import collapse_view_to_binary

    with pytest.raises(KeyError):
        collapse_view_to_binary("engine-bay")


# --------------------------------------------------------------- dataset query


def test_build_view_classifier_dataset_filters_low_confidence(
    db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    from car_lense_engine.training.view_classifier import build_view_classifier_dataset

    _seed_view_row(
        db,
        tmp_path=tmp_path,
        listing_id="l1",
        image_id="i1" * 16,
        view="front",
        view_score=0.55,
        split="train",
    )
    _seed_view_row(
        db,
        tmp_path=tmp_path,
        listing_id="l2",
        image_id="i2" * 16,
        view="front",
        view_score=0.65,
        split="train",
    )

    rows = build_view_classifier_dataset(db, split="train", min_view_score=0.6)
    assert len(rows) == 1
    path, cls_idx = rows[0]
    assert path.name == ("i2" * 16) + ".jpg"
    assert cls_idx == 0


def test_build_view_classifier_dataset_split_filter(
    db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    from car_lense_engine.training.view_classifier import build_view_classifier_dataset

    _seed_view_row(
        db,
        tmp_path=tmp_path,
        listing_id="l-train",
        image_id="a" * 32,
        view="front",
        view_score=0.9,
        split="train",
    )
    _seed_view_row(
        db,
        tmp_path=tmp_path,
        listing_id="l-val",
        image_id="b" * 32,
        view="rear",
        view_score=0.9,
        split="val",
    )
    _seed_view_row(
        db,
        tmp_path=tmp_path,
        listing_id="l-test",
        image_id="c" * 32,
        view="side",
        view_score=0.9,
        split="test",
    )

    train_rows = build_view_classifier_dataset(db, split="train", min_view_score=0.6)
    val_rows = build_view_classifier_dataset(db, split="val", min_view_score=0.6)
    test_rows = build_view_classifier_dataset(db, split="test", min_view_score=0.6)

    assert [r[1] for r in train_rows] == [0]
    assert [r[1] for r in val_rows] == [1]
    assert [r[1] for r in test_rows] == [2]


def test_build_view_classifier_dataset_excludes_null_view(
    db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    from car_lense_engine.training.view_classifier import build_view_classifier_dataset

    _seed_view_row(
        db,
        tmp_path=tmp_path,
        listing_id="l1",
        image_id="d" * 32,
        view=None,
        view_score=None,
        split="train",
    )
    _seed_view_row(
        db,
        tmp_path=tmp_path,
        listing_id="l2",
        image_id="e" * 32,
        view="three-quarter-front",
        view_score=0.8,
        split="train",
    )

    rows = build_view_classifier_dataset(db, split="train", min_view_score=0.6)
    assert len(rows) == 1
    assert rows[0][1] == 3  # three-quarter-front -> 3


def test_build_view_classifier_dataset_collapses_non_exterior(
    db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    """interior / detail / non-car all map to class index 5."""
    from car_lense_engine.training.view_classifier import build_view_classifier_dataset

    for i, view in enumerate(("interior", "detail", "non-car")):
        _seed_view_row(
            db,
            tmp_path=tmp_path,
            listing_id=f"l-{i}",
            image_id=str(i) * 32,
            view=view,
            view_score=0.9,
            split="train",
        )
    rows = build_view_classifier_dataset(db, split="train", min_view_score=0.6)
    assert {idx for _path, idx in rows} == {5}
    assert len(rows) == 3


def test_build_view_classifier_dataset_binary_includes_non_exterior(
    db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    """Binary mode pulls exterior rows from the split AND non-exterior
    rows regardless of ``images.split`` (NULL split is fine)."""
    from car_lense_engine.training.view_classifier import build_view_classifier_dataset

    # Exterior rows split=train should be included.
    _seed_view_row(
        db,
        tmp_path=tmp_path,
        listing_id="l-ext1",
        image_id="e" * 32,
        view="front",
        view_score=0.9,
        split="train",
    )
    _seed_view_row(
        db,
        tmp_path=tmp_path,
        listing_id="l-ext2",
        image_id="f" * 32,
        view="three-quarter-front",
        view_score=0.9,
        split="train",
    )
    # An exterior row with split=val should NOT show up when we request train.
    _seed_view_row(
        db,
        tmp_path=tmp_path,
        listing_id="l-ext-val",
        image_id="9" * 32,
        view="rear",
        view_score=0.9,
        split="val",
    )

    # Pick image_ids that land in train bucket for non-exterior rows.
    import hashlib

    candidates: list[tuple[str, str]] = []
    for view in ("interior", "detail", "non-car"):
        for n in range(1024):
            iid = f"{view}-{n:0>32}"
            bucket = hashlib.sha1(iid.encode("utf-8"), usedforsecurity=False).digest()[0] % 10
            if bucket < 8:  # train
                candidates.append((iid, view))
                break

    for i, (iid, view) in enumerate(candidates):
        _seed_view_row(
            db,
            tmp_path=tmp_path,
            listing_id=f"l-ne-{i}",
            image_id=iid,
            view=view,
            view_score=0.9,
            split=None,  # NULL split: only binary mode picks these up
        )

    rows = build_view_classifier_dataset(db, split="train", min_view_score=0.6, binary=True)
    labels = [idx for _path, idx in rows]
    assert labels.count(0) == 2  # two exterior train rows
    assert labels.count(1) == 3  # interior + detail + non-car


def test_build_view_classifier_dataset_binary_deterministic_split_for_non_exterior(
    db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    """Same image_ids -> same split assignment across two calls."""
    from car_lense_engine.training.view_classifier import build_view_classifier_dataset

    for i, view in enumerate(("interior", "detail", "non-car", "interior", "detail")):
        _seed_view_row(
            db,
            tmp_path=tmp_path,
            listing_id=f"l-ne-{i}",
            image_id=f"deterministic-{i:0>32}",
            view=view,
            view_score=0.9,
            split=None,
        )

    rows_a = build_view_classifier_dataset(db, split="train", min_view_score=0.6, binary=True)
    rows_b = build_view_classifier_dataset(db, split="train", min_view_score=0.6, binary=True)
    assert [str(p) for p, _ in rows_a] == [str(p) for p, _ in rows_b]

    # And the train rows + val rows + test rows partition the non-exterior set.
    train = build_view_classifier_dataset(db, split="train", min_view_score=0.6, binary=True)
    val = build_view_classifier_dataset(db, split="val", min_view_score=0.6, binary=True)
    test = build_view_classifier_dataset(db, split="test", min_view_score=0.6, binary=True)
    seen = sorted(str(p) for p, _ in train + val + test)
    # All 5 non-exterior rows should appear in exactly one split.
    assert len(seen) == 5
    assert len(set(seen)) == 5


def test_build_view_classifier_dataset_skips_unknown_view(
    db: sqlite3.Connection,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unexpected view label is dropped and surfaces via a WARNING log."""
    from car_lense_engine.training.view_classifier import build_view_classifier_dataset

    _seed_view_row(
        db,
        tmp_path=tmp_path,
        listing_id="l-unknown",
        image_id="0" * 32,
        view="engine-bay",
        view_score=0.9,
        split="train",
    )
    _seed_view_row(
        db,
        tmp_path=tmp_path,
        listing_id="l-known",
        image_id="1" * 32,
        view="rear",
        view_score=0.9,
        split="train",
    )
    with caplog.at_level("WARNING", logger="car_lense_engine.training.view_classifier"):
        rows = build_view_classifier_dataset(db, split="train", min_view_score=0.6)
    assert len(rows) == 1
    assert rows[0][1] == 1


# --------------------------------------------------------------- compute_class_weights


def test_compute_class_weights_inverse_sqrt_freq_normalized() -> None:
    from car_lense_engine.training.view_classifier import compute_class_weights

    weights = compute_class_weights([100, 100, 100, 100, 100, 100])
    assert all(abs(w - 1.0) < 1e-6 for w in weights)


def test_compute_class_weights_inverse_relation() -> None:
    """A rare class must receive a larger weight than a common class."""
    from car_lense_engine.training.view_classifier import compute_class_weights

    weights = compute_class_weights([10000, 10000, 10000, 10000, 100, 10])
    # Mean should be 1.0 (renormalized).
    assert abs(sum(weights) / len(weights) - 1.0) < 1e-6
    # Rare classes get larger weights.
    assert weights[4] > weights[0]
    assert weights[5] > weights[4]


def test_compute_class_weights_zero_count_yields_zero_weight() -> None:
    from car_lense_engine.training.view_classifier import compute_class_weights

    weights = compute_class_weights([100, 0, 100, 100, 100, 100])
    assert weights[1] == 0.0
    # The non-zero classes get bumped up to compensate.
    assert all(w > 0.0 for i, w in enumerate(weights) if i != 1)


# --------------------------------------------------------------- training smoke


def _seed_balanced_dataset(
    conn: sqlite3.Connection,
    tmp_path: Path,
    *,
    per_class_train: int,
    per_class_val: int,
) -> None:
    """Seed N train + M val images per exterior view class.

    Each class gets a distinct solid color so the cached features land
    in a distinct region of the embedding space and the head can in
    principle separate them.
    """
    counter = 0
    for cls_idx, view in enumerate(
        ("front", "rear", "side", "three-quarter-front", "three-quarter-rear")
    ):
        color = (
            (cls_idx * 41 + 10) % 256,
            (cls_idx * 83 + 20) % 256,
            (cls_idx * 167 + 30) % 256,
        )
        for split, n in (("train", per_class_train), ("val", per_class_val)):
            for _ in range(n):
                counter += 1
                _seed_view_row(
                    conn,
                    tmp_path=tmp_path,
                    listing_id=f"l-{counter}",
                    image_id=f"{counter:064d}",
                    view=view,
                    view_score=0.9,
                    split=split,
                    color=color,
                )


def test_train_view_classifier_smoke(
    stub_open_clip: _StubOpenClip,
    tmp_path: Path,
    db_path: Path,
) -> None:
    """End-to-end: 2 epochs over a tiny synthetic DB, check payload shape."""
    from car_lense_engine.training.view_classifier import (
        VIEW_CLASS_NAMES,
        ViewClassifierConfig,
        train_view_classifier,
    )

    conn = open_db(db_path)
    try:
        _seed_balanced_dataset(conn, tmp_path, per_class_train=4, per_class_val=2)
    finally:
        conn.close()

    config = ViewClassifierConfig(
        model_name="StubMobileCLIP",
        pretrained="stub",
        device="cpu",
        epochs=2,
        lr=1e-1,
        batch_size=8,
        backbone_batch_size=4,
        min_view_score=0.6,
        head_arch="linear",
        backbone_checkpoint=None,
        seed=123,
    )

    conn = open_db(db_path)
    try:
        payload = train_view_classifier(conn=conn, config=config)
    finally:
        conn.close()

    # Class names are the canonical 6-class vocabulary.
    assert payload.class_names == list(VIEW_CLASS_NAMES)

    # Report bookkeeping matches what we seeded (5 exterior classes ×
    # {4 train, 2 val}).
    assert payload.report.n_train == 5 * 4
    assert payload.report.n_val == 5 * 2
    assert len(payload.report.per_epoch) == 2
    assert payload.report.embed_dim == 16
    assert 0 <= payload.report.best_epoch < 2

    # Confusion matrix shape: 6x6 (even though only 5 classes had data).
    cm = payload.val_confusion_matrix
    assert tuple(cm.shape) == (6, 6)

    # head_state_dict has the expected linear-layer keys.
    assert "weight" in payload.head_state_dict
    assert "bias" in payload.head_state_dict
    assert tuple(payload.head_state_dict["weight"].shape) == (6, 16)
    assert tuple(payload.head_state_dict["bias"].shape) == (6,)

    # image_encoder_state_dict carries the stub backbone weights.
    assert "proj.weight" in payload.image_encoder_state_dict
    assert "proj.bias" in payload.image_encoder_state_dict

    # config dict is self-describing.
    assert payload.config["head_arch"] == "linear"
    assert payload.config["embed_dim"] == 16
    assert payload.config["class_weights_strategy"] == "inverse_sqrt_freq"
    assert payload.config["model_name"] == "StubMobileCLIP"
    assert payload.config["best_epoch"] == payload.report.best_epoch


def test_train_view_classifier_smoke_binary(
    stub_open_clip: _StubOpenClip,
    tmp_path: Path,
    db_path: Path,
) -> None:
    """Binary mode: 2 epochs over a tiny synthetic DB, check class_names + head shape."""
    from car_lense_engine.training.view_classifier import (
        BINARY_CLASS_NAMES,
        ViewClassifierConfig,
        train_view_classifier,
    )

    conn = open_db(db_path)
    try:
        _seed_balanced_dataset(conn, tmp_path, per_class_train=4, per_class_val=2)
        # Seed a few non-exterior rows too. Their split is NULL but the
        # binary builder derives split deterministically from image_id.
        for i, view in enumerate(("interior", "detail", "non-car") * 3):
            _seed_view_row(
                conn,
                tmp_path=tmp_path,
                listing_id=f"ne-{i}",
                image_id=f"binsmoke-{i:0>32}",
                view=view,
                view_score=0.9,
                split=None,
                color=(10 + i * 5, 200, 30),
            )
    finally:
        conn.close()

    config = ViewClassifierConfig(
        model_name="StubMobileCLIP",
        pretrained="stub",
        device="cpu",
        epochs=2,
        lr=1e-1,
        batch_size=8,
        backbone_batch_size=4,
        min_view_score=0.6,
        head_arch="linear",
        backbone_checkpoint=None,
        seed=123,
        binary=True,
    )

    conn = open_db(db_path)
    try:
        payload = train_view_classifier(conn=conn, config=config)
    finally:
        conn.close()

    # Class names are the binary vocabulary.
    assert payload.class_names == list(BINARY_CLASS_NAMES)

    # head out_features = 2.
    assert tuple(payload.head_state_dict["weight"].shape) == (2, 16)
    assert tuple(payload.head_state_dict["bias"].shape) == (2,)

    # Confusion matrix is 2x2.
    assert tuple(payload.val_confusion_matrix.shape) == (2, 2)


def test_train_view_classifier_mlp_head(
    stub_open_clip: _StubOpenClip,
    tmp_path: Path,
    db_path: Path,
) -> None:
    """MLP head exposes a hidden layer + final projection (4 state-dict keys)."""
    from car_lense_engine.training.view_classifier import (
        ViewClassifierConfig,
        train_view_classifier,
    )

    conn = open_db(db_path)
    try:
        _seed_balanced_dataset(conn, tmp_path, per_class_train=3, per_class_val=1)
    finally:
        conn.close()

    config = ViewClassifierConfig(
        model_name="StubMobileCLIP",
        pretrained="stub",
        device="cpu",
        epochs=1,
        lr=1e-1,
        batch_size=8,
        backbone_batch_size=4,
        min_view_score=0.6,
        head_arch="mlp",
        backbone_checkpoint=None,
        seed=7,
    )
    conn = open_db(db_path)
    try:
        payload = train_view_classifier(conn=conn, config=config)
    finally:
        conn.close()
    keys = sorted(payload.head_state_dict.keys())
    # Sequential MLP: 0=Linear(16->256), 3=Linear(256->6); ReLU/Dropout have no params.
    assert keys == ["0.bias", "0.weight", "3.bias", "3.weight"]
    assert tuple(payload.head_state_dict["0.weight"].shape) == (256, 16)
    assert tuple(payload.head_state_dict["3.weight"].shape) == (6, 256)


def test_train_view_classifier_empty_raises(
    stub_open_clip: _StubOpenClip,
    tmp_path: Path,
    db_path: Path,
) -> None:
    from car_lense_engine.training.view_classifier import (
        ViewClassifierConfig,
        train_view_classifier,
    )

    open_db(db_path).close()  # migrations only, no data
    config = ViewClassifierConfig(
        model_name="StubMobileCLIP",
        pretrained="stub",
        device="cpu",
        epochs=1,
        lr=1e-2,
        batch_size=4,
        backbone_batch_size=2,
        min_view_score=0.6,
        head_arch="linear",
        backbone_checkpoint=None,
        seed=42,
    )
    conn = open_db(db_path)
    try:
        with pytest.raises(ValueError, match="no train rows"):
            train_view_classifier(conn=conn, config=config)
    finally:
        conn.close()


# --------------------------------------------------------------- CLI


def test_cli_rejects_missing_db(tmp_path: Path) -> None:
    from car_lense_engine.training import view_classifier_cli

    missing = tmp_path / "nope.sqlite"
    with pytest.raises(SystemExit) as excinfo:
        view_classifier_cli.main(["--db", str(missing)])
    assert excinfo.value.code == 2


def test_cli_rejects_missing_backbone_checkpoint(tmp_path: Path, db_path: Path) -> None:
    from car_lense_engine.training import view_classifier_cli

    open_db(db_path).close()  # migrations only, but DB exists
    missing = tmp_path / "no_such_checkpoint.pt"
    with pytest.raises(SystemExit) as excinfo:
        view_classifier_cli.main(
            [
                "--db",
                str(db_path),
                "--backbone-checkpoint",
                str(missing),
            ]
        )
    assert excinfo.value.code == 2


def test_cli_runs_and_writes_outputs(
    stub_open_clip: _StubOpenClip,
    tmp_path: Path,
    db_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from car_lense_engine.training import view_classifier_cli
    from car_lense_engine.training.view_classifier import ViewClassifierReport

    conn = open_db(db_path)
    try:
        _seed_balanced_dataset(conn, tmp_path, per_class_train=2, per_class_val=1)
    finally:
        conn.close()

    output = tmp_path / "ckpts" / "view_classifier_v1.pt"
    report_path = tmp_path / "reports" / "p5_3_view.json"
    rc = view_classifier_cli.main(
        [
            "--db",
            str(db_path),
            "--backbone-checkpoint",
            "none",
            "--model",
            "StubMobileCLIP",
            "--pretrained",
            "stub",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--lr",
            "0.1",
            "--batch-size",
            "4",
            "--backbone-batch-size",
            "2",
            "--head-arch",
            "linear",
            "--output",
            str(output),
            "--report",
            str(report_path),
        ]
    )
    assert rc == 0
    assert output.exists()
    assert report_path.exists()

    report = ViewClassifierReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert report.n_train == 5 * 2
    assert report.n_val == 5 * 1
    assert report.class_names == [
        "front",
        "rear",
        "side",
        "three-quarter-front",
        "three-quarter-rear",
        "non-exterior",
    ]
    assert report.checkpoint_path == str(output)

    # The .pt round-trips and carries all the documented keys.
    payload = torch.load(output, map_location="cpu", weights_only=False)
    for key in (
        "head_state_dict",
        "image_encoder_state_dict",
        "class_names",
        "config",
        "val_confusion_matrix",
    ):
        assert key in payload
    assert payload["class_names"] == list(report.class_names)
    assert payload["config"]["head_arch"] == "linear"

    out = capsys.readouterr().out
    assert "train-view-classifier:" in out
    assert "best_top1=" in out
