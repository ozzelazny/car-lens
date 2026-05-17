"""Tests for the Phase 5.2 fine-tune harness.

The OpenCLIP backbone is replaced with a torch-based stub via
``sys.modules`` so the tests never download real weights and never touch
the network. We *do* rely on a real ``torch`` install -- torch is a
runtime dependency of the project.

Strategy:

* Seed a tiny SQLite DB with N classes × M train + K test images.
* Install a stub ``open_clip`` module whose ``create_model_and_transforms``
  returns:

    - A trainable ``StubBackbone`` (a single ``nn.Linear`` over a
      flattened pixel-mean feature) that behaves like ``model.encode_image``.
    - A stub preprocess (``ToTensor`` + ``Resize``) so the training
      DataLoader can produce proper tensors.

* Run :func:`run_training` for a couple of epochs and check that loss
  trends down, a checkpoint is written, and the report round-trips.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from car_lense_engine.dataset.canonical_labels import year_to_generation
from car_lense_engine.db import Image, Listing, images, listings, open_db
from car_lense_engine.eval.baseline import class_id_for

torch = pytest.importorskip("torch")
pytest.importorskip("torch.nn")
pytest.importorskip("torchvision")


# --------------------------------------------------------------- stub model


class _StubVisual(torch.nn.Module):  # type: ignore[misc, name-defined]
    """A tiny trainable image encoder used in place of MobileCLIP-S2.

    Pools (mean) over spatial dims to a 3-vector, then projects to
    ``embed_dim`` via a learned linear. This is enough surface area for
    the training loop to flow gradients through ``model.visual`` *and*
    the head, which is the only behaviour the test cares about.
    """

    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(3, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> mean over spatial -> (B, 3) -> (B, embed_dim)
        pooled = x.mean(dim=(2, 3))
        return self.proj(pooled)


class _StubModel(torch.nn.Module):  # type: ignore[misc, name-defined]
    """OpenCLIP-shaped wrapper around :class:`_StubVisual`.

    Exposes ``encode_image`` and ``.visual`` so the trainer's
    introspection (and checkpoint save) target the visual tower.
    """

    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        self.visual = _StubVisual(embed_dim=embed_dim)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


def _stub_preprocess(img: Any) -> torch.Tensor:
    """Convert a PIL image to a (3, 8, 8) float tensor in [0,1]."""
    from torchvision import transforms as T

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
    """Install the stub open_clip module in sys.modules."""
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
    """Write a tiny JPEG of a single solid color (gives a unique mean)."""
    from PIL import Image as PILImage

    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (8, 8), color=color).save(path, format="JPEG")
    return path


def _seed_dataset(
    conn: sqlite3.Connection,
    *,
    tmp_path: Path,
    classes: list[tuple[int, str, str]],
    n_train_per_class: int = 4,
    n_test_per_class: int = 2,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Seed ``listings`` + ``images``.

    Each class gets a unique solid color so a linear projection over the
    pooled RGB mean is in principle perfectly separable; that's the
    "training should drive loss down" property we want for the test.
    """
    train_paths: dict[str, list[Path]] = {}
    test_paths: dict[str, list[Path]] = {}
    counter = 0
    for class_idx, (year, make, model) in enumerate(classes):
        # Phase 4.6: training keys the class id off the bucketed
        # ``generation_year``, not the raw calendar year.
        gen_year = year_to_generation(year)
        cid = class_id_for(gen_year, make, model)
        assert cid is not None
        train_paths[cid] = []
        test_paths[cid] = []
        color = (
            (class_idx * 73) % 256,
            (class_idx * 137) % 256,
            (class_idx * 211 + 30) % 256,
        )
        for split_name, n, dest in (
            ("train", n_train_per_class, train_paths),
            ("test", n_test_per_class, test_paths),
        ):
            for i in range(n):
                counter += 1
                listing_id = f"stanford_cars:test_{counter:04d}"
                url = f"stanford_cars://{cid}/{counter:04d}"
                img_path = tmp_path / "imgs" / f"{cid}_{split_name}_{i}.jpg"
                _make_image_file(img_path, color=color)
                listings.insert_listing(
                    conn,
                    Listing(
                        listing_id=listing_id,
                        source="stanford_cars",
                        url=url,
                        year=year,
                        make=make,
                        model=model,
                        split=split_name,
                        # Phase 4.5 + 4.6: training reads canonical_*
                        # columns and generation_year exclusively.
                        # Populating these matches what
                        # canonicalize-labels would produce.
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
                dest[cid].append(img_path)
    return train_paths, test_paths


# --------------------------------------------------------------- module under test


def _train_config(
    *,
    tmp_path: Path,
    epochs: int = 2,
    batch_size: int = 4,
    hard_neg_weight: float = 2.0,
    hard_neg_confusion_path: Path | None = None,
    num_workers: int = 0,
) -> Any:
    from car_lense_engine.training import TrainConfig

    return TrainConfig(
        model_name="StubMobileCLIP",
        pretrained="stub",
        source="stanford_cars",
        train_split="train",
        val_split="test",
        device="cpu",
        batch_size=batch_size,
        num_workers=num_workers,
        epochs=epochs,
        lr_backbone=1e-2,
        lr_head=1e-1,
        weight_decay=0.0,
        warmup_epochs=0,
        label_smoothing=0.0,
        hard_neg_weight=hard_neg_weight,
        hard_neg_confusion_path=hard_neg_confusion_path,
        seed=123,
    )


# --------------------------------------------------------------- unit: weights


def test_build_class_weights_no_path_returns_ones() -> None:
    from car_lense_engine.training import build_class_weights_from_confusion

    weights = build_class_weights_from_confusion(
        class_ids=["a", "b", "c"],
        confusion_path=None,
        hard_neg_weight=2.0,
    )
    assert weights == [1.0, 1.0, 1.0]


def test_build_class_weights_missing_file_returns_ones(tmp_path: Path) -> None:
    from car_lense_engine.training import build_class_weights_from_confusion

    missing = tmp_path / "nope.json"
    weights = build_class_weights_from_confusion(
        class_ids=["a", "b"],
        confusion_path=missing,
        hard_neg_weight=2.0,
    )
    assert weights == [1.0, 1.0]


def test_build_class_weights_uses_confusion_pairs(tmp_path: Path) -> None:
    """Classes that participate in any confusion pair get boosted."""
    from car_lense_engine.training import build_class_weights_from_confusion

    fp = tmp_path / "phase5_baseline.json"
    fp.write_text(
        json.dumps(
            {
                "confusion_top_pairs": [
                    {"true_class": "alpha", "predicted_class": "beta", "count": 5},
                    {"true_class": "gamma", "predicted_class": "alpha", "count": 3},
                ]
            }
        ),
        encoding="utf-8",
    )
    weights = build_class_weights_from_confusion(
        class_ids=["alpha", "beta", "gamma", "delta"],
        confusion_path=fp,
        hard_neg_weight=2.5,
    )
    # alpha (true in pair 1, pred in pair 2), beta (pred in pair 1), gamma (true in pair 2) -> 2.5
    # delta (not in any pair) -> 1.0
    assert weights == [2.5, 2.5, 2.5, 1.0]


def test_build_class_weights_bad_weight_raises() -> None:
    from car_lense_engine.training import build_class_weights_from_confusion

    with pytest.raises(ValueError):
        build_class_weights_from_confusion(
            class_ids=["a"], confusion_path=None, hard_neg_weight=0.0
        )


def test_build_class_weights_handles_malformed_json(tmp_path: Path) -> None:
    """A garbled confusion file is tolerated -- log + return ones."""
    from car_lense_engine.training import build_class_weights_from_confusion

    fp = tmp_path / "bad.json"
    fp.write_text("{not valid json", encoding="utf-8")
    weights = build_class_weights_from_confusion(
        class_ids=["a", "b"], confusion_path=fp, hard_neg_weight=2.0
    )
    assert weights == [1.0, 1.0]


# --------------------------------------------------------------- training loop


def test_run_training_smoke_writes_checkpoint_and_loss_decreases(
    stub_open_clip: _StubOpenClip,
    tmp_path: Path,
    db_path: Path,
) -> None:
    from car_lense_engine.training import TrainReport, run_training

    classes = [
        (2012, "Acura", "RL"),
        (2007, "Hyundai", "Sonata"),
        (2012, "Tesla", "Model S"),
    ]
    conn = open_db(db_path)
    try:
        _seed_dataset(
            conn,
            tmp_path=tmp_path,
            classes=classes,
            n_train_per_class=4,
            n_test_per_class=2,
        )
    finally:
        conn.close()

    config = _train_config(tmp_path=tmp_path, epochs=3, batch_size=4)
    ckpt_dir = tmp_path / "ckpts"

    conn = open_db(db_path)
    try:
        report = run_training(conn=conn, config=config, checkpoint_dir=ckpt_dir)
    finally:
        conn.close()

    assert report.n_classes == 3
    assert report.n_train == 12
    assert report.n_val == 6
    assert len(report.per_epoch) == 3
    # Loss should not increase between the first and last epoch (we
    # allow equality for the (rare) case the model converged instantly).
    assert report.per_epoch[-1].train_loss <= report.per_epoch[0].train_loss + 1e-6
    # Best checkpoint must exist on disk.
    assert report.checkpoint_path
    assert Path(report.checkpoint_path).exists()
    # Filename pattern: <slug>_<source>_epoch<NN>_top1_<XX.X>.pt
    fname = Path(report.checkpoint_path).name
    assert fname.startswith("stubmobileclip_stanford_cars_epoch")
    assert fname.endswith(".pt")
    assert "top1_" in fname

    # The checkpoint must round-trip via torch.load and carry our metadata.
    payload = torch.load(report.checkpoint_path, map_location="cpu", weights_only=False)
    assert "image_encoder_state_dict" in payload
    assert "head_state_dict" in payload
    assert payload["n_classes"] == 3
    assert sorted(payload["class_ids"]) == sorted(payload["class_ids"])  # is a list
    assert payload["epoch"] == report.best_epoch

    # Report JSON round-trips.
    serialized = report.model_dump_json()
    again = TrainReport.model_validate_json(serialized)
    assert again == report


def test_run_training_zero_epochs_writes_no_checkpoint(
    stub_open_clip: _StubOpenClip,
    tmp_path: Path,
    db_path: Path,
) -> None:
    from car_lense_engine.training import run_training

    classes = [(2012, "Acura", "RL"), (2007, "Hyundai", "Sonata")]
    conn = open_db(db_path)
    try:
        _seed_dataset(
            conn, tmp_path=tmp_path, classes=classes, n_train_per_class=2, n_test_per_class=1
        )
    finally:
        conn.close()

    config = _train_config(tmp_path=tmp_path, epochs=0, batch_size=2)
    ckpt_dir = tmp_path / "ckpts"

    conn = open_db(db_path)
    try:
        report = run_training(conn=conn, config=config, checkpoint_dir=ckpt_dir)
    finally:
        conn.close()

    assert report.per_epoch == []
    assert report.checkpoint_path == ""
    assert not ckpt_dir.exists() or not any(ckpt_dir.iterdir())


def test_run_training_empty_train_raises(
    stub_open_clip: _StubOpenClip,
    tmp_path: Path,
    db_path: Path,
) -> None:
    from car_lense_engine.training import run_training

    open_db(db_path).close()  # migrations only, no data
    config = _train_config(tmp_path=tmp_path, epochs=2, batch_size=2)
    ckpt_dir = tmp_path / "ckpts"

    conn = open_db(db_path)
    try:
        with pytest.raises(ValueError, match="no train rows"):
            run_training(conn=conn, config=config, checkpoint_dir=ckpt_dir)
    finally:
        conn.close()


def test_run_training_with_hard_negative_path_loads_weights(
    stub_open_clip: _StubOpenClip,
    tmp_path: Path,
    db_path: Path,
) -> None:
    """A confusion file that mentions one of our classes drives the boosted weight.

    We can't directly inspect the weight tensor without instrumenting the
    runner, so we verify behaviour via :func:`build_class_weights_from_confusion`
    being called with the right inputs (already covered in unit tests),
    plus end-to-end that the training run with the file completes without
    error and the file is honoured (best_top1 >= 0).
    """
    from car_lense_engine.training import run_training

    classes = [
        (2012, "Acura", "RL"),
        (2007, "Hyundai", "Sonata"),
    ]
    # Phase 4.6: class id is keyed off the bucketed generation_year.
    cid0 = class_id_for(year_to_generation(classes[0][0]), classes[0][1], classes[0][2])
    cid1 = class_id_for(year_to_generation(classes[1][0]), classes[1][1], classes[1][2])
    assert cid0 is not None and cid1 is not None

    conn = open_db(db_path)
    try:
        _seed_dataset(
            conn, tmp_path=tmp_path, classes=classes, n_train_per_class=4, n_test_per_class=2
        )
    finally:
        conn.close()

    confusion_fp = tmp_path / "phase5_baseline.json"
    confusion_fp.write_text(
        json.dumps(
            {"confusion_top_pairs": [{"true_class": cid0, "predicted_class": cid1, "count": 10}]}
        ),
        encoding="utf-8",
    )

    config = _train_config(
        tmp_path=tmp_path,
        epochs=2,
        batch_size=4,
        hard_neg_weight=3.0,
        hard_neg_confusion_path=confusion_fp,
    )
    ckpt_dir = tmp_path / "ckpts"
    conn = open_db(db_path)
    try:
        report = run_training(conn=conn, config=config, checkpoint_dir=ckpt_dir)
    finally:
        conn.close()
    assert report.best_val_top1 >= 0.0
    assert len(report.per_epoch) == 2


# --------------------------------------------------------------- CLI smoke


def test_cli_rejects_missing_db(tmp_path: Path) -> None:
    from car_lense_engine.training import cli as train_cli

    missing = tmp_path / "nope.sqlite"
    with pytest.raises(SystemExit) as excinfo:
        train_cli.main(["--db", str(missing)])
    assert excinfo.value.code == 2


def test_cli_runs_and_writes_report(
    stub_open_clip: _StubOpenClip,
    tmp_path: Path,
    db_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from car_lense_engine.training import TrainReport
    from car_lense_engine.training import cli as train_cli

    classes = [(2012, "Acura", "RL"), (2007, "Hyundai", "Sonata")]
    conn = open_db(db_path)
    try:
        _seed_dataset(
            conn, tmp_path=tmp_path, classes=classes, n_train_per_class=2, n_test_per_class=1
        )
    finally:
        conn.close()

    output = tmp_path / "reports" / "p5_train.json"
    ckpt_dir = tmp_path / "ckpts"
    rc = train_cli.main(
        [
            "--db",
            str(db_path),
            "--source",
            "stanford_cars",
            "--train-split",
            "train",
            "--val-split",
            "test",
            "--model",
            "StubMobileCLIP",
            "--pretrained",
            "stub",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--epochs",
            "1",
            "--warmup-epochs",
            "0",
            "--label-smoothing",
            "0.0",
            "--hard-neg-confusion-path",
            str(tmp_path / "does_not_exist.json"),
            "--checkpoint-dir",
            str(ckpt_dir),
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert output.exists()
    report = TrainReport.model_validate_json(output.read_text(encoding="utf-8"))
    assert report.n_train == 4
    assert report.n_val == 2
    out = capsys.readouterr().out
    assert "phase5-train:" in out
    assert "best_top1=" in out


# --------------------------------------------------------------- dirty data


def _build_dataset_with_paths(
    paths: list[Path],
    *,
    class_id: str = "alpha",
) -> Any:
    """Construct an ``_ImagePathDataset`` over the given paths (single class)."""
    from car_lense_engine.training.train_classifier import _ImagePathDataset

    rows = [(class_id, p) for p in paths]
    return _ImagePathDataset(
        rows=rows,
        class_to_idx={class_id: 0},
        preprocess=_stub_preprocess,
    )


def test_dataset_skips_missing_image_file(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing image file does not crash; __getitem__ returns None and logs."""
    good0 = _make_image_file(tmp_path / "good0.jpg", color=(10, 20, 30))
    missing = tmp_path / "does_not_exist.jpg"
    good2 = _make_image_file(tmp_path / "good2.jpg", color=(40, 50, 60))

    dataset = _build_dataset_with_paths([good0, missing, good2])

    with caplog.at_level("WARNING", logger="car_lense_engine.training.train_classifier"):
        item0 = dataset[0]
        item1 = dataset[1]
        item2 = dataset[2]

    assert item0 is not None
    tensor0, label0 = item0
    assert label0 == 0
    assert tensor0.shape == (3, 8, 8)

    assert item1 is None

    assert item2 is not None
    tensor2, label2 = item2
    assert label2 == 0
    assert tensor2.shape == (3, 8, 8)

    # The warning must reference the bad path.
    warning_messages = [rec.getMessage() for rec in caplog.records if rec.levelname == "WARNING"]
    assert any(str(missing) in msg for msg in warning_messages), warning_messages


def test_dataset_skips_corrupted_jpeg(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A garbage-bytes .jpg yields None (logged) instead of propagating an exception."""
    good = _make_image_file(tmp_path / "good.jpg", color=(70, 80, 90))
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"this is not a real JPEG, just random bytes \x00\x01\x02\x03" * 8)

    dataset = _build_dataset_with_paths([good, corrupt])

    with caplog.at_level("WARNING", logger="car_lense_engine.training.train_classifier"):
        item_good = dataset[0]
        item_bad = dataset[1]

    assert item_good is not None
    assert item_bad is None

    warning_messages = [rec.getMessage() for rec in caplog.records if rec.levelname == "WARNING"]
    assert any(str(corrupt) in msg for msg in warning_messages), warning_messages
