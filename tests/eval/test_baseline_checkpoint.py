"""Tests for ``BaselineConfig.checkpoint_path`` (Phase 5.2 integration).

These tests verify that the baseline harness can optionally load a
fine-tuned image-encoder state dict on top of the pretrained OpenCLIP
weights. The OpenCLIP backbone is stubbed exactly the same way the
Phase 5.1 baseline tests stub it.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from car_lense_engine.db import Image, Listing, images, listings, open_db
from car_lense_engine.eval.baseline import (
    BaselineConfig,
    build_prototypes,
    class_id_for,
)

torch = pytest.importorskip("torch")
pytest.importorskip("torch.nn")
pytest.importorskip("torchvision")


# --------------------------------------------------------------- stub model


class _StubVisual(torch.nn.Module):  # type: ignore[misc, name-defined]
    """Real ``nn.Module`` with a single ``Linear`` so ``state_dict`` round-trips."""

    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        # Two parameters; we'll mutate them between save + load to prove
        # the load actually wrote new values.
        self.proj = torch.nn.Linear(3, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3)) if x.dim() == 4 else x
        return self.proj(pooled)


class _StubModel(torch.nn.Module):  # type: ignore[misc, name-defined]
    """OpenCLIP-shaped wrapper around :class:`_StubVisual`."""

    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.visual = _StubVisual(embed_dim=embed_dim)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        # Always returns a (B, embed_dim) L2-normalizable tensor.
        return self.visual(x)


def _stub_preprocess(img: Any) -> torch.Tensor:
    from torchvision import transforms as T

    return T.Compose([T.Resize((4, 4)), T.ToTensor()])(img)  # type: ignore[no-any-return]


class _StubOpenClip:
    """Drop-in replacement for the ``open_clip`` module.

    We re-instantiate the model on each ``create_model_and_transforms``
    call so the test can compare "fresh" state to "loaded" state.
    """

    def __init__(self) -> None:
        self.last_model: _StubModel | None = None

    def create_model_and_transforms(
        self,
        model_name: str,
        *,
        pretrained: str,
        device: str,
    ) -> tuple[_StubModel, None, Any]:
        m = _StubModel(embed_dim=4)
        self.last_model = m
        return m, None, _stub_preprocess


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
    from PIL import Image as PILImage

    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (4, 4), color=color).save(path, format="JPEG")
    return path


def _seed_one_class(conn: sqlite3.Connection, tmp_path: Path) -> dict[str, list[Path]]:
    from car_lense_engine.dataset.canonical_labels import year_to_generation

    classes = [(2012, "Acura", "RL")]
    out: dict[str, list[Path]] = {}
    counter = 0
    for year, make, model in classes:
        # Phase 4.6: class id is keyed off the bucketed generation_year.
        gen_year = year_to_generation(year)
        cid = class_id_for(gen_year, make, model)
        assert cid is not None
        out[cid] = []
        for split_name, n in (("train", 2), ("test", 1)):
            for i in range(n):
                counter += 1
                listing_id = f"stanford_cars:test_{counter:04d}"
                img_path = tmp_path / "imgs" / f"{cid}_{split_name}_{i}.jpg"
                _make_image_file(img_path)
                listings.insert_listing(
                    conn,
                    Listing(
                        listing_id=listing_id,
                        source="stanford_cars",
                        url=f"stanford_cars://{cid}/{counter:04d}",
                        year=year,
                        make=make,
                        model=model,
                        split=split_name,
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
                        source_url=f"stanford_cars://{cid}/{counter:04d}",
                        local_path=str(img_path),
                        position=1,
                    ),
                )
                # Mirror the listing-level split into ``images.split``
                # (Phase 3.5).
                with conn:
                    conn.execute(
                        "UPDATE images SET split = ? WHERE image_id = ?",
                        (split_name, image_id),
                    )
                out[cid].append(img_path)
    return out


# --------------------------------------------------------------- tests


def test_baseline_loads_checkpoint_into_visual(
    stub_open_clip: _StubOpenClip,
    tmp_path: Path,
    db_path: Path,
) -> None:
    """``BaselineConfig(checkpoint_path=...)`` overlays the saved state dict.

    We build a 'fake checkpoint' on disk that mutates the visual tower's
    weight to a recognisable value, then run the baseline harness and
    confirm the post-load weights match the checkpoint, not the freshly
    initialised values.
    """
    conn = open_db(db_path)
    try:
        _seed_one_class(conn, tmp_path)
    finally:
        conn.close()

    # Build a checkpoint by saving a known visual state.
    sentinel_visual = _StubVisual(embed_dim=4)
    with torch.no_grad():
        sentinel_visual.proj.weight.fill_(7.0)
        sentinel_visual.proj.bias.fill_(-3.0)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "image_encoder_state_dict": sentinel_visual.state_dict(),
            "head_state_dict": {},
            "n_classes": 1,
            "class_ids": ["2012|acura|rl"],
            "epoch": 0,
            "val_top1": 0.99,
        },
        ckpt_path,
    )

    config = BaselineConfig(
        model_name="StubMobileCLIP",
        pretrained="stub",
        device="cpu",
        batch_size=2,
        top_ks=(1,),
        checkpoint_path=ckpt_path,
    )

    conn = open_db(db_path)
    try:
        build_prototypes(conn=conn, config=config, source="stanford_cars", split="train")
    finally:
        conn.close()

    loaded_model = stub_open_clip.last_model
    assert loaded_model is not None
    # The freshly-constructed visual.proj would have small-random weights.
    # After load, every entry must be exactly 7.0 / -3.0 (no other code
    # path touched them).
    assert torch.allclose(loaded_model.visual.proj.weight, torch.full((4, 3), 7.0))
    assert torch.allclose(loaded_model.visual.proj.bias, torch.full((4,), -3.0))


def test_baseline_missing_checkpoint_raises(
    stub_open_clip: _StubOpenClip,
    tmp_path: Path,
    db_path: Path,
) -> None:
    conn = open_db(db_path)
    try:
        _seed_one_class(conn, tmp_path)
    finally:
        conn.close()

    config = BaselineConfig(
        model_name="StubMobileCLIP",
        pretrained="stub",
        device="cpu",
        batch_size=2,
        top_ks=(1,),
        checkpoint_path=tmp_path / "nope.pt",
    )

    conn = open_db(db_path)
    try:
        with pytest.raises(RuntimeError, match="checkpoint path does not exist"):
            build_prototypes(conn=conn, config=config, source="stanford_cars", split="train")
    finally:
        conn.close()


def test_baseline_bad_checkpoint_format_raises(
    stub_open_clip: _StubOpenClip,
    tmp_path: Path,
    db_path: Path,
) -> None:
    conn = open_db(db_path)
    try:
        _seed_one_class(conn, tmp_path)
    finally:
        conn.close()

    # A torch-loadable file that *doesn't* have the expected key.
    ckpt_path = tmp_path / "bad.pt"
    torch.save({"some_other_key": 1}, ckpt_path)

    config = BaselineConfig(
        model_name="StubMobileCLIP",
        pretrained="stub",
        device="cpu",
        batch_size=2,
        top_ks=(1,),
        checkpoint_path=ckpt_path,
    )

    conn = open_db(db_path)
    try:
        with pytest.raises(RuntimeError, match="not a Phase 5.2 training checkpoint"):
            build_prototypes(conn=conn, config=config, source="stanford_cars", split="train")
    finally:
        conn.close()


def test_eval_cli_accepts_checkpoint_flag(
    stub_open_clip: _StubOpenClip,
    tmp_path: Path,
    db_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``phase5-baseline --checkpoint ...`` parses and the eval runs."""
    from car_lense_engine.eval import cli as eval_cli

    conn = open_db(db_path)
    try:
        _seed_one_class(conn, tmp_path)
    finally:
        conn.close()

    sentinel = _StubVisual(embed_dim=4)
    with torch.no_grad():
        sentinel.proj.weight.fill_(0.5)
        sentinel.proj.bias.fill_(0.1)
    ckpt_path = tmp_path / "ck.pt"
    torch.save({"image_encoder_state_dict": sentinel.state_dict()}, ckpt_path)

    out_path = tmp_path / "reports" / "out.json"
    rc = eval_cli.main(
        [
            "--db",
            str(db_path),
            "--source",
            "stanford_cars",
            "--model",
            "StubMobileCLIP",
            "--pretrained",
            "stub",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--checkpoint",
            str(ckpt_path),
            "--output",
            str(out_path),
            "--per-class-top",
            "1",
        ]
    )
    assert rc == 0
    assert out_path.exists()

    # The model returned by the stub must have the sentinel weights.
    loaded = stub_open_clip.last_model
    assert loaded is not None
    assert torch.allclose(loaded.visual.proj.weight, torch.full((4, 3), 0.5))


def test_eval_cli_rejects_missing_checkpoint(tmp_path: Path, db_path: Path) -> None:
    """``--checkpoint nope.pt`` is rejected by argparse-level validation."""
    from car_lense_engine.eval import cli as eval_cli

    open_db(db_path).close()
    with pytest.raises(SystemExit) as excinfo:
        eval_cli.main(
            [
                "--db",
                str(db_path),
                "--checkpoint",
                str(tmp_path / "nope.pt"),
            ]
        )
    assert excinfo.value.code == 2
