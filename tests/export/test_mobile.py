"""Tests for the Phase 5.5 mobile export pipeline.

We stub the OpenCLIP backbone with a tiny ``nn.Module`` (a single
``Conv2d`` + spatial mean) so the export can run end-to-end without
downloading 100 MB of weights. The Core ML and TFLite paths are
exercised by monkey-patching :mod:`coremltools` / :mod:`onnx2tf`
import sites -- the goal is to assert the pipeline's *control flow*
(skips, fallbacks, report shape), not the converters themselves.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")


# --------------------------------------------------------------- stub backbone


class _StubVisual(torch.nn.Module):  # type: ignore[misc]
    """Tiny image encoder: Conv2d to 512 channels, spatial-mean to (B, 512)."""

    def __init__(self, embed_dim: int = 8) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, embed_dim, kernel_size=3, padding=1)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).mean(dim=(2, 3))


class _StubModel(torch.nn.Module):  # type: ignore[misc]
    """OpenCLIP-shaped wrapper around :class:`_StubVisual`."""

    def __init__(self, embed_dim: int = 8) -> None:
        super().__init__()
        self.visual = _StubVisual(embed_dim=embed_dim)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


class _StubPreprocess:
    """Tiny stand-in for the OpenCLIP ``Compose`` preprocess transform.

    Holds a ``.transforms`` attribute that ``_extract_preprocess_params``
    can introspect, so we don't have to import torchvision in tests.
    """

    def __init__(self, transforms: list[Any]) -> None:
        self.transforms = transforms


# NOTE: The class names below intentionally mirror torchvision's
# ``Resize`` / ``CenterCrop`` / ``Normalize`` because
# ``_extract_preprocess_params`` matches transforms by
# ``type(transform).__name__``. Renaming these classes breaks that
# match -- if you change them, also update the extractor.


class Resize:  # noqa: N801 -- mirrors torchvision class name
    def __init__(self, size: int, interpolation: str = "bicubic") -> None:
        self.size = size
        # Match the torchvision API: an object with ``.value`` (string).
        self.interpolation = type("_InterpMode", (), {"value": interpolation})()


class CenterCrop:  # noqa: N801 -- mirrors torchvision class name
    def __init__(self, size: int | tuple[int, int]) -> None:
        self.size = (size, size) if isinstance(size, int) else size


class Normalize:  # noqa: N801 -- mirrors torchvision class name
    def __init__(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        self.mean = list(mean)
        self.std = list(std)


def _make_mobileclip_b_preprocess() -> _StubPreprocess:
    """Synthetic transform pipeline that matches MobileCLIP-B's recipe."""
    return _StubPreprocess(
        [
            Resize(224, interpolation="bilinear"),
            CenterCrop(224),
            Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ]
    )


def _make_mobileclip_s2_preprocess() -> _StubPreprocess:
    """Synthetic transform pipeline that matches MobileCLIP-S2's recipe."""
    return _StubPreprocess(
        [
            Resize(256, interpolation="bicubic"),
            CenterCrop(256),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


class _StubOpenClip:
    """Drop-in replacement for ``open_clip`` for export tests."""

    def __init__(self, embed_dim: int = 8) -> None:
        self.embed_dim = embed_dim
        self.last_call: dict[str, Any] | None = None
        # Default to the S2-style synthetic preprocess so the
        # downstream extractor has the transforms it expects. Tests can
        # swap this for the B-style pipeline by reassigning the
        # attribute before calling load_backbone.
        self.preprocess: Any = _make_mobileclip_s2_preprocess()

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
        # Match the open_clip return shape (model, preprocess_train, preprocess_val).
        return _StubModel(embed_dim=self.embed_dim), None, self.preprocess


@pytest.fixture
def stub_open_clip(monkeypatch: pytest.MonkeyPatch) -> _StubOpenClip:
    stub = _StubOpenClip(embed_dim=8)
    fake_mod: Any = stub
    monkeypatch.setitem(sys.modules, "open_clip", fake_mod)
    return stub


# --------------------------------------------------------------- fixtures


@pytest.fixture
def backbone_checkpoint(tmp_path: Path, stub_open_clip: _StubOpenClip) -> Path:
    """Save a tiny Phase 5.2-style checkpoint to disk."""
    model = _StubModel(embed_dim=stub_open_clip.embed_dim)
    state = model.visual.state_dict()
    ckpt_path = tmp_path / "backbone.pt"
    torch.save(
        {
            "image_encoder_state_dict": state,
            "epoch": 0,
            "val_top1": 0.5,
        },
        ckpt_path,
    )
    return ckpt_path


@pytest.fixture
def prototypes_path(tmp_path: Path) -> Path:
    """Build a tiny prototype cache with 3 classes x 8 embedding dim."""
    proto = torch.randn(3, 8)
    proto = proto / proto.norm(dim=-1, keepdim=True)
    payload = {
        "class_ids": ["c0", "c1", "c2"],
        "display_names": ["Class Zero", "Class One", "Class Two"],
        "prototypes": proto,
        "config": {
            "model": "MobileCLIP-B",
            "pretrained": "datacompdr",
        },
    }
    p = tmp_path / "prototypes.pt"
    torch.save(payload, p)
    return p


@pytest.fixture
def view_classifier_checkpoint(tmp_path: Path) -> Path:
    """Save a tiny Phase 5.3 binary view-classifier checkpoint."""
    head = torch.nn.Linear(8, 2)
    state = head.state_dict()
    ckpt_path = tmp_path / "view_classifier.pt"
    torch.save(
        {
            "head_state_dict": state,
            "class_names": ["exterior", "non-exterior"],
            "config": {"embed_dim": 8},
        },
        ckpt_path,
    )
    return ckpt_path


# --------------------------------------------------------------- isolated tests


def test_load_backbone_overlays_checkpoint(
    stub_open_clip: _StubOpenClip,
    backbone_checkpoint: Path,
) -> None:
    from car_lense_engine.export.mobile import load_backbone

    encoder, torch_mod, _preprocess = load_backbone(
        model_name="MobileCLIP-B",
        pretrained="datacompdr",
        checkpoint_path=backbone_checkpoint,
    )
    # The wrapper emits L2-normalized embeddings.
    dummy = torch_mod.zeros((2, 3, 8, 8), dtype=torch_mod.float32)
    with torch_mod.no_grad():
        out = encoder(dummy)
    assert out.shape == (2, 8)
    norms = out.norm(dim=-1)
    # The stub conv on a zero input produces a constant per-channel
    # output (the conv bias). Cast to a unit norm and check.
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_export_onnx_produces_valid_file(
    stub_open_clip: _StubOpenClip,
    backbone_checkpoint: Path,
    tmp_path: Path,
) -> None:
    pytest.importorskip("onnx")
    from car_lense_engine.export.mobile import export_onnx, load_backbone

    encoder, torch_mod, _preprocess = load_backbone(
        model_name="MobileCLIP-B",
        pretrained="datacompdr",
        checkpoint_path=backbone_checkpoint,
    )
    onnx_path = tmp_path / "model.onnx"
    out = export_onnx(
        model=encoder,
        torch_mod=torch_mod,
        onnx_path=onnx_path,
        input_size=8,
        opset=17,
    )
    assert out == onnx_path
    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0

    # Re-load and run via onnxruntime; output shape should be (1, 8).
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {sess.get_inputs()[0].name: torch.zeros((1, 3, 8, 8)).numpy()}
    output = sess.run(None, feed)[0]
    assert output.shape == (1, 8)


def test_bundle_prototypes_round_trip(prototypes_path: Path, tmp_path: Path) -> None:
    from car_lense_engine.export.mobile import bundle_prototypes, read_prototypes_bin

    out = tmp_path / "prototypes.bin"
    n_classes, embed_dim = bundle_prototypes(
        prototypes_path=prototypes_path,
        output_path=out,
    )
    assert (n_classes, embed_dim) == (3, 8)
    arr = read_prototypes_bin(out, shape=(n_classes, embed_dim))
    # Each row should still be (approximately) unit norm after FP16
    # round-trip.
    norms = (arr * arr).sum(axis=-1) ** 0.5
    assert (norms > 0.95).all()


def test_class_names_json_round_trip(prototypes_path: Path, tmp_path: Path) -> None:
    from car_lense_engine.export.mobile import write_class_names_json

    out = tmp_path / "class_names.json"
    n = write_class_names_json(prototypes_path=prototypes_path, output_path=out)
    assert n == 3
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["class_ids"] == ["c0", "c1", "c2"]
    assert payload["display_names"] == ["Class Zero", "Class One", "Class Two"]
    assert len(payload["class_ids"]) == len(payload["display_names"])


def test_preprocessing_json_schema(tmp_path: Path) -> None:
    """Schema-only check: every required key is present and serializable.

    Values for ``input_size`` / ``mean`` / ``std`` / ``resize_interpolation``
    are derived from the actual model preprocess at export time, so we
    only assert the *shape* of the JSON here rather than hardcoding
    one model variant's recipe.
    """
    from car_lense_engine.export.mobile import write_preprocessing_json

    out = tmp_path / "preprocessing.json"
    write_preprocessing_json(output_path=out, n_classes=3, embed_dim=8)
    payload = json.loads(out.read_text(encoding="utf-8"))

    # Shape: required keys + types.
    for key in (
        "input_size",
        "mean",
        "std",
        "color_space",
        "resize_interpolation",
        "resize_strategy",
        "embedding_dim",
        "embedding_normalized",
        "prototypes_bin",
    ):
        assert key in payload, f"missing key {key!r} in preprocessing.json"

    assert isinstance(payload["input_size"], list)
    assert len(payload["input_size"]) == 2
    assert isinstance(payload["mean"], list)
    assert len(payload["mean"]) == 3
    assert isinstance(payload["std"], list)
    assert len(payload["std"]) == 3
    assert payload["color_space"] == "RGB"
    assert payload["resize_strategy"] == "resize_shortest_then_center_crop"
    assert payload["embedding_dim"] == 8
    assert payload["embedding_normalized"] is True
    assert payload["prototypes_bin"]["shape"] == [3, 8]
    assert payload["prototypes_bin"]["dtype"] == "fp16"
    assert payload["prototypes_bin"]["byte_order"] == "little"


def test_preprocessing_json_uses_extracted_params(tmp_path: Path) -> None:
    """``write_preprocessing_json`` should overwrite the template defaults."""
    from car_lense_engine.export.mobile import write_preprocessing_json

    out = tmp_path / "preprocessing.json"
    write_preprocessing_json(
        output_path=out,
        n_classes=3,
        embed_dim=8,
        preprocess_params={
            "input_size": (224, 224),
            "mean": (0.1, 0.2, 0.3),
            "std": (0.4, 0.5, 0.6),
            "resize_size": 224,
            "interpolation": "bilinear",
        },
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["input_size"] == [224, 224]
    assert payload["mean"] == [0.1, 0.2, 0.3]
    assert payload["std"] == [0.4, 0.5, 0.6]
    assert payload["resize_interpolation"] == "bilinear"


def test_extract_preprocess_params_handles_mobileclip_b() -> None:
    """MobileCLIP-B uses 224x224 + bilinear + mean=0 / std=1."""
    from car_lense_engine.export.mobile import _extract_preprocess_params

    params = _extract_preprocess_params(_make_mobileclip_b_preprocess())
    assert params["input_size"] == (224, 224)
    assert params["mean"] == (0.0, 0.0, 0.0)
    assert params["std"] == (1.0, 1.0, 1.0)
    assert params["resize_size"] == 224
    assert params["interpolation"] == "bilinear"


def test_extract_preprocess_params_handles_mobileclip_s2() -> None:
    """MobileCLIP-S2 uses 256x256 + bicubic + mean=std=0.5."""
    from car_lense_engine.export.mobile import _extract_preprocess_params

    params = _extract_preprocess_params(_make_mobileclip_s2_preprocess())
    assert params["input_size"] == (256, 256)
    assert params["mean"] == (0.5, 0.5, 0.5)
    assert params["std"] == (0.5, 0.5, 0.5)
    assert params["resize_size"] == 256
    assert params["interpolation"] == "bicubic"


def test_extract_preprocess_params_rejects_pipeline_without_centercrop() -> None:
    """A pipeline missing CenterCrop must error out, not silently default."""
    from car_lense_engine.export.mobile import _extract_preprocess_params

    bad = _StubPreprocess(
        [
            Resize(224),
            Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ]
    )
    with pytest.raises(RuntimeError, match="CenterCrop"):
        _extract_preprocess_params(bad)


def test_extract_preprocess_params_rejects_pipeline_without_normalize() -> None:
    """A pipeline missing Normalize must error out, not silently default."""
    from car_lense_engine.export.mobile import _extract_preprocess_params

    bad = _StubPreprocess(
        [
            Resize(224),
            CenterCrop(224),
        ]
    )
    with pytest.raises(RuntimeError, match="Normalize"):
        _extract_preprocess_params(bad)


def test_prototypes_bin_byte_layout(tmp_path: Path) -> None:
    """A known FP16 matrix round-trips through write_fp16_matrix / read_prototypes_bin."""
    import numpy as np

    from car_lense_engine.export.mobile import read_prototypes_bin, write_fp16_matrix

    arr = np.array(
        [
            [0.1, -0.2, 0.3, 0.4],
            [0.5, 0.6, -0.7, 0.8],
        ],
        dtype="float16",
    )
    out = tmp_path / "matrix.bin"
    write_fp16_matrix(arr, out)
    # The byte size must equal 2 * elements (FP16 = 2 bytes/element).
    assert out.stat().st_size == arr.size * 2

    loaded = read_prototypes_bin(out, shape=arr.shape)
    np.testing.assert_array_equal(loaded, arr)


def test_export_view_head_emits_onnx(
    stub_open_clip: _StubOpenClip,
    view_classifier_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("onnx")
    # See ``test_export_mobile_end_to_end_minimal`` for the rationale on
    # forcing the optional-converter import path to miss -- onnx2tf can
    # segfault on the trivial Linear stub when actually installed.
    monkeypatch.setitem(sys.modules, "onnx2tf", None)
    monkeypatch.setitem(sys.modules, "coremltools", None)
    from car_lense_engine.export.mobile import export_view_head

    ios_dir = tmp_path / "ios"
    android_dir = tmp_path / "android"
    common_dir = tmp_path / "common"
    paths = export_view_head(
        view_classifier_checkpoint=view_classifier_checkpoint,
        output_dir_ios=ios_dir,
        output_dir_android=android_dir,
        output_dir_common=common_dir,
        embed_dim=8,
    )
    assert paths["onnx"] is not None
    assert paths["onnx"].exists()
    # Core ML and TFLite may be None depending on what's installed --
    # the test just asserts the key shape.
    assert "coreml" in paths
    assert "tflite" in paths


def test_export_view_head_rejects_dim_mismatch(
    view_classifier_checkpoint: Path,
    tmp_path: Path,
) -> None:
    from car_lense_engine.export.mobile import export_view_head

    with pytest.raises(RuntimeError, match="expects"):
        export_view_head(
            view_classifier_checkpoint=view_classifier_checkpoint,
            output_dir_ios=tmp_path / "ios",
            output_dir_android=tmp_path / "android",
            output_dir_common=tmp_path / "common",
            embed_dim=16,  # wrong; head expects 8
        )


def test_export_mobile_end_to_end_minimal(
    stub_open_clip: _StubOpenClip,
    backbone_checkpoint: Path,
    prototypes_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the full pipeline with a tiny stub; Core ML / TFLite both skip."""
    pytest.importorskip("onnx")
    # ``onnx2tf`` is an optional dep that is occasionally installed in
    # the local venv. It segfaults on the trivial stub graph
    # (the Gemm op crashes on a 1x8 input), so force the import miss
    # path to keep this test deterministic across machines. The
    # production code path already handles the ImportError gracefully.
    monkeypatch.setitem(sys.modules, "onnx2tf", None)
    monkeypatch.setitem(sys.modules, "coremltools", None)
    from car_lense_engine.export.mobile import MobileExportConfig, export_mobile

    config = MobileExportConfig(
        backbone_checkpoint=backbone_checkpoint,
        view_classifier_checkpoint=None,
        prototypes_path=prototypes_path,
        output_dir=tmp_path / "dist",
        model_name="MobileCLIP-B",
        pretrained="datacompdr",
        quantize="fp16",
    )
    report = export_mobile(config=config)

    # ONNX must be present.
    assert report.onnx_path.exists()
    assert (tmp_path / "dist" / "common" / "model.onnx").exists()

    # Prototypes + class names + preprocessing in both ios + android.
    for sub in ("ios", "android"):
        d = tmp_path / "dist" / sub
        assert (d / "prototypes.bin").exists()
        assert (d / "class_names.json").exists()
        assert (d / "preprocessing.json").exists()

    # sizes_mb has at least onnx + prototypes + total.
    assert "onnx" in report.sizes_mb
    assert "prototypes" in report.sizes_mb
    assert "total" in report.sizes_mb

    # The extracted preprocess params are surfaced on the report and
    # baked into preprocessing.json. The stub returns the S2-style
    # (256, 256) recipe by default; the bundle should reflect that
    # exactly, not a stale hardcoded value.
    assert report.preprocess_params["input_size"] == (256, 256)
    assert report.preprocess_params["mean"] == (0.5, 0.5, 0.5)
    assert report.preprocess_params["std"] == (0.5, 0.5, 0.5)
    pre_payload = json.loads(
        (tmp_path / "dist" / "ios" / "preprocessing.json").read_text(encoding="utf-8")
    )
    assert pre_payload["input_size"] == [256, 256]
    assert pre_payload["mean"] == [0.5, 0.5, 0.5]
    assert pre_payload["std"] == [0.5, 0.5, 0.5]

    # Skipped notes are populated for Core ML (no coremltools in test env)
    # and TFLite (no onnx2tf).
    assert any("coreml" in s.lower() for s in report.skipped)


def test_export_mobile_picks_up_mobileclip_b_preprocess(
    stub_open_clip: _StubOpenClip,
    backbone_checkpoint: Path,
    prototypes_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switching the stub's preprocess to MobileCLIP-B style produces 224x224 outputs."""
    pytest.importorskip("onnx")
    # See ``test_export_mobile_end_to_end_minimal`` for the rationale on
    # forcing the optional-converter import path to miss.
    monkeypatch.setitem(sys.modules, "onnx2tf", None)
    monkeypatch.setitem(sys.modules, "coremltools", None)
    from car_lense_engine.export.mobile import MobileExportConfig, export_mobile

    stub_open_clip.preprocess = _make_mobileclip_b_preprocess()

    config = MobileExportConfig(
        backbone_checkpoint=backbone_checkpoint,
        view_classifier_checkpoint=None,
        prototypes_path=prototypes_path,
        output_dir=tmp_path / "dist",
        model_name="MobileCLIP-B",
        pretrained="datacompdr",
        quantize="fp16",
    )
    report = export_mobile(config=config)

    assert report.preprocess_params["input_size"] == (224, 224)
    assert report.preprocess_params["interpolation"] == "bilinear"
    pre_payload = json.loads(report.preprocessing_path.read_text(encoding="utf-8"))
    assert pre_payload["input_size"] == [224, 224]
    assert pre_payload["resize_interpolation"] == "bilinear"
    assert pre_payload["mean"] == [0.0, 0.0, 0.0]
    assert pre_payload["std"] == [1.0, 1.0, 1.0]


def test_export_coreml_skipped_on_import_error(
    stub_open_clip: _StubOpenClip,
    backbone_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If coremltools import fails, ``export_coreml`` returns ``None`` without raising."""
    from car_lense_engine.export.mobile import export_coreml, load_backbone

    # Force a fresh import attempt to miss.
    monkeypatch.setitem(sys.modules, "coremltools", None)

    encoder, torch_mod, _preprocess = load_backbone(
        model_name="MobileCLIP-B",
        pretrained="datacompdr",
        checkpoint_path=backbone_checkpoint,
    )
    out = export_coreml(
        model=encoder,
        torch_mod=torch_mod,
        coreml_path=tmp_path / "model.mlpackage",
        input_size=8,
        quantize="fp16",
    )
    assert out is None
    assert not (tmp_path / "model.mlpackage").exists()


def test_export_tflite_falls_back_when_onnx2tf_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If onnx2tf is missing, ``export_tflite`` records the fallback reason."""
    from car_lense_engine.export import mobile

    monkeypatch.setitem(sys.modules, "onnx2tf", None)

    # Create a fake .onnx file so the ORT-Mobile fallback has something
    # to chew on. We don't actually need a valid graph here -- we're
    # just asserting the control-flow.
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_bytes(b"not-a-real-onnx")
    out, reason = mobile.export_tflite(
        onnx_path=fake_onnx,
        tflite_path=tmp_path / "model.tflite",
        quantize="fp16",
    )
    assert out is None or out.suffix == ".ort"
    assert reason is not None
    assert "onnx2tf" in reason


def test_report_to_dict_is_json_serializable() -> None:
    from car_lense_engine.export.mobile import MobileExportReport

    report = MobileExportReport(
        onnx_path=Path("dist/common/model.onnx"),
        prototypes_path=Path("dist/ios/prototypes.bin"),
        class_names_path=Path("dist/ios/class_names.json"),
        preprocessing_path=Path("dist/ios/preprocessing.json"),
        coreml_path=None,
        tflite_path=Path("dist/android/model.tflite"),
        view_head_paths={"onnx": Path("dist/common/view_head.onnx"), "coreml": None},
        sizes_mb={"onnx": 1.2, "tflite": 1.5},
        skipped=["coreml -- coremltools missing"],
        notes=["test note"],
    )
    rendered = json.dumps(report.to_dict())
    payload = json.loads(rendered)
    assert payload["coreml_path"] is None
    assert payload["tflite_path"] == "dist/android/model.tflite"
    assert payload["view_head_paths"]["coreml"] is None
    assert payload["skipped"] == ["coreml -- coremltools missing"]


def test_load_backbone_rejects_bad_checkpoint(
    stub_open_clip: _StubOpenClip,
    tmp_path: Path,
) -> None:
    from car_lense_engine.export.mobile import load_backbone

    bad = tmp_path / "bad.pt"
    torch.save({"not_the_right_key": 1}, bad)
    with pytest.raises(RuntimeError, match="not a Phase 5.2 training checkpoint"):
        load_backbone(
            model_name="MobileCLIP-B",
            pretrained="datacompdr",
            checkpoint_path=bad,
        )


def test_input_size_rejects_unknown_model() -> None:
    from car_lense_engine.export.mobile import _input_size

    assert _input_size("MobileCLIP-B") == 256
    with pytest.raises(RuntimeError, match="unsupported model"):
        _input_size("ResNet-50")


@pytest.fixture
def isolated_test_dir(tmp_path: Path) -> Iterator[Path]:
    """Provide a clean directory for each export test."""
    yield tmp_path
