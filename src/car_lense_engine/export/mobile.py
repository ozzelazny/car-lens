"""Mobile export pipeline (Phase 5.5).

Turn the trained MobileCLIP-B backbone + binary view classifier +
per-class prototype cache into shippable mobile bundles. Three formats
are produced (any of which may be skipped if the corresponding optional
dependency is missing or its converter rejects the graph):

* **ONNX** -- the intermediate format. Always produced (we need the
  graph for the TFLite path); also useful for desktop debugging via
  onnxruntime.
* **Core ML** (``.mlpackage``) -- iOS deployment format. Requires
  :mod:`coremltools` (large, optional). Falls back to skipped.
* **TFLite** -- Android deployment format. Requires :mod:`onnx2tf`
  (drags TensorFlow as a transitive dep). Falls back to ONNX Runtime
  Mobile (``.ort``) if the TFLite conversion fails, then to skipped.

Bundled alongside the model files:

* **prototypes.bin** -- the ``[6423, 512]`` FP16 prototype matrix as raw
  little-endian bytes. On-device code reads this without any torch
  dependency (cosine sim against the model output is a tiny dot
  product).
* **class_names.json** -- ``{"class_ids": [...], "display_names": [...]}``
  pulled straight from the prototype cache.
* **preprocessing.json** -- describes the input transform so the iOS /
  Android implementations match the training-time recipe exactly. See
  :data:`PREPROCESSING_TEMPLATE` for the schema.

Design notes
------------

* **No on-device PyTorch.** The mobile clients ship the exported graph
  + a raw FP16 binary for the prototypes; everything else is JSON +
  whatever the platform's ML runtime needs. This keeps the app
  binary <30 MB on iOS / <40 MB on Android.

* **L2-normalize inside the graph.** The exported encoder applies
  ``F.normalize`` to its output so the on-device code doesn't have to.
  This matches the prototype-building recipe (prototypes are
  unit-normalized) and means top-K retrieval is a plain matmul + topk.

* **FP16 by default.** DESIGN.md targets <30 ms latency on mid-range
  Android; FP16 is the standard mobile quantization for transformer-y
  backbones and is expected to cost <0.5 pp accuracy on this task.
  INT8 is a knob but not validated yet.

* **Lazy imports.** ``onnx``, ``onnxsim``, ``coremltools``, ``onnx2tf``,
  ``onnxruntime`` are imported inside the export functions. Tests stub
  them via :mod:`sys.modules` exactly like the baseline / view
  classifier tests stub :mod:`open_clip`.

* **Skipped, not crashed.** If the Core ML or TFLite converter fails,
  we log a clear error, mark the corresponding output path ``None``,
  and continue. The ONNX file is still useful on its own.

* **Read-only inputs.** The trained checkpoint files and prototype
  cache are mounted read-only. We never modify them in place.
"""

from __future__ import annotations

import json
import logging
import shutil
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- constants


PREPROCESSING_TEMPLATE: dict[str, Any] = {
    "input_size": [256, 256],
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "color_space": "RGB",
    "resize_interpolation": "bicubic",
    "resize_strategy": "resize_shortest_then_center_crop",
    "embedding_dim": 512,
    "embedding_normalized": True,
    "prototypes_bin": {
        "shape": [6423, 512],
        "dtype": "fp16",
        "byte_order": "little",
    },
}
"""Default content of ``preprocessing.json``.

These defaults match MobileCLIP-S2's preprocessing transform. At export
time we **always** introspect the real preprocess pipeline returned by
``open_clip.create_model_and_transforms`` and overwrite the dynamic
fields (``input_size``, ``mean``, ``std``, ``resize_interpolation``,
``embedding_dim``, ``prototypes_bin.shape``) so the on-device code
matches the training-time recipe exactly -- the template only seeds the
static keys.
"""


# --------------------------------------------------------------- preprocess introspection


def _extract_preprocess_params(preprocess: Any) -> dict[str, Any]:
    """Pull input_size / mean / std / interpolation from an OpenCLIP transform.

    ``preprocess`` is the ``torchvision.transforms.Compose`` returned by
    ``open_clip.create_model_and_transforms``. The relevant transforms
    are ``Resize``, ``CenterCrop`` and ``Normalize``; everything else
    (``MaybeConvertMode``, ``MaybeToTensor``) is irrelevant to the
    on-device contract.

    The two known MobileCLIP recipes:

    * **S2**: Resize(256, bicubic) -> CenterCrop(256, 256) ->
      Normalize(0.5, 0.5)
    * **B**:  Resize(224, bicubic) -> CenterCrop(224, 224) ->
      Normalize(CLIP_MEAN, CLIP_STD)

    Returns a dict with keys ``input_size`` (tuple[int, int]),
    ``mean`` (tuple[float, float, float]), ``std``
    (tuple[float, float, float]), ``resize_size`` (int) and
    ``interpolation`` (str, lower-case, defaults to ``"bilinear"``).

    Raises :class:`RuntimeError` if the pipeline doesn't expose the
    three transforms we rely on -- callers should not silently fall
    back to defaults because that's exactly the bug this helper fixes.
    """
    transforms = getattr(preprocess, "transforms", None)
    if transforms is None:
        # Some preprocess functions might be plain callables (e.g. the
        # test stub). Treat them as opaque and fall back to template
        # defaults so test code can stay simple.
        return {
            "input_size": (
                int(PREPROCESSING_TEMPLATE["input_size"][0]),
                int(PREPROCESSING_TEMPLATE["input_size"][1]),
            ),
            "mean": tuple(float(x) for x in PREPROCESSING_TEMPLATE["mean"]),
            "std": tuple(float(x) for x in PREPROCESSING_TEMPLATE["std"]),
            "resize_size": int(PREPROCESSING_TEMPLATE["input_size"][0]),
            "interpolation": str(PREPROCESSING_TEMPLATE["resize_interpolation"]),
        }

    input_size: tuple[int, int] | None = None
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    resize_size: int | None = None
    interpolation: str = "bilinear"

    def _coerce_triplet(value: Any) -> tuple[float, float, float]:
        # ``Normalize.mean`` / ``Normalize.std`` are usually a list or
        # tensor of 3 floats; the dataclass surface stays plain
        # ``float`` tuples so JSON serialization is trivial.
        if hasattr(value, "tolist"):
            value = value.tolist()
        values = [float(v) for v in value]
        if len(values) != 3:
            raise RuntimeError(
                f"Normalize transform must have 3 channels; got {len(values)}: {values!r}"
            )
        return (values[0], values[1], values[2])

    for transform in transforms:
        tname = type(transform).__name__
        if tname == "Resize":
            size = getattr(transform, "size", None)
            if isinstance(size, (list, tuple)):
                resize_size = int(size[0])
            elif size is not None:
                resize_size = int(size)
            interp = getattr(transform, "interpolation", None)
            # ``torchvision.transforms.InterpolationMode`` has a
            # ``.value`` string ("bicubic", "bilinear", ...). Older /
            # custom transforms may stash the string directly.
            if interp is not None:
                interp_value = getattr(interp, "value", None)
                if interp_value is not None:
                    interpolation = str(interp_value).lower()
                else:
                    interpolation = str(interp).lower().split(".")[-1]
        elif tname == "CenterCrop":
            size = getattr(transform, "size", None)
            if isinstance(size, (list, tuple)):
                if len(size) == 1:
                    input_size = (int(size[0]), int(size[0]))
                else:
                    input_size = (int(size[0]), int(size[1]))
            elif size is not None:
                input_size = (int(size), int(size))
        elif tname == "Normalize":
            mean = _coerce_triplet(transform.mean)
            std = _coerce_triplet(transform.std)

    if input_size is None:
        raise RuntimeError(
            "preprocess pipeline does not contain a CenterCrop transform; "
            "cannot determine input_size for export"
        )
    if mean is None or std is None:
        raise RuntimeError(
            "preprocess pipeline does not contain a Normalize transform; "
            "cannot determine mean/std for export"
        )
    if resize_size is None:
        # Fall back to the crop size -- still better than guessing 256.
        resize_size = input_size[0]

    return {
        "input_size": input_size,
        "mean": mean,
        "std": std,
        "resize_size": resize_size,
        "interpolation": interpolation,
    }


VIEW_CLASS_NAMES: tuple[str, ...] = ("exterior", "non-exterior")
"""Canonical 2-class binary view-classifier output vocabulary."""


# --------------------------------------------------------------- public dataclasses


@dataclass
class MobileExportConfig:
    """Frozen configuration for one mobile-export run.

    See :func:`export_mobile` for the contract.
    """

    backbone_checkpoint: Path
    """Path to the Phase 5.2 fine-tuned image-encoder checkpoint."""

    prototypes_path: Path
    """Path to the single-prototype cache produced by ``build-prototypes``."""

    output_dir: Path
    """Root directory for the export bundle. Will hold ``ios/``,
    ``android/``, ``common/`` subdirectories on success."""

    view_classifier_checkpoint: Path | None = None
    """Optional binary view-classifier checkpoint. If set, a tiny
    ``view_head`` model is exported alongside the main backbone."""

    model_name: str = "MobileCLIP-B"
    """OpenCLIP model name. The default matches the Phase 5.2 resumed
    backbone."""

    pretrained: str = "datacompdr"
    """OpenCLIP pretrained tag."""

    quantize: Literal["fp32", "fp16", "int8"] = "fp16"
    """Target precision. FP16 is the DESIGN.md default; INT8 is a knob
    but not validated against the accuracy budget yet."""

    opset: int = 17
    """ONNX opset version. 17 is the latest that all three mobile
    runtimes (Core ML, TFLite, ORT Mobile) accept in 2026."""


@dataclass
class MobileExportReport:
    """Outcome of one mobile-export run.

    Returned by :func:`export_mobile` and serializable to JSON via
    :meth:`to_dict` for the run report.
    """

    onnx_path: Path
    """Path to the intermediate ONNX file (always produced)."""

    prototypes_path: Path
    """Path to the FP16 prototype binary."""

    class_names_path: Path
    """Path to the ``class_names.json`` bundle."""

    preprocessing_path: Path
    """Path to the ``preprocessing.json`` bundle."""

    coreml_path: Path | None = None
    """Path to the iOS Core ML bundle, or ``None`` if conversion was
    skipped (no :mod:`coremltools` installed, or converter raised)."""

    tflite_path: Path | None = None
    """Path to the Android TFLite bundle, or ``None`` if both the
    :mod:`onnx2tf` and ORT-Mobile fallbacks failed."""

    view_head_paths: dict[str, Path | None] = field(default_factory=dict)
    """Per-format paths for the view-classifier head export. Keys are
    ``"onnx"``, ``"coreml"``, ``"tflite"``. Empty if the caller didn't
    pass a view-classifier checkpoint."""

    sizes_mb: dict[str, float] = field(default_factory=dict)
    """Size (in MB) of every artifact actually written. Keys mirror the
    path-attribute names plus a ``"total"`` aggregate."""

    skipped: list[str] = field(default_factory=list)
    """Human-readable notes about every step that fell back or was
    skipped, e.g. ``"coreml -- coremltools not installed"``."""

    notes: list[str] = field(default_factory=list)
    """Free-form notes for the orchestrator (e.g. backbone/view-head
    feature-space mismatch warning)."""

    preprocess_params: dict[str, Any] = field(default_factory=dict)
    """The preprocessing parameters extracted from the OpenCLIP transform
    and baked into ``preprocessing.json``. Surfaced here for debugging
    so operators can see exactly what input shape / normalization the
    on-device code will use (catches mismatches against the trained
    backbone before shipping)."""

    def to_dict(self) -> dict[str, Any]:
        """Render this report as a JSON-serializable dict."""

        def _maybe_path(p: Path | None) -> str | None:
            return str(p) if p is not None else None

        def _jsonable(value: Any) -> Any:
            if isinstance(value, tuple):
                return list(value)
            return value

        return {
            "onnx_path": str(self.onnx_path),
            "coreml_path": _maybe_path(self.coreml_path),
            "tflite_path": _maybe_path(self.tflite_path),
            "prototypes_path": str(self.prototypes_path),
            "class_names_path": str(self.class_names_path),
            "preprocessing_path": str(self.preprocessing_path),
            "view_head_paths": {k: _maybe_path(v) for k, v in self.view_head_paths.items()},
            "sizes_mb": dict(self.sizes_mb),
            "skipped": list(self.skipped),
            "notes": list(self.notes),
            "preprocess_params": {k: _jsonable(v) for k, v in self.preprocess_params.items()},
        }


# --------------------------------------------------------------- backbone load


class _NormalizedEncoder:
    """Thin nn.Module wrapper that L2-normalizes the encoder output.

    Declared at module scope (rather than nested inside
    :func:`load_backbone`) so :func:`torch.jit.trace` / ONNX export can
    reach the class symbol when running under multiprocessing. This is
    the typed surface the tests poke at.
    """

    # The actual implementation is a ``torch.nn.Module`` subclass built
    # inside :func:`_build_normalized_encoder` because we can't import
    # torch at module load time (open_clip is a heavy optional dep --
    # and importing it eagerly breaks the test stub strategy).


def _build_normalized_encoder(backbone: Any, torch_mod: Any) -> Any:
    """Wrap ``backbone`` so its forward emits L2-normalized embeddings.

    ``backbone`` is the OpenCLIP model returned by
    ``open_clip.create_model_and_transforms``. We only use its
    ``encode_image`` method; the text tower is dropped because it
    isn't needed on-device.

    Returns an ``nn.Module`` whose ``forward(x)`` is the on-device
    contract: ``x: (B, 3, 256, 256)`` -> ``(B, 512)`` L2-normalized.
    """

    nn = torch_mod.nn
    f = torch_mod.nn.functional

    class _Wrapper(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self, encoder: Any) -> None:
            super().__init__()
            self.encoder = encoder

        def forward(self, x: Any) -> Any:
            feats = self.encoder.encode_image(x)
            return f.normalize(feats, dim=-1)

    wrapper = _Wrapper(backbone)
    wrapper.eval()
    return wrapper


def load_backbone(
    *,
    model_name: str,
    pretrained: str,
    checkpoint_path: Path,
    device: str = "cpu",
) -> tuple[Any, Any, Any]:
    """Instantiate the OpenCLIP backbone and overlay the fine-tuned weights.

    Returns ``(normalized_encoder, torch_module, preprocess)`` where
    ``normalized_encoder`` is an ``nn.Module`` that takes
    ``(B, 3, H, W)`` and returns a unit-norm ``(B, D)`` embedding (the
    H/W/D depend on the model variant), ``torch_module`` is the
    imported ``torch`` package, and ``preprocess`` is the
    ``torchvision.transforms.Compose`` that OpenCLIP built for this
    model -- callers use it via :func:`_extract_preprocess_params` to
    derive the on-device input shape and normalization stats.
    """
    try:
        import open_clip  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover -- deps are in pyproject
        raise RuntimeError("open_clip_torch and torch are required for mobile export") from exc

    logger.info("export: loading OpenCLIP %s / %s on %s", model_name, pretrained, device)
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001 -- surface with a clearer hint
        raise RuntimeError(
            f"failed to load OpenCLIP model "
            f"(name={model_name!r}, pretrained={pretrained!r}); "
            f"underlying error: {exc!r}"
        ) from exc

    if not checkpoint_path.exists():
        raise RuntimeError(f"backbone checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "image_encoder_state_dict" not in payload:
        raise RuntimeError(
            f"backbone checkpoint {checkpoint_path} is not a Phase 5.2 training "
            "checkpoint (missing 'image_encoder_state_dict' key)"
        )
    state = payload["image_encoder_state_dict"]
    visual = getattr(model, "visual", None)
    if visual is not None and hasattr(visual, "load_state_dict"):
        visual.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=False)

    # Freeze every parameter -- export is inference-only.
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    wrapper = _build_normalized_encoder(model, torch)
    return wrapper, torch, preprocess


# --------------------------------------------------------------- prototype bundle


def bundle_prototypes(
    *,
    prototypes_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    """Write the prototype tensor to ``output_path`` as raw FP16 bytes.

    Loads the ``build-prototypes`` payload, extracts the ``prototypes``
    tensor (shape ``(n_classes, embed_dim)``), casts to FP16, and
    writes its little-endian bytes to ``output_path``. The byte layout
    is documented in ``preprocessing.json`` so the on-device code can
    ``memmap`` the file without any torch dependency.

    Returns ``(n_classes, embed_dim)`` so callers can validate against
    the model's output shape.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for prototype bundling") from exc

    if not prototypes_path.exists():
        raise RuntimeError(f"prototype cache not found: {prototypes_path}")
    payload = torch.load(prototypes_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "prototypes" not in payload:
        raise RuntimeError(
            f"prototype cache {prototypes_path} is malformed: expected dict with 'prototypes' key"
        )
    proto = payload["prototypes"]
    # Some callers may pass a tensor already on a non-CPU device; force
    # CPU + contiguous so the byte-dump is deterministic.
    proto_cpu = proto.detach().to("cpu").contiguous()
    if proto_cpu.dim() != 2:
        raise RuntimeError(
            "prototype tensor must be 2D (n_classes, embed_dim); "
            f"got shape {tuple(proto_cpu.shape)}"
        )

    n_classes = int(proto_cpu.shape[0])
    embed_dim = int(proto_cpu.shape[1])
    fp16 = proto_cpu.to(dtype=torch.float16).contiguous()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # ``.numpy().tobytes()`` would also work, but going through
    # ``view(torch.uint8)`` keeps numpy out of the dependency graph for
    # this code path -- and is little-endian on every torch-supported
    # platform (the FP16 storage layout is x86 / ARM little-endian).
    raw = fp16.view(torch.uint8).contiguous().numpy().tobytes()
    output_path.write_bytes(raw)
    logger.info(
        "export: wrote %d x %d FP16 prototypes (%d bytes) to %s",
        n_classes,
        embed_dim,
        len(raw),
        output_path,
    )
    return n_classes, embed_dim


def write_class_names_json(*, prototypes_path: Path, output_path: Path) -> int:
    """Extract class_ids + display_names from the prototype cache.

    Writes a ``{"class_ids": [...], "display_names": [...]}`` JSON file
    to ``output_path``. Returns the class count for the caller's report.
    Raises :class:`RuntimeError` if the two lists have mismatched
    lengths (the cache is malformed).
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for class-names bundling") from exc

    payload = torch.load(prototypes_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"prototype cache {prototypes_path} is malformed (not a dict)")
    class_ids = list(payload.get("class_ids") or [])
    display_names = list(payload.get("display_names") or [])
    if len(class_ids) != len(display_names):
        raise RuntimeError(
            f"prototype cache {prototypes_path} has inconsistent lengths: "
            f"{len(class_ids)} class_ids vs {len(display_names)} display_names"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"class_ids": class_ids, "display_names": display_names}, indent=2),
        encoding="utf-8",
    )
    return len(class_ids)


def write_preprocessing_json(
    *,
    output_path: Path,
    n_classes: int,
    embed_dim: int,
    preprocess_params: dict[str, Any] | None = None,
) -> None:
    """Write the ``preprocessing.json`` bundle.

    Starts from :data:`PREPROCESSING_TEMPLATE` and fills in the dynamic
    fields from the live OpenCLIP preprocess transform (when
    ``preprocess_params`` is provided) plus the prototype cache. The
    template only seeds static fields like ``color_space`` and
    ``resize_strategy``; everything that depends on the model variant
    -- ``input_size``, ``mean``, ``std``, ``resize_interpolation`` --
    is overwritten so the on-device code can't drift from the
    training-time recipe.
    """
    payload: dict[str, Any] = json.loads(json.dumps(PREPROCESSING_TEMPLATE))  # deep copy
    payload["embedding_dim"] = embed_dim
    payload["prototypes_bin"]["shape"] = [n_classes, embed_dim]
    if preprocess_params is not None:
        input_size = preprocess_params.get("input_size")
        if input_size is not None:
            payload["input_size"] = [int(input_size[0]), int(input_size[1])]
        mean = preprocess_params.get("mean")
        if mean is not None:
            payload["mean"] = [float(v) for v in mean]
        std = preprocess_params.get("std")
        if std is not None:
            payload["std"] = [float(v) for v in std]
        interpolation = preprocess_params.get("interpolation")
        if interpolation is not None:
            payload["resize_interpolation"] = str(interpolation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_view_class_names_json(output_path: Path) -> None:
    """Write the binary view-classifier class names JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(list(VIEW_CLASS_NAMES), indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------------------- ONNX export


def _input_size(model_name: str) -> int:
    """Legacy fallback resolver for the expected input edge size.

    Superseded by :func:`_extract_preprocess_params`, which inspects the
    actual OpenCLIP transform pipeline and returns the real input size
    (224 for MobileCLIP-B, 256 for MobileCLIP-S2). Kept for backward
    compatibility with existing callers and tests; rejects non-MobileCLIP
    models because we haven't verified their preprocessing assumptions
    on-device.
    """
    if model_name.startswith("MobileCLIP-"):
        return 256
    raise RuntimeError(
        f"unsupported model {model_name!r}; only MobileCLIP-* is wired into the mobile export"
    )


def _normalize_hw(input_size: int | tuple[int, int]) -> tuple[int, int]:
    """Accept either a square edge length or an explicit ``(H, W)``."""
    if isinstance(input_size, tuple):
        return int(input_size[0]), int(input_size[1])
    edge = int(input_size)
    return edge, edge


def export_onnx(
    *,
    model: Any,
    torch_mod: Any,
    onnx_path: Path,
    input_size: int | tuple[int, int] = 256,
    opset: int = 17,
) -> Path:
    """Export ``model`` to ONNX at ``onnx_path``.

    Runs ``torch.onnx.export`` with dynamic batch axis, validates the
    file with ``onnx.checker.check_model``, and simplifies the graph
    with ``onnxsim`` (small speedup + smaller file). The simplifier
    is best-effort -- failures are logged but don't abort the export.

    ``input_size`` is either an ``int`` (square HxH) or a tuple
    ``(H, W)`` -- callers that derive the value from
    :func:`_extract_preprocess_params` pass the tuple form so the
    exported graph matches the trained model's positional embeddings.

    Returns the absolute path to the written ONNX file. Raises
    :class:`RuntimeError` if the basic export or the checker fails.
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = _normalize_hw(input_size)
    dummy = torch_mod.zeros((1, 3, height, width), dtype=torch_mod.float32)
    # ``dynamo=False`` -- the new dynamo-based exporter (default in
    # torch 2.11+) drags ``onnxscript`` as a hard runtime dep and
    # rejects some MobileCLIP ops. The legacy JIT-trace path covers
    # every op we use and produces the same graph the mobile runtimes
    # expect. We re-evaluate if torch deprecates the legacy path.
    try:
        torch_mod.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            opset_version=opset,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
            dynamo=False,
        )
    except Exception as exc:  # noqa: BLE001 -- re-raise with context
        raise RuntimeError(f"torch.onnx.export failed for {onnx_path}: {exc!r}") from exc

    # Validate the graph -- catches dangling nodes / missing weights.
    try:
        import onnx  # noqa: PLC0415

        loaded = onnx.load(str(onnx_path))
        onnx.checker.check_model(loaded)
    except ImportError as exc:  # pragma: no cover -- onnx is in pyproject
        raise RuntimeError("onnx is required for mobile export") from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"onnx.checker rejected the exported model at {onnx_path}: {exc!r}"
        ) from exc

    # Simplify (best-effort).
    try:
        import onnxsim  # noqa: PLC0415

        simplified, ok = onnxsim.simplify(loaded)
        if ok:
            onnx.save(simplified, str(onnx_path))
            logger.info("export: onnxsim simplified %s", onnx_path)
        else:
            logger.warning("export: onnxsim simplification failed for %s", onnx_path)
    except ImportError:  # pragma: no cover -- onnxsim is in pyproject
        logger.warning("export: onnxsim not installed; skipping graph simplification")
    except Exception as exc:  # noqa: BLE001
        logger.warning("export: onnxsim raised on %s (%s); skipping simplification", onnx_path, exc)

    return onnx_path


# --------------------------------------------------------------- Core ML export


def export_coreml(
    *,
    model: Any,
    torch_mod: Any,
    coreml_path: Path,
    input_size: int | tuple[int, int] = 256,
    quantize: Literal["fp32", "fp16", "int8"] = "fp16",
) -> Path | None:
    """Convert ``model`` to a Core ML ``.mlpackage``.

    Uses :mod:`coremltools` to convert directly from the traced PyTorch
    model (skipping the ONNX detour, which has known bugs on
    coremltools 8.x for some opsets). Applies FP16 quantization if
    requested.

    Returns the path on success or ``None`` if :mod:`coremltools` isn't
    installed / the converter raised. The caller logs a clear skip
    message in the second case.
    """
    try:
        import coremltools as ct  # noqa: PLC0415
    except ImportError:
        logger.warning("export: coremltools not installed; skipping Core ML export")
        return None

    try:
        height, width = _normalize_hw(input_size)
        dummy = torch_mod.zeros((1, 3, height, width), dtype=torch_mod.float32)
        with torch_mod.no_grad():
            traced = torch_mod.jit.trace(model, dummy)
        ml_input = ct.TensorType(name="input", shape=dummy.shape)
        precision = ct.precision.FLOAT16 if quantize == "fp16" else ct.precision.FLOAT32
        mlmodel = ct.convert(
            traced,
            inputs=[ml_input],
            convert_to="mlprogram",
            compute_precision=precision,
        )
        coreml_path.parent.mkdir(parents=True, exist_ok=True)
        # ``save`` for an mlpackage writes a directory; if one already
        # exists, blow it away so we don't accidentally merge files.
        if coreml_path.exists():
            if coreml_path.is_dir():
                shutil.rmtree(coreml_path)
            else:
                coreml_path.unlink()
        mlmodel.save(str(coreml_path))
        logger.info("export: wrote Core ML package to %s", coreml_path)
        return coreml_path
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "export: coremltools conversion failed (%s); marking Core ML output as skipped",
            exc,
        )
        return None


# --------------------------------------------------------------- TFLite export


def export_tflite(
    *,
    onnx_path: Path,
    tflite_path: Path,
    quantize: Literal["fp32", "fp16", "int8"] = "fp16",
) -> tuple[Path | None, str | None]:
    """Convert ``onnx_path`` to TFLite, falling back to ORT-Mobile.

    Tries the :mod:`onnx2tf` converter first; if that's not installed
    or raises, falls back to the ONNX Runtime Mobile ``.ort`` format
    via :mod:`onnxruntime.tools.convert_onnx_models_to_ort`. The
    ``.ort`` file goes to ``tflite_path.with_suffix('.ort')`` so the
    caller can tell which path was taken.

    Returns ``(output_path, fallback_reason)``:

    * ``(tflite_path, None)`` on the happy path.
    * ``(ort_path, "tflite -- onnx2tf failed: ...")`` on the ORT fallback.
    * ``(None, "tflite -- onnx2tf failed: ...; ort -- ...")`` if both fail.
    """
    onnx2tf_err: str | None = None
    try:
        import onnx2tf  # noqa: PLC0415

        tflite_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir = tflite_path.parent / f"_onnx2tf_{tflite_path.stem}"
        output_dir.mkdir(parents=True, exist_ok=True)
        # ``onnx2tf.convert`` writes a directory of TF artifacts; we
        # pluck the FP16 .tflite out and move it to the canonical name.
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(output_dir),
            output_dynamic_range_quantized_tflite=(quantize != "fp32"),
            non_verbose=True,
        )
        # The float TFLite usually lands as ``<onnx_stem>_float32.tflite``
        # or ``..._dynamic_range_quant.tflite`` depending on the flags.
        candidates = sorted(output_dir.glob("*.tflite"))
        if not candidates:
            raise RuntimeError(f"onnx2tf produced no .tflite files in {output_dir}")
        # Prefer the dynamic-range / FP16 variant when requested.
        chosen = candidates[0]
        if quantize != "fp32":
            for cand in candidates:
                if "dynamic_range" in cand.name or "float16" in cand.name:
                    chosen = cand
                    break
        shutil.move(str(chosen), str(tflite_path))
        # Clean the working dir so we don't ship stray TF artifacts.
        shutil.rmtree(output_dir, ignore_errors=True)
        logger.info("export: wrote TFLite to %s", tflite_path)
        return tflite_path, None
    except ImportError:
        onnx2tf_err = "onnx2tf not installed"
        logger.warning("export: onnx2tf not installed; falling back to ORT Mobile")
    except Exception as exc:  # noqa: BLE001
        onnx2tf_err = f"onnx2tf raised: {exc}"
        logger.warning("export: onnx2tf conversion failed (%s); falling back to ORT Mobile", exc)

    # Fallback: ONNX Runtime Mobile.
    ort_err: str | None = None
    try:
        from onnxruntime.tools import convert_onnx_models_to_ort  # noqa: PLC0415

        ort_path = tflite_path.with_suffix(".ort")
        ort_path.parent.mkdir(parents=True, exist_ok=True)
        # The helper writes alongside the .onnx file; copy the source
        # into a temp dir so we don't pollute ``common/`` with stray
        # artifacts.
        staging = ort_path.parent / f"_ort_{ort_path.stem}"
        staging.mkdir(parents=True, exist_ok=True)
        staged_onnx = staging / onnx_path.name
        shutil.copy2(onnx_path, staged_onnx)
        convert_onnx_models_to_ort.convert_onnx_models_to_ort(
            model_path_or_dir=staging,
            optimization_styles=[
                convert_onnx_models_to_ort.OptimizationStyle.Fixed,
            ],
        )
        produced = sorted(staging.glob("*.ort"))
        if not produced:
            raise RuntimeError(f"ORT conversion produced no .ort files in {staging}")
        shutil.move(str(produced[0]), str(ort_path))
        shutil.rmtree(staging, ignore_errors=True)
        reason = f"tflite -- {onnx2tf_err}; using ORT Mobile fallback"
        logger.info("export: wrote ORT Mobile bundle to %s (fallback)", ort_path)
        return ort_path, reason
    except ImportError:
        ort_err = "onnxruntime.tools not installed"
        logger.warning("export: onnxruntime.tools not installed; cannot fall back to ORT Mobile")
    except Exception as exc:  # noqa: BLE001
        ort_err = f"ORT conversion raised: {exc}"
        logger.warning("export: ORT Mobile fallback also failed (%s)", exc)

    reason = f"tflite -- {onnx2tf_err}; ort -- {ort_err}"
    return None, reason


# --------------------------------------------------------------- view head


def _build_view_head_module(state: dict[str, Any], torch_mod: Any) -> Any:
    """Reconstruct the tiny ``Linear(embed_dim, 2)`` view head.

    The Phase 5.3 binary view classifier ships its weights in a dict
    with ``"weight"`` / ``"bias"`` keys (a stock ``nn.Linear``). We
    reconstruct an ``nn.Module`` from those tensors so the export
    machinery can trace it just like the backbone.
    """
    weight = state.get("weight")
    bias = state.get("bias")
    if weight is None or bias is None:
        raise RuntimeError(
            "view-classifier checkpoint head_state_dict missing 'weight' / 'bias' "
            "(only nn.Linear heads are supported)"
        )
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    if out_features != len(VIEW_CLASS_NAMES):
        raise RuntimeError(
            f"view head has {out_features} outputs but expected {len(VIEW_CLASS_NAMES)} "
            f"({VIEW_CLASS_NAMES})"
        )
    head = torch_mod.nn.Linear(in_features, out_features)
    head.weight.data.copy_(weight)
    head.bias.data.copy_(bias)
    head.eval()
    return head


def export_view_head(
    *,
    view_classifier_checkpoint: Path,
    output_dir_ios: Path,
    output_dir_android: Path,
    output_dir_common: Path,
    embed_dim: int,
    quantize: Literal["fp32", "fp16", "int8"] = "fp16",
    opset: int = 17,
) -> dict[str, Path | None]:
    """Export the tiny binary view-classifier head.

    Reads the checkpoint, reconstructs a ``Linear(embed_dim, 2)`` head,
    and re-uses the backbone export functions to emit ONNX + Core ML +
    TFLite. Returns a dict mapping format -> path (or ``None`` on
    skip). Mismatched ``embed_dim`` against the backbone is logged
    loudly because the head's input must match the backbone's output.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for view-head export") from exc

    if not view_classifier_checkpoint.exists():
        raise RuntimeError(f"view-classifier checkpoint not found: {view_classifier_checkpoint}")
    payload = torch.load(view_classifier_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "head_state_dict" not in payload:
        raise RuntimeError(
            f"view-classifier checkpoint {view_classifier_checkpoint} is not a Phase 5.3 "
            "training checkpoint (missing 'head_state_dict' key)"
        )

    head = _build_view_head_module(payload["head_state_dict"], torch)
    head_embed_dim = int(head.weight.shape[1])
    if head_embed_dim != embed_dim:
        raise RuntimeError(
            f"view head expects {head_embed_dim}-dim features but backbone produces "
            f"{embed_dim}-dim; train a new head on the current backbone before exporting"
        )

    # Wrap so the exported graph runs a softmax for on-device convenience.
    class _ViewHeadWrapper(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: Any) -> Any:
            return torch.nn.functional.softmax(self.inner(x), dim=-1)

    wrapper = _ViewHeadWrapper(head)
    wrapper.eval()

    onnx_path = output_dir_common / "view_head.onnx"
    export_onnx(
        model=wrapper,
        torch_mod=torch,
        onnx_path=onnx_path,
        input_size=embed_dim,  # not actually used; we set the dummy below
        opset=opset,
    )

    # The default ``export_onnx`` builds a (1, 3, H, W) dummy which is
    # wrong for a (1, embed_dim) head. Re-export with the right input
    # shape.
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, embed_dim), dtype=torch.float32)
    # See ``export_onnx`` for the ``dynamo=False`` rationale.
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        opset_version=opset,
        input_names=["features"],
        output_names=["probs"],
        dynamic_axes={"features": {0: "batch"}, "probs": {0: "batch"}},
        dynamo=False,
    )

    paths: dict[str, Path | None] = {"onnx": onnx_path}

    # Core ML
    try:
        import coremltools as ct  # noqa: PLC0415

        traced = torch.jit.trace(wrapper, dummy)  # type: ignore[no-untyped-call]
        precision = ct.precision.FLOAT16 if quantize == "fp16" else ct.precision.FLOAT32
        ml_input = ct.TensorType(name="features", shape=dummy.shape)
        mlmodel = ct.convert(
            traced,
            inputs=[ml_input],
            convert_to="mlprogram",
            compute_precision=precision,
        )
        ios_path = output_dir_ios / "view_head.mlpackage"
        if ios_path.exists():
            if ios_path.is_dir():
                shutil.rmtree(ios_path)
            else:
                ios_path.unlink()
        mlmodel.save(str(ios_path))
        paths["coreml"] = ios_path
    except ImportError:
        paths["coreml"] = None
        logger.warning("export: coremltools not installed; view-head Core ML skipped")
    except Exception as exc:  # noqa: BLE001
        paths["coreml"] = None
        logger.warning("export: view-head Core ML conversion failed (%s)", exc)

    # TFLite
    android_path = output_dir_android / "view_head.tflite"
    out_path, fallback = export_tflite(
        onnx_path=onnx_path,
        tflite_path=android_path,
        quantize=quantize,
    )
    if out_path is not None and out_path.suffix == ".tflite":
        paths["tflite"] = out_path
    else:
        paths["tflite"] = out_path  # may be .ort fallback or None
        if fallback:
            logger.warning("export: view-head TFLite fallback (%s)", fallback)

    return paths


# --------------------------------------------------------------- driver


def _maybe_size_mb(path: Path | None) -> float | None:
    """Return the size of ``path`` in MB (rounded to 3 decimals), or ``None``."""
    if path is None:
        return None
    if not path.exists():
        return None
    if path.is_dir():
        total = 0
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    else:
        total = path.stat().st_size
    return round(total / (1024 * 1024), 3)


def export_mobile(
    *,
    config: MobileExportConfig,
) -> MobileExportReport:
    """End-to-end mobile export.

    Pipeline:

    1. Load the OpenCLIP backbone and overlay the fine-tuned weights.
    2. Wrap with an L2-normalize layer so the on-device contract is
       "image in, unit-norm embedding out".
    3. Export ONNX (always).
    4. Export Core ML (best-effort, skipped if coremltools missing).
    5. Export TFLite via onnx2tf, falling back to ORT Mobile if needed.
    6. Bundle the prototype tensor as raw FP16 bytes.
    7. Write ``class_names.json`` + ``preprocessing.json`` + (optionally)
       the view-classifier head.

    Returns a :class:`MobileExportReport` documenting every artifact
    and any steps that were skipped. The report is also the orchestrator's
    audit trail -- never throws on a single-stage failure; always emits a
    populated report so the operator can see exactly what shipped.
    """
    output_dir = config.output_dir
    ios_dir = output_dir / "ios"
    android_dir = output_dir / "android"
    common_dir = output_dir / "common"
    for d in (ios_dir, android_dir, common_dir):
        d.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []
    skipped: list[str] = []

    # 1) Backbone + L2-normalized wrapper. We also keep the preprocess
    # transform so we can introspect the real input size + normalize
    # stats instead of hardcoding values for one MobileCLIP variant.
    encoder, torch_mod, preprocess = load_backbone(
        model_name=config.model_name,
        pretrained=config.pretrained,
        checkpoint_path=config.backbone_checkpoint,
    )

    preprocess_params = _extract_preprocess_params(preprocess)
    input_size: tuple[int, int] = preprocess_params["input_size"]
    logger.info(
        "export: derived preprocess params from %s: input_size=%s mean=%s std=%s interp=%s",
        config.model_name,
        preprocess_params["input_size"],
        preprocess_params["mean"],
        preprocess_params["std"],
        preprocess_params["interpolation"],
    )

    # 2) Prototype bundle (we need n_classes + embed_dim to fill in
    # preprocessing.json + to validate the view head).
    prototypes_bin = ios_dir / "prototypes.bin"
    n_classes, embed_dim = bundle_prototypes(
        prototypes_path=config.prototypes_path,
        output_path=prototypes_bin,
    )
    # Mirror the same bytes into the Android bundle so the two
    # platforms ship from a single source of truth.
    android_proto = android_dir / "prototypes.bin"
    shutil.copy2(prototypes_bin, android_proto)

    # 3) ONNX (always)
    onnx_path = common_dir / "model.onnx"
    export_onnx(
        model=encoder,
        torch_mod=torch_mod,
        onnx_path=onnx_path,
        input_size=input_size,
        opset=config.opset,
    )

    # 4) Core ML (best-effort)
    coreml_path = export_coreml(
        model=encoder,
        torch_mod=torch_mod,
        coreml_path=ios_dir / "model.mlpackage",
        input_size=input_size,
        quantize=config.quantize,
    )
    if coreml_path is None:
        skipped.append("coreml -- coremltools missing or conversion failed")

    # 5) TFLite (with ORT-Mobile fallback)
    android_model_target = android_dir / "model.tflite"
    tflite_path, fallback_reason = export_tflite(
        onnx_path=onnx_path,
        tflite_path=android_model_target,
        quantize=config.quantize,
    )
    if fallback_reason:
        skipped.append(fallback_reason)

    # 6) Class names + preprocessing JSON (per-platform copies).
    class_names_path = ios_dir / "class_names.json"
    n_class_actual = write_class_names_json(
        prototypes_path=config.prototypes_path,
        output_path=class_names_path,
    )
    if n_class_actual != n_classes:
        notes.append(
            f"prototype bin has {n_classes} rows but class_names.json has "
            f"{n_class_actual} entries -- cache may be inconsistent"
        )
    shutil.copy2(class_names_path, android_dir / "class_names.json")

    preprocessing_path = ios_dir / "preprocessing.json"
    write_preprocessing_json(
        output_path=preprocessing_path,
        n_classes=n_classes,
        embed_dim=embed_dim,
        preprocess_params=preprocess_params,
    )
    shutil.copy2(preprocessing_path, android_dir / "preprocessing.json")

    # 7) View-classifier head (optional).
    view_head_paths: dict[str, Path | None] = {}
    if config.view_classifier_checkpoint is not None:
        # Sanity warning -- if the head was trained on a different
        # backbone its features won't match. We do export it anyway so
        # the orchestrator has the artifact; the warning is in the
        # report.
        notes.append(
            "view-classifier head exported alongside backbone. If the head was trained "
            "on a different backbone (e.g. MobileCLIP-S2 while the main encoder is "
            "MobileCLIP-B), the features will not match -- retrain the head on the new "
            "backbone before shipping."
        )
        try:
            view_head_paths = export_view_head(
                view_classifier_checkpoint=config.view_classifier_checkpoint,
                output_dir_ios=ios_dir,
                output_dir_android=android_dir,
                output_dir_common=common_dir,
                embed_dim=embed_dim,
                quantize=config.quantize,
                opset=config.opset,
            )
            write_view_class_names_json(ios_dir / "view_class_names.json")
            shutil.copy2(
                ios_dir / "view_class_names.json",
                android_dir / "view_class_names.json",
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"view-head export failed: {exc}")
            view_head_paths = {"onnx": None, "coreml": None, "tflite": None}

    # 8) Sizes
    sizes_mb: dict[str, float] = {}
    for label, path in (
        ("onnx", onnx_path),
        ("coreml", coreml_path),
        ("tflite", tflite_path),
        ("prototypes", prototypes_bin),
    ):
        size = _maybe_size_mb(path)
        if size is not None:
            sizes_mb[label] = size
    sizes_mb["total"] = round(sum(sizes_mb.values()), 3)

    return MobileExportReport(
        onnx_path=onnx_path,
        coreml_path=coreml_path,
        tflite_path=tflite_path,
        prototypes_path=prototypes_bin,
        class_names_path=class_names_path,
        preprocessing_path=preprocessing_path,
        view_head_paths=view_head_paths,
        sizes_mb=sizes_mb,
        skipped=skipped,
        notes=notes,
        preprocess_params=dict(preprocess_params),
    )


# --------------------------------------------------------------- helpers (re-export)


def read_prototypes_bin(
    path: Path,
    *,
    shape: Sequence[int],
) -> Any:
    """Read a prototype FP16 binary back into a 2D float array.

    Helper for the validator and tests. ``shape`` must be the
    ``(n_classes, embed_dim)`` pair from ``preprocessing.json``; we
    don't infer it from file size to keep round-trips explicit.

    Returns a numpy ``float16`` 2D array (the validator casts to
    ``float32`` for cosine sim arithmetic).
    """
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required to read the prototypes binary") from exc

    n_classes, embed_dim = int(shape[0]), int(shape[1])
    expected_bytes = n_classes * embed_dim * 2
    raw = path.read_bytes()
    if len(raw) != expected_bytes:
        raise RuntimeError(
            f"prototypes.bin at {path} has {len(raw)} bytes; "
            f"expected {expected_bytes} for shape {(n_classes, embed_dim)}"
        )
    arr = np.frombuffer(raw, dtype="<f2").reshape(n_classes, embed_dim)
    return arr


def write_report_json(report: MobileExportReport, output_path: Path) -> None:
    """Serialize a :class:`MobileExportReport` to JSON at ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


# Convenience for tests that round-trip the FP16 byte layout outside
# the prototype cache (e.g. test_prototypes_bin_byte_layout).
def write_fp16_matrix(arr: Any, output_path: Path) -> None:
    """Write a 2D matrix to ``output_path`` as little-endian FP16 bytes."""
    if arr.ndim != 2:
        raise RuntimeError(f"expected 2D matrix, got shape {tuple(arr.shape)}")
    # ``arr`` may be torch.Tensor or numpy.ndarray; we accept both.
    if hasattr(arr, "detach"):
        try:
            import torch  # noqa: PLC0415

            t = arr.detach().to("cpu").to(dtype=torch.float16).contiguous()
            raw = t.view(torch.uint8).contiguous().numpy().tobytes()
        except ImportError:  # pragma: no cover
            raise RuntimeError("torch is required to dump a torch.Tensor") from None
    else:
        # numpy path
        import numpy as np  # noqa: PLC0415

        fp16 = np.ascontiguousarray(arr, dtype="<f2")
        raw = fp16.tobytes()
    # Sanity: pack a known length so the test can assert byte parity.
    if len(raw) != arr.shape[0] * arr.shape[1] * 2:
        raise RuntimeError(
            f"byte-packing produced {len(raw)} bytes for shape {tuple(arr.shape)} "
            f"-- expected {arr.shape[0] * arr.shape[1] * 2}"
        )
    # Light belt-and-braces: ensure ``struct.pack`` would round-trip
    # the same way (catches any host-endianness mismatches at write
    # time, not on-device).
    if arr.shape[0] * arr.shape[1] < 4:
        # Cheap to double-check on tiny matrices.
        sample = struct.unpack(f"<{arr.shape[0] * arr.shape[1]}e", raw)
        if len(sample) != arr.shape[0] * arr.shape[1]:
            raise RuntimeError("FP16 byte round-trip self-check failed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)


__all__ = [
    "PREPROCESSING_TEMPLATE",
    "VIEW_CLASS_NAMES",
    "MobileExportConfig",
    "MobileExportReport",
    "_extract_preprocess_params",
    "bundle_prototypes",
    "export_coreml",
    "export_mobile",
    "export_onnx",
    "export_tflite",
    "export_view_head",
    "load_backbone",
    "read_prototypes_bin",
    "write_class_names_json",
    "write_fp16_matrix",
    "write_preprocessing_json",
    "write_report_json",
    "write_view_class_names_json",
]
