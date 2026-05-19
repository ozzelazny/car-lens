"""Mobile export validator (Phase 5.5).

Run PyTorch and each available exported model side-by-side on a small
batch of real test images and assert that:

1. Per-image embedding cosine similarity (PyTorch vs exported) has mean
   above 0.998 and minimum above 0.99.
2. The top-5 set against the loaded prototype matrix is identical for
   at least 98% of images.
3. The top-1 prediction is identical for at least 95% of images.

This is the parity check that gates "the mobile bundle ships". A
failure here means we'd be regressing accuracy on-device even though
all the converters succeeded.

The validator is intentionally light on infrastructure -- one helper
per runtime (ONNX Runtime, Core ML, TFLite), each guarded with a lazy
import so the test machine can validate whatever subset of the
exported formats is installed locally.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


COSINE_MEAN_MIN = 0.998
COSINE_MIN_MIN = 0.99
TOP5_IDENTITY_MIN = 0.98
TOP1_IDENTITY_MIN = 0.95


@dataclass
class FormatValidationResult:
    """One row of the validator's report (one format vs PyTorch)."""

    format: str
    cosine_mean: float
    cosine_min: float
    top1_identity: float
    top5_identity: float
    mean_latency_ms: float
    passed: bool
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """End-of-run summary across every available exported format."""

    pytorch_image_count: int
    results: list[FormatValidationResult] = field(default_factory=list)
    overall_passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "pytorch_image_count": self.pytorch_image_count,
            "overall_passed": self.overall_passed,
            "results": [
                {
                    "format": r.format,
                    "cosine_mean": r.cosine_mean,
                    "cosine_min": r.cosine_min,
                    "top1_identity": r.top1_identity,
                    "top5_identity": r.top5_identity,
                    "mean_latency_ms": r.mean_latency_ms,
                    "passed": r.passed,
                    "failure_reasons": list(r.failure_reasons),
                }
                for r in self.results
            ],
        }


# --------------------------------------------------------------- numeric helpers


def cosine_similarity_per_row(a: Any, b: Any) -> Any:
    """Per-row cosine similarity between two ``(N, D)`` arrays.

    Accepts numpy or torch tensors (whichever the caller has handy);
    returns a 1D array/tensor of length ``N``. Both inputs are assumed
    to be float (any precision); we don't normalize inside this
    function because both sides are expected to be unit-norm already.
    """
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for cosine similarity") from exc

    a_np = _to_numpy_f32(a)
    b_np = _to_numpy_f32(b)
    if a_np.shape != b_np.shape:
        raise RuntimeError(
            f"cosine_similarity_per_row: shape mismatch {a_np.shape} vs {b_np.shape}"
        )
    num = (a_np * b_np).sum(axis=-1)
    den = (np.linalg.norm(a_np, axis=-1) * np.linalg.norm(b_np, axis=-1)) + 1e-12
    return num / den


def topk_indices(matrix: Any, k: int) -> Any:
    """Top-K column indices per row of ``matrix`` (descending).

    Returns a numpy int64 array of shape ``(N, k)``.
    """
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for topk") from exc

    arr = _to_numpy_f32(matrix)
    if arr.ndim != 2:
        raise RuntimeError(f"topk_indices expects 2D matrix, got shape {arr.shape}")
    k_eff = min(k, arr.shape[1])
    # ``argpartition`` is faster than full sort; we then sort the
    # top-k slice descending for stable comparison.
    top_unsorted = np.argpartition(-arr, kth=k_eff - 1, axis=-1)[:, :k_eff]
    rows = np.arange(arr.shape[0])[:, None]
    top_vals = arr[rows, top_unsorted]
    order = np.argsort(-top_vals, axis=-1)
    return top_unsorted[rows, order]


def top1_identity(a: Any, b: Any) -> float:
    """Fraction of rows where the argmax matches between ``a`` and ``b``."""
    import numpy as np  # noqa: PLC0415

    a_np = _to_numpy_f32(a)
    b_np = _to_numpy_f32(b)
    a_top = np.argmax(a_np, axis=-1)
    b_top = np.argmax(b_np, axis=-1)
    return float(np.mean(a_top == b_top))


def top5_identity(a: Any, b: Any, k: int = 5) -> float:
    """Fraction of rows where the top-K *sets* match between ``a`` and ``b``."""
    a_top = topk_indices(a, k=k)
    b_top = topk_indices(b, k=k)
    if a_top.shape != b_top.shape:
        return 0.0
    matches = 0
    for i in range(a_top.shape[0]):
        if set(a_top[i].tolist()) == set(b_top[i].tolist()):
            matches += 1
    return float(matches) / float(max(1, a_top.shape[0]))


def _to_numpy_f32(x: Any) -> Any:
    """Convert torch / numpy / list inputs into a contiguous ``float32`` array."""
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for the validator") from exc

    if hasattr(x, "detach"):  # torch tensor
        return x.detach().to("cpu").to(dtype_for_fp32(x)).numpy().astype("float32", copy=False)
    arr = np.asarray(x)
    if arr.dtype != np.float32:
        arr = arr.astype("float32", copy=False)
    return np.ascontiguousarray(arr)


def dtype_for_fp32(x: Any) -> Any:
    """Return the torch float32 dtype symbol without importing torch eagerly."""
    import torch  # noqa: PLC0415

    return torch.float32


def evaluate_parity(
    *,
    pytorch_embeddings: Any,
    exported_embeddings: Any,
    pytorch_sims: Any,
    exported_sims: Any,
    format_name: str,
    mean_latency_ms: float,
) -> FormatValidationResult:
    """Score one (format, embedding-batch) pair against the thresholds.

    Inputs:

    * ``pytorch_embeddings``: ``(N, D)`` reference (unit-norm) embeddings.
    * ``exported_embeddings``: ``(N, D)`` exported embeddings.
    * ``pytorch_sims``: ``(N, C)`` PyTorch ``embedding @ prototypes.T``.
    * ``exported_sims``: ``(N, C)`` exported ``embedding @ prototypes.T``.
    * ``format_name``: the label used in the report (e.g. ``"onnx"``).
    * ``mean_latency_ms``: the timed forward latency for the exported
      runtime.

    Returns a populated :class:`FormatValidationResult` whose
    ``passed`` flag reflects the four thresholds. ``failure_reasons``
    lists every threshold that was violated, in stable order.
    """
    sims = cosine_similarity_per_row(pytorch_embeddings, exported_embeddings)
    cos_mean = float(sims.mean())
    cos_min = float(sims.min())
    t1 = top1_identity(pytorch_sims, exported_sims)
    t5 = top5_identity(pytorch_sims, exported_sims)

    reasons: list[str] = []
    if cos_mean < COSINE_MEAN_MIN:
        reasons.append(f"cosine_mean={cos_mean:.4f} < {COSINE_MEAN_MIN} (embedding parity broken)")
    if cos_min < COSINE_MIN_MIN:
        reasons.append(f"cosine_min={cos_min:.4f} < {COSINE_MIN_MIN} (at least one image diverges)")
    if t1 < TOP1_IDENTITY_MIN:
        reasons.append(f"top1_identity={t1:.4f} < {TOP1_IDENTITY_MIN} (predictions differ)")
    if t5 < TOP5_IDENTITY_MIN:
        reasons.append(f"top5_identity={t5:.4f} < {TOP5_IDENTITY_MIN} (top-5 sets differ)")

    return FormatValidationResult(
        format=format_name,
        cosine_mean=cos_mean,
        cosine_min=cos_min,
        top1_identity=t1,
        top5_identity=t5,
        mean_latency_ms=mean_latency_ms,
        passed=not reasons,
        failure_reasons=reasons,
    )


# --------------------------------------------------------------- runtimes


def run_onnxruntime(
    *,
    onnx_path: Path,
    batches: Sequence[Any],
    warmup: int = 10,
    timed: int = 100,
) -> tuple[Any, float]:
    """Run ``onnx_path`` on each batch via :mod:`onnxruntime`.

    Returns ``(embeddings, mean_latency_ms)`` where ``embeddings`` is a
    numpy ``(N, D)`` float32 array (stacked across batches). The
    latency is the mean of ``timed`` single-image forwards run after
    ``warmup`` warmup forwards on the first batch.
    """
    try:
        import numpy as np  # noqa: PLC0415
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("onnxruntime is required to validate the ONNX export") from exc

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    chunks: list[Any] = []
    for batch in batches:
        feed = {input_name: _to_numpy_f32(batch)}
        out = sess.run(None, feed)[0]
        chunks.append(out)
    embeddings = np.concatenate(chunks, axis=0)

    # Latency measurement (single-image forwards on a 1xCxHxW slice).
    first = _to_numpy_f32(batches[0])
    one = first[:1]
    for _ in range(warmup):
        sess.run(None, {input_name: one})
    t0 = time.perf_counter()
    for _ in range(timed):
        sess.run(None, {input_name: one})
    elapsed = time.perf_counter() - t0
    mean_ms = (elapsed / max(1, timed)) * 1000.0
    return embeddings, mean_ms


# --------------------------------------------------------------- driver


@dataclass
class ValidationInputs:
    """Inputs the validator needs from the caller / CLI."""

    onnx_path: Path | None
    coreml_path: Path | None
    tflite_path: Path | None
    prototypes_path: Path
    image_paths: list[Path]
    proto_shape: tuple[int, int]


def validate_export(
    *,
    inputs: ValidationInputs,
    pytorch_embeddings: Any,
) -> ValidationReport:
    """Cross-check every available exported format against PyTorch.

    The caller is responsible for running the PyTorch model and
    handing the embeddings + prototypes over. This keeps the heavy
    lifting (model load + image preprocessing) in one place outside
    the validator's purity boundary.

    Returns a :class:`ValidationReport` populated with one result per
    format that was actually loadable.
    """
    # numpy is required transitively for ``read_prototypes_bin`` and the
    # parity helpers below -- ensure it's importable up-front so the
    # caller sees a clear error.
    try:
        import numpy  # noqa: F401, PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for the validator") from exc

    from .mobile import read_prototypes_bin  # noqa: PLC0415

    proto_path = inputs.prototypes_path
    prototypes_fp16 = read_prototypes_bin(proto_path, shape=inputs.proto_shape)
    prototypes = prototypes_fp16.astype("float32", copy=False)

    pytorch_np = _to_numpy_f32(pytorch_embeddings)
    pytorch_sims = pytorch_np @ prototypes.T

    n_images = pytorch_np.shape[0]
    report = ValidationReport(pytorch_image_count=n_images)

    # ONNX is the only runtime we can run in tests without the heavy
    # mobile dependencies; iOS / TFLite runtimes are best-effort.
    if inputs.onnx_path is not None:
        try:
            # Caller loaded images already; we expect ``pytorch_embeddings``
            # was computed from the same preprocessed tensor in the caller.
            # The ONNX runner needs the preprocessed input tensors; we
            # reconstruct them lazily because the validator's contract
            # is "pass me the same images". For simplicity in this
            # round of validation, we re-load images here via the
            # caller-provided ``image_paths`` and let the runner
            # handle preprocessing. The heavy-image-load path is in
            # :func:`load_images_as_batch` below.
            onnx_emb, onnx_latency = run_onnxruntime(
                onnx_path=inputs.onnx_path,
                batches=[load_images_as_batch(inputs.image_paths)],
            )
            onnx_sims = onnx_emb @ prototypes.T
            result = evaluate_parity(
                pytorch_embeddings=pytorch_np,
                exported_embeddings=onnx_emb,
                pytorch_sims=pytorch_sims,
                exported_sims=onnx_sims,
                format_name="onnx",
                mean_latency_ms=onnx_latency,
            )
        except Exception as exc:  # noqa: BLE001
            result = FormatValidationResult(
                format="onnx",
                cosine_mean=0.0,
                cosine_min=0.0,
                top1_identity=0.0,
                top5_identity=0.0,
                mean_latency_ms=0.0,
                passed=False,
                failure_reasons=[f"onnx runtime failed to load/run: {exc}"],
            )
        report.results.append(result)

    # Core ML / TFLite runtimes are platform-specific (and the
    # corresponding Python loaders require macOS / TF respectively).
    # We skip them in the cross-platform validator path; the
    # orchestrator runs them on the macOS / Linux+TF machines that
    # actually have the runtimes installed.
    if inputs.coreml_path is not None:
        report.results.append(
            _try_coreml(
                coreml_path=inputs.coreml_path,
                image_paths=inputs.image_paths,
                pytorch_np=pytorch_np,
                pytorch_sims=pytorch_sims,
                prototypes=prototypes,
            )
        )
    if inputs.tflite_path is not None:
        report.results.append(
            _try_tflite(
                tflite_path=inputs.tflite_path,
                image_paths=inputs.image_paths,
                pytorch_np=pytorch_np,
                pytorch_sims=pytorch_sims,
                prototypes=prototypes,
            )
        )

    report.overall_passed = all(r.passed for r in report.results) if report.results else False
    return report


def _try_coreml(
    *,
    coreml_path: Path,
    image_paths: list[Path],
    pytorch_np: Any,
    pytorch_sims: Any,
    prototypes: Any,
) -> FormatValidationResult:
    """Best-effort Core ML parity check."""
    try:
        import coremltools as ct  # noqa: PLC0415
    except ImportError:
        return FormatValidationResult(
            format="coreml",
            cosine_mean=0.0,
            cosine_min=0.0,
            top1_identity=0.0,
            top5_identity=0.0,
            mean_latency_ms=0.0,
            passed=False,
            failure_reasons=["coremltools not installed; cannot validate Core ML bundle"],
        )

    try:
        model = ct.models.MLModel(str(coreml_path))
        batch = load_images_as_batch(image_paths)
        # Core ML expects per-image dicts; iterate one-by-one for parity.
        outs: list[Any] = []
        for i in range(batch.shape[0]):
            single = batch[i : i + 1]
            result = model.predict({"input": single})
            # The output key follows the input name convention; the
            # exporter uses "embedding".
            emb = next(iter(result.values()))
            outs.append(emb)
        import numpy as np  # noqa: PLC0415

        emb_arr = np.concatenate([np.asarray(x).reshape(1, -1) for x in outs], axis=0)
        sims = emb_arr @ prototypes.T

        t0 = time.perf_counter()
        for _ in range(50):
            model.predict({"input": batch[:1]})
        latency = ((time.perf_counter() - t0) / 50.0) * 1000.0
        return evaluate_parity(
            pytorch_embeddings=pytorch_np,
            exported_embeddings=emb_arr,
            pytorch_sims=pytorch_sims,
            exported_sims=sims,
            format_name="coreml",
            mean_latency_ms=latency,
        )
    except Exception as exc:  # noqa: BLE001
        return FormatValidationResult(
            format="coreml",
            cosine_mean=0.0,
            cosine_min=0.0,
            top1_identity=0.0,
            top5_identity=0.0,
            mean_latency_ms=0.0,
            passed=False,
            failure_reasons=[f"coreml runtime failed: {exc}"],
        )


def _try_tflite(
    *,
    tflite_path: Path,
    image_paths: list[Path],
    pytorch_np: Any,
    pytorch_sims: Any,
    prototypes: Any,
) -> FormatValidationResult:
    """Best-effort TFLite parity check."""
    try:
        import tensorflow as tf  # noqa: PLC0415
    except ImportError:
        return FormatValidationResult(
            format="tflite",
            cosine_mean=0.0,
            cosine_min=0.0,
            top1_identity=0.0,
            top5_identity=0.0,
            mean_latency_ms=0.0,
            passed=False,
            failure_reasons=["tensorflow not installed; cannot validate TFLite bundle"],
        )

    try:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        in_detail = interpreter.get_input_details()[0]
        out_detail = interpreter.get_output_details()[0]
        batch = load_images_as_batch(image_paths)
        outs: list[Any] = []
        for i in range(batch.shape[0]):
            single = batch[i : i + 1].astype(in_detail["dtype"], copy=False)
            interpreter.set_tensor(in_detail["index"], single)
            interpreter.invoke()
            outs.append(interpreter.get_tensor(out_detail["index"]))
        import numpy as np  # noqa: PLC0415

        emb_arr = np.concatenate(outs, axis=0)
        sims = emb_arr @ prototypes.T

        t0 = time.perf_counter()
        for _ in range(50):
            interpreter.set_tensor(in_detail["index"], batch[:1].astype(in_detail["dtype"]))
            interpreter.invoke()
        latency = ((time.perf_counter() - t0) / 50.0) * 1000.0
        return evaluate_parity(
            pytorch_embeddings=pytorch_np,
            exported_embeddings=emb_arr,
            pytorch_sims=pytorch_sims,
            exported_sims=sims,
            format_name="tflite",
            mean_latency_ms=latency,
        )
    except Exception as exc:  # noqa: BLE001
        return FormatValidationResult(
            format="tflite",
            cosine_mean=0.0,
            cosine_min=0.0,
            top1_identity=0.0,
            top5_identity=0.0,
            mean_latency_ms=0.0,
            passed=False,
            failure_reasons=[f"tflite runtime failed: {exc}"],
        )


# --------------------------------------------------------------- image loading


def load_images_as_batch(
    image_paths: Sequence[Path],
    *,
    input_size: int = 256,
) -> Any:
    """Load + preprocess ``image_paths`` into a ``(N, 3, H, W)`` float32 array.

    Uses the OpenCLIP MobileCLIP-B preprocessing recipe: bicubic resize
    to ``input_size`` shortest side, center-crop to ``(input_size,
    input_size)``, normalize to mean=0.5 / std=0.5. Implementation is
    a thin PIL + numpy pipeline so the validator doesn't drag
    torchvision / open_clip into the test path.
    """
    try:
        import numpy as np  # noqa: PLC0415
        from PIL import Image as PILImage  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PIL + numpy required to load validation images") from exc

    tensors: list[Any] = []
    for path in image_paths:
        with PILImage.open(path) as img:
            rgb = img.convert("RGB")
            # Resize shortest side to ``input_size``.
            w, h = rgb.size
            scale = input_size / float(min(w, h))
            new_w, new_h = (
                max(input_size, int(round(w * scale))),
                max(input_size, int(round(h * scale))),
            )
            # PIL 10+ moved interpolation constants under
            # ``Image.Resampling``; fall back to the legacy
            # ``Image.BICUBIC`` attribute for older Pillow versions.
            bicubic = getattr(
                getattr(PILImage, "Resampling", PILImage),
                "BICUBIC",
                getattr(PILImage, "BICUBIC", 3),
            )
            rgb = rgb.resize((new_w, new_h), bicubic)
            # Center crop.
            left = (new_w - input_size) // 2
            top = (new_h - input_size) // 2
            rgb = rgb.crop((left, top, left + input_size, top + input_size))
        arr = np.asarray(rgb, dtype="float32") / 255.0
        arr = (arr - 0.5) / 0.5  # mean=0.5, std=0.5
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        tensors.append(arr)
    return np.stack(tensors, axis=0)


def write_report_json(report: ValidationReport, output_path: Path) -> None:
    """Serialize a :class:`ValidationReport` to JSON at ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


# --------------------------------------------------------------- CLI


DEFAULT_NUM_IMAGES = 100


def _sample_test_image_paths(
    conn: sqlite3.Connection,
    *,
    source: str,
    split: str,
    num_images: int,
) -> list[Path]:
    """Sample ``num_images`` image paths from the test split."""
    cur = conn.execute(
        "SELECT images.local_path FROM images "
        "JOIN listings ON images.listing_id = listings.listing_id "
        "WHERE listings.source = ? AND images.split = ? "
        "AND images.local_path IS NOT NULL "
        "ORDER BY images.image_id "
        "LIMIT ?",
        (source, split, int(num_images)),
    )
    paths: list[Path] = []
    for row in cur.fetchall():
        p = row[0] if not hasattr(row, "keys") else row["local_path"]
        if p:
            paths.append(Path(str(p)))
    return paths


def _build_arg_parser() -> Any:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="validate-mobile-export",
        description=(
            "Phase 5.5 validator: compare PyTorch and exported model outputs on "
            "a small batch of test images. Outputs a JSON parity report."
        ),
    )
    parser.add_argument("--onnx-path", type=Path, default=None)
    parser.add_argument("--coreml-path", type=Path, default=None)
    parser.add_argument("--tflite-path", type=Path, default=None)
    parser.add_argument(
        "--prototypes-path",
        type=Path,
        required=True,
        help="path to the FP16 prototypes.bin produced by export-mobile",
    )
    parser.add_argument(
        "--preprocessing-path",
        type=Path,
        required=True,
        help="path to preprocessing.json (used to read the prototype shape)",
    )
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=str, default="MobileCLIP-B")
    parser.add_argument("--pretrained", type=str, default="datacompdr")
    parser.add_argument("--db", type=Path, default=Path("db/crawl.sqlite"))
    parser.add_argument("--source", type=str, default="compcars")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument(
        "--image-paths",
        type=Path,
        default=None,
        help="optional file containing one image path per line (overrides --db sampling)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("cpu", "cuda", "mps"),
        default="cpu",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/phase5_5_mobile_validate.json"),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``validate-mobile-export`` console script."""
    from car_lense_engine.db import open_db  # noqa: PLC0415

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pre = json.loads(args.preprocessing_path.read_text(encoding="utf-8"))
    shape = pre["prototypes_bin"]["shape"]

    if args.image_paths is not None:
        image_paths = [
            Path(line.strip())
            for line in args.image_paths.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        conn = open_db(args.db)
        try:
            image_paths = _sample_test_image_paths(
                conn,
                source=args.source,
                split=args.test_split,
                num_images=args.num_images,
            )
        finally:
            conn.close()

    if not image_paths:
        parser.error("no image paths to validate against")

    # Run PyTorch reference.
    from .mobile import load_backbone  # noqa: PLC0415

    encoder, torch_mod, _preprocess = load_backbone(
        model_name=args.model,
        pretrained=args.pretrained,
        checkpoint_path=args.backbone_checkpoint,
        device=args.device,
    )
    encoder.to(args.device)
    encoder.eval()
    batch = load_images_as_batch(image_paths)
    tens = torch_mod.from_numpy(batch).to(args.device)
    with torch_mod.no_grad():
        pytorch_embeddings = encoder(tens).cpu().numpy()

    report = validate_export(
        inputs=ValidationInputs(
            onnx_path=args.onnx_path,
            coreml_path=args.coreml_path,
            tflite_path=args.tflite_path,
            prototypes_path=args.prototypes_path,
            image_paths=image_paths,
            proto_shape=(int(shape[0]), int(shape[1])),
        ),
        pytorch_embeddings=pytorch_embeddings,
    )
    write_report_json(report, args.report)
    print(
        "validate-mobile-export: overall="
        f"{'PASS' if report.overall_passed else 'FAIL'} "
        f"({len(report.results)} formats checked)"
    )
    for result in report.results:
        flag = "PASS" if result.passed else "FAIL"
        print(
            f"  {result.format}: {flag} cos_mean={result.cosine_mean:.4f} "
            f"cos_min={result.cosine_min:.4f} top1={result.top1_identity:.4f} "
            f"top5={result.top5_identity:.4f} latency={result.mean_latency_ms:.2f}ms"
        )
        for reason in result.failure_reasons:
            print(f"    - {reason}")
    print(f"validate-mobile-export: report written to {args.report}")
    return 0 if report.overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "COSINE_MEAN_MIN",
    "COSINE_MIN_MIN",
    "TOP1_IDENTITY_MIN",
    "TOP5_IDENTITY_MIN",
    "FormatValidationResult",
    "ValidationInputs",
    "ValidationReport",
    "cosine_similarity_per_row",
    "evaluate_parity",
    "load_images_as_batch",
    "run_onnxruntime",
    "top1_identity",
    "top5_identity",
    "topk_indices",
    "validate_export",
    "write_report_json",
]
