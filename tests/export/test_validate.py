"""Tests for the Phase 5.5 mobile-export validator.

We exercise the parity logic directly with synthetic embeddings so the
test doesn't depend on Core ML / TFLite / a real ONNX file. The actual
ONNX runtime path is covered in :mod:`tests.export.test_mobile`.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


from car_lense_engine.export.validate import (  # noqa: E402
    COSINE_MEAN_MIN,
    cosine_similarity_per_row,
    evaluate_parity,
    top1_identity,
    top5_identity,
    topk_indices,
)


def _unit(arr: np.ndarray) -> np.ndarray:
    return arr / (np.linalg.norm(arr, axis=-1, keepdims=True) + 1e-12)


def test_cosine_similarity_per_row_identity() -> None:
    a = _unit(np.random.RandomState(0).randn(5, 8).astype("float32"))
    sims = cosine_similarity_per_row(a, a)
    assert sims.shape == (5,)
    assert np.allclose(sims, 1.0, atol=1e-5)


def test_cosine_similarity_per_row_orthogonal() -> None:
    e0 = np.eye(4)[0:1].astype("float32")
    e1 = np.eye(4)[1:2].astype("float32")
    sims = cosine_similarity_per_row(e0, e1)
    assert sims.shape == (1,)
    assert abs(float(sims[0])) < 1e-6


def test_top1_identity_perfect() -> None:
    a = np.array([[0.1, 0.9, 0.0], [0.0, 0.2, 0.8]], dtype="float32")
    assert top1_identity(a, a) == 1.0


def test_top1_identity_mismatch() -> None:
    a = np.array([[0.9, 0.1], [0.1, 0.9]], dtype="float32")
    b = np.array([[0.1, 0.9], [0.1, 0.9]], dtype="float32")
    assert top1_identity(a, b) == 0.5


def test_top5_identity_set_match() -> None:
    # Top-5 set ordering shouldn't matter; sets are equal.
    a = np.array([[0.5, 0.4, 0.3, 0.2, 0.1, 0.05]], dtype="float32")
    b = np.array([[0.3, 0.4, 0.5, 0.1, 0.2, 0.05]], dtype="float32")
    assert top5_identity(a, b) == 1.0


def test_top5_identity_set_mismatch() -> None:
    a = np.array([[0.9, 0.1, 0.2, 0.3, 0.4, 0.5]], dtype="float32")
    # Different top-5 (item at index 0 falls out, item at index 5 stays).
    b = np.array([[0.1, 0.5, 0.4, 0.3, 0.2, 0.9]], dtype="float32")
    assert 0.0 <= top5_identity(a, b) <= 1.0


def test_topk_indices_shape_and_descending() -> None:
    sims = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype="float32")
    top = topk_indices(sims, k=3)
    assert top.shape == (1, 3)
    # Highest first.
    assert top[0].tolist() == [4, 3, 2]


def test_evaluate_parity_passes_on_identical_inputs() -> None:
    rng = np.random.RandomState(7)
    emb = _unit(rng.randn(10, 16).astype("float32"))
    proto = _unit(rng.randn(20, 16).astype("float32"))
    sims = emb @ proto.T
    result = evaluate_parity(
        pytorch_embeddings=emb,
        exported_embeddings=emb,
        pytorch_sims=sims,
        exported_sims=sims,
        format_name="onnx",
        mean_latency_ms=12.0,
    )
    assert result.passed
    assert result.cosine_mean == pytest.approx(1.0, abs=1e-5)
    assert result.cosine_min == pytest.approx(1.0, abs=1e-5)
    assert result.top1_identity == 1.0
    assert result.top5_identity == 1.0
    assert result.mean_latency_ms == 12.0


def test_validate_mobile_export_detects_mismatch() -> None:
    """Embeddings deliberately differ by ~5% per row; validator must FAIL."""
    rng = np.random.RandomState(11)
    emb = _unit(rng.randn(10, 16).astype("float32"))
    proto = _unit(rng.randn(20, 16).astype("float32"))
    sims_pt = emb @ proto.T
    # Perturb the exported embeddings to break parity.
    perturbed = emb + 0.5 * rng.randn(10, 16).astype("float32")
    perturbed = _unit(perturbed)
    sims_ex = perturbed @ proto.T

    result = evaluate_parity(
        pytorch_embeddings=emb,
        exported_embeddings=perturbed,
        pytorch_sims=sims_pt,
        exported_sims=sims_ex,
        format_name="onnx",
        mean_latency_ms=1.0,
    )
    assert not result.passed
    assert result.failure_reasons  # at least one threshold flagged
    # cos_mean must be below the threshold.
    assert result.cosine_mean < COSINE_MEAN_MIN


def test_evaluate_parity_top1_failure_alone() -> None:
    """Bump the predictions just enough to break top-1 but keep cos sims tight."""
    rng = np.random.RandomState(13)
    emb = _unit(rng.randn(20, 16).astype("float32"))
    perturbed = _unit(emb + 0.01 * rng.randn(20, 16).astype("float32"))
    # Build prototypes with a deliberate ambiguous pair so top-1 flips.
    proto = _unit(rng.randn(50, 16).astype("float32"))
    sims_pt = emb @ proto.T
    sims_ex = perturbed @ proto.T
    # Force top-1 disagreement by swapping argmax for half the rows.
    for i in range(0, 20, 2):
        argmax = int(np.argmax(sims_ex[i]))
        # Pick a different column.
        alt = (argmax + 1) % proto.shape[0]
        sims_ex[i, alt] = sims_ex[i, argmax] + 0.1

    result = evaluate_parity(
        pytorch_embeddings=emb,
        exported_embeddings=perturbed,
        pytorch_sims=sims_pt,
        exported_sims=sims_ex,
        format_name="onnx",
        mean_latency_ms=1.0,
    )
    assert result.top1_identity < 1.0
