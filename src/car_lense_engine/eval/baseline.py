"""Zero-shot prototype-retrieval baseline (Phase 5.1).

This module produces the first concrete accuracy number for Car Lense: a
pre-trained MobileCLIP-S2 backbone is used to embed every train image of a
given source/split, mean-pool per class (with L2 normalization on both ends)
to form one prototype per class, and then top-K nearest-prototype retrieval
is evaluated against a held-out test set.

Design notes
------------

* **Single prototype per class.** No view conditioning -- that's Phase 5.2.
  A class is identified by the tuple ``(year, make, model)``; rows where
  any of those is NULL are excluded. The class id is rendered as the
  lower-cased ``"year|make|model"`` string and is what shows up in the
  report.

* **Mean of L2-normalized embeddings.** Standard CLIP retrieval recipe:
  L2-normalize each image embedding, take the unweighted mean across all
  train images for the class, then L2-normalize the mean. The cosine
  similarity against a test embedding is then a plain dot product.

* **Stateless.** No on-disk embedding cache; re-runs re-embed. For ~16k
  images this is acceptable for a baseline pass (single-digit hours on CPU)
  and avoids the cache-invalidation hazard.

* **Lazy heavyweight imports.** ``torch`` / ``open_clip`` / ``PIL`` are
  imported inside :class:`_BaselineRunner`. Tests can stub ``open_clip``
  via :mod:`sys.modules` exactly the way ``view_labeler`` tests do.

* **Bad / missing files** are logged at WARNING and skipped -- the run
  proceeds with whatever survived. Same precedent as
  :class:`~car_lense_engine.dataset.view_labeler.ViewLabeler`.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- public models


class BaselineConfig(BaseModel):
    """Frozen configuration of a zero-shot baseline evaluation."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    """OpenCLIP model name (e.g. ``"MobileCLIP-S2"``)."""

    pretrained: str
    """OpenCLIP pretrained tag (e.g. ``"datacompdr"``)."""

    device: str = "cpu"
    """torch device string (``"cpu"`` / ``"cuda"`` / ``"mps"``)."""

    batch_size: int = 16
    """Images per forward pass."""

    top_ks: tuple[int, ...] = (1, 3, 5, 10)
    """The ``K`` values at which top-K accuracy is reported."""

    checkpoint_path: Path | None = None
    """Optional path to a Phase 5.2 fine-tuned checkpoint. If set, the
    image encoder's state dict is loaded from this file *after* the
    pre-trained weights, so the baseline harness evaluates the
    fine-tuned model. The classification head is not used here -- the
    baseline always falls back to prototype-retrieval scoring."""


class ClassMetric(BaseModel):
    """Per-class accuracy slice for the report."""

    model_config = ConfigDict(extra="forbid")

    class_id: str
    n_train: int
    n_test: int
    top_1: float
    top_5: float


class ConfusionPair(BaseModel):
    """One (true_class, predicted_class, count) entry from the top-K confusion list."""

    model_config = ConfigDict(extra="forbid")

    true_class: str
    predicted_class: str
    count: int


class BaselineReport(BaseModel):
    """Top-line numbers + per-class breakdown for a baseline run."""

    model_config = ConfigDict(extra="forbid")

    config: BaselineConfig
    n_classes: int
    n_train_images: int
    n_test_images: int
    overall: dict[str, float] = Field(default_factory=dict)
    """Top-K accuracy keyed by ``"top_{k}"`` (e.g. ``{"top_1": 0.621, ...}``)."""

    per_class: list[ClassMetric] = Field(default_factory=list)
    """Per-class metrics, sorted by top-1 accuracy ascending (worst first)."""

    confusion_top_pairs: list[ConfusionPair] = Field(default_factory=list)
    """Top confusion pairs (true != predicted), most frequent first."""

    elapsed_seconds: float = 0.0


# --------------------------------------------------------------- public API


def build_prototypes(
    *,
    conn: sqlite3.Connection,
    config: BaselineConfig,
    source: str,
    split: str,
) -> tuple[list[str], Any]:
    """Build one mean-embedding prototype per class.

    For each class with at least one image in ``(source, split)`` whose
    ``(year, make, model)`` is fully populated, embed every image,
    L2-normalize, take the mean, and re-normalize. Classes with zero
    successfully-loaded images are dropped from the output -- they have no
    prototype to compare against and are therefore not predictable by this
    baseline.

    Returns
    -------
    (class_ids, proto_tensor)
        ``class_ids`` is the sorted list of class id strings; ``proto_tensor``
        is a ``(n_classes, embed_dim)`` float tensor on the configured device.
    """
    runner = _BaselineRunner(config)
    return runner.build_prototypes(conn=conn, source=source, split=split)


def evaluate(
    *,
    conn: sqlite3.Connection,
    config: BaselineConfig,
    prototypes: tuple[list[str], Any],
    source: str,
    split: str,
    per_class_top: int = 20,
) -> BaselineReport:
    """Evaluate ``prototypes`` against the ``(source, split)`` test set.

    Returns a :class:`BaselineReport` with overall top-K accuracy, per-class
    metrics (best + worst ``per_class_top`` classes), and the most frequent
    confusion pairs. Test rows whose ``class_id`` doesn't appear in
    ``prototypes`` are still embedded and counted -- their argmax simply can
    never be the true class, so they hurt the score (as they should).
    """
    runner = _BaselineRunner(config)
    return runner.evaluate(
        conn=conn,
        prototypes=prototypes,
        source=source,
        split=split,
        per_class_top=per_class_top,
    )


# --------------------------------------------------------------- internals


def class_id_for(year: int | None, make: str | None, model: str | None) -> str | None:
    """Render the canonical class id for a row, or ``None`` if any field is missing.

    Format: ``"<year>|<make_lower>|<model_lower>"``. Callers MUST pass
    the canonical (Phase 4.5) make / model strings -- the function still
    lower-cases them for stable class ids, but the cross-source
    deduplication happens upstream in the canonicalization pass. Rows
    whose canonical fields are NULL are skipped at the caller layer.
    """
    if year is None or make is None or model is None:
        return None
    make_s = str(make).strip().lower()
    model_s = str(model).strip().lower()
    if not make_s or not model_s:
        return None
    return f"{int(year)}|{make_s}|{model_s}"


def _select_rows(
    conn: sqlite3.Connection,
    *,
    source: str,
    split: str,
) -> list[tuple[str, Path]]:
    """Return ``(class_id, local_path)`` pairs for one (source, split) slice.

    Rows with NULL ``(year, canonical_make, canonical_model)`` are
    skipped. The query joins listings to images so a single listing with
    multiple images contributes all its images.

    Reads the canonical_make / canonical_model columns added by
    migration 8 (Phase 4.5). The user MUST run the
    ``canonicalize-labels`` CLI before calling this -- rows whose
    canonical fields are NULL are excluded from prototype building.
    """
    sql = (
        "SELECT listings.year AS year, "
        "       listings.canonical_make AS make, "
        "       listings.canonical_model AS model, "
        "       images.local_path AS local_path "
        "FROM listings "
        "JOIN images ON images.listing_id = listings.listing_id "
        "WHERE listings.source = ? AND listings.split = ? "
        "ORDER BY images.image_id"
    )
    cur = conn.execute(sql, (source, split))
    rows: list[tuple[str, Path]] = []
    for row in cur.fetchall():
        cid = class_id_for(row["year"], row["make"], row["model"])
        if cid is None:
            continue
        local_path = row["local_path"]
        if not local_path:
            continue
        rows.append((cid, Path(str(local_path))))
    return rows


def _chunked(items: list[tuple[str, Path]], n: int) -> Iterator[list[tuple[str, Path]]]:
    """Slice ``items`` into chunks of at most ``n`` elements (preserving order)."""
    if n <= 0:
        raise ValueError(f"batch size must be > 0, got {n}")
    for i in range(0, len(items), n):
        yield items[i : i + n]


class _BaselineRunner:
    """Holds the lazily-loaded OpenCLIP model and runs the actual numerics.

    Public entry points are :meth:`build_prototypes` and :meth:`evaluate`;
    we deliberately don't expose this class outside the module because the
    config + tensor handoff between the two phases is tricky enough that
    "function-only" is the contract we want.
    """

    def __init__(self, config: BaselineConfig) -> None:
        self._config = config
        self._torch: Any | None = None
        self._model: Any | None = None
        self._preprocess: Any | None = None

    # ----- public ------------------------------------------------------- #

    def build_prototypes(
        self,
        *,
        conn: sqlite3.Connection,
        source: str,
        split: str,
    ) -> tuple[list[str], Any]:
        rows = _select_rows(conn, source=source, split=split)
        if not rows:
            logger.warning(
                "baseline: no train rows for source=%s split=%s -- prototypes will be empty",
                source,
                split,
            )
            # No rows -> no need to load the model. Returning an empty tensor
            # with zero dims is fine because evaluate() guards on len(class_ids).
            import torch  # noqa: PLC0415

            return [], torch.zeros((0, 0))

        logger.info(
            "baseline: building prototypes from %d train images (source=%s split=%s)",
            len(rows),
            source,
            split,
        )
        embeddings, used_class_ids = self._embed_rows(rows)
        torch_mod = self._require_torch()

        # Group embeddings by class. ``embeddings`` is shape (N, D), normalized.
        sums: dict[str, Any] = {}
        counts: Counter[str] = Counter()
        for emb, cid in zip(embeddings, used_class_ids, strict=True):
            counts[cid] += 1
            if cid in sums:
                sums[cid] = sums[cid] + emb
            else:
                sums[cid] = emb.clone()

        class_ids = sorted(sums.keys())
        # Mean = sum / count, then re-normalize.
        mean_rows: list[Any] = []
        for cid in class_ids:
            mean = sums[cid] / float(counts[cid])
            norm = mean.norm()
            if norm.item() == 0.0:
                # Pathological: all-zero mean -- skip the class so it can't
                # be a top-K candidate. We carry it through with an explicit
                # zero vector so the index space stays aligned.
                mean_rows.append(mean)
            else:
                mean_rows.append(mean / norm)
        proto_tensor = torch_mod.stack(mean_rows, dim=0) if mean_rows else torch_mod.zeros((0, 0))
        return class_ids, proto_tensor

    def evaluate(
        self,
        *,
        conn: sqlite3.Connection,
        prototypes: tuple[list[str], Any],
        source: str,
        split: str,
        per_class_top: int,
    ) -> BaselineReport:
        if per_class_top < 0:
            raise ValueError(f"per_class_top must be >= 0, got {per_class_top}")
        start = time.perf_counter()
        class_ids, proto_tensor = prototypes
        config = self._config
        top_ks = tuple(config.top_ks)
        rows = _select_rows(conn, source=source, split=split)

        # Stats for the test set, computed before we touch the model so the
        # n_train_images figure in the report reflects what build_prototypes
        # actually used (sum of class counts inferred from the proto tensor's
        # row count is wrong if the caller passed in a bespoke proto set).
        n_train_images_total = _count_train_images(conn, source=source, split="train")
        if not rows:
            elapsed = time.perf_counter() - start
            return BaselineReport(
                config=config,
                n_classes=len(class_ids),
                n_train_images=n_train_images_total,
                n_test_images=0,
                overall={f"top_{k}": 0.0 for k in top_ks},
                per_class=[],
                confusion_top_pairs=[],
                elapsed_seconds=elapsed,
            )
        if not class_ids:
            elapsed = time.perf_counter() - start
            logger.warning(
                "baseline: no prototypes available -- cannot make predictions (n_test_rows=%d)",
                len(rows),
            )
            return BaselineReport(
                config=config,
                n_classes=0,
                n_train_images=n_train_images_total,
                n_test_images=len(rows),
                overall={f"top_{k}": 0.0 for k in top_ks},
                per_class=[],
                confusion_top_pairs=[],
                elapsed_seconds=elapsed,
            )

        logger.info(
            "baseline: evaluating %d test images against %d prototypes (source=%s split=%s)",
            len(rows),
            len(class_ids),
            source,
            split,
        )

        embeddings, used_class_ids = self._embed_rows(rows)
        if embeddings.shape[0] == 0:
            elapsed = time.perf_counter() - start
            return BaselineReport(
                config=config,
                n_classes=len(class_ids),
                n_train_images=n_train_images_total,
                n_test_images=0,
                overall={f"top_{k}": 0.0 for k in top_ks},
                per_class=[],
                confusion_top_pairs=[],
                elapsed_seconds=elapsed,
            )

        sims = embeddings @ proto_tensor.T  # (n_test, n_classes)
        max_k = max(top_ks)
        k_eff = min(max_k, sims.shape[1])
        # ``topk`` is descending by default.
        _, topk_idx = sims.topk(k=k_eff, dim=-1)
        topk_idx_list: list[list[int]] = topk_idx.tolist()

        class_to_idx = {cid: i for i, cid in enumerate(class_ids)}
        correct_by_k: dict[int, int] = dict.fromkeys(top_ks, 0)
        per_class_train_counts = _per_class_train_counts(conn, source=source)
        per_class_n_test: Counter[str] = Counter()
        per_class_top1_correct: Counter[str] = Counter()
        per_class_top5_correct: Counter[str] = Counter()
        confusion: Counter[tuple[str, str]] = Counter()

        n_evaluated = 0
        for true_cid, topk in zip(used_class_ids, topk_idx_list, strict=True):
            per_class_n_test[true_cid] += 1
            n_evaluated += 1
            true_idx = class_to_idx.get(true_cid)
            pred_idx = topk[0] if topk else None
            pred_cid: str | None = class_ids[pred_idx] if pred_idx is not None else None
            for k in top_ks:
                if true_idx is None:
                    # Test row's class has no prototype -- can never hit.
                    continue
                if true_idx in topk[:k]:
                    correct_by_k[k] += 1
            if true_idx is not None and pred_idx == true_idx:
                per_class_top1_correct[true_cid] += 1
            if true_idx is not None and true_idx in topk[:5]:
                per_class_top5_correct[true_cid] += 1
            if pred_cid is not None and pred_cid != true_cid:
                confusion[(true_cid, pred_cid)] += 1

        overall = {f"top_{k}": (correct_by_k[k] / n_evaluated) for k in top_ks}

        per_class_metrics: list[ClassMetric] = []
        for cid in sorted(per_class_n_test):
            n_test = per_class_n_test[cid]
            n_train = per_class_train_counts.get(cid, 0)
            top1 = per_class_top1_correct[cid] / n_test if n_test else 0.0
            top5 = per_class_top5_correct[cid] / n_test if n_test else 0.0
            per_class_metrics.append(
                ClassMetric(
                    class_id=cid,
                    n_train=n_train,
                    n_test=n_test,
                    top_1=top1,
                    top_5=top5,
                )
            )

        # Sort worst-first (lowest top-1), then alphabetically for stability.
        per_class_metrics.sort(key=lambda m: (m.top_1, m.class_id))
        # Take worst per_class_top + best per_class_top. If the total is
        # less than 2x per_class_top, just return the full list.
        if per_class_top > 0 and len(per_class_metrics) > 2 * per_class_top:
            worst = per_class_metrics[:per_class_top]
            best = per_class_metrics[-per_class_top:]
            per_class_report = worst + best
        else:
            per_class_report = per_class_metrics

        confusion_pairs = [
            ConfusionPair(true_class=t, predicted_class=p, count=n)
            for (t, p), n in confusion.most_common(per_class_top)
        ]

        elapsed = time.perf_counter() - start
        return BaselineReport(
            config=config,
            n_classes=len(class_ids),
            n_train_images=n_train_images_total,
            n_test_images=n_evaluated,
            overall=overall,
            per_class=per_class_report,
            confusion_top_pairs=confusion_pairs,
            elapsed_seconds=elapsed,
        )

    # ----- model + embedding ------------------------------------------- #

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip  # noqa: PLC0415
            import torch  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover -- deps are in pyproject
            raise RuntimeError(
                "open_clip_torch and torch are required for the baseline harness"
            ) from exc

        cfg = self._config
        logger.info(
            "baseline: loading OpenCLIP %s / %s on %s",
            cfg.model_name,
            cfg.pretrained,
            cfg.device,
        )
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                cfg.model_name,
                pretrained=cfg.pretrained,
                device=cfg.device,
            )
        except Exception as exc:  # noqa: BLE001 -- re-raise with a clearer hint
            raise RuntimeError(
                f"failed to load OpenCLIP model "
                f"(name={cfg.model_name!r}, pretrained={cfg.pretrained!r}); "
                "if MobileCLIP-S2 is missing, ensure open_clip_torch>=2.24 is installed "
                "and the pretrained tag is one of "
                f"`open_clip.list_pretrained('{cfg.model_name}')`. "
                f"Underlying error: {exc!r}"
            ) from exc
        # Optionally overlay a fine-tuned image-encoder state dict on top
        # of the pretrained weights. The checkpoint format is the one
        # produced by car_lense_engine.training (Phase 5.2): a dict with
        # an "image_encoder_state_dict" entry that targets `model.visual`.
        if cfg.checkpoint_path is not None:
            _load_image_encoder_checkpoint(
                model=model,
                checkpoint_path=cfg.checkpoint_path,
                device=cfg.device,
                torch_mod=torch,
            )
        model.eval()
        self._torch = torch
        self._model = model
        self._preprocess = preprocess

    def _embed_rows(
        self,
        rows: list[tuple[str, Path]],
    ) -> tuple[Any, list[str]]:
        """Embed every path in ``rows`` (L2-normalized).

        Returns ``(tensor, class_ids)`` where ``tensor`` is the stacked
        normalized embeddings for the rows that successfully loaded, and
        ``class_ids`` is the parallel list of class ids (one per surviving
        row). Bad / missing files are logged at WARNING and dropped.
        """
        self._ensure_model()
        torch_mod = self._require_torch()
        config = self._config
        all_chunks: list[Any] = []
        used_class_ids: list[str] = []
        n_skipped = 0
        n_total = len(rows)
        n_done = 0
        for chunk in _chunked(rows, config.batch_size):
            ok_class_ids: list[str] = []
            tensors: list[Any] = []
            for cid, path in chunk:
                try:
                    tensors.append(self._load_and_preprocess(path))
                except Exception as exc:  # noqa: BLE001 -- log + skip
                    logger.warning("baseline: skipping %s (%s)", path, exc)
                    n_skipped += 1
                    continue
                ok_class_ids.append(cid)
            if not tensors:
                continue
            batch = torch_mod.stack(tensors).to(config.device)
            with torch_mod.no_grad():
                features = self._encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
            all_chunks.append(features)
            used_class_ids.extend(ok_class_ids)
            n_done += len(ok_class_ids)
            if n_done and (n_done % max(50, config.batch_size * 4) < config.batch_size):
                logger.info("baseline: embedded %d / %d", n_done, n_total)
        if not all_chunks:
            return torch_mod.zeros((0, 0)), []
        return torch_mod.cat(all_chunks, dim=0), used_class_ids

    def _encode_image(self, batch: Any) -> Any:
        model = self._model
        if model is None:  # pragma: no cover -- defensive
            raise RuntimeError("model not loaded")
        return model.encode_image(batch)

    def _load_and_preprocess(self, path: Path) -> Any:
        from PIL import Image as PILImageModule  # noqa: PLC0415

        preprocess = self._preprocess
        if preprocess is None:  # pragma: no cover -- defensive
            raise RuntimeError("preprocess transform not loaded")
        with PILImageModule.open(path) as img:
            converted = img.convert("RGB")
        return cast(Any, preprocess(converted))

    def _require_torch(self) -> Any:
        if self._torch is None:  # pragma: no cover -- defensive
            raise RuntimeError("torch not loaded; call _ensure_model() first")
        return self._torch


def _load_image_encoder_checkpoint(
    *,
    model: Any,
    checkpoint_path: Path,
    device: str,
    torch_mod: Any,
) -> None:
    """Overlay a Phase 5.2 fine-tuned ``image_encoder_state_dict`` on ``model``.

    The checkpoint is the dict format produced by
    :func:`car_lense_engine.training.run_training`: it has
    ``"image_encoder_state_dict"`` keyed on ``model.visual``'s parameter
    names. We try ``model.visual.load_state_dict`` first; if the model
    doesn't expose a ``.visual`` attribute (e.g. test stubs), we fall
    back to ``model.load_state_dict`` on the whole model.

    Bad-checkpoint cases (missing file, wrong format) raise
    :class:`RuntimeError` with a clear message -- this is a user-visible
    error path because a fine-tuned eval that silently runs the
    pretrained backbone would be very confusing.
    """
    if not checkpoint_path.exists():
        raise RuntimeError(f"checkpoint path does not exist: {checkpoint_path}")
    payload = torch_mod.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "image_encoder_state_dict" not in payload:
        raise RuntimeError(
            f"checkpoint {checkpoint_path} is not a Phase 5.2 training checkpoint "
            "(missing 'image_encoder_state_dict' key)"
        )
    state = payload["image_encoder_state_dict"]
    visual = getattr(model, "visual", None)
    if visual is not None and hasattr(visual, "load_state_dict"):
        visual.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    logger.info(
        "baseline: loaded fine-tuned weights from %s (epoch=%s val_top1=%s)",
        checkpoint_path,
        payload.get("epoch"),
        payload.get("val_top1"),
    )


def _count_train_images(conn: sqlite3.Connection, *, source: str, split: str) -> int:
    """Count train images for ``(source, split)``. Used in the report header."""
    sql = (
        "SELECT COUNT(*) AS n "
        "FROM listings JOIN images ON images.listing_id = listings.listing_id "
        "WHERE listings.source = ? AND listings.split = ?"
    )
    cur = conn.execute(sql, (source, split))
    row = cur.fetchone()
    return int(row["n"]) if row is not None else 0


def _per_class_train_counts(
    conn: sqlite3.Connection,
    *,
    source: str,
) -> dict[str, int]:
    """Map ``class_id -> n_train_images`` for the given source's train split.

    Reads canonical_make / canonical_model (Phase 4.5); rows whose
    canonical fields are NULL are excluded from the per-class counts.
    """
    sql = (
        "SELECT listings.year AS year, "
        "       listings.canonical_make AS make, "
        "       listings.canonical_model AS model, "
        "       COUNT(images.image_id) AS n "
        "FROM listings JOIN images ON images.listing_id = listings.listing_id "
        "WHERE listings.source = ? AND listings.split = 'train' "
        "GROUP BY listings.year, listings.canonical_make, listings.canonical_model"
    )
    out: dict[str, int] = defaultdict(int)
    cur = conn.execute(sql, (source,))
    for row in cur.fetchall():
        cid = class_id_for(row["year"], row["make"], row["model"])
        if cid is None:
            continue
        out[cid] += int(row["n"])
    return dict(out)


def write_report(report: BaselineReport, path: Path) -> None:
    """Serialize a :class:`BaselineReport` to JSON at ``path`` (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def summary_line(report: BaselineReport) -> str:
    """Single-line stdout summary, e.g. ``"top_1=0.621 top_5=0.879 elapsed=243.7s"``."""
    pieces: list[str] = []
    for k in sorted(report.config.top_ks):
        key = f"top_{k}"
        value = report.overall.get(key, 0.0)
        pieces.append(f"{key}={value:.3f}")
    pieces.append(f"elapsed={report.elapsed_seconds:.1f}s")
    return " ".join(pieces)


__all__ = [
    "BaselineConfig",
    "BaselineReport",
    "ClassMetric",
    "ConfusionPair",
    "build_prototypes",
    "class_id_for",
    "evaluate",
    "summary_line",
    "write_report",
]
