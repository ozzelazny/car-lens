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


# The five Phase 3.3 exterior views that participate in view-conditional
# retrieval. The ``non-exterior`` class is handled separately (request
# rejection at inference time), and the view-classifier head ordering
# adds it at index 5 -- see
# :data:`car_lense_engine.training.view_classifier.VIEW_CLASS_NAMES`.
EXTERIOR_VIEWS: tuple[str, ...] = (
    "front",
    "rear",
    "side",
    "three-quarter-front",
    "three-quarter-rear",
)


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
    source: str | list[str],
    split: str,
) -> tuple[list[str], Any]:
    """Build one mean-embedding prototype per class.

    For each class with at least one image in ``(source, split)`` whose
    ``(year, make, model)`` is fully populated, embed every image,
    L2-normalize, take the mean, and re-normalize. Classes with zero
    successfully-loaded images are dropped from the output -- they have no
    prototype to compare against and are therefore not predictable by this
    baseline.

    ``source`` may be either a single source name (legacy single-dataset
    API) or a list of source names (Phase 5.5+ multi-source training);
    see :func:`_select_rows` for the SQL contract.

    Returns
    -------
    (class_ids, proto_tensor)
        ``class_ids`` is the sorted list of class id strings; ``proto_tensor``
        is a ``(n_classes, embed_dim)`` float tensor on the configured device.
    """
    runner = _BaselineRunner(config)
    return runner.build_prototypes(conn=conn, source=source, split=split)


def build_prototypes_by_view(
    *,
    conn: sqlite3.Connection,
    config: BaselineConfig,
    source: str | list[str],
    split: str,
) -> tuple[list[str], dict[str, Any]]:
    """Compute mean-pooled L2-normalized prototypes per ``(class, view)``.

    This is the Phase 6.1 view-conditional analogue of
    :func:`build_prototypes`. Instead of one prototype per class
    (collapsing all views), we produce one prototype per
    ``(class, view)`` cell for each of the five exterior views
    (:data:`EXTERIOR_VIEWS`). Non-exterior images are dropped at
    selection time -- they don't participate in retrieval at all.

    Returns
    -------
    (class_ids, prototypes_by_view)
        ``class_ids`` is the sorted list of class id strings (the global
        class index space). ``prototypes_by_view`` maps each exterior
        view name to a ``(n_classes, embed_dim)`` float tensor on the
        configured device, with row ``i`` holding the L2-normalized mean
        embedding for the class at ``class_ids[i]`` in that view.
        Classes missing a view's prototype get an all-zero row in that
        view's tensor, which is harmless for retrieval (cosine sim with
        any unit query vector is 0 -> the class will never be in the
        top-K from that view, exactly as desired).

    Notes
    -----

    * Both stages share a single backbone pass: every eligible image is
      embedded once and bucketed by ``(class, view)``. Re-using the
      :class:`_BaselineRunner` machinery keeps the model load + preprocess
      pipeline consistent with :func:`build_prototypes`.
    * Rows whose ``images.view`` is NULL or not in
      :data:`EXTERIOR_VIEWS` are silently skipped. This includes the
      ``interior`` / ``detail`` / ``non-car`` labels collapsed into the
      view classifier's "non-exterior" output -- those have no
      retrieval prototype because the service rejects them upstream.
    """
    runner = _BaselineRunner(config)
    return runner.build_prototypes_by_view(conn=conn, source=source, split=split)


def evaluate(
    *,
    conn: sqlite3.Connection,
    config: BaselineConfig,
    prototypes: tuple[list[str], Any],
    source: str | list[str],
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

    .. note::
       As of Phase 4.6 the SQL callers
       (:func:`_select_rows`, :func:`_per_class_train_counts`,
       :func:`car_lense_engine.training.train_classifier._select_train_rows`)
       pass ``listings.generation_year`` -- the 4-year bucket START year
       -- in the ``year`` argument, NOT the raw calendar year. This
       function is unchanged: it just formats whatever integer it
       receives. Concretely, any calendar year in ``[2012, 2015]`` now
       produces the class id ``"2012|honda|civic"`` because they all
       map to the same generation bucket. The raw ``listings.year``
       column is preserved for audit / display but is no longer the
       grouping key for retrieval / training.
    """
    if year is None or make is None or model is None:
        return None
    make_s = str(make).strip().lower()
    model_s = str(model).strip().lower()
    if not make_s or not model_s:
        return None
    return f"{int(year)}|{make_s}|{model_s}"


def _coerce_source_field(value: object) -> list[str]:
    """Coerce a pydantic ``source`` field input into a ``list[str]``.

    Used as a ``field_validator(..., mode="before")`` on the
    ``source`` field of :class:`TrainConfig`, :class:`BaselineConfig`,
    and :class:`EvaluationConfig`. Accepts:

    * ``str`` -- a single source name, e.g. ``"compcars"``. Returned
      as ``["compcars"]``. A comma-separated string like
      ``"compcars,vmmrdb"`` is split into ``["compcars", "vmmrdb"]``
      with whitespace trimmed around each entry.
    * ``list``/``tuple`` of strings -- returned as a fresh list,
      element strings stripped of whitespace.

    Empty / whitespace-only entries are dropped so a stray trailing
    comma in the CLI form doesn't produce an empty source. An empty
    final list raises :class:`ValueError`.
    """
    if isinstance(value, str):
        items = [s.strip() for s in value.split(",")]
    elif isinstance(value, list | tuple):
        items = [str(s).strip() for s in value]
    else:
        raise ValueError(f"source must be a string or list of strings, got {type(value).__name__}")
    filtered = [s for s in items if s]
    if not filtered:
        raise ValueError("source must contain at least one non-empty entry")
    return filtered


def _normalize_sources(source: str | list[str]) -> list[str]:
    """Normalize a single source or list-of-sources into a non-empty ``list[str]``.

    Accepts either a single string (backward-compat with the original
    Phase 5.1 / 5.2 API) or a list of strings (Phase 5.5+ multi-source).
    A bare ``str`` is wrapped in a single-element list; a list is copied
    and validated element-wise. Each source must be a non-empty string;
    empty / whitespace-only entries raise :class:`ValueError`. An empty
    list also raises :class:`ValueError` because there's nothing to
    filter on.
    """
    items = [source] if isinstance(source, str) else list(source)
    if not items:
        raise ValueError("source must be a non-empty string or list of strings")
    normalized: list[str] = []
    for entry in items:
        if not isinstance(entry, str):
            raise ValueError(f"source entries must be strings, got {type(entry).__name__}")
        stripped = entry.strip()
        if not stripped:
            raise ValueError("source entries must be non-empty strings")
        normalized.append(stripped)
    return normalized


def _source_where_clause(sources: list[str]) -> tuple[str, tuple[str, ...]]:
    """Build the ``listings.source`` ``WHERE`` fragment + parameter tuple.

    For a single-element list we keep the legacy ``listings.source = ?``
    form (cheaper plan, exact match on the existing index); for
    multi-source we emit ``listings.source IN (?, ?, ...)`` with one
    placeholder per source. SQLite handles both forms over the same
    index on ``(source, ...)``.
    """
    if len(sources) == 1:
        return "listings.source = ?", (sources[0],)
    placeholders = ", ".join("?" for _ in sources)
    return f"listings.source IN ({placeholders})", tuple(sources)


def _select_rows(
    conn: sqlite3.Connection,
    *,
    source: str | list[str],
    split: str,
) -> list[tuple[str, str | None, Path]]:
    """Return ``(class_id, view, local_path)`` triples for one (source, split) slice.

    ``source`` may be either a single source name (legacy API) or a list
    of source names. For a single source the SQL emits the original
    ``listings.source = ?`` filter; for multiple sources it emits an
    ``IN (?, ?, ...)`` clause so a Phase 5.5+ training run can pull
    rows from CompCars + VMMRdb + Stanford Cars in a single pass.

    Rows with NULL ``(generation_year, canonical_make, canonical_model)``
    are skipped. The query joins listings to images so a single listing
    with multiple images contributes all its images.

    Reads the canonical_make / canonical_model columns (migration 8,
    Phase 4.5) AND the generation_year column (migration 9, Phase 4.6).
    The user MUST run the ``canonicalize-labels`` CLI before calling
    this -- rows whose canonical / generation fields are NULL are
    excluded from prototype building.

    The bucket start year is passed as the ``year`` argument to
    :func:`class_id_for`, so any calendar year in the same 4-year
    bucket produces the same class id (e.g. 2012, 2013, 2014, 2015 all
    -> ``"2012|<make>|<model>"``).

    As of Phase 3.5 the filter is on ``images.split`` (migration 010),
    not ``listings.split``: splits are now stratified per-image by
    ``(class, view)`` because a single listing can contribute a front
    shot and a rear shot that legitimately fall into different splits.
    Rows whose ``images.split`` is NULL (e.g. non-exterior images
    intentionally left out of training) are excluded. The ``view``
    column is selected and returned alongside the path so future
    view-conditional callers can branch on it -- the prototype /
    training paths in this module do *not* yet consume it.
    """
    sources = _normalize_sources(source)
    source_clause, source_params = _source_where_clause(sources)
    sql = (
        "SELECT listings.generation_year AS year, "
        "       listings.canonical_make AS make, "
        "       listings.canonical_model AS model, "
        "       images.view AS view, "
        "       images.local_path AS local_path "
        "FROM listings "
        "JOIN images ON images.listing_id = listings.listing_id "
        f"WHERE {source_clause} AND images.split = ? "
        "ORDER BY images.image_id"
    )
    cur = conn.execute(sql, (*source_params, split))
    rows: list[tuple[str, str | None, Path]] = []
    for row in cur.fetchall():
        cid = class_id_for(row["year"], row["make"], row["model"])
        if cid is None:
            continue
        local_path = row["local_path"]
        if not local_path:
            continue
        view_raw = row["view"]
        view: str | None = str(view_raw) if view_raw is not None else None
        rows.append((cid, view, Path(str(local_path))))
    return rows


def _chunked(
    items: list[tuple[str, str | None, Path]], n: int
) -> Iterator[list[tuple[str, str | None, Path]]]:
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
        source: str | list[str],
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

    def build_prototypes_by_view(
        self,
        *,
        conn: sqlite3.Connection,
        source: str | list[str],
        split: str,
    ) -> tuple[list[str], dict[str, Any]]:
        """Per-(class, view) variant of :meth:`build_prototypes`.

        See :func:`build_prototypes_by_view` for the contract; this is the
        instance-method implementation that handles the actual embedding
        and bucketing.
        """
        all_rows = _select_rows(conn, source=source, split=split)
        # Drop rows whose view is NULL or non-exterior. We do this AFTER
        # the SQL fetch (rather than encoding the view set in the WHERE
        # clause) so the row-shape stays consistent with the single-
        # prototype path and existing tests don't have to be reshaped
        # for the view filter.
        exterior = set(EXTERIOR_VIEWS)
        rows: list[tuple[str, str | None, Path]] = [
            (cid, view, path) for (cid, view, path) in all_rows if view in exterior
        ]
        n_dropped = len(all_rows) - len(rows)
        if n_dropped:
            logger.info(
                "baseline (per-view): dropped %d/%d rows with NULL or non-exterior view",
                n_dropped,
                len(all_rows),
            )
        if not rows:
            logger.warning(
                "baseline (per-view): no exterior train rows for source=%s split=%s -- "
                "prototypes will be empty",
                source,
                split,
            )
            import torch  # noqa: PLC0415

            empty: dict[str, Any] = {v: torch.zeros((0, 0)) for v in EXTERIOR_VIEWS}
            return [], empty

        logger.info(
            "baseline (per-view): building prototypes from %d exterior train images "
            "(source=%s split=%s)",
            len(rows),
            source,
            split,
        )
        embeddings, used_class_ids, used_views = self._embed_rows_with_views(rows)
        torch_mod = self._require_torch()

        # First pass: collect all class ids so the per-view tensors share
        # the same row index space. Sorted for deterministic ordering.
        class_ids = sorted(set(used_class_ids))
        class_to_idx = {cid: i for i, cid in enumerate(class_ids)}

        # Group sums by (view, class_idx).
        per_view_sums: dict[str, dict[int, Any]] = {v: {} for v in EXTERIOR_VIEWS}
        per_view_counts: dict[str, Counter[int]] = {v: Counter() for v in EXTERIOR_VIEWS}
        for emb, cid, view in zip(embeddings, used_class_ids, used_views, strict=True):
            if view not in per_view_sums:
                # Should never happen given the up-front filter, but
                # cheap to be defensive.
                continue
            idx = class_to_idx[cid]
            per_view_counts[view][idx] += 1
            if idx in per_view_sums[view]:
                per_view_sums[view][idx] = per_view_sums[view][idx] + emb
            else:
                per_view_sums[view][idx] = emb.clone()

        # Embed dim is the trailing axis of any successfully-embedded row.
        embed_dim = int(embeddings.shape[1])
        # Match the device of the accumulated embeddings so the zero-fill
        # rows for missing (class, view) cells don't mix CPU + CUDA when
        # we later torch.stack them. Consumers (prototype cache writer,
        # recognize_api loader) expect CPU tensors, so we move the final
        # per-view tensors to CPU before returning.
        device = embeddings.device

        prototypes_by_view: dict[str, Any] = {}
        for view in EXTERIOR_VIEWS:
            rows_out: list[Any] = []
            sums = per_view_sums[view]
            counts = per_view_counts[view]
            for idx in range(len(class_ids)):
                if idx not in sums:
                    rows_out.append(torch_mod.zeros((embed_dim,), device=device))
                    continue
                mean = sums[idx] / float(counts[idx])
                norm = mean.norm()
                if norm.item() == 0.0:
                    rows_out.append(torch_mod.zeros((embed_dim,), device=device))
                else:
                    rows_out.append(mean / norm)
            if rows_out:
                prototypes_by_view[view] = torch_mod.stack(rows_out, dim=0).cpu()
            else:
                prototypes_by_view[view] = torch_mod.zeros((0, embed_dim), device=device).cpu()
        return class_ids, prototypes_by_view

    def evaluate(
        self,
        *,
        conn: sqlite3.Connection,
        prototypes: tuple[list[str], Any],
        source: str | list[str],
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
        rows: list[tuple[str, str | None, Path]],
    ) -> tuple[Any, list[str]]:
        """Embed every path in ``rows`` (L2-normalized).

        Returns ``(tensor, class_ids)`` where ``tensor`` is the stacked
        normalized embeddings for the rows that successfully loaded, and
        ``class_ids`` is the parallel list of class ids (one per surviving
        row). The per-row ``view`` value is part of the input tuple but
        is not yet consumed -- it's plumbed through for the upcoming
        view-conditional prototype work (Phase 3.5 Step 6). Bad /
        missing files are logged at WARNING and dropped.
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
            for cid, _view, path in chunk:
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

    def _embed_rows_with_views(
        self,
        rows: list[tuple[str, str | None, Path]],
    ) -> tuple[Any, list[str], list[str]]:
        """Per-view variant of :meth:`_embed_rows`.

        Identical to :meth:`_embed_rows` but also returns the parallel
        list of view strings for the surviving (successfully-loaded)
        rows. Used by :meth:`build_prototypes_by_view` to bucket
        embeddings by ``(class, view)``. Rows are assumed to have
        non-NULL view (the caller filters before calling).
        """
        self._ensure_model()
        torch_mod = self._require_torch()
        config = self._config
        all_chunks: list[Any] = []
        used_class_ids: list[str] = []
        used_views: list[str] = []
        n_skipped = 0
        n_total = len(rows)
        n_done = 0
        for chunk in _chunked(rows, config.batch_size):
            ok_class_ids: list[str] = []
            ok_views: list[str] = []
            tensors: list[Any] = []
            for cid, view, path in chunk:
                if view is None:  # pragma: no cover -- caller filters NULLs
                    continue
                try:
                    tensors.append(self._load_and_preprocess(path))
                except Exception as exc:  # noqa: BLE001 -- log + skip
                    logger.warning("baseline: skipping %s (%s)", path, exc)
                    n_skipped += 1
                    continue
                ok_class_ids.append(cid)
                ok_views.append(view)
            if not tensors:
                continue
            batch = torch_mod.stack(tensors).to(config.device)
            with torch_mod.no_grad():
                features = self._encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
            all_chunks.append(features)
            used_class_ids.extend(ok_class_ids)
            used_views.extend(ok_views)
            n_done += len(ok_class_ids)
            if n_done and (n_done % max(50, config.batch_size * 4) < config.batch_size):
                logger.info("baseline (per-view): embedded %d / %d", n_done, n_total)
        if not all_chunks:
            return torch_mod.zeros((0, 0)), [], []
        return torch_mod.cat(all_chunks, dim=0), used_class_ids, used_views

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


def _count_train_images(
    conn: sqlite3.Connection,
    *,
    source: str | list[str],
    split: str,
) -> int:
    """Count train images for ``(source, split)``. Used in the report header.

    ``source`` may be a single source name or a list -- see
    :func:`_select_rows` for the multi-source contract. Filters on
    ``images.split`` (migration 010) rather than ``listings.split`` --
    splits are now per-image (Phase 3.5).
    """
    sources = _normalize_sources(source)
    source_clause, source_params = _source_where_clause(sources)
    sql = (
        "SELECT COUNT(*) AS n "
        "FROM listings JOIN images ON images.listing_id = listings.listing_id "
        f"WHERE {source_clause} AND images.split = ?"
    )
    cur = conn.execute(sql, (*source_params, split))
    row = cur.fetchone()
    return int(row["n"]) if row is not None else 0


def _per_class_train_counts(
    conn: sqlite3.Connection,
    *,
    source: str | list[str],
) -> dict[str, int]:
    """Map ``class_id -> n_train_images`` for the given source(s) train split.

    ``source`` may be a single source name or a list of source names; when
    a list is passed, the counts are summed across all selected sources
    (the same class id can be contributed by more than one source after
    Phase 4.5 canonicalization).

    Reads canonical_make / canonical_model (Phase 4.5) AND
    generation_year (Phase 4.6); rows whose canonical / generation
    fields are NULL are excluded from the per-class counts. Filters
    on ``images.split = 'train'`` (migration 010), not
    ``listings.split`` (Phase 3.5).
    """
    sources = _normalize_sources(source)
    source_clause, source_params = _source_where_clause(sources)
    sql = (
        "SELECT listings.generation_year AS year, "
        "       listings.canonical_make AS make, "
        "       listings.canonical_model AS model, "
        "       COUNT(images.image_id) AS n "
        "FROM listings JOIN images ON images.listing_id = listings.listing_id "
        f"WHERE {source_clause} AND images.split = 'train' "
        "GROUP BY listings.generation_year, listings.canonical_make, listings.canonical_model"
    )
    out: dict[str, int] = defaultdict(int)
    cur = conn.execute(sql, source_params)
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
    "EXTERIOR_VIEWS",
    "BaselineConfig",
    "BaselineReport",
    "ClassMetric",
    "ConfusionPair",
    "build_prototypes",
    "build_prototypes_by_view",
    "class_id_for",
    "evaluate",
    "summary_line",
    "write_report",
]
