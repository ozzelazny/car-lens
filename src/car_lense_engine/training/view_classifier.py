"""View-classifier head training (Phase 5.3).

This module trains a small classification head on top of the
MobileCLIP-S2 image encoder to predict one of 6 view classes:

    front, rear, side, three-quarter-front, three-quarter-rear, non-exterior

The ``non-exterior`` class collapses the three raw "not a useful exterior
shot" labels emitted by the Phase 3.3 zero-shot view labeler --
``{interior, detail, non-car}``. Rows whose ``images.view`` is NULL are
excluded entirely (those are rows that haven't been labeled yet).

Design notes
------------

* **Frozen backbone.** The image encoder is held fixed (``requires_grad
  = False`` on every backbone parameter) and we train only a small head
  on top of cached features. This is the right recipe for a low-data,
  low-class-count task on top of an already-strong representation: the
  whole train pass is dominated by the cheap head update.

* **Feature caching.** The expensive operation is the backbone forward
  pass. With ~150k train images at ``embed_dim=512`` × ``float32`` we
  cache ~300 MB of features in RAM and run head training over the
  pre-computed tensor for many epochs. Caching is done once per
  ``train_view_classifier`` call (not once per epoch) over both the
  train and val splits.

* **Class imbalance.** The Phase 3.3 distribution skews towards
  ``three-quarter-*`` and ``side`` views (~75% combined) with ``rear``
  at ~7% and ``non-exterior`` at <0.05%. We compute ``CrossEntropyLoss``
  weights as ``1 / sqrt(n_per_class)`` then renormalize them so the
  mean weight is 1.0; this is the standard "inverse square-root
  frequency" reweighting from the long-tail literature and is gentler
  than pure inverse-frequency on the head classes.

* **Confidence filter.** We drop training rows whose ``view_score`` is
  below ``min_view_score`` (default 0.6). The zero-shot labeler is
  reliable above ~0.6 but the long tail of low-confidence labels
  injects noise that hurts the small head we're training.

* **Linear head by default.** ``nn.Linear(embed_dim, 6)``. An optional
  2-layer MLP head is supported (``--head-arch mlp``) for the case
  where the linear baseline saturates below the target val top-1.

The training CLI :mod:`car_lense_engine.training.view_classifier_cli`
drives this module; it never runs in unit tests (the smoke test stubs
the backbone). The full training run produces:

* ``models/checkpoints/view_classifier_v1.pt`` -- a dict with
  ``head_state_dict``, ``image_encoder_state_dict``, ``class_names``,
  ``config``, and the final ``val_confusion_matrix``. The encoder state
  is bundled so the checkpoint is self-contained (the recognize API
  can load both pieces from one file).
* ``reports/phase5_3_view_classifier.json`` -- per-epoch loss + val
  top-1 history plus the best-epoch number.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- constants


EXTERIOR_VIEWS: tuple[str, ...] = (
    "front",
    "rear",
    "side",
    "three-quarter-front",
    "three-quarter-rear",
)
"""The five Phase 3.3 view labels that map to indices 0..4 (preserved)."""

VIEW_CLASS_NAMES: tuple[str, ...] = (*EXTERIOR_VIEWS, "non-exterior")
"""The canonical 6-class output vocabulary in index order."""

BINARY_CLASS_NAMES: tuple[str, ...] = ("exterior", "non-exterior")
"""The 2-class output vocabulary for the binary rejection head."""

NON_EXTERIOR_RAW: frozenset[str] = frozenset({"interior", "detail", "non-car"})
"""Raw view labels that collapse into the single ``non-exterior`` class."""

_VIEW_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(EXTERIOR_VIEWS)}
for _raw in NON_EXTERIOR_RAW:
    _VIEW_TO_INDEX[_raw] = len(EXTERIOR_VIEWS)

_BINARY_VIEW_TO_INDEX: dict[str, int] = dict.fromkeys(EXTERIOR_VIEWS, 0)
for _raw in NON_EXTERIOR_RAW:
    _BINARY_VIEW_TO_INDEX[_raw] = 1


def collapse_view(view: str) -> int:
    """Map a raw view label to its 0..5 class index.

    Accepts any of the five exterior labels in :data:`EXTERIOR_VIEWS`
    (indices 0..4) or any of the three raw "non-exterior" labels in
    :data:`NON_EXTERIOR_RAW` (collapsed to index 5).

    Raises ``KeyError`` for any other input -- callers should pre-filter
    rows whose ``view`` is NULL or otherwise outside the vocabulary.
    """
    try:
        return _VIEW_TO_INDEX[view]
    except KeyError as exc:
        raise KeyError(
            f"view {view!r} is not a known label (expected one of {sorted(_VIEW_TO_INDEX)})"
        ) from exc


def collapse_view_to_binary(view: str) -> int:
    """Map a raw view label to its binary 0/1 class index.

    Returns 0 for any of the five exterior labels in :data:`EXTERIOR_VIEWS`,
    and 1 for any of the three "non-exterior" labels in
    :data:`NON_EXTERIOR_RAW`.

    Raises ``KeyError`` for any other input -- callers should pre-filter
    rows whose ``view`` is NULL or otherwise outside the vocabulary.
    """
    try:
        return _BINARY_VIEW_TO_INDEX[view]
    except KeyError as exc:
        raise KeyError(
            f"view {view!r} is not a known label (expected one of {sorted(_BINARY_VIEW_TO_INDEX)})"
        ) from exc


# --------------------------------------------------------------- heads


def _build_linear_head(embed_dim: int, out_features: int = len(VIEW_CLASS_NAMES)) -> Any:
    """Construct a single-layer linear classification head."""
    import torch.nn as nn  # noqa: PLC0415

    return nn.Linear(embed_dim, out_features)


def _build_mlp_head(embed_dim: int, out_features: int = len(VIEW_CLASS_NAMES)) -> Any:
    """Construct a 2-layer MLP head (``embed_dim -> 256 -> out_features``, ReLU, dropout)."""
    import torch.nn as nn  # noqa: PLC0415

    return nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(256, out_features),
    )


class LinearHead:
    """Factory for the single-layer linear head.

    Wraps :func:`_build_linear_head` in a small class so callers can
    introspect the architecture choice symbolically. The actual
    ``nn.Module`` is created by :meth:`build`.
    """

    arch_name = "linear"

    @staticmethod
    def build(embed_dim: int, out_features: int = len(VIEW_CLASS_NAMES)) -> Any:
        return _build_linear_head(embed_dim, out_features=out_features)


class MLPHead:
    """Factory for the 2-layer MLP head."""

    arch_name = "mlp"

    @staticmethod
    def build(embed_dim: int, out_features: int = len(VIEW_CLASS_NAMES)) -> Any:
        return _build_mlp_head(embed_dim, out_features=out_features)


def _resolve_head_factory(head_arch: str) -> type[LinearHead] | type[MLPHead]:
    if head_arch == "linear":
        return LinearHead
    if head_arch == "mlp":
        return MLPHead
    raise ValueError(f"unknown head_arch {head_arch!r} (expected 'linear' or 'mlp')")


# --------------------------------------------------------------- public models


class ViewClassifierConfig(BaseModel):
    """Frozen hyperparameters for a Phase 5.3 view-classifier run."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "MobileCLIP-S2"
    pretrained: str = "datacompdr"
    device: str = "cpu"
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 1024
    backbone_batch_size: int = 128
    min_view_score: float = 0.6
    head_arch: str = "linear"
    backbone_checkpoint: Path | None = None
    """Optional path to a fine-tuned Phase 5.2 checkpoint whose
    ``image_encoder_state_dict`` is overlaid on top of the pretrained
    backbone. ``None`` keeps the raw pretrained weights."""
    weight_decay: float = 0.0
    seed: int = 42
    binary: bool = False
    """If True, train a 2-class ``exterior`` vs ``non-exterior`` head
    instead of the 6-way view head. The dataset builder pulls
    non-exterior rows regardless of ``images.split`` (those rows have
    NULL split today) and deterministically buckets them into train /
    val / test via SHA-1 of the image_id."""


class ViewEpochMetrics(BaseModel):
    """Per-epoch training stats."""

    model_config = ConfigDict(extra="forbid")

    epoch: int
    train_loss: float
    val_top1: float
    elapsed_s: float


class ViewClassifierReport(BaseModel):
    """End-of-run summary written to JSON."""

    model_config = ConfigDict(extra="forbid")

    config: ViewClassifierConfig
    class_names: list[str]
    embed_dim: int
    n_train: int
    n_val: int
    per_class_train_counts: dict[str, int] = Field(default_factory=dict)
    class_weights: list[float] = Field(default_factory=list)
    per_epoch: list[ViewEpochMetrics] = Field(default_factory=list)
    best_epoch: int = 0
    best_val_top1: float = 0.0
    val_confusion_matrix: list[list[int]] = Field(default_factory=list)
    checkpoint_path: str = ""
    total_elapsed_s: float = 0.0


@dataclass
class CheckpointPayload:
    """Container returned by :func:`train_view_classifier`.

    Holds the trained head's state dict, a copy of the backbone state
    (for self-contained checkpoints), the canonical class names, the
    config dict, and the final 6x6 val confusion matrix as a tensor.
    The :mod:`view_classifier_cli` writes these to disk via
    ``torch.save``.
    """

    head_state_dict: dict[str, Any]
    image_encoder_state_dict: dict[str, Any]
    class_names: list[str]
    config: dict[str, Any]
    val_confusion_matrix: Any
    report: ViewClassifierReport


# --------------------------------------------------------------- dataset query


_BINARY_SPLIT_BY_BUCKET: dict[int, str] = {
    **dict.fromkeys(range(8), "train"),
    8: "val",
    9: "test",
}


def _derive_binary_split(image_id: str) -> str:
    """Deterministically bucket an image_id into train/val/test.

    Uses ``hashlib.sha1(image_id.encode("utf-8")).digest()[0] % 10``:
    buckets 0..7 → train, 8 → val, 9 → test. Stable across runs and
    machines so the same image_id always lands in the same split.
    """
    bucket = hashlib.sha1(image_id.encode("utf-8"), usedforsecurity=False).digest()[0] % 10
    return _BINARY_SPLIT_BY_BUCKET[bucket]


def build_view_classifier_dataset(
    conn: sqlite3.Connection,
    *,
    split: str,
    min_view_score: float,
    binary: bool = False,
) -> list[tuple[Path, int]]:
    """Return ``(local_path, class_idx)`` tuples for one split.

    Pulls **cross-source** rows -- the view classifier benefits from the
    full mix of crawled phone-camera shots + clean studio public-dataset
    images, so we don't filter by ``listings.source``.

    Filters (6-way mode, ``binary=False``):

    * ``images.view IS NOT NULL`` -- rows that haven't been labeled by
      the Phase 3.3 view labeler are excluded.
    * ``images.split = :split`` -- the per-image stratified split from
      Phase 3.5 (migration 010).
    * ``images.view_score >= min_view_score`` -- drop low-confidence
      labels. The zero-shot labeler is reliable above ~0.6.
    * ``images.local_path IS NOT NULL`` -- defensive; a row that lost
      its file path can't be loaded.

    Filters (binary mode, ``binary=True``):

    * Exterior rows (view in :data:`EXTERIOR_VIEWS`): same filter as
      above (``images.split = :split``).
    * Non-exterior rows (view in :data:`NON_EXTERIOR_RAW`): included
      regardless of ``images.split`` (those rows have NULL split today
      because ``make-splits`` only stratified the exterior pool). The
      split is derived deterministically from
      ``hashlib.sha1(image_id).digest()[0] % 10`` (0..7 = train, 8 = val,
      9 = test) and then filtered to the requested ``split`` value, so
      every run sees the same bucketing.

    Unknown view labels (anything outside :data:`EXTERIOR_VIEWS` U
    :data:`NON_EXTERIOR_RAW`) are dropped with a single aggregated
    WARN log so a stray label doesn't crash the run.
    """
    if binary:
        return _build_binary_dataset(conn, split=split, min_view_score=min_view_score)
    sql = (
        "SELECT images.local_path AS local_path, "
        "       images.view AS view "
        "FROM images "
        "WHERE images.view IS NOT NULL "
        "  AND images.split = ? "
        "  AND images.view_score >= ? "
        "  AND images.local_path IS NOT NULL "
        "ORDER BY images.image_id"
    )
    cur = conn.execute(sql, (split, float(min_view_score)))
    rows: list[tuple[Path, int]] = []
    n_unknown = 0
    for row in cur.fetchall():
        path_str = row["local_path"]
        view = row["view"]
        if not path_str or view is None:
            continue
        try:
            idx = collapse_view(str(view))
        except KeyError:
            n_unknown += 1
            continue
        rows.append((Path(str(path_str)), idx))
    if n_unknown:
        logger.warning(
            "view-classifier: dropped %d rows with unknown view labels in split=%r",
            n_unknown,
            split,
        )
    return rows


def _build_binary_dataset(
    conn: sqlite3.Connection,
    *,
    split: str,
    min_view_score: float,
) -> list[tuple[Path, int]]:
    """Binary-mode dataset builder.

    Pulls exterior rows from the requested ``images.split`` and
    non-exterior rows from anywhere in the table; non-exterior rows are
    deterministically assigned to a split via SHA-1 of their image_id
    (see :func:`_derive_binary_split`) and then filtered to ``split``.
    """
    rows: list[tuple[Path, int]] = []
    n_unknown = 0

    # 1) Exterior rows from the requested split.
    exterior_sql = (
        "SELECT images.local_path AS local_path, "
        "       images.view AS view "
        "FROM images "
        "WHERE images.view IS NOT NULL "
        "  AND images.split = ? "
        "  AND images.view_score >= ? "
        "  AND images.local_path IS NOT NULL "
        "ORDER BY images.image_id"
    )
    cur = conn.execute(exterior_sql, (split, float(min_view_score)))
    for row in cur.fetchall():
        path_str = row["local_path"]
        view = row["view"]
        if not path_str or view is None:
            continue
        view_str = str(view)
        if view_str not in EXTERIOR_VIEWS:
            # Skip non-exterior rows here; they're handled by the
            # second query below (which doesn't require a matching
            # images.split because those rows have NULL split).
            continue
        try:
            idx = collapse_view_to_binary(view_str)
        except KeyError:
            n_unknown += 1
            continue
        rows.append((Path(str(path_str)), idx))

    # 2) Non-exterior rows from anywhere in the table, bucketed
    #    deterministically by SHA-1(image_id) % 10.
    non_exterior_sql = (
        "SELECT images.image_id AS image_id, "
        "       images.local_path AS local_path, "
        "       images.view AS view "
        "FROM images "
        "WHERE images.view IN ('interior', 'detail', 'non-car') "
        "  AND images.view_score >= ? "
        "  AND images.local_path IS NOT NULL "
        "ORDER BY images.image_id"
    )
    cur = conn.execute(non_exterior_sql, (float(min_view_score),))
    for row in cur.fetchall():
        path_str = row["local_path"]
        view = row["view"]
        image_id = row["image_id"]
        if not path_str or view is None or image_id is None:
            continue
        derived_split = _derive_binary_split(str(image_id))
        if derived_split != split:
            continue
        try:
            idx = collapse_view_to_binary(str(view))
        except KeyError:
            n_unknown += 1
            continue
        rows.append((Path(str(path_str)), idx))

    if n_unknown:
        logger.warning(
            "view-classifier: dropped %d rows with unknown view labels in binary split=%r",
            n_unknown,
            split,
        )
    return rows


# --------------------------------------------------------------- class weights


def compute_class_weights(class_counts: list[int]) -> list[float]:
    """Inverse-square-root-frequency class weights, renormalized.

    ``w_i = 1 / sqrt(n_i)``, then ``w <- w * (len(w) / sum(w))`` so the
    mean weight is 1.0. Classes with zero observed examples get a
    weight of 0.0 -- those are dropped from the loss entirely (the
    optimizer can't learn a class it never sees).
    """
    raw = [1.0 / math.sqrt(n) if n > 0 else 0.0 for n in class_counts]
    total = sum(raw)
    if total <= 0:
        return [1.0 for _ in class_counts]
    scale = len(raw) / total
    return [w * scale for w in raw]


# --------------------------------------------------------------- training


def train_view_classifier(
    *,
    conn: sqlite3.Connection,
    config: ViewClassifierConfig,
) -> CheckpointPayload:
    """Run the full view-classifier head training and return the payload.

    This function never touches the network -- the OpenCLIP backbone
    has to be installed locally and the optional fine-tuned checkpoint
    must already be on disk. The runner caches features into RAM
    (~300 MB at 150k × 512 × 4B) then trains the head for
    ``config.epochs`` over the cached tensor. The best epoch (by val
    top-1) is the one whose head state ends up in the returned
    :class:`CheckpointPayload`.
    """
    runner = _ViewClassifierRunner(config=config)
    return runner.run(conn=conn)


# --------------------------------------------------------------- internals


class _ViewClassifierRunner:
    """Encapsulates lazy model loading + the feature-caching train loop."""

    def __init__(self, *, config: ViewClassifierConfig) -> None:
        self._config = config
        self._torch: Any | None = None
        self._open_clip: Any | None = None
        self._model: Any | None = None
        self._preprocess: Any | None = None

    # --------- public entry point ------------------------------------- #

    def run(self, *, conn: sqlite3.Connection) -> CheckpointPayload:
        start = time.perf_counter()
        cfg = self._config
        class_names: tuple[str, ...] = BINARY_CLASS_NAMES if cfg.binary else VIEW_CLASS_NAMES
        n_classes = len(class_names)

        self._seed_everything(cfg.seed)

        train_rows = build_view_classifier_dataset(
            conn,
            split="train",
            min_view_score=cfg.min_view_score,
            binary=cfg.binary,
        )
        val_rows = build_view_classifier_dataset(
            conn,
            split="val",
            min_view_score=cfg.min_view_score,
            binary=cfg.binary,
        )
        if not train_rows:
            raise ValueError(
                "no train rows available -- "
                "verify view labels are populated (run view-label) and that "
                "images.split = 'train' rows pass min_view_score "
                f"({cfg.min_view_score})"
            )

        per_class_train_counts = _count_by_class(train_rows, n_classes=n_classes)
        logger.info(
            "view-classifier: %d train rows, %d val rows; per-class train counts=%s",
            len(train_rows),
            len(val_rows),
            {class_names[i]: per_class_train_counts[i] for i in range(n_classes)},
        )

        self._ensure_model()
        torch_mod = self._require_torch()
        nn = torch_mod.nn

        # 1) Feature caching pass: backbone forward over all train + val
        #    images, store features as a contiguous tensor in CPU RAM.
        embed_dim = self._infer_embed_dim()
        train_feats, train_labels = self._cache_features(
            train_rows, label="train", n_classes=n_classes
        )
        val_feats, val_labels = self._cache_features(val_rows, label="val", n_classes=n_classes)

        # If the cache pass dropped some rows (PIL decode error etc.)
        # the label counts may shift; recompute per_class_train_counts
        # from the cached labels so the report matches what the head
        # actually saw.
        train_label_list = [int(x) for x in train_labels.tolist()]
        per_class_train_counts = [0] * n_classes
        for c in train_label_list:
            per_class_train_counts[c] += 1

        # 2) Class weights: inverse-sqrt-frequency, renormalized.
        weights_list = compute_class_weights(per_class_train_counts)
        weight_tensor = torch_mod.tensor(weights_list, dtype=torch_mod.float32).to(cfg.device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        logger.info("view-classifier: class_weights=%s", weights_list)

        # 3) Head + optimizer.
        head_factory = _resolve_head_factory(cfg.head_arch)
        head = head_factory.build(embed_dim, out_features=n_classes).to(cfg.device)
        optim = torch_mod.optim.AdamW(
            head.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # 4) Training loop over cached features.
        per_epoch: list[ViewEpochMetrics] = []
        best_top1 = -1.0
        best_epoch = 0
        best_head_state: dict[str, Any] = {
            k: v.detach().clone() for k, v in head.state_dict().items()
        }
        best_confusion: Any = torch_mod.zeros(
            (n_classes, n_classes),
            dtype=torch_mod.int64,
        )

        n_train = int(train_feats.shape[0])
        device = cfg.device

        for epoch in range(cfg.epochs):
            epoch_start = time.perf_counter()
            head.train()
            perm = torch_mod.randperm(n_train)
            total_loss = 0.0
            n_batches = 0
            for start_i in range(0, n_train, cfg.batch_size):
                end_i = min(start_i + cfg.batch_size, n_train)
                idx = perm[start_i:end_i]
                batch_feats = train_feats[idx].to(device, non_blocking=True)
                batch_labels = train_labels[idx].to(device, non_blocking=True)
                optim.zero_grad(set_to_none=True)
                logits = head(batch_feats)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optim.step()
                total_loss += float(loss.item())
                n_batches += 1
            train_loss = total_loss / max(1, n_batches)

            # Validation: top-1 + NxN confusion matrix.
            val_top1, confusion = self._evaluate(
                head=head,
                feats=val_feats,
                labels=val_labels,
                n_classes=n_classes,
            )

            elapsed = time.perf_counter() - epoch_start
            per_epoch.append(
                ViewEpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_top1=val_top1,
                    elapsed_s=elapsed,
                )
            )
            logger.info(
                "view-classifier: epoch=%d loss=%.4f val_top1=%.4f elapsed=%.1fs",
                epoch,
                train_loss,
                val_top1,
                elapsed,
            )

            if val_top1 > best_top1:
                best_top1 = val_top1
                best_epoch = epoch
                best_head_state = {
                    k: v.detach().cpu().clone() for k, v in head.state_dict().items()
                }
                best_confusion = confusion.detach().cpu().clone()

        # 5) Build the checkpoint payload.
        model = self._require_model()
        image_encoder_state = _image_encoder_state_dict(model)
        # Ship the encoder weights on CPU so the .pt is portable across
        # GPU-built / CPU-served deployments.
        image_encoder_cpu = {
            k: (v.detach().cpu().clone() if hasattr(v, "detach") else v)
            for k, v in image_encoder_state.items()
        }

        backbone_source = (
            str(cfg.backbone_checkpoint) if cfg.backbone_checkpoint is not None else "pretrained"
        )
        config_dict: dict[str, Any] = {
            "model_name": cfg.model_name,
            "pretrained": cfg.pretrained,
            "embed_dim": embed_dim,
            "backbone_source": backbone_source,
            "head_arch": cfg.head_arch,
            "min_view_score": cfg.min_view_score,
            "epochs_trained": cfg.epochs,
            "best_epoch": best_epoch,
            "best_val_top1": max(best_top1, 0.0),
            "class_weights_strategy": "inverse_sqrt_freq",
        }

        total_elapsed = time.perf_counter() - start

        report = ViewClassifierReport(
            config=cfg,
            class_names=list(class_names),
            embed_dim=embed_dim,
            n_train=int(train_feats.shape[0]),
            n_val=int(val_feats.shape[0]),
            per_class_train_counts={
                class_names[i]: per_class_train_counts[i] for i in range(n_classes)
            },
            class_weights=weights_list,
            per_epoch=per_epoch,
            best_epoch=best_epoch,
            best_val_top1=max(best_top1, 0.0),
            val_confusion_matrix=[[int(x) for x in row] for row in best_confusion.tolist()],
            checkpoint_path="",
            total_elapsed_s=total_elapsed,
        )

        return CheckpointPayload(
            head_state_dict=best_head_state,
            image_encoder_state_dict=image_encoder_cpu,
            class_names=list(class_names),
            config=config_dict,
            val_confusion_matrix=best_confusion,
            report=report,
        )

    # --------- evaluation helpers ------------------------------------- #

    def _evaluate(
        self,
        *,
        head: Any,
        feats: Any,
        labels: Any,
        n_classes: int,
    ) -> tuple[float, Any]:
        torch_mod = self._require_torch()
        cfg = self._config
        head.eval()
        n_total = int(feats.shape[0])
        if n_total == 0:
            return 0.0, torch_mod.zeros(
                (n_classes, n_classes),
                dtype=torch_mod.int64,
            )
        n_correct = 0
        confusion = torch_mod.zeros(
            (n_classes, n_classes),
            dtype=torch_mod.int64,
        )
        with torch_mod.no_grad():
            for start_i in range(0, n_total, cfg.batch_size):
                end_i = min(start_i + cfg.batch_size, n_total)
                batch_feats = feats[start_i:end_i].to(cfg.device, non_blocking=True)
                batch_labels = labels[start_i:end_i].to(cfg.device, non_blocking=True)
                logits = head(batch_feats)
                preds = logits.argmax(dim=-1)
                n_correct += int((preds == batch_labels).sum().item())
                preds_cpu = preds.detach().cpu()
                labels_cpu = batch_labels.detach().cpu()
                for true_i, pred_i in zip(
                    labels_cpu.tolist(),
                    preds_cpu.tolist(),
                    strict=True,
                ):
                    confusion[int(true_i), int(pred_i)] += 1
        return n_correct / n_total, confusion

    # --------- feature caching ---------------------------------------- #

    def _cache_features(
        self,
        rows: list[tuple[Path, int]],
        *,
        label: str,
        n_classes: int,
    ) -> tuple[Any, Any]:
        """Run the backbone over ``rows`` once, return ``(features, labels)``.

        Bad PIL files are logged + skipped. Caller gets a contiguous
        tensor of shape ``(n_ok, embed_dim)`` and an aligned int64
        label tensor of length ``n_ok``.
        """
        torch_mod = self._require_torch()
        model = self._require_model()
        preprocess = self._require_preprocess()
        cfg = self._config
        del n_classes  # unused, kept for symmetry with other helpers

        if not rows:
            embed_dim = self._infer_embed_dim()
            return (
                torch_mod.zeros((0, embed_dim), dtype=torch_mod.float32),
                torch_mod.zeros((0,), dtype=torch_mod.int64),
            )

        from PIL import Image as PILImageModule  # noqa: PLC0415
        from PIL import UnidentifiedImageError  # noqa: PLC0415

        feature_chunks: list[Any] = []
        label_buf: list[int] = []
        n_skipped = 0
        n_done = 0
        n_total = len(rows)

        for chunk in _chunked(rows, cfg.backbone_batch_size):
            tensors: list[Any] = []
            chunk_labels: list[int] = []
            for path, cls_idx in chunk:
                try:
                    with PILImageModule.open(path) as img:
                        converted = img.convert("RGB")
                    tensors.append(preprocess(converted))
                    chunk_labels.append(cls_idx)
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    logger.warning(
                        "view-classifier: skipping %s during %s feature cache (%s)",
                        path,
                        label,
                        exc,
                    )
                    n_skipped += 1
                    continue
            if not tensors:
                continue
            batch = torch_mod.stack(tensors).to(cfg.device)
            with torch_mod.inference_mode():
                feats = model.encode_image(batch)
            feature_chunks.append(feats.detach().to("cpu", dtype=torch_mod.float32))
            label_buf.extend(chunk_labels)
            n_done += len(chunk_labels)
            if n_done and (n_done % max(50, cfg.backbone_batch_size * 4) < cfg.backbone_batch_size):
                logger.info(
                    "view-classifier: cached %d / %d %s features",
                    n_done,
                    n_total,
                    label,
                )

        if not feature_chunks:
            embed_dim = self._infer_embed_dim()
            return (
                torch_mod.zeros((0, embed_dim), dtype=torch_mod.float32),
                torch_mod.zeros((0,), dtype=torch_mod.int64),
            )

        features = torch_mod.cat(feature_chunks, dim=0)
        labels_t = torch_mod.tensor(label_buf, dtype=torch_mod.int64)
        if n_skipped:
            logger.warning(
                "view-classifier: dropped %d / %d %s rows during feature cache",
                n_skipped,
                n_total,
                label,
            )
        return features, labels_t

    # --------- model loading ------------------------------------------ #

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip  # noqa: PLC0415
            import torch  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover -- deps are in pyproject
            raise RuntimeError(
                "open_clip_torch and torch are required for view-classifier training"
            ) from exc
        self._torch = torch
        self._open_clip = open_clip
        cfg = self._config
        logger.info(
            "view-classifier: loading OpenCLIP %s / %s on %s",
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
                f"underlying error: {exc!r}"
            ) from exc

        if cfg.backbone_checkpoint is not None:
            _load_image_encoder_checkpoint(
                model=model,
                checkpoint_path=cfg.backbone_checkpoint,
                device=cfg.device,
                torch_mod=torch,
            )

        # Freeze every backbone parameter.
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        self._model = model
        self._preprocess = preprocess

    def _infer_embed_dim(self) -> int:
        """Probe the loaded backbone for its image-encoder output dim."""
        torch_mod = self._require_torch()
        model = self._require_model()
        cfg = self._config
        # MobileCLIP-S2 expects 256x256; we try a few common sizes for
        # robustness against stubs.
        last_exc: Exception | None = None
        for size in (256, 224, 240, 192, 384):
            try:
                dummy = torch_mod.zeros(1, 3, size, size, device=cfg.device)
                with torch_mod.inference_mode():
                    feats = model.encode_image(dummy)
                return int(feats.shape[-1])
            except Exception as exc:  # noqa: BLE001 -- size probe is allowed to fail
                last_exc = exc
                continue
        raise RuntimeError("view-classifier: could not infer embed_dim from backbone") from last_exc

    # --------- accessors ---------------------------------------------- #

    def _require_torch(self) -> Any:
        if self._torch is None:  # pragma: no cover -- defensive
            raise RuntimeError("torch not loaded; call _ensure_model() first")
        return self._torch

    def _require_model(self) -> Any:
        if self._model is None:  # pragma: no cover -- defensive
            raise RuntimeError("model not loaded; call _ensure_model() first")
        return self._model

    def _require_preprocess(self) -> Any:
        if self._preprocess is None:  # pragma: no cover -- defensive
            raise RuntimeError("preprocess not built; call _ensure_model() first")
        return self._preprocess

    # --------- determinism -------------------------------------------- #

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        try:
            import numpy as np  # noqa: PLC0415

            np.random.seed(seed)
        except ImportError:  # pragma: no cover
            pass
        try:
            import torch  # noqa: PLC0415

            torch.manual_seed(seed)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:  # pragma: no cover
            pass


# --------------------------------------------------------------- helpers


def _count_by_class(
    rows: Iterable[tuple[Path, int]],
    *,
    n_classes: int = len(VIEW_CLASS_NAMES),
) -> list[int]:
    counts = [0] * n_classes
    for _path, idx in rows:
        if 0 <= idx < len(counts):
            counts[idx] += 1
    return counts


def _chunked(seq: list[tuple[Path, int]], size: int) -> Iterable[list[tuple[Path, int]]]:
    if size <= 0:
        raise ValueError(f"chunk size must be > 0, got {size}")
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _image_encoder_state_dict(model: Any) -> dict[str, Any]:
    visual = getattr(model, "visual", None)
    if visual is not None and hasattr(visual, "state_dict"):
        return cast(dict[str, Any], visual.state_dict())
    return cast(dict[str, Any], model.state_dict())


def _load_image_encoder_checkpoint(
    *,
    model: Any,
    checkpoint_path: Path,
    device: str,
    torch_mod: Any,
) -> None:
    """Overlay a Phase 5.2 fine-tuned ``image_encoder_state_dict`` on ``model``.

    Mirrors :func:`car_lense_engine.eval.baseline._load_image_encoder_checkpoint`
    but is duplicated here so the two modules don't grow a tight
    coupling. The checkpoint format is the dict produced by
    :func:`car_lense_engine.training.run_training`: ``payload`` must be a
    dict with an ``"image_encoder_state_dict"`` key keyed on
    ``model.visual``'s parameter names.

    Bad-checkpoint cases (missing file, wrong format, missing key)
    raise :class:`RuntimeError`. We do **not** silently fall back to
    the pretrained weights -- a misconfigured ``--backbone-checkpoint``
    must surface as an error so the user knows what they're training.
    """
    if not checkpoint_path.exists():
        raise RuntimeError(f"backbone checkpoint path does not exist: {checkpoint_path}")
    payload = torch_mod.load(checkpoint_path, map_location=device, weights_only=False)
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
    logger.info(
        "view-classifier: loaded fine-tuned backbone from %s (epoch=%s val_top1=%s)",
        checkpoint_path,
        payload.get("epoch"),
        payload.get("val_top1"),
    )


def write_report(report: ViewClassifierReport, path: Path) -> None:
    """Serialize a :class:`ViewClassifierReport` to JSON at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


__all__ = [
    "BINARY_CLASS_NAMES",
    "EXTERIOR_VIEWS",
    "NON_EXTERIOR_RAW",
    "VIEW_CLASS_NAMES",
    "CheckpointPayload",
    "LinearHead",
    "MLPHead",
    "ViewClassifierConfig",
    "ViewClassifierReport",
    "ViewEpochMetrics",
    "build_view_classifier_dataset",
    "collapse_view",
    "collapse_view_to_binary",
    "compute_class_weights",
    "train_view_classifier",
    "write_report",
]
