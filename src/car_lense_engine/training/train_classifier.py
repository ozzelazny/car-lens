"""Fine-tune MobileCLIP-S2 with a linear classification head (Phase 5.2).

This module trains a 196-way (or whatever the source dictates) car
classifier on top of an OpenCLIP image encoder. The recipe is the
standard "full backbone + new linear head" CLIP fine-tune that lifts
Stanford Cars top-1 from the ~88% zero-shot prototype baseline (Phase
5.1) into the 92-95% range typically reported in the literature.

Design notes
------------

* **Full backbone fine-tune.** The image encoder is *not* frozen. The
  backbone receives a small learning rate (``lr_backbone``) while the
  randomly-initialized linear head gets a larger one (``lr_head``). This
  two-param-group AdamW setup is the standard transfer-learning recipe
  for CLIP and is materially better than freezing the backbone on
  fine-grained tasks.

* **Hard-negative loss weighting.** The Phase 5.1 confusion-pair JSON
  identifies which classes the zero-shot model already confuses (Audi
  S5 vs A5 coupe, Silverado regular vs extended cab, etc.). Classes that
  participate as EITHER the true class OR the predicted class in any of
  those confusion pairs are boosted -- both sides of a confusion pair
  benefit -- by a multiplicative weight (``hard_neg_weight``, default
  2.0) on their cross-entropy loss. This is the simplest form of
  hard-negative mining that doesn't require pair-aware batch sampling
  -- it just pushes the optimizer to spend more capacity on the classes
  that need it.

* **Augmentation.** Train images get RandomResizedCrop + ColorJitter +
  HorizontalFlip (cars are bilaterally symmetric for the brand-id task,
  so horizontal flip is safe; vertical flip is NOT used because cars
  have up/down asymmetry). Validation uses the OpenCLIP-default
  resize-and-center-crop transform.

* **CPU-friendly.** AMP is only enabled when ``device.startswith("cuda")``;
  the CPU path runs in float32. ``num_workers=0`` is supported for
  Windows / WSL users who hit multiprocessing issues.

* **Lazy heavyweight imports.** ``torch`` / ``open_clip`` / ``PIL`` are
  imported inside :class:`_TrainingRunner` so unit tests can stub them
  the same way the baseline tests do.

Storage layout: checkpoints are written to ``models/checkpoints/`` (a
symlink to native fs per DESIGN.md) with the filename pattern
``<model>_<source>_epoch<NN>_top1_<XX.X>.pt``.
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from car_lense_engine.eval.baseline import class_id_for

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- public models


class TrainConfig(BaseModel):
    """Hyperparameters frozen at the start of a training run."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "MobileCLIP-S2"
    """OpenCLIP model name."""

    pretrained: str = "datacompdr"
    """OpenCLIP pretrained tag."""

    source: str = "stanford_cars"
    """``listings.source`` value to train against."""

    train_split: str = "train"
    val_split: str = "test"
    device: str = "cpu"
    batch_size: int = 64
    num_workers: int = 2
    epochs: int = 20

    lr_backbone: float = 1e-5
    """Small LR on the pre-trained backbone."""

    lr_head: float = 1e-3
    """Larger LR on the freshly-initialized linear head."""

    weight_decay: float = 0.01
    warmup_epochs: int = 1
    label_smoothing: float = 0.1

    hard_neg_weight: float = 2.0
    """Multiplicative weight applied to the CE loss for classes that
    participate in a Phase 5.1 confusion pair (as either the true or
    predicted class). ``1.0`` disables hard-negative weighting."""

    hard_neg_confusion_path: Path | None = None
    """Path to the Phase 5.1 baseline JSON report. If ``None`` or the
    file is missing, the class-weight vector is all-ones."""

    aug_random_resized_crop: bool = True
    aug_color_jitter: bool = True
    aug_horizontal_flip: bool = True

    seed: int = 42


class EpochMetrics(BaseModel):
    """One row in the per-epoch training log."""

    model_config = ConfigDict(extra="forbid")

    epoch: int
    train_loss: float
    val_top1: float
    val_top5: float
    lr_backbone: float
    lr_head: float
    elapsed_s: float


class TrainReport(BaseModel):
    """End-of-run summary: hyperparameters + every epoch's metrics."""

    model_config = ConfigDict(extra="forbid")

    config: TrainConfig
    n_classes: int
    n_train: int
    n_val: int
    per_epoch: list[EpochMetrics] = Field(default_factory=list)
    best_epoch: int = 0
    best_val_top1: float = 0.0
    best_val_top5: float = 0.0
    checkpoint_path: str = ""
    total_elapsed_s: float = 0.0


# --------------------------------------------------------------- helpers


def _read_confusion_classes(path: Path) -> set[str]:
    """Return the set of class ids participating in any confusion pair.

    The Phase 5.1 baseline JSON has a ``confusion_top_pairs`` list of
    ``{true_class, predicted_class, count}`` rows; we collect both sides
    so that boosting the loss for *either* the misclassified class or
    the confuser gets weight, since either correction helps.
    """
    if not path.exists():
        logger.warning(
            "train: hard-negative confusion file %s not found -- weights will be all ones",
            path,
        )
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "train: failed to parse confusion file %s (%s) -- weights will be all ones",
            path,
            exc,
        )
        return set()
    out: set[str] = set()
    for pair in payload.get("confusion_top_pairs", []) or []:
        t = pair.get("true_class")
        p = pair.get("predicted_class")
        if isinstance(t, str):
            out.add(t)
        if isinstance(p, str):
            out.add(p)
    return out


def build_class_weights_from_confusion(
    *,
    class_ids: list[str],
    confusion_path: Path | None,
    hard_neg_weight: float,
) -> list[float]:
    """Build the per-class CE weight vector.

    Returns a list aligned with ``class_ids`` where each entry is
    ``hard_neg_weight`` if the class appears in any confusion pair in
    the file at ``confusion_path``, else ``1.0``.

    A missing or unreadable file (or ``confusion_path=None``) yields a
    flat all-ones vector; a warning is logged.
    """
    if hard_neg_weight <= 0:
        raise ValueError(f"hard_neg_weight must be > 0, got {hard_neg_weight}")
    if not class_ids:
        return []
    if confusion_path is None:
        return [1.0] * len(class_ids)
    hard = _read_confusion_classes(confusion_path)
    if not hard:
        return [1.0] * len(class_ids)
    return [hard_neg_weight if cid in hard else 1.0 for cid in class_ids]


def _select_train_rows(
    conn: sqlite3.Connection,
    *,
    source: str,
    split: str,
) -> list[tuple[str, Path]]:
    """Return ``(class_id, local_path)`` pairs for ``(source, split)``.

    Mirrors the helper in :mod:`car_lense_engine.eval.baseline` but is
    duplicated here so the two modules don't grow a tight coupling on
    each other's private SQL helpers. Rows with NULL ``(year, make,
    model)`` are skipped.
    """
    sql = (
        "SELECT listings.year AS year, listings.make AS make, listings.model AS model, "
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


def write_report(report: TrainReport, path: Path) -> None:
    """Serialize a :class:`TrainReport` to JSON at ``path`` (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


# --------------------------------------------------------------- public API


def run_training(
    *,
    conn: sqlite3.Connection,
    config: TrainConfig,
    checkpoint_dir: Path,
) -> TrainReport:
    """Run the full Phase 5.2 fine-tune.

    Loads MobileCLIP-S2, attaches a ``Linear(embed_dim, n_classes)`` head,
    trains for ``config.epochs`` epochs over the ``(source, train_split)``
    slice with augmentation and hard-negative-weighted cross-entropy, and
    saves the best checkpoint (by val top-1 on ``(source, val_split)``)
    to ``checkpoint_dir``. Returns the full :class:`TrainReport`.
    """
    runner = _TrainingRunner(config=config, checkpoint_dir=checkpoint_dir)
    return runner.run(conn=conn)


# --------------------------------------------------------------- internals


class _TrainingRunner:
    """Encapsulates the lazy-loaded model + the training loop."""

    def __init__(self, *, config: TrainConfig, checkpoint_dir: Path) -> None:
        self._config = config
        self._checkpoint_dir = checkpoint_dir
        self._torch: Any | None = None
        self._open_clip: Any | None = None
        self._model: Any | None = None  # the OpenCLIP model (we use .visual)
        self._train_preprocess: Any | None = None
        self._val_preprocess: Any | None = None

    # ----- entry point -------------------------------------------------- #

    def run(self, *, conn: sqlite3.Connection) -> TrainReport:
        start = time.perf_counter()
        config = self._config

        self._seed_everything(config.seed)

        train_rows = _select_train_rows(conn, source=config.source, split=config.train_split)
        val_rows = _select_train_rows(conn, source=config.source, split=config.val_split)

        if not train_rows:
            raise ValueError(
                f"no train rows for source={config.source!r} split={config.train_split!r}"
            )

        # Class id space is the union of train + val so the head's output
        # dim covers any val-only class. (We still warn about val-only
        # classes -- they can never be learned without train samples.)
        class_ids = sorted({cid for cid, _ in train_rows} | {cid for cid, _ in val_rows})
        n_classes = len(class_ids)
        class_to_idx = {cid: i for i, cid in enumerate(class_ids)}

        train_only_classes = {cid for cid, _ in train_rows}
        val_only = [cid for cid in class_ids if cid not in train_only_classes]
        if val_only:
            logger.warning(
                "train: %d val-only classes have no train samples and cannot be learned: %s",
                len(val_only),
                val_only[:5],
            )

        logger.info(
            "train: %d classes, %d train images, %d val images",
            n_classes,
            len(train_rows),
            len(val_rows),
        )

        # Handle the trivial "no work to do" case before loading torch.
        if config.epochs <= 0:
            elapsed = time.perf_counter() - start
            return TrainReport(
                config=config,
                n_classes=n_classes,
                n_train=len(train_rows),
                n_val=len(val_rows),
                per_epoch=[],
                best_epoch=0,
                best_val_top1=0.0,
                best_val_top5=0.0,
                checkpoint_path="",
                total_elapsed_s=elapsed,
            )

        self._ensure_model()
        torch_mod = self._require_torch()
        nn = torch_mod.nn

        # Build the head. embed_dim is read off the loaded backbone.
        embed_dim = self._infer_embed_dim()
        head = nn.Linear(embed_dim, n_classes)
        head = head.to(config.device)

        # Move the model to the device too (open_clip.create_model_and_transforms
        # already does this when `device=` is passed, but be defensive).
        model = self._require_model()
        model = model.to(config.device)
        model.train()
        head.train()

        # Hard-negative-weighted CE.
        class_weights = build_class_weights_from_confusion(
            class_ids=class_ids,
            confusion_path=config.hard_neg_confusion_path,
            hard_neg_weight=config.hard_neg_weight,
        )
        weight_tensor = torch_mod.tensor(class_weights, dtype=torch_mod.float32).to(config.device)
        criterion = nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=config.label_smoothing,
        )

        # Two AdamW param groups: backbone (lr_backbone) + head (lr_head).
        optim = torch_mod.optim.AdamW(
            [
                {
                    "params": list(self._iter_backbone_params(model)),
                    "lr": config.lr_backbone,
                },
                {"params": list(head.parameters()), "lr": config.lr_head},
            ],
            weight_decay=config.weight_decay,
        )

        # Cosine schedule with linear warmup over `warmup_epochs`.
        warmup = max(0, config.warmup_epochs)
        total = config.epochs

        def lr_lambda(epoch: int) -> float:
            if total <= 0:
                return 1.0
            if warmup > 0 and epoch < warmup:
                return float(epoch + 1) / float(warmup + 1)
            # Cosine from epoch=warmup..total-1 -> 1.0 .. 0.0
            denom = max(1, total - warmup)
            t = (epoch - warmup) / denom
            return 0.5 * (1.0 + _cos(t * _PI))

        scheduler = torch_mod.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

        # DataLoaders.
        train_ds = _ImagePathDataset(
            rows=train_rows,
            class_to_idx=class_to_idx,
            preprocess=self._require_train_preprocess(),
        )
        val_ds = _ImagePathDataset(
            rows=val_rows,
            class_to_idx=class_to_idx,
            preprocess=self._require_val_preprocess(),
        )
        train_loader = self._make_loader(train_ds, shuffle=True)
        val_loader = self._make_loader(val_ds, shuffle=False)

        # AMP only on CUDA.
        use_amp = config.device.startswith("cuda")
        scaler = _make_grad_scaler(torch_mod, enabled=use_amp)

        # Training loop.
        per_epoch: list[EpochMetrics] = []
        best_top1 = -1.0
        best_top5 = 0.0
        best_epoch = 0
        best_path = ""

        for epoch in range(config.epochs):
            epoch_start = time.perf_counter()
            train_loss = self._train_one_epoch(
                model=model,
                head=head,
                loader=train_loader,
                criterion=criterion,
                optim=optim,
                scaler=scaler,
                use_amp=use_amp,
            )
            val_top1, val_top5 = self._validate(model=model, head=head, loader=val_loader)
            scheduler.step()

            lrs = [g["lr"] for g in optim.param_groups]
            lr_backbone = float(lrs[0]) if lrs else 0.0
            lr_head = float(lrs[1]) if len(lrs) > 1 else lr_backbone
            elapsed = time.perf_counter() - epoch_start
            per_epoch.append(
                EpochMetrics(
                    epoch=epoch,
                    train_loss=float(train_loss),
                    val_top1=float(val_top1),
                    val_top5=float(val_top5),
                    lr_backbone=lr_backbone,
                    lr_head=lr_head,
                    elapsed_s=float(elapsed),
                )
            )
            logger.info(
                "train: epoch=%d loss=%.4f val_top1=%.4f val_top5=%.4f elapsed=%.1fs",
                epoch,
                train_loss,
                val_top1,
                val_top5,
                elapsed,
            )

            if val_top1 > best_top1:
                best_top1 = float(val_top1)
                best_top5 = float(val_top5)
                best_epoch = epoch
                best_path = self._save_checkpoint(
                    model=model,
                    head=head,
                    epoch=epoch,
                    val_top1=best_top1,
                    class_ids=class_ids,
                )

        total_elapsed = time.perf_counter() - start
        return TrainReport(
            config=config,
            n_classes=n_classes,
            n_train=len(train_rows),
            n_val=len(val_rows),
            per_epoch=per_epoch,
            best_epoch=best_epoch,
            best_val_top1=max(best_top1, 0.0),
            best_val_top5=best_top5,
            checkpoint_path=best_path,
            total_elapsed_s=float(total_elapsed),
        )

    # ----- model + preprocess ------------------------------------------ #

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip  # noqa: PLC0415
            import torch  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover -- deps are in pyproject
            raise RuntimeError(
                "open_clip_torch and torch are required for the training harness"
            ) from exc
        self._torch = torch
        self._open_clip = open_clip
        cfg = self._config
        logger.info(
            "train: loading OpenCLIP %s / %s on %s",
            cfg.model_name,
            cfg.pretrained,
            cfg.device,
        )
        model, _, val_preprocess = open_clip.create_model_and_transforms(
            cfg.model_name,
            pretrained=cfg.pretrained,
            device=cfg.device,
        )
        train_preprocess = self._build_train_preprocess(val_preprocess)
        self._model = model
        self._val_preprocess = val_preprocess
        self._train_preprocess = train_preprocess

    def _build_train_preprocess(self, val_preprocess: Any) -> Any:
        """Build the train-time preprocess pipeline.

        Strategy: re-use the val preprocess to get the model's required
        size + normalization, but prepend augmentations. We do this by
        sniffing the ``Resize`` / ``CenterCrop`` to find the target size
        and building a fresh ``Compose`` whose final ``Normalize`` we
        copy from the val transform. If introspection fails, we fall
        back to the val preprocess (no augmentation -- safer than crash).
        """
        try:
            from torchvision import transforms as T  # noqa: PLC0415
        except ImportError:  # pragma: no cover -- torchvision is shipped with torch
            logger.warning("train: torchvision not available -- skipping train augmentation")
            return val_preprocess

        cfg = self._config
        target_size = _infer_target_size(val_preprocess)
        normalize = _find_normalize(val_preprocess)
        if target_size is None or normalize is None:
            logger.warning(
                "train: could not introspect val preprocess; falling back to no augmentation"
            )
            return val_preprocess

        steps: list[Any] = []
        if cfg.aug_random_resized_crop:
            steps.append(T.RandomResizedCrop(target_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)))
        else:
            steps.append(T.Resize(target_size))
            steps.append(T.CenterCrop(target_size))
        if cfg.aug_horizontal_flip:
            steps.append(T.RandomHorizontalFlip(p=0.5))
        if cfg.aug_color_jitter:
            steps.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
        steps.append(T.ToTensor())
        steps.append(normalize)
        return T.Compose(steps)

    def _infer_embed_dim(self) -> int:
        """Probe the loaded model for its image-encoder output dim."""
        torch_mod = self._require_torch()
        model = self._require_model()
        cfg = self._config
        with torch_mod.no_grad():
            # 3-channel dummy of the model's expected input size. We use
            # the val preprocess to convert a synthetic blank image so
            # the input is correctly normalized + shaped for the model.
            dummy = torch_mod.zeros(1, 3, 224, 224, device=cfg.device)
            # Try a few sizes if 224 isn't right. MobileCLIP-S2 uses 256.
            for size in (224, 256, 240, 192, 384):
                try:
                    dummy = torch_mod.zeros(1, 3, size, size, device=cfg.device)
                    feats = model.encode_image(dummy)
                    return int(feats.shape[-1])
                except Exception:  # noqa: BLE001 -- size probe is allowed to fail
                    continue
            # Last resort: try once more without try/except so the error
            # surfaces with full context.
            feats = model.encode_image(dummy)
        return int(feats.shape[-1])

    def _iter_backbone_params(self, model: Any) -> Iterator[Any]:
        """Iterate the image-encoder parameters we want to train.

        OpenCLIP models expose ``.visual`` (the image tower) plus a text
        tower we don't need for classification. To keep the optimizer
        small and avoid pointlessly updating the text encoder, we only
        grab the visual tower's params if present; otherwise we fall
        back to ``model.parameters()``.
        """
        visual = getattr(model, "visual", None)
        if visual is not None and hasattr(visual, "parameters"):
            yield from visual.parameters()
            return
        yield from model.parameters()

    # ----- training body ----------------------------------------------- #

    def _train_one_epoch(
        self,
        *,
        model: Any,
        head: Any,
        loader: Iterable[tuple[Any, Any]],
        criterion: Any,
        optim: Any,
        scaler: Any,
        use_amp: bool,
    ) -> float:
        torch_mod = self._require_torch()
        device = self._config.device
        model.train()
        head.train()
        total_loss = 0.0
        n_batches = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            if use_amp:
                with _amp_autocast(torch_mod):
                    feats = model.encode_image(images)
                    logits = head(feats)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                feats = model.encode_image(images)
                logits = head(feats)
                loss = criterion(logits, labels)
                loss.backward()
                optim.step()
            total_loss += float(loss.item())
            n_batches += 1
        return total_loss / max(1, n_batches)

    def _validate(
        self,
        *,
        model: Any,
        head: Any,
        loader: Iterable[tuple[Any, Any]],
    ) -> tuple[float, float]:
        torch_mod = self._require_torch()
        device = self._config.device
        model.eval()
        head.eval()
        n_total = 0
        n_top1 = 0
        n_top5 = 0
        with torch_mod.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                feats = model.encode_image(images)
                logits = head(feats)
                # top-5 (or fewer if n_classes < 5).
                k = min(5, logits.shape[-1])
                _, topk = logits.topk(k=k, dim=-1)
                top1 = topk[:, 0]
                n_total += int(labels.shape[0])
                n_top1 += int((top1 == labels).sum().item())
                # broadcast labels for membership check
                labels_e = labels.unsqueeze(1)
                n_top5 += int((topk == labels_e).any(dim=-1).sum().item())
        if n_total == 0:
            return 0.0, 0.0
        return n_top1 / n_total, n_top5 / n_total

    # ----- checkpointing ----------------------------------------------- #

    def _save_checkpoint(
        self,
        *,
        model: Any,
        head: Any,
        epoch: int,
        val_top1: float,
        class_ids: list[str],
    ) -> str:
        torch_mod = self._require_torch()
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        slug = self._config.model_name.lower().replace("-", "_").replace("/", "_")
        source_slug = self._config.source.lower().replace("/", "_")
        fname = f"{slug}_{source_slug}_epoch{epoch:02d}_top1_{val_top1 * 100:.1f}.pt"
        path = self._checkpoint_dir / fname
        # Save only the visual tower's state_dict (drops the text tower
        # we never update); falls back to the full model state if no
        # `.visual` attribute is present (stub-friendly).
        image_encoder_state = self._image_encoder_state_dict(model)
        payload = {
            "image_encoder_state_dict": image_encoder_state,
            "head_state_dict": head.state_dict(),
            "config": self._config.model_dump(mode="json"),
            "n_classes": len(class_ids),
            "class_ids": class_ids,
            "epoch": epoch,
            "val_top1": float(val_top1),
        }
        torch_mod.save(payload, path)
        logger.info("train: wrote checkpoint %s", path)
        return str(path)

    def _image_encoder_state_dict(self, model: Any) -> dict[str, Any]:
        visual = getattr(model, "visual", None)
        if visual is not None and hasattr(visual, "state_dict"):
            return cast(dict[str, Any], visual.state_dict())
        return cast(dict[str, Any], model.state_dict())

    # ----- data loader plumbing ---------------------------------------- #

    def _make_loader(self, dataset: _ImagePathDataset, *, shuffle: bool) -> Any:
        torch_mod = self._require_torch()
        cfg = self._config
        if len(dataset) == 0:
            # Empty val set is allowed (we'll just report 0.0 / 0.0).
            return _EmptyLoader()
        kwargs: dict[str, Any] = {
            "batch_size": cfg.batch_size,
            "shuffle": shuffle,
            "num_workers": cfg.num_workers,
            "drop_last": False,
            "collate_fn": _default_collate,
        }
        if cfg.num_workers > 0:
            kwargs["persistent_workers"] = True
        return torch_mod.utils.data.DataLoader(dataset, **kwargs)

    # ----- accessors --------------------------------------------------- #

    def _require_torch(self) -> Any:
        if self._torch is None:  # pragma: no cover -- defensive
            raise RuntimeError("torch not loaded; call _ensure_model() first")
        return self._torch

    def _require_model(self) -> Any:
        if self._model is None:  # pragma: no cover -- defensive
            raise RuntimeError("model not loaded; call _ensure_model() first")
        return self._model

    def _require_train_preprocess(self) -> Any:
        if self._train_preprocess is None:  # pragma: no cover -- defensive
            raise RuntimeError("train preprocess not built; call _ensure_model() first")
        return self._train_preprocess

    def _require_val_preprocess(self) -> Any:
        if self._val_preprocess is None:  # pragma: no cover -- defensive
            raise RuntimeError("val preprocess not built; call _ensure_model() first")
        return self._val_preprocess

    # ----- determinism ------------------------------------------------- #

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        try:
            import numpy as np  # noqa: PLC0415

            np.random.seed(seed)
        except ImportError:  # pragma: no cover -- numpy ships with torch
            pass
        # Torch may not be loaded yet (epochs=0 fast path); only seed if
        # the import is cheap. We use a deferred import to avoid forcing
        # torch loading in tests that never touch the model.
        try:
            import torch  # noqa: PLC0415

            torch.manual_seed(seed)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:  # pragma: no cover
            pass


# --------------------------------------------------------------- dataset


class _ImagePathDataset:
    """A torch-style ``Dataset`` over ``(class_id, local_path)`` rows.

    Lazy-imports ``PIL`` so the module remains import-cheap. Rows that
    fail to load (missing file, decode error, etc.) are skipped at
    iteration time: ``__getitem__`` returns ``None`` and the caller
    (via the collate function) drops the bad sample. This means batch
    sizes may shrink late in an epoch, but training is robust to dirty
    data.
    """

    def __init__(
        self,
        *,
        rows: list[tuple[str, Path]],
        class_to_idx: dict[str, int],
        preprocess: Any,
    ) -> None:
        self._rows = rows
        self._class_to_idx = class_to_idx
        self._preprocess = preprocess

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[Any, int] | None:
        cid, path = self._rows[idx]
        label = self._class_to_idx[cid]
        from PIL import Image as PILImageModule  # noqa: PLC0415
        from PIL import UnidentifiedImageError  # noqa: PLC0415

        try:
            with PILImageModule.open(path) as img:
                converted = img.convert("RGB")
            tensor = self._preprocess(converted)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            logger.warning("training: skipping %s (%s)", path, exc)
            return None
        return tensor, label


class _EmptyLoader:
    """Stand-in for ``DataLoader`` when the underlying dataset is empty."""

    def __iter__(self) -> Iterator[Any]:
        return iter(())

    def __len__(self) -> int:
        return 0


def _default_collate(batch: list[tuple[Any, int] | None]) -> tuple[Any, Any]:
    """Stack a list of (tensor, label) into a (batch_tensor, label_tensor).

    Skips any entries that are ``None`` (placeholder for failed loads).
    Imports torch lazily inside the function so this module is
    import-cheap.
    """
    import torch  # noqa: PLC0415

    tensors: list[Any] = []
    labels: list[int] = []
    for entry in batch:
        if entry is None:
            continue
        t, label = entry
        tensors.append(t)
        labels.append(int(label))
    if not tensors:
        return torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)
    return torch.stack(tensors, dim=0), torch.tensor(labels, dtype=torch.long)


# --------------------------------------------------------------- small utils


_PI = 3.141592653589793


def _cos(x: float) -> float:
    """Stdlib cos so this module doesn't need numpy at import time."""
    import math  # noqa: PLC0415

    return math.cos(x)


def _infer_target_size(transform: Any) -> int | None:
    """Walk a torchvision ``Compose`` to find the model's input HxW.

    Looks for a ``CenterCrop`` first (more specific than ``Resize``).
    Returns the integer side length, or ``None`` if introspection fails.
    """
    candidate: int | None = None
    transforms = getattr(transform, "transforms", None)
    if not transforms:
        return None
    for t in transforms:
        name = type(t).__name__
        if name == "CenterCrop":
            size = getattr(t, "size", None)
            if isinstance(size, int):
                return size
            if isinstance(size, tuple | list) and size:
                return int(size[0])
        if name == "Resize" and candidate is None:
            size = getattr(t, "size", None)
            if isinstance(size, int):
                candidate = size
            elif isinstance(size, tuple | list) and size:
                candidate = int(size[0])
    return candidate


def _find_normalize(transform: Any) -> Any | None:
    """Return the ``Normalize`` step inside a ``Compose``, if any."""
    transforms = getattr(transform, "transforms", None)
    if not transforms:
        return None
    for t in transforms:
        if type(t).__name__ == "Normalize":
            return t
    return None


def _make_grad_scaler(torch_mod: Any, *, enabled: bool) -> Any:
    """Build a ``GradScaler`` if AMP is enabled, else a no-op stand-in.

    Older torch versions exposed ``torch.cuda.amp.GradScaler``; newer
    ones recommend ``torch.amp.GradScaler('cuda')``. We try the new
    location first, fall back to the old.
    """
    if not enabled:
        return _NoOpScaler()
    try:
        return torch_mod.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        try:
            return torch_mod.cuda.amp.GradScaler()
        except AttributeError:
            return _NoOpScaler()


class _NoOpScaler:
    """Stand-in for ``GradScaler`` when AMP is disabled.

    Provides the same ``scale().backward() / step() / update()`` surface
    so the training loop is branchless.
    """

    def scale(self, loss: Any) -> Any:
        return loss

    def step(self, optim: Any) -> None:
        optim.step()

    def update(self) -> None:
        return None


class _AutocastNoOp:
    """Context manager that does nothing (used when AMP is off)."""

    def __enter__(self) -> _AutocastNoOp:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


def _amp_autocast(torch_mod: Any) -> Any:
    """Return a CUDA autocast context manager (or no-op if unavailable)."""
    try:
        return torch_mod.amp.autocast("cuda")
    except (AttributeError, TypeError):
        try:
            return torch_mod.cuda.amp.autocast()
        except AttributeError:
            return _AutocastNoOp()


__all__ = [
    "EpochMetrics",
    "TrainConfig",
    "TrainReport",
    "build_class_weights_from_confusion",
    "run_training",
    "write_report",
]
