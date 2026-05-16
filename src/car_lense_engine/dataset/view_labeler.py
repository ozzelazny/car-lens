"""OpenCLIP zero-shot view + content labeler.

Phase 3.3 of the recognition-engine pipeline. Given a list of local image
paths, assign each one a ``view`` in
``{front, rear, side, three-quarter-front, three-quarter-rear,
interior, detail, non-car}`` plus the winning-class softmax probability.

Implementation notes:

* The OpenCLIP model and ``torch`` are imported lazily inside the labeler
  (the first call to :meth:`ViewLabeler.label_batch` triggers the load).
  Simply importing this module does *not* require torch/open_clip — handy
  for unit tests that monkey-patch the model.
* For each view we encode an ensemble of prompts and average the (L2-norm)
  text embeddings; that mean is then re-normalized. This is the standard
  CLIP prompt-ensembling trick and noticeably reduces sensitivity to
  individual prompt wording.
* All inference runs under ``torch.no_grad()`` and ``model.eval()``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


ViewName = Literal[
    "front",
    "rear",
    "side",
    "three-quarter-front",
    "three-quarter-rear",
    "interior",
    "detail",
    "non-car",
]


# The canonical ordering used everywhere (text embeddings, logits, softmax).
VIEW_NAMES: tuple[ViewName, ...] = (
    "front",
    "rear",
    "side",
    "three-quarter-front",
    "three-quarter-rear",
    "interior",
    "detail",
    "non-car",
)


_VIEW_PROMPTS: dict[ViewName, list[str]] = {
    "front": [
        "a photo of the front of a car",
        "a head-on photo of a car from the front",
        "a car photographed from directly in front, showing the grille and headlights",
    ],
    "rear": [
        "a photo of the rear of a car",
        "a photo of the back of a car showing the tail lights",
        "a car photographed from directly behind",
    ],
    "side": [
        "a photo of the side profile of a car",
        "a car photographed from directly to the side, showing the doors",
        "a side-view photo of a car parked",
    ],
    "three-quarter-front": [
        "a photo of a car from the front three-quarter angle",
        "a car photographed at an angle showing the front and side",
        "a three-quarter front view of a car",
    ],
    "three-quarter-rear": [
        "a photo of a car from the rear three-quarter angle",
        "a car photographed at an angle showing the rear and side",
        "a three-quarter rear view of a car",
    ],
    "interior": [
        "a photo of the interior of a car",
        "a photo of a car dashboard and steering wheel from inside",
        "a photo of car seats and the cabin from inside",
    ],
    "detail": [
        "a close-up photo of a part of a car",
        "a closeup of a car badge or emblem",
        "a closeup of a car wheel or rim",
        "a closeup of a car taillight or headlight",
    ],
    "non-car": [
        "a photo that is not a car",
        "a photo of a person",
        "a photo of a document or paperwork",
        "a photo of an empty parking lot",
    ],
}


class ViewLabel(BaseModel):
    """One labeling decision returned by :meth:`ViewLabeler.label_batch`."""

    model_config = ConfigDict(extra="forbid")

    view: ViewName
    score: float  # 0..1 softmax probability of the winning view


class ViewLabeler:
    """Zero-shot view + content labeler over an OpenCLIP backbone.

    Heavy initialization (downloading / loading the OpenCLIP weights, encoding
    the prompt ensemble) is deferred to the first call to
    :meth:`label_batch`. Use the labeler as a context manager — or call
    :meth:`close` explicitly — to release any model state when done.
    """

    def __init__(
        self,
        *,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._device = device
        self._batch_size = batch_size
        # Populated lazily by _ensure_model().
        self._torch: Any | None = None
        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._tokenizer: Any | None = None
        self._text_embeds: Any | None = None  # shape: (num_views, embed_dim)
        # Cached scalar pulled from the model's learned ``logit_scale``
        # parameter on first load. OpenCLIP stores ``log(scale)``; we cache
        # ``exp().item()`` so the hot path doesn't repeatedly call torch.
        self._logit_scale: float | None = None

    # ----- public API -------------------------------------------------- #

    def label_batch(self, paths: list[Path]) -> list[tuple[Path, ViewLabel]]:
        """Label each image at ``paths``. Returns ``(path, label)`` tuples
        ONLY for successful loads.

        Empty input returns an empty list without touching the model.
        Images are processed in chunks of :attr:`batch_size` to keep memory
        steady when the caller hands us thousands of paths. Bad / missing
        files are logged at WARNING level and silently dropped from the
        result — the caller can detect skipped paths by comparing input vs
        output length, but typically just iterates the returned tuples.
        """
        if not paths:
            return []
        self._ensure_model()
        torch_mod = self._require_torch()
        results: list[tuple[Path, ViewLabel]] = []
        for chunk in _chunked(paths, self._batch_size):
            # Per-image load with isolation: a single bad/missing file must
            # not bring down the whole chunk. We collect only the successful
            # (path, tensor) pairs and proceed with whatever survived.
            ok_paths: list[Path] = []
            tensors: list[Any] = []
            for p in chunk:
                try:
                    tensors.append(self._load_and_preprocess(p))
                except Exception as exc:  # noqa: BLE001 - log + skip is the contract
                    logger.warning("view-label: skipping %s (%s)", p, exc)
                    continue
                ok_paths.append(p)
            if not tensors:
                continue
            batch = torch_mod.stack(tensors).to(self._device)
            with torch_mod.no_grad():
                image_features = self._encode_image(batch)
                image_features = self._l2_normalize(image_features)
                # Cosine similarity against the (num_views, dim) text embeds,
                # scaled by the model's learned ``logit_scale`` (cached on
                # first load) so softmax is meaningfully peaked.
                text_embeds = self._require_text_embeds()
                logit_scale = self._require_logit_scale()
                logits = logit_scale * image_features @ text_embeds.T
                probs = logits.softmax(dim=-1).cpu()
            for ok_path, row in zip(ok_paths, probs, strict=True):
                argmax_idx = int(row.argmax().item())
                score = float(row[argmax_idx].item())
                results.append(
                    (ok_path, ViewLabel(view=VIEW_NAMES[argmax_idx], score=score)),
                )
        return results

    def close(self) -> None:
        """Release the model and free any cached tensors."""
        # Drop references; CPython will GC the torch modules. We don't try
        # to be clever about torch.cuda.empty_cache() — if the caller wants
        # to keep using torch after we close, they're free to do so.
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_embeds = None
        self._logit_scale = None

    def __enter__(self) -> ViewLabeler:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ----- internals --------------------------------------------------- #

    def _ensure_model(self) -> None:
        """Lazy-load OpenCLIP + torch the first time we label."""
        if self._model is not None and self._text_embeds is not None:
            return
        # Imported here so importing this module costs nothing.
        import open_clip
        import torch  # noqa: F401  (held via self._torch)

        self._torch = torch
        logger.info(
            "loading OpenCLIP %s / %s on %s",
            self._model_name,
            self._pretrained,
            self._device,
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            self._model_name,
            pretrained=self._pretrained,
            device=self._device,
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(self._model_name)
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._text_embeds = self._build_text_embeds()
        # OpenCLIP stores ``log(scale)`` as a learned parameter; reading it
        # at load time future-proofs against model swaps (vs. hardcoding the
        # ~100.0 converged value). Cache the scalar so we don't keep paying
        # the .exp().item() cost on every batch.
        self._logit_scale = float(model.logit_scale.exp().item())

    def _build_text_embeds(self) -> Any:
        """Encode the prompt ensemble; one row per view, L2-normalized.

        For each view: encode every prompt, L2-normalize per-prompt, mean
        across prompts, then re-normalize. This is the standard CLIP
        prompt-ensembling pattern and matches the OpenAI reference.
        """
        torch_mod = self._require_torch()
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:  # pragma: no cover - defensive
            raise RuntimeError("model must be loaded before building text embeds")
        per_view: list[Any] = []
        with torch_mod.no_grad():
            for view in VIEW_NAMES:
                prompts = _VIEW_PROMPTS[view]
                tokens = tokenizer(prompts).to(self._device)
                embeds = self._encode_text(tokens)
                embeds = self._l2_normalize(embeds)
                mean = embeds.mean(dim=0)
                mean = mean / mean.norm()
                per_view.append(mean)
        return torch_mod.stack(per_view, dim=0)

    def _encode_image(self, batch: Any) -> Any:
        model = self._model
        if model is None:  # pragma: no cover - defensive
            raise RuntimeError("model not loaded")
        return model.encode_image(batch)

    def _encode_text(self, tokens: Any) -> Any:
        model = self._model
        if model is None:  # pragma: no cover - defensive
            raise RuntimeError("model not loaded")
        return model.encode_text(tokens)

    def _l2_normalize(self, tensor: Any) -> Any:
        # tensor shape: (N, D); divide by per-row L2 norm.
        return tensor / tensor.norm(dim=-1, keepdim=True)

    def _load_and_preprocess(self, path: Path) -> Any:
        from PIL import Image as PILImageModule

        preprocess = self._preprocess
        if preprocess is None:  # pragma: no cover - defensive
            raise RuntimeError("preprocess transform not loaded")
        with PILImageModule.open(path) as img:
            converted = img.convert("RGB")
        return preprocess(converted)

    def _require_torch(self) -> Any:
        if self._torch is None:  # pragma: no cover - defensive
            raise RuntimeError("torch not loaded; call _ensure_model() first")
        return self._torch

    def _require_text_embeds(self) -> Any:
        if self._text_embeds is None:  # pragma: no cover - defensive
            raise RuntimeError("text embeddings not built; call _ensure_model() first")
        return self._text_embeds

    def _require_logit_scale(self) -> float:
        if self._logit_scale is None:  # pragma: no cover - defensive
            raise RuntimeError("logit_scale not cached; call _ensure_model() first")
        return self._logit_scale


def _chunked(items: list[Path], n: int) -> list[list[Path]]:
    """Slice ``items`` into chunks of at most ``n`` elements (preserving order)."""
    if n <= 0:
        raise ValueError(f"batch size must be > 0, got {n}")
    return [items[i : i + n] for i in range(0, len(items), n)]


__all__ = [
    "VIEW_NAMES",
    "ViewLabel",
    "ViewLabeler",
    "ViewName",
]
