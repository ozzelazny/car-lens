"""FastAPI ``recognize()`` service for Phase 6.1.

Endpoints
---------

* ``GET  /health``       -> ``{"status": "ok", "model": ..., "n_classes": ...}``.
* ``POST /api/recognize``-> multipart ``image`` upload, returns top-5
  predictions as JSON. Body is the dump of :class:`RecognizeResponse`.

When the ``UI_ROOT`` env var points at a directory containing
``index.html``, that directory is mounted at ``/`` (catch-all) so the
same uvicorn process serves both the API and the static UI. This is
the single-process local-dev mode; the production deployment serves
the UI via a separate nginx container and leaves ``UI_ROOT`` unset.

Startup
-------

1. Load OpenCLIP MobileCLIP-S2 with the pre-trained ``datacompdr`` tag.
2. If ``MODEL_PATH`` points at a Phase 5.2 fine-tune checkpoint, overlay
   ``image_encoder_state_dict`` onto ``model.visual``.
3. Load the prototype tensor + class metadata from ``PROTOTYPES_PATH``
   (produced by the ``build-prototypes`` CLI). Move both model and
   prototypes to ``DEVICE`` (env var, default ``cpu``).

The model + prototypes are constructed exactly once during the FastAPI
lifespan startup; per-request handlers receive them through a dependency
injector so tests can swap in deterministic stand-ins. The lifespan
handler is used in preference to the deprecated ``on_event("startup")``
hook.

Validation
----------

* Upload content-type must start with ``image/`` (else HTTP 415).
* Upload size must be <= :data:`MAX_UPLOAD_BYTES` (else HTTP 413).
* Upload must successfully decode as a PIL image (else HTTP 400).
* Missing ``image`` field is a FastAPI 422 by default.
"""

from __future__ import annotations

import io
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- config


#: Default MobileCLIP-S2 / OpenCLIP descriptor; matched to the Phase 5.2
#: fine-tune harness so the architecture lines up before the
#: image_encoder_state_dict is overlaid.
DEFAULT_MODEL_NAME = "MobileCLIP-S2"
DEFAULT_PRETRAINED = "datacompdr"

#: Hard ceiling on the size of an uploaded image. 20 MB is generous for
#: phone-captured photos (typical 6-9 MB) and keeps the API from being
#: weaponized as a memory-pressure vector.
MAX_UPLOAD_BYTES: int = 20 * 1024 * 1024

#: How many predictions to return per request. The frontend renders 5.
DEFAULT_TOP_K: int = 5

#: Default reject threshold used by both view-classifier modes.
#:
#: In **binary mode** (2-class ``exterior`` / ``non-exterior`` head) the
#: semantics are: reject when the head predicts ``non-exterior`` AND its
#: softmax probability is >= this threshold. So 0.5 means "if the binary
#: classifier is more than 50% confident the image is non-exterior, drop
#: the request".
#:
#: In **6-way mode** (legacy view-conditional flow) the semantics flip:
#: reject when the top view's softmax probability is BELOW this
#: threshold OR the top view is ``non-exterior``. The 6-way default is
#: kept at 0.5 for parity with the original Phase 6.1 implementation
#: even though it is too aggressive for the noisy 6-way head -- a known
#: limitation we are stepping away from in favour of binary mode.
DEFAULT_VIEW_REJECT_THRESHOLD: float = 0.5

#: The five Phase 3.3 exterior view labels (indices 0..4 of the Phase 5.3
#: classifier head). Index 5 is the catch-all ``"non-exterior"`` class.
EXTERIOR_VIEWS: tuple[str, ...] = (
    "front",
    "rear",
    "side",
    "three-quarter-front",
    "three-quarter-rear",
)
NON_EXTERIOR_VIEW: str = "non-exterior"

#: Mode literal exposed via ``/health.view_classifier_mode``.
VIEW_CLASSIFIER_MODE_BINARY = "binary"
VIEW_CLASSIFIER_MODE_VIEW5 = "view5"


#: Default Anthropic model used for re-rank when no override is supplied
#: via the ``LLM_RERANK_MODEL`` env var. Sonnet 4.6 is a balanced
#: cost/latency choice for trim-year disambiguation.
DEFAULT_LLM_RERANK_MODEL = "claude-sonnet-4-6"

#: Below-this on-device top-1 confidence triggers an LLM re-rank.
#: Phase 5.4 shows top-5 contains the right answer ~98% of the time; a
#: 0.5 threshold conservatively re-ranks when the model isn't more
#: confident than a coin flip on its own top pick.
DEFAULT_LLM_RERANK_THRESHOLD: float = 0.5

#: Default per-call HTTP timeout (seconds) for the Anthropic API.
DEFAULT_LLM_RERANK_TIMEOUT_SECONDS: float = 10.0


@dataclass(frozen=True)
class ServiceSettings:
    """Resolved environment configuration for the recognize service."""

    model_path: Path | None
    prototypes_path: Path
    device: str
    model_name: str
    pretrained: str
    top_k: int
    ui_root: Path | None
    view_classifier_path: Path | None
    view_reject_threshold: float
    llm_rerank_enabled: bool = False
    llm_rerank_threshold: float = DEFAULT_LLM_RERANK_THRESHOLD
    llm_rerank_model: str = DEFAULT_LLM_RERANK_MODEL
    llm_rerank_timeout_seconds: float = DEFAULT_LLM_RERANK_TIMEOUT_SECONDS
    anthropic_api_key: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ServiceSettings:
        e = env if env is not None else dict(os.environ)
        model_env = e.get("MODEL_PATH", "").strip()
        model_path = Path(model_env) if model_env else None
        prototypes_path = Path(
            e.get("PROTOTYPES_PATH", "/app/cache/prototypes.pt").strip()
            or "/app/cache/prototypes.pt"
        )
        ui_env = e.get("UI_ROOT", "").strip()
        ui_root = Path(ui_env) if ui_env else None
        view_env = e.get("VIEW_CLASSIFIER_PATH", "").strip()
        view_classifier_path = Path(view_env) if view_env else None
        view_reject_threshold_raw = e.get(
            "VIEW_REJECT_THRESHOLD", str(DEFAULT_VIEW_REJECT_THRESHOLD)
        )
        try:
            view_reject_threshold = float(view_reject_threshold_raw)
        except ValueError as exc:
            raise RuntimeError(
                f"VIEW_REJECT_THRESHOLD must be a float, got {view_reject_threshold_raw!r}"
            ) from exc
        llm_rerank_enabled = _parse_bool_env(e.get("LLM_RERANK_ENABLED", ""))
        llm_rerank_threshold_raw = e.get("LLM_RERANK_THRESHOLD", str(DEFAULT_LLM_RERANK_THRESHOLD))
        try:
            llm_rerank_threshold = float(llm_rerank_threshold_raw)
        except ValueError as exc:
            raise RuntimeError(
                f"LLM_RERANK_THRESHOLD must be a float, got {llm_rerank_threshold_raw!r}"
            ) from exc
        llm_rerank_timeout_raw = e.get(
            "LLM_RERANK_TIMEOUT", str(DEFAULT_LLM_RERANK_TIMEOUT_SECONDS)
        )
        try:
            llm_rerank_timeout_seconds = float(llm_rerank_timeout_raw)
        except ValueError as exc:
            raise RuntimeError(
                f"LLM_RERANK_TIMEOUT must be a float, got {llm_rerank_timeout_raw!r}"
            ) from exc
        llm_rerank_model = e.get("LLM_RERANK_MODEL", DEFAULT_LLM_RERANK_MODEL).strip() or (
            DEFAULT_LLM_RERANK_MODEL
        )
        anthropic_api_key_raw = e.get("ANTHROPIC_API_KEY", "").strip()
        anthropic_api_key = anthropic_api_key_raw or None
        return cls(
            model_path=model_path,
            prototypes_path=prototypes_path,
            device=e.get("DEVICE", "cpu").strip() or "cpu",
            model_name=e.get("MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME,
            pretrained=e.get("PRETRAINED", DEFAULT_PRETRAINED).strip() or DEFAULT_PRETRAINED,
            top_k=int(e.get("TOP_K", str(DEFAULT_TOP_K))),
            ui_root=ui_root,
            view_classifier_path=view_classifier_path,
            view_reject_threshold=view_reject_threshold,
            llm_rerank_enabled=llm_rerank_enabled,
            llm_rerank_threshold=llm_rerank_threshold,
            llm_rerank_model=llm_rerank_model,
            llm_rerank_timeout_seconds=llm_rerank_timeout_seconds,
            anthropic_api_key=anthropic_api_key,
        )


def _parse_bool_env(raw: str) -> bool:
    """Parse a permissive truthy env var.

    Accepts ``"1"``, ``"true"``, ``"yes"``, ``"on"`` (case-insensitive)
    as true; everything else (including the empty string) as false.
    """
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_sibling_module(name: str) -> Any:
    """Load a sibling .py file via :mod:`importlib`.

    The recognize_api service is launched two ways: locally as
    ``services.recognize_api.app:app`` (package-style) and inside the
    Docker container as the bare ``app:app`` (no parent package). A
    plain ``from services.recognize_api.<name> import ...`` only works
    in the first mode; a relative ``from .<name> import ...`` only
    works in the first mode too because there is no ``__init__.py``
    in ``services/recognize_api``. importlib-by-path works in both.
    """
    import importlib.util  # noqa: PLC0415
    import sys  # noqa: PLC0415

    cache_key = f"recognize_api_{name}"
    cached = sys.modules.get(cache_key)
    if cached is not None:
        return cached
    here = Path(__file__).resolve().parent
    module_path = here / f"{name}.py"
    spec = importlib.util.spec_from_file_location(cache_key, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not locate sibling module {name!r} at {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec_module so dataclasses + pydantic can resolve
    # ``cls.__module__`` during their own module-level work.
    sys.modules[cache_key] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------- types


class Prediction(BaseModel):
    """Single (class, confidence) tuple in a recognize response."""

    model_config = ConfigDict(extra="forbid")

    class_id: str
    display: str
    confidence: float


class RecognizeResponse(BaseModel):
    """Body of a successful ``POST /api/recognize``."""

    model_config = ConfigDict(extra="forbid")

    predictions: list[Prediction] = Field(default_factory=list)
    elapsed_ms: float
    view: str | None = None
    """Predicted view name from the Phase 5.3 view classifier, or
    ``None`` when the view classifier isn't loaded (single-prototype
    backwards-compat path)."""

    view_score: float | None = None
    """Softmax probability the view classifier assigned to the predicted
    view, or ``None`` in the backwards-compat path."""

    rerank_applied: bool = False
    """``True`` when the LLM re-rank fallback successfully reordered
    the candidate list; ``False`` otherwise (re-rank disabled,
    high-confidence top-1, or LLM call failed)."""

    rerank_latency_ms: float | None = None
    """Wall-clock latency of the re-rank call (including the image
    upload + LLM round-trip) in milliseconds, or ``None`` when re-rank
    was not invoked for this request."""


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    model_config = ConfigDict(extra="forbid")

    status: str
    model: str
    n_classes: int
    device: str
    view_classifier_loaded: bool = False
    views_with_prototypes: list[str] = Field(default_factory=list)
    view_classifier_mode: str | None = None
    """Either ``"binary"`` (2-class exterior/non-exterior rejection
    gate), ``"view5"`` (legacy 6-way view-conditional retrieval), or
    ``None`` when no view classifier is loaded."""

    llm_rerank_enabled: bool = False
    """``True`` when an :class:`LLMReRanker` is wired up and ready to
    serve low-confidence top-1 predictions; ``False`` otherwise."""

    llm_rerank_model: str | None = None
    """The Anthropic model name configured for re-rank, or ``None``
    when re-rank is disabled."""


# --------------------------------------------------------------- service state


class RecognizerState:
    """Lifespan-scoped bundle: loaded encoder + prototypes + class labels.

    Held on ``app.state.recognizer`` after lifespan startup. The
    dependency :func:`get_recognizer` reaches into ``request.app.state``
    so each handler picks up the same instance.

    The state is intentionally **not** a Pydantic model -- it owns
    heavyweight torch tensors and the encoder model, which we don't want
    Pydantic to copy / validate.
    """

    def __init__(
        self,
        *,
        settings: ServiceSettings,
        model: Any,
        preprocess: Any,
        prototypes: Any | None,
        class_ids: list[str],
        display_names: list[str],
        torch_mod: Any,
        model_label: str,
        logit_scale: float = 100.0,
        prototypes_by_view: dict[str, Any] | None = None,
        view_head: Any | None = None,
        view_class_names: list[str] | None = None,
        view_reject_threshold: float = DEFAULT_VIEW_REJECT_THRESHOLD,
        view_classifier_mode: str | None = None,
        llm_reranker: Any | None = None,
    ) -> None:
        self.settings = settings
        self.model = model
        self.preprocess = preprocess
        self.prototypes = prototypes
        self.class_ids = class_ids
        self.display_names = display_names
        self.torch = torch_mod
        self.model_label = model_label
        # View-conditional state (Phase 6.1). When ``view_head`` is None
        # the service falls back to the legacy single-prototype path.
        self.prototypes_by_view = prototypes_by_view
        self.view_head = view_head
        self.view_class_names = view_class_names
        self.view_reject_threshold = float(view_reject_threshold)
        # Optional Phase 6.2 LLM re-rank fallback. None when disabled.
        self.llm_reranker = llm_reranker
        # Auto-derive the mode from the class-name count if not given
        # explicitly. Test fixtures that hand-build a RecognizerState
        # may still pass an explicit value.
        if view_classifier_mode is None and view_class_names is not None:
            n = len(view_class_names)
            if n == 2:
                view_classifier_mode = VIEW_CLASSIFIER_MODE_BINARY
            elif n == 6:
                view_classifier_mode = VIEW_CLASSIFIER_MODE_VIEW5
        self.view_classifier_mode = view_classifier_mode
        # Default CLIP-style softmax temperature. Real OpenCLIP models
        # expose ``model.logit_scale`` (a learned log-temperature); we
        # prefer reading that when available so the softmax matches
        # training. Falls back to the standard 100.0 = exp(log(100)).
        learned = getattr(model, "logit_scale", None)
        if learned is not None:
            try:
                self.logit_scale = float(learned.exp().item())
            except Exception:  # noqa: BLE001 -- be permissive with stubs
                self.logit_scale = float(logit_scale)
        else:
            self.logit_scale = float(logit_scale)

    @property
    def n_classes(self) -> int:
        return len(self.class_ids)


def _load_prototype_payload(path: Path, torch_mod: Any) -> dict[str, Any]:
    """Load and validate a ``build-prototypes`` payload.

    Supports both schema versions:

    * **v1** (legacy / default): dict with ``class_ids``,
      ``display_names``, ``prototypes`` (a single ``(n_classes, embed_dim)``
      tensor). No ``schema_version`` key.
    * **v2** (Phase 6.1 per-view): dict with ``schema_version == 2``,
      ``class_ids``, ``display_names``, ``prototypes_by_view`` (a dict
      mapping each exterior view name to a ``(n_classes, embed_dim)``
      tensor), and ``view_names`` listing the 5 exterior view keys.
    """
    if not path.exists():
        raise RuntimeError(
            f"prototype file not found at {path}; run the build-prototypes CLI "
            "first, e.g. `build-prototypes --source compcars --checkpoint "
            "models/checkpoints/mobileclip_s2_compcars_epoch09_top1_91.9.pt`"
        )
    payload = torch_mod.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"prototype file at {path} is malformed; expected a dict, got {type(payload).__name__}"
        )
    schema_version = int(payload.get("schema_version", 1))
    if schema_version == 2:
        required_v2 = ("class_ids", "display_names", "prototypes_by_view", "view_names")
        if any(k not in payload for k in required_v2):
            raise RuntimeError(
                f"prototype file at {path} (schema v2) is malformed; expected keys "
                f"{required_v2}, got {sorted(payload.keys())}"
            )
        prototypes_by_view = payload["prototypes_by_view"]
        if not isinstance(prototypes_by_view, dict):
            raise RuntimeError(
                f"prototype file at {path} has invalid 'prototypes_by_view' "
                f"(expected dict, got {type(prototypes_by_view).__name__})"
            )
        n_classes = len(payload["class_ids"])
        for view, tensor in prototypes_by_view.items():
            if int(tensor.shape[0]) != n_classes:
                raise RuntimeError(
                    f"prototype file at {path} has inconsistent shape for view "
                    f"{view!r}: {int(tensor.shape[0])} rows vs {n_classes} class_ids"
                )
        return dict(payload)
    # v1 path.
    required = ("class_ids", "display_names", "prototypes")
    if any(k not in payload for k in required):
        raise RuntimeError(
            f"prototype file at {path} is malformed; expected a dict with "
            f"keys {required}, got {sorted(payload.keys())}"
        )
    if len(payload["class_ids"]) != int(payload["prototypes"].shape[0]):
        raise RuntimeError(
            f"prototype file at {path} has inconsistent shape: "
            f"{len(payload['class_ids'])} class_ids vs "
            f"{int(payload['prototypes'].shape[0])} prototype rows"
        )
    return dict(payload)


def _load_view_classifier(
    path: Path,
    *,
    embed_dim: int,
    torch_mod: Any,
    device: str,
) -> tuple[Any, list[str]]:
    """Load the Phase 5.3 view classifier head from ``path``.

    The on-disk payload is the
    :class:`car_lense_engine.training.view_classifier.CheckpointPayload`
    serialized via ``torch.save``. We pull the head's state dict + the
    canonical class names, rebuild the head as
    ``nn.Linear(embed_dim, len(class_names))``, and load the weights.
    The class count is read from ``class_names`` so binary (2) and 6-way
    heads are both supported with the same loader. Raises
    :class:`RuntimeError` with a clear diagnostic on any failure -- no
    silent fallback (a service that silently dropped view conditioning
    would be very confusing to debug).
    """
    if not path.exists():
        raise RuntimeError(
            f"VIEW_CLASSIFIER_PATH does not exist: {path}; "
            "unset VIEW_CLASSIFIER_PATH to fall back to single-prototype retrieval"
        )
    payload = torch_mod.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"view classifier checkpoint at {path} is malformed; expected a dict, "
            f"got {type(payload).__name__}"
        )
    required = ("head_state_dict", "class_names")
    if any(k not in payload for k in required):
        raise RuntimeError(
            f"view classifier checkpoint at {path} is missing required keys "
            f"{required}; got {sorted(payload.keys())}"
        )
    class_names = list(payload["class_names"])
    if not class_names:
        raise RuntimeError(f"view classifier checkpoint at {path} has an empty class_names list")
    head_state = payload["head_state_dict"]
    import torch.nn as nn  # noqa: PLC0415

    head: Any = nn.Linear(embed_dim, len(class_names))
    try:
        head.load_state_dict(head_state)
    except Exception as exc:  # noqa: BLE001 -- re-raise with context
        raise RuntimeError(f"failed to load view classifier head from {path}: {exc!r}") from exc
    head.to(device)
    head.eval()
    return head, class_names


def _load_openclip_model(settings: ServiceSettings, torch_mod: Any) -> tuple[Any, Any, str]:
    """Construct the OpenCLIP backbone and overlay the fine-tune checkpoint.

    Returns ``(model, preprocess, label)`` where ``label`` is a short
    string suitable for ``/health`` (e.g.
    ``"mobileclip-s2-compcars-epoch09"`` derived from the checkpoint
    filename, or just the pretrained tag when no checkpoint is set).
    """
    try:
        import open_clip  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover -- deps are in pyproject
        raise RuntimeError("open_clip_torch must be installed in the recognize-api image") from exc

    logger.info(
        "recognize-api: loading OpenCLIP %s / %s on %s",
        settings.model_name,
        settings.pretrained,
        settings.device,
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        settings.model_name,
        pretrained=settings.pretrained,
        device=settings.device,
    )
    model_label = settings.pretrained
    if settings.model_path is not None:
        if not settings.model_path.exists():
            raise RuntimeError(f"MODEL_PATH does not exist: {settings.model_path}")
        payload = torch_mod.load(
            settings.model_path,
            map_location=settings.device,
            weights_only=False,
        )
        if not isinstance(payload, dict) or "image_encoder_state_dict" not in payload:
            raise RuntimeError(
                f"checkpoint at {settings.model_path} is not a Phase 5.2 "
                "training checkpoint (missing 'image_encoder_state_dict')"
            )
        state = payload["image_encoder_state_dict"]
        visual = getattr(model, "visual", None)
        if visual is not None and hasattr(visual, "load_state_dict"):
            visual.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=False)
        model_label = settings.model_path.stem
        logger.info(
            "recognize-api: loaded fine-tuned weights from %s (epoch=%s val_top1=%s)",
            settings.model_path,
            payload.get("epoch"),
            payload.get("val_top1"),
        )
    model.eval()
    return model, preprocess, model_label


def build_recognizer(settings: ServiceSettings) -> RecognizerState:
    """Construct a fully-loaded :class:`RecognizerState` from ``settings``.

    Separated from the FastAPI lifespan so tests can call it directly
    against a synthesized prototype file (or, more often, monkey-patch
    around it by injecting their own :class:`RecognizerState`).

    Two retrieval paths are supported:

    * **Legacy single-prototype (schema v1)**: ``VIEW_CLASSIFIER_PATH``
      unset, ``PROTOTYPES_PATH`` points at a v1 payload. The service
      runs the original cosine-vs-all-prototypes flow.
    * **View-conditional (schema v2)**: ``VIEW_CLASSIFIER_PATH`` set,
      ``PROTOTYPES_PATH`` points at a v2 payload with one prototype
      per ``(class, view)`` for the 5 exterior views. The view
      classifier predicts the query view first; non-exterior or
      low-confidence views are rejected; otherwise retrieval runs
      against only the matching view's prototypes.

    The two go together: setting ``VIEW_CLASSIFIER_PATH`` without a
    v2 prototype file (or vice versa with v1) is a configuration
    error that raises here.
    """
    import torch  # noqa: PLC0415

    model, preprocess, model_label = _load_openclip_model(settings, torch)
    payload = _load_prototype_payload(settings.prototypes_path, torch)
    schema_version = int(payload.get("schema_version", 1))

    prototypes: Any | None = None
    prototypes_by_view: dict[str, Any] | None = None
    embed_dim: int | None = None
    if schema_version == 2:
        prototypes_by_view = {
            view: tensor.to(settings.device)
            for view, tensor in payload["prototypes_by_view"].items()
        }
        # Pick embed dim from the first view tensor (all views share
        # the same trailing axis by construction).
        for tensor in prototypes_by_view.values():
            if int(tensor.shape[0]) > 0:
                embed_dim = int(tensor.shape[1])
                break
        if embed_dim is None:
            raise RuntimeError(
                f"prototype file at {settings.prototypes_path} (v2) has no populated "
                "view tensors -- cannot infer embed_dim"
            )
    else:
        prototypes = payload["prototypes"].to(settings.device)
        embed_dim = int(prototypes.shape[1])

    view_head: Any | None = None
    view_class_names: list[str] | None = None
    view_classifier_mode: str | None = None
    if settings.view_classifier_path is not None:
        view_head, view_class_names = _load_view_classifier(
            settings.view_classifier_path,
            embed_dim=embed_dim,
            torch_mod=torch,
            device=settings.device,
        )
        n_view_classes = len(view_class_names)
        # Auto-detect mode from class count and validate prototype schema.
        if n_view_classes == 2:
            view_classifier_mode = VIEW_CLASSIFIER_MODE_BINARY
            if prototypes_by_view is not None:
                raise RuntimeError(
                    "VIEW_CLASSIFIER_PATH points at a binary (2-class) head but the "
                    f"prototype file at {settings.prototypes_path} is schema v2 "
                    "(per-view). Binary mode is a rejection gate over single-prototype "
                    "retrieval -- re-run `build-prototypes` without `--per-view` and "
                    "point PROTOTYPES_PATH at the v1 cache, or train a 6-way view "
                    "classifier to match the v2 prototypes."
                )
        elif n_view_classes == 6:
            view_classifier_mode = VIEW_CLASSIFIER_MODE_VIEW5
            if prototypes_by_view is None:
                raise RuntimeError(
                    "VIEW_CLASSIFIER_PATH points at a 6-way head but the prototype "
                    f"file at {settings.prototypes_path} is schema v1 "
                    "(single-prototype). Re-run `build-prototypes --per-view` and "
                    "point PROTOTYPES_PATH at the new file, or unset "
                    "VIEW_CLASSIFIER_PATH to use the single-prototype fallback."
                )
        else:
            raise RuntimeError(
                f"view classifier at {settings.view_classifier_path} has "
                f"{n_view_classes} classes; expected 2 (binary) or 6 (view5)"
            )

    llm_reranker: Any | None = None
    if settings.llm_rerank_enabled:
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "LLM_RERANK_ENABLED=1 but ANTHROPIC_API_KEY is unset; "
                "set ANTHROPIC_API_KEY or unset LLM_RERANK_ENABLED to disable re-rank"
            )
        # The reranker module lives next to this one. Resolving via
        # importlib avoids the module-name ambiguity (services.recognize_api.app
        # vs the bare ``app:app`` Docker entrypoint) that breaks
        # plain ``from services.recognize_api.llm_rerank import`` for
        # both interpreters.
        llm_reranker_module = _load_sibling_module("llm_rerank")
        llm_reranker = llm_reranker_module.LLMReRanker(
            api_key=settings.anthropic_api_key,
            model=settings.llm_rerank_model,
            timeout_seconds=settings.llm_rerank_timeout_seconds,
            max_image_dim=768,
        )
        logger.info(
            "recognize-api: LLM re-rank enabled (model=%s threshold=%.2f timeout=%.1fs)",
            settings.llm_rerank_model,
            settings.llm_rerank_threshold,
            settings.llm_rerank_timeout_seconds,
        )

    return RecognizerState(
        settings=settings,
        model=model,
        preprocess=preprocess,
        prototypes=prototypes,
        class_ids=list(payload["class_ids"]),
        display_names=list(payload["display_names"]),
        torch_mod=torch,
        model_label=model_label,
        prototypes_by_view=prototypes_by_view,
        view_head=view_head,
        view_class_names=view_class_names,
        view_reject_threshold=settings.view_reject_threshold,
        view_classifier_mode=view_classifier_mode,
        llm_reranker=llm_reranker,
    )


# --------------------------------------------------------------- FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the recognizer on startup, drop the reference on shutdown.

    We don't want a partially-initialized service to start serving 5xxs:
    any error during model/prototype load is re-raised, which aborts the
    container with a clear traceback in the logs.
    """
    settings = ServiceSettings.from_env()
    logger.info(
        "recognize-api: starting on device=%s, prototypes=%s, model_path=%s",
        settings.device,
        settings.prototypes_path,
        settings.model_path,
    )
    app.state.recognizer = build_recognizer(settings)
    try:
        yield
    finally:
        app.state.recognizer = None


def create_app(
    *,
    use_lifespan: bool = True,
    settings: ServiceSettings | None = None,
) -> FastAPI:
    """Construct the FastAPI app.

    ``use_lifespan`` defaults to ``True`` so the production entry point
    (``app = create_app()`` at module level) gets the model + prototype
    load on startup. Tests pass ``use_lifespan=False`` to bypass the
    real load and inject a stub :class:`RecognizerState` onto
    ``app.state.recognizer`` directly -- the model file + prototypes
    don't exist in the test environment, and emulating them on disk
    would defeat the point of the in-memory stub.

    ``settings`` is read from the environment by default; tests can
    pass a synthesized :class:`ServiceSettings` to drive the optional
    ``ui_root`` static mount without touching ``os.environ``.
    """
    if settings is None:
        settings = ServiceSettings.from_env()

    app = FastAPI(
        title="Car Lense recognize() API",
        version="0.1.0",
        lifespan=lifespan if use_lifespan else None,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health(
        recognizer: RecognizerState = Depends(get_recognizer),  # noqa: B008
    ) -> HealthResponse:
        views_with_prototypes: list[str] = []
        if recognizer.prototypes_by_view is not None:
            views_with_prototypes = list(recognizer.prototypes_by_view.keys())
        llm_rerank_enabled = recognizer.llm_reranker is not None
        llm_rerank_model: str | None = None
        if llm_rerank_enabled:
            llm_rerank_model = getattr(recognizer.llm_reranker, "model", None)
        return HealthResponse(
            status="ok",
            model=recognizer.model_label,
            n_classes=recognizer.n_classes,
            device=recognizer.settings.device,
            view_classifier_loaded=recognizer.view_head is not None,
            views_with_prototypes=views_with_prototypes,
            view_classifier_mode=recognizer.view_classifier_mode,
            llm_rerank_enabled=llm_rerank_enabled,
            llm_rerank_model=llm_rerank_model,
        )

    @app.post(
        "/api/recognize",
        response_model=RecognizeResponse,
    )
    async def recognize_endpoint(
        image: UploadFile = File(...),  # noqa: B008
        recognizer: RecognizerState = Depends(get_recognizer),  # noqa: B008
    ) -> RecognizeResponse:
        return await recognize_image(image=image, recognizer=recognizer)

    # Optional single-process mode: mount the static UI under ``/`` so
    # the relative ``/api/recognize`` calls in ``app.js`` hit the same
    # origin. This must come AFTER ``/api/...`` and ``/health`` so
    # FastAPI matches those routes before falling through to the
    # catch-all static mount.
    if settings.ui_root is not None and settings.ui_root.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(settings.ui_root), html=True),
            name="ui",
        )

    return app


def get_recognizer(request: Request) -> RecognizerState:
    """FastAPI dependency that pulls the lifespan-scoped recognizer.

    FastAPI injects the active ``Request`` because of the parameter's
    type annotation; we read ``request.app.state.recognizer`` so each
    handler shares the same instance built during lifespan startup
    (or, in tests, installed manually onto ``app.state``).

    Raises a 503 if the recognizer hasn't been initialized -- this
    happens if the route is hit before lifespan startup completes
    (rare in practice, but the test suite explicitly covers it).
    """
    recognizer = getattr(request.app.state, "recognizer", None)
    if recognizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="recognizer not initialized",
        )
    return recognizer  # type: ignore[no-any-return]


async def recognize_image(
    *,
    image: UploadFile,
    recognizer: RecognizerState,
) -> RecognizeResponse:
    """Core inference path -- factored out so tests can call it directly.

    Validation order matches the HTTP spec layering:

    1. Content-Type must look like ``image/*`` (415 if not).
    2. Body must fit under :data:`MAX_UPLOAD_BYTES` (413 if not).
    3. Body must decode to a PIL image (400 if not).
    """
    content_type = (image.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"unsupported content-type: {image.content_type!r}; expected image/*",
        )
    raw = await image.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"image exceeds {MAX_UPLOAD_BYTES} bytes",
        )
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="empty image upload",
        )
    try:
        from PIL import Image as PILImage  # noqa: PLC0415
        from PIL import UnidentifiedImageError  # noqa: PLC0415

        with PILImage.open(io.BytesIO(raw)) as pil:
            rgb = pil.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"could not decode image: {exc}",
        ) from exc

    start = time.perf_counter()
    predictions, view_name, view_score = _predict(rgb, recognizer)
    rerank_applied = False
    rerank_latency_ms: float | None = None
    if _should_rerank(predictions, recognizer):
        predictions, rerank_applied, rerank_latency_ms = _apply_llm_rerank(
            predictions=predictions,
            image=rgb,
            recognizer=recognizer,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return RecognizeResponse(
        predictions=predictions,
        elapsed_ms=elapsed_ms,
        view=view_name,
        view_score=view_score,
        rerank_applied=rerank_applied,
        rerank_latency_ms=rerank_latency_ms,
    )


def _should_rerank(
    predictions: list[Prediction],
    recognizer: RecognizerState,
) -> bool:
    """Return True iff the LLM re-rank fallback should be invoked."""
    if recognizer.llm_reranker is None:
        return False
    if not predictions:
        return False
    threshold = float(recognizer.settings.llm_rerank_threshold)
    return predictions[0].confidence < threshold


def _apply_llm_rerank(
    *,
    predictions: list[Prediction],
    image: Any,
    recognizer: RecognizerState,
) -> tuple[list[Prediction], bool, float | None]:
    """Run the LLM re-rank pass.

    Returns ``(predictions, rerank_applied, rerank_latency_ms)``. When
    the LLM fails (or chooses an out-of-range index) the original
    ``predictions`` list is returned unchanged with
    ``rerank_applied=False``; the latency is still reported so the
    caller can observe how much overhead the failed call cost.
    """
    reranker = recognizer.llm_reranker
    assert reranker is not None  # noqa: S101 -- invariant; _should_rerank gates this
    candidates = [p.display for p in predictions]
    try:
        result = reranker.rerank(image=image, candidates=candidates)
    except Exception as exc:  # noqa: BLE001 -- defence: rerank() should never raise
        logger.warning("llm_rerank: unexpected exception escaped reranker: %s", exc)
        return predictions, False, None

    latency_ms = float(result.latency_ms)
    chosen = int(result.chosen_index)
    if chosen < 0 or chosen >= len(predictions):
        if result.error is not None:
            logger.info("llm_rerank: skipping reorder, error=%s", result.error)
        return predictions, False, latency_ms

    if chosen == 0:
        # LLM agreed with the original top-1. Still report it as
        # "applied" so callers can distinguish a confirmed call from a
        # skipped one; bump the confidence to reflect endorsement.
        boosted = _bump_top1_confidence(predictions, recognizer.settings.llm_rerank_threshold)
        return boosted, True, latency_ms

    reordered: list[Prediction] = [predictions[chosen]]
    for i, p in enumerate(predictions):
        if i == chosen:
            continue
        reordered.append(p)
    reordered = _bump_top1_confidence(reordered, recognizer.settings.llm_rerank_threshold)
    return reordered, True, latency_ms


def _bump_top1_confidence(
    predictions: list[Prediction],
    threshold: float,
) -> list[Prediction]:
    """Ensure the top-1 confidence is at least ``threshold``.

    The downstream UI displays a "low confidence" badge when the
    confidence dips below the rerank threshold; once the LLM has
    endorsed an option the badge is misleading, so we lift the
    reported confidence to at least the threshold.
    """
    if not predictions:
        return predictions
    top = predictions[0]
    bumped_conf = max(top.confidence, float(threshold))
    if bumped_conf == top.confidence:
        return predictions
    new_top = Prediction(class_id=top.class_id, display=top.display, confidence=bumped_conf)
    return [new_top, *predictions[1:]]


def _predict(
    pil_image: Any, recognizer: RecognizerState
) -> tuple[list[Prediction], str | None, float | None]:
    """Run encode -> (view-detect) -> cosine sim -> softmax -> top-K.

    The math mirrors :func:`car_lense_engine.eval.baseline.evaluate` for
    the legacy single-prototype path and supports two view-classifier
    modes:

    1. **Binary mode** (2-class ``exterior`` / ``non-exterior`` head):
       run the head on the L2-normalized features; if the top class is
       ``non-exterior`` AND its softmax probability >=
       ``view_reject_threshold``, reject the request via HTTP 422.
       Otherwise fall through to single-prototype retrieval (v1 cache).
       NOTE: the threshold semantics flip relative to 6-way mode -- here
       we reject when CONFIDENT it's non-exterior; uncertain non-exterior
       predictions still pass through so the user sees a top-5 they can
       review.
    2. **6-way / view5 mode** (legacy Phase 6.1): the head predicts one
       of 6 views (5 exterior + non-exterior); reject if the top view is
       ``non-exterior`` OR its softmax probability is BELOW
       ``view_reject_threshold``; otherwise retrieve against only the
       matching view's per-view prototypes (v2 cache).
    3. **No view classifier**: score against the single-prototype tensor.

    Returns ``(predictions, view_name, view_score)`` where the latter
    two are ``None`` when no view classifier is loaded so the response
    serializer can omit them cleanly.
    """
    torch_mod = recognizer.torch
    tensor = recognizer.preprocess(pil_image)
    if hasattr(tensor, "unsqueeze"):
        batch = tensor.unsqueeze(0).to(recognizer.settings.device)
    else:  # pragma: no cover -- stubs return arbitrary shapes in tests
        batch = tensor

    with torch_mod.no_grad():
        features = recognizer.model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)

        if recognizer.view_head is not None:
            view_logits = recognizer.view_head(features)
            view_probs = view_logits.softmax(dim=-1)[0]
            predicted_view_idx = int(view_probs.argmax().item())
            view_score = float(view_probs[predicted_view_idx].item())
            assert recognizer.view_class_names is not None  # noqa: S101 -- invariant
            predicted_view_name = recognizer.view_class_names[predicted_view_idx]

            if recognizer.view_classifier_mode == VIEW_CLASSIFIER_MODE_BINARY:
                # Binary mode: reject only when we are CONFIDENT the
                # image is non-exterior. Uncertain predictions pass
                # through to retrieval so the user sees a top-5 list.
                if (
                    predicted_view_name == NON_EXTERIOR_VIEW
                    and view_score >= recognizer.view_reject_threshold
                ):
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail={
                            "detail": "non-exterior view rejected",
                            "view": predicted_view_name,
                            "view_score": view_score,
                        },
                    )
                # Fall through to single-prototype retrieval.
                if recognizer.prototypes is None:  # pragma: no cover -- defensive
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="no prototypes loaded",
                    )
                sims = features @ recognizer.prototypes.T
                scaled = sims * recognizer.logit_scale
                probs = scaled.softmax(dim=-1)[0]
                top_predictions = _topk_predictions(probs, recognizer)
                return top_predictions, predicted_view_name, view_score

            # 6-way / view5 mode: reject non-exterior OR low-confidence
            # top view; otherwise retrieve against the predicted view's
            # per-view prototypes.
            if (
                predicted_view_name == NON_EXTERIOR_VIEW
                or view_score < recognizer.view_reject_threshold
            ):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={
                        "detail": "non-exterior view rejected",
                        "view": predicted_view_name,
                        "view_score": view_score,
                    },
                )
            assert recognizer.prototypes_by_view is not None  # noqa: S101 -- invariant
            proto_tensor = recognizer.prototypes_by_view[predicted_view_name]
            sims = features @ proto_tensor.T
            scaled = sims * recognizer.logit_scale
            probs = scaled.softmax(dim=-1)[0]
            top_predictions = _topk_predictions(probs, recognizer)
            return top_predictions, predicted_view_name, view_score

        # Legacy single-prototype path (no view classifier loaded).
        if recognizer.prototypes is None:  # pragma: no cover -- defensive
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="no prototypes loaded",
            )
        sims = features @ recognizer.prototypes.T
        scaled = sims * recognizer.logit_scale
        probs = scaled.softmax(dim=-1)[0]
    top_predictions = _topk_predictions(probs, recognizer)
    return top_predictions, None, None


def _topk_predictions(probs: Any, recognizer: RecognizerState) -> list[Prediction]:
    """Render the top-K predictions from a (n_classes,) probability vector."""
    k = min(recognizer.settings.top_k, recognizer.n_classes)
    top_values, top_idx = probs.topk(k=k, dim=-1)
    out: list[Prediction] = []
    for value, idx_t in zip(top_values.tolist(), top_idx.tolist(), strict=True):
        i = int(idx_t)
        out.append(
            Prediction(
                class_id=recognizer.class_ids[i],
                display=recognizer.display_names[i],
                confidence=float(value),
            )
        )
    return out


# The module-level ``app`` is what uvicorn imports as ``app:app``.
app = create_app()


__all__ = [
    "MAX_UPLOAD_BYTES",
    "Prediction",
    "RecognizeResponse",
    "RecognizerState",
    "ServiceSettings",
    "app",
    "build_recognizer",
    "create_app",
    "get_recognizer",
    "recognize_image",
]
