"""FastAPI ``recognize()`` service for Phase 6.1.

Endpoints
---------

* ``GET  /``             -> human-readable banner pointing at ``/docs``.
* ``GET  /health``       -> ``{"status": "ok", "model": ..., "n_classes": ...}``.
* ``POST /api/recognize``-> multipart ``image`` upload, returns top-5
  predictions as JSON. Body is the dump of :class:`RecognizeResponse`.

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
from fastapi.responses import PlainTextResponse
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


@dataclass(frozen=True)
class ServiceSettings:
    """Resolved environment configuration for the recognize service."""

    model_path: Path | None
    prototypes_path: Path
    device: str
    model_name: str
    pretrained: str
    top_k: int

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ServiceSettings:
        e = env if env is not None else dict(os.environ)
        model_env = e.get("MODEL_PATH", "").strip()
        model_path = Path(model_env) if model_env else None
        prototypes_path = Path(
            e.get("PROTOTYPES_PATH", "/app/cache/prototypes.pt").strip()
            or "/app/cache/prototypes.pt"
        )
        return cls(
            model_path=model_path,
            prototypes_path=prototypes_path,
            device=e.get("DEVICE", "cpu").strip() or "cpu",
            model_name=e.get("MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME,
            pretrained=e.get("PRETRAINED", DEFAULT_PRETRAINED).strip() or DEFAULT_PRETRAINED,
            top_k=int(e.get("TOP_K", str(DEFAULT_TOP_K))),
        )


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


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    model_config = ConfigDict(extra="forbid")

    status: str
    model: str
    n_classes: int
    device: str


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
        prototypes: Any,
        class_ids: list[str],
        display_names: list[str],
        torch_mod: Any,
        model_label: str,
        logit_scale: float = 100.0,
    ) -> None:
        self.settings = settings
        self.model = model
        self.preprocess = preprocess
        self.prototypes = prototypes
        self.class_ids = class_ids
        self.display_names = display_names
        self.torch = torch_mod
        self.model_label = model_label
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
    """Load and validate a ``build-prototypes`` payload."""
    if not path.exists():
        raise RuntimeError(
            f"prototype file not found at {path}; run the build-prototypes CLI "
            "first, e.g. `build-prototypes --source compcars --checkpoint "
            "models/checkpoints/mobileclip_s2_compcars_epoch09_top1_91.9.pt`"
        )
    payload = torch_mod.load(path, map_location="cpu", weights_only=False)
    required = ("class_ids", "display_names", "prototypes")
    if not isinstance(payload, dict) or any(k not in payload for k in required):
        raise RuntimeError(
            f"prototype file at {path} is malformed; expected a dict with "
            f"keys {required}, got {type(payload).__name__}"
        )
    if len(payload["class_ids"]) != int(payload["prototypes"].shape[0]):
        raise RuntimeError(
            f"prototype file at {path} has inconsistent shape: "
            f"{len(payload['class_ids'])} class_ids vs "
            f"{int(payload['prototypes'].shape[0])} prototype rows"
        )
    return dict(payload)


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
    """
    import torch  # noqa: PLC0415

    model, preprocess, model_label = _load_openclip_model(settings, torch)
    payload = _load_prototype_payload(settings.prototypes_path, torch)
    proto_tensor = payload["prototypes"].to(settings.device)
    return RecognizerState(
        settings=settings,
        model=model,
        preprocess=preprocess,
        prototypes=proto_tensor,
        class_ids=list(payload["class_ids"]),
        display_names=list(payload["display_names"]),
        torch_mod=torch,
        model_label=model_label,
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


def create_app(*, use_lifespan: bool = True) -> FastAPI:
    """Construct the FastAPI app.

    ``use_lifespan`` defaults to ``True`` so the production entry point
    (``app = create_app()`` at module level) gets the model + prototype
    load on startup. Tests pass ``use_lifespan=False`` to bypass the
    real load and inject a stub :class:`RecognizerState` onto
    ``app.state.recognizer`` directly -- the model file + prototypes
    don't exist in the test environment, and emulating them on disk
    would defeat the point of the in-memory stub.
    """
    app = FastAPI(
        title="Car Lense recognize() API",
        version="0.1.0",
        lifespan=lifespan if use_lifespan else None,
    )

    @app.get("/", response_class=PlainTextResponse)
    async def root() -> str:
        return "recognize-api ready, see /docs for OpenAPI"

    @app.get("/health", response_model=HealthResponse)
    async def health(
        recognizer: RecognizerState = Depends(get_recognizer),  # noqa: B008
    ) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model=recognizer.model_label,
            n_classes=recognizer.n_classes,
            device=recognizer.settings.device,
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
    predictions = _predict(rgb, recognizer)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return RecognizeResponse(predictions=predictions, elapsed_ms=elapsed_ms)


def _predict(pil_image: Any, recognizer: RecognizerState) -> list[Prediction]:
    """Run encode -> normalize -> cosine sim -> softmax -> top-K.

    The math mirrors :func:`car_lense_engine.eval.baseline.evaluate`: a
    single L2-normalized image embedding gets cosine-similarity against
    every L2-normalized prototype, scaled by ``logit_scale`` so the
    softmax has a useful dynamic range.
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
        sims = features @ recognizer.prototypes.T
        scaled = sims * recognizer.logit_scale
        probs = scaled.softmax(dim=-1)[0]
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
