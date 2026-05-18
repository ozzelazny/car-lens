"""Tests for the Phase 6.1 recognize() FastAPI service.

The OpenCLIP backbone and the prototype tensor are both stubbed in
memory; no model is downloaded and no checkpoint or prototype file is
read from disk. The strategy mirrors the Phase 5.1 baseline tests
(``tests/eval/test_baseline.py``) but operates at the HTTP layer.

We bypass the FastAPI lifespan handler entirely: a freshly-constructed
:class:`RecognizerState` is installed on ``app.state.recognizer`` so
the ``get_recognizer`` dependency picks it up. This decouples the HTTP
assertions from torch + open_clip + PIL bring-up cost, while still
exercising the real validation paths in ``recognize_image``.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# Skip the whole module if FastAPI isn't available -- it lives in the
# service container's image, not in the project's main pyproject.toml,
# so dev checkouts that haven't installed it will skip these tests
# instead of erroring during collection.
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("multipart")
torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")

from fastapi.testclient import TestClient  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_APP_PATH = _REPO_ROOT / "services" / "recognize_api" / "app.py"


def _import_app_module() -> Any:
    """Import ``services/recognize_api/app.py`` as a one-off module.

    We cache it under ``recognize_api_app`` so subsequent imports
    don't re-execute the FastAPI factory.
    """
    if "recognize_api_app" in sys.modules:
        return sys.modules["recognize_api_app"]
    spec = importlib.util.spec_from_file_location("recognize_api_app", _APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["recognize_api_app"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------- stubs


class _StubModel:
    """Tiny encoder that maps every batch to a fixed normalized vector.

    The vector is parameterized at construction time so different tests
    can verify that different inputs produce different rankings. The
    stub has a ``logit_scale`` attribute matching OpenCLIP's surface
    so :class:`RecognizerState` reads a deterministic temperature.
    """

    def __init__(self, embedding: Any) -> None:
        self._embedding = embedding
        # OpenCLIP stores logit_scale as a parameter whose ``.exp()``
        # is the temperature. We stash a torch tensor here so
        # ``logit_scale.exp().item()`` works the same way.
        import math

        self.logit_scale = torch.tensor(math.log(100.0))

    def encode_image(self, batch: Any) -> Any:
        n = batch.shape[0]
        return self._embedding.unsqueeze(0).expand(n, -1)

    def eval(self) -> _StubModel:
        return self


def _stub_preprocess(_img: Any) -> Any:
    """Return a deterministic dummy tensor. The stub model ignores it."""
    return torch.zeros((3, 4, 4))


def _make_recognizer(
    *,
    embedding: Any,
    prototypes: Any,
    class_ids: list[str],
    display_names: list[str],
) -> Any:
    """Construct a :class:`RecognizerState` from in-memory tensors."""
    app_mod = _import_app_module()
    settings = app_mod.ServiceSettings(
        model_path=None,
        prototypes_path=Path("/tmp/unused-prototypes.pt"),
        device="cpu",
        model_name="MobileCLIP-S2",
        pretrained="datacompdr",
        top_k=5,
    )
    return app_mod.RecognizerState(
        settings=settings,
        model=_StubModel(embedding=embedding),
        preprocess=_stub_preprocess,
        prototypes=prototypes,
        class_ids=class_ids,
        display_names=display_names,
        torch_mod=torch,
        model_label="stub-model",
    )


def _l2_normalize(t: Any) -> Any:
    return t / t.norm(dim=-1, keepdim=True)


def _png_bytes(color: tuple[int, int, int] = (200, 50, 50)) -> bytes:
    """Produce a tiny valid PNG byte payload."""
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (8, 8), color=color).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------- fixtures


@pytest.fixture
def app_module() -> Any:
    return _import_app_module()


@pytest.fixture
def client_with_recognizer(app_module: Any) -> Iterator[tuple[Any, Any]]:
    """Spin up a TestClient with a deterministic recognizer installed.

    The recognizer has 3 classes whose prototype directions are the
    three standard basis vectors in 4-D. The stub encoder always emits
    ``[1, 0, 0, 0]``, so the unambiguous top-1 is class 0.
    """
    # 3 classes, 4-D embeddings. Prototypes are the basis vectors.
    proto = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    proto = _l2_normalize(proto)
    embedding = _l2_normalize(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    recognizer = _make_recognizer(
        embedding=embedding,
        prototypes=proto,
        class_ids=["2012|honda|civic", "2012|toyota|camry", "2012|mazda|3"],
        display_names=[
            "2012-2015 Honda Civic",
            "2012-2015 Toyota Camry",
            "2012-2015 Mazda 3",
        ],
    )

    # Build a lifespan-less app so we never hit the real model loader
    # (which would download MobileCLIP from HF and look for a non-
    # existent prototypes.pt). The recognizer is injected onto
    # ``app.state`` directly; ``get_recognizer`` reads from there.
    app = app_module.create_app(use_lifespan=False)
    app.state.recognizer = recognizer
    client = TestClient(app)
    yield client, recognizer
    client.close()


# --------------------------------------------------------------- tests


def test_root_returns_banner(client_with_recognizer: tuple[Any, Any]) -> None:
    client, _ = client_with_recognizer
    resp = client.get("/")
    assert resp.status_code == 200
    assert "recognize-api" in resp.text


def test_health_returns_model_info(
    client_with_recognizer: tuple[Any, Any],
) -> None:
    client, recognizer = client_with_recognizer
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["n_classes"] == 3
    assert body["model"] == "stub-model"
    assert body["device"] == recognizer.settings.device


def test_recognize_returns_top_predictions(
    client_with_recognizer: tuple[Any, Any],
) -> None:
    client, _ = client_with_recognizer
    img = _png_bytes()
    resp = client.post(
        "/api/recognize",
        files={"image": ("car.png", img, "image/png")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 3 classes in the recognizer, top_k=5 -> we get min(5, 3) = 3.
    assert len(body["predictions"]) == 3
    assert body["predictions"][0]["class_id"] == "2012|honda|civic"
    assert body["predictions"][0]["display"] == "2012-2015 Honda Civic"
    # Confidence ranks should be descending.
    confs = [p["confidence"] for p in body["predictions"]]
    assert confs == sorted(confs, reverse=True)
    assert all(0.0 <= c <= 1.0 for c in confs)
    assert body["elapsed_ms"] >= 0.0


def test_recognize_rejects_non_image_content_type(
    client_with_recognizer: tuple[Any, Any],
) -> None:
    client, _ = client_with_recognizer
    resp = client.post(
        "/api/recognize",
        files={"image": ("note.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 415


def test_recognize_rejects_empty_upload(
    client_with_recognizer: tuple[Any, Any],
) -> None:
    client, _ = client_with_recognizer
    resp = client.post(
        "/api/recognize",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_recognize_rejects_undecodable_image(
    client_with_recognizer: tuple[Any, Any],
) -> None:
    client, _ = client_with_recognizer
    # image/* content-type so we pass the 415 gate, but bytes are not
    # a real image -- should fail PIL decode and hit the 400 branch.
    resp = client.post(
        "/api/recognize",
        files={"image": ("junk.jpg", b"\x00\x01\x02 not a real jpeg", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_recognize_rejects_missing_image_field(
    client_with_recognizer: tuple[Any, Any],
) -> None:
    client, _ = client_with_recognizer
    resp = client.post("/api/recognize")
    # FastAPI's default for a missing required form field is 422.
    assert resp.status_code == 422


def test_recognize_rejects_oversized_image(
    client_with_recognizer: tuple[Any, Any],
    app_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = client_with_recognizer
    # Shrink the size limit so we don't have to allocate 20 MB in the
    # test. The handler reads the env-derived constant from the
    # module, so we patch it at module level.
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 256)
    big = _png_bytes() + b"\x00" * 512
    resp = client.post(
        "/api/recognize",
        files={"image": ("big.png", big, "image/png")},
    )
    assert resp.status_code == 413


def test_get_recognizer_503_when_uninitialized(app_module: Any) -> None:
    """A request that arrives before lifespan startup returns 503.

    We build a lifespan-less app, leave ``app.state.recognizer``
    unset (i.e. None), and confirm the dependency raises 503 rather
    than a 500 traceback.
    """
    app = app_module.create_app(use_lifespan=False)
    app.state.recognizer = None
    client = TestClient(app)
    try:
        resp = client.get("/health")
        assert resp.status_code == 503
    finally:
        client.close()


def test_display_name_for_renders_year_range() -> None:
    """The build-prototypes display-name renderer expands bucketed years."""
    from car_lense_engine.eval.build_prototypes_cli import _display_name_for

    assert _display_name_for("2012|honda|civic") == "2012-2015 Honda Civic"
    assert _display_name_for("2016|toyota|camry") == "2016-2019 Toyota Camry"
    # Non-parseable input survives verbatim.
    assert _display_name_for("garbage") == "garbage"
    assert _display_name_for("notayear|honda|civic") == "notayear|honda|civic"


def test_service_settings_from_env_defaults() -> None:
    """Empty env yields the documented defaults."""
    app_mod = _import_app_module()
    settings = app_mod.ServiceSettings.from_env(env={})
    assert settings.device == "cpu"
    assert settings.model_path is None
    assert settings.prototypes_path == Path("/app/cache/prototypes.pt")
    assert settings.model_name == "MobileCLIP-S2"
    assert settings.pretrained == "datacompdr"
    assert settings.top_k == 5


def test_service_settings_from_env_overrides() -> None:
    """Env variables override defaults."""
    app_mod = _import_app_module()
    settings = app_mod.ServiceSettings.from_env(
        env={
            "MODEL_PATH": "/some/path.pt",
            "PROTOTYPES_PATH": "/another/proto.pt",
            "DEVICE": "cuda",
            "TOP_K": "10",
        }
    )
    assert settings.model_path == Path("/some/path.pt")
    assert settings.prototypes_path == Path("/another/proto.pt")
    assert settings.device == "cuda"
    assert settings.top_k == 10
