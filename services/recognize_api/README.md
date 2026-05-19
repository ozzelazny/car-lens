# recognize-api

FastAPI service that serves the Phase 5.2 fine-tuned MobileCLIP-S2
checkpoint as a `POST /api/recognize` endpoint. Used both by the
companion `recognize-ui` container and as a target for ad-hoc curl /
Postman tests.

## Endpoints

| Method | Path             | Description                                |
| ------ | ---------------- | ------------------------------------------ |
| GET    | `/health`        | `{status, model, n_classes, device, view_classifier_loaded, views_with_prototypes, view_classifier_mode, llm_rerank_enabled, llm_rerank_model}` |
| POST   | `/api/recognize` | multipart `image` upload, returns top-5 with optional LLM re-rank (`rerank_applied`, `rerank_latency_ms` in the response body) |
| GET    | `/docs`          | OpenAPI / Swagger UI                       |
| GET    | `/` (+ assets)   | Static UI when `UI_ROOT` is set (see below)|

## Configuration

Set via environment variables:

| Var                     | Default                                                       | Meaning                                  |
| ----------------------- | ------------------------------------------------------------- | ---------------------------------------- |
| `MODEL_PATH`            | `/app/models/mobileclip_s2_compcars_epoch09_top1_91.9.pt`     | Phase 5.2 checkpoint to overlay          |
| `PROTOTYPES_PATH`       | `/app/cache/prototypes.pt`                                    | Output of `build-prototypes`. May be either a **v1** (single-prototype) or **v2** (per-view; produced by `build-prototypes --per-view`) payload — the loader auto-detects via the `schema_version` key. |
| `DEVICE`                | `cpu`                                                         | torch device (`cpu` / `cuda` / `mps`)   |
| `MODEL_NAME`            | `MobileCLIP-S2`                                               | OpenCLIP model name                      |
| `PRETRAINED`            | `datacompdr`                                                  | OpenCLIP pretrained tag                  |
| `TOP_K`                 | `5`                                                           | Number of predictions returned           |
| `UI_ROOT`               | *(unset)*                                                     | Optional directory to mount at `/` as the static UI (single-process mode); leave unset in production where nginx serves the UI. |
| `VIEW_CLASSIFIER_PATH`  | *(unset)*                                                     | Optional path to the Phase 5.3 view-classifier checkpoint (`models/checkpoints/view_classifier_v1.pt`). When set, `PROTOTYPES_PATH` must point at a v2 payload; the service runs view-conditional retrieval (classify view → reject non-exterior → retrieve against the matching view's prototypes). When unset, the service falls back to the legacy single-prototype path. |
| `VIEW_REJECT_THRESHOLD` | `0.5`                                                         | Minimum softmax probability the view classifier must assign to its top view before the request is allowed through. Below this (or when the top view is `non-exterior`), the service returns HTTP 422 with body `{"detail": "non-exterior view rejected", "view": ..., "view_score": ...}`. Only consulted when `VIEW_CLASSIFIER_PATH` is set. |
| `LLM_RERANK_ENABLED`    | `0`                                                           | Set to `1` (or `true`/`yes`/`on`) to enable the Phase 6.2 Cloud-LLM re-rank fallback. When enabled, requests where the model's top-1 confidence is below `LLM_RERANK_THRESHOLD` are forwarded to the Anthropic Claude API to choose among the top-5 candidates using the image. **Requires `ANTHROPIC_API_KEY` to be set; the service refuses to start without it.** |
| `LLM_RERANK_THRESHOLD`  | `0.5`                                                         | Trigger the LLM re-rank when the on-device top-1 confidence is strictly below this value. With the Phase 5.2 backbone (top-5 ~97.9%) the right answer is almost always in the candidate set, so the LLM only needs to disambiguate trim/year. |
| `LLM_RERANK_MODEL`      | `claude-sonnet-4-6`                                           | Anthropic model identifier to use for re-rank (e.g. `claude-sonnet-4-6`, `claude-opus-4-7`). |
| `LLM_RERANK_TIMEOUT`    | `10.0`                                                        | Per-call HTTP timeout (seconds) for the Anthropic API. Failures (timeout, malformed response, out-of-range index) silently fall back to the original on-device ranking; the response carries `rerank_applied=false` and a non-null `rerank_latency_ms` so failures stay observable. |
| `ANTHROPIC_API_KEY`     | *(unset)*                                                     | Required when `LLM_RERANK_ENABLED=1`. The service raises a clear error at startup if missing. |

## Running locally (without docker)

```bash
# 1. Build prototypes once (see top-level services/README.md).
./.venv/bin/build-prototypes \
    --source compcars --train-split train \
    --checkpoint models/checkpoints/mobileclip_s2_compcars_epoch09_top1_91.9.pt \
    --output cache/prototypes.pt --device cuda

# 2a. Start the API only (uses the same .venv).
MODEL_PATH=models/checkpoints/mobileclip_s2_compcars_epoch09_top1_91.9.pt \
PROTOTYPES_PATH=cache/prototypes.pt DEVICE=cpu \
./.venv/bin/uvicorn services.recognize_api.app:app --port 8000

# 2b. ...or start API + static UI from a single uvicorn process by
#     setting UI_ROOT. The static files use relative /api/recognize
#     calls so same-origin avoids any CORS dance.
MODEL_PATH=models/checkpoints/mobileclip_s2_compcars_epoch09_top1_91.9.pt \
PROTOTYPES_PATH=cache/prototypes.pt DEVICE=cuda \
UI_ROOT=services/recognize_ui/static \
./.venv/bin/uvicorn services.recognize_api.app:create_app --factory \
    --host 0.0.0.0 --port 8000

# 3. Try it.
curl -F image=@some_car.jpg http://localhost:8000/api/recognize
```

## LLM re-rank fallback

The Phase 5.4 evaluation shows the failure mode of the retrieval-only
pipeline is *fine-grained intra-make trim/year confusion*
(e.g. `Infiniti QX80 2008-2011` vs `2012-2015`). With top-5 accuracy
~97.9% the right answer is almost always in the candidate set, so
when the on-device top-1 confidence is low it pays to ask a cloud
multimodal LLM to look at the image and choose among the top-5
candidates.

When `LLM_RERANK_ENABLED=1`, the recognize endpoint:

1. Runs the existing retrieval + view-classifier rejection flow.
2. If the top-1 prediction's confidence is `>= LLM_RERANK_THRESHOLD`,
   returns the result unchanged (`rerank_applied=false`,
   `rerank_latency_ms=null`).
3. Otherwise resizes the upload (longest side <= 768 px), JPEG-encodes
   at quality 85, and asks the Anthropic Claude API to pick the best
   candidate index. Successful re-ranks reorder the candidate list so
   the LLM's pick becomes the new top-1 and the top-1 confidence is
   raised to at least `LLM_RERANK_THRESHOLD` so downstream "low
   confidence" UI states do not fire on an LLM-endorsed answer.
4. Any failure (timeout, malformed response, out-of-range index,
   network error) preserves the on-device ranking and reports
   `rerank_applied=false`. The `rerank_latency_ms` field is still set
   so observability can track failed-call cost.

**Costs.** A single re-rank call sends one ~50 KB JPEG and a tiny
prompt (~100 tokens) and asks for at most 8 output tokens, so the
typical cost is ~$0.01 per call with Claude Sonnet 4.6. If you enable
re-rank for *every* request, costs scale linearly with QPS; the
threshold-gated default keeps spend bounded to "uncertain" requests
only.

**New response fields.**

```jsonc
{
  "predictions": [...],
  "elapsed_ms": 812.3,
  "view": "front",
  "view_score": 0.92,
  "rerank_applied": true,      // false when re-rank was off or skipped
  "rerank_latency_ms": 1240.5  // null when re-rank wasn't invoked
}
```

`GET /health` mirrors the runtime configuration:

```jsonc
{
  "status": "ok",
  "model": "...",
  "n_classes": 6423,
  "device": "cuda",
  "view_classifier_loaded": true,
  "view_classifier_mode": "binary",
  "views_with_prototypes": [],
  "llm_rerank_enabled": true,
  "llm_rerank_model": "claude-sonnet-4-6"
}
```

## Notes

* The fine-tuned weights are overlaid *after* the OpenCLIP pretrained
  weights -- if `MODEL_PATH` is unset or doesn't exist, the service
  refuses to start (silent fallback would mask a misconfiguration).
* Prototypes are loaded once at lifespan startup and held in memory.
  Re-running `build-prototypes` requires a container restart.
* GPU support is opt-in: build with the CUDA torch index and set
  `DEVICE=cuda`. The default image ships CPU torch only.
* The Anthropic SDK is an optional dependency. It ships in
  `pyproject.toml` and the production Docker image. For local-dev
  checkouts that don't yet have it installed, run
  `./.venv/bin/pip install --no-cache-dir 'anthropic>=0.40.0'`
  (do **not** `uv sync` -- that will bump the CUDA torch wheel and
  break the dev environment).
