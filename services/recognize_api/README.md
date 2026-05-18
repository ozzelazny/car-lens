# recognize-api

FastAPI service that serves the Phase 5.2 fine-tuned MobileCLIP-S2
checkpoint as a `POST /api/recognize` endpoint. Used both by the
companion `recognize-ui` container and as a target for ad-hoc curl /
Postman tests.

## Endpoints

| Method | Path             | Description                                |
| ------ | ---------------- | ------------------------------------------ |
| GET    | `/`              | Banner; points at `/docs`.                 |
| GET    | `/health`        | `{status, model, n_classes, device}`       |
| POST   | `/api/recognize` | multipart `image` upload, returns top-5    |
| GET    | `/docs`          | OpenAPI / Swagger UI                       |

## Configuration

Set via environment variables:

| Var              | Default                                                       | Meaning                                  |
| ---------------- | ------------------------------------------------------------- | ---------------------------------------- |
| `MODEL_PATH`     | `/app/models/mobileclip_s2_compcars_epoch09_top1_91.9.pt`     | Phase 5.2 checkpoint to overlay          |
| `PROTOTYPES_PATH`| `/app/cache/prototypes.pt`                                    | Output of `build-prototypes`             |
| `DEVICE`         | `cpu`                                                         | torch device (`cpu` / `cuda` / `mps`)   |
| `MODEL_NAME`     | `MobileCLIP-S2`                                               | OpenCLIP model name                      |
| `PRETRAINED`     | `datacompdr`                                                  | OpenCLIP pretrained tag                  |
| `TOP_K`          | `5`                                                           | Number of predictions returned           |

## Running locally (without docker)

```bash
# 1. Build prototypes once (see top-level services/README.md).
./.venv/bin/build-prototypes \
    --source compcars --train-split train \
    --checkpoint models/checkpoints/mobileclip_s2_compcars_epoch09_top1_91.9.pt \
    --output cache/prototypes.pt --device cuda

# 2. Start the API (uses the same .venv).
MODEL_PATH=models/checkpoints/mobileclip_s2_compcars_epoch09_top1_91.9.pt \
PROTOTYPES_PATH=cache/prototypes.pt DEVICE=cpu \
./.venv/bin/uvicorn services.recognize_api.app:app --port 8000

# 3. Try it.
curl -F image=@some_car.jpg http://localhost:8000/api/recognize
```

## Notes

* The fine-tuned weights are overlaid *after* the OpenCLIP pretrained
  weights -- if `MODEL_PATH` is unset or doesn't exist, the service
  refuses to start (silent fallback would mask a misconfiguration).
* Prototypes are loaded once at lifespan startup and held in memory.
  Re-running `build-prototypes` requires a container restart.
* GPU support is opt-in: build with the CUDA torch index and set
  `DEVICE=cuda`. The default image ships CPU torch only.
