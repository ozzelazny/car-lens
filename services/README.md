# services/

Phase 6.1 deployable surface for the Car Lense `recognize()` engine.
Two containers, wired together via `docker-compose.yml` at the repo
root:

| Service          | Image                       | Host port | Purpose                                  |
| ---------------- | --------------------------- | --------- | ---------------------------------------- |
| `recognize-api`  | `car-lense/recognize-api`   | `8000`    | FastAPI + MobileCLIP + prototypes        |
| `recognize-ui`   | `car-lense/recognize-ui`    | `8080`    | nginx + static SPA, proxies to the API   |

Sub-directories:

* `recognize_api/` -- backend `Dockerfile`, `app.py`, `requirements.txt`,
  `build_prototypes.py` shim.
* `recognize_ui/`  -- frontend `Dockerfile`, `nginx.conf`, `static/`
  (`index.html`, `style.css`, `app.js`).

## One-time prototype build

Prototypes are pre-computed once from the train split and mounted
read-only into the API container. Re-build only when the checkpoint or
the train split changes.

```bash
# Run from the repo root.
./.venv/bin/build-prototypes \
    --source compcars --train-split train \
    --checkpoint models/checkpoints/mobileclip_s2_compcars_epoch09_top1_91.9.pt \
    --output cache/prototypes.pt \
    --device cuda  # use cpu if no GPU
```

This writes `cache/prototypes.pt`, a torch state dict with:

* `class_ids`     -- `["2012|honda|civic", ...]`
* `display_names` -- `["2012-2015 Honda Civic", ...]`
* `prototypes`    -- `(n_classes, embed_dim)` L2-normalized tensor
* `config`        -- the run config + an ISO `built_at` timestamp

## Build + run the stack

```bash
docker compose up --build
```

The frontend depends on the backend `healthcheck`, so Compose will
wait for `/health` to return 200 before bringing up nginx. On a
fresh host the API takes ~30 s to load the model + prototypes (CPU,
~2.5k CompCars classes), so allow ~1 min before the UI is reachable.

Once up:

* Browser -> http://localhost:8080
* Drag a car photo onto the drop zone (or click to pick a file).
* The top-5 predictions render with horizontal confidence bars; the
  footer shows the live `/health` payload (model name + class count +
  device).

## Quick API smoke test

```bash
curl -s http://localhost:8000/health | jq .
curl -s -F image=@/path/to/car.jpg http://localhost:8000/api/recognize | jq .
```

## GPU mode (optional)

The default image is CPU-only. To run on a CUDA host:

1. Rebuild the API image with the CUDA torch wheel:

   ```bash
   docker compose build \
       --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 \
       recognize-api
   ```

2. Add a `runtime: nvidia` (or `gpus: all`) entry to the
   `recognize-api` service in `docker-compose.yml` and set
   `DEVICE: cuda` in the environment block.

3. `docker compose up --build`.
