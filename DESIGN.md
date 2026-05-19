# Car Lense — Design

## Product

A mobile (and eventually XR-glasses) app that identifies a vehicle from its image and returns full info: make / model / year / generation plus enrichment data (specs, recalls, valuation, history).

## Architecture (system-wide)

```
┌─────────────────────────────────────────────────────────────────┐
│  Client (iOS first, Android next, XR later)                     │
│   • Camera capture                                              │
│   • On-device recognition (MobileCLIP image encoder + catalog)  │
│   • Result UI / AR overlay                                      │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼ (only when online — for enrichment)
┌─────────────────────────────────────────────────────────────────┐
│  Enrichment API (cloud)                                         │
│   • NHTSA vPIC pass-through (specs, recalls)                    │
│   • EPA fuel economy                                            │
│   • Valuation provider(s)                                       │
│   • LLM fallback for "tell me more" / long-tail recognition     │
└─────────────────────────────────────────────────────────────────┘

                          ┌───────────────────────────────┐
                          │  Offline: recognition engine  │
                          │  build + training pipeline    │
                          └───────────────────────────────┘
```

Two distinct codebases will emerge:
- **`engine/`** — Python, offline. Crawls data, builds catalogs, trains models, exports for mobile.
- **`app/`** — Swift / Kotlin / XR runtime. Consumes the exported model + catalog. Not started.

This document covers `engine/`.

## Recognition engine — scope

The engine produces four deployable artifacts:

1. **Image encoder model** — **MobileCLIP-B** fine-tuned on car data (originally targeted MobileCLIP-S2; upgraded after MobileCLIP-S2's 5-epoch combined-corpus run hit only 75.67% test top-1 on 6,423 classes — B at the same epochs delivered +11.8 pp). 224×224 input, 512-dim embedding, L2-normalized in the exported graph. Trained checkpoint at `models/checkpoints/mobileclip_b_compcars-vmmrdb-stanford_cars_resumed_epoch09_top1_85.7.pt` (val_top1=85.66%, test_top1=81.07% / top-5=98.10%). Exported to ONNX → Core ML FP16 (165 MB `.mlpackage`) → TFLite (currently blocked by an `onnx2tf` segfault; ORT-Mobile fallback path retained). Inference target <30 ms on mid-range Android, <10 ms on iPhone 13+ NPU.
2. **View classifier** — **binary `exterior` / `non-exterior` head** over the same backbone for input gating (val_top1=99.88%). The original 5-way view classifier (front/rear/side/3Q-front/3Q-rear) was abandoned at val_top1=72.5% — the labels themselves (zero-shot CLIP) are noisy on the 3Q-front vs 3Q-rear distinction, and a 72.5% gate would route ~27% of queries to the wrong view's prototypes. Binary mode does input rejection only; retrieval stays single-prototype.
3. **Class catalog** — **6,423 (year-bucketed-make-model)** classes covering compcars + vmmrdb + stanford_cars (US-skewed: Ford 46k, Chevrolet 42k, Toyota 29k, Honda 26k). **Single prototype per class** (512-dim FP16, ~6.5 MB total) — view-conditional prototypes infrastructure exists (`build-prototypes --per-view` → schema-v2 cache) but is deferred until the view classifier is accurate enough to gate retrieval (>90% per design rule of thumb).
4. **`recognize()` interface** — Python module used by tests, evaluation, and the cloud API (`services/recognize_api/`). Same logical contract as the on-device code. iOS MVP scaffold under `app/ios/CarLense/` reproduces the same flow on-device with SwiftUI + AVFoundation + Vision + Core ML.

```python
recognize(image: PIL.Image) -> list[Match]
# Match = {make, model, generation, year_range, view, confidence, ...}
# Returns [] (with a "non-exterior" reason) if the view classifier rejects the input.
```

### Accuracy-first design decisions

Accuracy is the primary product priority. The architecture reflects:

- **View-conditional retrieval** *(deferred — infrastructure shipped, not currently enabled)*: original plan compared query embedding only against same-view prototypes. Abandoned in favor of single-prototype retrieval because the 5-way view classifier only hit 72.5% accuracy on label-noisy zero-shot training data — mis-routing 27% of queries to the wrong view's prototypes regressed end-to-end accuracy. `build_prototypes_by_view` + recognize-api's v2 path are kept for future re-activation if/when a more accurate view classifier ships.
- **Exterior-only v1** *(enforced via binary classifier gate)*: interior recognition is doable but lower-ceiling (~70–80% vs ~93–95% top-1) and not shipping in v1. Interior images are labeled and excluded from class-level training; the binary `non-exterior` classifier (99.88% val) gates inputs at the API boundary.
- **Hard-negative mining** during fine-tune: the loss explicitly contrasts visually similar but different classes (Camry vs Accord, F-150 vs Silverado, Civic vs Corolla) so the model learns discriminative features. Verified by today's eval — remaining top-1 misses are now overwhelmingly within-make trim/year-bucket adjacencies (Impala 04-07 vs 08-11, Civic vs Civic_Coupe), not cross-make confusions.
- **Cloud-LLM re-rank fallback** for low-confidence cases: when on-device top-1 confidence < threshold (default 0.5), the image + top-5 candidate names go to Claude via the Anthropic Messages API for a finer judgment. Code shipped (Phase 6.2 — `services/recognize_api/llm_rerank.py`); env-flag-gated. Top-5 at 98.10% means effective top-1 with re-rank should land ~92-94%.
- **Latency budget** stays ~30 ms on-device for the embedder. MobileCLIP-B is ~10 ms on iPhone NPU, ~30 ms on flagship Android — fits. App bundle ~172 MB with FP16 weights (vs MobileCLIP-S2's ~80 MB) — accepted trade for +11.8 pp test top-1.

## Recognition engine — pipeline

```
NHTSA vPIC ──► canonical catalog (year, make, model, trim)
                       │
                       ▼
            search-query generator
                       │
                       ▼
   ┌───────────────────┴───────────────────┐
   │              Crawler                  │
   │  Playwright + stealth + curl_cffi     │
   │  Targets: cars.com, AutoTrader,       │
   │           Craigslist, BaT/Hemmings/   │
   │           Cars&Bids                   │
   │  Politeness: 1 worker, 3–5s delay,    │
   │              off-peak only            │
   └───────────────────┬───────────────────┘
                       │
                       ▼
            SQLite metadata + image files
                       │
                       ▼
            pHash dedupe + quality filter
                       │
                       ▼
   CLIP zero-shot view + content labeling
   (drop non-car + interior; keep front/rear/
    side/three-quarter as labeled exteriors)
                       │
                       ▼
   train / val / test split, stratified by
   (class, view) — no view leaks across splits
                       │
                       ▼
   MobileCLIP-B fine-tune (resumed +10 epochs)
   with hard-negative-weighted CE
   (single-prototype retrieval; view-conditional
   infra retained but not enabled)
                       │
                       ▼
   evaluation harness: top-1 / top-5 per
   (make, view, era); confusion matrix
                       │
                       ▼
            ONNX export → Core ML / TFLite
```

## Catalog scope (v1)

- **Market**: US
- **Era**: all years (1981–present, per NHTSA vPIC coverage)
- **Granularity**: `(make, model, generation)` — generations collapsed via VIN decode rules and Wikipedia generation tables
- **Volume target**: top 2,000 combos by US fleet population

## Data sources

Primary (crawled):
- **cars.com** — modern listings, broad coverage
- **AutoTrader.com** — modern listings, broad coverage
- **Craigslist** — diverse real-world conditions, messier labels
- **Bring a Trailer / Hemmings / Cars & Bids** — vintage + enthusiast cars, fills the pre-2010 gap

Crawling profile: 1 worker per site, 3–5s jittered delay, off-peak only, custom User-Agent identifying the project, robots.txt respected where possible. Target: ~200k unique images post-dedup. **Personal research use; not for redistribution.**

Supplementary (downloads, not crawled):
- Stanford Cars (research-licensed)
- VMMRdb (research-licensed)
- CompCars (research-licensed)
- Wikimedia Commons (CC-licensed) — vintage gap

## Storage layout

```
engine/
  data/
    raw/                       # crawler output, organized by site
      <source>/<listing_id>/<sha256>.<ext>   # image content-addressed
      <source>/<listing_id>/meta.json
    public/                    # downloaded public datasets
      stanford_cars/
      vmmrdb/
      compcars/
    processed/                 # post-dedupe, label-normalized
      train/<class>/<hash>.jpg
      val/<class>/<hash>.jpg
      test/<class>/<hash>.jpg
  catalog/
    nhtsa_dump.json
    classes.json               # canonical (make, model, generation) list
  models/
    checkpoints/
    exported/                  # ONNX, CoreML, TFLite outputs
  db/
    crawl.sqlite               # listings, images, queue, dedupe index
```

### WSL2 storage convention

The repo lives on the Windows mount (`/mnt/c/...`), which is slow for the many-small-file I/O patterns image collection produces. To avoid that bottleneck:

- `data/raw/`, `data/public/`, `data/processed/`, `models/checkpoints/`, `models/exported/` are **symlinks** into a native Linux ext4 location, e.g. `/home/zelazny/car_lense_data/`.
- Code paths in the repo (and downstream tooling) continue to reference the in-repo paths — the symlinks are transparent.
- The symlinked subdirectories are gitignored (see `.gitignore`); the `.gitkeep` placeholders in `data/` and `models/` keep the parent directories tracked.
- On a fresh checkout, run:
  ```
  mkdir -p ~/car_lense_data/{raw,public,processed} ~/car_lense_data/models/{checkpoints,exported}
  ln -s ~/car_lense_data/raw       data/raw
  ln -s ~/car_lense_data/public    data/public
  ln -s ~/car_lense_data/processed data/processed
  ln -s ~/car_lense_data/models/checkpoints models/checkpoints
  ln -s ~/car_lense_data/models/exported    models/exported
  ```
- `db/crawl.sqlite` stays in-repo on the Windows mount: SQLite is a single file with sequential I/O, not the multi-thousand-tiny-files workload that motivates the symlink.

## Non-goals (for now)

- Live mobile app (separate workstream, after engine v1 works)
- Damage detection
- License plate / VIN OCR
- Cloud enrichment API (separate workstream)

## Open questions

- Generation boundaries: where do we get authoritative generation cutoffs per `(make, model)`? Wikipedia tables seem to be the de facto source; may need a one-time manual curation pass.
- Catalog refresh cadence: how often do we re-crawl for new model years? Probably annual.
- Long tail: do we ship v1 with the 2,000-class catalog or a smaller "high confidence" subset and let the LLM fallback handle the rest? TBD post-evaluation.
