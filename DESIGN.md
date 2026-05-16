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

1. **Image encoder model** — MobileCLIP-S2 fine-tuned on car data. Exported to ONNX, then converted to Core ML (iOS) and TFLite (Android). Inference target: <30ms on mid-range mobile.
2. **View classifier** — small head over the same backbone that routes an input image to one of `{front, rear, side, three-quarter-front, three-quarter-rear, non-exterior}`. Inference target: <5ms; non-exterior inputs are rejected ("please photograph the car from outside").
3. **Class catalog** — for each `(make, model, generation)`, **one prototype embedding per exterior view** (~5 views × ~2,000 classes ≈ 10k prototypes, ~25 MB). Distributed with the app.
4. **`recognize()` interface** — Python module used by tests, evaluation, and (initially) a thin cloud API. Same logical contract as the on-device code.

```python
recognize(image: PIL.Image) -> list[Match]
# Match = {make, model, generation, year_range, view, confidence, ...}
# Returns [] (with a "non-exterior" reason) if the view classifier rejects the input.
```

### Accuracy-first design decisions

Accuracy is the primary product priority. The architecture reflects:

- **View-conditional retrieval**: query embedding is compared only against prototypes of the *same view* (front-Civic vs front-Camry, not front-Civic vs side-Camry). Removes a major source of confusion in single-prototype CLIP retrieval.
- **Exterior-only v1**: interior recognition is doable but lower-ceiling (~70–80% vs ~93–95% top-1) and not shipping in v1. Interior images are *not* discarded from disk — they're labeled and excluded from training, so an "interior path" can be added in v1.1 as an additive feature with no API changes.
- **Hard-negative mining** during fine-tune: the loss explicitly contrasts visually similar but different classes (Camry vs Accord, F-150 vs Silverado, Civic vs Corolla) so the model learns the discriminative features rather than overall body shape.
- **Cloud-LLM re-rank fallback** for low-confidence cases: when on-device top-1 confidence < threshold (TBD post-eval), the top-K candidates plus the image go to the cloud LLM for a finer judgment. Preserves the "online enrichment" assumption already in the system architecture.
- **Latency budget** stays ~30 ms on-device for the embedder + ~5 ms for the view classifier. If post-eval shows the smaller MobileCLIP-S1 closes most of the gap on the larger MobileCLIP-B with much lower latency, we'll evaluate the trade-off before final export.

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
   MobileCLIP-S2 fine-tune with view-conditional
   contrastive loss + hard-negative mining
   (Camry↔Accord, F-150↔Silverado, etc.)
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
