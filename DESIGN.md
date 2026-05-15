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

The engine produces three deployable artifacts:

1. **Image encoder model** — MobileCLIP-S2 fine-tuned on car data. Exported to ONNX, then converted to Core ML (iOS) and TFLite (Android). Inference target: <30ms on mid-range mobile.
2. **Class catalog** — for each `(make, model, generation)`, a prototype embedding plus reference metadata (canonical name, year range, body style). Distributed with the app.
3. **`recognize()` interface** — Python module used by tests, evaluation, and (initially) a thin cloud API. Same logical contract as the on-device code.

```python
recognize(image: PIL.Image) -> list[Match]
# Match = {make, model, generation, year_range, confidence, ...}
```

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
            CLIP-similarity label-noise filter
                       │
                       ▼
            train / val / test split
                       │
                       ▼
            MobileCLIP-S2 fine-tune (PyTorch)
                       │
                       ▼
            evaluation harness (top-1 / top-5 / per-make)
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
      cars_com/
        listings/<id>/photo_<n>.jpg
        listings/<id>/meta.json
      autotrader/
      craigslist/
      bat/
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

## Non-goals (for now)

- Live mobile app (separate workstream, after engine v1 works)
- Damage detection
- License plate / VIN OCR
- Cloud enrichment API (separate workstream)

## Open questions

- Generation boundaries: where do we get authoritative generation cutoffs per `(make, model)`? Wikipedia tables seem to be the de facto source; may need a one-time manual curation pass.
- Catalog refresh cadence: how often do we re-crawl for new model years? Probably annual.
- Long tail: do we ship v1 with the 2,000-class catalog or a smaller "high confidence" subset and let the LLM fallback handle the rest? TBD post-evaluation.
