# Car Lense — Top-level TODO

See `DESIGN.md` for architecture. This file tracks high-level progress across the whole project.

## Current focus

Recognition engine — Phase 1 (catalog + crawler). No model training yet.

## Phase 1 — Catalog and crawler infrastructure

- [x] **1.1** Project scaffold (pyproject.toml, src/ layout, .gitignore, ruff config, pytest config)
- [x] **1.2** SQLite schema + DB access layer (listings, images, crawl_queue, dedupe)
- [x] **1.3** NHTSA vPIC catalog builder — pull canonical (year, make, model) list, cache locally
- [x] **1.4** Search-query generator — produce per-site search URLs for top-N (make, model, year) combos
- [x] **1.5** Crawler core — Playwright + stealth, request queue, retry/backoff, rate-limit, resume-after-restart

## Phase 2 — Per-site parsers

- [x] **2.1** cars.com listing parser + image extractor
- [x] **2.2** AutoTrader listing parser + image extractor
- [x] **2.3** Craigslist listing parser (free-text title → structured label)
- [x] **2.4** Bring a Trailer / Hemmings / Cars & Bids parsers

## Phase 3 — Image pipeline

- [ ] **3.1** Image downloader (curl_cffi, browser-fingerprint TLS)
- [ ] **3.2** pHash near-duplicate detection
- [ ] **3.3** CLIP-similarity label-noise filter
- [ ] **3.4** Quality filter (resolution, blur, aspect ratio)
- [ ] **3.5** Train / val / test split with class-stratified sampling

## Phase 4 — Public datasets

- [ ] **4.1** Stanford Cars downloader + label normalizer
- [ ] **4.2** VMMRdb downloader + label normalizer
- [ ] **4.3** CompCars downloader + label normalizer
- [ ] **4.4** Wikimedia Commons fetcher (vintage gap)
- [ ] **4.5** Unified label schema across all sources

## Phase 5 — Model training (not yet planned in detail)

- [ ] **5.1** Baseline: pre-trained MobileCLIP-S2 zero-shot prototype-retrieval, report top-1/top-5
- [ ] **5.2** Fine-tune MobileCLIP-S2 on combined dataset
- [ ] **5.3** Evaluation harness — held-out test set, per-make/per-era breakdown
- [ ] **5.4** ONNX export + Core ML + TFLite conversion
- [ ] **5.5** On-device latency benchmark

## Phase 6 — `recognize()` engine interface (not yet planned in detail)

- [ ] **6.1** `recognize()` Python interface with two backends (cloud-LLM, on-device)
- [ ] **6.2** CLI for one-shot inference + batch evaluation

## Known follow-ups (non-blocking)

- Catalog Title Case loses canonical capitalization for compound names: `"MCLAREN"` → `"Mclaren"`, not `"McLaren"`; same for `"BMW"` (becomes `"Bmw"`). Acceptable for taxonomy + matching, but the app's display layer will need a canonical-name override table eventually.

## Status legend

- `[ ]` pending
- `[~]` in progress
- `[x]` done
