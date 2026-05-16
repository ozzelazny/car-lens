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

- [x] **3.1** Image downloader (curl_cffi, browser-fingerprint TLS)
- [ ] **3.2** pHash near-duplicate detection
- [ ] **3.3** CLIP zero-shot **view + content labeling** — per image, assign `(view ∈ {front, rear, side, three-quarter-front, three-quarter-rear, interior, detail, non-car}, score)`. Drop non-car. Keep interior/detail rows in DB (excluded from training) so a v1.1 interior path is possible without re-crawling.
- [ ] **3.4** Quality filter (resolution, blur, aspect ratio) — applied after view labeling.
- [ ] **3.5** Train / val / test split **stratified by (class, view)** — no view leaks across splits. Exterior-only for v1.

## Phase 4 — Public datasets

- [ ] **4.1** Stanford Cars downloader + label normalizer
- [ ] **4.2** VMMRdb downloader + label normalizer
- [ ] **4.3** CompCars downloader + label normalizer
- [ ] **4.4** Wikimedia Commons fetcher (vintage gap)
- [ ] **4.5** Unified label schema across all sources

## Phase 5 — Model training (not yet planned in detail)

- [ ] **5.1** Baseline: pre-trained MobileCLIP-S2 **per-view** zero-shot prototype retrieval, report top-1/top-5 per view and overall.
- [ ] **5.2** Fine-tune MobileCLIP-S2 with view-conditional contrastive loss + **hard-negative mining** (Camry↔Accord, F-150↔Silverado, Civic↔Corolla, etc.). Train shared backbone; emit one embedding head per image.
- [ ] **5.3** Train the **view classifier** (small head on the same backbone over 5 exterior views + 1 "non-exterior" class).
- [ ] **5.4** Evaluation harness — held-out test set, top-1/top-5 broken down by `(make, view, era)`; full confusion matrix; per-view accuracy gap analysis.
- [ ] **5.5** ONNX export (embedder + view classifier) + Core ML + TFLite conversion.
- [ ] **5.6** On-device latency benchmark (embedder ≤ 30 ms target, view classifier ≤ 5 ms target).

## Phase 6 — `recognize()` engine interface (not yet planned in detail)

- [ ] **6.1** `recognize()` Python interface — view-detect → view-conditional retrieval → top-K candidates. Reject non-exterior inputs with a clear reason.
- [ ] **6.2** Cloud-LLM re-rank fallback for low-confidence top-1 (threshold TBD post-eval).
- [ ] **6.3** CLI for one-shot inference + batch evaluation.

## Known follow-ups (non-blocking)

- Catalog Title Case loses canonical capitalization for compound names: `"MCLAREN"` → `"Mclaren"`, not `"McLaren"`; same for `"BMW"` (becomes `"Bmw"`). Acceptable for taxonomy + matching, but the app's display layer will need a canonical-name override table eventually.
- BaT name-parse doesn't recognize two-word *models* like `"Del Sol"`, `"Type R"`. Two-word-make matching exists (Land Rover, Alfa Romeo) but no symmetric two-word-model logic. A `1997 Honda Del Sol` listing comes out as `model="Del", trim="Sol Si 5-Speed"`. Affects a handful of model families.
- `parse_proxy_url` still echoes the raw URL in the `"missing scheme"` `ValueError` (e.g. `//user:pass@host:8080` → message contains credentials). Triggered only by schemeless input, which is unusual but possible. The missing-host and missing-port paths were fixed in `b390fa4`; this third path remains as the only residual leak surface.
- Crawler smoke results are non-deterministic against Cloudflare-protected sites (cars.com observed flipping 200→403 between identical runs). Validation of parser fixes for those sites requires either consistent access via residential proxy OR a fixture-based test against saved HTML snapshots.
- AutoTrader vehicle-detail pages return an Akamai Bot Manager interstitial (HTTP 200, ~3.7 KB, title "Autotrader - page unavailable", assets under `/akamai-block/...`) when fetched via `CurlCffiFetcher(impersonate="chrome131")`. Sitemap URL discovery still works, but no detail data is extractable without a residential proxy. Fixture preserved at `tests/crawler/parsers/fixtures/real_world/autotrader_detail_curlcffi_chrome131_20260515T231622Z.html`. Reopen for parser work when a proxy pool is provisioned.

## Status legend

- `[ ]` pending
- `[~]` in progress
- `[x]` done
