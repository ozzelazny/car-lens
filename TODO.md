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
- [x] **3.3** CLIP zero-shot **view + content labeling** — per image, assign `(view ∈ {front, rear, side, three-quarter-front, three-quarter-rear, interior, detail, non-car}, score)`. Drop non-car. Keep interior/detail rows in DB (excluded from training) so a v1.1 interior path is possible without re-crawling.
- [ ] **3.4** Quality filter (resolution, blur, aspect ratio) — applied after view labeling.
- [ ] **3.5** Train / val / test split **stratified by (class, view)** — no view leaks across splits. Exterior-only for v1.

## Phase 4 — Public datasets

- [x] **4.1** Stanford Cars downloader + label normalizer — ingest verified end-to-end against `Multimodal-Fatima/StanfordCars_train` (HF mirror, int ClassLabel decoded via `ds.features['label'].int2str`).
- [x] **4.2** VMMRdb downloader + label normalizer — ingest module + CLI + tests built against `venetis/VMMRdb_make_model_{train,val,test}` (the only working HF mirror). HF mirror is **make_model only** (375 classes, no year), not the full 9,170-class year-make-model release. The label parser handles BOTH formats (year-suffix `honda_civic_2005` and no-year `acura_cl`); when ingesting from the venetis mirror, `listings.year` ends up `NULL`. Live ingest not yet run end-to-end against the live mirror — verify with `import-vmmrdb --limit 50` before kicking off the full run.
- [x] **4.3** CompCars downloader + label normalizer — ingest module + CLI + 33 tests against `JorgeLlorente/CompCars-Repository/Compcars_Data.zip` (16.5 GB ZIP, no `datasets`-streaming API). Custom path: download via direct curl (HF Xet kept rewinding chunks), iterate locally with `zipfile.ZipFile`, resolve integer make_id/model_id via `scipy.io.loadmat` on `misc/make_model_name.mat` (163 makes / 2004 models). `car_type.mat` in the live release ships only a `types` field — no model_type lookup — so `body_style` is always NULL for compcars rows. Live 20-row smoke verified end-to-end: year populated, sensible make/model labels. **ZIP cached on disk at `data/public/compcars/Compcars_Data.zip`; full ingest can run with `import-compcars --zip-path ...` (no re-download).**
- [ ] **4.4** Wikimedia Commons fetcher (vintage gap)
- [x] **4.5** Unified label schema across all sources — canonical_make / canonical_model columns (migration 008) + hand-curated alias map (BWM→BMW, Benz→Mercedes-Benz, Chevy→Chevrolet, Buck→Buick, lowercase→Title-Case) + Title-Case fallback. 152,848 rows canonicalized into 170 makes / 4,762 (year, make, model) classes. Phase 5 baseline + training pipelines now SELECT canonical fields. `canonicalize-labels` CLI is idempotent.

## Phase 5 — Model training (not yet planned in detail)

- [x] **5.1** Baseline: pre-trained MobileCLIP-S2 zero-shot prototype retrieval. Stanford Cars (196 classes, 8,014 test images): **top-1=87.88%, top-3=98.74%, top-5=99.64%, top-10=99.90%**, ~9 min on CPU. Per-view conditioning deferred to 5.2 — single-prototype baseline established first per industry convention.
- [x] **5.2** Fine-tune MobileCLIP-S2 with hard-negative-weighted CE. Stanford Cars 196 classes, 20 epochs on RTX 5090 (~19 min): **top-1=91.48%, top-5=98.85%** (vs zero-shot baseline 87.88%/99.64%). Best epoch 14. View-conditional retrieval still pending Phase 3.5 view-stratified split.
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
- Stanford Cars HF mirror (`Multimodal-Fatima/StanfordCars_train`) emits class strings lowercased (e.g. `"acura tl sedan 2012"`, not `"Acura TL Sedan 2012"`). Our parser preserves the casing it receives, so `listings.make` for `source='stanford_cars'` is lowercase while crawled sources store Title Case. Phase 4.5 (unified label schema) needs a normalize-on-save pass (or a case-insensitive catalog matcher) before training joins make labels across sources.
- Migration 004 (rebuild `listings` to widen `source` CHECK) has a narrow data-loss window between `DROP TABLE listings` and `ALTER TABLE listings_new RENAME TO listings` — if a crash hits in that gap, the retry's `DROP TABLE IF EXISTS listings_new` would destroy the only copy. Safer pattern: rename old → rename new → drop old. Two-statement gap with no I/O between, so probability is tiny; close in a follow-up. *Status: migration 006 (Phase 4.2) demonstrates the safer pattern — adds `PRAGMA legacy_alter_table = ON` to prevent SQLite's auto-rewriting of FKs in `images` when `listings` is RENAMEd. Migration 004 not retroactively rewritten (already-applied DBs would skip a fix anyway).*
- VMMRdb HF mirror (`venetis/VMMRdb_make_model_*`) is the only working HuggingFace upload as of 2026-05; it has **375 make-model classes (no year)**, not the full 9,170-class year-make-model VMMRdb. To get the full year-tagged dataset we'd need to download the original GitHub release (~50 GB) and ingest from disk. Closes the immediate Phase 4.2 ship gate (~100k images, ~375 classes); year-tagged ingest path deferred. Listing rows ingested from the venetis mirror will have `listings.year IS NULL`; Phase 4.5 (unified label schema) and the training-time stratified split need to account for this.

## Status legend

- `[ ]` pending
- `[~]` in progress
- `[x]` done
