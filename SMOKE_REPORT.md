# End-to-End Smoke Test — 2026-05-15

First real-world run of the Car Lense crawler with all 6 parsers registered,
one canonical search URL per site, conservative pacing (3-5s jittered
politeness delay), `max_items=20`, headless Chromium via Playwright.

This run is observational. No parser or crawler code was modified; findings
below are scoped as separate follow-up tasks.

## Environment

- OS: Linux (WSL2, `Linux 6.6.87.2-microsoft-standard-WSL2`)
- Python: 3.11.15 (uv-managed `.venv-smoke`)
- Playwright: 1.59.0 (Chromium 147.0.7727.15)
- Project: `car-lense-engine 0.0.1`, all dev deps installed via `uv pip install -e ".[dev]"`
- Politeness policy: `min_delay=3.0s`, `max_delay=5.0s`, off-peak gate OFF,
  idle-exit=10s
- User-Agent: native Chromium UA + `; CarLenseResearch/0.1`

## Run summary

- Total elapsed: **137.2 s**
- Crawler `exit_reason`: **`max_items_reached`** (hit the 20-item cap before
  the queue drained — there are 28 pending BaT listing URLs and 14 pending
  BaT image URLs left over, which is expected and good)
- Total fetches: **17 succeeded / 3 failed** (20 total)
- Listings inserted: **14** (all from Bring a Trailer)
- URLs enqueued: **42** (28 BaT listings + 14 BaT images)

## Per-site results

### cars.com
- Search page: **HTTP 403 (Cloudflare/anti-bot)**
- Listing URLs discovered: 0
- Listings parsed: 0
- Image URLs enqueued: 0
- Issues:
  - Search URL is blocked at the network layer before any HTML reaches the
    parser. The Playwright+stealth combo isn't sufficient to clear the
    challenge. Error captured: `fetch failed: HTTP 403 for https://www.cars.com/shopping/results/...`
  - No way to validate the cars.com parser logic from this run.

### AutoTrader
- Search page: **fetched (HTTP 200) but content was an interstitial / sparse
  shell, not the real results page**
- Listing URLs discovered: 0
- Listings parsed: 0
- Image URLs enqueued: 0
- Issues:
  - Direct re-fetch via Playwright returned only ~4.4 KB of HTML containing
    zero `cars-for-sale` hrefs, zero `vehicle-card` / `inventory-listing`
    markers. This is consistent with a JS-rendered SPA where the listing
    grid hasn't hydrated by `wait_until="domcontentloaded"` + 1.5 s settle,
    OR with a bot-detection interstitial.
  - Parser logged the expected note "no listing cards found on search page;
    selectors may need updating" — the parser is doing the right thing; the
    fetch isn't returning a page worth parsing.

### Craigslist
- Search page: **fetched (HTTP 200), but listing-card extraction returned 0
  matches in the actual smoke run**
- Listing URLs discovered: 0
- Listings parsed: 0
- Image URLs enqueued: 0
- Issues:
  - Inconsistent rendering: a separate diagnostic fetch in the same run
    returned 155 KB of HTML containing 81 `/d/` listing hrefs (which the
    parser's regex DOES match — verified directly). A *repeat* fetch a few
    minutes later returned only ~35 KB with 0 matches. Suggests timing /
    JS-hydration variance, or Craigslist serving a stripped page on
    repeated requests from the same IP/UA.
  - Parser regex itself works correctly when given the rich HTML; the issue
    is the page Playwright is observing in the smoke run.
  - **Hypothesis**: increasing `settle_ms` (currently 1500) or switching to
    `wait_until="networkidle"` may help. Not changed in this run.

### Bring a Trailer
- Search page: **fetched (HTTP 200), listing-card extraction worked**
- Listing URLs discovered: **28** in queue (14 done + 14 pending)
- Listings parsed: **14**
- Image URLs enqueued: **14**
- Issues:
  - Sample listing: `bat:1986-honda-civic-22` → year=1986, make=Honda,
    model=Civic, mileage=None, vin=None. Year extracted from JSON-LD
    `name` field (`"1986 Honda Civic Si"`) via `parse_year_safe`. Good.
  - **Make/model issue**: every parsed BaT listing comes back with
    make=Honda, model=Civic regardless of the actual vehicle. The DB
    contains slugs like `bat:1991-honda-crx-89`, `bat:1997-honda-del-sol-3`,
    `bat:2018-honda-civic-type-r-touring-5` — clearly CRX, del Sol, and
    Type R variants, but all labelled as Civic. Root cause: BaT's JSON-LD
    `Product` block has only `name`, `image`, `description`, `offers` —
    `brand`, `manufacturer`, and `model` are all `null`. The parser
    correctly falls back to queue `hints`, which are seeded as
    `(Honda, Civic)`, so every result inherits "Civic" even when it's a
    different model. Year is the only field genuinely extracted from BaT.
  - Image URLs were enqueued — those came from the JSON-LD `image` field,
    which IS populated by BaT (the parser worked here).
  - `mileage` and `vin` not populated — BaT auction listings put these in
    the prose description body, not in JSON-LD. This is a known gap.

### Hemmings
- Search page: **HTTP 403**
- Listing URLs discovered: 0
- Listings parsed: 0
- Image URLs enqueued: 0
- Issues:
  - Same shape as cars.com / carsandbids: blocked before the parser sees
    anything. Error: `fetch failed: HTTP 403 for https://www.hemmings.com/classifieds/cars-for-sale?Make=honda&Model=civic`

### Cars & Bids
- Search page: **HTTP 403**
- Listing URLs discovered: 0
- Listings parsed: 0
- Image URLs enqueued: 0
- Issues:
  - Cloudflare-style block. Error: `fetch failed: HTTP 403 for https://carsandbids.com/search?q=Honda+Civic`

## Identified follow-up tasks

Numbered in rough priority order. Each is small enough to be its own
Coder→Linter→Reviewer→Tester loop.

1. **BaT parser: derive make/model from the JSON-LD `name` field** when
   `brand`/`manufacturer`/`model` are absent. BaT consistently formats
   `name` as `"<year> <make> <model> [<trim>] [...]"` (e.g.
   `"1986 Honda Civic Si"`, `"No Reserve: Original-Owner 1986 Honda Civic Si"`).
   The current parser already reads year from `name`; extend to extract
   make and model. Without this fix, every BaT listing collapses to the
   queue's seed hint (Honda Civic in this run), making the dataset
   useless for non-Civic targets.

2. **Cars.com search 403 — Cloudflare block.** Investigate options:
   (a) longer `settle_ms` to let JS challenges resolve, (b) curl_cffi
   browser-fingerprint TLS fetcher as a fallback, (c) accept that
   cars.com is uncrawlable from this IP block and document the gap.
   DO NOT attempt to evade the block aggressively — respect the site.

3. **Hemmings search 403 — same Cloudflare class of issue as cars.com.**
   Same triage options apply. Hemmings may be more amenable than
   cars.com (smaller / less defended).

4. **Cars & Bids search 403 — same Cloudflare class of issue.**
   Cars & Bids ships a heavily client-rendered React app; even a clean
   fetch may need authenticated state for some pages. Worth checking
   whether their public sitemap/feed (e.g. `/past-auctions`) is more
   accessible.

5. **AutoTrader search returns an interstitial / unhydrated shell**
   (~4.4 KB HTML, 0 listing hrefs). The parser correctly logs "no
   listing cards found; selectors may need updating", but the real
   issue is that the page isn't fully rendered. Try:
   - `wait_until="networkidle"` instead of `domcontentloaded`
   - longer `settle_ms` (e.g. 4-6 s)
   - inspect for a specific selector that signals listing-grid ready
     (e.g. `[data-cmp="inventoryListing"]` or similar) and `wait_for_selector`

6. **Craigslist search hydration variance.** A direct test showed
   Craigslist returns either a full 155 KB page (81 listing hrefs, all
   matching the parser's existing regex) OR a stripped 35 KB page
   depending on when you fetch. Root cause unknown — could be
   anti-scrape rate-limiting or JS hydration timing. The parser regex
   itself is correct. Try the same fix as AutoTrader: longer settle /
   `networkidle` / explicit wait_for_selector on `.cl-search-result`.

7. **BaT listing: extract mileage and VIN from description prose.**
   These live in the listing-page body text on BaT, not JSON-LD. Low
   priority for catalog-building (mileage/VIN not needed for recognition
   training), but useful for downstream enrichment.

8. **General: consider a per-site fetcher fallback chain.** Default to
   Playwright; on 403/cloudflare, fall back to `curl_cffi` with a real
   browser TLS fingerprint. Today both fetcher and stealth are global;
   no per-source escape hatch.

## Open questions for the user

- For sites that 403 from this IP (cars.com, Hemmings, Cars & Bids), do
  we want to add `curl_cffi` as a fallback fetcher, accept the gap, or
  try a residential proxy? The crawler design doc says "personal
  research crawl; respect Cloudflare and rate limits" — the safest
  reading is "accept the gap and document" but worth confirming.
- Should the BaT/Cars & Bids parsers' name-parsing fallback be shared
  utility code (since both sites encode year/make/model in the JSON-LD
  `name`), or kept per-parser? I'd lean shared.
- The smoke harness committed here uses a hardcoded list of seed URLs;
  do we want it scriptable (e.g., read seeds from a YAML / catalog
  query) so it can be re-run against future search-query updates? Out
  of scope for this commit but a likely follow-up.
