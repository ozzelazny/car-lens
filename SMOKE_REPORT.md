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

## Second run — 2026-05-15 (after fixes)

Second smoke run after three follow-up fixes:

- `2b0304f` — BaT parser now extracts make/model from JSON-LD `name`.
- `d45c480` — `PlaywrightFetcher` gained configurable `wait_until` /
  `settle_ms` / `navigation_timeout_ms`; smoke harness uses
  `settle_ms=5000`, `navigation_timeout_ms=45000`,
  `wait_until="domcontentloaded"`.
- `b4aa9e3` — `CurlCffiFetcher` + per-source `MultiFetcher`; smoke
  harness routes `cars_com` and `hemmings` through curl_cffi (Chrome
  131 impersonation) to clear Cloudflare; everything else stays on
  Playwright.

### Status changes vs first run

| Site         | First run    | Second run                | Notes |
|--------------|--------------|---------------------------|-------|
| cars.com     | 403          | 200 (curl_cffi), 0 parsed | curl_cffi cleared block; parser misses path-relative hrefs in non-`<a>` elements |
| AutoTrader   | unhydrated   | unhydrated (still)        | `settle_ms=5000` not enough; still 0 hrefs |
| Craigslist   | inconsistent | 17 listing URLs found     | `settle_ms=5000` fixed the hydration variance |
| BaT          | works, all Civic | works, varied make/model | name-parse fix delivered: CRX, Del Sol, Civic Type R now distinct |
| Hemmings     | 403          | 200 (curl_cffi), redirected to category page | curl_cffi cleared block; site dropped `?Make=&Model=` query on redirect |
| Cars & Bids  | 403          | 403 (still)               | Unchanged — still Playwright; Cloudflare still blocks |

### Run summary

- Total elapsed: **186.4 s** (vs 137.2 s in run 1 — slower due to the
  longer Playwright settle and Hemmings now successfully fetching).
- Crawler `exit_reason`: **`max_items_reached`** (hit the 20-item cap
  again; 46 items remain pending: 14 BaT listings, 14 BaT images, 17
  Craigslist listings, 1 Hemmings pagination URL).
- Total fetches: **19 succeeded / 1 failed** (Cars & Bids was the only
  failure; first run had 3 failures).
- Listings inserted: **14** (all BaT — same count as first run, but
  this run hit `max_items` mid-queue, with Craigslist listings still
  pending).
- URLs enqueued: **60** (vs 42 first run — gain comes from the 17 new
  Craigslist listings and 1 Hemmings pagination URL).

### Per-site results

#### cars.com — curl_cffi cleared the 403, parser hit a different bug
- Search page: **HTTP 200 via curl_cffi** (1.02 MB of real HTML —
  `<title>` reads "New and Used 2020 Honda Civic for Sale Near San
  Mateo, CA | Cars.com"; 33 `vehicle-card` markers, 38
  `data-listing-id` markers, 95 occurrences of `/vehicledetail/` in
  the source).
- Listing URLs discovered: **0** (parser bug — see below)
- Listings parsed: 0
- Image URLs enqueued: 0
- **Headline**: curl_cffi cleared the Cloudflare block. The TLS
  fingerprint fix works against cars.com.
- **New bug**: the parser's `_LISTING_HREF_RE = ^/vehicledetail/[^/]+/?$`
  expects path-relative hrefs, but BeautifulSoup yields hrefs from
  `<a>` tags only, and those are now **absolute URLs**
  (`https://www.cars.com/vehicledetail/...`). The path-relative
  `/vehicledetail/...` versions in the source live in custom
  `<spark-link-button>` / `<spark-button>` elements that
  `soup.find_all("a", href=True)` doesn't pick up. Fix options:
  (a) broaden the selector to include `spark-link-button`, or
  (b) loosen the regex to match the absolute form, or
  (c) match on URL substring `/vehicledetail/` regardless of element
  tag.

#### AutoTrader — unhydrated shell, settle_ms=5000 didn't help
- Search page: **HTTP 200**, parser logged the same "no listing cards
  found on search page; selectors may need updating" note.
- Listing URLs discovered: 0
- Listings parsed: 0
- **Conclusion**: bumping `settle_ms` from 1500 to 5000 was
  insufficient. AutoTrader still serves an interstitial / unhydrated
  shell to headless Chromium. Next escalation: try `networkidle`, or
  `wait_for_selector` on a specific listing-grid marker, or route
  AutoTrader through curl_cffi the way we did cars.com.

#### Craigslist — settle_ms=5000 fixed the hydration variance
- Search page: **HTTP 200**, parser extracted 17 listing URLs.
- Listing URLs discovered: **17** (all `pending` — the run hit
  `max_items` before processing them).
- Listings parsed: 0 (queue cap, not a parser issue).
- **Headline**: the longer settle delivered consistent hydration.
  First-run variance (155 KB vs 35 KB depending on timing) is gone.
  Listing extraction now works; we just need more headroom to parse
  them.

#### Bring a Trailer — name-parse fix delivered varied makes/models
- Search page: **HTTP 200**, 28 listing URLs discovered (same as run 1).
- Listings parsed: **14** with **distinct models**:

| listing_id                            | year | make  | model                  |
|---------------------------------------|------|-------|------------------------|
| bat:1986-honda-civic-22               | 1986 | Honda | Civic Si               |
| bat:1988-honda-crx-si-33              | 1988 | Honda | Crx Si 5-Speed         |
| bat:1989-honda-crx-si-48              | 1989 | Honda | Crx Si 5-Speed         |
| bat:1991-honda-crx-89                 | 1991 | Honda | Crx 1.6I-Vt 5-Speed    |
| bat:1991-honda-crx-hf-4               | 1991 | Honda | Crx Hf 5-Speed         |
| bat:1991-honda-crx-si-14              | 1991 | Honda | Crx Si 5-Speed         |
| bat:1993-honda-civic-del-sol-37       | 1993 | Honda | Civic Del Sol Si       |
| bat:1993-honda-civic-del-sol-40       | 1993 | Honda | Civic Del Sol S        |
| bat:1994-honda-civic-del-sol-5        | 1994 | Honda | Civic Del Sol S        |
| bat:1995-honda-civic-5                | 1995 | Honda | Civic Cx Hatchback 5-Speed |
| bat:1997-honda-del-sol-3              | 1997 | Honda | Del Sol Si 5-Speed     |
| bat:2000-honda-civic-81               | 2000 | Honda | Civic Type Rx          |
| bat:2018-honda-civic-type-r-touring-5 | 2018 | Honda | Civic Type R Touring   |
| bat:2018-honda-civic-type-r-touring-7 | 2018 | Honda | Civic Type R Touring   |

- **Headline**: the JSON-LD `name`-parse fix worked. We no longer
  collapse every BaT listing to "Honda Civic". CRX (5 listings),
  Del Sol (4 listings), Civic Si (1), Civic Type R (2), Civic Type Rx
  (1), Civic CX Hatchback (1) are now distinct.
- **Known residual**: `model` is greedy — it absorbs trim and
  transmission tokens too (e.g. `Crx Si 5-Speed`, `Civic Cx Hatchback
  5-Speed`, `Civic Del Sol Si`). The `trim` column is still `None`
  for all rows. The name parser's token loop should stop earlier
  (likely at the first transmission / configuration token like
  "5-Speed", "Manual", "Coupe", "Hatchback") and stash the rest in
  `trim`. Lower-priority follow-up — model is at least correct in its
  prefix and the dataset is now usable for non-Civic targets.

#### Hemmings — curl_cffi cleared the 403, but Hemmings dropped the query
- Search page: **HTTP 200 via curl_cffi**, but final URL is
  `https://www.hemmings.com/classifieds/cars-for-sale` — the
  `?Make=honda&Model=civic` query params were stripped on a redirect.
  Result: 506 KB of HTML, but it's a generic "browse by category"
  landing page with 0 numeric-id listing hrefs.
- Listing URLs discovered: 0
- Listings parsed: 0
- **Status**: half-win — Cloudflare no longer blocks us, but Hemmings
  appears to redirect `?Make=...&Model=...` to the bare category page
  when the request doesn't carry the right cookies / session state.
  The parser DID find a pagination URL (`?page=2`), which is why
  there's a stray `pending` Hemmings search row in the queue.
- **Fix options**: (a) build the query URL differently (Hemmings may
  use a slug-style path like
  `/classifieds/cars-for-sale/honda/civic`), (b) follow up on the
  bare `cars-for-sale?page=2` listing-card extraction (it's a
  category page that may still have car cards).

#### Cars & Bids — still 403, unchanged
- Search page: **HTTP 403** (same as run 1; still on Playwright).
- This was an explicit non-goal of run 2 (the fix routed only
  cars.com and hemmings through curl_cffi). Next iteration could try
  routing Cars & Bids through curl_cffi too, but Cars & Bids ships a
  React SPA — even a clean fetch may give a near-empty shell.

### Most surprising finding

**cars.com curl_cffi worked, but the parser couldn't see the hrefs
because they live in `<spark-link-button>` custom elements, not
`<a>` tags.** That's a brand-new failure mode we didn't anticipate in
run 1 (when 403s hid it). The Cloudflare bypass exposed a
selector-incompleteness bug under it. cars.com is fixable in one
small parser PR.

The BaT fix overshoot — `model` absorbing trim words like
"5-Speed" / "Hatchback" — is a close runner-up. Annoying but the
data is still usable.

### Updated follow-up tasks

In rough priority order (replaces / amends list 1-8 above):

1. **cars.com parser: broaden listing-href selector.** Don't restrict
   to `<a href="...">`; match any element with an href / link attr
   whose value contains `/vehicledetail/`, OR loosen the regex to
   match both relative and absolute. The Cloudflare gate is solved;
   the parser is now the limiting factor.

2. **BaT name-parse: stop the model run at transmission /
   configuration tokens.** Add stop-words like `5-Speed`, `6-Speed`,
   `Manual`, `Automatic`, `Coupe`, `Hatchback`, `Sedan`, `Convertible`
   to the existing termination set, and route the leftover tail into
   `trim`. Low risk — covered by the existing BaT parser tests.

3. **Hemmings query-form rewrite.** The `?Make=honda&Model=civic`
   form redirects to the bare category page. Try the slug form
   (`/classifieds/cars-for-sale/honda/civic` or similar) instead;
   inspect what the site's own search UI actually links to.

4. **AutoTrader: escalate hydration fix.** `settle_ms=5000` didn't
   help. Try `wait_until="networkidle"`, `wait_for_selector` on a
   specific marker, or route through curl_cffi.

5. **Craigslist listing parsing.** Search now yields 17 URLs.
   Pending queue items prove fetch works; need a larger `max_items`
   to actually exercise the listing parser. Re-run with
   `max_items=40` to test.

6. **Cars & Bids fallback.** Try curl_cffi routing (would need to
   confirm the parser can work on the unhydrated shell, since C&B is
   a SPA).

7. Existing items 7 (BaT mileage/VIN from prose) and the design
   questions remain unchanged.

### Pytest / lint

- `ruff check .` — All checks passed.
- `pytest -v` — `297 passed in 14.63s` (no regressions from the
  fixes).
