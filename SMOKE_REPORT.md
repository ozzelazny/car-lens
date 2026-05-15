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

## Third run — 2026-05-15 (after polish + stealth upgrades)

Third smoke run after three follow-up fixes:

- `1e3a279` — BaT trim split into separate column + cars.com
  `<spark-link-button>` href support + Hemmings slug URL form.
- `a62390c` — `PlaywrightFetcher.wait_for_selector_by_source`; smoke
  harness pre-configures AutoTrader with an inventory-grid selector
  list.
- `f84ccb9` — `playwright-stealth` 2.x + Chromium automation-flag
  disable + `navigator.webdriver` override (targeting the Cars & Bids
  Cloudflare block).

### Status changes vs run 2

| Site         | Run 2                       | Run 3                       | Notes |
|--------------|-----------------------------|-----------------------------|-------|
| cars.com     | 200 + 0 listings            | **403 (regressed)**         | curl_cffi now blocked at the network layer where it cleared in run 2; spark-link-button code path never exercised |
| AutoTrader   | unhydrated 4 KB shell       | **200 + still 0 listings**  | wait_for_selector timed out (10 s) but search completed; parser still found no cards |
| Craigslist   | 17 listing URLs (all pending) | **17 listing URLs (all pending)** | unchanged config; consistent |
| BaT          | 14 listings, model absorbed trim | **14 listings, trim populated for most rows** | trim split fix delivered; one residual case ("Del Sol") split wrong |
| Hemmings     | 200 + redirected to bare category | **200 + still 0 listings** | slug URL form is reachable but the parser still finds 0 listing cards on the rendered HTML |
| Cars & Bids  | 403                         | **403 (still)**             | stealth 2.x + automation-flag disable was not enough |

### Run summary

- Total elapsed: **197.2 s** (vs 186.4 s in run 2 — almost identical;
  the 10 s AutoTrader selector wait + Playwright init eat the slack).
- Crawler `exit_reason`: **`max_items_reached`** (hit the 20-item cap
  again; 46 items remain pending: 14 BaT listings, 14 BaT images, 17
  Craigslist listings, 1 Hemmings pagination URL).
- Total fetches: **18 succeeded / 2 failed** (cars.com 403,
  Cars & Bids 403).
- Listings inserted: **14** (all BaT, same as run 2 — `max_items`
  again starves the Craigslist queue).
- URLs enqueued: **60** (28 BaT listings + 14 BaT images + 17
  Craigslist listings + 1 Hemmings pagination URL — identical to
  run 2).

### Per-site results

#### cars.com — regressed to 403

- Search page: **HTTP 403 via curl_cffi**. This is a regression from
  run 2 (which got 200). The `<spark-link-button>` parser support
  shipped in `1e3a279` was therefore not exercised at all.
- Listing URLs discovered: 0
- Listings parsed: 0
- **Status**: Cloudflare appears to have either (a) tightened
  fingerprinting against curl_cffi's Chrome 131 impersonation since
  run 2, or (b) IP-rate-limited / soft-banned the WSL2 egress after
  multiple back-to-back smoke runs. Either way, the parser-side fix
  in `1e3a279` is now untestable end-to-end. We need to either
  validate it against a saved fixture or wait out the rate-limit.

#### AutoTrader — wait_for_selector timed out but search completed

- Search page: **HTTP 200**. Browser warning:
  `wait_for_selector("[data-cmp='inventoryListing'], ...") timed out`
  after 10 s (the default `selector_timeout_ms`). After the timeout
  the fetcher returned whatever HTML was rendered.
- Listing URLs discovered: 0
- Listings parsed: 0
- Parser note: `"no listing cards found on search page; selectors
  may need updating"` (same as run 2).
- **Conclusion**: none of the four `data-cmp` / `data-qa` / class
  selectors we configured matched anything on AutoTrader's actual
  rendered DOM. Either (a) AutoTrader doesn't use those attribute
  names anymore, (b) the page is still serving an interstitial,
  rendering nothing under those selectors at all, or (c) we need a
  selector that matches whatever AutoTrader's current grid wrapper is
  called. Next step is to capture the rendered HTML for live
  inspection (e.g. dump the response body to disk and grep for
  candidate attributes).

#### Craigslist — same as run 2, queue cap still starves listing fetches

- Search page: **HTTP 200**, 17 listing URLs extracted (identical to
  run 2 — same listings, same order; the site state hasn't churned).
- Listing URLs discovered: **17** (all `pending` again — `max_items`
  fills up on BaT before any Craigslist listing pages get fetched).
- Listings parsed: 0 (queue cap, not a parser issue).
- **Action**: this is now a smoke harness issue — `max_items=20` is
  too small to exercise the Craigslist listing parser when BaT
  alone yields 28 candidates. Either bump `max_items` to 40+, or
  add a per-source cap to the smoke runner.

#### Bring a Trailer — trim split fix delivered for most rows

- Search page: **HTTP 200**, 28 listing URLs discovered.
- Listings parsed: **14** with trim split into its own column:

| listing_id                            | year | make  | model      | trim                  |
|---------------------------------------|------|-------|------------|-----------------------|
| bat:1986-honda-civic-22               | 1986 | Honda | Civic      | Si                    |
| bat:1988-honda-crx-si-33              | 1988 | Honda | Crx        | Si 5-Speed            |
| bat:1989-honda-crx-si-48              | 1989 | Honda | Crx        | Si 5-Speed            |
| bat:1991-honda-crx-89                 | 1991 | Honda | Crx        | 1.6I-Vt 5-Speed       |
| bat:1991-honda-crx-hf-4               | 1991 | Honda | Crx        | Hf 5-Speed            |
| bat:1991-honda-crx-si-14              | 1991 | Honda | Crx        | Si 5-Speed            |
| bat:1993-honda-civic-del-sol-37       | 1993 | Honda | Civic      | Del Sol Si 5-Speed    |
| bat:1993-honda-civic-del-sol-40       | 1993 | Honda | Civic      | Del Sol S 5-Speed     |
| bat:1994-honda-civic-del-sol-5        | 1994 | Honda | Civic      | Del Sol S 5-Speed     |
| bat:1995-honda-civic-5                | 1995 | Honda | Civic      | Cx Hatchback 5-Speed  |
| bat:1997-honda-del-sol-3              | 1997 | Honda | **Del**    | **Sol Si 5-Speed**    |
| bat:2000-honda-civic-81               | 2000 | Honda | Civic      | Type Rx               |
| bat:2018-honda-civic-type-r-touring-5 | 2018 | Honda | Civic      | Type R Touring        |
| bat:2018-honda-civic-type-r-touring-7 | 2018 | Honda | Civic      | Type R Touring        |

- **Headline**: the trim split worked. `trim` is now populated for
  every row that had a multi-token tail. Single-token models (Civic,
  Crx) correctly receive their non-model words as trim instead of
  swallowing them into `model`.
- **Headline win — single-word models stay single-word**:
  - Civic stays "Civic" (trim "Si" or "Type R Touring" etc.)
  - Crx stays "Crx" (trim "Si 5-Speed", "Hf 5-Speed", etc.)
- **Residual bug**: row `bat:1997-honda-del-sol-3` shows
  `model="Del", trim="Sol Si 5-Speed"`. The parser doesn't recognise
  "Del Sol" as a two-word model. The other Del Sol rows
  (1993, 1994) carried the prefix `Civic` (BaT named them
  `"... Civic Del Sol Si"`) so they got `model=Civic,
  trim="Del Sol ..."` — also not perfect, but at least `model`
  is the more-famous name. A two-word-model lookup (or a small
  hand-maintained alias list) would fix both.
- Trim values for `1991-honda-crx-89` show `1.6I-Vt` — that's the
  BaT name field's title-casing artifact (originally `1.6i-VT`).
  Cosmetic; downstream consumers can normalise.

#### Hemmings — slug URL form reachable, still 0 listings parsed

- Search page: **HTTP 200 via curl_cffi**. The slug URL form
  (`/classifieds/cars-for-sale?Make=honda&Model=civic`) returned a
  full page.
- Listing URLs discovered: 0
- Listings parsed: 0
- Parser note: `"no listing cards found on search page; selectors
  may need updating"`.
- The parser DID find a pagination URL (`?page=2` — still pending in
  the queue), so it's not pure noise — but the listing-card
  selector still doesn't match anything on the page.
- **Action**: capture and inspect the response body; the slug-URL
  fix gets us a real page back, but the parser's listing-card
  selectors are still wrong for the rendered HTML.

#### Cars & Bids — still 403, stealth 2.x didn't bypass

- Search page: **HTTP 403**.
- The stealth upgrade (playwright-stealth 2.x + automation-flag
  disable + `navigator.webdriver` override) was not sufficient.
  Cloudflare's challenge is fingerprinting at a deeper level —
  likely TLS / JA3 or Canvas / WebGL fingerprint — that
  playwright-stealth doesn't paper over.
- **Status**: this remains uncrawlable from headless Chromium.
  Next escalations (in order of effort):
  1. Try routing C&B through curl_cffi too. C&B is a React SPA so
     curl_cffi will only see the unhydrated shell; some C&B
     listing pages may still expose initial state in a JSON blob
     (`__NEXT_DATA__` style), in which case the parser could read
     from that.
  2. Try a non-headless Chromium with `--use-fake-ui-for-media-stream`
     and a real Canvas / WebGL fingerprint.
  3. Accept the gap; document; move on.

### Identified follow-up tasks

In rough priority order:

1. **AutoTrader: figure out the real listing-grid selector.** None
   of the four candidates we tried (`[data-cmp='inventoryListing']`,
   `[data-cmp='inventoryListingItem']`, `.inventory-listing`,
   `[data-qa='listing-card']`) matched. Need to dump the rendered
   HTML to disk and grep / inspect to find the actual wrapper class
   name. This is now blocking AutoTrader entirely.

2. **Hemmings: figure out the real listing-card selector.** The
   slug-URL fix gets us a 200 with content, but the parser's
   selectors still don't match. Same triage as AutoTrader — capture
   and inspect.

3. **cars.com 403 regression: investigate / wait out.** The fix
   from `1e3a279` is untested end-to-end. Either rotate egress IP,
   wait 24 h, or build a parser-unit-test fixture from the run-2
   HTML capture (if we still have it) to validate the
   spark-link-button code path without a live fetch.

4. **BaT two-word-model recognition.** "Del Sol" splits as
   `model=Del, trim=Sol ...`. Either (a) add a small alias list
   (`Del Sol`, `Type R`, etc. — already special-cased per the Type R
   rows, so the mechanism exists), or (b) use a real per-make model
   catalog. Low priority; data is still usable.

5. **Smoke harness: raise `max_items` or add per-source caps.**
   With BaT yielding 28 listing URLs, the 20-item cap blocks every
   other site's listing parser. A `max_items=50` run would
   exercise Craigslist's listing parser at least.

6. **Cars & Bids: try curl_cffi + JSON-blob extraction.** The
   stealth approach has hit diminishing returns. C&B ships a Next.js
   app; the initial HTML often contains `__NEXT_DATA__` with the
   full listing payload. Worth a parser experiment even if the page
   is otherwise empty.

7. Existing items (BaT mileage/VIN from prose; fetcher fallback
   chain) remain unchanged from prior runs.

### Open questions

- **Are we OK with the smoke run being non-deterministic re: 403s?**
  cars.com gave 200 in run 2 and 403 in run 3, with no parser-side
  change to either site. If Cloudflare is gating us based on recent
  request volume, every smoke run is gambling against a temporary
  rate-limit. Options: (a) accept the variance and re-run, (b) move
  to fixture-based parser tests so smokes only validate
  integration / network paths, (c) add a backoff between smokes.
- **Should the AutoTrader / Hemmings selector debugging happen in a
  one-off "rendered HTML dump" script** (cheaper, focused), or
  through the smoke run plus a `--save-html` flag (more general,
  more code)?
- **Is it worth one more attempt at Cars & Bids** (curl_cffi +
  `__NEXT_DATA__` extraction), or should we mark C&B as accepted-gap
  and move on?

### Pytest / lint

- `ruff check .` — All checks passed.
- `pytest` — `316 passed in 15.20s` (no regressions from the
  fixes; +19 new tests since run 2 covering BaT trim split,
  spark-link-button hrefs, slug URL form, and `wait_for_selector_by_source`).

## Fourth run — 2026-05-15 (after real-HTML parser fixes)

Fourth smoke run after the real-HTML parser fix loop:

- `12f8cc6` — block diagnostic; saved real production HTML fixtures
  for cars.com (via `curl_cffi(firefox133)`) and Hemmings (via
  `curl_cffi(chrome131)`); confirmed both endpoints are reachable
  network-side.
- `b0e29ac` — cars.com and Hemmings parsers rewritten against the
  saved real-HTML fixtures. Unit tests assert cars.com extracts ≥27
  distinct listing URLs from the saved page and Hemmings extracts
  ≥7 (deduped) listing URLs from the saved page.

Goal of this run: confirm the parser fixes hold end-to-end against
live HTML (i.e., not just the saved fixtures).

### Status changes vs run 3

| Site         | Run 3                       | Run 4                       | Notes |
|--------------|-----------------------------|-----------------------------|-------|
| cars.com     | 403 (regressed)             | **200 + 19 listing URLs enqueued, 0 followed up** | Network access restored AND the rewritten selector now extracts production listings. The 19 listings stay `pending` because `max_items=20` is consumed by BaT first. Headline win. |
| AutoTrader   | 200 + 0 listings            | **200 + 0 listings (still)** | Identical to run 3: 10 s `wait_for_selector` timeout, then 0 cards. Still need the real grid selector. |
| Craigslist   | 17 listing URLs (all pending) | **17 listing URLs (all pending)** | Unchanged; same `max_items` starvation. |
| BaT          | 14 listings                 | **14 listings**             | Identical (same site state). |
| Hemmings     | 200 + 0 listings            | **200 + 0 listings (still)** | The parser passes the real-HTML fixture (slug URL `/cars-for-sale/honda/civic`), but the smoke seed still uses the **query-param URL** `?Make=honda&Model=civic`. That URL appears to redirect to a different (empty / category) HTML shape that the rewritten selectors do not match. Smoke seed URL is now misaligned with the parser's real-HTML contract. |
| Cars & Bids  | 403                         | **403 (still)**             | Unchanged from runs 2 & 3. |

### Run summary

- Total elapsed: **196.2 s** (essentially identical to runs 2 & 3 —
  the 10 s AutoTrader selector wait + BaT page fetches still
  dominate).
- Crawler `exit_reason`: **`max_items_reached`** (hit the 20-item
  cap again; 65 items remain pending: 19 cars.com listings, 14 BaT
  listings, 14 BaT images, 17 Craigslist listings, 1 Hemmings
  pagination URL).
- Total fetches: **19 succeeded / 1 failed** (Cars & Bids 403 only —
  cars.com is back to 200 this run).
- Listings inserted: **14** (all BaT, same as runs 2 & 3 — the
  cars.com / Craigslist listing parsers were not exercised due to
  `max_items`).
- URLs enqueued: **79** (vs 60 in run 3). The +19 is the cars.com
  listing URLs that the rewritten parser now extracts — exactly the
  point of the run.

### Per-site results

#### cars.com — fixed (HTTP 200 + 19 listing URLs extracted)

- Search page: **HTTP 200 via curl_cffi(chrome131)**. The run-3
  regression cleared — we are not currently rate-limited.
- Listing URLs discovered: **19** (all `pending`; none reached the
  listing-parser stage because BaT consumed the `max_items=20` quota
  first).
- Listings parsed: 0 (queue cap, not a parser issue; the search
  parser worked).
- Sample URLs (all `/vehicledetail/<uuid>/` form, which the new
  selector matches):
  - `https://www.cars.com/vehicledetail/3715142b-250e-4689-a303-e5924eb2ceaa/`
  - `https://www.cars.com/vehicledetail/275e9719-3511-47e3-bc70-74c58a611288/`
  - `https://www.cars.com/vehicledetail/77a5d51d-e834-4d6d-a3eb-7e6b8ed09548/`
- **Headline**: the real-HTML fixture rewrite from `b0e29ac` works
  against live HTML. Production filter (`year_min=2020&year_max=2020`)
  narrows the count to 19 vs the 27 in the fixture (which was saved
  with no year filter) — but the parser is finding the cards. Next
  step is to actually fetch one of the listing pages, which is
  blocked only by `max_items`.

#### AutoTrader — unchanged (200 + 0 listings)

- Search page: **HTTP 200**, but
  `wait_for_selector("[data-cmp='inventoryListing'], ...")` timed
  out at 10 s (identical to run 3).
- Listing URLs discovered: 0.
- Parser note: `"no listing cards found on search page; selectors
  may need updating"`.
- **Conclusion**: no progress vs run 3. None of the four configured
  selectors match the rendered DOM. AutoTrader needs the same
  diagnostic + real-HTML fixture treatment that cars.com and Hemmings
  just received — the `12f8cc6` diagnostic only fetched
  `/robots.txt` for AutoTrader (because the search page is Akamai-
  blocked from curl_cffi). We need a Playwright-side capture of
  whatever HTML it eventually rendered, or evidence of an
  interstitial.

#### Craigslist — unchanged (17 listing URLs, all pending)

- Search page: **HTTP 200**, 17 listing URLs extracted (identical
  to runs 2 & 3 — same listings, same order).
- Listings parsed: 0 (queue cap, not a parser issue).
- **Action**: same as runs 2 & 3 — `max_items=20` is too small for
  five sites to share when BaT alone yields 28 candidates. With
  cars.com now contributing 19, the cap is even more biting.

#### Bring a Trailer — unchanged (14 listings)

- Search page: **HTTP 200**, 28 listing URLs discovered.
- Listings parsed: **14** with identical rows to run 3 (same site
  state — same auctions still active).
- Residual "Del Sol" model-split bug is still present
  (`bat:1997-honda-del-sol-3` shows `model="Del", trim="Sol Si
  5-Speed"`); the other Del Sol rows still carry the
  `model=Civic, trim="Del Sol ..."` pattern. Same as run 3.

#### Hemmings — still 0 listings end-to-end (URL mismatch)

- Search page: **HTTP 200 via curl_cffi(chrome131)**.
- Listing URLs discovered: 0.
- Listings parsed: 0.
- Parser note: `"no listing cards found on search page; selectors
  may need updating"`.
- A pagination URL (`?page=2`) is still discovered (same as run 3),
  so the parser is reaching *something* — just not listing cards.
- **Root cause** (newly identified this run): the smoke harness
  seeds Hemmings with the **query-param URL form**
  (`/cars-for-sale?Make=honda&Model=civic`), but the real-HTML
  fixture for the parser was saved from the **slug URL form**
  (`/cars-for-sale/honda/civic`). The two URLs return different
  HTML shapes. The parser is now contractually correct against the
  slug-URL shape (7 listings in the fixture, asserted by unit test),
  but the smoke seed asks for the other shape. Either (a) align the
  smoke seed to the slug URL, or (b) teach the Hemmings parser to
  cope with both shapes.

#### Cars & Bids — unchanged (HTTP 403)

- Search page: **HTTP 403** via Playwright (stealth still active).
- Status unchanged from runs 2 & 3. No new attempts this run.

### Identified follow-up tasks

In rough priority order:

1. **Hemmings: align smoke seed URL with parser fixture.** Change
   the smoke seed from `/cars-for-sale?Make=honda&Model=civic`
   (query-param) to `/cars-for-sale/honda/civic` (slug). The parser
   already passes the real-HTML fixture for the slug form. This is
   a one-line change in `scripts/smoke_e2e.py`.

2. **Raise `max_items` (or add per-source caps).** With cars.com now
   contributing 19 listing URLs on top of BaT's 28, the 20-item cap
   blocks every cars.com / Craigslist / Hemmings listing fetch. A
   `max_items=80+` run, or per-source caps (e.g., 20 per source),
   would actually exercise the listing parsers we just shipped.

3. **AutoTrader: capture real HTML and rewrite selectors.** Same
   treatment cars.com and Hemmings just got. The diagnostic should
   save a real Playwright-rendered AutoTrader HTML body and we
   rewrite the selectors against it (or confirm it's an interstitial
   /CAPTCHA and accept the gap).

4. **BaT two-word-model recognition.** Unchanged from run 3 —
   "Del Sol" still splits as `model=Del`. Low priority; data
   is still usable.

5. **Cars & Bids: try curl_cffi + `__NEXT_DATA__` extraction.**
   Unchanged from run 3.

### Open questions

- **Is the `max_items=20` cap intentional for smoke?** It now
  consistently starves cars.com and Craigslist (and would starve
  Hemmings once that seed is fixed). Bumping it would meaningfully
  improve smoke coverage. (Run 3 also flagged this.)
- **Should the Hemmings seed switch be a one-line fix in this run,
  or scheduled as a discrete task?** The fix is trivial but cosmetic
  to the smoke contract — the parser is already correct.

### Pytest / lint

- `ruff check .` — All checks passed.
- `pytest` — `360 passed in 15.45s` (+44 new tests since run 3;
  these cover the real-HTML fixtures for cars.com and Hemmings, the
  diagnostic / block characterization, and the parser rewrites from
  `b0e29ac`).
