# Block Diagnostic — 20260515T205423Z

Systematic probe of the four currently-failing crawler sources (cars.com,
AutoTrader, Hemmings, Cars & Bids) using multiple fetcher / impersonation
combinations and alternative endpoints. Goal: characterise the blocker
class per site so a follow-up parser-fix task can pick the right approach.

## Setup

- Run timestamp: `20260515T205423Z` UTC
- Min gap between requests to the same host: `7.0s`
- Max total requests cap: `30`
- Total probe requests issued: `24` (plus fixture re-fetches for usable
  probes)
- Fixtures dropped under: `tests/crawler/parsers/fixtures/real_world/`
- Note on `cloudflare_challenge` tags: the initial run flagged Hemmings
  search via `challenge-platform` substring, but inspection showed it was
  the Cloudflare JS beacon embedded in a normal 319KB page — not an
  interstitial. The blocker heuristic in `scripts/diagnose_blocks.py` has
  been tightened (removed `challenge-platform` and bare `Cloudflare Ray ID`
  from the strict-challenge marker set) and the Hemmings search fixture
  has been saved manually. The matrix below reflects the corrected
  interpretation.

## Blocker legend

- `http_403` / `http_404` / `http_5xx` — server returned that HTTP status.
- `cloudflare_challenge` — interstitial challenge page (e.g. "Just a
  moment...", `cf-error-details`). NOT the embedded CF JS beacon.
- `unhydrated_shell` — HTTP 200 but <8 KB; SPA hasn't hydrated.
- `content_markers:<...>` — recognised listing-shape substrings present.
- `parser_mismatch_candidate` — HTTP 200, non-trivial size, but no
  recognised listing markers; either parser-selector mismatch or alt
  content shape.

## cars_com

| probe | fetcher | status | bytes | blockers | saved |
|---|---|---|---|---|---|
| `search_playwright` | playwright | 403 | 0 | http_403 | — |
| `search_curlcffi_chrome131` | curl_cffi(chrome131) | 200 | 1,278,024 | content_markers:vehicle-card,/vehicledetail/,data-listing-id | (false-pos CF tag in v1; re-eval below) |
| `search_curlcffi_chrome120` | curl_cffi(chrome120) | 403 | 0 | http_403 | — |
| `search_curlcffi_firefox133` | curl_cffi(firefox133) | 200 | 1,220,733 | content_markers:vehicle-card,/vehicledetail/,data-listing-id | `cars_com_search_curlcffi_firefox_20260515T205423Z.html` |
| `sitemap_curlcffi_chrome131` | curl_cffi(chrome131) | 404 | 0 | http_404 | — |
| `robots_curlcffi_chrome131` | curl_cffi(chrome131) | 200 | 4,280 | content_markers:/vehicledetail/ | `cars_com_robots_curlcffi_chrome131_20260515T205423Z.html` *(will save in next run with updated heuristic)* |
| `listing_curlcffi_chrome131` | curl_cffi(chrome131) | 200 | 71,235 | content_markers:/vehicledetail/ | (404-shell; cars.com returns 200 + "page unavailable") |

Notes:
- The `chrome131` and `firefox133` profiles BOTH cleared cars.com's
  Cloudflare gate and returned the full ~1.2 MB SSR'd results page. The
  `chrome120` profile got a 403 — TLS-fingerprint sensitivity is fine-grained.
- The `search_curlcffi_chrome131` probe was originally flagged
  `cloudflare_challenge` due to the `challenge-platform` beacon path; the
  page itself was a fully-rendered 1.28 MB listings page with
  `vehicle-card` markers. The chrome131 search response is just as
  useful as the firefox133 one; saving the firefox133 fixture is fine.
- `sitemap.xml` is a 404 on cars.com — they don't expose one at the root.
- `/vehicledetail/123456789/` returned a 71 KB "page unavailable" shell
  rather than a 404 — useful signal that the listing-page surface IS
  accessible via curl_cffi.

### Recommendation

**Route `cars_com` through `CurlCffiFetcher(impersonate="chrome131")` or
`"firefox133"`.** Both impersonation profiles consistently bypass
Cloudflare from this vantage point and return the full SSR'd listings page
with `vehicle-card` / `/vehicledetail/` / `data-listing-id` markers. Avoid
`chrome120` (403). The current parser (`cars_com.py`) should now have
enough HTML to work against; verify against the saved fixture.

## autotrader

| probe | fetcher | status | bytes | blockers | saved |
|---|---|---|---|---|---|
| `search_playwright` | playwright | 200 | 4,443 | unhydrated_shell | — |
| `search_curlcffi_chrome131` | curl_cffi(chrome131) | 200 | 3,762 | unhydrated_shell (Akamai "page unavailable") | — |
| `search_curlcffi_chrome120` | curl_cffi(chrome120) | 200 | 3,761 | unhydrated_shell | — |
| `sitemap_curlcffi_chrome131` | curl_cffi(chrome131) | 200 | 696 | unhydrated_shell (sitemap index) | — |
| `robots_curlcffi_chrome131` | curl_cffi(chrome131) | 200 | 17,582 | content_markers:/cars-for-sale/vehicledetails/ | `autotrader_robots_curlcffi_chrome131_20260515T205423Z.html` |
| `listing_curlcffi_chrome131` | curl_cffi(chrome131) | 200 | 3,762 | unhydrated_shell (Akamai "page unavailable") | — |

Notes:
- Every non-meta probe returns a ~3.8 KB Akamai-served "Autotrader - page
  unavailable" interstitial — different from cars.com's Cloudflare gate
  but functionally equivalent at this IP. The first-200-char dumps
  confirm `<title>Autotrader - page unavailable</title>`.
- The sitemap returns a valid 696-byte sitemap *index* pointing to
  `sitemap_main.xml` — too small to flag as content but technically
  reachable. A follow-up could deepen this by walking the index.
- robots.txt is fully reachable (17 KB).

### Recommendation

**AutoTrader is the hardest of the four — every dynamic surface
(search and listing) is fronted by an Akamai bot-mitigation interstitial
that neither Playwright+stealth nor curl_cffi(chrome131/120) clears from
this IP.** Recommendations, in order of effort:
1. **Walk the sitemap tree** (sitemap.xml -> sitemap_main.xml -> per-section
   sitemaps) to harvest vehicledetail URLs without touching the search
   surface. This is the lowest-effort path that may work without proxies.
2. Try additional curl_cffi impersonation profiles (safari, edge) — TLS-fingerprint
   sensitivity is fine-grained, as cars.com demonstrated (chrome131 OK,
   chrome120 403).
3. Failing the above, AutoTrader requires a residential proxy.

## hemmings

| probe | fetcher | status | bytes | blockers | saved |
|---|---|---|---|---|---|
| `search_playwright` | playwright | 403 | 0 | http_403 | — |
| `search_curlcffi_chrome131` | curl_cffi(chrome131) | 200 | 319,038 | content_markers:listing-card | `hemmings_search_curlcffi_chrome131_20260515T205423Z.html` |
| `search_curlcffi_chrome120` | curl_cffi(chrome120) | 403 | 0 | http_403 | — |
| `sitemap_curlcffi_chrome131` | curl_cffi(chrome131) | 200 | 1,016 | (sitemap index) | (small) |
| `listing_curlcffi_chrome131` | curl_cffi(chrome131) | 404 | 0 | http_404 | (dead id) |

Notes:
- The Hemmings search probe via `curl_cffi(chrome131)` returns 319 KB
  containing 2 `listing-card` occurrences and the full Livewire-rendered
  classifieds shell. The initial blocker tag of `cloudflare_challenge`
  was a **false positive** — the marker was the `cdn-cgi/challenge-platform`
  beacon script path embedded in a normal page, not an interstitial. The
  fixture has been saved.
- However: only 2 `listing-card` occurrences were present despite the
  query being for Honda Civic. The page may be a "no results" state for
  that exact slug, or Hemmings only loads cards on scroll. Worth a
  follow-up probe with a broader slug (e.g. just `/cars-for-sale/honda/`
  or a higher-volume make/model combo) to confirm card density.
- `/classifieds/dealer/honda/civic/2845391/` returned a 404 — the ID was
  a guess. The shape of the URL is correct; future probes should use a
  real ID harvested from the search-page output.

### Recommendation

**Route `hemmings` through `CurlCffiFetcher(impersonate="chrome131")`.**
The slug URL works; the parser needs new selectors against the Livewire-
rendered output. The 319 KB fixture under
`hemmings_search_curlcffi_chrome131_20260515T205423Z.html` should drive
the parser fix. As a low-priority follow-up: confirm card density on a
broader/different slug.

## carsandbids

| probe | fetcher | status | bytes | blockers | saved |
|---|---|---|---|---|---|
| `search_playwright` | playwright | 403 | 0 | http_403 | — |
| `search_curlcffi_chrome131` | curl_cffi(chrome131) | 200 | 13,052 | parser_mismatch_candidate (Next.js SPA shell) | `carsandbids_search_curlcffi_chrome131_20260515T205423Z.html` |
| `search_curlcffi_chrome120` | curl_cffi(chrome120) | 200 | 13,052 | parser_mismatch_candidate | `carsandbids_search_curlcffi_chrome120_20260515T205423Z.html` |
| `past_auctions_curlcffi_chrome131` | curl_cffi(chrome131) | 200 | 13,219 | parser_mismatch_candidate | `carsandbids_past_auctions_curlcffi_chrome131_20260515T205423Z.html` |
| `sitemap_curlcffi_chrome131` | curl_cffi(chrome131) | 200 | 1,093 | (sitemap index) | (small) |
| `robots_curlcffi_chrome131` | curl_cffi(chrome131) | 200 | 213 | (robots.txt) | (small but valid) |

Notes:
- The first-200-char dump shows a `<meta data-react-helmet>` shell —
  Carsandbids.com is a client-rendered Next.js SPA. The 13 KB HTML is
  the *bundle bootstrap* and does NOT contain listing data inline; that
  data is fetched by the React app after hydration.
- However: a typical Next.js app inlines `__NEXT_DATA__` JSON in the
  initial HTML for SSR. The blocker tag did NOT see `__NEXT_DATA__` in
  the response (the content-markers heuristic includes it). This suggests
  the page truly is client-fetched.
- robots.txt reveals: `Sitemap: https://carsandbids.com/cab-sitemap/xml`
  — different path than `/sitemap.xml`. Worth a follow-up probe.
- Playwright via `search_playwright` returned a 403 — they fingerprint
  Chromium/CDP and challenge with Cloudflare *even with stealth*.

### Recommendation

**Cars & Bids needs a different strategy: hit the API/sitemap directly,
not the search SPA.** The shell HTML carries no listing data, so the
current parser approach can't extract anything regardless of selectors.
Options, in priority order:
1. **Walk `https://carsandbids.com/cab-sitemap/xml`** (discovered via
   robots.txt) to harvest auction URLs. This sidesteps the SPA entirely.
2. **Reverse-engineer the XHR API** the React app calls after hydration
   (likely `https://carsandbids.com/api/...`). curl_cffi(chrome131) can
   reach the origin; the question is whether the API needs auth/cookies.
3. **Use Playwright with longer settle** AND wait for a post-hydration
   selector (`.auction-list-item`, etc.) — but Playwright is currently
   getting 403'd at the entry, so step 1 or 2 is more promising.

## Summary table

| site | blocker class | recommended approach |
|---|---|---|
| cars.com | Cloudflare (TLS-fingerprint-sensitive) | `CurlCffiFetcher(impersonate="chrome131"\|"firefox133")` |
| autotrader | Akamai interstitial on every dynamic surface | Walk sitemap tree; otherwise needs residential proxy |
| hemmings | Cloudflare (TLS-fingerprint-sensitive) + parser selectors | `CurlCffiFetcher(impersonate="chrome131")` then update parser selectors |
| carsandbids | SPA — no listings in initial HTML | Walk `cab-sitemap/xml`; reverse-engineer XHR API |

## Saved fixtures (`tests/crawler/parsers/fixtures/real_world/`)

- `autotrader_robots_curlcffi_chrome131_20260515T205423Z.html` — AutoTrader robots.txt (17.6 KB)
- `cars_com_search_curlcffi_firefox_20260515T205423Z.html` — cars.com Honda Civic search page (1.22 MB, fully SSR'd)
- `carsandbids_past_auctions_curlcffi_chrome131_20260515T205423Z.html` — Cars & Bids past-auctions shell (13.2 KB, SPA bootstrap)
- `carsandbids_search_curlcffi_chrome120_20260515T205423Z.html` — Cars & Bids search shell (13.1 KB, SPA bootstrap)
- `carsandbids_search_curlcffi_chrome131_20260515T205423Z.html` — Cars & Bids search shell (13.1 KB, SPA bootstrap)
- `hemmings_search_curlcffi_chrome131_20260515T205423Z.html` — Hemmings Honda Civic search page (319 KB, Livewire-rendered)

## Next steps (out of scope for this commit)

1. Update the per-site routing in `MultiFetcher` so cars.com and Hemmings
   use `CurlCffiFetcher(impersonate="chrome131")` by default.
2. Re-run the cars.com parser against the saved 1.22 MB fixture; selectors
   should mostly work, validate against actual `vehicle-card` markup.
3. Write a new Hemmings parser test against the saved 319 KB fixture; the
   parser currently finds 0 listings, likely due to selector mismatch
   against the Livewire-rendered DOM.
4. Decide on a strategy for AutoTrader (sitemap-walk vs. proxy vs. drop).
5. Decide on a strategy for Cars & Bids (sitemap-walk vs. XHR API vs.
   Playwright-with-longer-settle).
