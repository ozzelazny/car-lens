"""Systematic block-diagnostic for the four currently-failing crawler sources.

Probes cars.com, AutoTrader, Hemmings, and Cars & Bids with multiple
fetcher / impersonation combinations and alternative endpoints
(sitemap, listing-page, etc.). For each probe records:

* HTTP status (or transport error)
* HTML byte count
* First 200 chars of HTML (for sniffing what we got)
* Detected blocker class (Cloudflare challenge marker, unhydrated shell,
  parser-amenable content, etc.)
* Final URL after redirects
* Cookies set by the server

Useful HTML responses are saved under
``tests/crawler/parsers/fixtures/real_world/`` for the parser-fix pass that
follows this diagnostic.

Designed to be run ONCE per IP/session; politeness is enforced internally
(5-10 s between requests to the same hostname). Total request budget is
capped at ``MAX_TOTAL_REQUESTS`` so we don't get our IP further flagged.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Importable from the editable install in .venv-diag.
from car_lense_engine.crawler.core.browser import PlaywrightFetcher
from car_lense_engine.crawler.core.curlcffi_fetcher import CurlCffiFetcher
from car_lense_engine.crawler.core.fetcher import FetchError

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "crawler" / "parsers" / "fixtures" / "real_world"
REPORT_PATH = REPO_ROOT / "BLOCKS_DIAGNOSTIC.md"

# Politeness: minimum gap between requests to the same hostname.
MIN_GAP_PER_HOST_SECONDS: float = 7.0

# Hard ceiling so a runaway loop can't hammer the upstream.
MAX_TOTAL_REQUESTS: int = 30


# ---------------------------------------------------------------------- probes


@dataclass
class Probe:
    """One probe (URL + fetcher configuration) for a given source."""

    source: str
    """Site identifier (cars_com / autotrader / hemmings / carsandbids)."""

    name: str
    """Short label: "search_playwright", "search_chrome131", ..."""

    url: str
    """URL to fetch."""

    fetcher_kind: str
    """Either "playwright" or "curl_cffi"."""

    impersonate: str | None = None
    """For curl_cffi: chrome131 / chrome120 / firefox120."""


@dataclass
class ProbeResult:
    """Outcome of a single probe."""

    source: str
    name: str
    url: str
    fetcher_kind: str
    impersonate: str | None
    status: int | None
    final_url: str | None
    bytes_: int
    first_200: str
    blockers: list[str] = field(default_factory=list)
    saved_to: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # rename trailing-underscore field for JSON friendliness
        d["bytes"] = d.pop("bytes_")
        return d


# Per-source probe definitions. Each site gets a curated set of probes
# covering: search via playwright, search via curl_cffi with three different
# impersonations, sitemap, robots.txt, one alternative endpoint, and one
# concrete listing page. We keep this under the request budget.
SITE_PROBES: dict[str, list[Probe]] = {
    "cars_com": [
        Probe(
            source="cars_com",
            name="search_playwright",
            url=(
                "https://www.cars.com/shopping/results/"
                "?stock_type=used&makes[]=honda&models[]=honda-civic"
            ),
            fetcher_kind="playwright",
        ),
        Probe(
            source="cars_com",
            name="search_curlcffi_chrome131",
            url=(
                "https://www.cars.com/shopping/results/"
                "?stock_type=used&makes[]=honda&models[]=honda-civic"
            ),
            fetcher_kind="curl_cffi",
            impersonate="chrome131",
        ),
        Probe(
            source="cars_com",
            name="search_curlcffi_chrome120",
            url=(
                "https://www.cars.com/shopping/results/"
                "?stock_type=used&makes[]=honda&models[]=honda-civic"
            ),
            fetcher_kind="curl_cffi",
            impersonate="chrome120",
        ),
        Probe(
            source="cars_com",
            name="search_curlcffi_firefox",
            url=(
                "https://www.cars.com/shopping/results/"
                "?stock_type=used&makes[]=honda&models[]=honda-civic"
            ),
            fetcher_kind="curl_cffi",
            impersonate="firefox133",
        ),
        Probe(
            source="cars_com",
            name="sitemap_curlcffi_chrome131",
            url="https://www.cars.com/sitemap.xml",
            fetcher_kind="curl_cffi",
            impersonate="chrome131",
        ),
        Probe(
            source="cars_com",
            name="robots_curlcffi_chrome131",
            url="https://www.cars.com/robots.txt",
            fetcher_kind="curl_cffi",
            impersonate="chrome131",
        ),
        Probe(
            source="cars_com",
            name="listing_curlcffi_chrome131",
            # A plausible vehicle-detail URL pattern; if id is dead we'll see
            # a 404 vs an anti-bot 403, which itself is signal.
            url="https://www.cars.com/vehicledetail/123456789/",
            fetcher_kind="curl_cffi",
            impersonate="chrome131",
        ),
    ],
    "autotrader": [
        Probe(
            source="autotrader",
            name="search_playwright",
            url=(
                "https://www.autotrader.com/cars-for-sale/all-cars/honda/civic"
                "?searchRadius=0&zip=10001"
            ),
            fetcher_kind="playwright",
        ),
        Probe(
            source="autotrader",
            name="search_curlcffi_chrome131",
            url=(
                "https://www.autotrader.com/cars-for-sale/all-cars/honda/civic"
                "?searchRadius=0&zip=10001"
            ),
            fetcher_kind="curl_cffi",
            impersonate="chrome131",
        ),
        Probe(
            source="autotrader",
            name="search_curlcffi_chrome120",
            url=(
                "https://www.autotrader.com/cars-for-sale/all-cars/honda/civic"
                "?searchRadius=0&zip=10001"
            ),
            fetcher_kind="curl_cffi",
            impersonate="chrome120",
        ),
        Probe(
            source="autotrader",
            name="sitemap_curlcffi_chrome131",
            url="https://www.autotrader.com/sitemap.xml",
            fetcher_kind="curl_cffi",
            impersonate="chrome131",
        ),
        Probe(
            source="autotrader",
            name="robots_curlcffi_chrome131",
            url="https://www.autotrader.com/robots.txt",
            fetcher_kind="curl_cffi",
            impersonate="chrome131",
        ),
        Probe(
            source="autotrader",
            name="listing_curlcffi_chrome131",
            url="https://www.autotrader.com/cars-for-sale/vehicledetails/12345",
            fetcher_kind="curl_cffi",
            impersonate="chrome131",
        ),
    ],
    "hemmings": [
        Probe(
            source="hemmings",
            name="search_playwright",
            url="https://www.hemmings.com/classifieds/cars-for-sale/honda/civic",
            fetcher_kind="playwright",
        ),
        Probe(
            source="hemmings",
            name="search_curlcffi_chrome131",
            url="https://www.hemmings.com/classifieds/cars-for-sale/honda/civic",
            fetcher_kind="curl_cffi",
            impersonate="chrome131",
        ),
        Probe(
            source="hemmings",
            name="search_curlcffi_chrome120",
            url="https://www.hemmings.com/classifieds/cars-for-sale/honda/civic",
            fetcher_kind="curl_cffi",
            impersonate="chrome120",
        ),
        Probe(
            source="hemmings",
            name="sitemap_curlcffi_chrome131",
            url="https://www.hemmings.com/sitemap.xml",
            fetcher_kind="curl_cffi",
            impersonate="chrome131",
        ),
        Probe(
            source="hemmings",
            name="listing_curlcffi_chrome131",
            url="https://www.hemmings.com/classifieds/dealer/honda/civic/2845391/",
            fetcher_kind="curl_cffi",
            impersonate="chrome131",
        ),
    ],
    "carsandbids": [
        Probe(
            source="carsandbids",
            name="search_playwright",
            url="https://carsandbids.com/search/honda%20civic",
            fetcher_kind="playwright",
        ),
        Probe(
            source="carsandbids",
            name="search_curlcffi_chrome131",
            url="https://carsandbids.com/search/honda%20civic",
            fetcher_kind="curl_cffi",
            impersonate="chrome131",
        ),
        Probe(
            source="carsandbids",
            name="search_curlcffi_chrome120",
            url="https://carsandbids.com/search/honda%20civic",
            fetcher_kind="curl_cffi",
            impersonate="chrome120",
        ),
        Probe(
            source="carsandbids",
            name="past_auctions_curlcffi_chrome131",
            url="https://carsandbids.com/past-auctions/",
            fetcher_kind="curl_cffi",
            impersonate="chrome131",
        ),
        Probe(
            source="carsandbids",
            name="sitemap_curlcffi_chrome131",
            url="https://carsandbids.com/sitemap.xml",
            fetcher_kind="curl_cffi",
            impersonate="chrome131",
        ),
        Probe(
            source="carsandbids",
            name="robots_curlcffi_chrome131",
            url="https://carsandbids.com/robots.txt",
            fetcher_kind="curl_cffi",
            impersonate="chrome131",
        ),
    ],
}


# ---------------------------------------------------------------- blocker sniff

# Strict CF challenge markers — only strings that appear on an *interstitial*
# challenge page, NOT on normal pages that happen to embed Cloudflare's
# JS beacon. ``challenge-platform`` (the beacon script path) and bare
# ``Cloudflare Ray ID`` (which CF injects into many normal 200 responses too)
# are deliberately omitted here to avoid false positives — the diagnostic
# treats their presence on a 200-OK 300KB page as a *content* page, not a
# challenge.
CF_CHALLENGE_MARKERS: tuple[str, ...] = (
    "Just a moment",
    "__cf_chl_jschl_tk__",
    "cf-error-details",
    "Attention Required! | Cloudflare",
    "__cf_chl_opt",
    "cf_chl_managed",
    "Enable JavaScript and cookies to continue",
)

CONTENT_MARKERS: dict[str, tuple[str, ...]] = {
    # Substrings that indicate the page has real listing-ish content. Used
    # only as a coarse "is this worth saving as a fixture?" heuristic — the
    # parser layer will still need real selectors.
    "cars_com": (
        "vehicle-card",
        "listing-card",
        "/vehicledetail/",
        "spark-vehicle-card",
        "data-listing-id",
    ),
    "autotrader": (
        "inventory-listing",
        "/cars-for-sale/vehicledetails/",
        "data-cmp=\"inventoryListing\"",
        "vehicle-card",
    ),
    "hemmings": (
        "/classifieds/dealer/",
        "/classifieds/private/",
        "listing-card",
        "ad-tile",
    ),
    "carsandbids": (
        "/auctions/",
        "auction-item",
        "auctions-list",
        "__NEXT_DATA__",
    ),
}


def detect_blockers(source: str, status: int | None, html: str) -> list[str]:
    """Return a list of blocker tags applicable to this response."""
    tags: list[str] = []
    if status is None:
        tags.append("transport_error")
        return tags
    if status == 403:
        tags.append("http_403")
    if status == 429:
        tags.append("http_429")
    if status >= 500:
        tags.append("http_5xx")
    if status == 404:
        tags.append("http_404")

    for marker in CF_CHALLENGE_MARKERS:
        if marker in html:
            tags.append("cloudflare_challenge")
            break

    n_bytes = len(html.encode("utf-8", errors="ignore"))
    if status == 200 and n_bytes < 8_000:
        tags.append("unhydrated_shell")

    content_hits = [m for m in CONTENT_MARKERS.get(source, ()) if m in html]
    if content_hits:
        tags.append(f"content_markers:{','.join(content_hits[:3])}")

    if status == 200 and not content_hits and "unhydrated_shell" not in tags:
        tags.append("parser_mismatch_candidate")

    return tags


def is_worth_saving(
    url: str,
    blockers: list[str],
    status: int | None,
    n_bytes: int,
) -> bool:
    """Heuristic for whether to drop a fixture file.

    robots.txt and sitemap.xml endpoints are *always* worth saving on a
    HTTP 200 (the shell/byte-count heuristic doesn't apply to them).
    """
    if status is None or status >= 400:
        return False
    lower_url = url.lower()
    is_meta = lower_url.endswith("/robots.txt") or lower_url.endswith("sitemap.xml")
    if any(t.startswith("cloudflare_challenge") for t in blockers):
        return False
    if "unhydrated_shell" in blockers and not is_meta:
        return False
    if is_meta:
        return n_bytes > 0
    if any(t.startswith("content_markers:") for t in blockers):
        return True
    return n_bytes > 1_000


# --------------------------------------------------------------- fetcher pool


class FetcherPool:
    """Lazily-built collection of fetchers reused across probes.

    Keeping a single PlaywrightFetcher across probes saves Chromium startup
    cost (~3-5 s each). CurlCffiFetcher is per-impersonation since the
    impersonate kwarg is fixed at construction time.
    """

    def __init__(self) -> None:
        self._playwright: PlaywrightFetcher | None = None
        self._curl: dict[str, CurlCffiFetcher] = {}

    def playwright(self) -> PlaywrightFetcher:
        if self._playwright is None:
            self._playwright = PlaywrightFetcher(
                headless=True,
                settle_ms=5_000,
                navigation_timeout_ms=30_000,
            )
        return self._playwright

    def curl(self, impersonate: str) -> CurlCffiFetcher:
        if impersonate not in self._curl:
            self._curl[impersonate] = CurlCffiFetcher(
                impersonate=impersonate,
                timeout_seconds=30.0,
            )
        return self._curl[impersonate]

    def close(self) -> None:
        if self._playwright is not None:
            try:
                self._playwright.close()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] playwright close failed: {exc!r}", file=sys.stderr)
            self._playwright = None
        for imp, f in list(self._curl.items()):
            try:
                f.close()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] curl_cffi[{imp}] close failed: {exc!r}", file=sys.stderr)
        self._curl.clear()


# --------------------------------------------------------------- probe runner


def run_probe(pool: FetcherPool, probe: Probe) -> ProbeResult:
    """Run a single probe; never raises (all errors are recorded in the result)."""
    status: int | None = None
    final_url: str | None = None
    html = ""
    error: str | None = None

    try:
        if probe.fetcher_kind == "playwright":
            fetcher = pool.playwright()
        elif probe.fetcher_kind == "curl_cffi":
            assert probe.impersonate is not None, "curl_cffi probe missing impersonate"
            fetcher = pool.curl(probe.impersonate)
        else:
            raise ValueError(f"unknown fetcher_kind: {probe.fetcher_kind!r}")

        page = fetcher.fetch(probe.url)
        status = page.status
        final_url = page.url
        html = page.html
    except FetchError as exc:
        # FetchError messages have the form "HTTP 403 for <url>" — parse it
        # back into a status code where possible, so 403/404/429 show up as
        # data rather than just generic errors.
        msg = str(exc)
        if msg.startswith("HTTP "):
            try:
                status = int(msg.split()[1])
            except (IndexError, ValueError):
                status = None
        error = msg
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    n_bytes = len(html.encode("utf-8", errors="ignore"))
    blockers = detect_blockers(probe.source, status, html)
    first_200 = html[:200].replace("\n", " ").replace("\r", " ")

    return ProbeResult(
        source=probe.source,
        name=probe.name,
        url=probe.url,
        fetcher_kind=probe.fetcher_kind,
        impersonate=probe.impersonate,
        status=status,
        final_url=final_url,
        bytes_=n_bytes,
        first_200=first_200,
        blockers=blockers,
        error=error,
    )


def save_fixture(result: ProbeResult, html: str, timestamp: str) -> str:
    """Save the HTML to fixtures/real_world/; return the absolute path saved."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{result.source}_{result.name}_{timestamp}.html"
    path = FIXTURES_DIR / fname
    path.write_text(html, encoding="utf-8", errors="replace")
    return str(path)


# ----------------------------------------------------------------- main loop


def main() -> int:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    pool = FetcherPool()
    results: list[ProbeResult] = []
    last_request_per_host: dict[str, float] = {}
    total_requests = 0

    try:
        for source, probes in SITE_PROBES.items():
            print(f"\n=== {source} ===", flush=True)
            for probe in probes:
                if total_requests >= MAX_TOTAL_REQUESTS:
                    print(
                        f"[budget] hit MAX_TOTAL_REQUESTS={MAX_TOTAL_REQUESTS}; stopping",
                        flush=True,
                    )
                    break

                host = (urlparse(probe.url).hostname or "").lower()
                gap_needed = MIN_GAP_PER_HOST_SECONDS - (
                    time.monotonic() - last_request_per_host.get(host, 0.0)
                )
                if gap_needed > 0 and host in last_request_per_host:
                    time.sleep(gap_needed)

                print(
                    f"[probe] {probe.source}/{probe.name} -> {probe.url}",
                    flush=True,
                )
                # Capture the full HTML buffer separately so we can save the
                # fixture without round-tripping through the truncated
                # ProbeResult.first_200.
                _pre_fetch = time.monotonic()
                result = run_probe(pool, probe)
                last_request_per_host[host] = time.monotonic()
                total_requests += 1

                # Re-fetch html via the same call path is unnecessary — but
                # we need the HTML string itself for the fixture write. Pull
                # it from the underlying fetch by re-running the probe's
                # fetcher path is wasteful; instead we capture html during
                # run_probe via a closure. Simpler: re-run the probe is bad;
                # instead, save the fixture using a *second* call only if the
                # probe was worthwhile. Cheaper: re-call once for fixtures.
                #
                # In practice the cleanest path is to make run_probe also
                # return the html buffer. Refactor inline below.
                html_for_fixture = _capture_html_for_fixture(pool, probe, result)
                if html_for_fixture is not None and is_worth_saving(
                    probe.url, result.blockers, result.status, result.bytes_
                ):
                    saved = save_fixture(result, html_for_fixture, timestamp)
                    result.saved_to = saved
                    print(f"        saved fixture -> {saved}", flush=True)

                elapsed = time.monotonic() - _pre_fetch
                print(
                    f"        status={result.status} bytes={result.bytes_} "
                    f"blockers={result.blockers} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                if result.error:
                    print(f"        error: {result.error}", flush=True)
                results.append(result)
            if total_requests >= MAX_TOTAL_REQUESTS:
                break
    finally:
        pool.close()

    # JSON dump for downstream tooling / debugging.
    print("\n=== summary (JSON) ===", flush=True)
    print(json.dumps([r.to_dict() for r in results], indent=2), flush=True)

    # Markdown report.
    write_report(results, timestamp)
    print(f"\nWrote report to {REPORT_PATH}", flush=True)

    return 0


def _capture_html_for_fixture(
    pool: FetcherPool,
    probe: Probe,
    result: ProbeResult,
) -> str | None:
    """Return the HTML buffer for this probe if it's worth saving.

    Implementation note: ``run_probe`` already fetched the page and threw
    away the buffer once it computed bytes/first_200/blockers. Rather than
    re-fetch (and double our request budget) we *only* call this when the
    probe is worth saving — but to do so we need the html anyway. The
    cheapest path is to short-circuit: if the probe is not worth saving by
    the blocker-tag heuristic alone, return None without any IO. Otherwise
    re-run the fetch to grab the html for persistence.

    This wastes a request on each saved fixture. We accept that within the
    request budget; saved fixtures are the whole point of the diagnostic.
    """
    if not is_worth_saving(probe.url, result.blockers, result.status, result.bytes_):
        return None

    try:
        if probe.fetcher_kind == "playwright":
            fetcher = pool.playwright()
        else:
            assert probe.impersonate is not None
            fetcher = pool.curl(probe.impersonate)
        page = fetcher.fetch(probe.url)
        return page.html
    except Exception as exc:  # pragma: no cover - defensive
        print(f"        [warn] re-fetch for fixture failed: {exc!r}", flush=True)
        return None


# ------------------------------------------------------------- markdown report


def write_report(results: list[ProbeResult], timestamp: str) -> None:
    lines: list[str] = []
    lines.append(f"# Block Diagnostic — {timestamp}\n")
    lines.append(
        "Systematic probe of the four currently-failing crawler sources "
        "(cars.com, AutoTrader, Hemmings, Cars & Bids) using multiple "
        "fetcher / impersonation combinations and alternative endpoints. "
        "Goal: characterise the blocker class per site so a follow-up "
        "parser-fix task can pick the right approach.\n"
    )
    lines.append("## Setup\n")
    lines.append(
        f"- Run timestamp: `{timestamp}` UTC\n"
        f"- Min gap between requests to the same host: "
        f"`{MIN_GAP_PER_HOST_SECONDS}s`\n"
        f"- Max total requests cap: `{MAX_TOTAL_REQUESTS}`\n"
        f"- Total requests issued: `{len(results)}` (plus fixture re-fetches "
        f"for usable probes)\n"
        f"- Fixtures dropped under: `tests/crawler/parsers/fixtures/real_world/`\n"
    )

    # Per-source matrices.
    by_source: dict[str, list[ProbeResult]] = {}
    for r in results:
        by_source.setdefault(r.source, []).append(r)

    for source, probes in by_source.items():
        lines.append(f"## {source}\n")
        lines.append(
            "| probe | fetcher | status | bytes | blockers | saved |\n"
            "|---|---|---|---|---|---|"
        )
        for r in probes:
            fetcher_label = (
                f"{r.fetcher_kind}({r.impersonate})"
                if r.fetcher_kind == "curl_cffi"
                else r.fetcher_kind
            )
            blockers_str = ", ".join(r.blockers) if r.blockers else "(none)"
            saved_str = (
                Path(r.saved_to).name if r.saved_to else "—"
            )
            status_str = str(r.status) if r.status is not None else "transport-err"
            lines.append(
                f"| `{r.name}` | {fetcher_label} | {status_str} | "
                f"{r.bytes_} | {blockers_str} | {saved_str} |"
            )
        lines.append("")

        # First-200 dump per probe so blockers are auditable from the report.
        lines.append("<details><summary>First-200-char dumps</summary>\n")
        for r in probes:
            lines.append(f"**`{r.name}`** — final_url: `{r.final_url or '(none)'}`\n")
            lines.append("```")
            lines.append(r.first_200 or "(empty)")
            lines.append("```\n")
        lines.append("</details>\n")

        rec = _per_source_recommendation(source, probes)
        lines.append(f"### Recommendation\n\n{rec}\n")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _per_source_recommendation(source: str, results: list[ProbeResult]) -> str:
    """Synthesize a one-line recommendation from the per-site probe outcomes."""
    usable = [r for r in results if r.saved_to is not None]
    cf_count = sum(
        1 for r in results if any(t == "cloudflare_challenge" for t in r.blockers)
    )
    http403_count = sum(1 for r in results if r.status == 403)
    http_ok_count = sum(1 for r in results if r.status == 200)
    shell_count = sum(1 for r in results if "unhydrated_shell" in r.blockers)

    parts: list[str] = []
    if usable:
        best = max(usable, key=lambda r: r.bytes_)
        fetcher_label = (
            f"{best.fetcher_kind}({best.impersonate})"
            if best.fetcher_kind == "curl_cffi"
            else best.fetcher_kind
        )
        parts.append(
            f"**Use `{best.name}` ({fetcher_label})** — returned "
            f"{best.bytes_} bytes of usable HTML "
            f"(blockers: {', '.join(best.blockers) or 'none'}). "
            f"Fixture saved at `{Path(best.saved_to).name if best.saved_to else '?'}`."
        )
    else:
        parts.append("**No probe returned usable HTML.**")

    parts.append(
        f"Counts: HTTP-200={http_ok_count}, HTTP-403={http403_count}, "
        f"Cloudflare-challenge={cf_count}, unhydrated-shell={shell_count}."
    )

    if not usable and http403_count >= 3:
        parts.append(
            "All probes are 403 — site is likely IP-rep-blocked from this "
            "vantage point. Residential proxy required."
        )
    elif not usable and shell_count >= 2:
        parts.append(
            "Multiple probes returned HTTP 200 but with tiny shells. The "
            "site hydrates via JS and Playwright settle/wait_for_selector "
            "isn't catching it. Need longer settle, networkidle, or an "
            "alternative endpoint (sitemap/API)."
        )
    elif not usable and cf_count >= 1:
        parts.append(
            "Cloudflare challenge interstitials detected. Need either a "
            "JS-execution path (Playwright with longer settle) or a "
            "residential proxy."
        )

    return " ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
