"""One-off diagnostic: AutoTrader + Cars & Bids sitemap probing.

Run with::

    python scripts/diag_sitemap_at_cab.py

Tasks (see Coder brief / SMOKE_REPORT.md fifth-run section):

* A) Walk AutoTrader's root sitemap (50-URL cap) and categorise the sample.
  The headline question: are AT's accessible sub-sitemaps full of actual
  vehicle-detail listings, or only marketing / dealer / static content?
* B) Probe Cars & Bids' sitemap endpoints directly to see what they actually
  return. The walker fifth-run showed ``cab-sitemap/xml`` is non-XML; this
  script captures what it *is*.

This is a one-shot script, kept in ``scripts/`` so it's reproducible. Politeness
matches the production crawler: ``min_delay_seconds=3.0``. At most ~10 actual
fetches across both diagnostics.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

# Make the package importable when running this script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from car_lense_engine.crawler.core.curlcffi_fetcher import CurlCffiFetcher  # noqa: E402
from car_lense_engine.crawler.core.sitemap import SitemapWalker  # noqa: E402

# AutoTrader listing-URL signature (matches the production filter in
# ``crawler.seed.sitemap_seed``).
_AT_VEHICLE_TAIL_RE = re.compile(r"(?:[/-])\d{6,}/?$")


_AT_BODY_STYLE_LEAVES = {
    "luxury",
    "coupe",
    "convertible",
    "commercial",
    "wagon",
    "van-minivan",
    "truck",
    "suv-crossover",
    "sedan",
    "hatchback",
}

_AT_MAKE_LEAVES = {
    "acura",
    "alfa-romeo",
    "aston-martin",
    "audi",
    "bentley",
    "bmw",
    "buick",
    "cadillac",
    "chevrolet",
    "chrysler",
    "daewoo",
    "dodge",
    "eagle",
    "ferrari",
    "fiat",
    "fisker",
    "ford",
    "genesis",
    "gmc",
    "honda",
    "hummer",
    "hyundai",
    "infiniti",
    "isuzu",
    "jaguar",
    "jeep",
    "kia",
    "lamborghini",
    "land-rover",
    "lexus",
    "lincoln",
    "lotus",
    "maserati",
    "maybach",
    "mazda",
    "mclaren",
    "mercedes-benz",
    "mercury",
    "mini",
    "mitsubishi",
    "nissan",
    "oldsmobile",
    "plymouth",
    "polestar",
    "pontiac",
    "porsche",
    "ram",
    "rivian",
    "rolls-royce",
    "saab",
    "saturn",
    "scion",
    "smart",
    "subaru",
    "suzuki",
    "tesla",
    "toyota",
    "volkswagen",
    "volvo",
}


def categorise_at_url(url: str) -> str:
    """Return a coarse category label for an AutoTrader sitemap URL."""
    parsed = urlparse(url)
    path = parsed.path
    if "/cars-for-sale/vehicledetails/" in path and _AT_VEHICLE_TAIL_RE.search(path):
        return "vehicle_listing"
    if "/dealers/" in path or "/dealer/" in path or path.startswith("/car-dealers"):
        return "dealer"
    if (
        "/car-news" in path
        or "/news/" in path
        or "/best-cars" in path
        or "/oversteer" in path
        or "/reviews" in path
        or "/car-reviews" in path
        or path.startswith("/archive")
    ):
        return "article_editorial"
    # Body-style / make landing pages (e.g. ``/luxury``, ``/acura``) are part
    # of the static taxonomy — they're search-result anchor pages, not
    # individual vehicle listings.
    leaf = path.strip("/").lower()
    if leaf in _AT_BODY_STYLE_LEAVES:
        return "static_category"
    if leaf in _AT_MAKE_LEAVES:
        return "static_category"
    if (
        path.startswith("/cars-for-sale/")
        or path.startswith("/research/")
        or path.startswith("/cars/")
        or path == "/"
        or path == ""
    ):
        return "static_category"
    return "other"


def diagnostic_a_autotrader_sample() -> None:
    """Walk the AT root sitemap, cap at 50 URLs, categorise."""
    print("=" * 72)
    print("DIAG A — AutoTrader sitemap content sample")
    print("=" * 72)
    fetcher = CurlCffiFetcher(impersonate="chrome131")
    try:
        # Cap at 50 URLs so we don't repeat the 10K walk from smoke run #5.
        walker = SitemapWalker(
            fetcher=fetcher,
            min_delay_seconds=3.0,
            max_urls=50,
        )
        urls: list[str] = []
        for url in walker.walk("https://www.autotrader.com/sitemap.xml"):
            urls.append(url)
            if len(urls) >= 50:
                break
    finally:
        fetcher.close()

    print(f"Walked URLs: {len(urls)}")
    if not urls:
        print("(no URLs yielded — sitemap walker returned an empty set)")
        return

    categories: Counter[str] = Counter()
    for url in urls:
        print(f"  {url}")
        categories[categorise_at_url(url)] += 1

    print()
    print("Category counts:")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count}")


def diagnostic_b_carsandbids_endpoint() -> None:
    """Probe Cars & Bids sitemap endpoints directly."""
    print()
    print("=" * 72)
    print("DIAG B — Cars & Bids sitemap endpoint probes")
    print("=" * 72)

    # Note: ``robots.txt`` actually advertises the real sitemap URL as
    # ``https://carsandbids.com/cab-sitemap/xml_sitemap.xml`` (verified
    # on the first run of this script). Probe that too — the bare
    # ``/cab-sitemap/xml`` path returns the SPA HTML shell.
    probes = [
        "https://carsandbids.com/cab-sitemap/xml",
        "https://carsandbids.com/cab-sitemap/xml_sitemap.xml",
        "https://carsandbids.com/sitemap.xml",
        "https://carsandbids.com/robots.txt",
    ]
    fetcher = CurlCffiFetcher(impersonate="chrome131")
    try:
        for url in probes:
            print()
            print(f"--- {url} ---")
            try:
                page = fetcher.fetch(url)
            except Exception as exc:
                print(f"  fetch failed: {exc!r}")
                continue
            body = page.html or ""
            print(f"  status      : {page.status}")
            print(f"  final URL   : {page.url}")
            print(f"  body length : {len(body)}")
            # Cheap classifier: XML / HTML / plain.
            stripped = body.lstrip()
            lower = stripped.lower()
            kind = "unknown"
            if (
                stripped.startswith("<?xml")
                or stripped.startswith("<urlset")
                or stripped.startswith("<sitemapindex")
            ):
                kind = "xml"
            elif lower.startswith("<!doctype html") or lower.startswith("<html"):
                kind = "html"
            elif body.startswith("User-agent") or "Sitemap:" in body[:1024]:
                kind = "robots_txt"
            print(f"  body kind   : {kind}")
            print(f"  first 500   : {body[:500]!r}")
    finally:
        fetcher.close()


def main() -> int:
    diagnostic_a_autotrader_sample()
    diagnostic_b_carsandbids_endpoint()
    return 0


if __name__ == "__main__":
    sys.exit(main())
