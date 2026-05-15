"""Per-site search-URL builders.

Each builder returns a ``list[str]`` because some sites (Craigslist) span
multiple cities, and we may add pagination URLs in the future. Single-URL
sites return a one-element list. URL patterns here are best-effort starting
points; if a site changes its query format the parser will discover that
empirically (zero results) and we'll iterate.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

BuilderFn = Callable[..., list[str]]


# Default Craigslist cities — a US-wide spread. Override with the ``cities``
# kwarg on ``craigslist`` for custom subsets.
DEFAULT_CRAIGSLIST_CITIES: list[str] = [
    "newyork",
    "losangeles",
    "sfbay",
    "chicago",
    "houston",
    "atlanta",
    "miami",
    "seattle",
    "boston",
    "dallas",
]


def _slug(value: str) -> str:
    """Convert ``Mercedes-Benz`` → ``mercedes-benz``; lowercase, spaces → hyphens."""
    return value.lower().replace(" ", "-")


def cars_com(make: str, model: str, year_min: int, year_max: int) -> list[str]:
    """Build the cars.com search URL for one ``(make, model, year-range)`` slice."""
    make_slug = _slug(make)
    model_slug = _slug(model)
    url = (
        f"https://www.cars.com/shopping/results/"
        f"?makes[]={make_slug}"
        f"&models[]={make_slug}-{model_slug}"
        f"&year_min={year_min}&year_max={year_max}"
        f"&stock_type=all"
    )
    return [url]


def autotrader(make: str, model: str, year_min: int, year_max: int) -> list[str]:
    """Build the AutoTrader search URL for one ``(make, model, year-range)`` slice."""
    make_slug = _slug(make)
    model_slug = _slug(model)
    url = (
        f"https://www.autotrader.com/cars-for-sale/all-cars/{make_slug}/{model_slug}"
        f"?yearMin={year_min}&yearMax={year_max}"
    )
    return [url]


def craigslist(
    make: str,
    model: str,
    year_min: int,
    year_max: int,
    cities: list[str] | None = None,
) -> list[str]:
    """Build one Craigslist search URL per city for a ``(make, model, year-range)``."""
    chosen = cities if cities is not None else DEFAULT_CRAIGSLIST_CITIES
    q = quote_plus(f"{make} {model}")
    urls: list[str] = []
    for city in chosen:
        url = (
            f"https://{city}.craigslist.org/search/cta"
            f"?auto_make_model={q}"
            f"&min_auto_year={year_min}&max_auto_year={year_max}"
            f"&query={q}"
        )
        urls.append(url)
    return urls


def bringatrailer(make: str, model: str, year_min: int, year_max: int) -> list[str]:
    """Build the Bring-a-Trailer search URL for one ``(make, model, year-range)``."""
    make_slug = _slug(make)
    model_slug = _slug(model)
    url = (
        f"https://bringatrailer.com/{make_slug}/{model_slug}/"
        f"?year_min={year_min}&year_max={year_max}"
    )
    return [url]


def hemmings(make: str, model: str, year_min: int, year_max: int) -> list[str]:
    """Build the Hemmings classifieds search URL for one ``(make, model, year-range)``.

    Hemmings strips ``?Make=&Model=`` query-string filters on the redirect
    chain and serves the bare category landing page, so we encode the
    make/model as slug path segments instead — the same convention
    cars.com and AutoTrader use. Year filters survive as query params.
    """
    make_slug = _slug(make)
    model_slug = _slug(model)
    url = (
        f"https://www.hemmings.com/classifieds/cars-for-sale/{make_slug}/{model_slug}"
        f"?YearFrom={year_min}&YearTo={year_max}"
    )
    return [url]


def carsandbids(make: str, model: str, year_min: int, year_max: int) -> list[str]:
    """Build the Cars & Bids search URL for one ``(make, model, year-range)``."""
    q = quote_plus(f"{make} {model}")
    url = f"https://carsandbids.com/search?q={q}&year_min={year_min}&year_max={year_max}"
    return [url]


# Registry mapping the DB-level source identifier to the builder function.
# Must stay in lockstep with the CHECK constraint on listings.source /
# crawl_queue.source in db/migrations/001_initial.sql.
SITE_BUILDERS: dict[str, BuilderFn] = {
    "cars_com": cars_com,
    "autotrader": autotrader,
    "craigslist": craigslist,
    "bat": bringatrailer,
    "hemmings": hemmings,
    "carsandbids": carsandbids,
}
