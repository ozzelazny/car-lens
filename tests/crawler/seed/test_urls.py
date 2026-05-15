"""Tests for per-site URL builders and the SITE_BUILDERS registry."""

from __future__ import annotations

from car_lense_engine.crawler.seed.urls import (
    DEFAULT_CRAIGSLIST_CITIES,
    SITE_BUILDERS,
    autotrader,
    bringatrailer,
    cars_com,
    carsandbids,
    craigslist,
    hemmings,
)

# Canonical fixture: exercises the Mercedes-Benz → mercedes-benz slug.
MAKE = "Mercedes-Benz"
MODEL = "GLA"
YMIN = 2020
YMAX = 2022


def test_cars_com_known_url() -> None:
    [url] = cars_com(MAKE, MODEL, YMIN, YMAX)
    assert url == (
        "https://www.cars.com/shopping/results/"
        "?makes[]=mercedes-benz"
        "&models[]=mercedes-benz-gla"
        "&year_min=2020&year_max=2022"
        "&stock_type=all"
    )


def test_autotrader_known_url() -> None:
    [url] = autotrader(MAKE, MODEL, YMIN, YMAX)
    assert url == (
        "https://www.autotrader.com/cars-for-sale/all-cars/mercedes-benz/gla"
        "?yearMin=2020&yearMax=2022"
    )


def test_craigslist_default_cities_produces_10_urls() -> None:
    urls = craigslist(MAKE, MODEL, YMIN, YMAX)
    assert len(urls) == len(DEFAULT_CRAIGSLIST_CITIES) == 10
    # Each URL is prefixed by one of the configured cities.
    for city, url in zip(DEFAULT_CRAIGSLIST_CITIES, urls, strict=True):
        assert url.startswith(f"https://{city}.craigslist.org/search/cta")
        assert "auto_make_model=Mercedes-Benz+GLA" in url
        assert "min_auto_year=2020" in url
        assert "max_auto_year=2022" in url
        assert "query=Mercedes-Benz+GLA" in url


def test_craigslist_custom_cities_override() -> None:
    urls = craigslist(MAKE, MODEL, YMIN, YMAX, cities=["newyork"])
    assert len(urls) == 1
    assert urls[0].startswith("https://newyork.craigslist.org/")


def test_bringatrailer_known_url() -> None:
    [url] = bringatrailer(MAKE, MODEL, YMIN, YMAX)
    assert url == ("https://bringatrailer.com/mercedes-benz/gla/?year_min=2020&year_max=2022")


def test_hemmings_known_url() -> None:
    [url] = hemmings(MAKE, MODEL, YMIN, YMAX)
    assert url == (
        "https://www.hemmings.com/classifieds/cars-for-sale"
        "?Make=Mercedes-Benz&Model=GLA&YearFrom=2020&YearTo=2022"
    )


def test_carsandbids_known_url() -> None:
    [url] = carsandbids(MAKE, MODEL, YMIN, YMAX)
    assert url == ("https://carsandbids.com/search?q=Mercedes-Benz+GLA&year_min=2020&year_max=2022")


def test_url_encoding_of_special_chars_slug_sites() -> None:
    """Make names with spaces become hyphen-slugs for slug-style sites."""
    [u1] = cars_com("Land Rover", "Range Rover", 2018, 2024)
    assert "makes[]=land-rover" in u1
    assert "models[]=land-rover-range-rover" in u1

    [u2] = autotrader("Land Rover", "Range Rover", 2018, 2024)
    assert "/land-rover/range-rover" in u2

    [u3] = bringatrailer("Land Rover", "Range Rover", 2018, 2024)
    assert "/land-rover/range-rover/" in u3


def test_url_encoding_of_special_chars_query_sites() -> None:
    """Make names with spaces become quote_plus'd (+) in query-style URLs."""
    [u1] = hemmings("Land Rover", "Range Rover", 2018, 2024)
    assert "Make=Land+Rover" in u1
    assert "Model=Range+Rover" in u1

    [u2] = carsandbids("Land Rover", "Range Rover", 2018, 2024)
    assert "q=Land+Rover+Range+Rover" in u2


def test_site_builders_registry_keys_match_db_check_constraint() -> None:
    """SITE_BUILDERS keys must match the DB CHECK constraint values exactly."""
    assert set(SITE_BUILDERS) == {
        "cars_com",
        "autotrader",
        "craigslist",
        "bat",
        "hemmings",
        "carsandbids",
    }


def test_site_builders_match_listings_source_check() -> None:
    """The registry keys must equal the listings.source CHECK list in the SQL migration."""
    from importlib import resources

    sql = (
        resources.files("car_lense_engine.db.migrations")
        .joinpath("001_initial.sql")
        .read_text(encoding="utf-8")
    )
    for site in SITE_BUILDERS:
        assert f"'{site}'" in sql, f"site {site} missing from migration CHECK constraints"
