"""Tests for the listings accessors and constraints."""

from __future__ import annotations

import sqlite3

import pytest
from pydantic import ValidationError

from car_lense_engine.db import Listing, listings


def _sample(**overrides: object) -> Listing:
    base: dict[str, object] = {
        "listing_id": "cars_com:1",
        "source": "cars_com",
        "url": "https://cars.com/vehicledetail/1/",
        "year": 2018,
        "make": "Honda",
        "model": "Civic",
        "trim": "EX",
        "body_style": "sedan",
        "mileage": 42000,
        "vin": "1HGFC2F59JH123456",
        "raw_html_sha256": "deadbeef",
    }
    base.update(overrides)
    return Listing(**base)  # type: ignore[arg-type]


def test_insert_and_get_listing(db: sqlite3.Connection) -> None:
    listing = _sample()
    listings.insert_listing(db, listing)

    fetched = listings.get_listing(db, listing.listing_id)
    assert fetched is not None
    assert fetched.listing_id == listing.listing_id
    assert fetched.year == 2018
    assert fetched.make == "Honda"
    assert fetched.scraped_at is not None  # DB default applied


def test_get_missing_returns_none(db: sqlite3.Connection) -> None:
    assert listings.get_listing(db, "cars_com:does-not-exist") is None


def test_unique_url_constraint(db: sqlite3.Connection) -> None:
    listings.insert_listing(db, _sample(listing_id="cars_com:1"))
    with pytest.raises(sqlite3.IntegrityError):
        listings.insert_listing(
            db,
            _sample(listing_id="cars_com:2"),  # same URL
        )


def test_invalid_source_rejected(db: sqlite3.Connection) -> None:
    # The pydantic Literal would block this at model construction, so build
    # a raw payload and bypass the model.
    with pytest.raises(ValidationError):
        _sample(source="ebay")

    # Also confirm the DB CHECK constraint independently — insert raw SQL.
    with pytest.raises(sqlite3.IntegrityError), db:
        db.execute(
            "INSERT INTO listings (listing_id, source, url) VALUES (?, ?, ?)",
            ("x:1", "ebay", "https://example.com/x/1"),
        )


def test_update_listing_known_field(db: sqlite3.Connection) -> None:
    listing = _sample()
    listings.insert_listing(db, listing)
    listings.update_listing(db, listing.listing_id, mileage=99999)

    fetched = listings.get_listing(db, listing.listing_id)
    assert fetched is not None
    assert fetched.mileage == 99999


def test_update_listing_rejects_unknown_field(db: sqlite3.Connection) -> None:
    listing = _sample()
    listings.insert_listing(db, listing)
    with pytest.raises(ValueError):
        listings.update_listing(db, listing.listing_id, scraped_at="2024-01-01")


def test_list_by_class_filters(db: sqlite3.Connection) -> None:
    listings.insert_listing(db, _sample(listing_id="cars_com:1", year=2018))
    listings.insert_listing(
        db,
        _sample(
            listing_id="cars_com:2",
            year=2020,
            url="https://cars.com/vehicledetail/2/",
        ),
    )
    listings.insert_listing(
        db,
        _sample(
            listing_id="autotrader:1",
            source="autotrader",
            year=2018,
            url="https://autotrader.com/1",
        ),
    )

    found = listings.list_by_class(db, year=2018)
    assert len(found) == 2

    found_cars = listings.list_by_class(db, year=2018, source="cars_com")
    assert len(found_cars) == 1
    assert found_cars[0].listing_id == "cars_com:1"
