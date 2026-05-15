"""Tests for the images accessors and FK cascade behavior."""

from __future__ import annotations

import sqlite3

import pytest

from car_lense_engine.db import Image, Listing, images, listings


def _listing() -> Listing:
    return Listing(
        listing_id="cars_com:1",
        source="cars_com",
        url="https://cars.com/vehicledetail/1/",
        year=2018,
        make="Honda",
        model="Civic",
    )


def _image(image_id: str = "a" * 64, position: int = 0) -> Image:
    return Image(
        image_id=image_id,
        listing_id="cars_com:1",
        source_url="https://example.com/photo_0.jpg",
        local_path="data/raw/cars_com/listings/1/photo_0.jpg",
        width=1024,
        height=768,
        bytes=120_000,
        position=position,
    )


def test_insert_image_and_get(db: sqlite3.Connection) -> None:
    listings.insert_listing(db, _listing())
    img = _image()
    images.insert_image(db, img)

    fetched = images.get_image_by_sha(db, img.image_id)
    assert fetched is not None
    assert fetched.image_id == img.image_id
    assert fetched.listing_id == img.listing_id
    assert fetched.position == 0


def test_fk_cascade_deletes_images(db: sqlite3.Connection) -> None:
    listings.insert_listing(db, _listing())
    images.insert_image(db, _image(image_id="a" * 64, position=0))
    images.insert_image(db, _image(image_id="b" * 64, position=1))

    with db:
        db.execute("DELETE FROM listings WHERE listing_id = ?", ("cars_com:1",))

    remaining = db.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"]
    assert int(remaining) == 0


def test_image_id_is_sha256_dedup(db: sqlite3.Connection) -> None:
    listings.insert_listing(db, _listing())
    img = _image()
    images.insert_image(db, img)
    with pytest.raises(sqlite3.IntegrityError):
        images.insert_image(db, img)


def test_update_phash(db: sqlite3.Connection) -> None:
    listings.insert_listing(db, _listing())
    img = _image()
    images.insert_image(db, img)

    images.update_phash(db, img.image_id, "0123456789abcdef")
    fetched = images.get_image_by_sha(db, img.image_id)
    assert fetched is not None
    assert fetched.phash == "0123456789abcdef"


def test_list_for_listing_orders_by_position(db: sqlite3.Connection) -> None:
    listings.insert_listing(db, _listing())
    images.insert_image(db, _image(image_id="c" * 64, position=2))
    images.insert_image(db, _image(image_id="a" * 64, position=0))
    images.insert_image(db, _image(image_id="b" * 64, position=1))

    result = images.list_for_listing(db, "cars_com:1")
    assert [img.position for img in result] == [0, 1, 2]
