"""Typed accessors for the ``images`` table."""

from __future__ import annotations

import sqlite3

from .models import Image

_INSERT_SQL = """
INSERT INTO images (
    image_id, listing_id, source_url, local_path, phash,
    width, height, bytes, position
) VALUES (
    :image_id, :listing_id, :source_url, :local_path, :phash,
    :width, :height, :bytes, :position
)
"""


def insert_image(conn: sqlite3.Connection, image: Image) -> None:
    """Insert a new image row. ``image_id`` is expected to be the SHA-256 of the bytes."""
    payload = image.model_dump(
        include={
            "image_id",
            "listing_id",
            "source_url",
            "local_path",
            "phash",
            "width",
            "height",
            "bytes",
            "position",
        }
    )
    with conn:
        conn.execute(_INSERT_SQL, payload)


def get_image_by_sha(conn: sqlite3.Connection, image_id: str) -> Image | None:
    """Fetch an image by its content hash (primary key)."""
    cur = conn.execute("SELECT * FROM images WHERE image_id = ?", (image_id,))
    row = cur.fetchone()
    if row is None:
        return None
    return Image.model_validate(dict(row))


def update_phash(conn: sqlite3.Connection, image_id: str, phash: str) -> None:
    """Set the perceptual hash for an existing image row."""
    with conn:
        conn.execute(
            "UPDATE images SET phash = ? WHERE image_id = ?",
            (phash, image_id),
        )


def list_for_listing(conn: sqlite3.Connection, listing_id: str) -> list[Image]:
    """Return all images attached to a listing, ordered by gallery position."""
    cur = conn.execute(
        "SELECT * FROM images WHERE listing_id = ? "
        "ORDER BY COALESCE(position, 1 << 30), downloaded_at",
        (listing_id,),
    )
    return [Image.model_validate(dict(r)) for r in cur.fetchall()]
