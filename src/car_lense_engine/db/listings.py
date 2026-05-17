"""Typed accessors for the ``listings`` table."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from .models import Listing

_INSERT_SQL = """
INSERT INTO listings (
    listing_id, source, url, year, make, model, trim, body_style,
    mileage, vin, raw_html_sha256, split, canonical_make, canonical_model
) VALUES (
    :listing_id, :source, :url, :year, :make, :model, :trim, :body_style,
    :mileage, :vin, :raw_html_sha256, :split, :canonical_make, :canonical_model
)
"""

_UPDATE_FIELDS: tuple[str, ...] = (
    "year",
    "make",
    "model",
    "trim",
    "body_style",
    "mileage",
    "vin",
    "raw_html_sha256",
    "split",
    "canonical_make",
    "canonical_model",
)


def insert_listing(conn: sqlite3.Connection, listing: Listing) -> None:
    """Insert a new listing row. Caller picks the listing_id."""
    payload = listing.model_dump(
        include={
            "listing_id",
            "source",
            "url",
            "year",
            "make",
            "model",
            "trim",
            "body_style",
            "mileage",
            "vin",
            "raw_html_sha256",
            "split",
            "canonical_make",
            "canonical_model",
        }
    )
    with conn:
        conn.execute(_INSERT_SQL, payload)


def get_listing(conn: sqlite3.Connection, listing_id: str) -> Listing | None:
    """Fetch a listing by primary key, or ``None`` if not present."""
    cur = conn.execute("SELECT * FROM listings WHERE listing_id = ?", (listing_id,))
    row = cur.fetchone()
    if row is None:
        return None
    return Listing.model_validate(dict(row))


def update_listing(conn: sqlite3.Connection, listing_id: str, **fields: object) -> None:
    """Update mutable fields on an existing listing row.

    Only fields in ``_UPDATE_FIELDS`` may be modified; passing any other key
    raises ``ValueError``. No-op if ``fields`` is empty.
    """
    if not fields:
        return
    unknown = set(fields) - set(_UPDATE_FIELDS)
    if unknown:
        raise ValueError(f"Cannot update listing fields: {sorted(unknown)}")
    assignments = ", ".join(f"{k} = :{k}" for k in fields)
    params: dict[str, object] = {**fields, "listing_id": listing_id}
    with conn:
        conn.execute(
            f"UPDATE listings SET {assignments} WHERE listing_id = :listing_id",
            params,
        )


def list_by_class(
    conn: sqlite3.Connection,
    year: int | None = None,
    make: str | None = None,
    model: str | None = None,
    source: str | None = None,
    limit: int | None = None,
) -> list[Listing]:
    """Return listings matching the given (year, make, model, source) filter.

    Any argument left as ``None`` is treated as a wildcard. Results are
    ordered by ``scraped_at DESC``.
    """
    clauses: list[str] = []
    params: list[object] = []
    if year is not None:
        clauses.append("year = ?")
        params.append(year)
    if make is not None:
        clauses.append("make = ?")
        params.append(make)
    if model is not None:
        clauses.append("model = ?")
        params.append(model)
    if source is not None:
        clauses.append("source = ?")
        params.append(source)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""
    sql = f"SELECT * FROM listings {where} ORDER BY scraped_at DESC {limit_sql}".strip()
    cur = conn.execute(sql, params)
    rows: Iterable[sqlite3.Row] = cur.fetchall()
    return [Listing.model_validate(dict(r)) for r in rows]
