"""Console script that populates the canonical label + generation-year columns.

Invoke via the ``canonicalize-labels`` entry point declared in
``pyproject.toml``::

    canonicalize-labels [--db PATH] [--source SOURCE] [--limit N] [--rebuild] [-v]

Walks every row in ``listings`` (optionally filtered by ``--source``)
and writes:

* ``canonical_make`` / ``canonical_model`` (migration 8, Phase 4.5):
  the alias-mapped / Title-Cased form of the raw make / model strings.
* ``generation_year`` (migration 9, Phase 4.6): the 4-year bucket
  start year derived from the raw ``year`` column. E.g. ``year=2014``
  -> ``generation_year=2012`` (bucket 2012-2015). This collapses
  adjacent model years of the same vehicle into one class, which
  matches the ~4-year redesign cycle and is the dominant source of
  confusion in the Phase 5.2 baseline.

Idempotent: rows are re-processed only if EITHER ``canonical_make``
OR ``generation_year`` is still NULL, unless ``--rebuild`` is passed
(which re-runs every row).

**IMPORTANT**: Phase 5 baseline + training read the canonical columns
+ ``generation_year`` exclusively. Rows whose ``canonical_make`` /
``canonical_model`` / ``generation_year`` is NULL are excluded from
prototype-building and from the training data loader -- the same way
rows with NULL ``year`` are excluded today. **You must run this CLI
before** ``phase5-baseline`` / ``phase5-train``.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from car_lense_engine.db import open_db

from .canonical_labels import normalize_make, normalize_model, year_to_generation

DEFAULT_DB = Path("db/crawl.sqlite")
DEFAULT_PROGRESS_INTERVAL = 5000

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser for the ``canonicalize-labels`` command."""
    parser = argparse.ArgumentParser(
        prog="canonicalize-labels",
        description=(
            "Populate listings.canonical_make / canonical_model from the raw "
            "make / model fields (Phase 4.5 alias map + Title Case fallback) "
            "AND listings.generation_year from the raw year (Phase 4.6 4-year "
            "bucketing). Idempotent: re-runs skip rows already fully populated "
            "unless --rebuild is passed. Phase 5 baseline + training read "
            "these columns exclusively; you MUST run this CLI before "
            "phase5-baseline / phase5-train."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"path to the crawler SQLite DB (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "restrict the pass to one listings.source value (e.g. "
            "'compcars'). If omitted, every row is considered."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of rows visited this run (smoke-test hatch)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "re-canonicalize every row, including those whose "
            "canonical_make is already non-NULL. Use after editing the "
            "alias map."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def _select_rows(
    conn: sqlite3.Connection,
    *,
    source: str | None,
    rebuild: bool,
    limit: int | None,
) -> list[tuple[str, str | None, str | None, int | None]]:
    """Return ``(listing_id, make, model, year)`` rows that need canonicalization.

    When ``rebuild`` is False, rows are skipped only if BOTH
    ``canonical_make`` AND ``generation_year`` are already populated --
    so a Phase 4.5 DB whose canonical_* columns are already filled in
    will still get its ``generation_year`` backfilled on the next run.
    When True, every row matching ``source`` / ``limit`` is returned
    regardless of current canonical state.

    The fetched ``year`` is the raw integer column; the CLI computes
    the bucket start year via :func:`year_to_generation` before the
    UPDATE.
    """
    clauses: list[str] = []
    params: list[object] = []
    if source is not None:
        clauses.append("source = ?")
        params.append(source)
    if not rebuild:
        # Phase 4.6: a row counts as "already canonicalized" only when
        # BOTH the make/model AND the generation bucket are filled in.
        clauses.append("(canonical_make IS NULL OR generation_year IS NULL)")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""
    sql = (
        f"SELECT listing_id, make, model, year FROM listings "
        f"{where} ORDER BY listing_id {limit_sql}"
    ).strip()
    cur = conn.execute(sql, params)
    return [(str(r["listing_id"]), r["make"], r["model"], r["year"]) for r in cur.fetchall()]


def _update_canonical(
    conn: sqlite3.Connection,
    *,
    rows: list[tuple[str, str | None, str | None, int | None]],
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> int:
    """Apply the canonical-label + generation-year transforms and UPDATE.

    For each row we write:

    * ``canonical_make`` <- :func:`normalize_make` of the raw make
    * ``canonical_model`` <- :func:`normalize_model` of the raw model
    * ``generation_year`` <- :func:`year_to_generation` of the raw year
      (the 4-year bucket start year; Phase 4.6).

    Progress is logged every ``progress_interval`` rows. Returns the
    number of rows actually written.
    """
    n_updated = 0
    sql = (
        "UPDATE listings "
        "SET canonical_make = :canonical_make, "
        "    canonical_model = :canonical_model, "
        "    generation_year = :generation_year "
        "WHERE listing_id = :listing_id"
    )
    with conn:
        for i, (listing_id, raw_make, raw_model, raw_year) in enumerate(rows, start=1):
            canonical_make = normalize_make(raw_make)
            canonical_model = normalize_model(raw_model)
            generation_year = year_to_generation(raw_year)
            conn.execute(
                sql,
                {
                    "listing_id": listing_id,
                    "canonical_make": canonical_make,
                    "canonical_model": canonical_model,
                    "generation_year": generation_year,
                },
            )
            n_updated += 1
            if i % progress_interval == 0:
                logger.info(
                    "canonicalize-labels: processed %d / %d rows",
                    i,
                    len(rows),
                )
    return n_updated


def _final_stats(
    conn: sqlite3.Connection,
    *,
    source: str | None,
) -> tuple[int, Counter[str]]:
    """Return ``(total_rows, canonical_make_counts)`` for the final summary."""
    where = "WHERE source = ?" if source is not None else ""
    params: tuple[object, ...] = (source,) if source is not None else ()
    total = conn.execute(f"SELECT COUNT(*) AS n FROM listings {where}", params).fetchone()["n"]
    counts_sql = (
        f"SELECT canonical_make AS m, COUNT(*) AS n FROM listings {where} GROUP BY canonical_make"
    )
    counts: Counter[str] = Counter()
    for row in conn.execute(counts_sql, params).fetchall():
        key = row["m"] if row["m"] is not None else "<NULL>"
        counts[str(key)] = int(row["n"])
    return int(total), counts


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``canonicalize-labels`` console script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.limit is not None and args.limit <= 0:
        parser.error(f"--limit must be > 0, got {args.limit}")

    db_path: Path = args.db
    if not db_path.exists():
        parser.error(f"db path does not exist: {db_path}")

    conn = open_db(db_path)
    try:
        rows = _select_rows(
            conn,
            source=args.source,
            rebuild=args.rebuild,
            limit=args.limit,
        )
        logger.info(
            "canonicalize-labels: %d candidate rows (source=%s rebuild=%s limit=%s)",
            len(rows),
            args.source,
            args.rebuild,
            args.limit,
        )
        n_updated = _update_canonical(conn, rows=rows)
        total_rows, counts = _final_stats(conn, source=args.source)
    finally:
        conn.close()

    distinct = sum(1 for k in counts if k != "<NULL>")
    n_null = counts.get("<NULL>", 0)
    print(
        "canonicalize-labels: "
        f"total_rows={total_rows} "
        f"updated={n_updated} "
        f"distinct_canonical_makes={distinct} "
        f"null_canonical={n_null}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
