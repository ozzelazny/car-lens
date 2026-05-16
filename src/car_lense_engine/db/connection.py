"""SQLite connection helper with pragma setup and inline migration runner."""

from __future__ import annotations

import logging
import sqlite3
from importlib import resources
from pathlib import Path

logger = logging.getLogger(__name__)

# Ordered list of (version, sql_resource_name) pairs. Append new migrations here.
_MIGRATIONS: list[tuple[int, str]] = [
    (1, "001_initial.sql"),
    (2, "002_image_listing_link.sql"),
    (3, "003_image_view_label.sql"),
    (4, "004_source_stanford_cars.sql"),
]


def open_db(path: Path | str) -> sqlite3.Connection:
    """Open (and if needed create) the crawler SQLite database.

    Sets the project's standard pragmas (WAL journal, foreign keys on,
    NORMAL synchronous) and applies any pending migrations. Safe to call
    repeatedly: the migration runner is idempotent.

    The returned connection uses ``sqlite3.Row`` as its row factory.
    """
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        db_path,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        isolation_level="DEFERRED",
    )
    conn.row_factory = sqlite3.Row

    # Pragmas. Some (journal_mode) persist on the file; others are per-connection.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")

    _apply_migrations(conn)
    return conn


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Run any migrations newer than the current schema_version."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (  version INTEGER NOT NULL PRIMARY KEY)"
    )
    cur = conn.execute("SELECT COALESCE(MAX(version), 0) AS v FROM schema_version")
    row = cur.fetchone()
    current_version: int = int(row["v"]) if row is not None else 0

    for version, resource_name in _MIGRATIONS:
        if version <= current_version:
            continue
        sql = _read_migration_sql(resource_name)
        logger.info("Applying migration %d (%s)", version, resource_name)
        with conn:  # transaction
            conn.executescript(sql)
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))


def _read_migration_sql(resource_name: str) -> str:
    """Load a migration SQL file from the packaged ``migrations`` resources."""
    pkg = "car_lense_engine.db.migrations"
    return resources.files(pkg).joinpath(resource_name).read_text(encoding="utf-8")
