-- Widen the listings.source CHECK constraint to include 'compcars'.
--
-- CompCars is ingested from a Hugging Face ZIP archive (one synthetic listing
-- per image) and never goes through crawl_queue, so we only widen
-- listings.source here. crawl_queue.source is left untouched.
--
-- This migration uses the SAFER rebuild-table pattern (no DROP-before-RENAME
-- window): the old and new tables coexist by their final names until the very
-- last statement, so a crash at any point leaves recoverable data. Same shape
-- as migration 006 (vmmrdb).
--
-- IMPORTANT — legacy_alter_table interaction with foreign keys:
-- Modern SQLite (>= 3.25) auto-rewrites FK references in OTHER tables when a
-- referenced table is RENAMEd. images.listing_id has an FK to
-- listings(listing_id); without intervention, ``ALTER TABLE listings RENAME
-- TO listings_old`` would silently rewrite the FK on images to point at
-- listings_old, and subsequent INSERTs into images would crash with
-- "no such table: main.listings_old". To prevent that, we set
-- ``PRAGMA legacy_alter_table = ON`` for the duration of the migration so
-- RENAME is a pure metadata operation that leaves dependent FKs untouched.
--
-- Sequence:
--   1. PRAGMA foreign_keys = OFF so dropping listings_old doesn't cascade
--      into images via the FK on images.listing_id.
--   2. PRAGMA legacy_alter_table = ON so RENAME doesn't rewrite FKs in
--      images. (See note above.)
--   3. DROP TABLE IF EXISTS listings_old — defensive; covers the case where a
--      previous run of this migration crashed between RENAME and DROP.
--   4. ALTER TABLE listings RENAME TO listings_old — the old data is now under
--      a stable temporary name, NOT dropped.
--   5. CREATE TABLE listings (new schema) — the canonical name now points at
--      an empty table with the widened CHECK.
--   6. INSERT INTO listings SELECT * FROM listings_old — copy data over. Both
--      tables exist; at no point is the data on disk only in a soon-to-be-
--      dropped table.
--   7. DROP TABLE listings_old — last step; from here on, the old copy is
--      gone but the new one is fully populated and queryable.
--   8. CREATE INDEX (re-create the secondary indexes).
--   9. PRAGMA legacy_alter_table = OFF (restore default).
--  10. PRAGMA foreign_keys = ON.

PRAGMA foreign_keys = OFF;
PRAGMA legacy_alter_table = ON;

DROP TABLE IF EXISTS listings_old;

ALTER TABLE listings RENAME TO listings_old;

CREATE TABLE listings (
    listing_id      TEXT PRIMARY KEY,
    source          TEXT NOT NULL CHECK (source IN (
                        'cars_com',
                        'autotrader',
                        'craigslist',
                        'bat',
                        'hemmings',
                        'carsandbids',
                        'stanford_cars',
                        'vmmrdb',
                        'compcars'
                    )),
    url             TEXT NOT NULL UNIQUE,
    year            INTEGER,
    make            TEXT,
    model           TEXT,
    trim            TEXT,
    body_style      TEXT,
    mileage         INTEGER,
    vin             TEXT,
    raw_html_sha256 TEXT,
    scraped_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    split           TEXT
);

INSERT INTO listings (
    listing_id, source, url, year, make, model, trim, body_style,
    mileage, vin, raw_html_sha256, scraped_at, split
)
SELECT
    listing_id, source, url, year, make, model, trim, body_style,
    mileage, vin, raw_html_sha256, scraped_at, split
FROM listings_old;

DROP TABLE listings_old;

CREATE INDEX IF NOT EXISTS idx_listings_source_ymm
    ON listings (source, year, make, model);

CREATE INDEX IF NOT EXISTS idx_listings_vin
    ON listings (vin) WHERE vin IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_listings_source_split
    ON listings (source, split);

PRAGMA legacy_alter_table = OFF;
PRAGMA foreign_keys = ON;
