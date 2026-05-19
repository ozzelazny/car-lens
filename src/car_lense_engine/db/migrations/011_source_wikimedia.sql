-- Widen the listings.source CHECK constraint to include 'wikimedia_commons'.
--
-- Wikimedia Commons is ingested directly into listings + images (one
-- synthetic listing per image) and never goes through crawl_queue, so we
-- only widen listings.source here. crawl_queue.source is left untouched.
--
-- This migration uses the SAFER rebuild-table pattern (no DROP-before-RENAME
-- window): the old and new tables coexist by their final names until the
-- very last statement, so a crash at any point leaves recoverable data.
-- Same shape as migrations 006 (vmmrdb) and 007 (compcars).
--
-- IMPORTANT — legacy_alter_table interaction with foreign keys:
-- Modern SQLite (>= 3.25) auto-rewrites FK references in OTHER tables when
-- a referenced table is RENAMEd. images.listing_id has an FK to
-- listings(listing_id); without intervention, ``ALTER TABLE listings RENAME
-- TO listings_old`` would silently rewrite the FK on images to point at
-- listings_old, and subsequent INSERTs into images would crash with
-- "no such table: main.listings_old". To prevent that, we set
-- ``PRAGMA legacy_alter_table = ON`` for the duration of the migration so
-- RENAME is a pure metadata operation that leaves dependent FKs untouched.
--
-- Migrations 008 (canonical_labels), 009 (generation_year), and 010
-- (images_split) added columns via plain ALTER TABLE ADD COLUMN and did
-- not rebuild the table; we preserve those columns by listing them
-- explicitly in the INSERT SELECT below.

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
                        'compcars',
                        'wikimedia_commons'
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
    split           TEXT,
    canonical_make  TEXT,
    canonical_model TEXT,
    generation_year INTEGER
);

INSERT INTO listings (
    listing_id, source, url, year, make, model, trim, body_style,
    mileage, vin, raw_html_sha256, scraped_at, split,
    canonical_make, canonical_model, generation_year
)
SELECT
    listing_id, source, url, year, make, model, trim, body_style,
    mileage, vin, raw_html_sha256, scraped_at, split,
    canonical_make, canonical_model, generation_year
FROM listings_old;

DROP TABLE listings_old;

CREATE INDEX IF NOT EXISTS idx_listings_source_ymm
    ON listings (source, year, make, model);

CREATE INDEX IF NOT EXISTS idx_listings_vin
    ON listings (vin) WHERE vin IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_listings_source_split
    ON listings (source, split);

CREATE INDEX IF NOT EXISTS idx_listings_canonical_class
    ON listings (canonical_make, canonical_model, year)
    WHERE canonical_make IS NOT NULL AND canonical_model IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_listings_canonical_generation
    ON listings (canonical_make, canonical_model, generation_year)
    WHERE canonical_make IS NOT NULL AND canonical_model IS NOT NULL AND generation_year IS NOT NULL;

PRAGMA legacy_alter_table = OFF;
PRAGMA foreign_keys = ON;
