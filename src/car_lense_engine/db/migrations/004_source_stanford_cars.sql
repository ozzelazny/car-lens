-- Widen the listings.source CHECK constraint to include 'stanford_cars'.
--
-- Stanford Cars is ingested directly into listings + images (one synthetic
-- listing per image) and never goes through crawl_queue, so we only widen
-- listings.source here. crawl_queue.source is left untouched.
--
-- SQLite cannot ALTER CHECK in place, so this is a rebuild-table migration:
-- create a new table, copy data, drop the old one, rename. Because
-- executescript() runs in autocommit mode (it commits any open transaction
-- before executing), individual statements are NOT rollback-protected.
-- The DROP TABLE IF EXISTS listings_new guard at the top makes retries safe
-- if a previous attempt crashed between CREATE and RENAME.
--
-- foreign_keys is disabled for the duration so dropping the old listings
-- table doesn't cascade into images via the FK on images.listing_id.

PRAGMA foreign_keys = OFF;

DROP TABLE IF EXISTS listings_new;

CREATE TABLE listings_new (
    listing_id      TEXT PRIMARY KEY,
    source          TEXT NOT NULL CHECK (source IN (
                        'cars_com',
                        'autotrader',
                        'craigslist',
                        'bat',
                        'hemmings',
                        'carsandbids',
                        'stanford_cars'
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
    scraped_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO listings_new (
    listing_id, source, url, year, make, model, trim, body_style,
    mileage, vin, raw_html_sha256, scraped_at
)
SELECT
    listing_id, source, url, year, make, model, trim, body_style,
    mileage, vin, raw_html_sha256, scraped_at
FROM listings;

DROP TABLE listings;

ALTER TABLE listings_new RENAME TO listings;

CREATE INDEX IF NOT EXISTS idx_listings_source_ymm
    ON listings (source, year, make, model);

CREATE INDEX IF NOT EXISTS idx_listings_vin
    ON listings (vin) WHERE vin IS NOT NULL;

PRAGMA foreign_keys = ON;
