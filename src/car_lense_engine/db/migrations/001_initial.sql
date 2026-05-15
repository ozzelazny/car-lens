-- Initial schema for the Car Lense crawler database.
-- All tables live in a single SQLite file (default: db/crawl.sqlite).

CREATE TABLE IF NOT EXISTS listings (
    listing_id      TEXT PRIMARY KEY,
    source          TEXT NOT NULL CHECK (source IN (
                        'cars_com',
                        'autotrader',
                        'craigslist',
                        'bat',
                        'hemmings',
                        'carsandbids'
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

CREATE INDEX IF NOT EXISTS idx_listings_source_ymm
    ON listings (source, year, make, model);

CREATE INDEX IF NOT EXISTS idx_listings_vin
    ON listings (vin) WHERE vin IS NOT NULL;


CREATE TABLE IF NOT EXISTS images (
    image_id        TEXT PRIMARY KEY,
    listing_id      TEXT NOT NULL
                    REFERENCES listings (listing_id) ON DELETE CASCADE,
    source_url      TEXT NOT NULL,
    local_path      TEXT NOT NULL,
    phash           TEXT,
    width           INTEGER,
    height          INTEGER,
    bytes           INTEGER,
    position        INTEGER,
    downloaded_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_images_listing
    ON images (listing_id);

CREATE INDEX IF NOT EXISTS idx_images_phash
    ON images (phash) WHERE phash IS NOT NULL;


CREATE TABLE IF NOT EXISTS crawl_queue (
    url             TEXT PRIMARY KEY,
    source          TEXT NOT NULL CHECK (source IN (
                        'cars_com',
                        'autotrader',
                        'craigslist',
                        'bat',
                        'hemmings',
                        'carsandbids'
                    )),
    kind            TEXT NOT NULL CHECK (kind IN ('search', 'listing', 'image')),
    target_year     INTEGER,
    target_make     TEXT,
    target_model    TEXT,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'in_progress', 'done', 'failed', 'dead')),
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    next_try_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    enqueued_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    claimed_at      TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_queue_poll
    ON crawl_queue (status, next_try_at, source);

CREATE INDEX IF NOT EXISTS idx_queue_source_kind
    ON crawl_queue (source, kind);


CREATE TABLE IF NOT EXISTS dedupe_phash (
    phash                   TEXT PRIMARY KEY,
    representative_image_id TEXT NOT NULL REFERENCES images (image_id),
    cluster_size            INTEGER NOT NULL DEFAULT 1
);
