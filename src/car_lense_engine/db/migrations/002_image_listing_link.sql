-- Link image-kind queue items back to the listing they came from so the
-- worker can build the on-disk path (<data_root>/<source>/<listing_id>/...)
-- and the images row's listing_id FK without re-fetching the parent page.
ALTER TABLE crawl_queue ADD COLUMN parent_listing_id TEXT;

CREATE INDEX IF NOT EXISTS idx_queue_parent_listing
    ON crawl_queue (parent_listing_id) WHERE parent_listing_id IS NOT NULL;
