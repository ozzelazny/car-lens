-- Add a split label to listings so train/val/test partitions are persisted.
-- Nullable: crawled rows haven't been split yet; Phase 3.5 will do that pass.
ALTER TABLE listings ADD COLUMN split TEXT;

-- Backfill: the existing stanford_cars rows were ingested via
-- `import-stanford-cars --split train`, so mark them as 'train'.
UPDATE listings SET split = 'train' WHERE source = 'stanford_cars' AND split IS NULL;

-- Index on (source, split) -- Phase 5 evaluations select by both.
CREATE INDEX IF NOT EXISTS idx_listings_source_split ON listings (source, split);
