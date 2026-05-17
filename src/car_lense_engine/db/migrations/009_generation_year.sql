-- Add generation_year column to listings (Phase 4.6).
--
-- Phase 5.2 fine-tune on the CompCars baseline (4,421 classes) hit
-- top-1 = 64.76%. The dominant confusion mode is year-pairs of the
-- SAME model -- adjacent model years are visually near-identical
-- because car redesigns happen on a ~4-year cycle, not yearly. Top
-- confusion pairs:
--   - BYD Qin 2012 <-> 2014           (7 swaps)
--   - Kia K2 sedan 2012 <-> 2015      (6 swaps)
--   - Zotye V10 2011 <-> 2012         (6 swaps)
--
-- Fix: collapse year into 4-year "generation buckets" anchored at
-- 1980. ``generation_year`` stores the BUCKET START YEAR (an integer)
-- so the same redesign cycle rolls up into a single class:
--   year=2012 -> generation_year=2012  (bucket 2012-2015)
--   year=2014 -> generation_year=2012  (same bucket)
--   year=2016 -> generation_year=2016  (next bucket)
--
-- The human-readable label ("2012-2015") is derivable from the int
-- column at display time; we don't bake the label string into the DB.
-- The raw ``year`` column is preserved for audit / display purposes.
--
-- ``ALTER TABLE ADD COLUMN`` is safe in SQLite (no rebuild needed),
-- mirroring migration 8.
--
-- The partial index covers the Phase 5 training-side query (filter
-- rows whose canonical + generation fields are populated, group by
-- the new bucketed class id).

ALTER TABLE listings ADD COLUMN generation_year INTEGER;

CREATE INDEX IF NOT EXISTS idx_listings_canonical_generation
    ON listings (canonical_make, canonical_model, generation_year)
    WHERE canonical_make IS NOT NULL AND canonical_model IS NOT NULL AND generation_year IS NOT NULL;
