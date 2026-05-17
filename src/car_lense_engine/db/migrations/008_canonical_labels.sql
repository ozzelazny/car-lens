-- Add canonical_make and canonical_model columns to listings (Phase 4.5).
--
-- Cross-source training requires a single canonical class_id per
-- (make, model, year). Today the same logical class has different
-- ``listings.make`` strings depending on source:
--   - crawled:  "Chevrolet"   (Title Case)
--   - Stanford: "chevrolet"   (lowercase)
--   - CompCars: "Chevy"       (alias) or "BWM" (typo) or "MAZDA" (all caps)
--
-- This migration adds two nullable TEXT columns that the
-- ``canonicalize-labels`` CLI populates via a hand-curated alias map +
-- Title Case fallback. The original raw ``make`` / ``model`` fields are
-- preserved for audit; downstream training joins (Phase 5) target the
-- canonical columns instead.
--
-- ``ALTER TABLE ADD COLUMN`` is safe in SQLite (no rebuild needed),
-- so unlike migrations 004 / 006 / 007 we don't need the
-- rename-rebuild-drop dance.
--
-- The partial index covers the common training-side query (filter rows
-- whose canonical fields are populated, group by canonical class id).

ALTER TABLE listings ADD COLUMN canonical_make TEXT;
ALTER TABLE listings ADD COLUMN canonical_model TEXT;

CREATE INDEX IF NOT EXISTS idx_listings_canonical_class
    ON listings (canonical_make, canonical_model, year)
    WHERE canonical_make IS NOT NULL AND canonical_model IS NOT NULL;
