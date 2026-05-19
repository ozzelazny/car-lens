-- Phase 3.5 — per-image train/val/test split.
--
-- The Phase 5.2 training and Phase 5.1 baseline both currently filter on
-- ``listings.split`` (migration 005). That granularity is wrong: a single
-- listing can have a front shot + a rear shot, and the spec calls for splits
-- stratified by ``(class, view)`` — so the split must live per-image, not
-- per-listing.
--
-- This migration adds the per-image column; the ``make-splits`` CLI
-- (``car_lense_engine.dataset.make_splits``) populates it. Non-exterior rows
-- (interior, detail, non-car, NULL view) are intentionally left NULL — they
-- stay in the DB but are excluded from training.
--
-- The partial index covers the Phase 5 training-side query (filter rows whose
-- split is populated and group by split).
--
-- ``ALTER TABLE ADD COLUMN`` is safe in SQLite (no rebuild needed), mirroring
-- migrations 8 and 9.

ALTER TABLE images ADD COLUMN split TEXT;

CREATE INDEX IF NOT EXISTS idx_images_split
    ON images (split) WHERE split IS NOT NULL;
