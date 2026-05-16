-- Phase 3.3 — CLIP zero-shot view + content labeling.
-- Adds the per-image view label (and confidence score) populated by the
-- `view-label` CLI. All three columns are nullable so existing rows remain
-- valid until the labeler is run; idx_images_view is a partial index so we
-- can cheaply filter "labeled vs unlabeled" rows.
ALTER TABLE images ADD COLUMN view TEXT;
ALTER TABLE images ADD COLUMN view_score REAL;
ALTER TABLE images ADD COLUMN view_labeled_at TIMESTAMP;

CREATE INDEX IF NOT EXISTS idx_images_view
    ON images (view) WHERE view IS NOT NULL;
