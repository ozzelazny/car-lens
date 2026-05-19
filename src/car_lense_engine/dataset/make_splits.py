"""Stratified train/val/test split over the per-image view labels (Phase 3.5).

The Phase 5 baseline + training pipelines need a split assignment that is:

1. **Per-image, not per-listing.** A listing can contribute a front shot AND
   a rear shot; stratifying at the image level keeps each ``(class, view)``
   cell evenly represented across train/val/test.
2. **Stratified by ``(canonical_make, canonical_model, generation_year, view)``.**
   This is the actual recognition class we train on (Phase 4.6's bucketed
   class id, joined with the per-image view label).
3. **Listing-coherent by default.** Two photos of the same physical car
   trivially leak signal; we assign whole listings to a single split inside
   each ``(class, view)`` cell so the val/test sets measure generalization
   to *unseen vehicles*, not just unseen shots of seen vehicles.
4. **Deterministic.** Seeded shuffling so re-runs against the same DB +
   eligible set produce the same assignment.

Eligible rows are exterior photos (``view`` in the 5 exterior categories)
whose listing has both canonical_make and canonical_model populated. Other
rows — interior, detail, non-car, NULL view, or rows missing canonical
labels — are left with ``images.split = NULL`` and silently dropped from
training/eval downstream.

The module is pure stdlib + sqlite3; no torch / model load. It plays nicely
with an in-memory DB in tests.
"""

from __future__ import annotations

import logging
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# The 5 exterior view labels eligible for the train/val/test pool.
_ELIGIBLE_VIEWS: tuple[str, ...] = (
    "front",
    "rear",
    "side",
    "three-quarter-front",
    "three-quarter-rear",
)

# ``split`` values written to the DB.
_TRAIN: str = "train"
_VAL: str = "val"
_TEST: str = "test"

# Per-cell key type: (canonical_make, canonical_model, generation_year, view).
# generation_year can be NULL for sources that don't carry year info (e.g.
# the VMMRdb venetis HF mirror); we still want to stratify by the rest of
# the key so we keep it as Optional[int] inside the key tuple.
_CellKey = tuple[str, str, int | None, str]


@dataclass(frozen=True)
class SplitSummary:
    """Aggregate counters returned by :func:`make_splits`.

    All counts refer to **images**, not listings. ``per_view_counts`` is a
    mapping from view name to ``{"train": N, "val": N, "test": N}``.
    ``per_class_coverage_p10`` is the 10th-percentile of per-class image
    counts (i.e. how thin the long tail of classes is); useful for spotting
    classes that fell entirely into the "small group -> all train" bucket.
    """

    total_eligible: int
    total_train: int
    total_val: int
    total_test: int
    total_skipped_small_group: int
    total_excluded_non_exterior: int
    per_view_counts: dict[str, dict[str, int]]
    per_class_coverage_p10: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot of the summary."""
        return {
            "total_eligible": self.total_eligible,
            "total_train": self.total_train,
            "total_val": self.total_val,
            "total_test": self.total_test,
            "total_skipped_small_group": self.total_skipped_small_group,
            "total_excluded_non_exterior": self.total_excluded_non_exterior,
            "per_view_counts": self.per_view_counts,
            "per_class_coverage_p10": self.per_class_coverage_p10,
        }


@dataclass(frozen=True)
class _ImageRow:
    """One eligible image row read out of the DB."""

    image_id: str
    listing_id: str
    canonical_make: str
    canonical_model: str
    generation_year: int | None
    view: str


def make_splits(
    conn: sqlite3.Connection,
    *,
    source: str,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    min_group_size: int = 10,
    rebuild: bool = False,
    listing_coherent: bool = True,
    dry_run: bool = False,
) -> SplitSummary:
    """Compute (and persist) the per-image train/val/test split.

    Parameters
    ----------
    conn:
        Open SQLite connection with the Phase 3.5 schema applied (migration
        010 must be present so ``images.split`` exists).
    source:
        Listings.source filter (e.g. ``"compcars"``). Only rows whose
        listing belongs to this source are considered.
    seed:
        Seed for the deterministic shuffle of listings inside each
        ``(class, view)`` cell.
    train_frac, val_frac:
        Target proportions for the train and val splits. Test is the
        remainder. Must satisfy ``0 < train_frac + val_frac <= 1``.
    min_group_size:
        Cells with fewer than this many *images* get **all** their rows in
        ``train`` (no eval signal for that cell, but the class still
        participates in training).
    rebuild:
        If False (default), only rows whose ``images.split`` is currently
        NULL get an assignment; existing assignments are left untouched.
        If True, every eligible row's split is recomputed and overwritten.
    listing_coherent:
        If True (default), whole listings are assigned to a single split
        inside each ``(class, view)`` cell — so two photos of the same
        physical car can never appear in different splits. If False, each
        image is shuffled independently (faster, but allows cross-split
        leakage of the same vehicle).
    dry_run:
        If True, compute the split but skip the DB write.

    Returns
    -------
    SplitSummary
        Aggregate counters. The summary reflects the assignment that was
        *computed*, even when ``dry_run`` is True.
    """
    if not 0.0 < train_frac + val_frac <= 1.0:
        raise ValueError(
            "train_frac + val_frac must be in (0, 1]; "
            f"got train_frac={train_frac!r}, val_frac={val_frac!r}"
        )
    if train_frac < 0 or val_frac < 0:
        raise ValueError(
            f"train_frac and val_frac must be non-negative; got "
            f"train_frac={train_frac!r}, val_frac={val_frac!r}"
        )
    if min_group_size <= 0:
        raise ValueError(f"min_group_size must be > 0, got {min_group_size!r}")

    excluded_non_exterior = _count_excluded_non_exterior(conn, source=source)
    eligible_rows = _fetch_eligible_rows(conn, source=source)
    logger.info(
        "make-splits: %d eligible rows (source=%s); %d non-exterior excluded",
        len(eligible_rows),
        source,
        excluded_non_exterior,
    )

    # Bucket eligible rows by (class, view) cell.
    cells: dict[_CellKey, list[_ImageRow]] = defaultdict(list)
    for row in eligible_rows:
        cell_key: _CellKey = (
            row.canonical_make,
            row.canonical_model,
            row.generation_year,
            row.view,
        )
        cells[cell_key].append(row)

    # Compute the assignment per cell.
    assignments: dict[str, str] = {}
    skipped_small_group_total = 0
    rng = random.Random(seed)
    for _cell_key, cell_rows in cells.items():
        cell_assignments, n_small = _assign_cell(
            cell_rows=cell_rows,
            train_frac=train_frac,
            val_frac=val_frac,
            min_group_size=min_group_size,
            listing_coherent=listing_coherent,
            rng=rng,
        )
        skipped_small_group_total += n_small
        assignments.update(cell_assignments)

    # Tally summary BEFORE we filter for rebuild — the summary reports what
    # we *would* assign on a clean run, regardless of the idempotency knob.
    summary = _build_summary(
        eligible_rows=eligible_rows,
        assignments=assignments,
        excluded_non_exterior=excluded_non_exterior,
        skipped_small_group=skipped_small_group_total,
    )

    if dry_run:
        logger.info("make-splits: dry-run; skipping DB writes")
        return summary

    # Filter to rows we actually want to write back.
    rows_to_write: list[tuple[str, str]] = _select_writes(
        conn=conn,
        assignments=assignments,
        rebuild=rebuild,
    )
    if not rows_to_write:
        logger.info("make-splits: nothing to write (rebuild=%s)", rebuild)
        return summary

    logger.info(
        "make-splits: writing %d image rows (rebuild=%s)",
        len(rows_to_write),
        rebuild,
    )
    with conn:
        conn.executemany(
            "UPDATE images SET split = ? WHERE image_id = ?",
            [(split, image_id) for image_id, split in rows_to_write],
        )
    return summary


# --------------------------------------------------------------- internals


def _fetch_eligible_rows(
    conn: sqlite3.Connection,
    *,
    source: str,
) -> list[_ImageRow]:
    """Return all eligible image rows for the source, ordered deterministically.

    Eligibility is the same gating used by Phase 5 readers: exterior view,
    canonical_make and canonical_model both non-NULL. We also pull
    generation_year (nullable) and view so the stratification key is
    complete.
    """
    placeholders = ", ".join(["?"] * len(_ELIGIBLE_VIEWS))
    sql = (
        "SELECT images.image_id AS image_id, "
        "       images.listing_id AS listing_id, "
        "       images.view AS view, "
        "       listings.canonical_make AS canonical_make, "
        "       listings.canonical_model AS canonical_model, "
        "       listings.generation_year AS generation_year "
        "FROM images "
        "JOIN listings ON listings.listing_id = images.listing_id "
        f"WHERE images.view IN ({placeholders}) "
        "  AND listings.canonical_make IS NOT NULL "
        "  AND listings.canonical_model IS NOT NULL "
        "  AND listings.source = ? "
        "ORDER BY images.image_id"
    )
    params: list[object] = [*_ELIGIBLE_VIEWS, source]
    cur = conn.execute(sql, params)
    rows: list[_ImageRow] = []
    for r in cur.fetchall():
        rows.append(
            _ImageRow(
                image_id=str(r["image_id"]),
                listing_id=str(r["listing_id"]),
                canonical_make=str(r["canonical_make"]),
                canonical_model=str(r["canonical_model"]),
                generation_year=(
                    int(r["generation_year"]) if r["generation_year"] is not None else None
                ),
                view=str(r["view"]),
            )
        )
    return rows


def _count_excluded_non_exterior(conn: sqlite3.Connection, *, source: str) -> int:
    """Return the count of rows we deliberately leave with ``split = NULL``.

    These are non-exterior rows (interior, detail, non-car, NULL view) or
    rows whose listing has no canonical label. They stay in the DB but are
    invisible to training/eval — reported as a separate counter so the
    operator can sanity-check the labeler ran fully.
    """
    placeholders = ", ".join(["?"] * len(_ELIGIBLE_VIEWS))
    sql = (
        "SELECT COUNT(*) AS n FROM images "
        "JOIN listings ON listings.listing_id = images.listing_id "
        f"WHERE listings.source = ? "
        f"  AND (images.view IS NULL OR images.view NOT IN ({placeholders}) "
        "       OR listings.canonical_make IS NULL "
        "       OR listings.canonical_model IS NULL)"
    )
    params: list[object] = [source, *_ELIGIBLE_VIEWS]
    row = conn.execute(sql, params).fetchone()
    return int(row["n"]) if row is not None else 0


def _assign_cell(
    *,
    cell_rows: list[_ImageRow],
    train_frac: float,
    val_frac: float,
    min_group_size: int,
    listing_coherent: bool,
    rng: random.Random,
) -> tuple[dict[str, str], int]:
    """Assign every image in one ``(class, view)`` cell to a split.

    Returns ``(assignments, n_small_group_images)`` — the latter is non-zero
    only when the whole cell fell into the "below ``min_group_size``, all
    train" bucket.
    """
    n_images = len(cell_rows)
    if n_images == 0:  # pragma: no cover - filtered upstream
        return {}, 0

    # Small-group rule: dump everything in train, no val/test for this cell.
    if n_images < min_group_size:
        return {row.image_id: _TRAIN for row in cell_rows}, n_images

    if listing_coherent:
        return _assign_cell_listing_coherent(
            cell_rows=cell_rows,
            train_frac=train_frac,
            val_frac=val_frac,
            rng=rng,
        ), 0
    return _assign_cell_image_level(
        cell_rows=cell_rows,
        train_frac=train_frac,
        val_frac=val_frac,
        rng=rng,
    ), 0


def _assign_cell_listing_coherent(
    *,
    cell_rows: list[_ImageRow],
    train_frac: float,
    val_frac: float,
    rng: random.Random,
) -> dict[str, str]:
    """Assign whole listings (not individual images) to a single split.

    Shuffle the distinct listing ids deterministically, then walk the
    sorted list and assign whole listings to train until the train target
    is met (by **image count**), then val, then test. Each ``(class, view)``
    cell uses its own pass over the rng — but because ``rng`` is a single
    seeded ``random.Random`` consumed in dict-iteration order across cells,
    the overall assignment is fully deterministic when the eligible row set
    is unchanged.
    """
    # Group images by listing_id.
    images_by_listing: dict[str, list[_ImageRow]] = defaultdict(list)
    for row in cell_rows:
        images_by_listing[row.listing_id].append(row)

    listing_ids = sorted(images_by_listing.keys())
    rng.shuffle(listing_ids)

    n_total = len(cell_rows)
    n_train_target = int(round(train_frac * n_total))
    n_val_target = int(round(val_frac * n_total))

    assignments: dict[str, str] = {}
    train_count = 0
    val_count = 0
    for listing_id in listing_ids:
        listing_images = images_by_listing[listing_id]
        if train_count < n_train_target:
            split = _TRAIN
            train_count += len(listing_images)
        elif val_count < n_val_target:
            split = _VAL
            val_count += len(listing_images)
        else:
            split = _TEST
        for row in listing_images:
            assignments[row.image_id] = split
    return assignments


def _assign_cell_image_level(
    *,
    cell_rows: list[_ImageRow],
    train_frac: float,
    val_frac: float,
    rng: random.Random,
) -> dict[str, str]:
    """Assign each image independently (no listing coherency).

    Shuffle the image ids, take the first ``train_frac`` for train, the
    next ``val_frac`` for val, the rest for test. Allows two photos of the
    same vehicle to land in different splits.
    """
    shuffled = sorted(row.image_id for row in cell_rows)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(round(train_frac * n_total))
    n_val = int(round(val_frac * n_total))
    assignments: dict[str, str] = {}
    for i, image_id in enumerate(shuffled):
        if i < n_train:
            assignments[image_id] = _TRAIN
        elif i < n_train + n_val:
            assignments[image_id] = _VAL
        else:
            assignments[image_id] = _TEST
    return assignments


def _build_summary(
    *,
    eligible_rows: list[_ImageRow],
    assignments: dict[str, str],
    excluded_non_exterior: int,
    skipped_small_group: int,
) -> SplitSummary:
    """Tally the per-view / per-class counters for the JSON report."""
    per_view_counts: dict[str, dict[str, int]] = {
        v: {_TRAIN: 0, _VAL: 0, _TEST: 0} for v in _ELIGIBLE_VIEWS
    }
    total_train = 0
    total_val = 0
    total_test = 0
    per_class_counts: Counter[tuple[str, str, int | None]] = Counter()
    for row in eligible_rows:
        split = assignments.get(row.image_id)
        if split is None:  # pragma: no cover - every eligible row gets assigned
            continue
        per_view_counts[row.view][split] = per_view_counts[row.view].get(split, 0) + 1
        if split == _TRAIN:
            total_train += 1
        elif split == _VAL:
            total_val += 1
        elif split == _TEST:
            total_test += 1
        per_class_counts[(row.canonical_make, row.canonical_model, row.generation_year)] += 1

    p10 = _percentile(sorted(per_class_counts.values()), 10) if per_class_counts else 0
    return SplitSummary(
        total_eligible=len(eligible_rows),
        total_train=total_train,
        total_val=total_val,
        total_test=total_test,
        total_skipped_small_group=skipped_small_group,
        total_excluded_non_exterior=excluded_non_exterior,
        per_view_counts=per_view_counts,
        per_class_coverage_p10=p10,
    )


def _percentile(sorted_values: list[int], pct: int) -> int:
    """Return the ``pct``-th percentile of ``sorted_values`` (nearest-rank).

    Uses the nearest-rank method (no interpolation) since the values are
    integer image counts and we just want a coarse "how thin is the tail".
    """
    if not sorted_values:
        return 0
    if not 0 <= pct <= 100:
        raise ValueError(f"pct must be in [0, 100], got {pct!r}")
    # Nearest-rank: index = ceil(pct/100 * N) - 1, clamped to [0, N-1].
    n = len(sorted_values)
    idx = max(0, min(n - 1, (pct * n + 99) // 100 - 1))
    return sorted_values[idx]


def _select_writes(
    *,
    conn: sqlite3.Connection,
    assignments: dict[str, str],
    rebuild: bool,
) -> list[tuple[str, str]]:
    """Return ``(image_id, split)`` rows that should actually be UPDATE'd.

    When ``rebuild`` is True, every computed assignment is written.
    Otherwise, only rows whose ``images.split`` is currently NULL are
    written — preserving any prior hand-edited assignments.
    """
    if not assignments:
        return []
    if rebuild:
        return list(assignments.items())

    # Pull the current split for each image we're about to touch and skip
    # the ones that already carry a value.
    image_ids = list(assignments.keys())
    existing: set[str] = set()
    # Batch the SELECT so we don't blow past SQLite's 999-variable limit.
    batch_size = 500
    for start in range(0, len(image_ids), batch_size):
        batch = image_ids[start : start + batch_size]
        placeholders = ", ".join(["?"] * len(batch))
        sql = (
            f"SELECT image_id FROM images WHERE image_id IN ({placeholders}) AND split IS NOT NULL"
        )
        cur = conn.execute(sql, batch)
        for row in cur.fetchall():
            existing.add(str(row["image_id"]))
    return [
        (image_id, split) for image_id, split in assignments.items() if image_id not in existing
    ]


__all__ = [
    "SplitSummary",
    "make_splits",
]
