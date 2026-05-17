"""CompCars dataset ingest (Phase 4.3).

Ingests the CompCars dataset (CUHK, Yang et al. 2015) from a Hugging Face
mirror packaged as a single 16.5 GB ZIP archive
(``JorgeLlorente/CompCars-Repository``). The on-disk layout inside the
archive is:

* ``image/<make_id>/<model_id>/<year>/<sha>.jpg`` — ~136,727 JPEGs.
* ``misc/make_model_name.mat`` — MATLAB v5 cell-arrays mapping integer IDs
  to make/model name strings.
* ``misc/car_type.mat`` — MATLAB v5 mapping model_id to a body-style index.
* ``label/`` — per-image text labels (unused; everything we need lives in
  the path + the .mat tables).

Design choices (mirror :mod:`stanford_cars` / :mod:`vmmrdb`):

* **One listing per image** — there's no "listing" container at the source;
  we synthesize ``listing_id = f"compcars:{image_id}"``.
* **Content-addressed storage** — SHA-256 of the on-disk image bytes is the
  filename and the image_id, matching the rest of the data pipeline.
* **Synthetic URL** — ``compcars://<make_id>/<model_id>/<year>/<image_id[:8]>``
  keeps ``listings.url UNIQUE`` satisfied without inventing fake HTTP URLs.
* **No streaming** — there is no HuggingFace-``datasets``-streaming mirror
  for CompCars (as of the Phase 4.3 probe). We download the ZIP via
  ``huggingface_hub.hf_hub_download`` (resumable, cached) and iterate it
  with stdlib :mod:`zipfile`. The download is skipped if ``zip_path`` is
  passed (smoke-tests + repeat-ingests).
* **Idempotent** — re-runs check ``images.get_image_by_sha`` per row and
  skip already-stored images. The on-disk file is re-written only when
  absent (atomic-rename pattern).
* **Lazy imports** — :mod:`scipy.io`, :mod:`zipfile`, and
  :mod:`huggingface_hub` are imported only when this function actually runs,
  not at module-import time. Same pattern as :mod:`view_labeler`.

The view labeler is NOT auto-run after ingest. Run
``view-label --source compcars`` separately when you want per-image view
labels.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import re
import sqlite3
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from car_lense_engine.db import images, listings
from car_lense_engine.db.models import Image, Listing

from .compcars_labels import (
    CompCarsBodyTypeTable,
    CompCarsLabelError,
    CompCarsNameTable,
    parse_image_path,
)

if TYPE_CHECKING:
    import zipfile as _zipfile_t

logger = logging.getLogger(__name__)

_SOURCE: str = "compcars"
_DEFAULT_HF_REPO: str = "JorgeLlorente/CompCars-Repository"
_DEFAULT_HF_FILENAME: str = "Compcars_Data.zip"

# Match canonical CompCars image entries inside the ZIP. We also accept a
# leading folder component (some packagings wrap the dataset under a top-level
# ``Compcars/`` or ``data/`` dir); the optional prefix is captured but
# discarded at parse time.
_IMAGE_ENTRY_RE = re.compile(r"(?:.*/)?image/\d+/\d+/[^/]+/[^/]+\.jpg$")

# Paths to the two .mat label tables inside the ZIP. Allow an optional
# leading folder prefix for the same reason.
_MAKE_MODEL_MAT_RE = re.compile(r"(?:.*/)?misc/make_model_name\.mat$")
_CAR_TYPE_MAT_RE = re.compile(r"(?:.*/)?misc/car_type\.mat$")


@dataclass(frozen=True)
class ImportStats:
    """Per-run ingest counters."""

    processed: int = 0
    inserted_listings: int = 0
    inserted_images: int = 0
    skipped_existing: int = 0
    skipped_parse_failures: int = 0
    skipped_no_year: int = 0
    skipped_unknown_class: int = 0


def import_compcars(
    *,
    conn: sqlite3.Connection,
    out_dir: pathlib.Path,
    zip_path: pathlib.Path | None = None,
    hf_repo: str = _DEFAULT_HF_REPO,
    hf_filename: str = _DEFAULT_HF_FILENAME,
    split: str = "train",
    limit: int | None = None,
    log_every: int = 1000,
) -> ImportStats:
    """Ingest CompCars from a HuggingFace-hosted ZIP into the crawler DB.

    Parameters
    ----------
    conn:
        Open SQLite connection (migrations applied).
    out_dir:
        Root directory where images will be written. Typically
        ``data/public/compcars``.
    zip_path:
        Path to a pre-downloaded ``Compcars_Data.zip``. If ``None``, the
        archive is fetched via ``huggingface_hub.hf_hub_download`` (resumable,
        cached) — the on-disk location is determined by the HF cache.
    hf_repo, hf_filename:
        Hugging Face dataset id and filename of the ZIP. Defaults to the
        ``JorgeLlorente/CompCars-Repository`` mirror confirmed by the Phase
        4.3 probe.
    split:
        Semantic split tag recorded in ``listings.split``. CompCars has no
        canonical pre-defined train/test split inside this ZIP; the caller
        passes whatever tag they want.
    limit:
        If given, stop after this many image entries have been processed.
    log_every:
        Emit a progress log line every ``log_every`` processed rows.

    Returns
    -------
    ImportStats
        Per-run counters. ``processed`` counts every image entry pulled from
        the archive; the rest are decomposed sub-counts.
    """
    if log_every <= 0:
        raise ValueError(f"log_every must be > 0, got {log_every!r}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be > 0 or None, got {limit!r}")

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the ZIP path. If None, download via huggingface_hub (resumable).
    if zip_path is None:
        zip_path = _download_zip(hf_repo=hf_repo, hf_filename=hf_filename, out_dir=out_dir)
    zip_path = pathlib.Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"CompCars ZIP not found: {zip_path}")

    zipfile = _import_zipfile()
    logger.info("compcars: opening archive %s", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        name_table, body_table = _load_label_tables(zf)
        logger.info(
            "compcars: loaded label tables: %d makes, %d models",
            len(name_table),
            name_table.model_count,
        )

        return _iterate_zip(
            zf=zf,
            conn=conn,
            out_dir=out_dir,
            name_table=name_table,
            body_table=body_table,
            split=split,
            limit=limit,
            log_every=log_every,
        )


# --------------------------------------------------------------- internals


def _download_zip(*, hf_repo: str, hf_filename: str, out_dir: pathlib.Path) -> pathlib.Path:
    """Resumable-download the CompCars ZIP via ``huggingface_hub``.

    Returns the local filesystem path. ``hf_hub_download`` is content-aware:
    a previously-completed download is re-used without re-fetching, and
    partial downloads resume from the byte offset they reached.
    """
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dep listed transitively
        raise RuntimeError(
            "huggingface_hub is required to download CompCars; install it via `uv pip install -e .`"
        ) from exc
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "compcars: downloading %s/%s via hf_hub_download (resumable)",
        hf_repo,
        hf_filename,
    )
    local_path = hf_hub_download(
        repo_id=hf_repo,
        filename=hf_filename,
        repo_type="dataset",
        local_dir=str(out_dir),
    )
    return pathlib.Path(local_path)


def _import_zipfile() -> Any:
    """Lazy-import the stdlib ``zipfile`` module."""
    import zipfile  # noqa: PLC0415

    return zipfile


def _load_label_tables(
    zf: _zipfile_t.ZipFile,
) -> tuple[CompCarsNameTable, CompCarsBodyTypeTable]:
    """Find + load the two label .mat files from the archive.

    Raises :class:`CompCarsLabelError` when either file is missing.
    """
    name_mat_bytes: bytes | None = None
    type_mat_bytes: bytes | None = None
    for name in zf.namelist():
        if name_mat_bytes is None and _MAKE_MODEL_MAT_RE.match(name):
            name_mat_bytes = zf.read(name)
        elif type_mat_bytes is None and _CAR_TYPE_MAT_RE.match(name):
            type_mat_bytes = zf.read(name)
        if name_mat_bytes is not None and type_mat_bytes is not None:
            break
    if name_mat_bytes is None:
        raise CompCarsLabelError("misc/make_model_name.mat not found in archive")
    if type_mat_bytes is None:
        raise CompCarsLabelError("misc/car_type.mat not found in archive")
    return CompCarsNameTable(name_mat_bytes), CompCarsBodyTypeTable(type_mat_bytes)


def _iterate_zip(
    *,
    zf: _zipfile_t.ZipFile,
    conn: sqlite3.Connection,
    out_dir: pathlib.Path,
    name_table: CompCarsNameTable,
    body_table: CompCarsBodyTypeTable,
    split: str,
    limit: int | None,
    log_every: int,
) -> ImportStats:
    """Stream through the ZIP, writing images + inserting rows."""
    processed = 0
    inserted_listings = 0
    inserted_images = 0
    skipped_existing = 0
    skipped_parse_failures = 0
    skipped_no_year = 0
    skipped_unknown_class = 0

    for entry_name in zf.namelist():
        if not _IMAGE_ENTRY_RE.match(entry_name):
            continue
        if limit is not None and processed >= limit:
            break
        processed += 1

        # Strip any top-level prefix so the path parser sees the canonical
        # ``image/<make_id>/<model_id>/<year>/<sha>.jpg``.
        canonical_path = _strip_leading_prefix(entry_name)

        try:
            make_id, model_id, year = parse_image_path(canonical_path)
        except CompCarsLabelError as exc:
            # Disambiguate "no/bad year" (expected for ~351 paths) from
            # otherwise-malformed paths so the stats are useful.
            msg = str(exc)
            if "year" in msg:
                skipped_no_year += 1
            else:
                skipped_parse_failures += 1
                logger.warning("compcars: parse failure (skipped): %s", exc)
            _maybe_log_progress(
                processed,
                inserted_listings,
                inserted_images,
                skipped_existing,
                skipped_parse_failures,
                skipped_no_year,
                skipped_unknown_class,
                log_every,
            )
            continue

        try:
            make, model = name_table.resolve(make_id, model_id)
        except CompCarsLabelError as exc:
            skipped_unknown_class += 1
            logger.warning(
                "compcars: unknown class (skipped) make_id=%d model_id=%d: %s",
                make_id,
                model_id,
                exc,
            )
            _maybe_log_progress(
                processed,
                inserted_listings,
                inserted_images,
                skipped_existing,
                skipped_parse_failures,
                skipped_no_year,
                skipped_unknown_class,
                log_every,
            )
            continue

        body_style = body_table.resolve(model_id)

        body = zf.read(entry_name)
        image_id = hashlib.sha256(body).hexdigest()

        class_id = _class_id_for(year=year, make=make, model=model)
        target_path = out_dir / class_id / f"{image_id}.jpg"

        listing_id = f"{_SOURCE}:{image_id}"

        # Idempotency: image already known -> skip the whole row.
        if images.get_image_by_sha(conn, image_id) is not None:
            skipped_existing += 1
            _maybe_log_progress(
                processed,
                inserted_listings,
                inserted_images,
                skipped_existing,
                skipped_parse_failures,
                skipped_no_year,
                skipped_unknown_class,
                log_every,
            )
            continue

        _atomic_write_bytes(target_path, body)

        listing_inserted = _insert_listing_if_new(
            conn,
            listing_id=listing_id,
            make_id=make_id,
            model_id=model_id,
            year=year,
            make=make,
            model=model,
            body_style=body_style,
            image_id=image_id,
            split=split,
        )
        if listing_inserted:
            inserted_listings += 1

        image_inserted = _insert_image_if_new(
            conn,
            image_id=image_id,
            listing_id=listing_id,
            make_id=make_id,
            model_id=model_id,
            year=year,
            target_path=target_path,
            byte_count=len(body),
        )
        if image_inserted:
            inserted_images += 1

        _maybe_log_progress(
            processed,
            inserted_listings,
            inserted_images,
            skipped_existing,
            skipped_parse_failures,
            skipped_no_year,
            skipped_unknown_class,
            log_every,
        )

    stats = ImportStats(
        processed=processed,
        inserted_listings=inserted_listings,
        inserted_images=inserted_images,
        skipped_existing=skipped_existing,
        skipped_parse_failures=skipped_parse_failures,
        skipped_no_year=skipped_no_year,
        skipped_unknown_class=skipped_unknown_class,
    )
    logger.info(
        "compcars: done: processed=%d inserted_listings=%d inserted_images=%d "
        "skipped_existing=%d skipped_parse_failures=%d skipped_no_year=%d "
        "skipped_unknown_class=%d",
        stats.processed,
        stats.inserted_listings,
        stats.inserted_images,
        stats.skipped_existing,
        stats.skipped_parse_failures,
        stats.skipped_no_year,
        stats.skipped_unknown_class,
    )
    return stats


def _strip_leading_prefix(entry: str) -> str:
    """Strip any leading folder prefix so the parser sees ``image/...``.

    The canonical layout is rooted at ``image/`` and ``misc/``; some
    packagings wrap them under e.g. ``Compcars_Data/`` or ``data/``. We
    drop everything before (and including) the *last* slash that precedes
    the literal ``image/`` token. Idempotent on already-canonical paths.
    """
    if entry.startswith("image/"):
        return entry
    idx = entry.rfind("/image/")
    if idx >= 0:
        return entry[idx + 1 :]
    return entry


def _class_id_for(*, year: int, make: str, model: str) -> str:
    """Build a stable on-disk class directory name from the resolved label.

    Format: ``<year>_<make>_<model>`` slugified. Stable across runs because
    it's a pure function of the resolved label.
    """
    return _slugify(f"{year}_{make}_{model}")


def _slugify(text: str) -> str:
    """Replace whitespace + filesystem-hostile characters with underscores."""
    out_chars: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            out_chars.append(ch)
        else:
            out_chars.append("_")
    slug = "".join(out_chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _atomic_write_bytes(path: pathlib.Path, body: bytes) -> None:
    """Write ``body`` to ``path`` via a ``.tmp`` rename. Skip if file exists."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError:
        if tmp.exists():
            with suppress(OSError):
                tmp.unlink()
        raise


def _insert_listing_if_new(
    conn: sqlite3.Connection,
    *,
    listing_id: str,
    make_id: int,
    model_id: int,
    year: int,
    make: str,
    model: str,
    body_style: str | None,
    image_id: str,
    split: str,
) -> bool:
    """Insert a listing row, returning True if a new row was created."""
    from car_lense_engine.db.listings import get_listing  # noqa: PLC0415

    if get_listing(conn, listing_id) is not None:
        return False

    listing = Listing(
        listing_id=listing_id,
        source="compcars",
        url=_synthetic_url(make_id=make_id, model_id=model_id, year=year, image_id=image_id),
        year=year,
        make=make,
        model=model,
        body_style=body_style,
        split=split,
    )
    try:
        listings.insert_listing(conn, listing)
    except sqlite3.IntegrityError as exc:
        logger.debug("compcars: listing insert race for %s: %r", listing_id, exc)
        return False
    return True


def _insert_image_if_new(
    conn: sqlite3.Connection,
    *,
    image_id: str,
    listing_id: str,
    make_id: int,
    model_id: int,
    year: int,
    target_path: pathlib.Path,
    byte_count: int,
) -> bool:
    """Insert an image row, returning True if a new row was created."""
    if images.get_image_by_sha(conn, image_id) is not None:
        return False

    image = Image(
        image_id=image_id,
        listing_id=listing_id,
        source_url=_synthetic_url(make_id=make_id, model_id=model_id, year=year, image_id=image_id),
        local_path=str(target_path),
        bytes=byte_count,
        position=1,
    )
    try:
        images.insert_image(conn, image)
    except sqlite3.IntegrityError as exc:
        logger.debug("compcars: image insert race for %s: %r", image_id[:12], exc)
        return False
    return True


def _synthetic_url(*, make_id: int, model_id: int, year: int, image_id: str) -> str:
    """Build the synthetic ``compcars://`` URL string.

    Mirrors the ``stanford_cars://`` / ``vmmrdb://`` convention; the
    image_id is truncated to 8 hex chars (collision-safe enough given the
    upstream paths are disambiguated by ``(make_id, model_id, year)``) so
    the URL stays readable.
    """
    return f"compcars://{make_id}/{model_id}/{year}/{image_id[:8]}"


def _maybe_log_progress(
    processed: int,
    inserted_listings: int,
    inserted_images: int,
    skipped_existing: int,
    skipped_parse_failures: int,
    skipped_no_year: int,
    skipped_unknown_class: int,
    log_every: int,
) -> None:
    """Emit a progress line every ``log_every`` processed rows."""
    if processed % log_every != 0:
        return
    logger.info(
        "compcars: progress processed=%d inserted_listings=%d "
        "inserted_images=%d skipped_existing=%d skipped_parse_failures=%d "
        "skipped_no_year=%d skipped_unknown_class=%d",
        processed,
        inserted_listings,
        inserted_images,
        skipped_existing,
        skipped_parse_failures,
        skipped_no_year,
        skipped_unknown_class,
    )
