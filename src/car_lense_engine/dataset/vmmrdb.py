"""VMMRdb dataset ingest (Phase 4.2).

Streams the VMMRdb (Vehicle Make and Model Recognition Database) from a
Hugging Face mirror via ``datasets.load_dataset``, parses each class string
into structured ``(year, make, model)`` using
:mod:`car_lense_engine.dataset.vmmrdb_labels`, and persists each image to
``data/public/vmmrdb/<class_id>/<sha256>.jpg`` while inserting one synthetic
listing + one image row per image into the SQLite DB.

Design choices (mirror :mod:`stanford_cars`):

* **One listing per image** — VMMRdb gives us a class label per image; we
  synthesize ``listing_id = f"vmmrdb:{image_id}"``.
* **Content-addressed storage** — same SHA-256-of-bytes convention as the
  crawler's :class:`ImageDownloader` and the Stanford Cars ingest. Future
  dedupe stages (Phase 3.2) don't need to special-case VMMRdb.
* **Synthetic URL** — ``vmmrdb://<class_id>/<image_id>`` keeps the
  ``listings.url UNIQUE`` constraint satisfied without inventing fake HTTP
  URLs.
* **Streaming load** — VMMRdb is ~292k images on the original release (the
  ``venetis`` HF mirror trims to 375 make-model classes). ``streaming=True``
  so we iterate row-by-row without materializing the whole dataset.
* **Idempotent** — re-running over the same dataset skips already-present
  on-disk files and already-inserted DB rows.
* **Lazy HF import** — :func:`import_vmmrdb` imports ``datasets`` lazily so
  plain ``pytest`` collection doesn't require the HF install path.
* **Higher log frequency** — VMMRdb is bigger than Stanford Cars, so the
  default log cadence is 1000 rows (not 500).

The view labeler is NOT auto-run after ingest. Run
``view-label --source vmmrdb`` separately when you want per-image view labels.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import pathlib
import sqlite3
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from car_lense_engine.db import images, listings
from car_lense_engine.db.models import Image, Listing

from .vmmrdb_labels import VmmrdbLabel, VmmrdbParseError, parse_class

logger = logging.getLogger(__name__)

_SOURCE: str = "vmmrdb"
_JPEG_QUALITY: int = 95


@dataclass(frozen=True)
class ImportStats:
    """Per-run ingest counters."""

    processed: int = 0
    inserted_listings: int = 0
    inserted_images: int = 0
    skipped_existing: int = 0
    skipped_parse_failures: int = 0


def import_vmmrdb(
    *,
    conn: sqlite3.Connection,
    out_dir: pathlib.Path,
    catalog_path: pathlib.Path,
    hf_dataset: str,
    split: str = "train",
    limit: int | None = None,
    log_every: int = 1000,
) -> ImportStats:
    """Stream VMMRdb from Hugging Face into the crawler DB + ``out_dir``.

    Each row's PIL image is JPEG-encoded at quality 95, hashed
    (``image_id = sha256(bytes)``), and written atomically to
    ``out_dir / <class_id> / <image_id>.jpg``. One listing row
    (``source='vmmrdb'``, synthetic ``vmmrdb://`` URL) and one image row
    (``position=1``, no pHash yet) are inserted per image.

    Parameters
    ----------
    conn:
        Open SQLite connection (migrations applied).
    out_dir:
        Root directory where images will be written. Typically
        ``data/public/vmmrdb``.
    catalog_path:
        Path to ``catalog/classes.json``; accepted for symmetry with
        :func:`stanford_cars.import_stanford_cars` and to confirm the file
        exists before we start a long ingest. Not consulted by the label
        parser (VMMRdb make matching is single-token first-of-underscore;
        no longest-prefix lookup needed).
    hf_dataset:
        Hugging Face dataset id. Required — VMMRdb has multiple distinct
        mirrors (per-split: ``venetis/VMMRdb_make_model_train`` /
        ``_val`` / ``_test``) and no single canonical name.
    split:
        Dataset split to record in ``listings.split``. Note that the
        ``venetis`` mirror exposes only a single ``"train"`` split per
        dataset id; pass ``split="val"`` when ingesting the ``_val`` mirror
        so the row is labelled correctly downstream.
    limit:
        If given, stop after this many rows have been *processed*. Useful
        for smoke tests.
    log_every:
        Emit a progress log line every ``log_every`` processed rows.

    Returns
    -------
    ImportStats
        Per-run counters. ``processed`` counts every row pulled from the
        stream; the rest are decomposed sub-counts.
    """
    if log_every <= 0:
        raise ValueError(f"log_every must be > 0, got {log_every!r}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be > 0 or None, got {limit!r}")

    catalog_path = pathlib.Path(catalog_path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"catalog file does not exist: {catalog_path}")

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stream = _open_stream(hf_dataset=hf_dataset, split=_stream_split(hf_dataset, split))
    decode_class = _make_label_decoder(stream)

    processed = 0
    inserted_listings = 0
    inserted_images = 0
    skipped_existing = 0
    skipped_parse_failures = 0

    for row in stream:
        if limit is not None and processed >= limit:
            break
        processed += 1

        try:
            raw_class = decode_class(row)
            label = parse_class(raw_class)
        except VmmrdbParseError as exc:
            skipped_parse_failures += 1
            logger.warning(
                "vmmrdb: parse failure (skipped): err=%s row_keys=%r",
                exc,
                sorted(row.keys()),
            )
            if processed % log_every == 0:
                _log_progress(
                    processed,
                    inserted_listings,
                    inserted_images,
                    skipped_existing,
                    skipped_parse_failures,
                )
            continue

        body = _encode_image(row)
        image_id = hashlib.sha256(body).hexdigest()

        class_id = _class_id_for(label)
        target_path = out_dir / class_id / f"{image_id}.jpg"

        listing_id = f"vmmrdb:{image_id}"

        # Idempotency: if the image row already exists, skip the whole row.
        # We still ensure the file is on disk in case the previous run
        # crashed between write + insert.
        if images.get_image_by_sha(conn, image_id) is not None:
            skipped_existing += 1
            if processed % log_every == 0:
                _log_progress(
                    processed,
                    inserted_listings,
                    inserted_images,
                    skipped_existing,
                    skipped_parse_failures,
                )
            continue

        _atomic_write_bytes(target_path, body)

        listing_inserted = _insert_listing_if_new(
            conn,
            listing_id=listing_id,
            class_id=class_id,
            image_id=image_id,
            label=label,
            split=split,
        )
        if listing_inserted:
            inserted_listings += 1

        image_inserted = _insert_image_if_new(
            conn,
            image_id=image_id,
            listing_id=listing_id,
            class_id=class_id,
            target_path=target_path,
            byte_count=len(body),
        )
        if image_inserted:
            inserted_images += 1

        if processed % log_every == 0:
            _log_progress(
                processed,
                inserted_listings,
                inserted_images,
                skipped_existing,
                skipped_parse_failures,
            )

    stats = ImportStats(
        processed=processed,
        inserted_listings=inserted_listings,
        inserted_images=inserted_images,
        skipped_existing=skipped_existing,
        skipped_parse_failures=skipped_parse_failures,
    )
    logger.info(
        "vmmrdb: done: processed=%d inserted_listings=%d inserted_images=%d "
        "skipped_existing=%d skipped_parse_failures=%d",
        stats.processed,
        stats.inserted_listings,
        stats.inserted_images,
        stats.skipped_existing,
        stats.skipped_parse_failures,
    )
    return stats


# --------------------------------------------------------------- internals


def _stream_split(hf_dataset: str, split: str) -> str:
    """Translate the semantic ``listings.split`` label to the actual HF split name.

    The ``venetis/VMMRdb_make_model_*`` mirrors expose a single ``"train"``
    split per dataset id, even for the ``_val`` and ``_test`` variants
    (the partition lives in the dataset id, not the split name). Callers
    pass the *semantic* split (``"train"`` / ``"val"`` / ``"test"``) via
    the ``split=`` kwarg so it gets recorded correctly in
    ``listings.split``, but the underlying stream must be opened with the
    split name that mirror actually exposes.

    For the ``venetis/VMMRdb_make_model_*`` mirrors, that's always ``"train"``.
    Any other mirror is assumed to honour the semantic name as-given.
    """
    if hf_dataset.startswith("venetis/VMMRdb_make_model"):
        return "train"
    return split


def _open_stream(*, hf_dataset: str, split: str) -> Any:
    """Lazy-import ``datasets.load_dataset`` and open the streaming split."""
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dep listed in pyproject
        raise RuntimeError(
            "the 'datasets' package is required to ingest VMMRdb; "
            "install it via `uv pip install -e .`"
        ) from exc
    logger.info("vmmrdb: opening HF stream dataset=%s split=%s", hf_dataset, split)
    return load_dataset(hf_dataset, split=split, streaming=True)


def _make_label_decoder(ds: Any) -> Callable[[dict[str, Any]], str]:
    """Build a function that turns whatever the mirror puts in 'label' into a class string.

    Some mirrors emit string class names directly under ``class`` /
    ``label_name``. Others (notably ``venetis/VMMRdb_make_model_*``) emit
    only an integer ``ClassLabel`` under ``label``, which must be resolved
    via ``ds.features['label'].int2str``.

    The returned callable first tries pre-resolved string fields; if those
    are absent it falls through to the integer path using the captured
    feature decoders. Raises :class:`VmmrdbParseError` if neither path
    yields a class string.
    """
    features = getattr(ds, "features", None) or {}
    # Map column-name -> ClassLabel.int2str. Duck-typed so a stub
    # ``_FakeClassLabel`` works without importing ``datasets``.
    int2str: dict[str, Callable[[int], str]] = {}
    for key, feat in features.items():
        if hasattr(feat, "int2str") and hasattr(feat, "names"):
            int2str[key] = feat.int2str

    def decode(row: dict[str, Any]) -> str:
        # 1) Direct string fields first (mirrors that pre-resolve labels).
        for k in ("class", "label_name", "labels", "label"):
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # 2) Int fields routed through the captured feature decoder.
        for k in ("label", "labels"):
            v = row.get(k)
            # ``bool`` is a subclass of ``int``; exclude it explicitly so we
            # never try to look up True/False as a class index.
            if isinstance(v, int) and not isinstance(v, bool) and k in int2str:
                return str(int2str[k](v))
        raise VmmrdbParseError(f"row has no usable class field (keys={sorted(row.keys())!r})")

    return decode


def _encode_image(row: dict[str, Any]) -> bytes:
    """JPEG-encode the PIL image at row['image'] into bytes (quality=95)."""
    image_obj = row.get("image")
    if image_obj is None:
        raise VmmrdbParseError("row has no 'image' field")
    try:
        from PIL import Image as PILImage  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dep listed in pyproject
        raise RuntimeError("Pillow is required") from exc

    # HF returns a PIL.Image.Image; if a mirror gave us raw bytes, accept them too.
    if isinstance(image_obj, (bytes, bytearray)):
        return bytes(image_obj)

    if not isinstance(image_obj, PILImage.Image):
        raise VmmrdbParseError(f"unsupported image type: {type(image_obj).__name__}")

    buf = io.BytesIO()
    pil_image = image_obj
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return buf.getvalue()


def _class_id_for(label: VmmrdbLabel) -> str:
    """Derive a stable on-disk class directory name from the structured label.

    Format: ``<year>_<make>_<model>`` when year is present, otherwise
    ``<make>_<model>``. Spaces and filesystem-hostile characters are replaced
    by underscores so the slug is always a safe directory name.
    """
    parts: list[str] = []
    if label.year is not None:
        parts.append(str(label.year))
    parts.append(label.make)
    parts.append(label.model)
    raw = "_".join(parts)
    return _slugify(raw)


def _slugify(text: str) -> str:
    """Replace whitespace + filesystem-hostile characters with underscores."""
    out_chars: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            out_chars.append(ch)
        else:
            out_chars.append("_")
    # Collapse runs of underscores for readability.
    slug = "".join(out_chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _atomic_write_bytes(path: pathlib.Path, body: bytes) -> None:
    """Write ``body`` to ``path`` via a ``.tmp`` rename. Skip if file exists.

    Same convention as the crawler's :func:`image_downloader._atomic_write_bytes`
    — we trust the SHA-256 in the filename and never overwrite.
    """
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
        # Clean up the tmp file so a retry isn't blocked by a stale .tmp.
        if tmp.exists():
            with suppress(OSError):
                tmp.unlink()
        raise


def _insert_listing_if_new(
    conn: sqlite3.Connection,
    *,
    listing_id: str,
    class_id: str,
    image_id: str,
    label: VmmrdbLabel,
    split: str,
) -> bool:
    """Insert a listing row, returning True if a new row was created.

    A pre-check via ``get_listing`` covers the idempotent re-run case; an
    IntegrityError on PK / unique-URL still returns False so two ingests
    racing don't crash.
    """
    from car_lense_engine.db.listings import get_listing  # noqa: PLC0415

    if get_listing(conn, listing_id) is not None:
        return False

    listing = Listing(
        listing_id=listing_id,
        source="vmmrdb",
        url=f"vmmrdb://{class_id}/{image_id}",
        year=label.year,
        make=label.make,
        model=label.model,
        split=split,
    )
    try:
        listings.insert_listing(conn, listing)
    except sqlite3.IntegrityError as exc:
        logger.debug("vmmrdb: listing insert race for %s: %r", listing_id, exc)
        return False
    return True


def _insert_image_if_new(
    conn: sqlite3.Connection,
    *,
    image_id: str,
    listing_id: str,
    class_id: str,
    target_path: pathlib.Path,
    byte_count: int,
) -> bool:
    """Insert an image row, returning True if a new row was created."""
    if images.get_image_by_sha(conn, image_id) is not None:
        return False

    image = Image(
        image_id=image_id,
        listing_id=listing_id,
        source_url=f"vmmrdb://{class_id}/{image_id}",
        local_path=str(target_path),
        bytes=byte_count,
        position=1,
    )
    try:
        images.insert_image(conn, image)
    except sqlite3.IntegrityError as exc:
        logger.debug("vmmrdb: image insert race for %s: %r", image_id[:12], exc)
        return False
    return True


def _log_progress(
    processed: int,
    inserted_listings: int,
    inserted_images: int,
    skipped_existing: int,
    skipped_parse_failures: int,
) -> None:
    logger.info(
        "vmmrdb: progress processed=%d inserted_listings=%d "
        "inserted_images=%d skipped_existing=%d skipped_parse_failures=%d",
        processed,
        inserted_listings,
        inserted_images,
        skipped_existing,
        skipped_parse_failures,
    )
