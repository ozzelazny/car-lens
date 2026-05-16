"""Stanford Cars dataset ingest (Phase 4.1).

Streams the Stanford Cars dataset from a Hugging Face mirror via
``datasets.load_dataset``, parses each class string into structured
``(year, make, model, body_style)`` using
:mod:`car_lense_engine.dataset.stanford_cars_labels`, and persists each
image to ``data/public/stanford_cars/<class_id>/<sha256>.jpg`` while
inserting one synthetic listing + one image row into the SQLite DB.

Design choices:

* **One listing per image** — Stanford gives us a class label per image;
  there's no real "listing" container, so we synthesize ``listing_id =
  f"stanford_cars:{image_id}"``. Mirrors how the crawler treats listings:
  a stable PK independent of the gallery position.
* **Content-addressed storage** — same SHA-256-of-bytes convention as the
  crawler's :class:`ImageDownloader`. Future pHash / dedupe stages (Phase
  3.2) don't need to special-case Stanford vs crawled data.
* **Synthetic URL** — ``stanford_cars://<class_id>/<image_id>`` keeps the
  ``listings.url UNIQUE`` constraint satisfied without inventing fake HTTP
  URLs that could mislead downstream tooling.
* **Streaming load** — ``streaming=True`` so a 16k-image dataset isn't
  fully materialized in memory; we iterate row-by-row.
* **Idempotent** — re-running over the same dataset skips already-present
  on-disk files and already-inserted DB rows. Stats track the difference.
* **Lazy HF import** — importing this module never touches the
  ``datasets`` library; the import happens inside
  :func:`import_stanford_cars` so plain ``pytest`` collection (and code
  paths that never run the ingest) don't require a HF install.

The view labeler is NOT auto-run after ingest. Run
``view-label --source stanford_cars`` separately when you want per-image
view labels.
"""

from __future__ import annotations

import hashlib
import io
import json
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

from .stanford_cars_labels import (
    StanfordCarsLabel,
    StanfordCarsParseError,
    parse_class,
)

logger = logging.getLogger(__name__)

_SOURCE: str = "stanford_cars"
_JPEG_QUALITY: int = 95


@dataclass(frozen=True)
class ImportStats:
    """Per-run ingest counters."""

    processed: int = 0
    inserted_listings: int = 0
    inserted_images: int = 0
    skipped_existing: int = 0
    skipped_parse_failures: int = 0


def import_stanford_cars(
    *,
    conn: sqlite3.Connection,
    out_dir: pathlib.Path,
    catalog_path: pathlib.Path,
    hf_dataset: str = "Multimodal-Fatima/StanfordCars_train",
    split: str = "train",
    limit: int | None = None,
    log_every: int = 500,
) -> ImportStats:
    """Stream Stanford Cars from Hugging Face into the crawler DB + ``out_dir``.

    Each row's PIL image is JPEG-encoded at quality 95, hashed
    (``image_id = sha256(bytes)``), and written atomically to
    ``out_dir / <class_id> / <image_id>.jpg``. One listing row
    (``source='stanford_cars'``, synthetic ``stanford_cars://`` URL) and one
    image row (``position=1``, no pHash yet) are inserted per image.

    Parameters
    ----------
    conn:
        Open SQLite connection (migrations applied).
    out_dir:
        Root directory where images will be written. Typically
        ``data/public/stanford_cars``.
    catalog_path:
        Path to ``catalog/classes.json``; used to seed the known-make set
        for the label parser.
    hf_dataset:
        Hugging Face dataset id. Defaults to the canonical Stanford Cars
        train mirror.
    split:
        Dataset split to ingest (``"train"`` or ``"test"``).
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

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    known_makes = _load_known_makes(catalog_path)
    logger.info(
        "stanford_cars: loaded %d known makes from %s",
        len(known_makes),
        catalog_path,
    )

    stream = _open_stream(hf_dataset=hf_dataset, split=split)
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
            label = parse_class(raw_class, known_makes)
        except StanfordCarsParseError as exc:
            skipped_parse_failures += 1
            logger.warning(
                "stanford_cars: parse failure (skipped): err=%s row_keys=%r",
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

        listing_id = f"stanford_cars:{image_id}"

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
        "stanford_cars: done: processed=%d inserted_listings=%d inserted_images=%d "
        "skipped_existing=%d skipped_parse_failures=%d",
        stats.processed,
        stats.inserted_listings,
        stats.inserted_images,
        stats.skipped_existing,
        stats.skipped_parse_failures,
    )
    return stats


# --------------------------------------------------------------- internals


def _open_stream(*, hf_dataset: str, split: str) -> Any:
    """Lazy-import ``datasets.load_dataset`` and open the streaming split."""
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dep listed in pyproject
        raise RuntimeError(
            "the 'datasets' package is required to ingest Stanford Cars; "
            "install it via `uv pip install -e .`"
        ) from exc
    logger.info("stanford_cars: opening HF stream dataset=%s split=%s", hf_dataset, split)
    return load_dataset(hf_dataset, split=split, streaming=True)


def _load_known_makes(catalog_path: pathlib.Path) -> set[str]:
    """Read ``classes.json`` and return the set of canonical make names."""
    path = pathlib.Path(catalog_path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    makes_field = data.get("makes")
    if not isinstance(makes_field, list):
        raise ValueError(
            f"catalog at {path} has no 'makes' list (got {type(makes_field).__name__})"
        )
    out: set[str] = set()
    for entry in makes_field:
        if isinstance(entry, dict):
            name = entry.get("make_name")
            if isinstance(name, str) and name.strip():
                out.add(name.strip())
    return out


def _make_label_decoder(ds: Any) -> Callable[[dict[str, Any]], str]:
    """Build a function that turns whatever the mirror puts in 'label' into a class string.

    Some mirrors emit string class names directly under ``class`` /
    ``label_name``. Others (notably ``Multimodal-Fatima/StanfordCars_train``)
    emit only an integer ``ClassLabel`` under ``label``, which must be
    resolved via ``ds.features['label'].int2str``.

    The returned callable first tries pre-resolved string fields; if those
    are absent it falls through to the integer path using the captured
    feature decoders. Raises :class:`StanfordCarsParseError` if neither
    path yields a class string.
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
        raise StanfordCarsParseError(f"row has no usable class field (keys={sorted(row.keys())!r})")

    return decode


def _encode_image(row: dict[str, Any]) -> bytes:
    """JPEG-encode the PIL image at row['image'] into bytes (quality=95)."""
    image_obj = row.get("image")
    if image_obj is None:
        raise StanfordCarsParseError("row has no 'image' field")
    try:
        from PIL import Image as PILImage  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dep listed in pyproject
        raise RuntimeError("Pillow is required") from exc

    # HF returns a PIL.Image.Image; if a mirror gave us raw bytes, accept them too.
    if isinstance(image_obj, (bytes, bytearray)):
        return bytes(image_obj)

    if not isinstance(image_obj, PILImage.Image):
        raise StanfordCarsParseError(f"unsupported image type: {type(image_obj).__name__}")

    buf = io.BytesIO()
    pil_image = image_obj
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return buf.getvalue()


def _class_id_for(label: StanfordCarsLabel) -> str:
    """Derive a stable on-disk class directory name from the structured label.

    Format: ``<year>_<make>_<model>[_<body_style>]`` with spaces and
    filesystem-hostile characters replaced by underscores. Stable across
    runs because it's a pure function of the parsed label.
    """
    parts = [str(label.year), label.make, label.model]
    if label.body_style:
        parts.append(label.body_style)
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
    label: StanfordCarsLabel,
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
        source="stanford_cars",
        url=f"stanford_cars://{class_id}/{image_id}",
        year=label.year,
        make=label.make,
        model=label.model,
        body_style=label.body_style,
    )
    try:
        listings.insert_listing(conn, listing)
    except sqlite3.IntegrityError as exc:
        logger.debug("stanford_cars: listing insert race for %s: %r", listing_id, exc)
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
        source_url=f"stanford_cars://{class_id}/{image_id}",
        local_path=str(target_path),
        bytes=byte_count,
        position=1,
    )
    try:
        images.insert_image(conn, image)
    except sqlite3.IntegrityError as exc:
        logger.debug("stanford_cars: image insert race for %s: %r", image_id[:12], exc)
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
        "stanford_cars: progress processed=%d inserted_listings=%d "
        "inserted_images=%d skipped_existing=%d skipped_parse_failures=%d",
        processed,
        inserted_listings,
        inserted_images,
        skipped_existing,
        skipped_parse_failures,
    )
