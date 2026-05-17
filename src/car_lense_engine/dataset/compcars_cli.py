"""Console script for the CompCars ingest (Phase 4.3).

Invoke via the ``import-compcars`` entry point declared in ``pyproject.toml``::

    import-compcars [--db PATH] [--out-dir PATH] [--zip-path PATH]
                    [--hf-repo NAME] [--hf-filename NAME]
                    [--split train|val|test] [--limit N] [-v]

Downloads (or re-uses) the ``JorgeLlorente/CompCars-Repository`` ZIP via
``huggingface_hub.hf_hub_download``, iterates the entries, and inserts one
listing + one image row per JPEG into the crawler DB. Idempotent — re-runs
skip rows already present.

The 16.5 GB download is resumable: pass ``--zip-path`` to point at a
pre-downloaded archive (e.g. for smoke-tests), or omit it to fetch into
``--out-dir``. Re-runs of ``hf_hub_download`` are content-aware and don't
re-fetch when the local copy already matches.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from car_lense_engine.db import open_db

from .compcars import import_compcars

DEFAULT_DB = Path("db/crawl.sqlite")
DEFAULT_OUT_DIR = Path("data/public/compcars")
DEFAULT_HF_REPO = "JorgeLlorente/CompCars-Repository"
DEFAULT_HF_FILENAME = "Compcars_Data.zip"
DEFAULT_SPLIT = "train"
SPLIT_CHOICES: tuple[str, ...] = ("train", "val", "test")

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser for the ``import-compcars`` command."""
    parser = argparse.ArgumentParser(
        prog="import-compcars",
        description=(
            "Ingest the CompCars dataset from a Hugging Face ZIP mirror. "
            "Downloads (resumable) or re-uses a local copy, resolves "
            "(make_id, model_id) integer IDs via the bundled .mat tables, "
            "and inserts one synthetic listing + one image row per JPEG "
            "into the crawler DB. Idempotent."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"path to the crawler SQLite DB (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"directory for written JPEGs (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help=(
            "path to a pre-downloaded Compcars_Data.zip; if omitted, the "
            "ZIP is fetched via huggingface_hub.hf_hub_download (resumable)."
        ),
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_HF_REPO,
        help=f"Hugging Face dataset id holding the ZIP (default: {DEFAULT_HF_REPO})",
    )
    parser.add_argument(
        "--hf-filename",
        type=str,
        default=DEFAULT_HF_FILENAME,
        help=f"filename inside the HF repo (default: {DEFAULT_HF_FILENAME})",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=SPLIT_CHOICES,
        default=DEFAULT_SPLIT,
        help=(
            f"semantic split tag recorded in listings.split (default: {DEFAULT_SPLIT}). "
            "CompCars has no canonical pre-defined split inside this ZIP; pick what "
            "you want downstream training joins to filter on."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of image entries ingested this run (smoke-test hatch)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``import-compcars`` console script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.limit is not None and args.limit <= 0:
        parser.error(f"--limit must be > 0, got {args.limit}")

    if args.zip_path is not None and not args.zip_path.exists():
        parser.error(f"--zip-path does not exist: {args.zip_path}")

    db_path: Path = args.db
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = open_db(db_path)
    try:
        stats = import_compcars(
            conn=conn,
            out_dir=args.out_dir,
            zip_path=args.zip_path,
            hf_repo=args.hf_repo,
            hf_filename=args.hf_filename,
            split=args.split,
            limit=args.limit,
        )
    finally:
        conn.close()

    print(
        "import-compcars: "
        f"processed={stats.processed} "
        f"inserted_listings={stats.inserted_listings} "
        f"inserted_images={stats.inserted_images} "
        f"skipped_existing={stats.skipped_existing} "
        f"skipped_parse_failures={stats.skipped_parse_failures} "
        f"skipped_no_year={stats.skipped_no_year} "
        f"skipped_unknown_class={stats.skipped_unknown_class}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
