"""Console script for the VMMRdb ingest (Phase 4.2).

Invoke via the ``import-vmmrdb`` entry point declared in ``pyproject.toml``::

    import-vmmrdb [--db PATH] [--out-dir PATH] [--catalog PATH]
                  [--split train|val|test] [--limit N]
                  [--hf-dataset NAME] [-v]

Streams a VMMRdb Hugging Face mirror, normalizes each class string into
``(year, make, model)``, and inserts one listing + one image row per image
into the crawler DB. Idempotent — re-runs skip rows already present.

The default ``--hf-dataset`` targets the train partition of the
``venetis/VMMRdb_make_model_*`` family. Override with ``--hf-dataset`` +
``--split`` to ingest the val/test partitions. Note that the
``venetis`` mirror exposes a single ``"train"`` HF split per dataset id;
the value of ``--split`` is recorded in ``listings.split`` for
downstream training joins.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from car_lense_engine.db import open_db

from .vmmrdb import import_vmmrdb

DEFAULT_DB = Path("db/crawl.sqlite")
DEFAULT_OUT_DIR = Path("data/public/vmmrdb")
DEFAULT_CATALOG = Path("catalog/classes.json")
DEFAULT_HF_DATASET = "venetis/VMMRdb_make_model_train"
DEFAULT_SPLIT = "train"
SPLIT_CHOICES: tuple[str, ...] = ("train", "val", "test")

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser for the ``import-vmmrdb`` command."""
    parser = argparse.ArgumentParser(
        prog="import-vmmrdb",
        description=(
            "Stream the VMMRdb dataset from a Hugging Face mirror, normalize "
            "each class string into (year, make, model), and insert one "
            "synthetic listing + one image row per image into the crawler "
            "DB. Idempotent."
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
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help=f"path to the NHTSA catalog classes.json (default: {DEFAULT_CATALOG})",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=SPLIT_CHOICES,
        default=DEFAULT_SPLIT,
        help=(
            f"semantic split tag recorded in listings.split (default: {DEFAULT_SPLIT}). "
            "Pair with --hf-dataset on the venetis mirrors: train -> _train, "
            "val -> _val, test -> _test."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of rows ingested this run (smoke-test hatch)",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=DEFAULT_HF_DATASET,
        help=f"Hugging Face dataset id (default: {DEFAULT_HF_DATASET})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``import-vmmrdb`` console script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.limit is not None and args.limit <= 0:
        parser.error(f"--limit must be > 0, got {args.limit}")

    catalog_path: Path = args.catalog
    if not catalog_path.exists():
        parser.error(f"catalog path does not exist: {catalog_path}")

    db_path: Path = args.db
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = open_db(db_path)
    try:
        stats = import_vmmrdb(
            conn=conn,
            out_dir=args.out_dir,
            catalog_path=catalog_path,
            hf_dataset=args.hf_dataset,
            split=args.split,
            limit=args.limit,
        )
    finally:
        conn.close()

    print(
        "import-vmmrdb: "
        f"processed={stats.processed} "
        f"inserted_listings={stats.inserted_listings} "
        f"inserted_images={stats.inserted_images} "
        f"skipped_existing={stats.skipped_existing} "
        f"skipped_parse_failures={stats.skipped_parse_failures}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
