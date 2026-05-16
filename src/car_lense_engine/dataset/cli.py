"""Console script for the CLIP zero-shot view + content labeler.

Invoke via the ``view-label`` entry point declared in ``pyproject.toml``::

    view-label [--db PATH] [--rebuild] [--batch-size N]
               [--device cpu|cuda|mps] [--model ViT-L-14]
               [--pretrained laion2b_s32b_b82k] [--source SOURCE_ID]
               [--limit N] [-v]

Idempotent by default: rows where ``view IS NOT NULL`` are skipped unless
``--rebuild`` is passed.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from car_lense_engine.db import open_db

from .view_labeler import ViewLabel, ViewLabeler

DEFAULT_DB = Path("db/crawl.sqlite")
DEFAULT_BATCH_SIZE = 16
DEFAULT_MODEL = "ViT-L-14"
DEFAULT_PRETRAINED = "laion2b_s32b_b82k"
DEFAULT_DEVICE = "cpu"
DEVICE_CHOICES: tuple[str, ...] = ("cpu", "cuda", "mps")
PROGRESS_INTERVAL = 100

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser for the ``view-label`` command."""
    parser = argparse.ArgumentParser(
        prog="view-label",
        description=(
            "Run the OpenCLIP zero-shot view + content labeler over every "
            "image row in the crawler DB. Idempotent: rows that already have "
            "a view are skipped unless --rebuild is set."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"path to the crawler SQLite DB (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="re-label every row, even those that already have a view",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"images per OpenCLIP forward pass (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=DEVICE_CHOICES,
        default=DEFAULT_DEVICE,
        help=f"torch device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenCLIP model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=DEFAULT_PRETRAINED,
        help=f"OpenCLIP pretrained tag (default: {DEFAULT_PRETRAINED})",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="restrict to images whose listing belongs to this source (e.g. cars_com)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of images labeled this run (smoke-test hatch)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def _select_targets(
    conn: sqlite3.Connection,
    *,
    rebuild: bool,
    source: str | None,
    limit: int | None,
) -> list[tuple[str, Path]]:
    """Return ``(image_id, local_path)`` tuples for rows to label.

    Joined against ``listings`` so the ``--source`` filter actually filters by
    the listing's source — the images table doesn't carry the source itself.
    """
    clauses: list[str] = []
    params: list[object] = []
    if not rebuild:
        clauses.append("images.view IS NULL")
    if source is not None:
        clauses.append("listings.source = ?")
        params.append(source)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""
    sql = (
        "SELECT images.image_id AS image_id, images.local_path AS local_path "
        "FROM images "
        "JOIN listings ON listings.listing_id = images.listing_id "
        f"{where} "
        "ORDER BY images.downloaded_at, images.image_id "
        f"{limit_sql}"
    ).strip()
    cur = conn.execute(sql, params)
    return [(str(row["image_id"]), Path(str(row["local_path"]))) for row in cur.fetchall()]


def _persist_labels(
    conn: sqlite3.Connection,
    rows: Iterable[tuple[str, ViewLabel]],
) -> None:
    """Persist a batch of ``(image_id, ViewLabel)`` rows in one transaction.

    Uses ``CURRENT_TIMESTAMP`` server-side so all rows in this batch share a
    consistent ``view_labeled_at``.
    """
    payload = [
        {"image_id": image_id, "view": label.view, "view_score": label.score}
        for image_id, label in rows
    ]
    if not payload:
        return
    with conn:
        conn.executemany(
            "UPDATE images "
            "SET view = :view, view_score = :view_score, "
            "    view_labeled_at = CURRENT_TIMESTAMP "
            "WHERE image_id = :image_id",
            payload,
        )


def _print_distribution(counts: Counter[str], total: int) -> None:
    """Print a per-view distribution summary to stdout."""
    if total == 0:
        print("view-label: no rows labeled")
        return
    print(f"view-label: labeled {total} image(s)")
    width = max((len(v) for v in counts), default=0)
    # Sort by count desc, then view name for stable output.
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    for view, n in ordered:
        pct = 100.0 * n / total
        print(f"  {view:<{width}}  {n:>6d}  ({pct:5.1f}%)")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``view-label`` console script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.batch_size <= 0:
        parser.error(f"--batch-size must be > 0, got {args.batch_size}")
    if args.limit is not None and args.limit <= 0:
        parser.error(f"--limit must be > 0, got {args.limit}")

    db_path: Path = args.db
    if not db_path.exists():
        parser.error(f"DB path does not exist: {db_path}")

    conn = open_db(db_path)
    counts: Counter[str] = Counter()
    total_labeled = 0
    try:
        targets = _select_targets(
            conn,
            rebuild=args.rebuild,
            source=args.source,
            limit=args.limit,
        )
        if not targets:
            logger.info(
                "view-label: nothing to do (rebuild=%s source=%s limit=%s)",
                args.rebuild,
                args.source or "<all>",
                args.limit,
            )
            print("view-label: nothing to do")
            return 0

        logger.info(
            "view-label: labeling %d image(s) "
            "(model=%s pretrained=%s device=%s batch=%d rebuild=%s source=%s)",
            len(targets),
            args.model,
            args.pretrained,
            args.device,
            args.batch_size,
            args.rebuild,
            args.source or "<all>",
        )

        with ViewLabeler(
            model_name=args.model,
            pretrained=args.pretrained,
            device=args.device,
            batch_size=args.batch_size,
        ) as labeler:
            # Process in batch-sized slices so progress logging is meaningful
            # and DB writes stay small.
            batch = args.batch_size
            # Map path -> image_id for the slice so we can re-associate
            # ``label_batch`` results (which now skip bad / missing files)
            # with the right DB row. Paths are unique within a slice because
            # ``_select_targets`` returns one row per image_id.
            for start in range(0, len(targets), batch):
                slice_ = targets[start : start + batch]
                path_to_image_id = {p: image_id for image_id, p in slice_}
                paths = [p for _, p in slice_]
                results = labeler.label_batch(paths)
                _persist_labels(
                    conn,
                    ((path_to_image_id[p], label) for p, label in results),
                )
                for _, label in results:
                    counts[label.view] += 1
                total_labeled += len(results)
                if total_labeled % PROGRESS_INTERVAL < batch:
                    logger.info("view-label: progress %d / %d", total_labeled, len(targets))
    finally:
        conn.close()

    _print_distribution(counts, total_labeled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
