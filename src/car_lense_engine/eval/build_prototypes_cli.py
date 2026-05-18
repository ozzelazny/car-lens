"""Console script for pre-computing prototype embeddings (Phase 6.1).

The Phase 6.1 ``recognize()`` service container needs one prototype
embedding per class to do nearest-neighbor retrieval at request time.
Building prototypes on every container start would be slow (re-embedding
all ~122k CompCars train images) and unnecessary -- prototypes are
deterministic given the fine-tuned checkpoint + the train split. This
CLI pre-computes them once and writes a portable ``.pt`` file the
service mounts read-only.

Invoke via the ``build-prototypes`` entry point declared in
``pyproject.toml``::

    build-prototypes [--db PATH] [--source compcars]
                     [--train-split train]
                     [--checkpoint PATH]
                     [--model MobileCLIP-S2] [--pretrained datacompdr]
                     [--device cpu|cuda|mps] [--batch-size N]
                     [--output PATH] [-v]

The output is a torch state dict with four keys:

* ``class_ids`` -- list[str], one ``"<year>|<make>|<model>"`` per class
  in the same order as the rows of ``prototypes``.
* ``display_names`` -- list[str], the human-readable
  ``"<year-range> <Canonical Make> <Canonical Model>"`` per class.
* ``prototypes`` -- ``torch.Tensor`` of shape ``(n_classes, embed_dim)``,
  L2-normalized, on CPU (the caller can ``.to(device)``).
* ``config`` -- dict with ``model``, ``pretrained``, ``checkpoint``,
  ``source``, ``split``, and ``built_at`` (ISO timestamp).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Any

from car_lense_engine.dataset.canonical_labels import generation_label
from car_lense_engine.db import open_db

from .baseline import BaselineConfig, build_prototypes

DEFAULT_DB = Path("db/crawl.sqlite")
DEFAULT_SOURCE = "compcars"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_MODEL = "MobileCLIP-S2"
DEFAULT_PRETRAINED = "datacompdr"
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 64
DEFAULT_OUTPUT = Path("cache/prototypes.pt")
DEVICE_CHOICES: tuple[str, ...] = ("cpu", "cuda", "mps")

logger = logging.getLogger(__name__)


def _display_name_for(class_id: str) -> str:
    """Render the human-readable display name for a class id.

    Class id format is ``"<year>|<make_lower>|<model_lower>"`` (see
    :func:`car_lense_engine.eval.baseline.class_id_for`). The year is the
    Phase 4.6 4-year bucket START year; we render the bucket as a
    dash-joined range (e.g. ``"2012-2015"``). The make / model are
    lower-cased in the id but we Title-Case them for display -- the
    canonical capitalization is lost at id-render time, so this is a
    best-effort cosmetic step. Brand-correct casing (``"BMW"``,
    ``"McLaren"``, ``"FIAT"``) is a Phase 6.3 follow-up.

    Returns the input id verbatim if the format doesn't parse -- callers
    should never see this in practice but it keeps the function total.
    """
    parts = class_id.split("|")
    if len(parts) != 3:
        return class_id
    year_str, make, model = parts
    try:
        year = int(year_str)
    except ValueError:
        return class_id
    label = generation_label(year) or str(year)
    return f"{label} {make.title()} {model.title()}"


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser for the ``build-prototypes`` command."""
    parser = argparse.ArgumentParser(
        prog="build-prototypes",
        description=(
            "Phase 6.1 prototype builder: embed every train image of a given "
            "(source, split) with a fine-tuned MobileCLIP-S2 checkpoint, "
            "mean-pool per class, L2-normalize, and write a single .pt file "
            "the recognize() service can load read-only."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"path to the crawler SQLite DB (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE,
        help=f"listings.source to build prototypes from (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default=DEFAULT_TRAIN_SPLIT,
        help=f"split used to build prototypes (default: {DEFAULT_TRAIN_SPLIT})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "optional path to a Phase 5.2 fine-tuned checkpoint .pt file; "
            "when set, the image-encoder weights are loaded from the "
            "checkpoint on top of the pretrained tag (default: none)"
        ),
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
        "--device",
        type=str,
        choices=DEVICE_CHOICES,
        default=DEFAULT_DEVICE,
        help=f"torch device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"images per forward pass (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"path for the prototypes .pt file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``build-prototypes`` console script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.batch_size <= 0:
        parser.error(f"--batch-size must be > 0, got {args.batch_size}")

    db_path: Path = args.db
    if not db_path.exists():
        parser.error(f"DB path does not exist: {db_path}")

    checkpoint_path: Path | None = args.checkpoint
    if checkpoint_path is not None and not checkpoint_path.exists():
        parser.error(f"--checkpoint path does not exist: {checkpoint_path}")

    config = BaselineConfig(
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
        checkpoint_path=checkpoint_path,
    )

    conn = open_db(db_path)
    try:
        class_ids, proto_tensor = build_prototypes(
            conn=conn,
            config=config,
            source=args.source,
            split=args.train_split,
        )
    finally:
        conn.close()

    if not class_ids:
        parser.error(
            f"no prototypes built -- no eligible train rows for "
            f"source={args.source!r} split={args.train_split!r}; "
            "run canonicalize-labels first and verify the DB has data"
        )

    import torch  # noqa: PLC0415

    display_names = [_display_name_for(cid) for cid in class_ids]
    # Always serialize on CPU so the .pt file is portable between
    # GPU-built (faster) and CPU-served (default) deployments.
    proto_cpu: Any = proto_tensor.detach().to("cpu")

    payload: dict[str, Any] = {
        "class_ids": list(class_ids),
        "display_names": display_names,
        "prototypes": proto_cpu,
        "config": {
            "model": args.model,
            "pretrained": args.pretrained,
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            "source": args.source,
            "split": args.train_split,
            "built_at": dt.datetime.now(dt.UTC).isoformat(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"build-prototypes: wrote {len(class_ids)} prototypes "
        f"(embed_dim={int(proto_cpu.shape[1])}) to {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
