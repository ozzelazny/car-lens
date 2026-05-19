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

    build-prototypes [--db PATH] [--source compcars[,vmmrdb,...]]
                     [--train-split train]
                     [--checkpoint PATH]
                     [--model MobileCLIP-S2] [--pretrained datacompdr]
                     [--device cpu|cuda|mps] [--batch-size N]
                     [--output PATH] [-v]

The ``--source`` flag accepts one or more comma-separated source names
(e.g. ``compcars,vmmrdb,stanford_cars``) so the prototype cache can
span every dataset the deployed model was trained on.

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

from .baseline import EXTERIOR_VIEWS, BaselineConfig, build_prototypes, build_prototypes_by_view

DEFAULT_DB = Path("db/crawl.sqlite")
DEFAULT_SOURCE = "compcars"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_MODEL = "MobileCLIP-S2"
DEFAULT_PRETRAINED = "datacompdr"
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 64
DEFAULT_OUTPUT = Path("cache/prototypes.pt")
DEFAULT_OUTPUT_PER_VIEW = Path("cache/prototypes_by_view.pt")
DEVICE_CHOICES: tuple[str, ...] = ("cpu", "cuda", "mps")
PROTOTYPE_SCHEMA_V2 = 2

logger = logging.getLogger(__name__)


def _parse_sources(raw: str) -> list[str]:
    """Parse a comma-separated ``--source`` argument into a non-empty list.

    Empty / whitespace-only entries (e.g. from a stray trailing comma)
    are dropped. Raises :class:`ValueError` if the final list is empty.
    """
    items = [s.strip() for s in raw.split(",") if s.strip()]
    if not items:
        raise ValueError("at least one non-empty source is required")
    return items


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
        help=(
            "one or more comma-separated ``listings.source`` values to "
            f"build prototypes from (default: {DEFAULT_SOURCE}). Examples: "
            "`compcars`, `compcars,vmmrdb`, "
            "`compcars,vmmrdb,stanford_cars`."
        ),
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
        default=None,
        help=(
            f"path for the prototypes .pt file "
            f"(default: {DEFAULT_OUTPUT} for single-prototype mode, "
            f"{DEFAULT_OUTPUT_PER_VIEW} for --per-view)"
        ),
    )
    parser.add_argument(
        "--per-view",
        action="store_true",
        help=(
            "build one prototype per (class, view) for the 5 exterior views "
            "(Phase 6.1 view-conditional retrieval). Non-exterior images "
            "are dropped at selection time. Output payload is v2 with a "
            "``prototypes_by_view`` dict; default output path is "
            f"{DEFAULT_OUTPUT_PER_VIEW} to avoid clobbering the v1 cache."
        ),
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

    try:
        sources = _parse_sources(args.source)
    except ValueError as exc:
        parser.error(f"--source: {exc}")

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

    # Pick the default output path based on the mode (per-view writes to
    # a separate file so it never clobbers the v1 single-prototype cache
    # the legacy service path still expects).
    output_path: Path
    if args.output is not None:
        output_path = args.output
    elif args.per_view:
        output_path = DEFAULT_OUTPUT_PER_VIEW
    else:
        output_path = DEFAULT_OUTPUT

    conn = open_db(db_path)
    try:
        if args.per_view:
            class_ids, prototypes_by_view = build_prototypes_by_view(
                conn=conn,
                config=config,
                source=sources,
                split=args.train_split,
            )
        else:
            class_ids, proto_tensor = build_prototypes(
                conn=conn,
                config=config,
                source=sources,
                split=args.train_split,
            )
    finally:
        conn.close()

    if not class_ids:
        parser.error(
            f"no prototypes built -- no eligible train rows for "
            f"source={sources!r} split={args.train_split!r}; "
            "run canonicalize-labels first and verify the DB has data"
        )

    import torch  # noqa: PLC0415

    display_names = [_display_name_for(cid) for cid in class_ids]
    payload: dict[str, Any]
    if args.per_view:
        # Move every per-view tensor to CPU so the .pt file is portable.
        proto_by_view_cpu: dict[str, Any] = {
            view: prototypes_by_view[view].detach().to("cpu") for view in EXTERIOR_VIEWS
        }
        # Determine the embed dim from any non-empty view tensor.
        embed_dim = 0
        for view in EXTERIOR_VIEWS:
            shape = proto_by_view_cpu[view].shape
            if len(shape) >= 2 and int(shape[1]) > 0:
                embed_dim = int(shape[1])
                break
        payload = {
            "schema_version": PROTOTYPE_SCHEMA_V2,
            "class_ids": list(class_ids),
            "display_names": display_names,
            "prototypes_by_view": proto_by_view_cpu,
            "view_names": list(EXTERIOR_VIEWS),
            "config": {
                "model": args.model,
                "pretrained": args.pretrained,
                "embed_dim": embed_dim,
                "source": ",".join(sources),
                "split": args.train_split,
                "checkpoint_path_used": (
                    str(checkpoint_path) if checkpoint_path is not None else None
                ),
                "built_at": dt.datetime.now(dt.UTC).isoformat(),
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
        view_counts = {
            view: int(
                (proto_by_view_cpu[view].norm(dim=-1) > 0).sum().item()
                if int(proto_by_view_cpu[view].shape[0]) > 0
                else 0
            )
            for view in EXTERIOR_VIEWS
        }
        print(
            f"build-prototypes: wrote per-view prototypes for {len(class_ids)} classes "
            f"(embed_dim={embed_dim}) to {output_path}; "
            f"populated rows per view: {view_counts}"
        )
        return 0

    # Always serialize on CPU so the .pt file is portable between
    # GPU-built (faster) and CPU-served (default) deployments.
    proto_cpu: Any = proto_tensor.detach().to("cpu")
    payload = {
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        f"build-prototypes: wrote {len(class_ids)} prototypes "
        f"(embed_dim={int(proto_cpu.shape[1])}) to {output_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
