"""Console script for the Phase 5.4 evaluation harness (``evaluate-recognize``).

Invoke via the ``evaluate-recognize`` entry point declared in
``pyproject.toml``::

    evaluate-recognize [--db PATH] [--source compcars[,vmmrdb,...]]
                       [--test-split test]
                       [--checkpoint PATH] [--prototypes PATH]
                       [--model MobileCLIP-S2] [--pretrained datacompdr]
                       [--device cuda|cpu|mps] [--batch-size N]
                       [--era-bucket-years 10]
                       [--report PATH]
                       [--print-top-makes N] [--print-top-confusions N]
                       [-v]

Runs the full retrieval pipeline against the held-out test set,
producing a JSON :class:`~car_lense_engine.eval.evaluate.EvaluationReport`
with overall + per-(make, view, era) top-K accuracy and the top
confusion pairs.

The ``--source`` flag accepts one or more comma-separated source names
(e.g. ``compcars,vmmrdb,stanford_cars``) so a single evaluation run
can score against every dataset the deployed model was trained on.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from car_lense_engine.db import open_db

from .evaluate import (
    EvaluationConfig,
    EvaluationReport,
    evaluate,
    format_cell_table,
    format_confusions,
    summarize,
    write_report,
)

DEFAULT_DB = Path("db/crawl.sqlite")
DEFAULT_SOURCE = "compcars"
DEFAULT_TEST_SPLIT = "test"
DEFAULT_CHECKPOINT = Path("models/checkpoints/mobileclip_s2_compcars_epoch09_top1_91.9.pt")
DEFAULT_PROTOTYPES = Path("cache/prototypes.pt")
DEFAULT_MODEL = "MobileCLIP-S2"
DEFAULT_PRETRAINED = "datacompdr"
DEFAULT_DEVICE = "cuda"
DEFAULT_BATCH_SIZE = 128
DEFAULT_ERA_BUCKET_YEARS = 10
DEFAULT_REPORT = Path("reports/phase5_4_eval.json")
DEFAULT_PRINT_TOP_MAKES = 20
DEFAULT_PRINT_TOP_CONFUSIONS = 20
DEVICE_CHOICES: tuple[str, ...] = ("cpu", "cuda", "mps")

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


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser for the ``evaluate-recognize`` command."""
    parser = argparse.ArgumentParser(
        prog="evaluate-recognize",
        description=(
            "Phase 5.4 evaluation harness: measure top-K accuracy of the "
            "deployed retrieval pipeline (fine-tuned MobileCLIP-S2 backbone "
            "+ v1 prototype cache) against the held-out test set, with "
            "per-(make, view, era) breakdowns and a top confusion table."
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
            f"evaluate against (default: {DEFAULT_SOURCE}). Examples: "
            "`compcars`, `compcars,vmmrdb`, "
            "`compcars,vmmrdb,stanford_cars`."
        ),
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default=DEFAULT_TEST_SPLIT,
        help=f"images.split used for evaluation (default: {DEFAULT_TEST_SPLIT})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=(
            "path to the Phase 5.2 fine-tuned image-encoder checkpoint "
            f"(default: {DEFAULT_CHECKPOINT}). Use a literal empty string "
            "to skip checkpoint loading and evaluate the pretrained backbone."
        ),
    )
    parser.add_argument(
        "--prototypes",
        type=Path,
        default=DEFAULT_PROTOTYPES,
        help=(
            "path to the v1 single-prototype cache (.pt). The harness will "
            "REJECT v2 per-view caches; pass the v1 file produced by "
            f"`build-prototypes` (default: {DEFAULT_PROTOTYPES})."
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
        help=(
            f"torch device (default: {DEFAULT_DEVICE}). If `cuda` is selected "
            "but unavailable at runtime, the harness falls back to `cpu` with "
            "a warning."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"images per forward pass (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--era-bucket-years",
        type=int,
        default=DEFAULT_ERA_BUCKET_YEARS,
        help=(
            f"era bucket width in years (default: {DEFAULT_ERA_BUCKET_YEARS}). "
            "10 produces decade labels like '2000s' / '2010s'; other widths "
            "produce '<start>-<end>' labels."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"path for the JSON report (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--print-top-makes",
        type=int,
        default=DEFAULT_PRINT_TOP_MAKES,
        help=(
            "limit the per-make table printed to stdout to this many worst rows "
            f"(default: {DEFAULT_PRINT_TOP_MAKES})"
        ),
    )
    parser.add_argument(
        "--print-top-confusions",
        type=int,
        default=DEFAULT_PRINT_TOP_CONFUSIONS,
        help=(
            "limit the global confusion table printed to stdout to this many rows "
            f"(default: {DEFAULT_PRINT_TOP_CONFUSIONS})"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def _resolve_device(requested: str) -> str:
    """Resolve the requested torch device, falling back to ``cpu`` if needed.

    ``cuda`` falls back to ``cpu`` if torch reports cuda unavailable; we
    deliberately don't try to import torch at module-import time so the
    CLI's ``--help`` doesn't pay the import cost.
    """
    if requested != "cuda":
        return requested
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover -- torch is a runtime dep
        logger.warning("evaluate-recognize: torch missing; using cpu")
        return "cpu"
    if not torch.cuda.is_available():
        logger.warning(
            "evaluate-recognize: --device=cuda requested but torch.cuda.is_available() "
            "is False; falling back to cpu"
        )
        return "cpu"
    return "cuda"


def _print_summary(
    report: EvaluationReport,
    *,
    top_makes: int,
    top_confusions: int,
) -> None:
    """Render a multi-section summary to stdout."""
    top_ks = tuple(report.config.top_k)
    print()
    print("=" * 80)
    print(f"evaluate-recognize: {summarize(report)}")
    print("=" * 80)
    print()

    print("Per-view top-K accuracy (worst-first by top-1):")
    print(format_cell_table(report.per_view, top_ks=top_ks, header="view"))
    print()

    print("Per-era top-K accuracy (worst-first by top-1):")
    print(format_cell_table(report.per_era, top_ks=top_ks, header="era"))
    print()

    print(f"Per-make top-K accuracy (worst {top_makes} by top-1):")
    print(
        format_cell_table(
            report.per_make,
            top_ks=top_ks,
            limit=top_makes,
            header="make",
        )
    )
    print()

    print(f"Top {top_confusions} confusion pairs (over top-1 misses):")
    print(format_confusions(report.top_confusions, limit=top_confusions))
    print()


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``evaluate-recognize`` console script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.batch_size <= 0:
        parser.error(f"--batch-size must be > 0, got {args.batch_size}")
    if args.era_bucket_years <= 0:
        parser.error(f"--era-bucket-years must be > 0, got {args.era_bucket_years}")
    if args.print_top_makes < 0:
        parser.error(f"--print-top-makes must be >= 0, got {args.print_top_makes}")
    if args.print_top_confusions < 0:
        parser.error(f"--print-top-confusions must be >= 0, got {args.print_top_confusions}")

    db_path: Path = args.db
    if not db_path.exists():
        parser.error(f"DB path does not exist: {db_path}")

    try:
        sources = _parse_sources(args.source)
    except ValueError as exc:
        parser.error(f"--source: {exc}")

    # Empty-string sentinel: ``--checkpoint ""`` means "no checkpoint";
    # argparse turns that into ``Path('.')`` which exists, so we have to
    # detect the empty string explicitly via the raw argv. The clean way
    # is to compare the parsed value's string form.
    checkpoint_path: Path | None = args.checkpoint
    if checkpoint_path is not None and str(checkpoint_path) in {"", "."}:
        checkpoint_path = None
    if checkpoint_path is not None and not checkpoint_path.exists():
        parser.error(f"--checkpoint path does not exist: {checkpoint_path}")

    prototypes_path: Path = args.prototypes
    if not prototypes_path.exists():
        parser.error(
            f"--prototypes path does not exist: {prototypes_path}; run `build-prototypes` first"
        )

    device = _resolve_device(args.device)

    config = EvaluationConfig(
        db_path=db_path,
        source=sources,
        test_split=args.test_split,
        checkpoint_path=checkpoint_path,
        prototypes_path=prototypes_path,
        model_name=args.model,
        pretrained=args.pretrained,
        device=device,
        batch_size=args.batch_size,
        era_bucket_years=args.era_bucket_years,
    )

    conn = open_db(db_path)
    try:
        report = evaluate(conn=conn, config=config)
    finally:
        conn.close()

    write_report(report, args.report)
    _print_summary(
        report,
        top_makes=args.print_top_makes,
        top_confusions=args.print_top_confusions,
    )
    print(f"evaluate-recognize: report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
