"""Console script for the Phase 5.5 mobile export pipeline.

Invoke via the ``export-mobile`` entry point declared in
``pyproject.toml``::

    export-mobile [--db PATH]
                  --backbone-checkpoint PATH
                  [--view-classifier-checkpoint PATH | none]
                  [--prototypes PATH]
                  [--model MobileCLIP-B]
                  [--pretrained datacompdr]
                  [--quantize fp32|fp16|int8]
                  [--output-dir PATH]
                  [--report PATH]
                  [-v]

Runs :func:`car_lense_engine.export.mobile.export_mobile` and writes a
JSON run report. ``--view-classifier-checkpoint none`` (literal string
"none") explicitly skips the view-head export -- useful when the head
hasn't been retrained against the current backbone yet.

The ``--db`` flag is accepted for parity with the other Phase 5
scripts but isn't currently consumed (the export pipeline reads the
backbone checkpoint + prototype cache, not the SQLite DB). It's
preserved so a single orchestrator wrapper can shell out to every
phase with the same arg set.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal, cast

from .mobile import MobileExportConfig, export_mobile, write_report_json

DEFAULT_DB = Path("db/crawl.sqlite")
DEFAULT_BACKBONE = Path(
    "models/checkpoints/mobileclip_b_compcars-vmmrdb-stanford_cars_resumed_epoch09_top1_85.7.pt"
)
DEFAULT_VIEW_CLASSIFIER = Path("models/checkpoints/exterior_classifier_v1.pt")
DEFAULT_PROTOTYPES = Path("cache/prototypes_combined_b_resumed.pt")
DEFAULT_MODEL = "MobileCLIP-B"
DEFAULT_PRETRAINED = "datacompdr"
DEFAULT_QUANTIZE = "fp16"
DEFAULT_OUTPUT_DIR = Path("dist/")
DEFAULT_REPORT = Path("reports/phase5_5_mobile_export.json")
QUANTIZE_CHOICES: tuple[str, ...] = ("fp32", "fp16", "int8")

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``argparse`` parser for the ``export-mobile`` command."""
    parser = argparse.ArgumentParser(
        prog="export-mobile",
        description=(
            "Phase 5.5 mobile export: convert the trained backbone + view "
            "classifier + prototype cache into iOS Core ML + Android "
            "TFLite (or ORT Mobile) bundles, plus an intermediate ONNX."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=(
            f"path to the crawler SQLite DB (default: {DEFAULT_DB}); "
            "not currently used by the export, accepted for arg-set parity"
        ),
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=DEFAULT_BACKBONE,
        help=f"path to the fine-tuned backbone checkpoint (default: {DEFAULT_BACKBONE})",
    )
    parser.add_argument(
        "--view-classifier-checkpoint",
        type=str,
        default=str(DEFAULT_VIEW_CLASSIFIER),
        help=(
            f"path to the view-classifier checkpoint (default: {DEFAULT_VIEW_CLASSIFIER}); "
            "pass the literal string 'none' to skip the view-head export"
        ),
    )
    parser.add_argument(
        "--prototypes",
        type=Path,
        default=DEFAULT_PROTOTYPES,
        help=f"path to the prototype cache .pt file (default: {DEFAULT_PROTOTYPES})",
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
        "--quantize",
        type=str,
        choices=QUANTIZE_CHOICES,
        default=DEFAULT_QUANTIZE,
        help=f"target precision for the mobile bundles (default: {DEFAULT_QUANTIZE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory for the bundle (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"path for the JSON run report (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def _parse_view_classifier(value: str) -> Path | None:
    """Parse the ``--view-classifier-checkpoint`` argument.

    The literal string ``"none"`` (case-insensitive) maps to ``None``
    so callers can explicitly opt out. Everything else is parsed as
    a path and must exist on disk.
    """
    if value.strip().lower() == "none":
        return None
    return Path(value)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``export-mobile`` console script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    backbone_path: Path = args.backbone_checkpoint
    if not backbone_path.exists():
        parser.error(f"--backbone-checkpoint path does not exist: {backbone_path}")

    prototypes_path: Path = args.prototypes
    if not prototypes_path.exists():
        parser.error(f"--prototypes path does not exist: {prototypes_path}")

    view_path = _parse_view_classifier(args.view_classifier_checkpoint)
    if view_path is not None and not view_path.exists():
        parser.error(f"--view-classifier-checkpoint path does not exist: {view_path}")

    quantize_value: Literal["fp32", "fp16", "int8"] = cast(
        Literal["fp32", "fp16", "int8"], args.quantize
    )

    config = MobileExportConfig(
        backbone_checkpoint=backbone_path,
        view_classifier_checkpoint=view_path,
        prototypes_path=prototypes_path,
        output_dir=args.output_dir,
        model_name=args.model,
        pretrained=args.pretrained,
        quantize=quantize_value,
    )

    report = export_mobile(config=config)
    write_report_json(report, args.report)
    print(
        "export-mobile: wrote bundle to "
        f"{args.output_dir} (onnx={report.onnx_path}, "
        f"coreml={report.coreml_path}, tflite={report.tflite_path})"
    )
    if report.skipped:
        print("export-mobile: skipped steps:")
        for note in report.skipped:
            print(f"  - {note}")
    if report.notes:
        print("export-mobile: notes:")
        for note in report.notes:
            print(f"  - {note}")
    print(f"export-mobile: report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
