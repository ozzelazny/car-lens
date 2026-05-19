"""Mobile export pipeline (Phase 5.5).

This package turns the trained MobileCLIP-B backbone + binary view
classifier + per-class prototype cache into shippable iOS (Core ML) and
Android (TFLite / ONNX Runtime Mobile) bundles, plus an intermediate
ONNX file kept for debugging.

See :func:`car_lense_engine.export.mobile.export_mobile` for the full
pipeline, and :mod:`car_lense_engine.export.validate` for the
PyTorch-vs-exported parity checker.
"""

from __future__ import annotations

from .mobile import (
    MobileExportConfig,
    MobileExportReport,
    bundle_prototypes,
    export_coreml,
    export_mobile,
    export_onnx,
    export_tflite,
    export_view_head,
    load_backbone,
    write_class_names_json,
    write_preprocessing_json,
)

__all__ = [
    "MobileExportConfig",
    "MobileExportReport",
    "bundle_prototypes",
    "export_coreml",
    "export_mobile",
    "export_onnx",
    "export_tflite",
    "export_view_head",
    "load_backbone",
    "write_class_names_json",
    "write_preprocessing_json",
]
