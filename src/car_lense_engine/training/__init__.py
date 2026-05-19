"""Training pipelines for the recognition engine.

Phase 5.2 ships :mod:`car_lense_engine.training.train_classifier` -- a
classification-head fine-tune of the MobileCLIP-S2 image encoder over the
196 Stanford Cars classes, with hard-negative-aware cross-entropy
weighting derived from the Phase 5.1 confusion-pair leaderboard.

Phase 5.3 ships :mod:`car_lense_engine.training.view_classifier` -- a
small 6-way head on top of the same backbone for view classification
(front / rear / side / three-quarter-front / three-quarter-rear /
non-exterior). The head is trained with the backbone frozen and
features cached in RAM.
"""

from __future__ import annotations

from .train_classifier import (
    EpochMetrics,
    TrainConfig,
    TrainReport,
    build_class_weights_from_confusion,
    run_training,
)
from .train_classifier import (
    write_report as write_train_report,
)
from .view_classifier import (
    EXTERIOR_VIEWS,
    NON_EXTERIOR_RAW,
    VIEW_CLASS_NAMES,
    CheckpointPayload,
    LinearHead,
    MLPHead,
    ViewClassifierConfig,
    ViewClassifierReport,
    ViewEpochMetrics,
    build_view_classifier_dataset,
    collapse_view,
    compute_class_weights,
    train_view_classifier,
)
from .view_classifier import (
    write_report as write_view_classifier_report,
)

# Back-compat alias: existing call sites import ``write_report`` from this
# package and refer to the training-report writer.
write_report = write_train_report

__all__ = [
    "EXTERIOR_VIEWS",
    "NON_EXTERIOR_RAW",
    "VIEW_CLASS_NAMES",
    "CheckpointPayload",
    "EpochMetrics",
    "LinearHead",
    "MLPHead",
    "TrainConfig",
    "TrainReport",
    "ViewClassifierConfig",
    "ViewClassifierReport",
    "ViewEpochMetrics",
    "build_class_weights_from_confusion",
    "build_view_classifier_dataset",
    "collapse_view",
    "compute_class_weights",
    "run_training",
    "train_view_classifier",
    "write_report",
    "write_train_report",
    "write_view_classifier_report",
]
