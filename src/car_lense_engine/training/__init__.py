"""Training pipelines for the recognition engine.

Phase 5.2 ships :mod:`car_lense_engine.training.train_classifier` -- a
classification-head fine-tune of the MobileCLIP-S2 image encoder over the
196 Stanford Cars classes, with hard-negative-aware cross-entropy
weighting derived from the Phase 5.1 confusion-pair leaderboard.
"""

from __future__ import annotations

from .train_classifier import (
    EpochMetrics,
    TrainConfig,
    TrainReport,
    build_class_weights_from_confusion,
    run_training,
    write_report,
)

__all__ = [
    "EpochMetrics",
    "TrainConfig",
    "TrainReport",
    "build_class_weights_from_confusion",
    "run_training",
    "write_report",
]
