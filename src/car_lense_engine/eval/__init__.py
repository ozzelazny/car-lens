"""Evaluation harnesses for the recognition engine.

Phase 5.1 ships :mod:`car_lense_engine.eval.baseline` — a pre-trained
zero-shot prototype retrieval baseline (one mean-embedding prototype per
class). View-conditioning is Phase 5.2 fine-tune territory and lives in a
separate module.
"""

from __future__ import annotations

from .baseline import (
    BaselineConfig,
    BaselineReport,
    ClassMetric,
    ConfusionPair,
    build_prototypes,
    evaluate,
)

__all__ = [
    "BaselineConfig",
    "BaselineReport",
    "ClassMetric",
    "ConfusionPair",
    "build_prototypes",
    "evaluate",
]
