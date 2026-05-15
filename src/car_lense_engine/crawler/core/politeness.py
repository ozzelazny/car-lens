"""Politeness policy: jittered per-request delay, optional off-peak gate.

All time- and randomness-dependent helpers accept injectable ``rng`` and
``now`` arguments so tests can pin behaviour deterministically.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class PolicyConfig(BaseModel):
    """Knobs controlling crawler pacing and the off-peak window."""

    model_config = ConfigDict(extra="forbid")

    min_delay_seconds: float = Field(default=3.0, ge=0.0)
    """Lower bound (inclusive) for the per-request jittered sleep."""

    max_delay_seconds: float = Field(default=5.0, ge=0.0)
    """Upper bound (exclusive) for the per-request jittered sleep."""

    off_peak_only: bool = False
    """When ``True``, the runner blocks until the off-peak window opens."""

    off_peak_start_hour: int = Field(default=2, ge=0, le=23)
    """Local-time hour (inclusive) when the off-peak window opens."""

    off_peak_end_hour: int = Field(default=6, ge=0, le=23)
    """Local-time hour (exclusive) when the off-peak window closes."""

    idle_exit_seconds: int = Field(default=60, ge=0)
    """If the queue stays empty this long, the runner exits."""


def jittered_delay(cfg: PolicyConfig, rng: random.Random | None = None) -> float:
    """Return a uniformly random delay in ``[min_delay_seconds, max_delay_seconds)``.

    Falls back to the module-level :mod:`random` instance when ``rng`` is ``None``.
    If ``min == max`` we return exactly that value.
    """
    if cfg.max_delay_seconds < cfg.min_delay_seconds:
        raise ValueError(
            "PolicyConfig.max_delay_seconds must be >= min_delay_seconds; "
            f"got min={cfg.min_delay_seconds} max={cfg.max_delay_seconds}"
        )
    if cfg.max_delay_seconds == cfg.min_delay_seconds:
        return float(cfg.min_delay_seconds)
    source = rng if rng is not None else random
    return source.uniform(cfg.min_delay_seconds, cfg.max_delay_seconds)


def is_off_peak(cfg: PolicyConfig, now: datetime | None = None) -> bool:
    """Return ``True`` when ``now`` falls inside the off-peak window.

    The window is half-open: ``[start_hour, end_hour)``. Wrap-around is supported
    (e.g., ``start=22 end=4`` covers 22:00-04:00 local time).
    """
    when = now if now is not None else datetime.now()
    start = cfg.off_peak_start_hour
    end = cfg.off_peak_end_hour
    hour = when.hour
    if start == end:
        # Degenerate config: treat as "never off-peak" rather than "always".
        return False
    if start < end:
        return start <= hour < end
    # Wrap-around: window crosses midnight.
    return hour >= start or hour < end


def _seconds_until_off_peak(cfg: PolicyConfig, now: datetime) -> float:
    """Compute how long to sleep until the next off-peak window opens.

    Returns ``0.0`` if we're already inside the window.
    """
    if is_off_peak(cfg, now):
        return 0.0
    candidate = now.replace(hour=cfg.off_peak_start_hour, minute=0, second=0, microsecond=0)
    if candidate <= now:
        candidate = candidate + timedelta(days=1)
    return (candidate - now).total_seconds()


def sleep_until_off_peak(
    cfg: PolicyConfig,
    now_fn: Callable[[], datetime] = datetime.now,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Block until the configured off-peak window opens.

    Returns immediately if ``off_peak_only`` is ``False`` or we're already
    inside the window. Logs how long it intends to wait.
    """
    if not cfg.off_peak_only:
        return
    now = now_fn()
    if is_off_peak(cfg, now):
        return
    wait = _seconds_until_off_peak(cfg, now)
    logger.info(
        "off-peak gate: sleeping %.0fs until %02d:00 local",
        wait,
        cfg.off_peak_start_hour,
    )
    sleep_fn(wait)
