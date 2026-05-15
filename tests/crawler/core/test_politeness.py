"""Tests for the politeness policy helpers."""

from __future__ import annotations

import random
from datetime import datetime

from car_lense_engine.crawler.core.politeness import (
    PolicyConfig,
    is_off_peak,
    jittered_delay,
    sleep_until_off_peak,
)


def test_jittered_delay_within_range() -> None:
    cfg = PolicyConfig(min_delay_seconds=3.0, max_delay_seconds=5.0)
    rng = random.Random(42)
    samples = [jittered_delay(cfg, rng) for _ in range(1000)]
    assert all(3.0 <= s <= 5.0 for s in samples)
    # Sanity: the range is actually exercised, not always a single value.
    assert min(samples) < 3.5
    assert max(samples) > 4.5


def test_jittered_delay_deterministic_with_seed() -> None:
    cfg = PolicyConfig(min_delay_seconds=3.0, max_delay_seconds=5.0)
    a = [jittered_delay(cfg, random.Random(7)) for _ in range(5)]
    b = [jittered_delay(cfg, random.Random(7)) for _ in range(5)]
    assert a == b


def test_jittered_delay_degenerate_range_returns_min() -> None:
    cfg = PolicyConfig(min_delay_seconds=4.2, max_delay_seconds=4.2)
    assert jittered_delay(cfg, random.Random(0)) == 4.2


def test_is_off_peak_in_window() -> None:
    cfg = PolicyConfig(off_peak_start_hour=2, off_peak_end_hour=6)
    pinned = datetime(2026, 5, 14, 3, 30, 0)
    assert is_off_peak(cfg, pinned) is True


def test_is_off_peak_outside_window() -> None:
    cfg = PolicyConfig(off_peak_start_hour=2, off_peak_end_hour=6)
    pinned = datetime(2026, 5, 14, 14, 0, 0)
    assert is_off_peak(cfg, pinned) is False


def test_is_off_peak_wraparound() -> None:
    cfg = PolicyConfig(off_peak_start_hour=22, off_peak_end_hour=4)
    assert is_off_peak(cfg, datetime(2026, 5, 14, 23, 0)) is True
    assert is_off_peak(cfg, datetime(2026, 5, 14, 3, 0)) is True
    assert is_off_peak(cfg, datetime(2026, 5, 14, 12, 0)) is False


def test_is_off_peak_boundary_inclusive_start_exclusive_end() -> None:
    cfg = PolicyConfig(off_peak_start_hour=2, off_peak_end_hour=6)
    assert is_off_peak(cfg, datetime(2026, 5, 14, 2, 0)) is True
    assert is_off_peak(cfg, datetime(2026, 5, 14, 6, 0)) is False


def test_sleep_until_off_peak_no_op_when_disabled() -> None:
    cfg = PolicyConfig(off_peak_only=False, off_peak_start_hour=2, off_peak_end_hour=6)
    sleeps: list[float] = []
    now = datetime(2026, 5, 14, 14, 0)
    sleep_until_off_peak(cfg, now_fn=lambda: now, sleep_fn=sleeps.append)
    assert sleeps == []


def test_sleep_until_off_peak_no_op_when_already_in_window() -> None:
    cfg = PolicyConfig(off_peak_only=True, off_peak_start_hour=2, off_peak_end_hour=6)
    sleeps: list[float] = []
    now = datetime(2026, 5, 14, 3, 0)
    sleep_until_off_peak(cfg, now_fn=lambda: now, sleep_fn=sleeps.append)
    assert sleeps == []


def test_sleep_until_off_peak_waits_until_window_opens() -> None:
    cfg = PolicyConfig(off_peak_only=True, off_peak_start_hour=2, off_peak_end_hour=6)
    sleeps: list[float] = []
    # 23:00 → next off-peak opens at 02:00 the following day = 3 hours
    now = datetime(2026, 5, 14, 23, 0)
    sleep_until_off_peak(cfg, now_fn=lambda: now, sleep_fn=sleeps.append)
    assert len(sleeps) == 1
    assert sleeps[0] == 3 * 3600


def test_sleep_until_off_peak_same_day_future_window() -> None:
    cfg = PolicyConfig(off_peak_only=True, off_peak_start_hour=22, off_peak_end_hour=23)
    sleeps: list[float] = []
    now = datetime(2026, 5, 14, 14, 0)
    sleep_until_off_peak(cfg, now_fn=lambda: now, sleep_fn=sleeps.append)
    assert sleeps == [8 * 3600]
