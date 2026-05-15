"""Structural tests for :class:`PlaywrightFetcher`.

These tests must NOT instantiate Chromium — Playwright is intentionally
lazy-imported inside ``__init__`` so unit tests can run without
``playwright install chromium`` having been executed. Validation logic for
the user-facing parameters lives BEFORE the lazy import so we can exercise it
here in a sandbox without launching a browser.
"""

from __future__ import annotations

import sys

import pytest

from car_lense_engine.crawler.core import browser as browser_mod
from car_lense_engine.crawler.core.browser import (
    DEFAULT_NAVIGATION_TIMEOUT_MS,
    DEFAULT_SETTLE_MS,
    DEFAULT_WAIT_UNTIL,
    PlaywrightFetcher,
)


def test_module_does_not_eagerly_import_playwright() -> None:
    """Importing browser.py must not pull in the ``playwright`` package."""
    # The module was imported at test-collection time; assert the contract.
    assert "playwright" not in sys.modules


def test_fetcher_default_settle_ms_is_3000() -> None:
    """The new default settle_ms (bumped from 1500) must be 3000."""
    assert DEFAULT_SETTLE_MS == 3000


def test_fetcher_default_wait_until_is_domcontentloaded() -> None:
    """Default wait_until stays domcontentloaded (most robust)."""
    assert DEFAULT_WAIT_UNTIL == "domcontentloaded"


def test_fetcher_default_navigation_timeout_is_30s() -> None:
    """Default navigation timeout remains 30s in ms."""
    assert DEFAULT_NAVIGATION_TIMEOUT_MS == 30_000


def test_fetcher_wait_until_validation() -> None:
    """An invalid wait_until must raise ValueError before any Playwright call."""
    with pytest.raises(ValueError, match="wait_until"):
        # type: ignore[arg-type] — feeding a bad value intentionally
        PlaywrightFetcher(wait_until="invalid")  # type: ignore[arg-type]
    # And the lazy-import contract must still hold after a rejected call.
    assert "playwright" not in sys.modules


def test_fetcher_negative_settle_rejected() -> None:
    """settle_ms < 0 must raise ValueError before any Playwright call."""
    with pytest.raises(ValueError, match="settle_ms"):
        PlaywrightFetcher(settle_ms=-1)
    assert "playwright" not in sys.modules


def test_fetcher_zero_or_negative_navigation_timeout_rejected() -> None:
    """navigation_timeout_ms <= 0 must raise ValueError before any Playwright call."""
    with pytest.raises(ValueError, match="navigation_timeout_ms"):
        PlaywrightFetcher(navigation_timeout_ms=0)
    with pytest.raises(ValueError, match="navigation_timeout_ms"):
        PlaywrightFetcher(navigation_timeout_ms=-100)
    assert "playwright" not in sys.modules


def test_fetcher_init_parameters_are_documented() -> None:
    """The public PlaywrightFetcher __init__ exposes all configurable knobs."""
    params = PlaywrightFetcher.__init__.__annotations__
    for name in (
        "headless",
        "ua_suffix",
        "wait_until",
        "settle_ms",
        "navigation_timeout_ms",
    ):
        assert name in params, f"expected PlaywrightFetcher.__init__ to declare {name}"


def test_wait_until_literal_values() -> None:
    """The WaitUntil Literal must enumerate exactly the three Playwright values."""
    assert browser_mod._WAIT_UNTIL_VALUES == (
        "domcontentloaded",
        "load",
        "networkidle",
    )
