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
    DEFAULT_SELECTOR_TIMEOUT_MS,
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
        "wait_for_selector_by_source",
        "selector_timeout_ms",
    ):
        assert name in params, f"expected PlaywrightFetcher.__init__ to declare {name}"


def test_fetcher_default_selector_timeout_is_10s() -> None:
    """Default selector timeout is 10s in ms."""
    assert DEFAULT_SELECTOR_TIMEOUT_MS == 10_000


# A sentinel exception we inject to escape the constructor right after
# validation but before Playwright actually launches Chromium. This lets us
# assert that validation passed (no ValueError raised) without ever spinning
# up a real browser. The fence catches any Exception subclass.
class _BarrierError(Exception):
    """Raised by a monkey-patched ``sync_playwright`` to stop construction."""


def _install_post_validation_barrier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the lazy-imported ``sync_playwright`` so init can't proceed."""
    import playwright.sync_api as pw_sync  # noqa: PLC0415 - test-only

    def _raise(*_a: object, **_kw: object) -> None:
        raise _BarrierError("post-validation barrier")

    monkeypatch.setattr(pw_sync, "sync_playwright", _raise)


def test_fetcher_default_wait_for_selector_by_source_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default / None / {} argument validates cleanly (no ValueError)."""
    _install_post_validation_barrier(monkeypatch)
    # None — validation accepts; barrier fires after.
    with pytest.raises(_BarrierError):
        PlaywrightFetcher(wait_for_selector_by_source=None)
    with pytest.raises(_BarrierError):
        PlaywrightFetcher(wait_for_selector_by_source={})


def test_fetcher_wait_for_selector_validates_known_sources() -> None:
    """An unknown source key must raise ValueError before any Playwright call."""
    with pytest.raises(ValueError, match="unknown source"):
        PlaywrightFetcher(wait_for_selector_by_source={"ebay": ".foo"})


def test_fetcher_wait_for_selector_accepts_known_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple known sources validate cleanly (no ValueError)."""
    _install_post_validation_barrier(monkeypatch)
    with pytest.raises(_BarrierError):
        PlaywrightFetcher(
            wait_for_selector_by_source={
                "autotrader": "[data-cmp='inventoryListing']",
                "cars_com": ".vehicle-card",
            }
        )


def test_fetcher_wait_for_selector_rejects_empty_string_value() -> None:
    """An empty / whitespace-only selector value must raise ValueError."""
    with pytest.raises(ValueError, match="non-empty string"):
        PlaywrightFetcher(wait_for_selector_by_source={"autotrader": ""})
    with pytest.raises(ValueError, match="non-empty string"):
        PlaywrightFetcher(wait_for_selector_by_source={"autotrader": "   "})


def test_fetcher_selector_timeout_must_be_positive() -> None:
    """selector_timeout_ms <= 0 must raise ValueError before any Playwright call."""
    with pytest.raises(ValueError, match="selector_timeout_ms"):
        PlaywrightFetcher(selector_timeout_ms=0)
    with pytest.raises(ValueError, match="selector_timeout_ms"):
        PlaywrightFetcher(selector_timeout_ms=-100)


def test_wait_until_literal_values() -> None:
    """The WaitUntil Literal must enumerate exactly the three Playwright values."""
    assert browser_mod._WAIT_UNTIL_VALUES == (
        "domcontentloaded",
        "load",
        "networkidle",
    )


def test_chromium_launch_args_disable_automation() -> None:
    """Chromium launch args must include the automation-suppression flag.

    Without ``--disable-blink-features=AutomationControlled`` Chromium exposes
    a clear ``navigator.webdriver`` signal that Cloudflare-class gates use to
    serve 403 / Turnstile challenges to Playwright sessions. This flag is the
    single most impactful piece of stealth hardening we apply at launch time.
    """
    assert "--disable-blink-features=AutomationControlled" in browser_mod._CHROMIUM_LAUNCH_ARGS


def test_webdriver_override_init_script_targets_navigator_webdriver() -> None:
    """The init script must clobber ``navigator.webdriver``.

    Pure structural check of the constant; the per-page integration is
    covered by ``test_fetcher_init_script_includes_webdriver_override`` below
    via the sentinel-barrier pattern.
    """
    script = browser_mod._WEBDRIVER_OVERRIDE_SCRIPT
    assert "navigator" in script
    assert "webdriver" in script
    assert "undefined" in script


def test_fetcher_init_script_includes_webdriver_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing the fetcher must call ``add_init_script`` with the override.

    We intercept ``sync_playwright`` with a fake Playwright object that
    records every ``add_init_script`` call made on the browser context, then
    assert the navigator.webdriver override is among them. The fake also
    short-circuits the rest of construction so we never need real Chromium.
    """
    import playwright.sync_api as pw_sync  # noqa: PLC0415 - test-only

    recorded_init_scripts: list[str] = []

    class _FakeContext:
        def add_init_script(self, script: str) -> None:
            recorded_init_scripts.append(script)

        def set_default_navigation_timeout(self, _ms: int) -> None:
            return None

        def set_default_timeout(self, _ms: int) -> None:
            return None

        def new_page(self) -> object:  # pragma: no cover - unused in this test
            raise RuntimeError("not used in this test")

    class _FakeBrowser:
        def new_context(self, **_kwargs: object) -> _FakeContext:
            return _FakeContext()

        def close(self) -> None:
            return None

    class _FakeChromium:
        def launch(self, **_kwargs: object) -> _FakeBrowser:
            return _FakeBrowser()

    class _FakePlaywright:
        def __init__(self) -> None:
            self.chromium = _FakeChromium()

        def stop(self) -> None:
            return None

    class _FakeRunner:
        def start(self) -> _FakePlaywright:
            return _FakePlaywright()

    def _fake_sync_playwright() -> _FakeRunner:
        return _FakeRunner()

    monkeypatch.setattr(pw_sync, "sync_playwright", _fake_sync_playwright)
    # Bypass the UA-discovery probe (which would try to open a real page).
    monkeypatch.setattr(
        PlaywrightFetcher,
        "_discover_default_user_agent",
        lambda _self: "Mozilla/5.0 (FakeUA)",
    )

    # Construct the fetcher. We don't care about the resulting instance — we
    # only care that ``add_init_script`` was called with our explicit
    # navigator.webdriver override during construction (independent of any
    # init scripts ``playwright-stealth`` may also add at the same time).
    fetcher = PlaywrightFetcher()
    try:
        assert browser_mod._WEBDRIVER_OVERRIDE_SCRIPT in recorded_init_scripts, (
            f"expected explicit navigator.webdriver override "
            f"{browser_mod._WEBDRIVER_OVERRIDE_SCRIPT!r} in init scripts; "
            f"got {recorded_init_scripts!r}"
        )
    finally:
        fetcher.close()
