"""Default :class:`Fetcher` implementation backed by Playwright Chromium + stealth.

Playwright is *lazy-imported* inside :meth:`PlaywrightFetcher.__init__` and
:meth:`PlaywrightFetcher.fetch` so unit tests for the rest of the crawler do
not require ``playwright install chromium`` to have been run. The user is
expected to run ``playwright install chromium`` once before kicking off a real
crawl.

The fetcher launches a single persistent ``BrowserContext`` per session so
cookies stick across navigations within a run; this matters for sites that
issue anti-bot interstitials on a fresh context. A custom User-Agent
identifies the project (``CarLenseResearch/0.1``) by appending to the
underlying Chromium UA so we don't lie about being a browser.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from .fetcher import FetchedPage, FetchError
from .proxy import mask_proxy_url, parse_proxy_url

if TYPE_CHECKING:  # pragma: no cover - typing only
    from playwright.sync_api import (
        Browser,
        BrowserContext,
        Page,
        Playwright,
    )

logger = logging.getLogger(__name__)

WaitUntil = Literal["domcontentloaded", "load", "networkidle"]
"""Accepted values for ``page.goto(wait_until=...)``."""

_WAIT_UNTIL_VALUES: tuple[str, ...] = get_args(WaitUntil)

DEFAULT_VIEWPORT: dict[str, int] = {"width": 1920, "height": 1080}
DEFAULT_LOCALE: str = "en-US"
DEFAULT_TIMEZONE: str = "America/Los_Angeles"
DEFAULT_NAVIGATION_TIMEOUT_MS: int = 30_000
DEFAULT_SETTLE_MS: int = 3_000
DEFAULT_WAIT_UNTIL: WaitUntil = "domcontentloaded"
DEFAULT_SELECTOR_TIMEOUT_MS: int = 10_000

# Chromium launch args to suppress Playwright/CDP automation tells. The first
# is the single most impactful flag against fingerprint-based bot detection
# (Cloudflare et al inspect ``navigator.webdriver`` and a handful of related
# automation markers). ``--no-sandbox`` is also pragmatic under WSL / CI where
# the user-namespace sandbox often isn't available; without it Chromium can
# fail to launch entirely.
_CHROMIUM_LAUNCH_ARGS: tuple[str, ...] = (
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
)

# Init script injected into every page before navigation. ``playwright-stealth``
# 2.x patches navigator.webdriver in most cases, but the override below is
# both a belt-and-suspenders backup AND the canonical signal we test against
# in unit tests (so we can assert the contract without spinning up Chromium).
_WEBDRIVER_OVERRIDE_SCRIPT: str = (
    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
)


class PlaywrightFetcher:
    """Fetch URLs through a Playwright Chromium browser with stealth patches.

    Construction launches Chromium and creates a persistent context; the
    caller must call :meth:`close` (or use the class as a context manager) to
    release browser resources.
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        ua_suffix: str = "CarLenseResearch/0.1",
        wait_until: WaitUntil = DEFAULT_WAIT_UNTIL,
        settle_ms: int = DEFAULT_SETTLE_MS,
        navigation_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS,
        wait_for_selector_by_source: dict[str, str] | None = None,
        selector_timeout_ms: int = DEFAULT_SELECTOR_TIMEOUT_MS,
        viewport: dict[str, int] | None = None,
        locale: str = DEFAULT_LOCALE,
        timezone_id: str = DEFAULT_TIMEZONE,
        proxy: str | None = None,
    ) -> None:
        # Validate parameters BEFORE launching Chromium so structural tests can
        # exercise validation without a Playwright install. The lazy-import
        # contract requires that an invalid call raise without touching the
        # ``playwright`` module.
        if wait_until not in _WAIT_UNTIL_VALUES:
            raise ValueError(
                f"wait_until must be one of {_WAIT_UNTIL_VALUES!r}, got {wait_until!r}"
            )
        if settle_ms < 0:
            raise ValueError(f"settle_ms must be >= 0, got {settle_ms!r}")
        if navigation_timeout_ms <= 0:
            raise ValueError(f"navigation_timeout_ms must be > 0, got {navigation_timeout_ms!r}")
        if selector_timeout_ms <= 0:
            raise ValueError(f"selector_timeout_ms must be > 0, got {selector_timeout_ms!r}")

        # Validate per-source selectors at construction so misconfiguration
        # surfaces immediately (and without touching the ``playwright`` module).
        # We import ``known_sources`` from the routing module — it's a tiny
        # pure-Python helper with no Playwright dependency, so the lazy-import
        # contract is preserved.
        from .routing import known_sources  # noqa: PLC0415 - keep coupling local

        validated_selectors: dict[str, str] = {}
        if wait_for_selector_by_source:
            valid = known_sources()
            for source, selector in wait_for_selector_by_source.items():
                if source not in valid:
                    raise ValueError(
                        f"wait_for_selector_by_source: unknown source {source!r}; "
                        f"valid choices are {sorted(valid)}"
                    )
                if not isinstance(selector, str) or not selector.strip():
                    raise ValueError(
                        f"wait_for_selector_by_source[{source!r}]: selector must be "
                        f"a non-empty string, got {selector!r}"
                    )
                validated_selectors[source] = selector

        # Validate (and parse) the proxy URL BEFORE the lazy Playwright import
        # so misconfiguration fails fast without launching Chromium. The proxy
        # dict is held until the chromium.launch call below.
        proxy_dict: dict[str, str] | None = None
        if proxy is not None:
            proxy_dict = parse_proxy_url(proxy)

        self._headless = headless
        self._ua_suffix = ua_suffix
        self._wait_until: WaitUntil = wait_until
        self._settle_ms = settle_ms
        self._navigation_timeout_ms = navigation_timeout_ms
        self._wait_for_selector_by_source: dict[str, str] = validated_selectors
        self._selector_timeout_ms = selector_timeout_ms
        self._viewport = dict(viewport) if viewport is not None else dict(DEFAULT_VIEWPORT)
        self._locale = locale
        self._timezone_id = timezone_id
        # We do NOT store the raw proxy URL on the instance — only the parsed
        # dict is needed for the browser launch, and stashing the original URL
        # (credentials and all) widens the leak surface for no benefit.
        self._proxy_configured: bool = proxy_dict is not None
        # Mask the URL for the startup log — credentials must never appear.
        self._proxy_log_repr: str | None = mask_proxy_url(proxy) if proxy is not None else None

        # Lazy import to keep unit tests free of a Playwright requirement.
        from playwright.sync_api import sync_playwright  # noqa: PLC0415 - intentional lazy import

        self._pw: Playwright = sync_playwright().start()
        # Launch Chromium with automation-flag suppression. The most important
        # one is ``--disable-blink-features=AutomationControlled`` which keeps
        # ``navigator.webdriver`` from being a clear ``true`` and removes the
        # ``Chrome-AutomationControlled`` infobar signal that Cloudflare and
        # similar gates fingerprint on.
        #
        # When a proxy is configured, attach it at the browser level (simpler
        # than per-context). Playwright accepts the dict shape returned by
        # :func:`parse_proxy_url`. We cast at the boundary because Playwright
        # types the kwarg as a TypedDict.
        launch_kwargs: dict[str, Any] = {
            "headless": self._headless,
            "args": list(_CHROMIUM_LAUNCH_ARGS),
        }
        if proxy_dict is not None:
            launch_kwargs["proxy"] = cast(Any, proxy_dict)
        self._browser: Browser = self._pw.chromium.launch(**launch_kwargs)

        # Get the default UA so we can append our project identifier.
        # If anything goes wrong, fall back to a plain Chromium-like UA.
        ua = self._compose_user_agent()
        # Playwright expects a TypedDict for viewport — cast our dict at the boundary.
        # Realistic context options: a desktop 1920x1080 viewport, a US locale
        # / Pacific timezone, an explicit device scale factor, and a
        # pre-granted geolocation permission make us look like a typical
        # consumer browser rather than a freshly-constructed automation
        # context.
        self._context: BrowserContext = self._browser.new_context(
            user_agent=ua,
            viewport=cast(Any, self._viewport),
            locale=self._locale,
            timezone_id=self._timezone_id,
            device_scale_factor=1.0,
            permissions=["geolocation"],
        )
        self._context.set_default_navigation_timeout(self._navigation_timeout_ms)
        self._context.set_default_timeout(self._navigation_timeout_ms)

        # Belt-and-suspenders: explicitly clobber navigator.webdriver via an
        # init script BEFORE navigation. ``playwright-stealth`` 2.x usually
        # patches this already, but applying it again at the context level
        # guarantees the override on every page regardless of stealth's
        # internal evasion enable/disable knobs.
        self._context.add_init_script(_WEBDRIVER_OVERRIDE_SCRIPT)

        # Apply playwright-stealth patches once at context level. The 2.x API
        # is ``Stealth().apply_stealth_sync(page_or_context)``; we apply to
        # the context so every page created from it inherits the patches.
        stealth_applied = False
        try:
            from playwright_stealth import Stealth  # noqa: PLC0415

            Stealth().apply_stealth_sync(self._context)
            stealth_applied = True
        except ImportError:  # pragma: no cover - dependency listed in pyproject
            logger.debug("playwright-stealth not installed; running without stealth patches")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("playwright-stealth apply failed: %r", exc)
        self._stealth_applied = stealth_applied

        logger.info(
            "PlaywrightFetcher ready: headless=%s ua_suffix=%s wait_until=%s "
            "settle_ms=%d navigation_timeout_ms=%d selector_timeout_ms=%d "
            "wait_for_selector_by_source=%s viewport=%s tz=%s stealth=%s "
            "proxy=%s",
            self._headless,
            self._ua_suffix,
            self._wait_until,
            self._settle_ms,
            self._navigation_timeout_ms,
            self._selector_timeout_ms,
            self._wait_for_selector_by_source,
            self._viewport,
            self._timezone_id,
            self._stealth_applied,
            self._proxy_log_repr if self._proxy_configured else "<none>",
        )

    # ------------------------------------------------------------- public API

    def fetch(self, url: str) -> FetchedPage:
        """Fetch ``url`` and return rendered HTML. Raises :class:`FetchError` on failure."""
        # Stealth patches and the navigator.webdriver override are already
        # applied at the context level (see ``__init__``), so every new page
        # inherits them; no per-page setup is required here.
        page: Page = self._context.new_page()
        try:
            response = page.goto(
                url,
                wait_until=self._wait_until,
                timeout=self._navigation_timeout_ms,
            )
            # If a wait_for_selector hint is configured for this URL's source,
            # poll for it before the time-based settle. A missing selector is
            # NOT a fetch failure — log a warning and fall through. The
            # downstream parser will surface "no listings" as a note if the
            # page genuinely lacks them.
            if self._wait_for_selector_by_source:
                # Lazy import to keep the lazy-import contract clean (routing
                # is pure Python — this is for symmetry with __init__'s
                # in-method import and to avoid module-level coupling).
                from .routing import source_for_url  # noqa: PLC0415

                source = source_for_url(page.url)
                selector = self._wait_for_selector_by_source.get(source) if source else None
                if selector:
                    try:
                        page.wait_for_selector(selector, timeout=self._selector_timeout_ms)
                    except Exception as exc:
                        logger.warning(
                            "wait_for_selector(%r) timed out for source=%s url=%s: %s",
                            selector,
                            source,
                            url,
                            exc,
                        )
            page.wait_for_timeout(self._settle_ms)
            if response is None:
                raise FetchError(f"no response object returned for {url}")
            status = response.status
            if status >= 400:
                raise FetchError(f"HTTP {status} for {url}")
            html = page.content()
            final_url = page.url
            return FetchedPage(
                url=final_url,
                status=status,
                html=html,
                fetched_at=datetime.now(UTC).replace(tzinfo=None),
            )
        except FetchError:
            raise
        except Exception as exc:
            raise FetchError(f"playwright error for {url}: {exc!r}") from exc
        finally:
            with contextlib.suppress(Exception):  # pragma: no cover - defensive
                page.close()

    def close(self) -> None:
        """Tear down browser, context, and the Playwright driver."""
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            self._context.close()
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            self._browser.close()
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            self._pw.stop()

    def __enter__(self) -> PlaywrightFetcher:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ----------------------------------------------------------------- helpers

    def _compose_user_agent(self) -> str:
        """Build the User-Agent string: native Chromium UA + project suffix."""
        base = self._discover_default_user_agent()
        if base is None:
            base = (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        return f"{base}; {self._ua_suffix}"

    def _discover_default_user_agent(self) -> str | None:
        """Best-effort: spin up a throwaway context to read the browser's UA."""
        try:
            tmp_ctx: BrowserContext = self._browser.new_context()
            try:
                tmp_page: Page = tmp_ctx.new_page()
                try:
                    ua: Any = tmp_page.evaluate("() => navigator.userAgent")
                    if isinstance(ua, str) and ua:
                        return ua
                finally:
                    tmp_page.close()
            finally:
                tmp_ctx.close()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("failed to discover default UA: %r", exc)
        return None
