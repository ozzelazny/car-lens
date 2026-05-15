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
from typing import TYPE_CHECKING, Any, cast

from .fetcher import FetchedPage, FetchError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from playwright.sync_api import (
        Browser,
        BrowserContext,
        Page,
        Playwright,
    )

logger = logging.getLogger(__name__)

DEFAULT_VIEWPORT: dict[str, int] = {"width": 1366, "height": 900}
DEFAULT_LOCALE: str = "en-US"
DEFAULT_TIMEZONE: str = "America/New_York"
DEFAULT_NAV_TIMEOUT_MS: int = 30_000
DEFAULT_SETTLE_MS: int = 1500


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
        nav_timeout_ms: int = DEFAULT_NAV_TIMEOUT_MS,
        settle_ms: int = DEFAULT_SETTLE_MS,
        viewport: dict[str, int] | None = None,
        locale: str = DEFAULT_LOCALE,
        timezone_id: str = DEFAULT_TIMEZONE,
    ) -> None:
        self._headless = headless
        self._ua_suffix = ua_suffix
        self._nav_timeout_ms = nav_timeout_ms
        self._settle_ms = settle_ms
        self._viewport = dict(viewport) if viewport is not None else dict(DEFAULT_VIEWPORT)
        self._locale = locale
        self._timezone_id = timezone_id

        # Lazy import to keep unit tests free of a Playwright requirement.
        from playwright.sync_api import sync_playwright  # noqa: PLC0415 - intentional lazy import

        self._pw: Playwright = sync_playwright().start()
        self._browser: Browser = self._pw.chromium.launch(headless=self._headless)

        # Get the default UA so we can append our project identifier.
        # If anything goes wrong, fall back to a plain Chromium-like UA.
        ua = self._compose_user_agent()
        # Playwright expects a TypedDict for viewport — cast our dict at the boundary.
        self._context: BrowserContext = self._browser.new_context(
            user_agent=ua,
            viewport=cast(Any, self._viewport),
            locale=self._locale,
            timezone_id=self._timezone_id,
        )
        self._context.set_default_navigation_timeout(self._nav_timeout_ms)
        self._context.set_default_timeout(self._nav_timeout_ms)

        # Apply playwright-stealth patches once at context level.
        stealth_sync: Any | None
        try:
            from playwright_stealth import stealth_sync as _stealth_sync  # noqa: PLC0415

            stealth_sync = _stealth_sync
        except ImportError:  # pragma: no cover - dependency listed in pyproject
            stealth_sync = None
        self._stealth_sync = stealth_sync

        logger.info(
            "PlaywrightFetcher ready: headless=%s ua_suffix=%s viewport=%s tz=%s",
            self._headless,
            self._ua_suffix,
            self._viewport,
            self._timezone_id,
        )

    # ------------------------------------------------------------- public API

    def fetch(self, url: str) -> FetchedPage:
        """Fetch ``url`` and return rendered HTML. Raises :class:`FetchError` on failure."""
        page: Page = self._context.new_page()
        if self._stealth_sync is not None:
            try:
                self._stealth_sync(page)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("playwright-stealth patch failed: %r", exc)
        try:
            response = page.goto(url, wait_until="domcontentloaded")
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
