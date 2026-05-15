"""Alternate :class:`Fetcher` implementation backed by ``curl_cffi``.

``curl_cffi`` wraps ``curl-impersonate`` — a libcurl fork that mimics the TLS
handshake (JA3/JA4 fingerprints) of real Chrome / Safari / Firefox builds.
For server-rendered pages it often clears basic Cloudflare browser-fingerprint
challenges where headless Playwright is detected and 403'd.

This fetcher is **request-based** and does NOT execute JavaScript. Sites that
hydrate listings via JS (e.g. Cars & Bids' React SPA, AutoTrader's heavy
client app) will not work here; use :class:`PlaywrightFetcher` for those.

The ``curl_cffi`` import is deferred until first use so importing this module
does not fail if the dependency is missing (it is declared in ``pyproject``,
but lazy-loading keeps the contract symmetric with :mod:`browser`).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

from .fetcher import FetchedPage, FetchError

logger = logging.getLogger(__name__)

DEFAULT_IMPERSONATE: str = "chrome131"
DEFAULT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_UA_SUFFIX: str = "CarLenseResearch/0.1"

# Hardcoded base UA paired with the default ``chrome131`` impersonation
# profile. The TLS / JA3 fingerprint is what defeats Cloudflare; the UA only
# has to look plausible alongside it. We expose the project suffix so target
# sites can identify the bot if they inspect logs.
_BASE_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


class CurlCffiFetcher:
    """Fetcher backed by ``curl_cffi`` with browser TLS impersonation.

    Fast and lightweight (no browser process); works against server-rendered
    pages and basic Cloudflare protections that detect browser fingerprints.
    Does NOT execute JavaScript — sites that hydrate listings via JS won't
    work here; use :class:`PlaywrightFetcher` for those.
    """

    def __init__(
        self,
        *,
        impersonate: str = DEFAULT_IMPERSONATE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        ua_suffix: str = DEFAULT_UA_SUFFIX,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds!r}")
        if not impersonate:
            raise ValueError("impersonate must be a non-empty profile name")

        self._impersonate = impersonate
        self._timeout_seconds = timeout_seconds
        self._ua_suffix = ua_suffix
        self._user_agent = f"{_BASE_USER_AGENT}; {ua_suffix}" if ua_suffix else _BASE_USER_AGENT
        # Typed as Any because the underlying ``curl_cffi.requests.Session`` is
        # a Generic[R] whose type parameter we don't care about here, and we
        # also want to allow tests to substitute a fake via monkeypatch.
        self._session: Any | None = None
        self._closed = False

        logger.info(
            "CurlCffiFetcher configured: impersonate=%s timeout=%.1fs ua_suffix=%s",
            self._impersonate,
            self._timeout_seconds,
            self._ua_suffix,
        )

    # ------------------------------------------------------------ public API

    def fetch(self, url: str) -> FetchedPage:
        """Fetch ``url`` synchronously; raise :class:`FetchError` on failure.

        Returns a :class:`FetchedPage` with the final URL after redirects,
        HTTP status, response text, and a naive UTC ``fetched_at`` timestamp.
        """
        session = self._ensure_session()
        try:
            response: Any = session.get(
                url,
                allow_redirects=True,
                timeout=self._timeout_seconds,
            )
        except FetchError:
            raise
        except Exception as exc:
            raise FetchError(f"transport: {exc!r}") from exc

        status = int(getattr(response, "status_code", 0))
        if status >= 400:
            raise FetchError(f"HTTP {status} for {url}")

        final_url = str(getattr(response, "url", url))
        html = str(getattr(response, "text", ""))
        return FetchedPage(
            url=final_url,
            status=status,
            html=html,
            fetched_at=datetime.now(UTC).replace(tzinfo=None),
        )

    def close(self) -> None:
        """Close the underlying Session; safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        session = self._session
        self._session = None
        if session is None:
            return
        try:
            session.close()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("CurlCffiFetcher session close raised: %r", exc)

    def __enter__(self) -> CurlCffiFetcher:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --------------------------------------------------------------- helpers

    def _ensure_session(self) -> Any:
        """Lazy-create the underlying ``curl_cffi.requests.Session``."""
        if self._closed:
            raise FetchError("CurlCffiFetcher has been closed")
        if self._session is not None:
            return self._session
        try:
            from curl_cffi.requests import Session as _Session  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - dep listed in pyproject
            raise FetchError("curl_cffi is not installed; run `uv pip install -e .[dev]`") from exc
        # ``impersonate`` accepts a Literal of known profile names. We pass
        # whatever the caller provided; curl_cffi raises ImpersonateError on
        # unknown values, which we surface as a FetchError on first fetch().
        session: Any = _Session(impersonate=self._impersonate)  # type: ignore[arg-type]
        # Override the User-Agent so target sites can identify the project.
        # The TLS / JA3 fingerprint comes from the impersonation profile and
        # is independent of the UA string.
        session.headers.update({"User-Agent": self._user_agent})
        self._session = session
        return session
