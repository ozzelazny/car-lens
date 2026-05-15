"""Tests for :class:`CurlCffiFetcher`.

We never make real HTTP calls; the ``curl_cffi.requests.Session`` class is
monkey-patched with a fake that records inputs and returns a canned response
(or raises). This keeps the suite fast and offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from car_lense_engine.crawler.core import curlcffi_fetcher as ccf_mod
from car_lense_engine.crawler.core.curlcffi_fetcher import (
    DEFAULT_IMPERSONATE,
    CurlCffiFetcher,
)
from car_lense_engine.crawler.core.fetcher import FetchError


@dataclass
class _FakeResponse:
    status_code: int
    text: str
    url: str


@dataclass
class _FakeHeaders:
    data: dict[str, str] = field(default_factory=dict)

    def update(self, other: dict[str, str]) -> None:
        self.data.update(other)


@dataclass
class _FakeSession:
    """Records get() calls; returns ``response_for(url)`` or raises ``raise_exc``."""

    impersonate: str | None = None
    headers: _FakeHeaders = field(default_factory=_FakeHeaders)
    calls: list[dict[str, Any]] = field(default_factory=list)
    response_for: dict[str, _FakeResponse] = field(default_factory=dict)
    default_response: _FakeResponse | None = None
    raise_exc: Exception | None = None
    closed: bool = False
    close_count: int = 0

    def get(
        self,
        url: str,
        *,
        allow_redirects: bool = True,
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.calls.append({"url": url, "allow_redirects": allow_redirects, "timeout": timeout})
        if self.raise_exc is not None:
            raise self.raise_exc
        if url in self.response_for:
            return self.response_for[url]
        if self.default_response is not None:
            return self.default_response
        # Synthesize a default 200.
        return _FakeResponse(status_code=200, text="<html></html>", url=url)

    def close(self) -> None:
        self.closed = True
        self.close_count += 1


@pytest.fixture
def fake_session_holder(monkeypatch: pytest.MonkeyPatch) -> list[_FakeSession]:
    """Patch ``curl_cffi.requests.Session`` so construction returns ``_FakeSession``.

    Returns a list that captures every constructed session, so tests can
    assert what was passed into the patched constructor.
    """
    sessions: list[_FakeSession] = []

    def _factory(**kwargs: Any) -> _FakeSession:
        session = _FakeSession(impersonate=kwargs.get("impersonate"))
        sessions.append(session)
        return session

    # Patch at the curl_cffi level so the lazy-import inside the fetcher
    # picks up our fake when it imports curl_cffi.requests.Session.
    import curl_cffi.requests

    monkeypatch.setattr(curl_cffi.requests, "Session", _factory)
    return sessions


def test_module_imports_without_instantiating_curl_cffi() -> None:
    """Importing the fetcher module must not crash even though curl_cffi is lazy.

    We can't easily assert ``curl_cffi`` is absent from ``sys.modules`` because
    it IS installed in the dev environment and is imported by the test
    machinery. The contract we DO verify: this module's top-level statements
    succeed and the class object is available without touching curl_cffi.
    """
    assert ccf_mod.CurlCffiFetcher is CurlCffiFetcher
    assert ccf_mod.DEFAULT_IMPERSONATE == "chrome131"


def test_fetcher_returns_fetched_page_on_200(fake_session_holder: list[_FakeSession]) -> None:
    fetcher = CurlCffiFetcher()
    # Pre-register a canned response. The session is created lazily on first
    # fetch(), so we wire it up via fake_session_holder after fetch fires.
    url = "https://www.cars.com/shopping/results/"
    # Trigger session creation so we can configure it.
    fetcher._ensure_session()  # noqa: SLF001 - test access
    assert fake_session_holder, "expected fake Session to have been constructed"
    session = fake_session_holder[-1]
    session.response_for[url] = _FakeResponse(status_code=200, text="<html>ok</html>", url=url)

    page = fetcher.fetch(url)
    assert page.status == 200
    assert page.html == "<html>ok</html>"
    assert page.url == url
    assert page.fetched_at is not None
    fetcher.close()


def test_fetcher_raises_on_403(fake_session_holder: list[_FakeSession]) -> None:
    fetcher = CurlCffiFetcher()
    fetcher._ensure_session()  # noqa: SLF001
    session = fake_session_holder[-1]
    url = "https://www.hemmings.com/classifieds"
    session.response_for[url] = _FakeResponse(status_code=403, text="forbidden", url=url)

    with pytest.raises(FetchError, match="HTTP 403"):
        fetcher.fetch(url)
    fetcher.close()


def test_fetcher_raises_on_5xx(fake_session_holder: list[_FakeSession]) -> None:
    fetcher = CurlCffiFetcher()
    fetcher._ensure_session()  # noqa: SLF001
    session = fake_session_holder[-1]
    url = "https://www.cars.com/x"
    session.response_for[url] = _FakeResponse(status_code=503, text="busy", url=url)

    with pytest.raises(FetchError, match="HTTP 503"):
        fetcher.fetch(url)
    fetcher.close()


def test_fetcher_raises_on_timeout(fake_session_holder: list[_FakeSession]) -> None:
    fetcher = CurlCffiFetcher()
    fetcher._ensure_session()  # noqa: SLF001
    session = fake_session_holder[-1]
    session.raise_exc = TimeoutError("read timed out")

    with pytest.raises(FetchError, match="transport:"):
        fetcher.fetch("https://www.cars.com/x")
    fetcher.close()


def test_fetcher_ua_suffix_appended_to_session_header(
    fake_session_holder: list[_FakeSession],
) -> None:
    fetcher = CurlCffiFetcher(ua_suffix="MyBot/9.9")
    fetcher._ensure_session()  # noqa: SLF001
    session = fake_session_holder[-1]
    ua = session.headers.data.get("User-Agent")
    assert ua is not None
    # The composed UA must include the impersonation-profile base AND our suffix.
    assert "Chrome/" in ua
    assert ua.endswith("; MyBot/9.9")
    fetcher.close()


def test_fetcher_close_idempotent(fake_session_holder: list[_FakeSession]) -> None:
    fetcher = CurlCffiFetcher()
    fetcher._ensure_session()  # noqa: SLF001
    session = fake_session_holder[-1]

    fetcher.close()
    fetcher.close()
    fetcher.close()
    assert session.close_count == 1


def test_fetcher_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        CurlCffiFetcher(timeout_seconds=0.0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        CurlCffiFetcher(timeout_seconds=-1.0)


def test_fetcher_rejects_empty_impersonate() -> None:
    with pytest.raises(ValueError, match="impersonate"):
        CurlCffiFetcher(impersonate="")


def test_fetcher_default_impersonate_is_chrome131(
    fake_session_holder: list[_FakeSession],
) -> None:
    fetcher = CurlCffiFetcher()
    fetcher._ensure_session()  # noqa: SLF001
    session = fake_session_holder[-1]
    assert session.impersonate == DEFAULT_IMPERSONATE == "chrome131"
    fetcher.close()


def test_fetcher_passes_timeout_to_session_get(
    fake_session_holder: list[_FakeSession],
) -> None:
    fetcher = CurlCffiFetcher(timeout_seconds=12.5)
    fetcher._ensure_session()  # noqa: SLF001
    session = fake_session_holder[-1]
    session.default_response = _FakeResponse(status_code=200, text="x", url="https://x.test/")

    fetcher.fetch("https://x.test/")
    assert session.calls[-1]["timeout"] == 12.5
    assert session.calls[-1]["allow_redirects"] is True
    fetcher.close()


def test_fetcher_uses_response_final_url(
    fake_session_holder: list[_FakeSession],
) -> None:
    """After redirects, FetchedPage.url must reflect the response.url."""
    fetcher = CurlCffiFetcher()
    fetcher._ensure_session()  # noqa: SLF001
    session = fake_session_holder[-1]
    requested = "https://www.cars.com/short"
    final = "https://www.cars.com/long/after/redirect"
    session.response_for[requested] = _FakeResponse(
        status_code=200, text="<html></html>", url=final
    )

    page = fetcher.fetch(requested)
    assert page.url == final
    fetcher.close()


def test_fetcher_after_close_cannot_fetch(fake_session_holder: list[_FakeSession]) -> None:
    fetcher = CurlCffiFetcher()
    fetcher._ensure_session()  # noqa: SLF001
    fetcher.close()
    with pytest.raises(FetchError, match="closed"):
        fetcher.fetch("https://x.test/")


def test_fetcher_context_manager_closes(fake_session_holder: list[_FakeSession]) -> None:
    with CurlCffiFetcher() as fetcher:
        fetcher._ensure_session()  # noqa: SLF001
        session = fake_session_holder[-1]
        session.default_response = _FakeResponse(status_code=200, text="ok", url="https://x.test/")
        fetcher.fetch("https://x.test/")
    assert session.closed is True


# ----------------------------------------------------------- proxy support


def test_curlcffi_fetcher_accepts_proxy(fake_session_holder: list[_FakeSession]) -> None:
    """A valid proxy URL configures session.proxies on the underlying Session."""
    fetcher = CurlCffiFetcher(proxy="http://u:p@gate.example.com:7000")
    fetcher._ensure_session()  # noqa: SLF001
    session = fake_session_holder[-1]
    # _FakeSession doesn't pre-declare a proxies attr; the fetcher assigns it.
    proxies = getattr(session, "proxies", None)
    assert isinstance(proxies, dict)
    assert proxies["http"] == "http://u:p@gate.example.com:7000"
    assert proxies["https"] == "http://u:p@gate.example.com:7000"
    fetcher.close()


def test_curlcffi_fetcher_no_proxy_leaves_session_proxies_unset(
    fake_session_holder: list[_FakeSession],
) -> None:
    """Without a proxy kwarg, the fetcher must NOT touch session.proxies.

    The fake session has no ``proxies`` attribute by default; if the fetcher
    set one we'd see it via ``hasattr``.
    """
    fetcher = CurlCffiFetcher()
    fetcher._ensure_session()  # noqa: SLF001
    session = fake_session_holder[-1]
    assert not hasattr(session, "proxies")
    fetcher.close()


def test_curlcffi_fetcher_rejects_invalid_proxy() -> None:
    """A bad proxy URL must raise ValueError before any session is created."""
    with pytest.raises(ValueError):
        CurlCffiFetcher(proxy="not-a-url")


def test_curlcffi_fetcher_rejects_empty_proxy_string() -> None:
    """An empty-string proxy URL must raise (not silently no-op)."""
    with pytest.raises(ValueError, match="empty proxy URL"):
        CurlCffiFetcher(proxy="")


def test_curlcffi_fetcher_rejects_unsupported_proxy_scheme() -> None:
    """ftp:// is not an accepted proxy scheme."""
    with pytest.raises(ValueError, match="unsupported proxy scheme"):
        CurlCffiFetcher(proxy="ftp://host:21")


def test_curlcffi_fetcher_proxy_credentials_not_in_logs(
    fake_session_holder: list[_FakeSession],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The startup log line must NOT contain proxy credentials."""
    caplog.set_level("INFO", logger="car_lense_engine.crawler.core.curlcffi_fetcher")
    fetcher = CurlCffiFetcher(proxy="http://supersecretuser:supersecretpass@gate.example.com:7000")
    try:
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "supersecretuser" not in messages
        assert "supersecretpass" not in messages
        assert "gate.example.com:7000" in messages
    finally:
        fetcher.close()
