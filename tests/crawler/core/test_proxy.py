"""Tests for :mod:`car_lense_engine.crawler.core.proxy`.

Validation must run without importing Playwright or curl_cffi (the fetcher
back-ends both delegate URL validation to this module so they can fail fast
on bad config without touching their heavy lazy imports).
"""

from __future__ import annotations

import pytest

from car_lense_engine.crawler.core.proxy import (
    mask_proxy_url,
    parse_proxy_url,
    proxy_url_to_curl_dict,
    validate_proxy_url,
)

# ----------------------------------------------------------- parse_proxy_url


def test_parse_proxy_url_basic() -> None:
    """http://host:8080 → {'server': 'http://host:8080'} (no creds)."""
    assert parse_proxy_url("http://host:8080") == {"server": "http://host:8080"}


def test_parse_proxy_url_with_credentials() -> None:
    """http://u:p@host:8080 must include username and password."""
    result = parse_proxy_url("http://u:p@host:8080")
    assert result["server"] == "http://host:8080"
    assert result["username"] == "u"
    assert result["password"] == "p"


def test_parse_proxy_url_socks5() -> None:
    """socks5://host:1080 must preserve the socks5 scheme on the server URL."""
    result = parse_proxy_url("socks5://host:1080")
    assert result == {"server": "socks5://host:1080"}


def test_parse_proxy_url_https() -> None:
    """https://host:443 round-trips through the parser cleanly."""
    assert parse_proxy_url("https://host:443") == {"server": "https://host:443"}


def test_parse_proxy_url_socks4_with_credentials() -> None:
    """SOCKS4 with credentials still produces the expected dict shape."""
    result = parse_proxy_url("socks4://alice:secret@host:1080")
    assert result["server"] == "socks4://host:1080"
    assert result["username"] == "alice"
    assert result["password"] == "secret"


def test_parse_proxy_url_missing_scheme() -> None:
    """A bare host:port (no scheme) must raise ValueError."""
    with pytest.raises(ValueError, match="scheme"):
        parse_proxy_url("host:8080")


def test_parse_proxy_url_unsupported_scheme() -> None:
    """ftp:// is not an accepted proxy scheme."""
    with pytest.raises(ValueError, match="unsupported proxy scheme"):
        parse_proxy_url("ftp://host:21")


def test_parse_proxy_url_missing_port() -> None:
    """A URL without a port must raise ValueError."""
    with pytest.raises(ValueError, match="port"):
        parse_proxy_url("http://host")


def test_parse_proxy_url_missing_host() -> None:
    """A URL without a hostname must raise ValueError."""
    with pytest.raises(ValueError, match="host"):
        parse_proxy_url("http://:8080")


def test_parse_proxy_url_empty() -> None:
    """An empty string must raise ValueError with a clear message."""
    with pytest.raises(ValueError, match="empty proxy URL"):
        parse_proxy_url("")


def test_parse_proxy_url_whitespace() -> None:
    """A whitespace-only URL is treated as empty."""
    with pytest.raises(ValueError, match="empty proxy URL"):
        parse_proxy_url("   ")


def test_parse_proxy_url_missing_port_does_not_leak_credentials() -> None:
    """Missing-port error must not include user:pass credentials."""
    url = "http://supersecretuser:supersecretpass@host"  # no port
    try:
        parse_proxy_url(url)
    except ValueError as exc:
        msg = str(exc)
        assert "supersecretpass" not in msg
        assert "supersecretuser" not in msg
    else:
        raise AssertionError("expected ValueError")


def test_parse_proxy_url_missing_host_does_not_leak_credentials() -> None:
    """Missing-host error must not include user:pass credentials."""
    url = "http://supersecretuser:supersecretpass@:8080"  # creds present, no hostname
    try:
        parse_proxy_url(url)
    except ValueError as exc:
        msg = str(exc)
        assert "supersecretpass" not in msg
        assert "supersecretuser" not in msg
    else:
        raise AssertionError("expected ValueError")


# ----------------------------------------------------------- validate_proxy_url


def test_validate_proxy_url_returns_original() -> None:
    """validate_proxy_url returns the original URL unchanged on success."""
    url = "http://u:p@host:8080"
    assert validate_proxy_url(url) == url


def test_validate_proxy_url_rejects_bad_input() -> None:
    """validate_proxy_url raises on every invalid form parse_proxy_url rejects."""
    with pytest.raises(ValueError):
        validate_proxy_url("")
    with pytest.raises(ValueError):
        validate_proxy_url("not-a-url")
    with pytest.raises(ValueError):
        validate_proxy_url("ftp://host:21")


# ----------------------------------------------------------- proxy_url_to_curl_dict


def test_proxy_url_to_curl_dict() -> None:
    """The curl_cffi shape is {'http': url, 'https': url} with the full URL."""
    url = "http://host:8080"
    assert proxy_url_to_curl_dict(url) == {"http": url, "https": url}


def test_proxy_url_to_curl_dict_preserves_credentials() -> None:
    """Credentials must remain in the URL passed to curl_cffi."""
    url = "http://u:p@gate.example.com:7000"
    assert proxy_url_to_curl_dict(url) == {"http": url, "https": url}


def test_proxy_url_to_curl_dict_rejects_invalid() -> None:
    """Bad URLs raise ValueError before producing a dict."""
    with pytest.raises(ValueError):
        proxy_url_to_curl_dict("")
    with pytest.raises(ValueError):
        proxy_url_to_curl_dict("ftp://host:21")


# ----------------------------------------------------------- mask_proxy_url


def test_mask_proxy_url_strips_credentials() -> None:
    """Masking must drop the user:pass@ portion."""
    masked = mask_proxy_url("http://user:pass@gate.smartproxy.com:7000")
    assert "user" not in masked
    assert "pass" not in masked
    assert masked == "http://gate.smartproxy.com:7000"


def test_mask_proxy_url_no_credentials_round_trip() -> None:
    """URLs without credentials should round-trip unchanged."""
    assert mask_proxy_url("http://host:8080") == "http://host:8080"


def test_mask_proxy_url_unparseable_falls_back_safely() -> None:
    """An unparseable URL produces a safe placeholder, not the raw input."""
    masked = mask_proxy_url("not-a-real-url")
    assert "not-a-real-url" not in masked
