"""Shared proxy-URL parsing for the Playwright and curl_cffi fetchers.

Users supply a single proxy URL via ``--proxy`` / ``PROXY_URL``; this module
converts that URL into the shape each fetcher needs:

* :func:`parse_proxy_url` — Playwright's expected dict
  (``{"server": "...", "username": "...", "password": "..."}``).
* :func:`proxy_url_to_curl_dict` — curl_cffi's expected dict
  (``{"http": url, "https": url}``).
* :func:`validate_proxy_url` — round-trip validation that returns the
  original URL after checking; raises :class:`ValueError` on invalid input.

Accepted schemes: ``http://``, ``https://``, ``socks4://``, ``socks5://``.
Optional ``user:pass@`` credentials are preserved verbatim.

Validation runs without importing Playwright or curl_cffi so callers can
fail fast on bad configuration before launching either fetcher (preserves
the lazy-import contracts in :mod:`browser` and :mod:`curlcffi_fetcher`).

Security note: callers MUST NOT log the full proxy URL (it contains
credentials). Use :func:`mask_proxy_url` to emit a credentials-free
representation suitable for startup logs.
"""

from __future__ import annotations

from urllib.parse import urlparse

ALLOWED_PROXY_SCHEMES: frozenset[str] = frozenset({"http", "https", "socks4", "socks5"})
"""Proxy schemes accepted by both fetcher back-ends."""


def validate_proxy_url(url: str) -> str:
    """Return ``url`` unchanged after structural validation; raise on invalid.

    Validates that the URL has an allowed scheme (``http``, ``https``,
    ``socks4``, ``socks5``) and includes both a host and a port. Credentials
    (``user:pass@``) are optional.

    Raises :class:`ValueError` with a clear message on any of:

    * empty / whitespace-only URL
    * unsupported / missing scheme
    * missing host
    * missing port
    """
    if not url or not url.strip():
        raise ValueError("empty proxy URL")
    parsed = urlparse(url)
    if not parsed.scheme:
        raise ValueError(f"proxy URL must include a scheme: {url!r}")
    if parsed.scheme not in ALLOWED_PROXY_SCHEMES:
        raise ValueError(
            f"unsupported proxy scheme: {parsed.scheme!r} "
            f"(allowed: {sorted(ALLOWED_PROXY_SCHEMES)})"
        )
    # Build a credentials-free reference for any remaining error messages.
    # NEVER include `url`, `parsed.netloc`, `parsed.username`, or `parsed.password`
    # in error messages -- they flow to stderr / CI logs / error reporters.
    safe_endpoint = f"{parsed.scheme}://{parsed.hostname or '<no-host>'}"
    if not parsed.hostname:
        raise ValueError(f"proxy URL must include host: {safe_endpoint}")
    if parsed.port is None:
        raise ValueError(f"proxy URL must include port: {safe_endpoint}")
    return url


def parse_proxy_url(url: str) -> dict[str, str]:
    """Parse a proxy URL into Playwright's proxy dict format.

    Accepts ``http://``, ``https://``, ``socks4://``, ``socks5://`` URLs with
    optional ``user:pass@`` credentials. Returns
    ``{"server": "scheme://host:port"}`` plus ``"username"`` and
    ``"password"`` if credentials were present.

    Raises :class:`ValueError` on invalid input (see :func:`validate_proxy_url`).
    """
    validate_proxy_url(url)
    parsed = urlparse(url)
    # validate_proxy_url guarantees scheme, hostname, and port are present.
    assert parsed.hostname is not None
    assert parsed.port is not None
    result: dict[str, str] = {"server": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"}
    if parsed.username:
        result["username"] = parsed.username
    if parsed.password:
        result["password"] = parsed.password
    return result


def proxy_url_to_curl_dict(url: str) -> dict[str, str]:
    """Convert a proxy URL into curl_cffi's ``proxies={"http": ..., "https": ...}`` dict.

    curl_cffi uses the same URL form for both schemes, so we hand it the full
    URL (credentials intact) for both keys. Raises :class:`ValueError` on
    invalid input (see :func:`validate_proxy_url`).
    """
    validate_proxy_url(url)
    return {"http": url, "https": url}


def mask_proxy_url(url: str) -> str:
    """Return a credentials-free ``scheme://host:port`` for safe logging.

    Does NOT validate — call :func:`validate_proxy_url` first when the input
    is user-supplied. If the URL is unparseable we return the literal string
    ``"<unparseable proxy URL>"`` rather than leaking the input back into logs.
    """
    try:
        parsed = urlparse(url)
    except Exception:  # pragma: no cover - urlparse is permissive
        return "<unparseable proxy URL>"
    if not parsed.scheme or not parsed.hostname or parsed.port is None:
        return "<unparseable proxy URL>"
    return f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
