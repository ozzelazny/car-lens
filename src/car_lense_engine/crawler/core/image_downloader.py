"""HTTP image downloader: fetch bytes, write atomically, compute pHash, insert row.

Listings parsed in Phase 2 enqueue image URLs as ``kind='image'`` queue items.
The Worker delegates those items to :class:`ImageDownloader`, which:

1. Fetches the bytes via ``curl_cffi`` (browser-impersonating TLS) so CDNs
   that cargo-cult Cloudflare's fingerprint checks still serve us bytes.
2. Validates ``Content-Type`` and a max-byte cap (defense against runaway
   downloads or HTML error pages served with image URLs).
3. Hashes the bytes (``image_id = sha256``), writes them atomically to
   ``<data_root>/<source>/<listing_id>/<image_id>.<ext>``.
4. Computes a perceptual hash (pHash) + image dimensions via Pillow /
   imagehash for downstream dedupe (Phase 3.3).
5. Inserts an :class:`Image` row, treating a PRIMARY KEY conflict on
   ``image_id`` as an idempotent success (the same bytes already exist —
   another listing shared the photo, or this is a retry of a partial run).

Retry/backoff is the queue layer's job (``mark_failed``/``next_try_at``);
this class raises :class:`ImageDownloadError` on any unrecoverable failure
and leaves it to the worker to call ``queue.mark_failed``.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import pathlib
import sqlite3
from contextlib import suppress
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

from car_lense_engine.db import images
from car_lense_engine.db.models import Image

from .proxy import mask_proxy_url, proxy_url_to_curl_dict

logger = logging.getLogger(__name__)

DEFAULT_IMPERSONATE: str = "chrome131"
DEFAULT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_UA_SUFFIX: str = "CarLenseResearch/0.1"
DEFAULT_MAX_BYTES: int = 25 * 1024 * 1024

# 64 KB strikes a balance between syscall overhead and the granularity at
# which an oversized body is detected and aborted.
_STREAM_CHUNK_BYTES: int = 64 * 1024

# Hardcoded base UA, matched to the ``chrome131`` impersonation profile.
# Kept in sync with CurlCffiFetcher: the JA3/JA4 fingerprint is what defeats
# fingerprint-based bot detection; the UA only has to look plausible.
_BASE_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Allowed image content-types and their file extensions. We never trust the
# URL's apparent extension — many CDNs serve ``/photo123`` (no extension)
# and the Content-Type header is the only reliable signal.
_CONTENT_TYPE_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


class ImageDownloadError(Exception):
    """Raised when an image cannot be downloaded or persisted."""


class ImageDownloader:
    """Fetch image bytes via curl_cffi and persist them to disk + DB.

    All side-effecting dependencies are configurable; the underlying
    ``curl_cffi.requests.Session`` is lazy-imported so importing this module
    does not require curl_cffi to be loadable (matches the contract used by
    :class:`CurlCffiFetcher`).
    """

    def __init__(
        self,
        *,
        data_root: pathlib.Path,
        impersonate: str = DEFAULT_IMPERSONATE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        ua_suffix: str = DEFAULT_UA_SUFFIX,
        proxy: str | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds!r}")
        if not impersonate:
            raise ValueError("impersonate must be a non-empty profile name")
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be > 0, got {max_bytes!r}")

        # Validate the proxy URL eagerly so a bad config fails fast — BEFORE
        # the lazy curl_cffi import that would otherwise be the first failure
        # site. See CurlCffiFetcher for the matching pattern.
        proxies: dict[str, str] | None = None
        if proxy is not None:
            proxies = proxy_url_to_curl_dict(proxy)

        self._data_root = pathlib.Path(data_root)
        self._impersonate = impersonate
        self._timeout_seconds = timeout_seconds
        self._ua_suffix = ua_suffix
        self._user_agent = f"{_BASE_USER_AGENT}; {ua_suffix}" if ua_suffix else _BASE_USER_AGENT
        self._max_bytes = max_bytes
        # ``Any``-typed because curl_cffi.requests.Session is Generic[R] and
        # we don't care about its type parameter; also lets tests substitute
        # a fake via monkeypatch.
        self._session: Any | None = None
        self._closed = False
        self._proxies: dict[str, str] | None = proxies
        self._proxy_log_repr: str | None = mask_proxy_url(proxy) if proxy is not None else None

        logger.info(
            "ImageDownloader configured: data_root=%s impersonate=%s timeout=%.1fs "
            "max_bytes=%d ua_suffix=%s proxy=%s",
            self._data_root,
            self._impersonate,
            self._timeout_seconds,
            self._max_bytes,
            self._ua_suffix,
            self._proxy_log_repr if self._proxies is not None else "<none>",
        )

    # ----------------------------------------------------------- public API

    def download(
        self,
        conn: sqlite3.Connection,
        url: str,
        *,
        source: str,
        listing_id: str,
        position: int | None = None,
    ) -> Image | None:
        """Fetch ``url``, persist bytes + DB row. Return :class:`Image` or ``None``.

        Returns ``None`` when the bytes are already present (same ``image_id``
        already in the ``images`` table) — treat as idempotent success.
        Raises :class:`ImageDownloadError` on any unrecoverable failure.
        """
        body, content_type = self._fetch_bytes(url)

        ext = _CONTENT_TYPE_EXT.get(content_type.lower())
        if ext is None:
            raise ImageDownloadError(
                f"unsupported content-type {content_type!r} for {url} "
                f"(allowed: {sorted(set(_CONTENT_TYPE_EXT))})"
            )

        image_id = hashlib.sha256(body).hexdigest()

        # Pre-check whether the same bytes are already persisted. This narrows
        # the IntegrityError window below to "real" violations (foreign key on
        # listing_id, NOT NULL, etc.) which must be surfaced — not swallowed —
        # to avoid data loss when the parent listing was deleted between
        # enqueue and download.
        if images.get_image_by_sha(conn, image_id) is not None:
            logger.debug(
                "image already in DB, skipping: source=%s listing=%s sha=%s",
                source,
                listing_id,
                image_id[:12],
            )
            return None

        width, height, phash = _decode_dimensions_and_phash(body, url=url)

        path = self._data_root / source / listing_id / f"{image_id}{ext}"
        _atomic_write_bytes(path, body)

        image = Image(
            image_id=image_id,
            listing_id=listing_id,
            source_url=url,
            local_path=str(path),
            phash=phash,
            width=width,
            height=height,
            bytes=len(body),
            position=position,
            downloaded_at=datetime.now(UTC).replace(tzinfo=None),
        )
        try:
            images.insert_image(conn, image)
        except sqlite3.IntegrityError as exc:
            # The pre-check above ruled out PK conflicts as far as this
            # process knows, but another concurrent worker could have raced
            # us. Re-check; if a row now exists, the race resolved cleanly.
            # Otherwise the violation is something else (FK on listing_id,
            # NOT NULL, ...) and must propagate — leave the on-disk bytes in
            # place so a subsequent retry can re-insert via the same path
            # without re-downloading.
            if images.get_image_by_sha(conn, image_id) is not None:
                logger.debug(
                    "image inserted concurrently, treating as success: source=%s listing=%s sha=%s",
                    source,
                    listing_id,
                    image_id[:12],
                )
                return None
            raise ImageDownloadError(
                f"insert failed for {url} (listing={listing_id}): {exc!r}"
            ) from exc

        logger.info(
            "downloaded: source=%s listing=%s bytes=%d sha=%s url=%s",
            source,
            listing_id,
            len(body),
            image_id[:12],
            url,
        )
        return image

    def close(self) -> None:
        """Close the underlying Session; safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        session = self._session
        self._session = None
        if session is None:
            return
        with suppress(Exception):  # pragma: no cover - defensive
            session.close()

    def __enter__(self) -> ImageDownloader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --------------------------------------------------------------- helpers

    def _fetch_bytes(self, url: str) -> tuple[bytes, str]:
        """Run the HTTP GET; return ``(body_bytes, content_type)``.

        Streams the body and enforces ``max_bytes`` defensively at two layers:
        first via ``Content-Length`` (cheap reject before any bytes are read),
        then via a running total over ``iter_content`` chunks (handles servers
        that omit or lie about Content-Length). This prevents a malicious or
        misconfigured server from forcing us to buffer a multi-GB body just
        to discover it exceeds the cap.
        """
        session = self._ensure_session()
        try:
            response: Any = session.get(
                url,
                allow_redirects=True,
                timeout=self._timeout_seconds,
                stream=True,
            )
        except ImageDownloadError:
            raise
        except Exception as exc:
            raise ImageDownloadError(f"transport: {exc!r}") from exc

        status = int(getattr(response, "status_code", 0))
        if status >= 400:
            logger.warning("image HTTP %d for %s", status, url)
            raise ImageDownloadError(f"HTTP {status} for {url}")

        headers = getattr(response, "headers", {}) or {}
        content_type = _extract_content_type(headers)

        declared = _content_length(headers)
        if declared is not None and declared > self._max_bytes:
            logger.warning(
                "image Content-Length exceeds max_bytes: url=%s declared=%d cap=%d",
                url,
                declared,
                self._max_bytes,
            )
            raise ImageDownloadError(
                f"response Content-Length {declared} exceeds cap {self._max_bytes} for {url}"
            )

        body = self._consume_body(response, url=url)
        return body, content_type

    def _consume_body(self, response: Any, *, url: str) -> bytes:
        """Read the response body, aborting as soon as ``max_bytes`` is crossed."""
        buf = bytearray()
        cap = self._max_bytes
        iter_content = getattr(response, "iter_content", None)
        if callable(iter_content):
            try:
                chunks = iter_content(chunk_size=_STREAM_CHUNK_BYTES)
            except TypeError:
                # Some clients accept positional only; fall back.
                chunks = iter_content(_STREAM_CHUNK_BYTES)
            try:
                for chunk in chunks:
                    if not chunk:
                        continue
                    if not isinstance(chunk, (bytes, bytearray)):  # pragma: no cover
                        raise ImageDownloadError(
                            f"expected bytes from iter_content, got {type(chunk).__name__}"
                        )
                    buf.extend(chunk)
                    if len(buf) > cap:
                        logger.warning(
                            "image exceeds max_bytes mid-stream: url=%s bytes=%d cap=%d",
                            url,
                            len(buf),
                            cap,
                        )
                        raise ImageDownloadError(
                            f"image too large: {len(buf)} bytes > cap {cap} for {url}"
                        )
            except ImageDownloadError:
                raise
            except Exception as exc:
                raise ImageDownloadError(f"transport: {exc!r}") from exc
            return bytes(buf)

        # Fallback: client doesn't support streaming. Content-Length pre-check
        # is our only line of defense here, so the worst case is a single
        # ``max_bytes``-sized buffered read (not multi-GB).
        body: bytes = getattr(response, "content", b"") or b""
        if not isinstance(body, (bytes, bytearray)):  # pragma: no cover - defensive
            raise ImageDownloadError(
                f"expected bytes from session.get(...).content, got {type(body).__name__}"
            )
        body = bytes(body)
        if len(body) > cap:
            logger.warning(
                "image exceeds max_bytes: url=%s bytes=%d cap=%d",
                url,
                len(body),
                cap,
            )
            raise ImageDownloadError(f"image too large: {len(body)} bytes > cap {cap} for {url}")
        return body

    def _ensure_session(self) -> Any:
        """Lazy-create the underlying ``curl_cffi.requests.Session``."""
        if self._closed:
            raise ImageDownloadError("ImageDownloader has been closed")
        if self._session is not None:
            return self._session
        try:
            from curl_cffi.requests import Session as _Session  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - dep listed in pyproject
            raise ImageDownloadError(
                "curl_cffi is not installed; run `uv pip install -e .[dev]`"
            ) from exc
        session: Any = _Session(impersonate=self._impersonate)  # type: ignore[arg-type]
        session.headers.update({"User-Agent": self._user_agent})
        if self._proxies is not None:
            session.proxies = dict(self._proxies)
        self._session = session
        return session


# --------------------------------------------------------- module-private helpers


def _extract_content_type(headers: Any) -> str:
    """Pull the bare media type out of a Content-Type header, lowercased.

    Accepts any mapping-ish object with ``.get`` (curl_cffi response headers
    behave like a case-insensitive dict). Strips any ``; charset=...`` /
    boundary parameters.
    """
    raw: str | None = None
    if hasattr(headers, "get"):
        raw = headers.get("content-type") or headers.get("Content-Type")
    if raw is None and isinstance(headers, dict):
        # Last-ditch: case-insensitive scan if .get didn't find it.
        for k, v in headers.items():
            if str(k).lower() == "content-type":
                raw = str(v)
                break
    if not raw:
        return ""
    return str(raw).split(";", 1)[0].strip().lower()


def _content_length(headers: Any) -> int | None:
    """Return the ``Content-Length`` header as an int, or ``None`` if missing/invalid."""
    raw: str | None = None
    if hasattr(headers, "get"):
        raw = headers.get("content-length") or headers.get("Content-Length")
    if raw is None and isinstance(headers, dict):
        for k, v in headers.items():
            if str(k).lower() == "content-length":
                raw = str(v)
                break
    if raw is None:
        return None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


def _decode_dimensions_and_phash(body: bytes, *, url: str) -> tuple[int, int, str]:
    """Return ``(width, height, phash_hex)`` for the image bytes.

    Pillow's ``Image.verify()`` invalidates the instance for further use, so
    we open once, read ``.size``, then convert + hash without re-opening from
    bytes a second time. Decode failures raise :class:`ImageDownloadError`.
    """
    try:
        from PIL import Image as PILImage  # noqa: PLC0415
        from PIL import UnidentifiedImageError  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dep listed in pyproject
        raise ImageDownloadError("Pillow is not installed") from exc
    try:
        import imagehash  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dep listed in pyproject
        raise ImageDownloadError("imagehash is not installed") from exc

    try:
        with PILImage.open(io.BytesIO(body)) as img:
            img.load()
            width, height = img.size
            # imagehash.phash needs an image it can convert to grayscale; use
            # a copy so we control the lifecycle (the original closes on exit).
            phash = imagehash.phash(img.copy())
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageDownloadError(f"decode failed for {url}: {exc!r}") from exc

    return int(width), int(height), str(phash)


def _atomic_write_bytes(path: pathlib.Path, body: bytes) -> None:
    """Write ``body`` to ``path`` via a ``.tmp`` rename. Idempotent on existing files.

    If ``path`` already exists we trust the SHA-256 in the filename and skip
    the write. Cleans up the temp file if ``os.replace`` raises so a partial
    write doesn't block a future retry.
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        with suppress(OSError):
            if tmp.exists():
                tmp.unlink()
        raise ImageDownloadError(f"write failed for {path}: {exc!r}") from exc
