"""Tests for :class:`ImageDownloader`.

We never make real HTTP calls — ``curl_cffi.requests.Session`` is replaced
with a fake that returns canned ``status_code`` / ``content`` / ``headers``.
"""

from __future__ import annotations

import io
import os
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from PIL import Image as PILImage

from car_lense_engine.crawler.core.image_downloader import (
    ImageDownloader,
    ImageDownloadError,
    _atomic_write_bytes,
)
from car_lense_engine.db import Listing, listings, open_db

# ---------- DB / listing fixtures -------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = open_db(tmp_path / "crawl.sqlite")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def listing_id(db: sqlite3.Connection) -> str:
    """Insert a parent listing row so the FK on ``images.listing_id`` is satisfied."""
    lid = "cars_com:1"
    listings.insert_listing(
        db,
        Listing(
            listing_id=lid,
            source="cars_com",
            url="https://cars.com/listing/1",
        ),
    )
    return lid


# ---------- Fake curl_cffi.requests.Session ---------------------------------


@dataclass
class _FakeHeaders:
    data: dict[str, str] = field(default_factory=dict)

    def update(self, other: dict[str, str]) -> None:
        self.data.update(other)

    def get(self, key: str, default: Any = None) -> Any:
        for k, v in self.data.items():
            if k.lower() == key.lower():
                return v
        return default


@dataclass
class _FakeResponse:
    status_code: int
    content: bytes
    headers: dict[str, str]
    url: str = ""
    # Optional override: when provided, iter_content yields these chunks
    # verbatim (used for tests that need to assert mid-stream behavior).
    chunks: list[bytes] | None = None
    iter_calls: int = 0

    def iter_content(self, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
        self.iter_calls += 1
        if self.chunks is not None:
            yield from self.chunks
            return
        body = self.content
        for i in range(0, len(body), chunk_size):
            yield body[i : i + chunk_size]


@dataclass
class _FakeSession:
    impersonate: str | None = None
    headers: _FakeHeaders = field(default_factory=_FakeHeaders)
    calls: list[dict[str, Any]] = field(default_factory=list)
    response_for: dict[str, _FakeResponse] = field(default_factory=dict)
    default_response: _FakeResponse | None = None
    raise_exc: Exception | None = None
    closed: bool = False

    def get(
        self,
        url: str,
        *,
        allow_redirects: bool = True,
        timeout: float | None = None,
        stream: bool = False,
    ) -> _FakeResponse:
        self.calls.append(
            {
                "url": url,
                "allow_redirects": allow_redirects,
                "timeout": timeout,
                "stream": stream,
            }
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        if url in self.response_for:
            return self.response_for[url]
        if self.default_response is not None:
            return self.default_response
        raise AssertionError(f"_FakeSession has no response configured for {url}")

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_sessions(monkeypatch: pytest.MonkeyPatch) -> list[_FakeSession]:
    sessions: list[_FakeSession] = []

    def _factory(**kwargs: Any) -> _FakeSession:
        session = _FakeSession(impersonate=kwargs.get("impersonate"))
        sessions.append(session)
        return session

    import curl_cffi.requests

    monkeypatch.setattr(curl_cffi.requests, "Session", _factory)
    return sessions


# ---------- helpers ----------------------------------------------------------


def _png_bytes(w: int = 32, h: int = 32, color: tuple[int, int, int] = (200, 50, 50)) -> bytes:
    """Generate real PNG bytes via Pillow so decode/pHash actually succeed."""
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _set_response(
    fake_sessions: list[_FakeSession],
    url: str,
    *,
    status: int = 200,
    body: bytes = b"",
    content_type: str = "image/png",
    content_length: int | None = None,
    chunks: list[bytes] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> _FakeResponse:
    """Pre-register a response for ``url`` against the most recent fake session.

    ImageDownloader builds the session lazily on first ``download(...)``; tests
    that need to pre-register a response trigger ``_ensure_session`` first.
    """
    assert fake_sessions, "expected at least one fake session to have been built"
    session = fake_sessions[-1]
    headers: dict[str, str] = {"Content-Type": content_type}
    if content_length is not None:
        headers["Content-Length"] = str(content_length)
    if extra_headers:
        headers.update(extra_headers)
    response = _FakeResponse(
        status_code=status,
        content=body,
        headers=headers,
        url=url,
        chunks=chunks,
    )
    session.response_for[url] = response
    return response


# =============================================================================
# Test 1 — successful download
# =============================================================================


def test_successful_download_writes_file_and_inserts_row(
    db: sqlite3.Connection,
    listing_id: str,
    tmp_path: Path,
    fake_sessions: list[_FakeSession],
) -> None:
    body = _png_bytes()
    url = "https://cdn.example.com/photo123"

    dl = ImageDownloader(data_root=tmp_path / "raw")
    dl._ensure_session()  # noqa: SLF001 — trigger lazy session so we can configure it
    _set_response(fake_sessions, url, body=body, content_type="image/png")

    image = dl.download(db, url, source="cars_com", listing_id=listing_id, position=0)
    dl.close()

    assert image is not None
    assert image.image_id  # sha256 hex
    assert len(image.image_id) == 64
    assert image.bytes == len(body)
    assert image.width == 32
    assert image.height == 32
    assert image.phash is not None and image.phash != ""
    assert image.position == 0

    # File written to <data_root>/<source>/<listing_id>/<sha>.png
    expected_path = tmp_path / "raw" / "cars_com" / listing_id / f"{image.image_id}.png"
    assert expected_path.exists()
    assert expected_path.read_bytes() == body
    assert image.local_path == str(expected_path)

    # Row inserted.
    row = db.execute(
        "SELECT image_id, listing_id, source_url, local_path, bytes, width, height, phash "
        "FROM images WHERE image_id = ?",
        (image.image_id,),
    ).fetchone()
    assert row is not None
    assert row["listing_id"] == listing_id
    assert row["source_url"] == url
    assert row["bytes"] == len(body)
    assert row["phash"] == image.phash


# =============================================================================
# Test 2 — content-type rejection
# =============================================================================


def test_rejects_non_image_content_type(
    db: sqlite3.Connection,
    listing_id: str,
    tmp_path: Path,
    fake_sessions: list[_FakeSession],
) -> None:
    url = "https://example.com/decoy"
    dl = ImageDownloader(data_root=tmp_path / "raw")
    dl._ensure_session()  # noqa: SLF001
    _set_response(fake_sessions, url, body=b"<html>not an image</html>", content_type="text/html")

    with pytest.raises(ImageDownloadError, match="unsupported content-type"):
        dl.download(db, url, source="cars_com", listing_id=listing_id)
    dl.close()

    # No file written.
    assert list((tmp_path / "raw").rglob("*")) == [] or all(
        p.is_dir() for p in (tmp_path / "raw").rglob("*")
    )
    # No row inserted.
    n = db.execute("SELECT COUNT(*) AS n FROM images").fetchone()
    assert int(n["n"]) == 0


# =============================================================================
# Test 3 — oversized body
# =============================================================================


def test_rejects_oversized_body(
    db: sqlite3.Connection,
    listing_id: str,
    tmp_path: Path,
    fake_sessions: list[_FakeSession],
) -> None:
    url = "https://cdn.example.com/huge"
    body = b"x" * 1024  # 1 KB
    dl = ImageDownloader(data_root=tmp_path / "raw", max_bytes=100)
    dl._ensure_session()  # noqa: SLF001
    _set_response(fake_sessions, url, body=body, content_type="image/png")

    with pytest.raises(ImageDownloadError, match="too large"):
        dl.download(db, url, source="cars_com", listing_id=listing_id)
    dl.close()

    n = db.execute("SELECT COUNT(*) AS n FROM images").fetchone()
    assert int(n["n"]) == 0


# =============================================================================
# Test 4 — HTTP 404
# =============================================================================


def test_raises_on_http_error(
    db: sqlite3.Connection,
    listing_id: str,
    tmp_path: Path,
    fake_sessions: list[_FakeSession],
) -> None:
    url = "https://cdn.example.com/gone"
    dl = ImageDownloader(data_root=tmp_path / "raw")
    dl._ensure_session()  # noqa: SLF001
    _set_response(fake_sessions, url, status=404, body=b"", content_type="text/html")

    with pytest.raises(ImageDownloadError, match="HTTP 404"):
        dl.download(db, url, source="cars_com", listing_id=listing_id)
    dl.close()


# =============================================================================
# Test 5 — idempotent duplicate
# =============================================================================


def test_idempotent_duplicate_returns_none(
    db: sqlite3.Connection,
    listing_id: str,
    tmp_path: Path,
    fake_sessions: list[_FakeSession],
) -> None:
    body = _png_bytes(color=(10, 20, 30))
    url = "https://cdn.example.com/once"

    dl = ImageDownloader(data_root=tmp_path / "raw")
    dl._ensure_session()  # noqa: SLF001
    _set_response(fake_sessions, url, body=body, content_type="image/png")

    first = dl.download(db, url, source="cars_com", listing_id=listing_id)
    assert first is not None
    expected_path = tmp_path / "raw" / "cars_com" / listing_id / f"{first.image_id}.png"
    assert expected_path.exists()

    # Second download of identical bytes: no exception, returns None, no
    # duplicate row, file untouched (same hash means it was already on disk).
    mtime_before = expected_path.stat().st_mtime_ns
    second = dl.download(db, url, source="cars_com", listing_id=listing_id)
    assert second is None
    assert expected_path.stat().st_mtime_ns == mtime_before

    n = db.execute("SELECT COUNT(*) AS n FROM images").fetchone()
    assert int(n["n"]) == 1
    dl.close()


# =============================================================================
# Test 6 — decode failure
# =============================================================================


def test_raises_on_decode_failure(
    db: sqlite3.Connection,
    listing_id: str,
    tmp_path: Path,
    fake_sessions: list[_FakeSession],
) -> None:
    url = "https://cdn.example.com/garbage"
    dl = ImageDownloader(data_root=tmp_path / "raw")
    dl._ensure_session()  # noqa: SLF001
    _set_response(
        fake_sessions,
        url,
        body=b"not a real image",
        content_type="image/png",
    )

    with pytest.raises(ImageDownloadError, match="decode failed"):
        dl.download(db, url, source="cars_com", listing_id=listing_id)
    dl.close()

    # Disk untouched; DB untouched.
    n = db.execute("SELECT COUNT(*) AS n FROM images").fetchone()
    assert int(n["n"]) == 0


# =============================================================================
# Test 7 — invalid proxy URL rejected before any HTTP
# =============================================================================


def test_invalid_proxy_raises_before_session_built(tmp_path: Path) -> None:
    # No fake_sessions fixture — if the constructor ever reached
    # _ensure_session, the real curl_cffi.requests.Session would be invoked
    # and the test would fail differently (or hit the network).
    with pytest.raises(ValueError):
        ImageDownloader(data_root=tmp_path / "raw", proxy="not-a-url")


def test_empty_proxy_raises_before_session_built(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty proxy URL"):
        ImageDownloader(data_root=tmp_path / "raw", proxy="")


# =============================================================================
# Test 8 — atomic write: replace() failure cleans up .tmp
# =============================================================================


def test_atomic_write_cleans_up_tmp_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "out" / "abc.png"

    def _boom(src: str, dst: str) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(ImageDownloadError, match="write failed"):
        _atomic_write_bytes(target, b"some-bytes")

    # Final file should not exist; .tmp should also be cleaned up so a retry
    # can proceed without tripping on stale state.
    assert not target.exists()
    tmp_files = list(target.parent.glob("*.tmp"))
    assert tmp_files == []


# =============================================================================
# Constructor / config validation
# =============================================================================


def test_rejects_invalid_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        ImageDownloader(data_root=tmp_path / "raw", timeout_seconds=0.0)


def test_rejects_invalid_max_bytes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        ImageDownloader(data_root=tmp_path / "raw", max_bytes=0)


def test_rejects_empty_impersonate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="impersonate"):
        ImageDownloader(data_root=tmp_path / "raw", impersonate="")


def test_proxy_credentials_not_logged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="car_lense_engine.crawler.core.image_downloader")
    dl = ImageDownloader(
        data_root=tmp_path / "raw",
        proxy="http://hideuser:hidepw@gate.example.com:7000",
    )
    try:
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "hideuser" not in messages
        assert "hidepw" not in messages
        assert "gate.example.com:7000" in messages
    finally:
        dl.close()


def test_context_manager_closes_session(
    tmp_path: Path,
    fake_sessions: list[_FakeSession],
) -> None:
    with ImageDownloader(data_root=tmp_path / "raw") as dl:
        dl._ensure_session()  # noqa: SLF001
    session = fake_sessions[-1]
    assert session.closed is True


def test_download_after_close_raises(
    db: sqlite3.Connection,
    listing_id: str,
    tmp_path: Path,
    fake_sessions: list[_FakeSession],
) -> None:
    dl = ImageDownloader(data_root=tmp_path / "raw")
    dl._ensure_session()  # noqa: SLF001
    dl.close()
    with pytest.raises(ImageDownloadError, match="closed"):
        dl.download(db, "https://x/y", source="cars_com", listing_id=listing_id)


# =============================================================================
# FK violation: parent listing was deleted between enqueue and download.
# Must raise ImageDownloadError, not silently discard the row.
# =============================================================================


def test_orphan_listing_id_raises_image_download_error(
    db: sqlite3.Connection,
    tmp_path: Path,
    fake_sessions: list[_FakeSession],
) -> None:
    body = _png_bytes(color=(7, 9, 11))
    url = "https://cdn.example.com/orphan"
    orphan_listing_id = "cars_com:does-not-exist"

    dl = ImageDownloader(data_root=tmp_path / "raw")
    dl._ensure_session()  # noqa: SLF001
    _set_response(fake_sessions, url, body=body, content_type="image/png")

    with pytest.raises(ImageDownloadError, match="insert failed"):
        dl.download(db, url, source="cars_com", listing_id=orphan_listing_id)
    dl.close()

    # No row was created.
    n = db.execute("SELECT COUNT(*) AS n FROM images").fetchone()
    assert int(n["n"]) == 0

    # The bytes are intentionally left on disk so the next retry can re-detect
    # via the file-exists fast path and re-insert (same SHA-256 -> same path).
    import hashlib  # local import keeps the test self-contained

    sha = hashlib.sha256(body).hexdigest()
    expected_path = tmp_path / "raw" / "cars_com" / orphan_listing_id / f"{sha}.png"
    assert expected_path.exists()


# =============================================================================
# Streaming size cap: Content-Length pre-check rejects without body read.
# =============================================================================


def test_content_length_header_too_large_rejected_before_body(
    db: sqlite3.Connection,
    listing_id: str,
    tmp_path: Path,
    fake_sessions: list[_FakeSession],
) -> None:
    url = "https://cdn.example.com/declared-huge"
    dl = ImageDownloader(data_root=tmp_path / "raw", max_bytes=1024)
    dl._ensure_session()  # noqa: SLF001
    response = _set_response(
        fake_sessions,
        url,
        body=b"",  # body must never be read
        content_type="image/jpeg",
        content_length=999_999_999,
    )

    with pytest.raises(ImageDownloadError, match="Content-Length"):
        dl.download(db, url, source="cars_com", listing_id=listing_id)
    dl.close()

    # iter_content must not have been invoked at all.
    assert response.iter_calls == 0
    # No file, no row.
    n = db.execute("SELECT COUNT(*) AS n FROM images").fetchone()
    assert int(n["n"]) == 0


# =============================================================================
# Streaming size cap: running-total check aborts mid-stream when
# Content-Length is absent (or lies).
# =============================================================================


def test_streaming_running_total_exceeds_cap(
    db: sqlite3.Connection,
    listing_id: str,
    tmp_path: Path,
    fake_sessions: list[_FakeSession],
) -> None:
    url = "https://cdn.example.com/sneaky"
    # 5 chunks of 1 MiB; cap is 2 MiB. Cap is crossed on chunk 3.
    chunk = b"\xab" * (1024 * 1024)
    chunks = [chunk] * 5
    max_bytes = 2 * 1024 * 1024

    dl = ImageDownloader(data_root=tmp_path / "raw", max_bytes=max_bytes)
    dl._ensure_session()  # noqa: SLF001
    _set_response(
        fake_sessions,
        url,
        body=b"",  # iter_content uses chunks override; .content is ignored
        content_type="image/jpeg",
        chunks=chunks,
        # No Content-Length header: forces the running-total path.
    )

    with pytest.raises(ImageDownloadError, match="too large"):
        dl.download(db, url, source="cars_com", listing_id=listing_id)
    dl.close()

    n = db.execute("SELECT COUNT(*) AS n FROM images").fetchone()
    assert int(n["n"]) == 0
