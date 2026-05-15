"""Disk cache for NHTSA vPIC HTTP responses.

The cache is keyed by URL path: a sanitized filename derived from the full URL is
used to store the raw JSON response. On a cache hit we skip the HTTP call entirely.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize(url: str) -> str:
    """Produce a filesystem-safe filename from an arbitrary URL."""
    stripped = re.sub(r"^https?://", "", url)
    return _SANITIZE_RE.sub("_", stripped)


class JSONFileCache:
    """Tiny per-URL JSON cache backed by one file per request."""

    def __init__(self, cache_dir: Path) -> None:
        """Create the cache rooted at ``cache_dir`` (created if missing)."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, url: str) -> Path:
        return self.cache_dir / f"{_sanitize(url)}.json"

    def get(self, url: str) -> dict[str, Any] | None:
        """Return the cached payload for ``url`` or ``None`` if there's no hit."""
        path = self._path_for(url)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("cache read failed for %s: %s — ignoring", url, exc)
            return None
        if not isinstance(data, dict):
            logger.warning("cache file %s did not contain a JSON object — ignoring", path)
            return None
        return data

    def set(self, url: str, payload: dict[str, Any]) -> None:
        """Persist ``payload`` to disk for ``url``."""
        path = self._path_for(url)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        tmp.replace(path)

    def clear(self) -> None:
        """Delete every cached entry (used by ``--rebuild``)."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
