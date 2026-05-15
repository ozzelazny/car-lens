"""Catalog module — canonical (make, model, generation) data.

Public API:

* :func:`build_catalog` — orchestrate a full NHTSA vPIC pull.
* :func:`write_catalog` — serialize a :class:`Catalog` to JSON.
* :class:`NHTSAClient` — thin httpx wrapper with caching, retry, rate limit.
* :class:`JSONFileCache` — per-URL JSON file cache.
* Schema models: :class:`Catalog`, :class:`Make`, :class:`Model`, :class:`Meta`.
"""

from __future__ import annotations

from .build_catalog import build_catalog, write_catalog
from .cache import JSONFileCache
from .nhtsa_client import NHTSAClient, NHTSAError, TokenBucket
from .schema import Catalog, Make, Meta, Model

__all__ = [
    "Catalog",
    "JSONFileCache",
    "Make",
    "Meta",
    "Model",
    "NHTSAClient",
    "NHTSAError",
    "TokenBucket",
    "build_catalog",
    "write_catalog",
]
