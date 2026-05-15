"""Per-site listing parsers.

Phase 2 ships one concrete parser per source, plus a shared
:mod:`car_lense_engine.crawler.parsers.common` utilities module. Each parser
is registered against
:class:`~car_lense_engine.crawler.core.registry.ParserRegistry` and dispatched
by ``QueueItem.source`` in the worker loop.
"""

from __future__ import annotations

from .autotrader import AutoTraderParser
from .base import DiscoveredUrl, ParsedListing, Parser, ParseResult
from .cars_com import CarsComParser
from .common import (
    extract_jsonld,
    find_jsonld_by_type,
    find_links,
    normalize_url,
    parse_int_safe,
    parse_year_safe,
    sha256_text,
)
from .craigslist import CraigslistParser

__all__ = [
    "AutoTraderParser",
    "CarsComParser",
    "CraigslistParser",
    "DiscoveredUrl",
    "ParseResult",
    "ParsedListing",
    "Parser",
    "extract_jsonld",
    "find_jsonld_by_type",
    "find_links",
    "normalize_url",
    "parse_int_safe",
    "parse_year_safe",
    "sha256_text",
]
