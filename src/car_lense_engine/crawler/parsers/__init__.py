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
from .bat import BringATrailerParser
from .cars_com import CarsComParser
from .carsandbids import CarsAndBidsParser
from .common import (
    extract_jsonld,
    find_jsonld_by_type,
    find_links,
    find_next_page,
    is_next_link,
    normalize_url,
    parse_int_safe,
    parse_year_safe,
    sha256_text,
)
from .craigslist import CraigslistParser
from .hemmings import HemmingsParser

__all__ = [
    "AutoTraderParser",
    "BringATrailerParser",
    "CarsAndBidsParser",
    "CarsComParser",
    "CraigslistParser",
    "DiscoveredUrl",
    "HemmingsParser",
    "ParseResult",
    "ParsedListing",
    "Parser",
    "extract_jsonld",
    "find_jsonld_by_type",
    "find_links",
    "find_next_page",
    "is_next_link",
    "normalize_url",
    "parse_int_safe",
    "parse_year_safe",
    "sha256_text",
]
