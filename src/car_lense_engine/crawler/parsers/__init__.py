"""Per-site listing parsers.

Phase 2 ships one concrete parser per source, plus a shared
:mod:`car_lense_engine.crawler.parsers.common` utilities module. Each parser
is registered against
:class:`~car_lense_engine.crawler.core.registry.ParserRegistry` and dispatched
by ``QueueItem.source`` in the worker loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from ..core.registry import ParserRegistry


def register_all(registry: ParserRegistry) -> None:
    """Register every production parser into the given registry.

    Single source of truth for the production parser set — the CLI and the
    smoke harness both go through here so they can never drift.
    """
    registry.register(CarsComParser())
    registry.register(AutoTraderParser())
    registry.register(CraigslistParser())
    registry.register(BringATrailerParser())
    registry.register(HemmingsParser())
    registry.register(CarsAndBidsParser())


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
    "register_all",
    "sha256_text",
]
