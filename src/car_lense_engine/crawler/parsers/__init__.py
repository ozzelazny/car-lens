"""Per-site listing parsers.

Phase 1 ships only the protocol and result dataclasses; Phase 2 will add one
concrete parser per source and register them against
:class:`~car_lense_engine.crawler.core.registry.ParserRegistry`.
"""

from __future__ import annotations

from .base import DiscoveredUrl, ParsedListing, Parser, ParseResult

__all__ = ["DiscoveredUrl", "ParseResult", "ParsedListing", "Parser"]
