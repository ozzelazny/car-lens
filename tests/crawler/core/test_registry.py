"""Tests for the parser registry."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from car_lense_engine.crawler.core.registry import ParserRegistry
from car_lense_engine.crawler.parsers.base import ParseResult


@dataclass
class _StubParser:
    source: str
    calls: list[str] = field(default_factory=list)

    def parse(
        self,
        *,
        html: str,
        url: str,
        kind: str,
        hints: dict[str, str | int | None],
    ) -> ParseResult:
        self.calls.append(url)
        return ParseResult()


def test_register_and_get() -> None:
    reg = ParserRegistry()
    parser = _StubParser(source="cars_com")
    reg.register(parser)
    assert reg.get("cars_com") is parser


def test_get_unknown_source_raises_keyerror_with_clear_message() -> None:
    reg = ParserRegistry()
    reg.register(_StubParser(source="cars_com"))
    with pytest.raises(KeyError) as excinfo:
        reg.get("autotrader")
    msg = str(excinfo.value)
    assert "autotrader" in msg
    assert "cars_com" in msg


def test_sources_lists_registered() -> None:
    reg = ParserRegistry()
    reg.register(_StubParser(source="cars_com"))
    reg.register(_StubParser(source="autotrader"))
    assert reg.sources() == ["autotrader", "cars_com"]


def test_register_duplicate_source_raises() -> None:
    reg = ParserRegistry()
    reg.register(_StubParser(source="cars_com"))
    with pytest.raises(ValueError, match="cars_com"):
        reg.register(_StubParser(source="cars_com"))


def test_has_returns_true_only_for_registered_sources() -> None:
    reg = ParserRegistry()
    assert reg.has("cars_com") is False
    reg.register(_StubParser(source="cars_com"))
    assert reg.has("cars_com") is True
    assert reg.has("missing") is False


def test_unregister_removes_parser() -> None:
    reg = ParserRegistry()
    reg.register(_StubParser(source="cars_com"))
    reg.unregister("cars_com")
    assert reg.has("cars_com") is False
    # No-op on unknown sources.
    reg.unregister("never_registered")
