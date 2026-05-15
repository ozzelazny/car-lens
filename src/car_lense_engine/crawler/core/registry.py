"""Parser registry — the only way per-site parsers reach the worker.

The crawler core never imports concrete parsers. Phase 2 packages will provide
a ``register_all(registry)`` helper that populates an instance of
:class:`ParserRegistry`; the CLI / a caller hands that populated registry to
:func:`~car_lense_engine.crawler.core.runner.run_crawler`.
"""

from __future__ import annotations

import logging

from car_lense_engine.crawler.parsers.base import Parser

logger = logging.getLogger(__name__)


class ParserRegistry:
    """A mapping from ``source`` string to :class:`Parser` instance."""

    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {}

    def register(self, parser: Parser) -> None:
        """Register ``parser`` under its declared ``source``.

        Raises :class:`ValueError` if a parser is already registered for that
        source — explicit overwrites should call :meth:`unregister` first.
        """
        source = parser.source
        if source in self._parsers:
            raise ValueError(
                f"parser for source {source!r} is already registered: "
                f"{type(self._parsers[source]).__name__}"
            )
        self._parsers[source] = parser
        logger.debug("registered parser for source=%s (%s)", source, type(parser).__name__)

    def unregister(self, source: str) -> None:
        """Remove the parser for ``source``; no-op if absent."""
        self._parsers.pop(source, None)

    def get(self, source: str) -> Parser:
        """Return the registered parser for ``source``.

        Raises :class:`KeyError` with the list of known sources in the message
        so callers can produce useful diagnostics for the queue.
        """
        try:
            return self._parsers[source]
        except KeyError as exc:
            known = sorted(self._parsers)
            raise KeyError(
                f"no parser registered for source {source!r}; known sources: {known}"
            ) from exc

    def has(self, source: str) -> bool:
        """Return ``True`` if a parser is registered for ``source``."""
        return source in self._parsers

    def sources(self) -> list[str]:
        """Return a sorted list of registered source identifiers."""
        return sorted(self._parsers)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._parsers)
