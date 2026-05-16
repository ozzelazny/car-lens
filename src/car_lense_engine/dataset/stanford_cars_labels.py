"""Stanford Cars class-name normalizer.

Stanford Cars' 196 classes are encoded as English strings like
``"Acura RL Sedan 2012"`` or ``"Chevrolet Silverado 1500 Hybrid Crew Cab 2012"``.
This module parses each string into structured ``(year, make, model,
body_style)`` so we can insert clean rows into the ``listings`` table and
cross-reference the NHTSA catalog.

Parsing strategy:

1. Strip the trailing 4-digit year token.
2. Match a make at the start using the longest-prefix from the catalog's
   known-make set (handles multi-word makes like "Aston Martin" or
   "Mercedes-Benz" — though hyphenated names are a single whitespace token
   anyway).
3. From the right, pull off a body-style suffix if the trailing token (or
   trailing two tokens, for ``Crew Cab``-style cab suffixes) is in a fixed
   dictionary.
4. Everything between the make and the body style (or end-of-string if no
   body style was matched) is the model.

The body-style dictionary is intentionally small. Stanford only uses ~12
distinct body styles; we list them here rather than reading them from any
external taxonomy so the normalizer is self-contained and deterministic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Single-token body styles. Match against the last whitespace-delimited
# token of the post-year prefix.
_BODY_STYLES: frozenset[str] = frozenset(
    {
        "Sedan",
        "Coupe",
        "Hatchback",
        "Wagon",
        "Convertible",
        "SUV",
        "Van",
        "Minivan",
        "Truck",
        "Pickup",
        "Roadster",
        "Cabriolet",
        "GranCoupe",
        "GranTurismo",
    }
)


# Two-token body-style suffixes. Match against the last two tokens of the
# post-year prefix. Order doesn't matter — we check membership.
_BODY_STYLE_SUFFIXES: frozenset[tuple[str, str]] = frozenset(
    {
        ("Crew", "Cab"),
        ("Extended", "Cab"),
        ("Regular", "Cab"),
        ("Super", "Cab"),
        ("Quad", "Cab"),
        ("Club", "Cab"),
        ("Standard", "Cab"),
    }
)


_TRAILING_YEAR_RE = re.compile(r"\s+(\d{4})$")


class StanfordCarsParseError(ValueError):
    """Raised when a Stanford Cars class string cannot be parsed."""


@dataclass(frozen=True)
class StanfordCarsLabel:
    """Structured (year, make, model, body_style) parsed from a class string."""

    year: int
    make: str
    model: str
    body_style: str | None
    raw_class: str


def parse_class(raw: str, known_makes: set[str]) -> StanfordCarsLabel:
    """Parse a Stanford Cars class string into a :class:`StanfordCarsLabel`.

    Examples (with ``known_makes`` populated from the NHTSA catalog)::

        parse_class("Acura RL Sedan 2012", makes) ==
            StanfordCarsLabel(2012, "Acura", "RL", "Sedan", ...)

        parse_class("AM General Hummer SUV 2000", makes) ==
            StanfordCarsLabel(2000, "AM General", "Hummer", "SUV", ...)

        parse_class("Hyundai Sonata 2007", makes) ==
            StanfordCarsLabel(2007, "Hyundai", "Sonata", None, ...)

    Raises :class:`StanfordCarsParseError` when:

    * the input is empty / whitespace only,
    * no trailing 4-digit year is present, or
    * the make can't be matched and no fallback prefix is recoverable.

    The function never raises for "missing body style" — that's the common
    case for older listings (e.g. "Hyundai Sonata 2007").
    """
    if raw is None or not str(raw).strip():
        raise StanfordCarsParseError("empty class string")

    s = str(raw).strip()

    year_match = _TRAILING_YEAR_RE.search(s)
    if year_match is None:
        raise StanfordCarsParseError(f"no trailing year in class string: {raw!r}")
    year = int(year_match.group(1))
    prefix = s[: year_match.start()].strip()
    if not prefix:
        raise StanfordCarsParseError(f"no make/model before year in: {raw!r}")

    tokens = prefix.split()

    # ---- Make: longest-prefix match against known_makes.
    make = _match_make(tokens, known_makes)
    if make is None:
        # Fallback: first token is the make. Log a warning so we can audit
        # parser drift later, but don't fail — the caller may still want to
        # surface the row.
        logger.warning(
            "stanford_cars: make not in catalog (raw=%r); falling back to first token",
            raw,
        )
        make = tokens[0]
        make_token_count = 1
    else:
        make_token_count = len(make.split())

    remaining = tokens[make_token_count:]
    if not remaining:
        raise StanfordCarsParseError(f"no model after make in: {raw!r}")

    # ---- Body style: try two-token suffix first, then single-token.
    body_style: str | None = None
    if len(remaining) >= 2:
        last_two = (remaining[-2], remaining[-1])
        if last_two in _BODY_STYLE_SUFFIXES:
            body_style = f"{last_two[0]} {last_two[1]}"
            remaining = remaining[:-2]
    if body_style is None and remaining and remaining[-1] in _BODY_STYLES:
        body_style = remaining[-1]
        remaining = remaining[:-1]

    if not remaining:
        raise StanfordCarsParseError(f"no model tokens left after stripping body style in: {raw!r}")

    model = " ".join(remaining)

    return StanfordCarsLabel(
        year=year,
        make=make,
        model=model,
        body_style=body_style,
        raw_class=raw,
    )


def _match_make(tokens: list[str], known_makes: set[str]) -> str | None:
    """Return the longest known-make prefix of ``tokens``, or ``None``.

    Tries 3-token, 2-token, 1-token prefixes in that order. Comparison is
    case-insensitive (the catalog is Title Case but we mirror Stanford's
    original casing in the returned string).
    """
    # Normalize known_makes to a lowercase lookup so case mismatches between
    # Stanford ("AM General") and the catalog ("Am General") don't drop a
    # match. The returned make preserves Stanford's casing as-is.
    lookup = {m.lower() for m in known_makes}

    for n in (3, 2, 1):
        if len(tokens) < n:
            continue
        candidate = " ".join(tokens[:n])
        if candidate.lower() in lookup:
            return candidate
    return None
