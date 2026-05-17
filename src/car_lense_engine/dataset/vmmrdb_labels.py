"""VMMRdb class-name normalizer.

VMMRdb (Vehicle Make and Model Recognition Database, Tafazzoli et al. 2017)
encodes classes as underscore-joined strings. Two formats appear in the wild:

* **Original GitHub release** — ``"<make>_<model>_<year>"`` where ``<year>`` is
  a trailing 4-digit token. Examples::

      honda_civic_2005
      toyota_camry_2007
      ford_f-150_2010
      chevrolet_silverado_1500_2012

  Make is the first underscore-delimited token; year is the last; everything
  in between is the model (which may itself contain underscores).

* **HuggingFace ``venetis/VMMRdb_make_model_*`` mirror** —
  ``"<make>_<model>"`` (no year). Make is the first token. Multi-word makes
  are joined with a *space* (``"mercedes benz_s550"``), multi-word models
  with a *space* or a dash (``"chevrolet_bel air"``, ``"honda_cr-v"``). The
  first underscore separates make from model.

The parser handles both:

1. Strip the trailing 4-digit year token if present (year-suffix format).
2. Split on the *first* underscore; left side is make, right side is model.
3. Preserve casing as-given. Replace internal spaces in make/model with
   underscores would be lossy; we keep them so callers see exactly what the
   dataset labelled the class.

We intentionally don't try to canonicalize ``"mercedes benz"`` →
``"Mercedes-Benz"`` here — that's a Phase 4.5 (unified label schema) concern.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Matches a trailing _NNNN at the very end of the class string.
_TRAILING_YEAR_RE = re.compile(r"_(\d{4})$")


class VmmrdbParseError(ValueError):
    """Raised when a VMMRdb class string cannot be parsed."""


@dataclass(frozen=True)
class VmmrdbLabel:
    """Structured (year, make, model) parsed from a VMMRdb class string.

    ``year`` is ``None`` for mirror formats that omit it (notably
    ``venetis/VMMRdb_make_model_*``). All other fields are required.
    """

    year: int | None
    make: str
    model: str
    raw_class: str


def parse_class(raw: str) -> VmmrdbLabel:
    """Parse a VMMRdb class string into a :class:`VmmrdbLabel`.

    Examples::

        parse_class("honda_civic_2005") ==
            VmmrdbLabel(year=2005, make="honda", model="civic", ...)

        parse_class("ford_f-150_2010") ==
            VmmrdbLabel(year=2010, make="ford", model="f-150", ...)

        parse_class("chevrolet_silverado_1500_2012") ==
            VmmrdbLabel(year=2012, make="chevrolet", model="silverado_1500", ...)

        parse_class("mercedes benz_s550") ==
            VmmrdbLabel(year=None, make="mercedes benz", model="s550", ...)

    Raises :class:`VmmrdbParseError` when:

    * the input is empty / whitespace only,
    * after stripping any trailing year, no underscore remains to separate
      make from model, or
    * the make or model would be empty after splitting.
    """
    if raw is None or not str(raw).strip():
        raise VmmrdbParseError("empty class string")

    s = str(raw).strip()

    year: int | None = None
    body = s
    year_match = _TRAILING_YEAR_RE.search(s)
    if year_match is not None:
        year = int(year_match.group(1))
        body = s[: year_match.start()]

    if "_" not in body:
        raise VmmrdbParseError(f"no underscore separating make from model in: {raw!r}")

    make, _, model = body.partition("_")
    make = make.strip()
    model = model.strip()

    if not make:
        raise VmmrdbParseError(f"empty make token in: {raw!r}")
    if not model:
        raise VmmrdbParseError(f"empty model token in: {raw!r}")

    return VmmrdbLabel(
        year=year,
        make=make,
        model=model,
        raw_class=raw,
    )
