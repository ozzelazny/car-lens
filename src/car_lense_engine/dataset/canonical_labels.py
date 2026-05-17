"""Hand-curated make/model normalizer for the Phase 4.5 unified label schema.

Cross-source training requires a single canonical ``class_id`` per
``(year, make, model)``. Different sources spell the same logical make
differently: ``"Chevrolet"`` (Title Case, crawled), ``"chevrolet"``
(lowercase, Stanford Cars HF mirror), ``"Chevy"`` (alias, CompCars),
``"BWM"`` (typo, CompCars), ``"MAZDA"`` (all caps, CompCars). This
module maps any of those raw strings to a single canonical form
(``"Chevrolet"``, ``"BMW"``, ``"Mazda"``).

Design notes
------------

* **Hand-curated alias map first.** ``_MAKE_ALIAS_MAP`` is the
  source-of-truth for makes whose Title Case is wrong (``"BMW"``,
  ``"FIAT"``, ``"MINI"``), for cross-source typos (``"BWM"``,
  ``"Lamorghini"``, ``"Buck"``), and for aliases (``"Chevy"`` ->
  ``"Chevrolet"``, ``"Benz"`` -> ``"Mercedes-Benz"``).
* **Title Case fallback.** Anything not in the alias map gets Python's
  ``str.title()``, which is correct for the long tail (Acura, Geely,
  Honda, ...). Compound names with internal capitals (``McLaren``)
  aren't perfect after Title Case (``"Mclaren"``) but are stable across
  sources; pre-existing TODO.md note tracks the catalog Title Case
  follow-up.
* **No fuzzy matching.** We do NOT try to match against the NHTSA
  catalog or use string-similarity heuristics. The alias map covers the
  inventory of cross-source patterns we've observed; new sources can
  add entries as they're ingested.
* **Models: case-only normalization.** ``normalize_model`` only does
  Title Case for v1. Stanford's body-style suffix (``"rl sedan"``) and
  CompCars' make-prefix-baked-in models (``"ABT A3"``) are NOT cleaned
  up here -- both result in distinct ``(make, model)`` tuples by
  design.
"""

from __future__ import annotations

# Hand-curated alias map. Drives Phase 4.5 unified labels.
# Keys are LOWERCASED + stripped raw strings; values are the canonical form
# (case as it should appear on the brand or per industry convention).
_MAKE_ALIAS_MAP: dict[str, str] = {
    # CompCars-style typos / aliases (observed in the live dataset).
    "benz": "Mercedes-Benz",
    "mercedes benz": "Mercedes-Benz",
    "mercedes-benz": "Mercedes-Benz",
    "bwm": "BMW",
    "bmw": "BMW",
    "chevy": "Chevrolet",
    "chevrolet": "Chevrolet",
    "kia": "Kia",
    "mazda": "Mazda",
    "buck": "Buick",
    "buick": "Buick",
    "chrey": "Chery",
    "lamorghini": "Lamborghini",
    "lamborghini": "Lamborghini",
    "land-rover": "Land Rover",
    "land rover": "Land Rover",
    "landrover": "Land Rover",
    "tesla": "Tesla",
    "saab": "Saab",
    "fiat": "FIAT",  # FIAT canonical is uppercase per the brand
    "vw": "Volkswagen",
    "vauxhall": "Vauxhall",
    "mini": "MINI",  # MINI canonical is uppercase
    "smart": "smart",  # smart is lowercase per the brand
    "rolls-royce": "Rolls-Royce",
    "rolls royce": "Rolls-Royce",
    "alfa romeo": "Alfa Romeo",
    "aston martin": "Aston Martin",
    "am general": "AM General",
    "ds": "DS",
    "byd": "BYD",
    "gmc": "GMC",
    "mg": "MG",
    "abt": "ABT",
    # Stanford-style lowercase is already covered by the lower()-lookup;
    # we only add an explicit entry when the Title Case fallback would
    # be wrong for that brand.
}


def normalize_make(raw: str | None) -> str | None:
    """Map a raw ``make`` string to its canonical form.

    Strategy:

    1. ``None`` / empty / whitespace-only input -> ``None``.
    2. Lowercase + strip the input, look up in :data:`_MAKE_ALIAS_MAP`;
       if found, return the mapped canonical form.
    3. Otherwise: Title Case the stripped input. ``"acura"`` -> ``"Acura"``;
       ``"geely"`` -> ``"Geely"``. This handles the long tail of makes
       that aren't in the alias map.

    The function is **idempotent** by construction: every canonical form
    in :data:`_MAKE_ALIAS_MAP` is itself a value AND, after lowercasing,
    a key (or its Title Case matches the canonical case). E.g.
    ``normalize_make("BMW")`` -> ``"BMW"`` (lowercase key ``"bmw"``
    maps to ``"BMW"``); ``normalize_make("Acura")`` -> ``"Acura"``
    (no alias, Title Case of ``"acura"`` is ``"Acura"``).
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    key = stripped.lower()
    mapped = _MAKE_ALIAS_MAP.get(key)
    if mapped is not None:
        return mapped
    return stripped.title()


def normalize_model(raw: str | None) -> str | None:
    """Normalize a ``model`` string.

    v1: case-only normalization (``str.lower().title()``). Stanford
    Cars emits models like ``"rl sedan"`` (lowercase, with body-style
    suffix); CompCars emits ``"ABT A3"`` (uppercase, with make prefix
    baked in). We do NOT strip the body-style suffix or the make
    prefix -- different ``(make, model)`` strings produce distinct
    classes, which is intentional for v1. The catalog-level alignment
    is out of scope for this phase.

    ``None`` / empty / whitespace-only -> ``None``.
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    return stripped.title()


__all__ = [
    "normalize_make",
    "normalize_model",
]
