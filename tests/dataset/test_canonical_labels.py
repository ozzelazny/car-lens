"""Tests for the Phase 4.5 canonical-label normalizer.

Covers the alias map (CompCars typos, aliases, brand-specific casing),
the Title Case fallback for unknown makes, edge cases (None, empty,
whitespace), and idempotency (normalizing the canonical form returns
the same string).
"""

from __future__ import annotations

import pytest

from car_lense_engine.dataset.canonical_labels import (
    GENERATION_ANCHOR_YEAR,
    GENERATION_BUCKET_YEARS,
    generation_label,
    normalize_make,
    normalize_model,
    year_to_generation,
)

# --------------------------------------------------------------- make: alias map


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # CompCars-style typos and aliases.
        ("Benz", "Mercedes-Benz"),
        ("mercedes benz", "Mercedes-Benz"),
        ("Mercedes-Benz", "Mercedes-Benz"),
        ("BWM", "BMW"),  # typo
        ("BMW", "BMW"),
        ("bmw", "BMW"),
        ("Chevy", "Chevrolet"),
        ("Chevrolet", "Chevrolet"),
        ("CHEVROLET", "Chevrolet"),
        ("MAZDA", "Mazda"),
        ("Buck", "Buick"),  # CompCars typo
        ("Buick", "Buick"),
        ("Chrey", "Chery"),  # CompCars typo
        ("Lamorghini", "Lamborghini"),  # CompCars typo
        ("Lamborghini", "Lamborghini"),
        ("land-rover", "Land Rover"),
        ("Land Rover", "Land Rover"),
        ("landrover", "Land Rover"),
        # Brand-specific casing.
        ("FIAT", "FIAT"),
        ("Fiat", "FIAT"),
        ("fiat", "FIAT"),
        ("MINI", "MINI"),
        ("Mini", "MINI"),
        ("smart", "smart"),
        ("SMART", "smart"),
        ("VW", "Volkswagen"),
        ("vw", "Volkswagen"),
        ("Rolls-Royce", "Rolls-Royce"),
        ("rolls royce", "Rolls-Royce"),
        ("alfa romeo", "Alfa Romeo"),
        ("aston martin", "Aston Martin"),
        ("ds", "DS"),
        ("byd", "BYD"),
        ("gmc", "GMC"),
        ("mg", "MG"),
        ("abt", "ABT"),
    ],
)
def test_normalize_make_alias_map(raw: str, expected: str) -> None:
    """Every alias-map entry produces its canonical form, regardless of input case."""
    assert normalize_make(raw) == expected


# --------------------------------------------------------------- make: title case fallback


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Long-tail makes not in the alias map: Title Case fallback.
        ("acura", "Acura"),
        ("Acura", "Acura"),
        ("ACURA", "Acura"),
        ("geely", "Geely"),
        ("honda", "Honda"),
        ("toyota", "Toyota"),
        ("ford", "Ford"),
        ("nissan", "Nissan"),
        ("hyundai", "Hyundai"),
    ],
)
def test_normalize_make_title_case_fallback(raw: str, expected: str) -> None:
    """Makes not in the alias map fall back to ``str.title()``."""
    assert normalize_make(raw) == expected


# --------------------------------------------------------------- make: edge cases


def test_normalize_make_none() -> None:
    assert normalize_make(None) is None


def test_normalize_make_empty_string() -> None:
    assert normalize_make("") is None


def test_normalize_make_whitespace_only() -> None:
    assert normalize_make("   ") is None
    assert normalize_make("\t\n  ") is None


def test_normalize_make_strips_whitespace() -> None:
    """Leading / trailing whitespace is stripped before lookup."""
    assert normalize_make("  BMW  ") == "BMW"
    assert normalize_make("\tacura\n") == "Acura"
    assert normalize_make(" chevy ") == "Chevrolet"


# --------------------------------------------------------------- make: idempotency


@pytest.mark.parametrize(
    "canonical",
    [
        "BMW",
        "Mercedes-Benz",
        "Chevrolet",
        "FIAT",
        "MINI",
        "smart",
        "Volkswagen",
        "Land Rover",
        "Rolls-Royce",
        "Alfa Romeo",
        "Aston Martin",
        "Acura",
        "Honda",
        "Toyota",
        "Geely",
    ],
)
def test_normalize_make_idempotent(canonical: str) -> None:
    """Re-normalizing a canonical form returns the same canonical form."""
    assert normalize_make(canonical) == canonical
    # Doubly idempotent: f(f(x)) == f(x).
    assert normalize_make(normalize_make(canonical)) == canonical


# --------------------------------------------------------------- model normalization


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Lowercase Stanford-style models.
        ("rl sedan", "Rl Sedan"),
        ("model s", "Model S"),
        ("civic", "Civic"),
        # All-caps CompCars-style models.
        ("ABT A3", "Abt A3"),
        ("MUSTANG", "Mustang"),
        # Already-canonical Title Case passes through.
        ("Civic", "Civic"),
        ("Model S", "Model S"),
    ],
)
def test_normalize_model_case(raw: str, expected: str) -> None:
    """``normalize_model`` is currently a case-only normalizer."""
    assert normalize_model(raw) == expected


def test_normalize_model_none() -> None:
    assert normalize_model(None) is None


def test_normalize_model_empty_string() -> None:
    assert normalize_model("") is None


def test_normalize_model_whitespace_only() -> None:
    assert normalize_model("   ") is None


def test_normalize_model_strips_whitespace() -> None:
    assert normalize_model("  civic  ") == "Civic"
    assert normalize_model("\tmodel s\n") == "Model S"


def test_normalize_model_preserves_body_style_suffix() -> None:
    """Phase 4.5 deliberately does not strip Stanford-style body-style suffixes."""
    assert normalize_model("rl sedan") == "Rl Sedan"
    assert normalize_model("tt rs coupe") == "Tt Rs Coupe"


def test_normalize_model_preserves_compcars_make_prefix() -> None:
    """Phase 4.5 deliberately does not strip CompCars-style make prefixes."""
    assert normalize_model("ABT A3") == "Abt A3"


# --------------------------------------------------------------- generation buckets


def test_generation_constants() -> None:
    """Bucketing uses a 4-year width anchored at 1980 (Phase 4.6)."""
    assert GENERATION_BUCKET_YEARS == 4
    assert GENERATION_ANCHOR_YEAR == 1980


@pytest.mark.parametrize(
    ("year", "expected_start"),
    [
        # Anchor boundary.
        (1980, 1980),
        (1981, 1980),
        (1982, 1980),
        (1983, 1980),
        # Next bucket starts at 1984.
        (1984, 1984),
        (1987, 1984),
        # Modern years that drive the Phase 5.2 confusion pairs --
        # BYD Qin 2012 and 2014 must land in the same bucket.
        (2010, 2008),
        (2011, 2008),
        (2012, 2012),
        (2013, 2012),
        (2014, 2012),
        (2015, 2012),
        (2016, 2016),
        # Bucket containing 2020-2023.
        (2020, 2020),
        (2023, 2020),
        (2024, 2024),
        # Future years still bucket.
        (2025, 2024),
        (2027, 2024),
        (2028, 2028),
    ],
)
def test_year_to_generation_modern(year: int, expected_start: int) -> None:
    """4-year buckets anchored at 1980 produce the expected start year."""
    assert year_to_generation(year) == expected_start


@pytest.mark.parametrize(
    ("year", "expected_start"),
    [
        # Pre-anchor years still bucket via floor division.
        (1979, 1976),  # (1979-1980)//4 == -1 -> anchor + -4 = 1976
        (1976, 1976),
        (1975, 1972),
        (1972, 1972),
        (1971, 1968),
    ],
)
def test_year_to_generation_pre_anchor(year: int, expected_start: int) -> None:
    """Pre-anchor years still bucket downward (no special-casing)."""
    assert year_to_generation(year) == expected_start


def test_year_to_generation_none_is_none() -> None:
    """``None`` propagates through unchanged."""
    assert year_to_generation(None) is None


def test_year_to_generation_idempotent() -> None:
    """Re-bucketing a bucket start year returns the same start year."""
    for start in (1980, 1984, 2012, 2016, 2020, 2024):
        assert year_to_generation(start) == start
        assert year_to_generation(year_to_generation(start)) == start


def test_year_to_generation_pair_collision() -> None:
    """The Phase 5.2 top-confusion pairs collapse to one class each."""
    # BYD Qin 2012 vs 2014.
    assert year_to_generation(2012) == year_to_generation(2014)
    # Kia K2 sedan 2012 vs 2015.
    assert year_to_generation(2012) == year_to_generation(2015)
    # Zotye V10 2011 vs 2012 -- these DO end up in different buckets
    # because 2011 is in 2008-2011 and 2012 is in 2012-2015. That's an
    # expected limitation of the fixed-grid bucketing; the design note
    # in the migration calls this out.
    assert year_to_generation(2011) != year_to_generation(2012)


@pytest.mark.parametrize(
    ("year", "expected_label"),
    [
        (1980, "1980-1983"),
        (1983, "1980-1983"),
        (1984, "1984-1987"),
        (2012, "2012-2015"),
        (2014, "2012-2015"),
        (2015, "2012-2015"),
        (2016, "2016-2019"),
        (2020, "2020-2023"),
        (2024, "2024-2027"),
    ],
)
def test_generation_label_modern(year: int, expected_label: str) -> None:
    """The human-readable label is ``start-end`` with end = start + 3."""
    assert generation_label(year) == expected_label


def test_generation_label_none_is_none() -> None:
    assert generation_label(None) is None


def test_generation_label_pre_anchor() -> None:
    """Pre-anchor labels still render as ``start-end``."""
    assert generation_label(1976) == "1976-1979"
    assert generation_label(1979) == "1976-1979"
