"""Tests for the Phase 4.5 canonical-label normalizer.

Covers the alias map (CompCars typos, aliases, brand-specific casing),
the Title Case fallback for unknown makes, edge cases (None, empty,
whitespace), and idempotency (normalizing the canonical form returns
the same string).
"""

from __future__ import annotations

import pytest

from car_lense_engine.dataset.canonical_labels import (
    normalize_make,
    normalize_model,
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
