"""Tests for the Stanford Cars class-name normalizer."""

from __future__ import annotations

import logging

import pytest

from car_lense_engine.dataset.stanford_cars_labels import (
    StanfordCarsLabel,
    StanfordCarsParseError,
    parse_class,
)

# Catalog-style known makes. Capitalization here mirrors what the NHTSA
# catalog produces (Title Case for compound names).
KNOWN_MAKES: set[str] = {
    "Acura",
    "AM General",
    "Alfa Romeo",
    "Aston Martin",
    "Audi",
    "BMW",
    "Buick",
    "Cadillac",
    "Chevrolet",
    "Chrysler",
    "Dodge",
    "Eagle",
    "Fiat",
    "Ford",
    "Geo",
    "GMC",
    "Honda",
    "Hyundai",
    "Infiniti",
    "Isuzu",
    "Jaguar",
    "Jeep",
    "Lamborghini",
    "Land Rover",
    "Lincoln",
    "Mazda",
    "Mercedes-Benz",
    "MINI",
    "Mitsubishi",
    "Nissan",
    "Plymouth",
    "Porsche",
    "Ram",
    "Rolls-Royce",
    "Scion",
    "Smart",
    "Spyker",
    "Suzuki",
    "Tesla",
    "Toyota",
    "Volkswagen",
    "Volvo",
}


def test_parse_simple_sedan() -> None:
    label = parse_class("Acura RL Sedan 2012", KNOWN_MAKES)
    assert label == StanfordCarsLabel(
        year=2012,
        make="Acura",
        model="RL",
        body_style="Sedan",
        raw_class="Acura RL Sedan 2012",
    )


def test_parse_two_word_make() -> None:
    label = parse_class("AM General Hummer SUV 2000", KNOWN_MAKES)
    assert label.year == 2000
    assert label.make == "AM General"
    assert label.model == "Hummer"
    assert label.body_style == "SUV"


def test_parse_hyphenated_make() -> None:
    label = parse_class("Mercedes-Benz Sprinter Van 2012", KNOWN_MAKES)
    assert label.year == 2012
    assert label.make == "Mercedes-Benz"
    assert label.model == "Sprinter"
    assert label.body_style == "Van"


def test_parse_multi_word_model_with_two_word_body_style() -> None:
    label = parse_class("Chevrolet Silverado 1500 Hybrid Crew Cab 2012", KNOWN_MAKES)
    assert label.year == 2012
    assert label.make == "Chevrolet"
    assert label.model == "Silverado 1500 Hybrid"
    assert label.body_style == "Crew Cab"


def test_parse_no_body_style() -> None:
    label = parse_class("Hyundai Sonata 2007", KNOWN_MAKES)
    assert label.year == 2007
    assert label.make == "Hyundai"
    assert label.model == "Sonata"
    assert label.body_style is None


def test_parse_tesla_multi_word_model() -> None:
    label = parse_class("Tesla Model S Sedan 2012", KNOWN_MAKES)
    assert label.year == 2012
    assert label.make == "Tesla"
    assert label.model == "Model S"
    assert label.body_style == "Sedan"


def test_parse_volkswagen_hatchback() -> None:
    label = parse_class("Volkswagen Golf Hatchback 1991", KNOWN_MAKES)
    assert label.year == 1991
    assert label.make == "Volkswagen"
    assert label.model == "Golf"
    assert label.body_style == "Hatchback"


def test_parse_empty_string_raises() -> None:
    with pytest.raises(StanfordCarsParseError):
        parse_class("", KNOWN_MAKES)


def test_parse_whitespace_only_raises() -> None:
    with pytest.raises(StanfordCarsParseError):
        parse_class("   ", KNOWN_MAKES)


def test_parse_missing_year_raises() -> None:
    with pytest.raises(StanfordCarsParseError):
        parse_class("Acura RL Sedan", KNOWN_MAKES)


def test_parse_year_only_raises() -> None:
    # The match consumes "_____ 2012" — but the regex requires whitespace
    # before the year, so a bare "2012" has no preceding char. It still has
    # no leading prefix, which we reject explicitly.
    with pytest.raises(StanfordCarsParseError):
        parse_class("Acura 2012", KNOWN_MAKES)


def test_unknown_make_falls_back_to_first_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # "Banana" isn't in KNOWN_MAKES; the parser falls back to "Banana" as
    # the make and "Bread" as the model, with a logged warning.
    with caplog.at_level(logging.WARNING):
        label = parse_class("Banana Bread Sedan 2012", KNOWN_MAKES)
    assert label.make == "Banana"
    assert label.model == "Bread"
    assert label.body_style == "Sedan"
    assert label.year == 2012
    assert any("make not in catalog" in r.message for r in caplog.records)


def test_case_insensitive_make_match() -> None:
    # Stanford-style casing ("AM General") matches catalog Title-Case
    # ("Am General") and vice versa.
    makes = {"Am General"}
    label = parse_class("AM General Hummer SUV 2000", makes)
    assert label.make == "AM General"


def test_raw_class_is_preserved() -> None:
    raw = "Acura RL Sedan 2012"
    label = parse_class(raw, KNOWN_MAKES)
    assert label.raw_class == raw
