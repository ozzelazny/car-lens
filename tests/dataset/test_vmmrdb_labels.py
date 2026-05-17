"""Tests for the VMMRdb class-name normalizer."""

from __future__ import annotations

import pytest

from car_lense_engine.dataset.vmmrdb_labels import (
    VmmrdbLabel,
    VmmrdbParseError,
    parse_class,
)

# --------------------------------------------------------- year-suffix format


def test_parse_simple_year_suffix() -> None:
    label = parse_class("honda_civic_2005")
    assert label == VmmrdbLabel(
        year=2005,
        make="honda",
        model="civic",
        raw_class="honda_civic_2005",
    )


def test_parse_toyota_camry() -> None:
    label = parse_class("toyota_camry_2007")
    assert label.year == 2007
    assert label.make == "toyota"
    assert label.model == "camry"


def test_parse_hyphenated_model() -> None:
    """``f-150`` stays a single underscore-free token in the model field."""
    label = parse_class("ford_f-150_2010")
    assert label.year == 2010
    assert label.make == "ford"
    assert label.model == "f-150"


def test_parse_multi_word_model_with_year() -> None:
    """Model carries embedded underscores when the original name has multiple words."""
    label = parse_class("chevrolet_silverado_1500_2012")
    assert label.year == 2012
    assert label.make == "chevrolet"
    assert label.model == "silverado_1500"


def test_parse_three_word_model_with_year() -> None:
    label = parse_class("ford_super_duty_f350_1999")
    assert label.year == 1999
    assert label.make == "ford"
    assert label.model == "super_duty_f350"


# --------------------------------------------------------- no-year format
# Matches the ``venetis/VMMRdb_make_model_*`` HF mirror.


def test_parse_no_year_make_model() -> None:
    label = parse_class("acura_cl")
    assert label.year is None
    assert label.make == "acura"
    assert label.model == "cl"


def test_parse_no_year_multi_word_make() -> None:
    """The venetis mirror joins multi-word makes with a space, not underscore."""
    label = parse_class("mercedes benz_s550")
    assert label.year is None
    assert label.make == "mercedes benz"
    assert label.model == "s550"


def test_parse_no_year_multi_word_model() -> None:
    """Models can also contain spaces (e.g. ``bel air``) on the venetis mirror."""
    label = parse_class("chevrolet_bel air")
    assert label.year is None
    assert label.make == "chevrolet"
    assert label.model == "bel air"


def test_parse_no_year_hyphenated_model() -> None:
    label = parse_class("honda_cr-v")
    assert label.year is None
    assert label.make == "honda"
    assert label.model == "cr-v"


# --------------------------------------------------------- edge cases


def test_parse_empty_string_raises() -> None:
    with pytest.raises(VmmrdbParseError):
        parse_class("")


def test_parse_whitespace_only_raises() -> None:
    with pytest.raises(VmmrdbParseError):
        parse_class("   ")


def test_parse_no_underscore_raises() -> None:
    """Single token with no underscore can't yield a (make, model) pair."""
    with pytest.raises(VmmrdbParseError):
        parse_class("acura")


def test_parse_year_only_raises() -> None:
    """A trailing year strips off, but the remainder must still contain ``_``."""
    with pytest.raises(VmmrdbParseError):
        parse_class("honda_2005")


def test_parse_empty_model_raises() -> None:
    """Underscore at the end (no model token) is rejected."""
    with pytest.raises(VmmrdbParseError):
        parse_class("honda_")


def test_parse_empty_make_raises() -> None:
    """Leading underscore (empty make) is rejected."""
    with pytest.raises(VmmrdbParseError):
        parse_class("_civic")


def test_parse_preserves_casing() -> None:
    """We don't normalize case here — Phase 4.5 will do that across sources."""
    label = parse_class("Honda_Civic_2005")
    assert label.make == "Honda"
    assert label.model == "Civic"


def test_parse_strips_surrounding_whitespace() -> None:
    label = parse_class("  honda_civic_2005  ")
    assert label.make == "honda"
    assert label.model == "civic"
    assert label.year == 2005


def test_parse_raw_class_is_preserved() -> None:
    raw = "honda_civic_2005"
    label = parse_class(raw)
    assert label.raw_class == raw


def test_parse_3_digit_trailing_token_is_not_year() -> None:
    """A 3-digit trailing token is NOT a year — treated as part of the model."""
    label = parse_class("ford_f-150_999")
    # Trailing year regex requires exactly 4 digits, so 999 stays in the model.
    assert label.year is None
    assert label.make == "ford"
    assert label.model == "f-150_999"


def test_parse_5_digit_trailing_token_is_not_year() -> None:
    """A 5-digit trailing token is NOT a year — treated as part of the model."""
    label = parse_class("ford_truck_99999")
    assert label.year is None
    assert label.make == "ford"
    # The regex requires _NNNN$ — a 5-digit trailing token doesn't match.
    assert label.model == "truck_99999"


def test_parse_none_raises() -> None:
    with pytest.raises(VmmrdbParseError):
        parse_class(None)  # type: ignore[arg-type]
