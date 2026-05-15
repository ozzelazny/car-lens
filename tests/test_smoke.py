"""Smoke test — verifies the package imports and reports its version."""

import car_lense_engine


def test_version() -> None:
    assert car_lense_engine.__version__ == "0.0.1"
