"""Tests for the export CLI argument parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from car_lense_engine.export.cli import _parse_view_classifier


def test_parse_view_classifier_none() -> None:
    assert _parse_view_classifier("none") is None
    assert _parse_view_classifier("NONE") is None
    assert _parse_view_classifier(" none ") is None


def test_parse_view_classifier_path() -> None:
    out = _parse_view_classifier("models/view.pt")
    assert isinstance(out, Path)
    assert str(out) == "models/view.pt"


def test_cli_help_does_not_crash() -> None:
    """Running with --help exits cleanly with status 0."""
    from car_lense_engine.export import cli

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
