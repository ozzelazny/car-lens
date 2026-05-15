"""Tests for the popularity-based class ranker."""

from __future__ import annotations

from car_lense_engine.catalog.schema import Catalog, Make, Meta, Model
from car_lense_engine.crawler.seed.ranker import (
    MAKE_POPULARITY,
    rank_models,
    recency_weight,
)


def _make_catalog(makes: list[Make]) -> Catalog:
    return Catalog(
        meta=Meta(
            generated_at="2026-01-01T00:00:00Z",
            source="test",
            year_range=(1980, 2026),
            total_makes=len(makes),
            total_models=sum(len(m.models) for m in makes),
            total_class_entries=sum(len(md.years) for m in makes for md in m.models),
        ),
        makes=makes,
    )


def test_score_monotonic_in_recency() -> None:
    """Same make, newer year_max → higher score."""
    catalog = _make_catalog(
        [
            Make(
                make_id=1,
                make_name="Honda",
                models=[
                    Model(model_id=1, model_name="Old", years=[2000]),
                    Model(model_id=2, model_name="New", years=[2024]),
                ],
            ),
        ]
    )
    ranked = rank_models(catalog, top_n=10)
    by_model = {r.model: r.score for r in ranked}
    assert by_model["New"] > by_model["Old"]


def test_score_monotonic_in_make_popularity() -> None:
    """Same year, more-popular make → higher score."""
    catalog = _make_catalog(
        [
            Make(
                make_id=1,
                make_name="Honda",  # weight 1.0
                models=[Model(model_id=1, model_name="X", years=[2024])],
            ),
            Make(
                make_id=2,
                make_name="Maserati",  # weight 0.35
                models=[Model(model_id=2, model_name="X", years=[2024])],
            ),
        ]
    )
    ranked = rank_models(catalog, top_n=10)
    by_make = {r.make: r.score for r in ranked}
    assert by_make["Honda"] > by_make["Maserati"]


def test_top_n_deterministic(tiny_catalog: Catalog) -> None:
    """Identical inputs produce identical output ordering across runs."""
    a = rank_models(tiny_catalog, top_n=10)
    b = rank_models(tiny_catalog, top_n=10)
    assert [(r.make, r.model, r.score) for r in a] == [(r.make, r.model, r.score) for r in b]


def test_top_n_zero_returns_empty(tiny_catalog: Catalog) -> None:
    assert rank_models(tiny_catalog, top_n=0) == []


def test_top_n_larger_than_population_returns_all(tiny_catalog: Catalog) -> None:
    """When top_n > available, return all classes sorted."""
    expected_count = sum(len(mk.models) for mk in tiny_catalog.makes)
    ranked = rank_models(tiny_catalog, top_n=10_000)
    assert len(ranked) == expected_count


def test_tie_break_alpha_make_then_model() -> None:
    """Identical score → sort by make ascending, then model ascending."""
    catalog = _make_catalog(
        [
            Make(
                make_id=1,
                make_name="Honda",
                models=[
                    Model(model_id=2, model_name="Zeta", years=[2024]),
                    Model(model_id=1, model_name="Alpha", years=[2024]),
                ],
            ),
            Make(
                make_id=2,
                make_name="Acura",  # popularity 0.65 — different score, so not in this tie
                models=[Model(model_id=3, model_name="Beta", years=[2024])],
            ),
        ]
    )
    ranked = rank_models(catalog, top_n=10)
    # Honda Alpha and Honda Zeta share a score → Alpha first.
    honda_models = [r.model for r in ranked if r.make == "Honda"]
    assert honda_models == ["Alpha", "Zeta"]


def test_unknown_make_uses_default_weight() -> None:
    """A make absent from MAKE_POPULARITY scores at the default 0.3."""
    catalog = _make_catalog(
        [
            Make(
                make_id=99,
                make_name="GhostMake",
                models=[Model(model_id=1, model_name="X", years=[2024])],
            ),
        ]
    )
    ranked = rank_models(catalog, top_n=1)
    assert "GhostMake" not in MAKE_POPULARITY
    # Score should be (default weight) * recency_weight(2024).
    expected = 0.3 * recency_weight(2024)
    assert ranked[0].score == expected


def test_models_without_years_are_skipped() -> None:
    """Defensive: a model with empty ``years`` should not crash the ranker."""
    catalog = _make_catalog(
        [
            Make(
                make_id=1,
                make_name="Honda",
                models=[
                    Model(model_id=1, model_name="HasYears", years=[2024]),
                    Model(model_id=2, model_name="NoYears", years=[]),
                ],
            ),
        ]
    )
    ranked = rank_models(catalog, top_n=10)
    names = [r.model for r in ranked]
    assert "HasYears" in names
    assert "NoYears" not in names
