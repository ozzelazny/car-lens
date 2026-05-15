"""Score and rank ``(make, model)`` combos from the NHTSA catalog by US popularity."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from car_lense_engine.catalog.schema import Catalog

logger = logging.getLogger(__name__)


# Hardcoded US best-selling-makes weights. Keys are Title Case to match the
# NHTSA-derived catalog (which loses canonical mixed-case like "BMW" → "Bmw"
# and "McLaren" → "Mclaren"). Anything not listed gets ``_DEFAULT_MAKE_WEIGHT``.
MAKE_POPULARITY: dict[str, float] = {
    "Ford": 1.0,
    "Toyota": 1.0,
    "Chevrolet": 1.0,
    "Honda": 1.0,
    "Nissan": 0.95,
    "Ram": 0.9,
    "Jeep": 0.9,
    "Gmc": 0.9,
    "Hyundai": 0.9,
    "Kia": 0.85,
    "Subaru": 0.85,
    "Mazda": 0.8,
    "Volkswagen": 0.8,
    "Bmw": 0.8,
    "Mercedes-Benz": 0.8,
    "Audi": 0.75,
    "Tesla": 0.75,
    "Lexus": 0.7,
    "Acura": 0.65,
    "Buick": 0.65,
    "Cadillac": 0.65,
    "Volvo": 0.6,
    "Infiniti": 0.55,
    "Lincoln": 0.55,
    "Chrysler": 0.55,
    "Dodge": 0.7,
    "Mitsubishi": 0.5,
    "Mini": 0.5,
    "Porsche": 0.55,
    "Land Rover": 0.5,
    "Jaguar": 0.45,
    "Genesis": 0.5,
    "Fiat": 0.4,
    "Alfa Romeo": 0.4,
    "Mclaren": 0.35,
    "Maserati": 0.35,
}

_DEFAULT_MAKE_WEIGHT: float = 0.3

# Half-life for the recency curve, in years. exp(-0/15)=1, exp(-20/15)≈0.26,
# exp(-40/15)≈0.069 — we floor the very-old tail to 0.02 so dropped-from-market
# models still survive long enough for the top-N to include some classics.
_RECENCY_TAU: float = 15.0
_RECENCY_FLOOR_AGE: int = 40
_RECENCY_FLOOR: float = 0.02


class RankedClass(BaseModel):
    """One ``(make, model)`` candidate with its popularity score and year span."""

    model_config = ConfigDict(extra="forbid")

    make: str
    model: str
    make_id: int
    model_id: int
    year_min: int
    year_max: int
    score: float


def _current_year() -> int:
    """Return the current calendar year in UTC (factored out for testability)."""
    return datetime.now(UTC).year


def recency_weight(year_max: int, *, current_year: int | None = None) -> float:
    """Smooth recency curve: 1.0 today, ~0.26 at 20y old, floored to 0.02 past 40y."""
    cy = current_year if current_year is not None else _current_year()
    age = cy - year_max
    if age <= 0:
        return 1.0
    if age >= _RECENCY_FLOOR_AGE:
        return _RECENCY_FLOOR
    return math.exp(-age / _RECENCY_TAU)


def make_popularity_weight(make: str) -> float:
    """Look up a make's hardcoded popularity weight (default ``0.3``)."""
    return MAKE_POPULARITY.get(make, _DEFAULT_MAKE_WEIGHT)


def rank_models(catalog: Catalog, top_n: int = 2000) -> list[RankedClass]:
    """Score every ``(make, model)`` in the catalog and return the top ``top_n``."""
    if top_n < 0:
        raise ValueError(f"top_n must be >= 0, got {top_n}")
    if top_n == 0:
        return []

    cy = _current_year()
    ranked: list[RankedClass] = []
    for make in catalog.makes:
        m_weight = make_popularity_weight(make.make_name)
        for model in make.models:
            if not model.years:
                continue
            ymin = min(model.years)
            ymax = max(model.years)
            r_weight = recency_weight(ymax, current_year=cy)
            score = m_weight * r_weight
            ranked.append(
                RankedClass(
                    make=make.make_name,
                    model=model.model_name,
                    make_id=make.make_id,
                    model_id=model.model_id,
                    year_min=ymin,
                    year_max=ymax,
                    score=score,
                )
            )

    # Deterministic ordering: descending score, then make asc, then model asc.
    ranked.sort(key=lambda r: (-r.score, r.make, r.model))
    return ranked[:top_n]
