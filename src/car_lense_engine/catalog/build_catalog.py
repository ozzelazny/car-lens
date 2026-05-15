"""Orchestrator that builds the canonical (year, make, model) catalog.

Outer loop iterates years; inner loop iterates makes. Results are merged into a
single :class:`Catalog` keyed by ``make_id`` / ``model_id`` so that each
``(make, model)`` carries the union of years it was produced.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

from .nhtsa_client import NHTSAClient
from .schema import Catalog, Make, Meta, Model

logger = logging.getLogger(__name__)


@dataclass
class _MakeAccumulator:
    """In-progress accumulation of a single make's models and their years."""

    make_id: int
    make_name: str
    models: dict[int, _ModelAccumulator] = field(default_factory=dict)


@dataclass
class _ModelAccumulator:
    """In-progress accumulation of a single model's set of production years."""

    model_id: int
    model_name: str
    years: set[int] = field(default_factory=set)


def build_catalog(
    client: NHTSAClient,
    year_range: tuple[int, int],
    *,
    max_makes: int | None = None,
    progress: bool = True,
) -> Catalog:
    """Iterate years x makes and return a fully populated :class:`Catalog`."""
    start, end = year_range
    if start > end:
        raise ValueError(f"invalid year range: {year_range}")
    years = list(range(start, end + 1))

    makes = client.get_car_makes()
    if max_makes is not None:
        makes = makes[:max_makes]
    logger.info("fetched %d makes; iterating %d years", len(makes), len(years))

    accumulators: dict[int, _MakeAccumulator] = {}
    year_iter: Iterable[int] = tqdm(years, desc="years", unit="yr") if progress else years
    for year in year_iter:
        for make in makes:
            records = client.get_models_for_make_year(make.make_id, year)
            if not records:
                continue
            acc = accumulators.setdefault(
                make.make_id,
                _MakeAccumulator(make_id=make.make_id, make_name=make.make_name),
            )
            for rec in records:
                m = acc.models.setdefault(
                    rec.model_id,
                    _ModelAccumulator(model_id=rec.model_id, model_name=rec.model_name),
                )
                m.years.add(year)

    return _finalize(accumulators, year_range)


def _finalize(
    accumulators: dict[int, _MakeAccumulator],
    year_range: tuple[int, int],
) -> Catalog:
    """Convert the mutable accumulators into the immutable output schema."""
    makes_out: list[Make] = []
    total_models = 0
    total_entries = 0
    for acc in accumulators.values():
        models_out: list[Model] = []
        for m in acc.models.values():
            sorted_years = sorted(m.years)
            models_out.append(
                Model(model_id=m.model_id, model_name=m.model_name, years=sorted_years)
            )
            total_entries += len(sorted_years)
        if not models_out:
            continue
        models_out.sort(key=lambda x: x.model_name)
        total_models += len(models_out)
        makes_out.append(Make(make_id=acc.make_id, make_name=acc.make_name, models=models_out))
    makes_out.sort(key=lambda x: x.make_name)

    meta = Meta(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source="NHTSA vPIC",
        year_range=year_range,
        total_makes=len(makes_out),
        total_models=total_models,
        total_class_entries=total_entries,
    )
    return Catalog(meta=meta, makes=makes_out)


def write_catalog(catalog: Catalog, output_path: Path) -> None:
    """Serialize ``catalog`` to ``output_path`` as pretty JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = catalog.model_dump(mode="json")
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
