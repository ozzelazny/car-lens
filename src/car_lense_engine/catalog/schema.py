"""Pydantic models for the NHTSA vPIC catalog output."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    """Base config — forbid unknown fields so we catch drift early."""

    model_config = ConfigDict(extra="forbid")


class Model(_Base):
    """One vehicle model belonging to a make, plus the years it was produced."""

    model_id: int
    model_name: str
    years: list[int] = Field(default_factory=list)


class Make(_Base):
    """One vehicle make and its set of models."""

    make_id: int
    make_name: str
    models: list[Model] = Field(default_factory=list)


class Meta(_Base):
    """Top-level metadata describing how/when the catalog was built."""

    generated_at: str
    source: str
    year_range: tuple[int, int]
    total_makes: int
    total_models: int
    total_class_entries: int


class Catalog(_Base):
    """Full canonical catalog written to ``classes.json``."""

    meta: Meta
    makes: list[Make] = Field(default_factory=list)
