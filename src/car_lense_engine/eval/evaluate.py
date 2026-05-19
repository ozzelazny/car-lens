"""Phase 5.4 evaluation harness for the deployed retrieval pipeline.

This module measures the *real* held-out test-set accuracy of the
fine-tuned MobileCLIP-S2 backbone + the v1 single-prototype cache the
``recognize()`` service container loads in production. Unlike the
:mod:`car_lense_engine.eval.baseline` Phase 5.1 harness (which builds
prototypes from scratch every run from the train split), the Phase 5.4
harness *loads* the prototypes from the on-disk cache produced by the
``build-prototypes`` CLI -- so it answers the question "given the
exact bits the service uses, what accuracy would users see?".

Breakdowns
----------

Top-K accuracy is reported overall and broken down by three axes:

* **make** -- canonical_make from listings.
* **view** -- images.view (the 5 exterior views from Phase 3.3).
* **era** -- decade bucket derived from listings.generation_year, e.g.
  ``"2000s"``, ``"2010s"``. NULL years go into the ``"unknown"`` bucket.

In addition, the top-N most frequent top-1 confusion pairs are
reported, plus the most-confused class within each make (filtered to
makes with at least 50 test images).

Design notes
------------

* **Single prototype per class only.** The harness rejects v2
  schema_version cache files; per-view evaluation is a separate
  question and would need a different runner that consults the view
  for retrieval. The intent of this module is: how does the deployed
  v1 path actually perform on the held-out test set?
* **No binary-classifier rejection.** This harness measures the
  backbone + prototype pipeline in isolation; the exterior-vs-not
  rejection knob is layered on by the service and not part of the
  retrieval accuracy question.
* **Reuses the Phase 5.1 runner.** The image-loading, batching,
  encoding, L2-norm, and cosine-similarity machinery already lives
  on :class:`car_lense_engine.eval.baseline._BaselineRunner`. We
  build a runner from the config, run a sibling helper
  :func:`_evaluate_with_predictions` that returns per-row top-K
  predictions plus the row's view/era/make context, then bucket
  pure-Pythonically.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .baseline import (
    BaselineConfig,
    _BaselineRunner,
    _chunked,
    _coerce_source_field,
    _normalize_sources,
    _select_rows,
    _source_where_clause,
    class_id_for,
)
from .build_prototypes_cli import _display_name_for

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- constants

PROTOTYPE_SCHEMA_V1 = 1
PROTOTYPE_SCHEMA_V2 = 2

ERA_UNKNOWN = "unknown"

# Minimum number of test images for a make to appear in
# ``top_confusions_per_make``. Makes with fewer test rows don't have
# enough signal to claim a "most-confused" pair reliably.
MIN_TEST_IMAGES_FOR_PER_MAKE_CONFUSION = 50

# How many top confusion pairs to keep in the global report.
TOP_CONFUSION_LIMIT = 50

# How many top confusions to keep per make (when the make qualifies).
TOP_CONFUSION_PER_MAKE_LIMIT = 3


# --------------------------------------------------------------- public models


class EvaluationConfig(BaseModel):
    """Frozen configuration of a Phase 5.4 evaluation run.

    ``source`` is a list of ``listings.source`` values to evaluate
    against. A single source (``["compcars"]``) reproduces the legacy
    Phase 5.4 behaviour; passing multiple sources (e.g.
    ``["compcars", "vmmrdb", "stanford_cars"]``) joins them in a single
    SQL ``IN`` clause so the test set spans every dataset the caller
    listed. For backward compat the validator also accepts a bare
    string (legacy single-source config) and a comma-separated string
    (e.g. ``"compcars,vmmrdb"``).
    """

    model_config = ConfigDict(extra="forbid")

    db_path: Path
    source: list[str] = Field(default_factory=lambda: ["compcars"])
    test_split: str = "test"
    checkpoint_path: Path | None = None
    prototypes_path: Path
    model_name: str = "MobileCLIP-S2"
    pretrained: str = "datacompdr"
    device: str = "cpu"
    batch_size: int = 64
    top_k: tuple[int, ...] = (1, 3, 5, 10)
    era_bucket_years: int = 10

    @field_validator("source", mode="before")
    @classmethod
    def _coerce_source(cls, value: object) -> list[str]:
        """Accept a bare string, comma-separated string, or list of strings.

        Backwards compatibility: existing call sites that pass
        ``source="compcars"`` continue to work; the validator wraps
        them in a single-element list. A comma-separated string like
        ``"compcars,vmmrdb"`` is split into ``["compcars", "vmmrdb"]``
        so the CLI argv form round-trips through the pydantic config
        without extra plumbing.
        """
        return _coerce_source_field(value)


class CellMetrics(BaseModel):
    """Per-cell (make / view / era) accuracy bucket."""

    model_config = ConfigDict(extra="forbid")

    n: int
    top_k_correct: dict[int, int] = Field(default_factory=dict)
    """Map ``k -> count_of_correct_top_k_predictions``. Accuracy at K is
    ``top_k_correct[k] / n`` (caller arithmetic; we don't pre-divide so
    cells with ``n == 0`` can still be represented unambiguously)."""


class ConfusionRow(BaseModel):
    """One ``(true, pred, count, rate)`` confusion-table entry."""

    model_config = ConfigDict(extra="forbid")

    true: str
    pred: str
    count: int
    rate: float


class EvaluationReport(BaseModel):
    """Full Phase 5.4 evaluation report."""

    model_config = ConfigDict(extra="forbid")

    overall: CellMetrics
    per_make: dict[str, CellMetrics] = Field(default_factory=dict)
    per_view: dict[str, CellMetrics] = Field(default_factory=dict)
    per_era: dict[str, CellMetrics] = Field(default_factory=dict)
    top_confusions: list[ConfusionRow] = Field(default_factory=list)
    top_confusions_per_make: dict[str, list[ConfusionRow]] = Field(default_factory=dict)
    config: EvaluationConfig
    n_classes: int
    elapsed_seconds: float


# --------------------------------------------------------------- public API


def evaluate(*, conn: sqlite3.Connection, config: EvaluationConfig) -> EvaluationReport:
    """Run the Phase 5.4 evaluation against ``(source, test_split)``.

    Loads the v1 prototype cache from ``config.prototypes_path``, builds
    a :class:`_BaselineRunner` from the config (the runner owns the
    image-encoder + preprocess and overlays the optional fine-tuned
    checkpoint), evaluates every test row, and returns a structured
    report with overall + per-(make, view, era) top-K accuracy plus the
    top confusion pairs.

    Raises
    ------
    RuntimeError
        If the prototype cache is missing, malformed, or uses
        ``schema_version == 2`` (per-view cache -- not supported here).
    """
    start = time.monotonic()

    baseline_cfg = BaselineConfig(
        model_name=config.model_name,
        pretrained=config.pretrained,
        device=config.device,
        batch_size=config.batch_size,
        top_ks=tuple(config.top_k),
        checkpoint_path=config.checkpoint_path,
    )
    class_ids, proto_tensor, display_by_cid = _load_v1_prototypes(config.prototypes_path)
    n_classes = len(class_ids)

    runner = _BaselineRunner(baseline_cfg)
    per_row = _evaluate_with_predictions(
        runner=runner,
        conn=conn,
        class_ids=class_ids,
        proto_tensor=proto_tensor,
        source=config.source,
        split=config.test_split,
        top_ks=tuple(config.top_k),
        era_bucket_years=config.era_bucket_years,
    )

    overall = _build_overall(per_row, config.top_k)
    per_make = _build_breakdown(per_row, key="make", top_ks=config.top_k)
    per_view = _build_breakdown(per_row, key="view", top_ks=config.top_k)
    per_era = _build_breakdown(per_row, key="era", top_ks=config.top_k)

    top_confusions, per_make_confusions = _build_confusions(per_row, display_by_cid=display_by_cid)

    elapsed = time.monotonic() - start
    return EvaluationReport(
        overall=overall,
        per_make=per_make,
        per_view=per_view,
        per_era=per_era,
        top_confusions=top_confusions,
        top_confusions_per_make=per_make_confusions,
        config=config,
        n_classes=n_classes,
        elapsed_seconds=elapsed,
    )


# --------------------------------------------------------------- internals


class _RowResult(BaseModel):
    """Per-test-row eval result. Internal -- not part of the report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    true_class_id: str
    pred_class_id: str | None
    top_k_hit: dict[int, bool]
    """Map ``k -> whether the true class is in the top-K predictions``."""
    view: str | None
    era: str
    make: str


def _load_v1_prototypes(path: Path) -> tuple[list[str], Any, dict[str, str]]:
    """Load a v1 prototype cache and return ``(class_ids, proto_tensor, displays)``.

    Raises ``RuntimeError`` for missing files, malformed payloads, or
    v2 schema files (per-view caches are not supported by this harness).
    """
    if not path.exists():
        raise RuntimeError(
            f"prototype cache does not exist at {path}; run `build-prototypes` to produce it first"
        )
    import torch  # noqa: PLC0415

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"prototype cache at {path} is malformed; expected a dict, got {type(payload).__name__}"
        )
    schema_version = int(payload.get("schema_version", PROTOTYPE_SCHEMA_V1))
    if schema_version == PROTOTYPE_SCHEMA_V2:
        raise RuntimeError(
            f"prototype cache at {path} is schema_version=2 (per-view); "
            "the Phase 5.4 evaluate-recognize harness only supports v1 "
            "single-prototype caches. Re-run `build-prototypes` without "
            "`--per-view` to produce the v1 cache."
        )
    required = ("class_ids", "prototypes")
    if any(k not in payload for k in required):
        raise RuntimeError(
            f"prototype cache at {path} is malformed; expected keys "
            f"{required}, got {sorted(payload.keys())}"
        )
    class_ids_raw = payload["class_ids"]
    proto_tensor = payload["prototypes"]
    if not isinstance(class_ids_raw, list):
        raise RuntimeError(
            f"prototype cache at {path}: class_ids must be a list, "
            f"got {type(class_ids_raw).__name__}"
        )
    class_ids: list[str] = [str(c) for c in class_ids_raw]
    if int(proto_tensor.shape[0]) != len(class_ids):
        raise RuntimeError(
            f"prototype cache at {path} has inconsistent shape: "
            f"{len(class_ids)} class_ids vs "
            f"{int(proto_tensor.shape[0])} prototype rows"
        )

    # Display names: prefer the cache's own column (cosmetically nicer);
    # fall back to deriving from the class id if missing.
    display_names_raw = payload.get("display_names")
    if isinstance(display_names_raw, list) and len(display_names_raw) == len(class_ids):
        displays = {cid: str(name) for cid, name in zip(class_ids, display_names_raw, strict=True)}
    else:
        displays = {cid: _display_name_for(cid) for cid in class_ids}

    return class_ids, proto_tensor, displays


def _evaluate_with_predictions(
    *,
    runner: _BaselineRunner,
    conn: sqlite3.Connection,
    class_ids: list[str],
    proto_tensor: Any,
    source: str | list[str],
    split: str,
    top_ks: tuple[int, ...],
    era_bucket_years: int,
) -> list[_RowResult]:
    """Per-row top-K evaluation against the loaded prototypes.

    Returns one :class:`_RowResult` per test image that was
    successfully loaded. Test rows whose true ``class_id`` doesn't
    appear in the prototype index are still included (they always
    count as a miss because the true class can never be in any top-K),
    matching the Phase 5.1 runner's behaviour.
    """
    if not class_ids:
        # Defensive: if the prototype index is empty the evaluation
        # cannot make any prediction. Return an empty result list so
        # the caller's breakdowns produce all-zero cells.
        logger.warning("evaluate-recognize: prototype cache is empty -- no predictions possible")
        return []

    rows = _select_rows(conn, source=source, split=split)
    if not rows:
        logger.warning(
            "evaluate-recognize: no test rows for source=%s split=%s",
            source,
            split,
        )
        return []

    # Pull the per-row make + era context from the same join, keyed by
    # local_path so we can attach it to embedded rows below. We can't
    # add it to ``_select_rows`` without changing the public surface
    # everyone else (training, build_prototypes) uses, so we make a
    # second focused query here.
    context_by_path = _select_row_context(
        conn,
        source=source,
        split=split,
        era_bucket_years=era_bucket_years,
    )

    logger.info(
        "evaluate-recognize: evaluating %d test images against %d prototypes (source=%s split=%s)",
        len(rows),
        len(class_ids),
        source,
        split,
    )

    embeddings, used_class_ids, survivor_paths = _embed_rows_with_paths(runner, rows)
    if int(embeddings.shape[0]) == 0:
        return []

    # Compute per-row top-K predictions.
    sims = embeddings @ proto_tensor.to(embeddings.device).T  # (n_test, n_classes)
    max_k = max(top_ks)
    k_eff = min(max_k, int(sims.shape[1]))
    _, topk_idx = sims.topk(k=k_eff, dim=-1)
    topk_idx_list: list[list[int]] = topk_idx.tolist()

    class_to_idx = {cid: i for i, cid in enumerate(class_ids)}
    results: list[_RowResult] = []
    n_attached = min(len(used_class_ids), len(survivor_paths), len(topk_idx_list))
    for i in range(n_attached):
        true_cid = used_class_ids[i]
        topk = topk_idx_list[i]
        pred_cid: str | None = class_ids[topk[0]] if topk else None
        true_idx = class_to_idx.get(true_cid)
        top_k_hit: dict[int, bool] = {}
        for k in top_ks:
            if true_idx is None:
                top_k_hit[k] = False
            else:
                top_k_hit[k] = true_idx in topk[:k]
        path = survivor_paths[i]
        ctx = context_by_path.get(str(path))
        if ctx is None:
            # Shouldn't happen -- the context query and the row query
            # use identical WHERE clauses. Default to neutral values
            # so the row still contributes to the overall metric.
            ctx_view: str | None = None
            ctx_era = ERA_UNKNOWN
            ctx_make = "unknown"
        else:
            ctx_view, ctx_era, ctx_make = ctx
        results.append(
            _RowResult(
                true_class_id=true_cid,
                pred_class_id=pred_cid,
                top_k_hit=top_k_hit,
                view=ctx_view,
                era=ctx_era,
                make=ctx_make,
            )
        )
    return results


def _embed_rows_with_paths(
    runner: _BaselineRunner,
    rows: list[tuple[str, str | None, Path]],
) -> tuple[Any, list[str], list[Path]]:
    """Embed ``rows`` and return ``(embeddings, class_ids, paths)`` aligned.

    This mirrors :meth:`_BaselineRunner._embed_rows` exactly but also
    keeps the per-row ``path`` alongside the surviving class ids so
    callers can attach per-row context (view / make / era) that the
    baseline runner doesn't track. We deliberately don't modify
    ``_embed_rows`` because the public Phase 5.1 callers don't need the
    paths and the extra return value would broaden the contract.
    """
    runner._ensure_model()  # noqa: SLF001
    torch_mod = runner._require_torch()  # noqa: SLF001
    config = runner._config  # noqa: SLF001
    all_chunks: list[Any] = []
    used_class_ids: list[str] = []
    used_paths: list[Path] = []
    n_total = len(rows)
    n_done = 0
    for chunk in _chunked(rows, config.batch_size):
        ok_class_ids: list[str] = []
        ok_paths: list[Path] = []
        tensors: list[Any] = []
        for cid, _view, path in chunk:
            try:
                tensors.append(runner._load_and_preprocess(path))  # noqa: SLF001
            except Exception as exc:  # noqa: BLE001 -- log + skip
                logger.warning("evaluate-recognize: skipping %s (%s)", path, exc)
                continue
            ok_class_ids.append(cid)
            ok_paths.append(path)
        if not tensors:
            continue
        batch = torch_mod.stack(tensors).to(config.device)
        with torch_mod.no_grad():
            features = runner._encode_image(batch)  # noqa: SLF001
            features = features / features.norm(dim=-1, keepdim=True)
        all_chunks.append(features)
        used_class_ids.extend(ok_class_ids)
        used_paths.extend(ok_paths)
        n_done += len(ok_class_ids)
        if n_done and (n_done % max(50, config.batch_size * 4) < config.batch_size):
            logger.info("evaluate-recognize: embedded %d / %d", n_done, n_total)
    if not all_chunks:
        return torch_mod.zeros((0, 0)), [], []
    return torch_mod.cat(all_chunks, dim=0), used_class_ids, used_paths


def _select_row_context(
    conn: sqlite3.Connection,
    *,
    source: str | list[str],
    split: str,
    era_bucket_years: int,
) -> dict[str, tuple[str | None, str, str]]:
    """Return ``{local_path: (view, era_bucket, make)}`` for every row.

    Uses the same WHERE clause as :func:`_select_rows` so the keys
    line up exactly. ``source`` may be a single source name or a list
    (multi-source eval); the SQL switches between ``listings.source = ?``
    and ``listings.source IN (?, ?, ...)`` accordingly. ``view`` is the
    raw ``images.view`` column; ``era_bucket`` is derived from
    ``listings.generation_year`` (NULL -> ``"unknown"``); ``make`` is
    ``listings.canonical_make``.
    """
    sources = _normalize_sources(source)
    source_clause, source_params = _source_where_clause(sources)
    sql = (
        "SELECT listings.canonical_make AS make, "
        "       listings.generation_year AS year, "
        "       images.view AS view, "
        "       images.local_path AS local_path "
        "FROM listings "
        "JOIN images ON images.listing_id = listings.listing_id "
        f"WHERE {source_clause} AND images.split = ? "
        "ORDER BY images.image_id"
    )
    cur = conn.execute(sql, (*source_params, split))
    out: dict[str, tuple[str | None, str, str]] = {}
    for row in cur.fetchall():
        local_path = row["local_path"]
        if not local_path:
            continue
        # Only keep rows that pass the same canonical-fields gate as
        # ``_select_rows`` -- the eval set is the intersection.
        make_raw = row["make"]
        if make_raw is None or not str(make_raw).strip():
            continue
        # We don't have model in this projection but _select_rows also
        # filters on canonical_model NOT NULL. To stay in sync without
        # a second query, we'd need to refetch model. Pragmatic: also
        # check generation_year because that's part of the class_id
        # gate. Rows missing canonical_model will still appear here
        # but they'll never match a survivor_path (because _select_rows
        # drops them). The result is harmless: extra keys in
        # ``out`` that no caller looks up.
        view_raw = row["view"]
        view: str | None = str(view_raw) if view_raw is not None else None
        era = _era_bucket(row["year"], era_bucket_years)
        make = str(make_raw).strip()
        out[str(local_path)] = (view, era, make)
    return out


def _era_bucket(year: int | None, era_bucket_years: int) -> str:
    """Map a generation_year integer to a decade-style era bucket.

    ``era_bucket_years=10`` (default) produces ``"2000s"`` / ``"2010s"``.
    Other bucket sizes produce ``"<start>-<start+size-1>"``. NULL
    years go into the ``ERA_UNKNOWN`` bucket.
    """
    if year is None:
        return ERA_UNKNOWN
    try:
        y = int(year)
    except (TypeError, ValueError):
        return ERA_UNKNOWN
    if era_bucket_years <= 0:
        # Pathological -- treat as raw year.
        return str(y)
    start = (y // era_bucket_years) * era_bucket_years
    if era_bucket_years == 10:
        return f"{start}s"
    return f"{start}-{start + era_bucket_years - 1}"


def _build_overall(rows: list[_RowResult], top_ks: tuple[int, ...]) -> CellMetrics:
    """Aggregate the overall top-K hit counts across every row."""
    n = len(rows)
    counts: dict[int, int] = dict.fromkeys(top_ks, 0)
    for r in rows:
        for k in top_ks:
            if r.top_k_hit.get(k):
                counts[k] += 1
    return CellMetrics(n=n, top_k_correct=counts)


def _build_breakdown(
    rows: list[_RowResult],
    *,
    key: str,
    top_ks: tuple[int, ...],
) -> dict[str, CellMetrics]:
    """Group ``rows`` by ``key`` (one of ``"make"``, ``"view"``, ``"era"``)
    and compute top-K accuracy per bucket.

    ``view`` may be NULL on a row; such rows are collected under the
    ``"unknown"`` bucket so they aren't silently dropped.
    """
    bucketed: dict[str, list[_RowResult]] = defaultdict(list)
    for r in rows:
        value: str
        if key == "make":
            value = r.make
        elif key == "view":
            value = r.view if r.view is not None else "unknown"
        elif key == "era":
            value = r.era
        else:  # pragma: no cover -- defensive
            raise ValueError(f"unknown breakdown key: {key!r}")
        bucketed[value].append(r)

    out: dict[str, CellMetrics] = {}
    for value, bucket_rows in bucketed.items():
        counts: dict[int, int] = dict.fromkeys(top_ks, 0)
        for r in bucket_rows:
            for k in top_ks:
                if r.top_k_hit.get(k):
                    counts[k] += 1
        out[value] = CellMetrics(n=len(bucket_rows), top_k_correct=counts)
    return out


def _build_confusions(
    rows: list[_RowResult],
    *,
    display_by_cid: dict[str, str],
) -> tuple[list[ConfusionRow], dict[str, list[ConfusionRow]]]:
    """Compute the global top-N confusion table + per-make top-3."""
    global_conf: Counter[tuple[str, str]] = Counter()
    total_top1_misses = 0
    per_make_conf: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    per_make_n_test: Counter[str] = Counter()

    for r in rows:
        per_make_n_test[r.make] += 1
        if r.top_k_hit.get(1):
            continue
        # Top-1 miss.
        total_top1_misses += 1
        if r.pred_class_id is None or r.pred_class_id == r.true_class_id:
            # ``pred == true`` would be a hit; if pred is None there's
            # nothing to record. Defensive.
            continue
        pair = (r.true_class_id, r.pred_class_id)
        global_conf[pair] += 1
        per_make_conf[r.make][pair] += 1

    def _to_row(pair: tuple[str, str], count: int, denom: int) -> ConfusionRow:
        true_cid, pred_cid = pair
        rate = (count / denom) if denom > 0 else 0.0
        return ConfusionRow(
            true=display_by_cid.get(true_cid, true_cid),
            pred=display_by_cid.get(pred_cid, pred_cid),
            count=count,
            rate=rate,
        )

    top_confusions = [
        _to_row(pair, count, total_top1_misses)
        for pair, count in global_conf.most_common(TOP_CONFUSION_LIMIT)
    ]

    per_make_out: dict[str, list[ConfusionRow]] = {}
    for make, n_test in per_make_n_test.items():
        if n_test < MIN_TEST_IMAGES_FOR_PER_MAKE_CONFUSION:
            continue
        conf = per_make_conf.get(make)
        if not conf:
            continue
        make_total_misses = sum(conf.values())
        per_make_out[make] = [
            _to_row(pair, count, make_total_misses)
            for pair, count in conf.most_common(TOP_CONFUSION_PER_MAKE_LIMIT)
        ]
    return top_confusions, per_make_out


def write_report(report: EvaluationReport, path: Path) -> None:
    """Serialize an :class:`EvaluationReport` to JSON at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def summarize(report: EvaluationReport) -> str:
    """Single-line stdout summary of the top-line numbers."""
    overall = report.overall
    n = overall.n
    pieces: list[str] = []
    for k in sorted(report.config.top_k):
        correct = overall.top_k_correct.get(k, 0)
        acc = correct / n if n else 0.0
        pieces.append(f"top_{k}={acc:.4f}")
    pieces.append(f"n_test={n}")
    pieces.append(f"n_classes={report.n_classes}")
    pieces.append(f"elapsed={report.elapsed_seconds:.1f}s")
    return " ".join(pieces)


def format_cell_table(
    cells: dict[str, CellMetrics],
    *,
    top_ks: Iterable[int],
    sort_by_k: int = 1,
    limit: int | None = None,
    header: str = "key",
) -> str:
    """Render a per-cell breakdown as a fixed-width text table.

    Rows are sorted ascending by accuracy at ``sort_by_k`` (worst
    first), then by descending ``n`` to put well-sampled rows on top
    of ties. If ``limit`` is set, only the worst ``limit`` rows are
    rendered.
    """
    if not cells:
        return f"(no {header} cells)"
    top_ks_t = tuple(top_ks)

    def _acc(cell: CellMetrics, k: int) -> float:
        return (cell.top_k_correct.get(k, 0) / cell.n) if cell.n else 0.0

    sortable = list(cells.items())
    sortable.sort(key=lambda kv: (_acc(kv[1], sort_by_k), -kv[1].n, kv[0]))
    if limit is not None:
        sortable = sortable[:limit]

    # Compute column widths.
    key_w = max(len(header), max((len(k) for k, _ in sortable), default=0))
    n_w = max(len("n"), max((len(str(c.n)) for _, c in sortable), default=1))
    k_w = 8  # "0.0000" + padding
    parts_header = [header.ljust(key_w), "n".rjust(n_w)]
    parts_header.extend(f"top_{k}".rjust(k_w) for k in top_ks_t)
    lines = [" | ".join(parts_header)]
    lines.append("-+-".join("-" * len(p) for p in parts_header))
    for key, cell in sortable:
        parts = [key.ljust(key_w), str(cell.n).rjust(n_w)]
        for k in top_ks_t:
            parts.append(f"{_acc(cell, k):.4f}".rjust(k_w))
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def format_confusions(confusions: list[ConfusionRow], *, limit: int | None = None) -> str:
    """Render the global top-N confusions as a fixed-width text table."""
    if not confusions:
        return "(no confusions)"
    rows = confusions[:limit] if limit is not None else confusions
    true_w = max(len("true"), max(len(r.true) for r in rows))
    pred_w = max(len("pred"), max(len(r.pred) for r in rows))
    count_w = max(len("count"), max(len(str(r.count)) for r in rows))
    rate_w = 8
    header = " | ".join(
        [
            "true".ljust(true_w),
            "pred".ljust(pred_w),
            "count".rjust(count_w),
            "rate".rjust(rate_w),
        ]
    )
    sep = "-+-".join(
        [
            "-" * true_w,
            "-" * pred_w,
            "-" * count_w,
            "-" * rate_w,
        ]
    )
    lines = [header, sep]
    for r in rows:
        lines.append(
            " | ".join(
                [
                    r.true.ljust(true_w),
                    r.pred.ljust(pred_w),
                    str(r.count).rjust(count_w),
                    f"{r.rate:.4f}".rjust(rate_w),
                ]
            )
        )
    return "\n".join(lines)


__all__ = [
    "CellMetrics",
    "ConfusionRow",
    "EvaluationConfig",
    "EvaluationReport",
    "evaluate",
    "format_cell_table",
    "format_confusions",
    "summarize",
    "write_report",
    # Re-exports for the CLI:
    "class_id_for",
]
