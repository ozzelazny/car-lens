"""Tests for the Wikimedia Commons ingest module (Phase 4.4).

All tests stub the HTTP session so no real Wikimedia API calls are made.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from car_lense_engine.dataset.wikimedia import (
    WikimediaIngestConfig,
    WikimediaIngestSummary,
    _RateLimiter,
    extract_label_triple,
    ingest_wikimedia,
    iter_category_files,
)
from car_lense_engine.db import images, listings, open_db

# --------------------------------------------------------- fixtures


@pytest.fixture
def db(tmp_path: Path) -> Iterable[sqlite3.Connection]:
    conn = open_db(tmp_path / "crawl.sqlite")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "wikimedia_out"


# --------------------------------------------------------- helpers


@dataclass
class _FakeResponse:
    """Minimal stand-in for an httpx.Response.

    Carries a fixed JSON payload (for API calls) or raw bytes (for image
    fetches). ``raise_for_status`` is intentionally absent — the ingest
    code never calls it; it checks ``status_code`` directly.
    """

    status_code: int = 200
    payload: dict[str, Any] = field(default_factory=dict)
    content: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> dict[str, Any]:
        return self.payload


@dataclass
class _FakeSession:
    """Records every .get(...) call and returns scripted responses.

    Construct with ``script`` = an iterable of ``_FakeResponse``; each .get
    pops the next response. ``calls`` accumulates the URL + params for
    assertions.
    """

    script: list[_FakeResponse]
    calls: list[tuple[str, dict[str, Any] | None]] = field(default_factory=list)

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.calls.append((url, params))
        if not self.script:
            raise AssertionError(
                f"FakeSession ran out of scripted responses on .get({url!r}, params={params!r})"
            )
        return self.script.pop(0)


def _png_bytes(n: int) -> bytes:
    """Make a unique byte blob (each ``n`` returns distinct content).

    The actual bytes don't need to be a real image — the ingest doesn't
    decode them (Phase 3.2's pHash work is out of scope here).
    """
    return f"fake-png-{n}".encode() * 16


# --------------------------------------------------------- label extraction


def test_extract_label_triple_year_make_model() -> None:
    """Year-prefix category + sibling 'Ford Mustang' yields (1965, Ford, Mustang)."""
    triple = extract_label_triple(["Category:1965 Ford Mustang", "Category:Ford Mustang"])
    assert triple == (1965, "Ford", "Mustang")


def test_extract_label_triple_explicit_introduced_in() -> None:
    """The 'Cars introduced in YYYY' canonical form also works."""
    triple = extract_label_triple(
        ["Category:Cars introduced in 1972", "Category:Chevrolet Chevelle"]
    )
    assert triple == (1972, "Chevrolet", "Chevelle")


def test_extract_label_triple_decade_only_drops() -> None:
    """A decade-only category (no make) drops the row entirely.

    The year fallback yields 1965 (the 1960s midpoint) but ``_find_make``
    has nothing to grab, so the triple is rejected as a whole.
    """
    triple = extract_label_triple(["Category:1960s automobiles"])
    assert triple is None


def test_extract_label_triple_no_make() -> None:
    """A year-only year category with no make-bearing sibling drops."""
    triple = extract_label_triple(
        [
            "Category:Cars introduced in 1970",
            "Category:Black and white photographs of automobiles",
        ]
    )
    assert triple is None


def test_extract_label_triple_make_alias() -> None:
    """'Chevy' aliases to 'Chevrolet' via the Phase 4.5 alias map."""
    triple = extract_label_triple(["Category:1972 Chevy", "Category:Chevrolet Chevelle"])
    assert triple == (1972, "Chevrolet", "Chevelle")


def test_extract_label_triple_brand_caps() -> None:
    """Brand-cased makes (BMW) come through with the right casing."""
    triple = extract_label_triple(["Category:Cars introduced in 1989", "Category:BMW M3"])
    assert triple == (1989, "BMW", "M3")


def test_extract_label_triple_year_out_of_range_dropped() -> None:
    """Years outside [year_min, year_max] are rejected at extraction."""
    triple = extract_label_triple(
        ["Category:1965 Ford Mustang", "Category:Ford Mustang"],
        year_min=1980,
        year_max=2000,
    )
    assert triple is None


def test_extract_label_triple_decade_midpoint_with_make() -> None:
    """Decade-midpoint year IS accepted when a separate make category exists."""
    triple = extract_label_triple(["Category:1960s automobiles", "Category:Ford Mustang"])
    # 1960s midpoint = 1965; make=Ford via "Ford Mustang"; model=Mustang.
    assert triple == (1965, "Ford", "Mustang")


def test_extract_label_triple_with_namespace_prefix_or_underscores() -> None:
    """``Category:`` namespace + underscores are normalized away."""
    triple = extract_label_triple(["Category:1965_Ford_Mustang", "Category:Ford_Mustang"])
    assert triple == (1965, "Ford", "Mustang")


# --------------------------------------------------------- iter_category_files


def test_iter_category_files_paginates() -> None:
    """Two-page response: yields all members + honors ``cmcontinue``."""
    script = [
        _FakeResponse(
            status_code=200,
            payload={
                "query": {
                    "categorymembers": [
                        {"pageid": 1, "ns": 6, "title": "File:A.jpg"},
                        {"pageid": 2, "ns": 6, "title": "File:B.jpg"},
                    ]
                },
                "continue": {"cmcontinue": "tok-xyz", "continue": "-||"},
            },
        ),
        _FakeResponse(
            status_code=200,
            payload={
                "query": {
                    "categorymembers": [
                        {"pageid": 3, "ns": 6, "title": "File:C.jpg"},
                    ]
                },
                # no "continue" -> stop.
            },
        ),
    ]
    session = _FakeSession(script=script)
    rate = _RateLimiter(min_delay=0.0)

    got = list(
        iter_category_files(
            "http://api.example/w/api.php",
            "Category:Cars introduced in 1965",
            session=session,
            rate_limit=rate,
        )
    )
    assert [m["title"] for m in got] == ["File:A.jpg", "File:B.jpg", "File:C.jpg"]
    # Second call should have included cmcontinue.
    assert len(session.calls) == 2
    second_params = session.calls[1][1]
    assert second_params is not None
    assert second_params.get("cmcontinue") == "tok-xyz"
    # First call should NOT carry cmcontinue.
    assert "cmcontinue" not in (session.calls[0][1] or {})


def test_iter_category_files_max_files() -> None:
    """max_files caps the iterator mid-stream."""
    script = [
        _FakeResponse(
            status_code=200,
            payload={
                "query": {
                    "categorymembers": [
                        {"pageid": i, "ns": 6, "title": f"File:{i}.jpg"} for i in range(10)
                    ]
                },
                "continue": {"cmcontinue": "would-paginate"},
            },
        ),
    ]
    session = _FakeSession(script=script)
    rate = _RateLimiter(min_delay=0.0)

    got = list(
        iter_category_files(
            "http://api.example/w/api.php",
            "Category:X",
            session=session,
            rate_limit=rate,
            max_files=3,
        )
    )
    assert len(got) == 3
    # We stopped mid-page; the second page request should NOT have been made.
    assert len(session.calls) == 1


def test_iter_category_files_adds_category_prefix() -> None:
    """A category title without 'Category:' prefix gets one added."""
    script = [
        _FakeResponse(status_code=200, payload={"query": {"categorymembers": []}}),
    ]
    session = _FakeSession(script=script)
    rate = _RateLimiter(min_delay=0.0)
    list(
        iter_category_files(
            "http://api.example/w/api.php",
            "Cars introduced in 1965",  # no prefix
            session=session,
            rate_limit=rate,
        )
    )
    params = session.calls[0][1]
    assert params is not None
    assert params["cmtitle"] == "Category:Cars introduced in 1965"


# --------------------------------------------------------- rate limiter


def test_rate_limiter_enforces_min_delay() -> None:
    """The clock + sleep stubs see a single ``sleep(remaining)`` call."""
    fake_now = {"t": 100.0}
    sleeps: list[float] = []

    def clock() -> float:
        return fake_now["t"]

    def sleep(s: float) -> None:
        sleeps.append(s)
        fake_now["t"] += s  # advance the clock as if we'd really slept

    rate = _RateLimiter(min_delay=1.0, clock=clock, sleep=sleep)
    # First call: no prior, no sleep.
    rate.wait()
    assert sleeps == []
    # Advance the clock by only 0.2 s (under the 1 s threshold).
    fake_now["t"] += 0.2
    rate.wait()
    # Should have slept for ~0.8 s.
    assert len(sleeps) == 1
    assert 0.79 <= sleeps[0] <= 0.81

    # Advance well past the threshold and call again.
    fake_now["t"] += 5.0
    rate.wait()
    assert len(sleeps) == 1  # no extra sleep needed


def test_rate_limiter_zero_disabled() -> None:
    """min_delay=0 short-circuits entirely (no sleep, no clock reads)."""
    sleeps: list[float] = []

    def clock() -> float:
        return 0.0

    def sleep(s: float) -> None:  # pragma: no cover - shouldn't be reached
        sleeps.append(s)

    rate = _RateLimiter(min_delay=0.0, clock=clock, sleep=sleep)
    for _ in range(5):
        rate.wait()
    assert sleeps == []


# --------------------------------------------------------- ingest_wikimedia


def _build_ingest_script(
    *,
    files: list[tuple[str, list[str], bytes]],
) -> list[_FakeResponse]:
    """Build the scripted responses for a single-seed ingest test.

    ``files`` is a list of ``(title, categories, body)`` tuples. The script
    returns:

    1. One categorymembers page listing the titles (no pagination).
    2. One imageinfo+categories metadata page (all titles batched).
    3. One image-download response per (title, body) pair.
    """
    cm_page = _FakeResponse(
        status_code=200,
        payload={
            "query": {
                "categorymembers": [
                    {"pageid": i + 1, "ns": 6, "title": title}
                    for i, (title, _, _) in enumerate(files)
                ]
            }
        },
    )
    pages: list[dict[str, Any]] = []
    for title, cats, body in files:
        pages.append(
            {
                "title": title,
                "categories": [{"title": c} for c in cats],
                "imageinfo": [
                    {
                        "url": f"https://upload.example/wiki/{title}",
                        "size": len(body),
                        "mime": "image/jpeg",
                    }
                ],
            }
        )
    meta_page = _FakeResponse(
        status_code=200,
        payload={"query": {"pages": pages}},
    )
    script = [cm_page, meta_page]
    for _, _, body in files:
        script.append(
            _FakeResponse(
                status_code=200,
                content=body,
                headers={"content-type": "image/jpeg"},
            )
        )
    return script


def test_ingest_smoke(
    db: sqlite3.Connection,
    output_dir: Path,
) -> None:
    """3 well-labelled files -> 3 listings + 3 images, all with source='wikimedia_commons'."""
    files = [
        (
            "File:Ford_Mustang_1965.jpg",
            ["Category:Cars introduced in 1965", "Category:Ford Mustang"],
            _png_bytes(1),
        ),
        (
            "File:Chevy_Chevelle_1972.jpg",
            ["Category:1972 Chevy", "Category:Chevrolet Chevelle"],
            _png_bytes(2),
        ),
        (
            "File:BMW_M3_1989.jpg",
            ["Category:Cars introduced in 1989", "Category:BMW M3"],
            _png_bytes(3),
        ),
    ]
    session = _FakeSession(script=_build_ingest_script(files=files))

    config = WikimediaIngestConfig(
        output_dir=output_dir,
        seed_categories=("Category:Test seed",),
        year_min=1900,
        year_max=1999,
        max_images_per_category=500,
        rate_limit_seconds=0.0,  # don't slow tests down
        split="train",
    )

    summary = ingest_wikimedia(conn=db, config=config, session=session)

    assert summary.processed == 3
    assert summary.listings_inserted == 3
    assert summary.images_inserted == 3
    assert summary.skipped_no_label == 0

    rows = listings.list_by_class(db, source="wikimedia_commons")
    assert len(rows) == 3
    by_make = {r.make: r for r in rows}
    assert set(by_make.keys()) == {"Ford", "Chevrolet", "BMW"}
    # Canonical fields populated via Phase 4.5 normalizer.
    assert by_make["Chevrolet"].canonical_make == "Chevrolet"
    assert by_make["BMW"].canonical_make == "BMW"
    # Year preserved.
    assert by_make["Ford"].year == 1965
    assert by_make["Chevrolet"].year == 1972
    assert by_make["BMW"].year == 1989
    # All rows tagged with the right split + source.
    for r in rows:
        assert r.split == "train"
        assert r.source == "wikimedia_commons"
        img_rows = images.list_for_listing(db, r.listing_id)
        assert len(img_rows) == 1
        assert Path(img_rows[0].local_path).exists()


def test_ingest_skips_unlabelled_rows(
    db: sqlite3.Connection,
    output_dir: Path,
) -> None:
    """Files without an extractable label get counted but not inserted."""
    files = [
        (
            "File:Ford_Mustang_1965.jpg",
            ["Category:Cars introduced in 1965", "Category:Ford Mustang"],
            _png_bytes(1),
        ),
        (
            "File:Random_photo.jpg",
            ["Category:Concept cars", "Category:Black and white photographs"],
            _png_bytes(2),
        ),
    ]
    # Only the FIRST file is expected to be downloaded; the second has no
    # label so the image fetch should be skipped. Truncate the image
    # responses accordingly.
    script = _build_ingest_script(files=files)
    # The downloader is only called once -- drop the second image response.
    script = script[:-1]
    session = _FakeSession(script=script)

    config = WikimediaIngestConfig(
        output_dir=output_dir,
        seed_categories=("Category:Test seed",),
        rate_limit_seconds=0.0,
    )
    summary = ingest_wikimedia(conn=db, config=config, session=session)

    assert summary.processed == 2
    assert summary.listings_inserted == 1
    assert summary.images_inserted == 1
    assert summary.skipped_no_label == 1


def test_ingest_dry_run_no_writes(
    db: sqlite3.Connection,
    output_dir: Path,
) -> None:
    """--dry-run leaves the DB and disk empty even with valid labels."""
    files = [
        (
            "File:Ford_Mustang_1965.jpg",
            ["Category:Cars introduced in 1965", "Category:Ford Mustang"],
            _png_bytes(1),
        ),
    ]
    # No image-fetch responses scripted -- dry-run mustn't try to download.
    script = _build_ingest_script(files=files)[:2]
    session = _FakeSession(script=script)

    config = WikimediaIngestConfig(
        output_dir=output_dir,
        seed_categories=("Category:Test seed",),
        rate_limit_seconds=0.0,
        dry_run=True,
    )
    summary = ingest_wikimedia(conn=db, config=config, session=session)

    assert summary.processed == 1
    assert summary.listings_inserted == 0
    assert summary.images_inserted == 0
    # No listings written.
    rows = listings.list_by_class(db, source="wikimedia_commons")
    assert rows == []
    # No image files written -- the output_dir may not even exist.
    if output_dir.exists():
        assert list(output_dir.rglob("*")) == []


def test_ingest_idempotent_skips_existing(
    db: sqlite3.Connection,
    output_dir: Path,
) -> None:
    """Second run on the same file is a clean skip-existing."""
    files = [
        (
            "File:Ford_Mustang_1965.jpg",
            ["Category:Cars introduced in 1965", "Category:Ford Mustang"],
            _png_bytes(1),
        ),
    ]
    config = WikimediaIngestConfig(
        output_dir=output_dir,
        seed_categories=("Category:Test seed",),
        rate_limit_seconds=0.0,
    )

    s1 = _FakeSession(script=_build_ingest_script(files=files))
    first = ingest_wikimedia(conn=db, config=config, session=s1)
    assert first.images_inserted == 1

    s2 = _FakeSession(script=_build_ingest_script(files=files))
    second = ingest_wikimedia(conn=db, config=config, session=s2)
    assert second.processed == 1
    assert second.images_inserted == 0
    assert second.skipped_existing == 1


def test_ingest_respects_limit(
    db: sqlite3.Connection,
    output_dir: Path,
) -> None:
    """--limit halts processing mid-batch."""
    # 3 files but limit=1 -- processing should stop after the first label
    # decision (still might process all 3 from the batch but only insert 1).
    files = [
        (
            "File:A.jpg",
            ["Category:Cars introduced in 1965", "Category:Ford Mustang"],
            _png_bytes(10),
        ),
        (
            "File:B.jpg",
            ["Category:Cars introduced in 1966", "Category:Ford Mustang"],
            _png_bytes(11),
        ),
        (
            "File:C.jpg",
            ["Category:Cars introduced in 1967", "Category:Ford Mustang"],
            _png_bytes(12),
        ),
    ]
    script = _build_ingest_script(files=files)
    # With limit=1 only one image is downloaded; trim image responses.
    script = script[:2] + script[2:3]
    session = _FakeSession(script=script)

    config = WikimediaIngestConfig(
        output_dir=output_dir,
        seed_categories=("Category:Test seed",),
        rate_limit_seconds=0.0,
        limit=1,
    )
    summary = ingest_wikimedia(conn=db, config=config, session=session)

    assert summary.processed == 1
    assert summary.listings_inserted == 1
    assert summary.images_inserted == 1


def test_ingest_records_source_wikimedia_commons(
    db: sqlite3.Connection,
    output_dir: Path,
) -> None:
    """The CHECK-widening migration was applied -- 'wikimedia_commons' is accepted."""
    files = [
        (
            "File:A.jpg",
            ["Category:Cars introduced in 1965", "Category:Ford Mustang"],
            _png_bytes(1),
        ),
    ]
    session = _FakeSession(script=_build_ingest_script(files=files))
    config = WikimediaIngestConfig(
        output_dir=output_dir,
        seed_categories=("Category:Test seed",),
        rate_limit_seconds=0.0,
    )
    ingest_wikimedia(conn=db, config=config, session=session)
    cur = db.execute("SELECT DISTINCT source FROM listings")
    assert {r["source"] for r in cur.fetchall()} == {"wikimedia_commons"}


# --------------------------------------------------------- summary type


def test_summary_is_pydantic_serializable() -> None:
    """The summary round-trips through model_dump for JSON reports."""
    s = WikimediaIngestSummary(processed=10, listings_inserted=5)
    d = s.model_dump()
    assert d["processed"] == 10
    assert d["listings_inserted"] == 5
    # All fields default to 0.
    for key in (
        "images_inserted",
        "skipped_existing",
        "skipped_no_label",
        "skipped_out_of_year_range",
        "skipped_unsupported_type",
        "skipped_download_failures",
        "api_errors",
    ):
        assert d[key] == 0


# --------------------------------------------------------- config validation


def test_config_rejects_year_min_gt_year_max(output_dir: Path) -> None:
    cfg = WikimediaIngestConfig(
        output_dir=output_dir,
        seed_categories=("Category:X",),
        year_min=2000,
        year_max=1999,
    )
    with pytest.raises(ValueError, match="year_min"):
        cfg.validated()


def test_config_rejects_empty_seed_categories(output_dir: Path) -> None:
    cfg = WikimediaIngestConfig(
        output_dir=output_dir,
        seed_categories=(),
    )
    with pytest.raises(ValueError, match="seed_category"):
        cfg.validated()


def test_config_rejects_zero_max_per_category(output_dir: Path) -> None:
    cfg = WikimediaIngestConfig(
        output_dir=output_dir,
        seed_categories=("Category:X",),
        max_images_per_category=0,
    )
    with pytest.raises(ValueError, match="max_images_per_category"):
        cfg.validated()


def test_config_rejects_negative_rate_limit(output_dir: Path) -> None:
    cfg = WikimediaIngestConfig(
        output_dir=output_dir,
        seed_categories=("Category:X",),
        rate_limit_seconds=-1.0,
    )
    with pytest.raises(ValueError, match="rate_limit_seconds"):
        cfg.validated()
