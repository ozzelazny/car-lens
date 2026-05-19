"""Tests for the Phase 5.1 zero-shot baseline harness.

The OpenCLIP backbone is replaced with a torch-based stub via
``monkeypatch`` so the tests never download real weights and never touch
the network. We do rely on a real ``torch`` install for tensor math --
torch is a runtime dependency of the project.

Strategy:

* Seed a tiny SQLite DB with 3 classes, 4 train images and 2 test images
  per class.
* Stub ``open_clip.create_model_and_transforms`` to return a model that
  maps each on-disk image to a pre-arranged embedding based on its filename
  prefix. The prefix encodes the class id (and is mirrored by the
  prototype direction we want) so we can compute the expected top-1 score
  by hand.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from car_lense_engine.dataset.canonical_labels import year_to_generation
from car_lense_engine.db import Image, Listing, images, listings, open_db
from car_lense_engine.eval import baseline as baseline_mod
from car_lense_engine.eval.baseline import (
    EXTERIOR_VIEWS,
    BaselineConfig,
    BaselineReport,
    build_prototypes,
    build_prototypes_by_view,
    class_id_for,
    evaluate,
)

torch = pytest.importorskip("torch")


# --------------------------------------------------------------- stub model


class _StubModel:
    """Stub of the OpenCLIP image-encoder surface used by the baseline.

    ``encode_image`` doesn't actually look at pixels; instead the stub's
    ``encode_image`` is wired to consult a ``path -> embedding`` table that
    the test populated. We use the preprocess step to smuggle the path
    through: the stub preprocess returns a (D+1,) tensor whose last slot
    encodes the path's index, and ``encode_image`` strips that off before
    returning the corresponding embedding.

    This keeps the test deterministic and order-independent.
    """

    def __init__(self, *, path_index: dict[str, int], embeddings: torch.Tensor) -> None:
        self._path_index = path_index
        self._embeddings = embeddings
        self.eval_called = False

    def eval(self) -> _StubModel:
        self.eval_called = True
        return self

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        # The stub preprocess encodes the path's row index in the last
        # element of the (1, D+1) preprocess output. ``batch`` here is
        # shape (N, D+1); the trailing column carries the index.
        indices = batch[:, -1].long().tolist()
        return self._embeddings[indices]


class _StubOpenClip:
    """Module-style stub installed into ``sys.modules['open_clip']``."""

    def __init__(self, model: _StubModel, preprocess: Any) -> None:
        self._model = model
        self._preprocess = preprocess

    def create_model_and_transforms(
        self,
        model_name: str,
        *,
        pretrained: str,
        device: str,
    ) -> tuple[_StubModel, None, Any]:
        # Capture call args for assertion if needed.
        self.last_call = {"model_name": model_name, "pretrained": pretrained, "device": device}
        return self._model, None, self._preprocess

    def get_tokenizer(self, model_name: str) -> Any:  # pragma: no cover -- unused here
        raise AssertionError("baseline must not call get_tokenizer")


def _install_stub_open_clip(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embeddings_by_path: dict[Path, torch.Tensor],
) -> _StubOpenClip:
    """Install an ``open_clip`` stub keyed by absolute path -> embedding.

    All embeddings must share the same dimensionality; we stack them once
    and let the stub's ``encode_image`` look up by index.
    """
    paths = list(embeddings_by_path.keys())
    path_index = {str(p): i for i, p in enumerate(paths)}
    if paths:
        embeds = torch.stack([embeddings_by_path[p] for p in paths], dim=0)
        embed_dim = int(embeds.shape[1])
    else:
        # Empty table -- the stub model will never be called, but it still
        # needs valid shape metadata for the preprocess to construct a
        # vector of the right size.
        embed_dim = 8
        embeds = torch.zeros((0, embed_dim))

    def stub_preprocess(img: Any) -> torch.Tensor:
        path_str: str = img._test_path  # type: ignore[attr-defined]
        idx = path_index[path_str]
        # Pack index into the last slot of a (D+1,)-vector.
        out = torch.zeros(embed_dim + 1)
        out[-1] = float(idx)
        return out

    model = _StubModel(path_index=path_index, embeddings=_pad_embeddings(embeds))
    stub = _StubOpenClip(model, stub_preprocess)
    fake_mod: Any = stub
    monkeypatch.setitem(sys.modules, "open_clip", fake_mod)
    return stub


def _pad_embeddings(embeds: torch.Tensor) -> torch.Tensor:
    """Append a zero column so the stub model can index by the trailing slot.

    The stub preprocess writes the path-index into the last column; we
    stash the real embedding in columns 0..D-1 and use the last column only
    as an index carrier. ``encode_image`` returns the row's embedding from
    a separate ``self._embeddings`` table, so the padding is just to keep
    the encoder's input shape consistent if anyone inspects it.
    """
    n, d = embeds.shape
    padded = torch.zeros((n, d + 1))
    padded[:, :d] = embeds
    return padded


# --------------------------------------------------------------- PIL patch


@pytest.fixture
def patch_pil_to_carry_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``PIL.Image.open`` attach ``_test_path`` to the returned image.

    The stub preprocess needs to know which path it's processing so the
    correct embedding can be returned. Real CLIP preprocesses don't need
    this -- they read pixels -- but the test stub does. We monkey-patch
    ``PIL.Image.open`` to record the path on the returned object inside a
    context manager that mimics the real signature.
    """
    from PIL import Image as PILImage

    original_open = PILImage.open

    def open_with_path(fp: Any, *args: Any, **kwargs: Any) -> Any:
        img = original_open(fp, *args, **kwargs)
        img._test_path = str(fp)  # type: ignore[attr-defined]
        # When the caller does ``img.convert(...)``, that returns a new
        # image; carry the path forward.
        original_convert = img.convert

        def convert_with_path(mode: str, *cargs: Any, **ckwargs: Any) -> Any:
            converted = original_convert(mode, *cargs, **ckwargs)
            converted._test_path = img._test_path
            return converted

        img.convert = convert_with_path  # type: ignore[method-assign]
        return img

    monkeypatch.setattr(PILImage, "open", open_with_path)


# --------------------------------------------------------------- fixtures


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "crawl.sqlite"


@pytest.fixture
def db(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = open_db(db_path)
    try:
        yield conn
    finally:
        conn.close()


def _make_image_file(path: Path) -> Path:
    """Write a tiny JPEG so PIL.open succeeds."""
    from PIL import Image as PILImage

    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (4, 4), color=(255, 255, 255)).save(path, format="JPEG")
    return path


def _seed_dataset(
    conn: sqlite3.Connection,
    *,
    tmp_path: Path,
    classes: list[tuple[int, str, str]],
    n_train_per_class: int = 4,
    n_test_per_class: int = 2,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Seed ``listings`` + ``images`` with ``classes`` * (train+test) rows.

    Returns ``(train_paths_by_class, test_paths_by_class)`` so tests can
    map paths back to their class id when building stub embeddings.
    """
    train_paths: dict[str, list[Path]] = {}
    test_paths: dict[str, list[Path]] = {}
    counter = 0
    for year, make, model in classes:
        # Phase 4.6: baseline keys the class id off the bucketed
        # ``generation_year``, not the raw calendar year. The test
        # class_id for retrieval is therefore derived from the bucket
        # start year (e.g. year=2007 -> generation_year=2004 ->
        # class_id "2004|hyundai|sonata").
        gen_year = year_to_generation(year)
        cid = class_id_for(gen_year, make, model)
        assert cid is not None
        train_paths[cid] = []
        test_paths[cid] = []
        for split_name, n, dest in (
            ("train", n_train_per_class, train_paths),
            ("test", n_test_per_class, test_paths),
        ):
            for i in range(n):
                counter += 1
                listing_id = f"stanford_cars:test_{counter:04d}"
                url = f"stanford_cars://{cid}/{counter:04d}"
                img_path = tmp_path / "imgs" / f"{cid}_{split_name}_{i}.jpg"
                _make_image_file(img_path)
                listings.insert_listing(
                    conn,
                    Listing(
                        listing_id=listing_id,
                        source="stanford_cars",
                        url=url,
                        year=year,
                        make=make,
                        model=model,
                        split=split_name,
                        # Phase 4.5 + 4.6: the baseline reads
                        # canonical_make / canonical_model and
                        # generation_year exclusively. Populating those
                        # here matches what the canonicalize-labels CLI
                        # would produce for Title Case + integer-year
                        # inputs.
                        canonical_make=make,
                        canonical_model=model,
                        generation_year=gen_year,
                    ),
                )
                image_id = f"{counter:064d}"
                images.insert_image(
                    conn,
                    Image(
                        image_id=image_id,
                        listing_id=listing_id,
                        source_url=url,
                        local_path=str(img_path),
                        position=1,
                    ),
                )
                # Phase 3.5: the baseline filters on ``images.split``
                # (migration 010), not ``listings.split``. Mirror the
                # listing-level split into the per-image column so the
                # fixture reflects what the make-splits CLI would have
                # produced. The listing-level value is left in place
                # for backwards compatibility but is no longer read.
                with conn:
                    conn.execute(
                        "UPDATE images SET split = ? WHERE image_id = ?",
                        (split_name, image_id),
                    )
                dest[cid].append(img_path)
    return train_paths, test_paths


# --------------------------------------------------------------- migration


def test_migration_adds_split_column(db: sqlite3.Connection) -> None:
    cur = db.execute("PRAGMA table_info(listings)")
    cols = {str(row["name"]) for row in cur.fetchall()}
    assert "split" in cols
    cur = db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_listings_source_split'"
    )
    assert cur.fetchone() is not None


def test_insert_listing_persists_split(db: sqlite3.Connection) -> None:
    listings.insert_listing(
        db,
        Listing(
            listing_id="stanford_cars:abc",
            source="stanford_cars",
            url="stanford_cars://x/abc",
            year=2012,
            make="acura",
            model="rl",
            split="test",
        ),
    )
    fetched = listings.get_listing(db, "stanford_cars:abc")
    assert fetched is not None
    assert fetched.split == "test"


# --------------------------------------------------------------- unit: class_id_for


def test_class_id_for_drops_nulls() -> None:
    # ``class_id_for`` is unchanged by Phase 4.6 -- it just formats
    # whatever integer it receives. The SQL callers now pass the
    # bucketed generation_year, but the function itself is agnostic.
    assert class_id_for(2012, "Acura", "RL") == "2012|acura|rl"
    assert class_id_for(None, "Acura", "RL") is None
    assert class_id_for(2012, None, "RL") is None
    assert class_id_for(2012, "Acura", None) is None


def test_class_id_for_drops_empty_strings() -> None:
    assert class_id_for(2012, "", "RL") is None
    assert class_id_for(2012, "Acura", "") is None


# --------------------------------------------------------------- _select_rows multi-source


def test_select_rows_multiple_sources(db_path: Path, tmp_path: Path) -> None:
    """``_select_rows`` filters by ``listings.source IN (...)`` when given a list.

    Seed three rows across three different sources (``compcars``,
    ``vmmrdb``, ``stanford_cars``). Passing ``["compcars", "vmmrdb"]``
    must return exactly those two rows; the ``stanford_cars`` row must
    be excluded. Passing a single source via the list form
    (``["compcars"]``) must still work (back-compat). And passing the
    legacy bare string still works (back-compat).
    """
    from car_lense_engine.eval.baseline import _select_rows

    # Seed three listings, each with one image in the train split.
    seeds = [
        ("compcars", 2012, "Honda", "Civic"),
        ("vmmrdb", 2012, "Toyota", "Camry"),
        ("stanford_cars", 2012, "Tesla", "Model S"),
    ]
    conn = open_db(db_path)
    try:
        for i, (src, year, make, model) in enumerate(seeds):
            gen_year = year_to_generation(year)
            listing_id = f"{src}:multi_{i:04d}"
            url = f"{src}://x/{i:04d}"
            img_path = tmp_path / "imgs" / f"{src}_{i}.jpg"
            _make_image_file(img_path)
            listings.insert_listing(
                conn,
                Listing(
                    listing_id=listing_id,
                    source=src,
                    url=url,
                    year=year,
                    make=make,
                    model=model,
                    split="train",
                    canonical_make=make,
                    canonical_model=model,
                    generation_year=gen_year,
                ),
            )
            image_id = f"{i:064d}"
            images.insert_image(
                conn,
                Image(
                    image_id=image_id,
                    listing_id=listing_id,
                    source_url=url,
                    local_path=str(img_path),
                    position=1,
                ),
            )
            with conn:
                conn.execute(
                    "UPDATE images SET split = ? WHERE image_id = ?",
                    ("train", image_id),
                )
    finally:
        conn.close()

    # Multi-source: list of two -> exactly two rows, stanford_cars excluded.
    conn = open_db(db_path)
    try:
        rows = _select_rows(conn, source=["compcars", "vmmrdb"], split="train")
    finally:
        conn.close()
    cids = sorted(cid for cid, _view, _path in rows)
    assert cids == sorted(
        [
            class_id_for(year_to_generation(2012), "Honda", "Civic"),
            class_id_for(year_to_generation(2012), "Toyota", "Camry"),
        ]
    )
    # stanford_cars (Tesla Model S) must not appear.
    tesla = class_id_for(year_to_generation(2012), "Tesla", "Model S")
    assert tesla not in cids

    # Single-source list -> identical to legacy bare-string call.
    conn = open_db(db_path)
    try:
        rows_list = _select_rows(conn, source=["compcars"], split="train")
        rows_str = _select_rows(conn, source="compcars", split="train")
    finally:
        conn.close()
    assert sorted(c for c, _v, _p in rows_list) == sorted(c for c, _v, _p in rows_str)
    assert len(rows_list) == 1

    # All three sources -> all three rows.
    conn = open_db(db_path)
    try:
        rows_all = _select_rows(conn, source=["compcars", "vmmrdb", "stanford_cars"], split="train")
    finally:
        conn.close()
    assert len(rows_all) == 3


def test_select_rows_rejects_empty_source_list(db_path: Path) -> None:
    """An empty source list raises ValueError -- there's nothing to filter on."""
    from car_lense_engine.eval.baseline import _select_rows

    open_db(db_path).close()  # apply migrations
    conn = open_db(db_path)
    try:
        with pytest.raises(ValueError):
            _select_rows(conn, source=[], split="train")
    finally:
        conn.close()


def test_select_rows_rejects_empty_string_entry(db_path: Path) -> None:
    """An empty / whitespace string entry raises ValueError."""
    from car_lense_engine.eval.baseline import _select_rows

    open_db(db_path).close()
    conn = open_db(db_path)
    try:
        with pytest.raises(ValueError):
            _select_rows(conn, source=["compcars", ""], split="train")
        with pytest.raises(ValueError):
            _select_rows(conn, source="   ", split="train")
    finally:
        conn.close()


# --------------------------------------------------------------- baseline e2e


def _basis_embedding(class_index: int, embed_dim: int = 8, lead: float = 5.0) -> torch.Tensor:
    """Return a (D,) vector with a strong lead at ``class_index`` and small noise elsewhere."""
    v = torch.full((embed_dim,), 0.1)
    v[class_index] = lead
    return v


def test_baseline_top1_matches_expected_fraction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """Three classes × 4 train + 2 test, perfect stub embeddings ⇒ 100% top-1."""
    classes = [(2012, "Acura", "RL"), (2007, "Hyundai", "Sonata"), (2012, "Tesla", "Model S")]
    conn = open_db(db_path)
    try:
        train_paths, test_paths = _seed_dataset(
            conn, tmp_path=tmp_path, classes=classes, n_train_per_class=4, n_test_per_class=2
        )
    finally:
        conn.close()

    # Build a per-path embedding table. Class i gets basis vector e_i for
    # every image (both train and test) -> prototypes are exactly e_i and
    # top-1 is 100%. Phase 4.6: class id uses the bucketed
    # ``generation_year``, not the raw calendar year.
    embeddings_by_path: dict[Path, torch.Tensor] = {}
    for i, (_, make, model) in enumerate(classes):
        cid = class_id_for(year_to_generation(classes[i][0]), make, model)
        assert cid is not None
        for p in train_paths[cid]:
            embeddings_by_path[p] = _basis_embedding(i)
        for p in test_paths[cid]:
            embeddings_by_path[p] = _basis_embedding(i)
    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    config = BaselineConfig(
        model_name="MobileCLIP-S2",
        pretrained="datacompdr",
        device="cpu",
        batch_size=4,
        top_ks=(1, 3, 5),
    )
    conn = open_db(db_path)
    try:
        prototypes = build_prototypes(
            conn=conn, config=config, source="stanford_cars", split="train"
        )
        report = evaluate(
            conn=conn,
            config=config,
            prototypes=prototypes,
            source="stanford_cars",
            split="test",
            per_class_top=10,
        )
    finally:
        conn.close()

    assert report.n_classes == 3
    assert report.n_test_images == 6
    assert report.n_train_images == 12
    assert report.overall["top_1"] == pytest.approx(1.0)
    assert report.overall["top_3"] == pytest.approx(1.0)
    # No confusion when every prediction is correct.
    assert report.confusion_top_pairs == []
    # All 3 classes appear in per_class. Note: Phase 4.6 buckets
    # year=2007 into bucket 2004-2007 (start year 2004), so the
    # Hyundai Sonata class id is "2004|hyundai|sonata".
    assert {m.class_id for m in report.per_class} == {
        "2012|acura|rl",
        "2004|hyundai|sonata",
        "2012|tesla|model s",
    }
    for m in report.per_class:
        assert m.top_1 == pytest.approx(1.0)
        assert m.n_train == 4
        assert m.n_test == 2


def test_baseline_partial_accuracy_and_confusion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """One test image from class 0 deliberately points at class 1 ⇒ top-1 = 5/6, one confusion."""
    classes = [(2012, "Acura", "RL"), (2007, "Hyundai", "Sonata"), (2012, "Tesla", "Model S")]
    conn = open_db(db_path)
    try:
        train_paths, test_paths = _seed_dataset(
            conn, tmp_path=tmp_path, classes=classes, n_train_per_class=4, n_test_per_class=2
        )
    finally:
        conn.close()

    embeddings_by_path: dict[Path, torch.Tensor] = {}
    # Phase 4.6: class id is keyed off generation_year, not raw year.
    cids = [class_id_for(year_to_generation(y), m, mo) for y, m, mo in classes]
    for i, cid in enumerate(cids):
        assert cid is not None
        for p in train_paths[cid]:
            embeddings_by_path[p] = _basis_embedding(i)
        for p in test_paths[cid]:
            embeddings_by_path[p] = _basis_embedding(i)

    # Sabotage one of class 0's test images: give it class 1's embedding.
    cid_0 = cids[0]
    assert cid_0 is not None
    saboteur = test_paths[cid_0][0]
    embeddings_by_path[saboteur] = _basis_embedding(1)

    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    config = BaselineConfig(
        model_name="MobileCLIP-S2",
        pretrained="datacompdr",
        device="cpu",
        batch_size=4,
        top_ks=(1, 3),
    )
    conn = open_db(db_path)
    try:
        prototypes = build_prototypes(
            conn=conn, config=config, source="stanford_cars", split="train"
        )
        report = evaluate(
            conn=conn,
            config=config,
            prototypes=prototypes,
            source="stanford_cars",
            split="test",
            per_class_top=10,
        )
    finally:
        conn.close()

    assert report.overall["top_1"] == pytest.approx(5.0 / 6.0)
    assert report.overall["top_3"] == pytest.approx(1.0)
    # Exactly one confusion: class 0 misclassified as class 1.
    assert len(report.confusion_top_pairs) == 1
    pair = report.confusion_top_pairs[0]
    assert pair.true_class == cids[0]
    assert pair.predicted_class == cids[1]
    assert pair.count == 1


def test_baseline_report_is_json_round_trippable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    classes = [(2012, "Acura", "RL"), (2007, "Hyundai", "Sonata")]
    conn = open_db(db_path)
    try:
        train_paths, test_paths = _seed_dataset(
            conn, tmp_path=tmp_path, classes=classes, n_train_per_class=2, n_test_per_class=1
        )
    finally:
        conn.close()

    embeddings_by_path: dict[Path, torch.Tensor] = {}
    for i, (y, m, mo) in enumerate(classes):
        # Phase 4.6: class id is keyed off generation_year.
        cid = class_id_for(year_to_generation(y), m, mo)
        assert cid is not None
        for p in train_paths[cid]:
            embeddings_by_path[p] = _basis_embedding(i)
        for p in test_paths[cid]:
            embeddings_by_path[p] = _basis_embedding(i)
    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    config = BaselineConfig(
        model_name="MobileCLIP-S2",
        pretrained="datacompdr",
        device="cpu",
        batch_size=2,
        top_ks=(1, 5),
    )
    conn = open_db(db_path)
    try:
        prototypes = build_prototypes(
            conn=conn, config=config, source="stanford_cars", split="train"
        )
        report = evaluate(
            conn=conn,
            config=config,
            prototypes=prototypes,
            source="stanford_cars",
            split="test",
        )
    finally:
        conn.close()

    payload = report.model_dump_json()
    round_tripped = BaselineReport.model_validate_json(payload)
    assert round_tripped == report

    # ``write_report`` writes the same content to disk + creates parents.
    target = tmp_path / "reports" / "phase5_baseline.json"
    baseline_mod.write_report(report, target)
    assert target.exists()
    on_disk = BaselineReport.model_validate_json(target.read_text(encoding="utf-8"))
    assert on_disk == report


# --------------------------------------------------------------- edge cases


def test_empty_test_set_returns_zeroed_overall(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """With train rows but no test rows, overall is all zeros and n_test_images=0."""
    classes = [(2012, "Acura", "RL")]
    conn = open_db(db_path)
    try:
        train_paths, _ = _seed_dataset(
            conn, tmp_path=tmp_path, classes=classes, n_train_per_class=2, n_test_per_class=0
        )
    finally:
        conn.close()

    embeddings_by_path: dict[Path, torch.Tensor] = {
        p: _basis_embedding(0)
        for p in train_paths[class_id_for(year_to_generation(2012), "Acura", "RL")]  # type: ignore[index]
    }
    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    config = BaselineConfig(
        model_name="MobileCLIP-S2",
        pretrained="datacompdr",
        device="cpu",
        batch_size=2,
        top_ks=(1, 5),
    )
    conn = open_db(db_path)
    try:
        prototypes = build_prototypes(
            conn=conn, config=config, source="stanford_cars", split="train"
        )
        report = evaluate(
            conn=conn,
            config=config,
            prototypes=prototypes,
            source="stanford_cars",
            split="test",
        )
    finally:
        conn.close()

    assert report.n_test_images == 0
    assert report.overall == {"top_1": 0.0, "top_5": 0.0}
    assert report.per_class == []


def test_missing_image_file_is_skipped_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A test row whose ``local_path`` doesn't exist on disk must be logged + skipped."""
    classes = [(2012, "Acura", "RL")]
    conn = open_db(db_path)
    try:
        train_paths, test_paths = _seed_dataset(
            conn, tmp_path=tmp_path, classes=classes, n_train_per_class=2, n_test_per_class=2
        )
        # Delete one test file from disk after seeding.
        cid = class_id_for(year_to_generation(2012), "Acura", "RL")
        assert cid is not None
        doomed = test_paths[cid][0]
        doomed.unlink()
    finally:
        conn.close()

    embeddings_by_path: dict[Path, torch.Tensor] = {}
    cid = class_id_for(year_to_generation(2012), "Acura", "RL")
    assert cid is not None
    for p in train_paths[cid]:
        embeddings_by_path[p] = _basis_embedding(0)
    # Only register the surviving test file; the deleted one shouldn't be
    # looked up at all.
    embeddings_by_path[test_paths[cid][1]] = _basis_embedding(0)
    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    config = BaselineConfig(
        model_name="MobileCLIP-S2",
        pretrained="datacompdr",
        device="cpu",
        batch_size=2,
        top_ks=(1,),
    )
    conn = open_db(db_path)
    try:
        with caplog.at_level("WARNING", logger="car_lense_engine.eval.baseline"):
            prototypes = build_prototypes(
                conn=conn, config=config, source="stanford_cars", split="train"
            )
            report = evaluate(
                conn=conn,
                config=config,
                prototypes=prototypes,
                source="stanford_cars",
                split="test",
            )
    finally:
        conn.close()

    # Only the surviving test image counted; top-1 should be 1.0 for it.
    assert report.n_test_images == 1
    assert report.overall["top_1"] == pytest.approx(1.0)
    assert any("skipping" in r.message for r in caplog.records if r.levelname == "WARNING")


def test_class_with_zero_train_excluded_from_prototypes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """A class present only in test has no prototype; its predictions miss."""
    # Two classes in train, third class only in test.
    train_classes = [(2012, "Acura", "RL"), (2007, "Hyundai", "Sonata")]
    test_only_class = (2012, "Tesla", "Model S")

    conn = open_db(db_path)
    try:
        train_paths, test_paths = _seed_dataset(
            conn,
            tmp_path=tmp_path,
            classes=train_classes,
            n_train_per_class=2,
            n_test_per_class=1,
        )
        # Add a test-only row for the third class. Phase 4.6: key the
        # class id off the bucketed generation_year.
        third_gen = year_to_generation(test_only_class[0])
        third_cid = class_id_for(third_gen, test_only_class[1], test_only_class[2])
        assert third_cid is not None
        third_path = tmp_path / "imgs" / f"{third_cid}_test_solo.jpg"
        _make_image_file(third_path)
        listings.insert_listing(
            conn,
            Listing(
                listing_id="stanford_cars:solo",
                source="stanford_cars",
                url="stanford_cars://solo/solo",
                year=test_only_class[0],
                make=test_only_class[1],
                model=test_only_class[2],
                split="test",
                canonical_make=test_only_class[1],
                canonical_model=test_only_class[2],
                generation_year=third_gen,
            ),
        )
        images.insert_image(
            conn,
            Image(
                image_id="s" * 64,
                listing_id="stanford_cars:solo",
                source_url="stanford_cars://solo/solo",
                local_path=str(third_path),
                position=1,
            ),
        )
        # Mirror the listing-level split into ``images.split`` (Phase 3.5).
        with conn:
            conn.execute(
                "UPDATE images SET split = ? WHERE image_id = ?",
                ("test", "s" * 64),
            )
    finally:
        conn.close()

    embeddings_by_path: dict[Path, torch.Tensor] = {}
    # Phase 4.6: class id is keyed off generation_year.
    cids = [class_id_for(year_to_generation(y), m, mo) for y, m, mo in train_classes]
    for i, cid in enumerate(cids):
        assert cid is not None
        for p in train_paths[cid]:
            embeddings_by_path[p] = _basis_embedding(i)
        for p in test_paths[cid]:
            embeddings_by_path[p] = _basis_embedding(i)
    # The orphan test row gets a unique direction unrelated to the train
    # prototypes -- it'll be misclassified.
    embeddings_by_path[third_path] = _basis_embedding(5)

    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    config = BaselineConfig(
        model_name="MobileCLIP-S2",
        pretrained="datacompdr",
        device="cpu",
        batch_size=2,
        top_ks=(1,),
    )
    conn = open_db(db_path)
    try:
        prototypes = build_prototypes(
            conn=conn, config=config, source="stanford_cars", split="train"
        )
        report = evaluate(
            conn=conn,
            config=config,
            prototypes=prototypes,
            source="stanford_cars",
            split="test",
        )
    finally:
        conn.close()

    # 2 in-prototype test rows + 1 orphan = 3 total; 2 hits.
    assert report.n_test_images == 3
    assert report.overall["top_1"] == pytest.approx(2.0 / 3.0)
    # Only the train classes have prototypes -> n_classes == 2.
    assert report.n_classes == 2


def test_baseline_uses_canonical_fields_not_raw(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """The baseline reads canonical_make / canonical_model, NOT raw make / model.

    We seed two rows whose RAW make differs from CANONICAL make and
    verify the class_id reported by the baseline matches the
    canonical-derived form, not the raw one.
    """
    # Raw make values are deliberately wrong / unmapped; canonical
    # values are the actually-correct ones. If baseline.py reads
    # the raw column, n_classes would split (two separate raw makes);
    # if it reads canonical, the two rows roll up into one class.
    cid_canonical = "2012|chevrolet|tahoe"
    conn = open_db(db_path)
    try:
        for i, raw_make in enumerate(["RAW1_DIFFERENT", "RAW2_ALSO_DIFFERENT"]):
            for split_name in ("train", "test"):
                listing_id = f"x-{i}-{split_name}"
                img_path = tmp_path / "imgs" / f"{listing_id}.jpg"
                _make_image_file(img_path)
                listings.insert_listing(
                    conn,
                    Listing(
                        listing_id=listing_id,
                        source="stanford_cars",
                        url=f"x://{listing_id}",
                        year=2012,
                        make=raw_make,
                        model=f"RAW_MODEL_{i}",
                        split=split_name,
                        canonical_make="Chevrolet",
                        canonical_model="Tahoe",
                        generation_year=2012,
                    ),
                )
                image_id = f"{i:030d}{split_name:>034}"
                images.insert_image(
                    conn,
                    Image(
                        image_id=image_id,
                        listing_id=listing_id,
                        source_url=f"x://{listing_id}",
                        local_path=str(img_path),
                        position=1,
                    ),
                )
                # Mirror the listing-level split into ``images.split``
                # (Phase 3.5).
                with conn:
                    conn.execute(
                        "UPDATE images SET split = ? WHERE image_id = ?",
                        (split_name, image_id),
                    )
    finally:
        conn.close()

    # All rows get the same embedding so the single canonical class is
    # 100% predictable.
    paths = sorted((tmp_path / "imgs").iterdir())
    embeddings_by_path: dict[Path, torch.Tensor] = {p: _basis_embedding(0) for p in paths}
    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    config = BaselineConfig(
        model_name="MobileCLIP-S2",
        pretrained="datacompdr",
        device="cpu",
        batch_size=2,
        top_ks=(1,),
    )
    conn = open_db(db_path)
    try:
        prototypes = build_prototypes(
            conn=conn, config=config, source="stanford_cars", split="train"
        )
        report = evaluate(
            conn=conn,
            config=config,
            prototypes=prototypes,
            source="stanford_cars",
            split="test",
        )
    finally:
        conn.close()

    # Single canonical class even though the raw makes differed.
    assert report.n_classes == 1
    assert {m.class_id for m in report.per_class} == {cid_canonical}


def test_no_train_rows_returns_empty_prototypes(
    monkeypatch: pytest.MonkeyPatch,
    db_path: Path,
) -> None:
    """If the train split is empty, build_prototypes returns ([], zeros((0,0)))."""
    open_db(db_path).close()  # apply migrations on a fresh DB
    _install_stub_open_clip(monkeypatch, embeddings_by_path={})

    config = BaselineConfig(
        model_name="MobileCLIP-S2",
        pretrained="datacompdr",
        device="cpu",
        batch_size=2,
    )
    conn = open_db(db_path)
    try:
        cids, proto = build_prototypes(
            conn=conn, config=config, source="stanford_cars", split="train"
        )
    finally:
        conn.close()
    assert cids == []
    assert proto.shape == (0, 0)


# --------------------------------------------------------------- per-view prototypes


def _seed_per_view_dataset(
    conn: sqlite3.Connection,
    *,
    tmp_path: Path,
    images_spec: list[tuple[str, str | None, int]],
) -> dict[Path, tuple[str, str | None]]:
    """Seed a tiny DB with one image per ``(class_id, view, class_index)`` entry.

    ``images_spec`` is a list of ``(class_id, view, class_index)`` triples.
    ``view`` may be one of the 5 exterior labels, ``"interior"`` for
    non-exterior, or ``None``. Returns a ``path -> (class_id, view)``
    map so the test can wire stub embeddings.
    """
    # Parse class_ids back to year/make/model.
    out: dict[Path, tuple[str, str | None]] = {}
    for offset, (class_id, view, _idx) in enumerate(images_spec):
        year_s, make_lower, model_lower = class_id.split("|")
        year = int(year_s)
        counter = offset + 1
        listing_id = f"stanford_cars:pv_{counter:04d}"
        url = f"stanford_cars://{class_id}/{counter:04d}"
        view_token = view if view is not None else "noview"
        img_path = tmp_path / "imgs" / f"{class_id}_{view_token}_{counter}.jpg"
        _make_image_file(img_path)
        listings.insert_listing(
            conn,
            Listing(
                listing_id=listing_id,
                source="stanford_cars",
                url=url,
                year=year,
                make=make_lower,
                model=model_lower,
                split="train",
                canonical_make=make_lower,
                canonical_model=model_lower,
                generation_year=year,
            ),
        )
        image_id = f"{counter:064d}"
        images.insert_image(
            conn,
            Image(
                image_id=image_id,
                listing_id=listing_id,
                source_url=url,
                local_path=str(img_path),
                position=1,
            ),
        )
        # ``insert_image`` doesn't write the ``view`` / ``split`` columns
        # (they're populated by Phase 3.3 / 3.5 CLIs), so push them via
        # UPDATE the same way the production pipeline does.
        with conn:
            conn.execute(
                "UPDATE images SET split = ?, view = ? WHERE image_id = ?",
                ("train", view, image_id),
            )
        out[img_path] = (class_id, view)
    return out


def test_build_prototypes_by_view(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """Per-view prototype builder buckets by ``(class, view)`` and zeros gaps.

    We seed three classes with varying view coverage:

    * Class A has both ``front`` and ``rear`` images.
    * Class B has only ``side`` images.
    * Class C has only ``front`` images.
    * One ``interior`` (non-exterior) row is included to verify it
      gets filtered out before embedding.
    * One row with ``view IS NULL`` is also included; same filter applies.

    Expected output:

    * ``class_ids`` are sorted across all three classes.
    * Each per-view tensor has ``(3, embed_dim)`` shape.
    * Rows for the present ``(class, view)`` cells have unit L2 norm.
    * Rows for missing cells are all-zero.
    * The total number of populated rows equals the number of
      ``(class, view)`` cells that had at least one exterior row.
    """
    cid_a = "2012|honda|civic"
    cid_b = "2012|toyota|camry"
    cid_c = "2012|mazda|3"
    # Build the seed list. Use 2 rows per cell to verify mean-pooling
    # actually happens (a single row trivially L2-normalizes).
    spec: list[tuple[str, str | None, int]] = [
        (cid_a, "front", 0),
        (cid_a, "front", 0),
        (cid_a, "rear", 0),
        (cid_a, "rear", 0),
        (cid_b, "side", 1),
        (cid_b, "side", 1),
        (cid_c, "front", 2),
        (cid_c, "front", 2),
        # Filtered out:
        (cid_a, "interior", 0),
        (cid_a, None, 0),
    ]
    conn = open_db(db_path)
    try:
        path_map = _seed_per_view_dataset(conn, tmp_path=tmp_path, images_spec=spec)
    finally:
        conn.close()

    # Embeddings: each (class, view) cell shares the same direction so
    # the per-row mean is just that direction. Distinct directions per
    # (class, view) so the L2-normalized rows are well-defined.
    direction_for = {
        (cid_a, "front"): _basis_embedding(0),
        (cid_a, "rear"): _basis_embedding(1),
        (cid_b, "side"): _basis_embedding(2),
        (cid_c, "front"): _basis_embedding(3),
    }
    embeddings_by_path: dict[Path, torch.Tensor] = {}
    for path, (cid, view) in path_map.items():
        if (cid, view) in direction_for:
            embeddings_by_path[path] = direction_for[(cid, view)]  # type: ignore[index]
        else:
            # Non-exterior / NULL rows shouldn't be embedded at all but
            # we still register a dummy so the stub's path_index covers
            # them (the production filter drops them before embedding).
            embeddings_by_path[path] = _basis_embedding(7)
    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    config = BaselineConfig(
        model_name="MobileCLIP-S2",
        pretrained="datacompdr",
        device="cpu",
        batch_size=4,
    )
    conn = open_db(db_path)
    try:
        class_ids, prototypes_by_view = build_prototypes_by_view(
            conn=conn, config=config, source="stanford_cars", split="train"
        )
    finally:
        conn.close()

    # Class ids are sorted; all 3 classes appear because each has at
    # least one exterior row.
    assert class_ids == sorted([cid_a, cid_b, cid_c])
    idx_a = class_ids.index(cid_a)
    idx_b = class_ids.index(cid_b)
    idx_c = class_ids.index(cid_c)

    # Every exterior view name must be present in the output dict,
    # even when no class has a prototype for it.
    assert set(prototypes_by_view.keys()) == set(EXTERIOR_VIEWS)

    embed_dim = int(prototypes_by_view["front"].shape[1])
    for view in EXTERIOR_VIEWS:
        tensor = prototypes_by_view[view]
        assert tensor.shape == (len(class_ids), embed_dim), view

    # Populated rows: norm == 1.0 (approximately, after L2-normalize).
    populated_cells = {
        ("front", idx_a),
        ("rear", idx_a),
        ("side", idx_b),
        ("front", idx_c),
    }
    for view in EXTERIOR_VIEWS:
        tensor = prototypes_by_view[view]
        for i in range(len(class_ids)):
            norm = float(tensor[i].norm().item())
            if (view, i) in populated_cells:
                assert norm == pytest.approx(1.0, abs=1e-5), (view, i, norm)
            else:
                assert norm == pytest.approx(0.0, abs=1e-7), (view, i, norm)

    # Three-quarter views had no training data: tensors must be all zeros.
    assert float(prototypes_by_view["three-quarter-front"].abs().sum().item()) == 0.0
    assert float(prototypes_by_view["three-quarter-rear"].abs().sum().item()) == 0.0


def test_build_prototypes_by_view_returns_cpu_tensors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """Regression: zero-fill rows for missing ``(class, view)`` cells used to
    be allocated on CPU while real prototypes lived on the embedding
    device (e.g. ``cuda:0``). ``torch.stack`` then crashed with::

        RuntimeError: Expected all tensors to be on the same device, but got
        tensors is on cuda:0, different from other tensors on cpu

    The fix routes the zero allocation through ``embeddings.device`` and
    moves the final per-view tensor to CPU before returning (consumers --
    prototype cache, recognize_api -- expect CPU tensors).

    We can't easily drive a CUDA tensor through the existing CPU-only
    stubs, but the contract that catches the bug is symmetric: assert
    every returned tensor lives on CPU. Combined with the existing
    ``test_build_prototypes_by_view`` coverage (which exercises the
    missing-cell path on multiple (class, view) combinations), this
    locks in the post-``.cpu()`` behaviour.
    """
    cid_a = "2012|honda|civic"
    cid_b = "2012|toyota|camry"
    # Class A has only ``front``; class B has only ``side``. Every other
    # exterior cell is missing and exercises the zero-fill path.
    spec: list[tuple[str, str | None, int]] = [
        (cid_a, "front", 0),
        (cid_a, "front", 0),
        (cid_b, "side", 1),
        (cid_b, "side", 1),
    ]
    conn = open_db(db_path)
    try:
        path_map = _seed_per_view_dataset(conn, tmp_path=tmp_path, images_spec=spec)
    finally:
        conn.close()

    direction_for = {
        (cid_a, "front"): _basis_embedding(0),
        (cid_b, "side"): _basis_embedding(1),
    }
    embeddings_by_path: dict[Path, torch.Tensor] = {
        path: direction_for[(cid, view)]  # type: ignore[index]
        for path, (cid, view) in path_map.items()
    }
    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    config = BaselineConfig(
        model_name="MobileCLIP-S2",
        pretrained="datacompdr",
        device="cpu",
        batch_size=2,
    )
    conn = open_db(db_path)
    try:
        class_ids, prototypes_by_view = build_prototypes_by_view(
            conn=conn, config=config, source="stanford_cars", split="train"
        )
    finally:
        conn.close()

    assert class_ids == sorted([cid_a, cid_b])
    # Every per-view tensor (populated or fully zero-filled) must live on
    # CPU after the ``.cpu()`` migration the fix introduces.
    for view in EXTERIOR_VIEWS:
        tensor = prototypes_by_view[view]
        assert tensor.device.type == "cpu", (view, tensor.device)


def test_build_prototypes_by_view_empty_when_no_exterior_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
) -> None:
    """When every row is non-exterior, the builder returns empty class_ids."""
    cid = "2012|honda|civic"
    spec: list[tuple[str, str | None, int]] = [
        (cid, "interior", 0),
        (cid, None, 0),
    ]
    conn = open_db(db_path)
    try:
        path_map = _seed_per_view_dataset(conn, tmp_path=tmp_path, images_spec=spec)
    finally:
        conn.close()

    embeddings_by_path: dict[Path, torch.Tensor] = {p: _basis_embedding(0) for p in path_map}
    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    config = BaselineConfig(
        model_name="MobileCLIP-S2",
        pretrained="datacompdr",
        device="cpu",
        batch_size=2,
    )
    conn = open_db(db_path)
    try:
        class_ids, prototypes_by_view = build_prototypes_by_view(
            conn=conn, config=config, source="stanford_cars", split="train"
        )
    finally:
        conn.close()

    assert class_ids == []
    assert set(prototypes_by_view.keys()) == set(EXTERIOR_VIEWS)
    for view in EXTERIOR_VIEWS:
        assert int(prototypes_by_view[view].shape[0]) == 0


# --------------------------------------------------------------- CLI smoke


def test_cli_rejects_missing_db(tmp_path: Path) -> None:
    from car_lense_engine.eval import cli as eval_cli

    missing = tmp_path / "nope.sqlite"
    with pytest.raises(SystemExit) as excinfo:
        eval_cli.main(["--db", str(missing)])
    assert excinfo.value.code == 2


def test_cli_writes_report_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    patch_pil_to_carry_path: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from car_lense_engine.eval import cli as eval_cli

    classes = [(2012, "Acura", "RL")]
    conn = open_db(db_path)
    try:
        train_paths, test_paths = _seed_dataset(
            conn, tmp_path=tmp_path, classes=classes, n_train_per_class=2, n_test_per_class=1
        )
    finally:
        conn.close()
    cid = class_id_for(year_to_generation(2012), "Acura", "RL")
    assert cid is not None
    embeddings_by_path: dict[Path, torch.Tensor] = {
        p: _basis_embedding(0) for p in (*train_paths[cid], *test_paths[cid])
    }
    _install_stub_open_clip(monkeypatch, embeddings_by_path=embeddings_by_path)

    output_path = tmp_path / "reports" / "p5.json"
    rc = eval_cli.main(
        [
            "--db",
            str(db_path),
            "--source",
            "stanford_cars",
            "--train-split",
            "train",
            "--test-split",
            "test",
            "--model",
            "MobileCLIP-S2",
            "--pretrained",
            "datacompdr",
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--output",
            str(output_path),
            "--per-class-top",
            "5",
        ]
    )
    assert rc == 0
    assert output_path.exists()
    on_disk = BaselineReport.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert on_disk.overall["top_1"] == pytest.approx(1.0)

    out = capsys.readouterr().out
    assert "top_1=1.000" in out
    assert "n_classes=1" in out
