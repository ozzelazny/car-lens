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

from car_lense_engine.db import Image, Listing, images, listings, open_db
from car_lense_engine.eval import baseline as baseline_mod
from car_lense_engine.eval.baseline import (
    BaselineConfig,
    BaselineReport,
    build_prototypes,
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
        cid = class_id_for(year, make, model)
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
    assert class_id_for(2012, "Acura", "RL") == "2012|acura|rl"
    assert class_id_for(None, "Acura", "RL") is None
    assert class_id_for(2012, None, "RL") is None
    assert class_id_for(2012, "Acura", None) is None


def test_class_id_for_drops_empty_strings() -> None:
    assert class_id_for(2012, "", "RL") is None
    assert class_id_for(2012, "Acura", "") is None


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
    # top-1 is 100%.
    embeddings_by_path: dict[Path, torch.Tensor] = {}
    for i, (_, make, model) in enumerate(classes):
        cid = class_id_for(classes[i][0], make, model)
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
    # All 3 classes appear in per_class.
    assert {m.class_id for m in report.per_class} == {
        "2012|acura|rl",
        "2007|hyundai|sonata",
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
    cids = [class_id_for(y, m, mo) for y, m, mo in classes]
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
        cid = class_id_for(y, m, mo)
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
        for p in train_paths[class_id_for(2012, "Acura", "RL")]  # type: ignore[index]
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
        cid = class_id_for(2012, "Acura", "RL")
        assert cid is not None
        doomed = test_paths[cid][0]
        doomed.unlink()
    finally:
        conn.close()

    embeddings_by_path: dict[Path, torch.Tensor] = {}
    cid = class_id_for(2012, "Acura", "RL")
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
        # Add a test-only row for the third class.
        third_cid = class_id_for(*test_only_class)
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
    finally:
        conn.close()

    embeddings_by_path: dict[Path, torch.Tensor] = {}
    cids = [class_id_for(y, m, mo) for y, m, mo in train_classes]
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
    cid = class_id_for(2012, "Acura", "RL")
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
