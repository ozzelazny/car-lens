"""Tests for the OpenCLIP view labeler and the ``view-label`` CLI.

The OpenCLIP backbone is replaced with a tiny torch-based stub via
``monkeypatch`` so the tests never download real weights and never touch
the network. We do still rely on a real torch install for tensor math —
torch is a runtime dependency of the project, so tests can assume it's
importable.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from car_lense_engine.dataset import cli as view_cli
from car_lense_engine.dataset.view_labeler import (
    VIEW_NAMES,
    ViewLabel,
    ViewLabeler,
)
from car_lense_engine.db import Image, Listing, images, listings, open_db

torch = pytest.importorskip("torch")


# ----- stub OpenCLIP --------------------------------------------------- #


class _StubTokenizer:
    """A trivial tokenizer that just records the prompts it was handed."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, prompts: Sequence[str]) -> torch.Tensor:
        # Snapshot the prompts so tests can inspect ensemble construction.
        self.calls.append(list(prompts))
        # Encode each prompt as an integer index by stable position in the
        # cumulative call history; that index is what the stub model uses
        # to look up its prearranged text embedding.
        flat: list[str] = []
        for c in self.calls:
            flat.extend(c)
        token_ids = [flat.index(p) for p in prompts]
        return torch.tensor(token_ids, dtype=torch.long).unsqueeze(1)


class _StubModel:
    """A stub of the OpenCLIP nn.Module surface we use.

    * ``encode_text(tokens)`` returns a deterministic per-prompt embedding
      pulled from ``text_table`` (keyed by the prompt string the tokenizer
      saw at construction time). The labeler L2-normalizes whatever we
      return, so the tests verify the *direction* not the magnitude.
    * ``encode_image(batch)`` returns ``image_table[i]`` for the i-th image
      in the batch. The labeler chunks by ``batch_size`` so the indices
      reset across forward passes; the test fixture lays out one row per
      image_total ordering and feeds them in one chunk.
    """

    def __init__(
        self,
        *,
        tokenizer: _StubTokenizer,
        text_table: dict[str, torch.Tensor],
        image_rows: torch.Tensor,
    ) -> None:
        self._tokenizer = tokenizer
        self._text_table = text_table
        self._image_rows = image_rows
        self._image_cursor = 0
        self.eval_called = False
        # Mirror the OpenCLIP API: ``logit_scale`` is a learned parameter
        # stored as ``log(scale)``. We pin it to log(100.0) so the labeler's
        # cached scalar matches the previous hardcoded value and the
        # existing argmax / softmax-score expectations hold.
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(100.0)), requires_grad=False)

    def eval(self) -> _StubModel:
        self.eval_called = True
        return self

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        # The tokenizer stored each call's prompts in order; the most recent
        # call corresponds to *this* set of tokens.
        prompts = self._tokenizer.calls[-1]
        rows = [self._text_table[p] for p in prompts]
        return torch.stack(rows, dim=0)

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        n = batch.shape[0]
        out = self._image_rows[self._image_cursor : self._image_cursor + n]
        self._image_cursor += n
        return out


def _stub_preprocess(img: Any) -> torch.Tensor:
    """Stub preprocess: any PIL image becomes a 3x4x4 zero tensor."""
    return torch.zeros((3, 4, 4))


def _install_stub_open_clip(
    monkeypatch: pytest.MonkeyPatch,
    *,
    text_table: dict[str, torch.Tensor],
    image_rows: torch.Tensor,
) -> tuple[_StubModel, _StubTokenizer]:
    """Install a fake ``open_clip`` module in sys.modules.

    Returns the stub model + tokenizer so the test can poke at their state.
    """
    tokenizer = _StubTokenizer()
    model = _StubModel(
        tokenizer=tokenizer,
        text_table=text_table,
        image_rows=image_rows,
    )

    class _StubOpenClip:
        @staticmethod
        def create_model_and_transforms(
            model_name: str,
            *,
            pretrained: str,
            device: str,
        ) -> tuple[_StubModel, None, Any]:
            return model, None, _stub_preprocess

        @staticmethod
        def get_tokenizer(model_name: str) -> _StubTokenizer:
            return tokenizer

    monkeypatch.setitem(sys.modules, "open_clip", _StubOpenClip())
    return model, tokenizer


# ----- helpers --------------------------------------------------------- #


def _basis_text_table(winning_view: str, embed_dim: int = 8) -> dict[str, torch.Tensor]:
    """Build a per-prompt text embedding table.

    Every prompt for view ``v`` gets the v-th standard basis vector — so the
    per-view mean (and re-normalized result) is also that basis vector, and
    the dot product against the image embedding singles out the matching
    view cleanly.
    """
    from car_lense_engine.dataset.view_labeler import _VIEW_PROMPTS

    table: dict[str, torch.Tensor] = {}
    for view_index, view in enumerate(VIEW_NAMES):
        basis = torch.zeros(embed_dim)
        basis[view_index] = 1.0
        for prompt in _VIEW_PROMPTS[view]:
            table[prompt] = basis.clone()
    # ``winning_view`` is just used by callers to remember which axis they
    # want to peg high in the image embedding; the table itself is identical.
    assert winning_view in VIEW_NAMES
    return table


def _image_row_for_view(view: str, *, embed_dim: int = 8, lead: float = 5.0) -> torch.Tensor:
    """Build a single image embedding whose argmax cosine is ``view``."""
    idx = VIEW_NAMES.index(view)  # type: ignore[arg-type]
    row = torch.full((embed_dim,), 0.1)
    row[idx] = lead
    return row


def _make_image_file(tmp_path: Path, name: str) -> Path:
    """Write a 4x4 white JPEG so PIL.open succeeds."""
    from PIL import Image as PILImage

    path = tmp_path / name
    PILImage.new("RGB", (4, 4), color=(255, 255, 255)).save(path, format="JPEG")
    return path


# ----- fixtures -------------------------------------------------------- #


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


def _seed_listing_and_image(
    conn: sqlite3.Connection,
    *,
    image_id: str,
    local_path: Path,
    listing_id: str = "cars_com:1",
    source: str = "cars_com",
) -> None:
    # Insert listing if not already present.
    cur = conn.execute("SELECT 1 FROM listings WHERE listing_id = ?", (listing_id,))
    if cur.fetchone() is None:
        listings.insert_listing(
            conn,
            Listing(
                listing_id=listing_id,
                source=source,  # type: ignore[arg-type]
                url=f"https://example.com/{listing_id}",
                year=2020,
                make="Honda",
                model="Civic",
            ),
        )
    images.insert_image(
        conn,
        Image(
            image_id=image_id,
            listing_id=listing_id,
            source_url="https://example.com/photo.jpg",
            local_path=str(local_path),
            width=1024,
            height=768,
            bytes=1234,
            position=0,
        ),
    )


# ----- migration test -------------------------------------------------- #


def test_migration_adds_view_columns(db: sqlite3.Connection) -> None:
    cur = db.execute("PRAGMA table_info(images)")
    cols = {str(row["name"]) for row in cur.fetchall()}
    assert {"view", "view_score", "view_labeled_at"}.issubset(cols)
    # idx_images_view exists.
    cur = db.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_images_view'")
    assert cur.fetchone() is not None


# ----- unit tests for ViewLabeler ------------------------------------- #


def test_label_batch_picks_argmax_and_returns_softmax_score(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    text_table = _basis_text_table(winning_view="front")
    # Three images, one per view target.
    image_rows = torch.stack(
        [
            _image_row_for_view("front"),
            _image_row_for_view("rear"),
            _image_row_for_view("interior"),
        ]
    )
    _install_stub_open_clip(monkeypatch, text_table=text_table, image_rows=image_rows)

    paths = [_make_image_file(tmp_path, f"img_{i}.jpg") for i in range(3)]

    with ViewLabeler(batch_size=3) as labeler:
        results = labeler.label_batch(paths)

    # label_batch returns (path, ViewLabel) tuples, one per successfully
    # loaded image, in the same order as the input ``paths``.
    assert [p for p, _ in results] == paths
    assert [label.view for _, label in results] == ["front", "rear", "interior"]
    # softmax of (logit_scale=100 * cosine) for our lead-vs-noise vectors is
    # essentially 1.0; assert a comfortable margin.
    for _, label in results:
        assert label.score > 0.99
        assert 0.0 <= label.score <= 1.0


def test_multi_prompt_ensemble_uses_mean_of_text_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If the prompts for one view disagree on direction, the labeler should
    average them (then re-normalize) before scoring.

    Construction: give every "front" prompt the same basis vector except
    the last one, which is mirrored (negated). With three prompts, the mean
    of (e0, e0, -e0) = e0/3, pointing in the *same* direction as e0 after
    normalization — so an image embedding aligned with e0 should still pick
    "front" as the argmax. (This verifies we're averaging, not just using
    the first prompt.)
    """
    from car_lense_engine.dataset.view_labeler import _VIEW_PROMPTS

    text_table = _basis_text_table(winning_view="front")
    embed_dim = 8
    # Override "front" prompts: first two = +e0, third = -e0. Mean = (1/3) e0.
    front_prompts = _VIEW_PROMPTS["front"]
    assert len(front_prompts) == 3
    e0 = torch.zeros(embed_dim)
    e0[0] = 1.0
    text_table[front_prompts[0]] = e0.clone()
    text_table[front_prompts[1]] = e0.clone()
    text_table[front_prompts[2]] = -e0.clone()

    image_rows = torch.stack([_image_row_for_view("front")])
    _install_stub_open_clip(monkeypatch, text_table=text_table, image_rows=image_rows)

    paths = [_make_image_file(tmp_path, "img.jpg")]
    with ViewLabeler(batch_size=1) as labeler:
        results = labeler.label_batch(paths)
    assert results[0][0] == paths[0]
    assert results[0][1].view == "front"


def test_label_batch_handles_empty_paths_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel: list[str] = []

    class _BoomOpenClip:
        @staticmethod
        def create_model_and_transforms(*a: Any, **kw: Any) -> Any:
            sentinel.append("loaded")
            raise AssertionError("should not be called for empty input")

        @staticmethod
        def get_tokenizer(*a: Any, **kw: Any) -> Any:
            raise AssertionError("should not be called for empty input")

    monkeypatch.setitem(sys.modules, "open_clip", _BoomOpenClip())

    labeler = ViewLabeler()
    assert labeler.label_batch([]) == []
    assert sentinel == []


def test_close_releases_model_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    text_table = _basis_text_table(winning_view="front")
    image_rows = torch.stack([_image_row_for_view("front")])
    _install_stub_open_clip(monkeypatch, text_table=text_table, image_rows=image_rows)

    labeler = ViewLabeler(batch_size=1)
    labeler.label_batch([_make_image_file(tmp_path, "img.jpg")])
    assert labeler._model is not None
    assert labeler._text_embeds is not None
    labeler.close()
    assert labeler._model is None
    assert labeler._text_embeds is None


def test_context_manager_closes_on_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    text_table = _basis_text_table(winning_view="front")
    image_rows = torch.stack([_image_row_for_view("front")])
    _install_stub_open_clip(monkeypatch, text_table=text_table, image_rows=image_rows)

    with ViewLabeler(batch_size=1) as labeler:
        labeler.label_batch([_make_image_file(tmp_path, "img.jpg")])
        assert labeler._model is not None
    assert labeler._model is None
    assert labeler._text_embeds is None


def test_view_label_model_validates_view_name() -> None:
    # ``ViewLabel`` must reject unknown view names — the pydantic Literal
    # contract is part of the public schema.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ViewLabel(view="bogus", score=0.5)  # type: ignore[arg-type]


def test_label_batch_skips_missing_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-existent file mid-batch must be logged and skipped, not raise."""
    text_table = _basis_text_table(winning_view="front")
    # The stub yields one image row per successful encode_image call. Two
    # files exist on disk and survive _load_and_preprocess; one does not.
    image_rows = torch.stack(
        [
            _image_row_for_view("front"),
            _image_row_for_view("rear"),
        ]
    )
    _install_stub_open_clip(monkeypatch, text_table=text_table, image_rows=image_rows)

    good_a = _make_image_file(tmp_path, "ok_a.jpg")
    missing = tmp_path / "does_not_exist.jpg"
    good_b = _make_image_file(tmp_path, "ok_b.jpg")
    paths = [good_a, missing, good_b]

    with (
        caplog.at_level("WARNING", logger="car_lense_engine.dataset.view_labeler"),
        ViewLabeler(batch_size=3) as labeler,
    ):
        results = labeler.label_batch(paths)

    # Only the two existing paths come back, in input order.
    assert [p for p, _ in results] == [good_a, good_b]
    assert [label.view for _, label in results] == ["front", "rear"]
    # The missing path is logged at WARNING level with its path in the message.
    assert any(
        "skipping" in record.message and str(missing) in record.message
        for record in caplog.records
        if record.levelname == "WARNING"
    )


def test_label_batch_skips_corrupted_jpeg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A file that exists but isn't a valid image must be logged and skipped."""
    text_table = _basis_text_table(winning_view="front")
    # No images are expected to reach encode_image; pass an empty tensor.
    image_rows = torch.empty((0, 8))
    _install_stub_open_clip(monkeypatch, text_table=text_table, image_rows=image_rows)

    corrupted = tmp_path / "garbage.jpg"
    corrupted.write_bytes(b"this is not a jpeg, just some bytes")

    with (
        caplog.at_level("WARNING", logger="car_lense_engine.dataset.view_labeler"),
        ViewLabeler(batch_size=1) as labeler,
    ):
        results = labeler.label_batch([corrupted])

    assert results == []
    assert any(
        "skipping" in record.message and str(corrupted) in record.message
        for record in caplog.records
        if record.levelname == "WARNING"
    )


# ----- CLI integration tests ------------------------------------------ #


def test_cli_skips_already_labeled_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Seed DB: one labeled row, one unlabeled.
    img1 = _make_image_file(tmp_path, "a.jpg")
    img2 = _make_image_file(tmp_path, "b.jpg")
    conn = open_db(db_path)
    try:
        _seed_listing_and_image(conn, image_id="a" * 64, local_path=img1)
        _seed_listing_and_image(conn, image_id="b" * 64, local_path=img2)
        with conn:
            conn.execute(
                "UPDATE images SET view = 'rear', view_score = 0.9, "
                "view_labeled_at = CURRENT_TIMESTAMP WHERE image_id = ?",
                ("a" * 64,),
            )
    finally:
        conn.close()

    text_table = _basis_text_table(winning_view="front")
    image_rows = torch.stack([_image_row_for_view("front")])
    _install_stub_open_clip(monkeypatch, text_table=text_table, image_rows=image_rows)

    rc = view_cli.main(["--db", str(db_path), "--batch-size", "4"])
    assert rc == 0

    conn = open_db(db_path)
    try:
        a_row = conn.execute("SELECT view FROM images WHERE image_id = ?", ("a" * 64,)).fetchone()
        b_row = conn.execute("SELECT view FROM images WHERE image_id = ?", ("b" * 64,)).fetchone()
    finally:
        conn.close()
    # Pre-existing label preserved; new row labeled.
    assert a_row["view"] == "rear"
    assert b_row["view"] == "front"
    out = capsys.readouterr().out
    assert "labeled 1" in out


def test_cli_rebuild_relabels_everything(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
) -> None:
    img1 = _make_image_file(tmp_path, "a.jpg")
    img2 = _make_image_file(tmp_path, "b.jpg")
    conn = open_db(db_path)
    try:
        _seed_listing_and_image(conn, image_id="a" * 64, local_path=img1)
        _seed_listing_and_image(conn, image_id="b" * 64, local_path=img2)
        # Both rows already labeled with stale data.
        with conn:
            conn.execute(
                "UPDATE images SET view = 'detail', view_score = 0.5, "
                "view_labeled_at = CURRENT_TIMESTAMP"
            )
    finally:
        conn.close()

    text_table = _basis_text_table(winning_view="rear")
    # Both images point at "rear".
    image_rows = torch.stack([_image_row_for_view("rear"), _image_row_for_view("rear")])
    _install_stub_open_clip(monkeypatch, text_table=text_table, image_rows=image_rows)

    rc = view_cli.main(["--db", str(db_path), "--rebuild", "--batch-size", "4"])
    assert rc == 0

    conn = open_db(db_path)
    try:
        rows = conn.execute("SELECT image_id, view FROM images ORDER BY image_id").fetchall()
    finally:
        conn.close()
    assert [r["view"] for r in rows] == ["rear", "rear"]


def test_cli_filters_by_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
) -> None:
    img1 = _make_image_file(tmp_path, "a.jpg")
    img2 = _make_image_file(tmp_path, "b.jpg")
    conn = open_db(db_path)
    try:
        _seed_listing_and_image(
            conn,
            image_id="a" * 64,
            local_path=img1,
            listing_id="cars_com:1",
            source="cars_com",
        )
        _seed_listing_and_image(
            conn,
            image_id="b" * 64,
            local_path=img2,
            listing_id="bat:1",
            source="bat",
        )
    finally:
        conn.close()

    text_table = _basis_text_table(winning_view="side")
    image_rows = torch.stack([_image_row_for_view("side")])
    _install_stub_open_clip(monkeypatch, text_table=text_table, image_rows=image_rows)

    rc = view_cli.main(["--db", str(db_path), "--source", "cars_com", "--batch-size", "4"])
    assert rc == 0

    conn = open_db(db_path)
    try:
        a_row = conn.execute("SELECT view FROM images WHERE image_id = ?", ("a" * 64,)).fetchone()
        b_row = conn.execute("SELECT view FROM images WHERE image_id = ?", ("b" * 64,)).fetchone()
    finally:
        conn.close()
    assert a_row["view"] == "side"
    assert b_row["view"] is None  # bat source untouched


def test_cli_no_work_no_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    db_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # No rows at all in DB → CLI must short-circuit before touching open_clip.
    open_db(db_path).close()  # apply migrations on a fresh DB

    class _BoomOpenClip:
        @staticmethod
        def create_model_and_transforms(*a: Any, **kw: Any) -> Any:
            raise AssertionError("model should not load when there's nothing to do")

        @staticmethod
        def get_tokenizer(*a: Any, **kw: Any) -> Any:
            raise AssertionError("tokenizer should not load when there's nothing to do")

    monkeypatch.setitem(sys.modules, "open_clip", _BoomOpenClip())

    rc = view_cli.main(["--db", str(db_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "nothing to do" in out


def test_cli_rejects_missing_db(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "nope.sqlite"
    with pytest.raises(SystemExit) as excinfo:
        view_cli.main(["--db", str(missing)])
    assert excinfo.value.code == 2
