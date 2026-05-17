"""Tests for the CompCars label resolver.

The .mat-loading tests build a real scipy.io.savemat payload in-memory so we
exercise the same loader-and-coercion path that production hits, without
needing the 16.5 GB ZIP on disk.
"""

from __future__ import annotations

import io

import pytest

# Skip the whole module if scipy isn't available — the production code
# lazily imports it and only fails when actually invoked.
scipy_io = pytest.importorskip("scipy.io")
np = pytest.importorskip("numpy")

from car_lense_engine.dataset.compcars_labels import (  # noqa: E402
    CompCarsBodyTypeTable,
    CompCarsLabelError,
    CompCarsNameTable,
    parse_image_path,
)

# --------------------------------------------------------- path parsing


def test_parse_image_path_happy_path() -> None:
    make_id, model_id, year = parse_image_path("image/12/345/2014/abc123.jpg")
    assert make_id == 12
    assert model_id == 345
    assert year == 2014


def test_parse_image_path_accepts_long_sha_filename() -> None:
    # The on-disk filename is a 32-char hex (the CompCars 'sha' token is
    # actually an MD5 in the original release). Verify the path regex
    # doesn't impose a length restriction on the leaf.
    path = "image/1/2/2010/" + "a" * 32 + ".jpg"
    assert parse_image_path(path) == (1, 2, 2010)


def test_parse_image_path_rejects_bad_format() -> None:
    with pytest.raises(CompCarsLabelError):
        parse_image_path("not/a/compcars/path.jpg")
    with pytest.raises(CompCarsLabelError):
        parse_image_path("image/12/345/2014/abc.png")  # wrong extension
    with pytest.raises(CompCarsLabelError):
        parse_image_path("image/12/345/abc.jpg")  # missing year segment


def test_parse_image_path_rejects_nan_year() -> None:
    with pytest.raises(CompCarsLabelError):
        parse_image_path("image/12/345/nan/abc.jpg")


def test_parse_image_path_rejects_5008_sentinel() -> None:
    # Documented CompCars sentinel for "year unknown" — treat as NaN.
    with pytest.raises(CompCarsLabelError):
        parse_image_path("image/12/345/5008/abc.jpg")


def test_parse_image_path_rejects_year_out_of_range() -> None:
    with pytest.raises(CompCarsLabelError):
        parse_image_path("image/12/345/1899/abc.jpg")
    with pytest.raises(CompCarsLabelError):
        parse_image_path("image/12/345/2100/abc.jpg")


def test_parse_image_path_rejects_non_positive_ids() -> None:
    # The regex only allows digits, but 0 should still be rejected as a
    # MATLAB-1-based-index violation.
    with pytest.raises(CompCarsLabelError):
        parse_image_path("image/0/345/2014/abc.jpg")
    with pytest.raises(CompCarsLabelError):
        parse_image_path("image/12/0/2014/abc.jpg")


def test_parse_image_path_rejects_empty() -> None:
    with pytest.raises(CompCarsLabelError):
        parse_image_path("")


# --------------------------------------------------------- name table


def _build_name_mat(makes: list[str], models: list[str]) -> bytes:
    """Build a make_model_name.mat-shaped byte blob via scipy.io.savemat."""
    buf = io.BytesIO()
    # Use ``object`` arrays so scipy writes them as MATLAB cell arrays.
    make_arr = np.array(makes, dtype=object).reshape(-1, 1)
    model_arr = np.array(models, dtype=object).reshape(-1, 1)
    scipy_io.savemat(buf, {"make_names": make_arr, "model_names": model_arr})
    return buf.getvalue()


def test_name_table_resolves_known_ids() -> None:
    mat = _build_name_mat(
        makes=["Audi", "BMW", "Mercedes-Benz"],
        models=["A4", "3 Series", "C-Class", "5 Series"],
    )
    table = CompCarsNameTable(mat)
    assert table.resolve(1, 1) == ("Audi", "A4")
    assert table.resolve(2, 2) == ("BMW", "3 Series")
    assert table.resolve(3, 3) == ("Mercedes-Benz", "C-Class")


def test_name_table_rejects_out_of_range_ids() -> None:
    mat = _build_name_mat(makes=["Audi"], models=["A4"])
    table = CompCarsNameTable(mat)
    with pytest.raises(CompCarsLabelError):
        table.resolve(0, 1)
    with pytest.raises(CompCarsLabelError):
        table.resolve(2, 1)  # make_id out of range
    with pytest.raises(CompCarsLabelError):
        table.resolve(1, 0)
    with pytest.raises(CompCarsLabelError):
        table.resolve(1, 2)  # model_id out of range


def test_name_table_rejects_empty_cell() -> None:
    # An empty string in the cell should fail the resolve check.
    mat = _build_name_mat(makes=["", "BMW"], models=["A4", ""])
    table = CompCarsNameTable(mat)
    with pytest.raises(CompCarsLabelError):
        table.resolve(1, 1)  # empty make
    with pytest.raises(CompCarsLabelError):
        table.resolve(2, 2)  # empty model


def test_name_table_rejects_missing_fields() -> None:
    buf = io.BytesIO()
    scipy_io.savemat(buf, {"unrelated": np.array([1, 2, 3])})
    with pytest.raises(CompCarsLabelError):
        CompCarsNameTable(buf.getvalue())


def test_name_table_reports_counts() -> None:
    mat = _build_name_mat(makes=["A", "B"], models=["X", "Y", "Z"])
    table = CompCarsNameTable(mat)
    assert len(table) == 2
    assert table.model_count == 3


# --------------------------------------------------------- body-type table


def _build_body_mat(car_types: list[str], model_to_type: list[int]) -> bytes:
    """Build a car_type.mat-shaped byte blob via scipy.io.savemat."""
    buf = io.BytesIO()
    car_arr = np.array(car_types, dtype=object).reshape(1, -1)
    mt_arr = np.array(model_to_type, dtype=np.int32).reshape(-1, 1)
    scipy_io.savemat(buf, {"car_type": car_arr, "model_type": mt_arr})
    return buf.getvalue()


def test_body_type_resolves_known_model() -> None:
    mat = _build_body_mat(
        car_types=["MPV", "SUV", "sedan", "hatchback"],
        model_to_type=[3, 2, 0, 4],
    )
    table = CompCarsBodyTypeTable(mat)
    assert table.resolve(1) == "sedan"
    assert table.resolve(2) == "SUV"
    assert table.resolve(4) == "hatchback"


def test_body_type_returns_none_for_unknown() -> None:
    mat = _build_body_mat(
        car_types=["MPV", "SUV", "sedan"],
        model_to_type=[1, 0, 2],
    )
    table = CompCarsBodyTypeTable(mat)
    # ``0`` in model_type means unknown -> None.
    assert table.resolve(2) is None


def test_body_type_returns_none_for_out_of_range_model() -> None:
    mat = _build_body_mat(car_types=["sedan"], model_to_type=[1])
    table = CompCarsBodyTypeTable(mat)
    assert table.resolve(0) is None
    assert table.resolve(2) is None
    assert table.resolve(-5) is None


def test_body_type_returns_none_for_bad_type_index() -> None:
    # A model_type value greater than len(car_type) should resolve to None
    # rather than crash — defensive against truncated tables.
    mat = _build_body_mat(car_types=["sedan"], model_to_type=[99])
    table = CompCarsBodyTypeTable(mat)
    assert table.resolve(1) is None


def test_body_type_accepts_model2type_alias() -> None:
    """Some CompCars snapshots use ``model2type`` instead of ``model_type``."""
    buf = io.BytesIO()
    car_arr = np.array(["sedan", "SUV"], dtype=object).reshape(1, -1)
    mt_arr = np.array([2, 1], dtype=np.int32).reshape(-1, 1)
    scipy_io.savemat(buf, {"car_type": car_arr, "model2type": mt_arr})
    table = CompCarsBodyTypeTable(buf.getvalue())
    assert table.resolve(1) == "SUV"
    assert table.resolve(2) == "sedan"


def test_body_type_handles_missing_fields_as_noop() -> None:
    """A .mat file with neither car_type/types nor model_type still loads.

    The live CUHK release of car_type.mat omits the model_type lookup
    array entirely; the resolver must degrade to a no-op instead of
    raising at construction time.
    """
    buf = io.BytesIO()
    scipy_io.savemat(buf, {"unrelated": np.array([1, 2, 3])})
    table = CompCarsBodyTypeTable(buf.getvalue())
    assert table.resolve(1) is None
    assert table.resolve(999) is None


def test_body_type_table_handles_only_types_field() -> None:
    """Real-world car_type.mat has only 'types' — no model_type lookup.

    Resolver should return ``None`` for every input.
    """
    buf = io.BytesIO()
    scipy_io.savemat(buf, {"types": np.array(["MPV", "SUV", "sedan"], dtype=object)})
    table = CompCarsBodyTypeTable(buf.getvalue())
    assert table.resolve(1) is None
    assert table.resolve(999) is None


def test_body_type_table_accepts_types_field_with_model_type() -> None:
    """If a future mirror ships both ``types`` and ``model_type``, resolution works."""
    buf = io.BytesIO()
    types_arr = np.array(["MPV", "SUV", "sedan"], dtype=object).reshape(1, -1)
    mt_arr = np.array([3, 1, 2], dtype=np.int32).reshape(-1, 1)
    scipy_io.savemat(buf, {"types": types_arr, "model_type": mt_arr})
    table = CompCarsBodyTypeTable(buf.getvalue())
    assert table.resolve(1) == "sedan"
    assert table.resolve(2) == "MPV"
    assert table.resolve(3) == "SUV"
