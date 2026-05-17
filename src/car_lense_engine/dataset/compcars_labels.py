"""CompCars label resolver (Phase 4.3).

CompCars (Yang et al., CUHK, 2015) stores per-image metadata via integer IDs
embedded in the on-disk path. Each image lives at:

    image/<make_id>/<model_id>/<year>/<sha>.jpg

where ``make_id`` and ``model_id`` are 1-based integer indices into two
look-up tables shipped under ``misc/``:

* ``misc/make_model_name.mat`` — a MATLAB v5 file containing:
    - ``make_names``  : 163×1 cell-array of make-name strings.
    - ``model_names`` : N×1 cell-array (N≈1716) of model-name strings.
  Both arrays are **1-indexed** in the dataset's path convention (the
  MATLAB convention), so ``make_id = 1`` maps to ``make_names[0]`` in
  Python after we load via ``scipy.io.loadmat(squeeze_me=True)``.

* ``misc/car_type.mat`` — a MATLAB v5 file containing:
    - ``car_type``  : 1×12 (or 12×1) cell-array of body-style name strings,
      e.g. ``["MPV", "SUV", "sedan", "hatchback", "minibus", "fastback",
      "estate", "pickup", "hardtop convertible", "sports", "crossover",
      "convertible"]``.
    - ``model_type`` : N×1 integer array mapping each ``model_id`` to a
      body-style index (1-based) or ``0`` when unknown.

The .mat files vary slightly across packaging snapshots (squeezed vs
unsqueezed, ``model_type`` vs ``model2type`` field name); the loaders here
duck-type the loaded structure so subtle re-packagings still parse.

Path year handling: a small fraction of CompCars images encode the year as
NaN-equivalent strings ("nan") or as the sentinel ``"5008"``. Those are
rejected at the path-parse boundary so the ingest can count + skip them
cleanly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# --- Public types ----------------------------------------------------------


class CompCarsLabelError(ValueError):
    """Raised when a CompCars image path or label table cannot be resolved."""


@dataclass(frozen=True)
class CompCarsLabel:
    """Structured (year, make, model, body_style) parsed for one image."""

    year: int
    make: str
    model: str
    body_style: str | None
    raw_path: str


# --- Path parsing ----------------------------------------------------------


# Match the canonical CompCars in-ZIP path: image/<make_id>/<model_id>/<year>/<sha>.jpg
# year may be "nan" / "5008" (we tolerate the parse and let callers filter).
_IMAGE_PATH_RE = re.compile(
    r"^image/(?P<make_id>\d+)/(?P<model_id>\d+)/(?P<year>[^/]+)/(?P<sha>[^/]+)\.jpg$"
)

# Years 1900-2099 are accepted; anything else (including "5008", "nan", "")
# is treated as a missing year and triggers a CompCarsLabelError below.
_VALID_YEAR_RE = re.compile(r"^\d{4}$")

# The CompCars README documents "5008" as a placeholder sentinel for "year
# unknown" appearing in a handful of paths. Treat it identically to NaN.
_YEAR_SENTINELS: frozenset[str] = frozenset({"5008", "nan", "NaN", "NAN", ""})


def parse_image_path(path: str) -> tuple[int, int, int]:
    """Parse ``image/<make_id>/<model_id>/<year>/<sha>.jpg`` into integers.

    Returns the triple ``(make_id, model_id, year)``. Raises
    :class:`CompCarsLabelError` when:

    * the path doesn't match the canonical layout,
    * any of ``make_id`` / ``model_id`` is not a positive integer,
    * the year token is missing / NaN / the ``5008`` sentinel, or
    * the year is not in a sane historical range (1900-2099).
    """
    if not isinstance(path, str) or not path:
        raise CompCarsLabelError(f"empty or non-string path: {path!r}")

    m = _IMAGE_PATH_RE.match(path)
    if m is None:
        raise CompCarsLabelError(f"not a canonical CompCars image path: {path!r}")

    make_token = m.group("make_id")
    model_token = m.group("model_id")
    year_token = m.group("year")

    try:
        make_id = int(make_token)
        model_id = int(model_token)
    except ValueError as exc:  # pragma: no cover - regex already restricts to \d+
        raise CompCarsLabelError(f"non-integer id in path: {path!r}") from exc

    if make_id <= 0 or model_id <= 0:
        raise CompCarsLabelError(f"non-positive id in path: {path!r}")

    if year_token in _YEAR_SENTINELS or not _VALID_YEAR_RE.match(year_token):
        raise CompCarsLabelError(f"unusable year token {year_token!r} in path: {path!r}")

    year = int(year_token)
    if year < 1900 or year > 2099:
        raise CompCarsLabelError(f"year out of range in path: {path!r}")

    return make_id, model_id, year


# --- .mat loaders ----------------------------------------------------------


def _load_mat(mat_bytes: bytes) -> dict[str, Any]:
    """Lazy-load a MATLAB .mat byte blob via scipy.io.loadmat.

    ``squeeze_me=True`` collapses singleton dims (so a 163×1 cell array comes
    out as a 1-D numpy ``object`` array of bare strings). ``struct_as_record
    =False`` would give us attribute-style structs; we don't need it because
    CompCars uses top-level arrays, not nested structs.
    """
    try:
        from io import BytesIO  # noqa: PLC0415

        from scipy.io import loadmat  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dep listed in pyproject
        raise RuntimeError(
            "scipy is required to load CompCars .mat label tables; "
            "install it via `uv pip install -e .`"
        ) from exc
    result: dict[str, Any] = loadmat(BytesIO(mat_bytes), squeeze_me=True)
    return result


def _coerce_str(cell_value: Any) -> str | None:
    """Best-effort coercion of a scipy-loaded cell entry to a python str.

    scipy.io.loadmat returns cell strings as ``numpy.str_`` (after squeeze),
    or as 0-d / 1-element ``numpy.ndarray`` for awkward layouts. Empty cells
    come through as zero-length arrays. Returns ``None`` for empty / blank.
    """
    if cell_value is None:
        return None
    # numpy.str_ / str both pass through cleanly.
    if isinstance(cell_value, str):
        s = cell_value.strip()
        return s or None
    # 0-d or 1-element ndarrays carrying a string-like scalar.
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:  # pragma: no cover - numpy is required by scipy
        np = None  # type: ignore[assignment]
    if np is not None and isinstance(cell_value, np.ndarray):
        if cell_value.size == 0:
            return None
        if cell_value.size == 1:
            return _coerce_str(cell_value.item())
    # Fall back: stringify and hope.
    s = str(cell_value).strip()
    return s or None


def _to_1d_list(arr: Any) -> list[Any]:
    """Flatten a (possibly 2-D) cell array into a 1-D python list.

    Handles 163×1, 1×163, and 163-element flat shapes. Empty -> [].
    """
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:  # pragma: no cover - numpy is required by scipy
        # Pure-python fallback: assume it's already iterable.
        return list(arr) if arr is not None else []
    if arr is None:
        return []
    if isinstance(arr, np.ndarray):
        return list(arr.ravel())
    if isinstance(arr, (list, tuple)):
        return list(arr)
    # Scalar — wrap it.
    return [arr]


class CompCarsNameTable:
    """Resolve ``(make_id, model_id)`` to ``(make_name, model_name)``.

    Loads ``misc/make_model_name.mat`` once. The dataset uses 1-based IDs
    (MATLAB convention); we store the cells in a 0-indexed python list and
    translate on lookup.
    """

    def __init__(self, mat_bytes: bytes) -> None:
        raw = _load_mat(mat_bytes)
        make_names_arr = raw.get("make_names")
        model_names_arr = raw.get("model_names")
        if make_names_arr is None or model_names_arr is None:
            raise CompCarsLabelError(
                "make_model_name.mat missing 'make_names' or 'model_names' "
                f"(have: {sorted(k for k in raw if not k.startswith('__'))!r})"
            )
        self._make_names: list[str | None] = [_coerce_str(c) for c in _to_1d_list(make_names_arr)]
        self._model_names: list[str | None] = [_coerce_str(c) for c in _to_1d_list(model_names_arr)]

    def resolve(self, make_id: int, model_id: int) -> tuple[str, str]:
        """Return ``(make_name, model_name)`` for the given 1-based IDs.

        Raises :class:`CompCarsLabelError` when either ID is out of range or
        maps to an empty cell.
        """
        if make_id <= 0 or make_id > len(self._make_names):
            raise CompCarsLabelError(f"make_id {make_id} out of range (1..{len(self._make_names)})")
        if model_id <= 0 or model_id > len(self._model_names):
            raise CompCarsLabelError(
                f"model_id {model_id} out of range (1..{len(self._model_names)})"
            )
        make = self._make_names[make_id - 1]
        model = self._model_names[model_id - 1]
        if not make:
            raise CompCarsLabelError(f"empty make name for make_id={make_id}")
        if not model:
            raise CompCarsLabelError(f"empty model name for model_id={model_id}")
        return make, model

    def __len__(self) -> int:
        """Number of make entries — handy for diagnostic logging."""
        return len(self._make_names)

    @property
    def model_count(self) -> int:
        """Number of model entries — handy for diagnostic logging."""
        return len(self._model_names)


class CompCarsBodyTypeTable:
    """Resolve ``model_id`` to a body-style name, or ``None`` when unknown.

    Loads ``misc/car_type.mat`` once. The file contains:

    * ``car_type``  — a 12-element cell array of body-style names.
    * ``model_type`` — a 1-D integer array indexed by ``model_id`` (1-based)
      with values in ``1..12`` (a 1-based index into ``car_type``) or ``0``
      meaning "unknown".

    Some snapshots name the second field ``model2type`` instead; we accept
    either spelling.
    """

    def __init__(self, mat_bytes: bytes) -> None:
        raw = _load_mat(mat_bytes)
        car_type_arr = raw.get("car_type")
        # Accept either spelling.
        model_type_arr = raw.get("model_type")
        if model_type_arr is None:
            model_type_arr = raw.get("model2type")
        if car_type_arr is None or model_type_arr is None:
            raise CompCarsLabelError(
                "car_type.mat missing 'car_type' or 'model_type'/'model2type' "
                f"(have: {sorted(k for k in raw if not k.startswith('__'))!r})"
            )
        self._car_types: list[str | None] = [_coerce_str(c) for c in _to_1d_list(car_type_arr)]

        # model_type is an int array — coerce to a plain python list of ints.
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:  # pragma: no cover - numpy is required by scipy
            self._model_to_type: list[int] = [int(x) for x in _to_1d_list(model_type_arr)]
        else:
            if isinstance(model_type_arr, np.ndarray):
                self._model_to_type = [int(x) for x in model_type_arr.ravel().tolist()]
            else:
                self._model_to_type = [int(x) for x in _to_1d_list(model_type_arr)]

    def resolve(self, model_id: int) -> str | None:
        """Return the body-style name for the given 1-based ``model_id``.

        Returns ``None`` when ``model_id`` is out of range or maps to ``0``
        (unknown). Never raises.
        """
        if model_id <= 0 or model_id > len(self._model_to_type):
            return None
        type_idx = self._model_to_type[model_id - 1]
        if type_idx <= 0 or type_idx > len(self._car_types):
            return None
        return self._car_types[type_idx - 1]
