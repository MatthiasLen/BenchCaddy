from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from numbers import Real
from typing import Any

StoredReturnValue = bool | int | float | str | list[float]

_VECTOR_TYPES = (list, tuple)

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy is optional
    _np = None
else:  # pragma: no cover - exercised when numpy is available
    _VECTOR_TYPES = (list, tuple, _np.ndarray)


def _is_numeric_scalar(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _as_numeric_vector(value: object) -> list[float] | None:
    if not isinstance(value, _VECTOR_TYPES):
        return None

    if _np is not None:  # pragma: no branch
        try:
            array = _np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
        if array.ndim != 1:
            return None
        return [float(item) for item in array.tolist()]

    vector: list[float] = []
    for item in value:
        if not _is_numeric_scalar(item):
            return None
        vector.append(float(item))
    return vector


def normalize_return_value(value: Any) -> StoredReturnValue:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if _is_numeric_scalar(value):
        return float(value)
    if (vector := _as_numeric_vector(value)) is not None:
        return vector
    raise TypeError(
        "Unsupported return value type. Expected one of: "
        "bool, int, float, str, or a one-dimensional numeric array/list/tuple."
    )


def return_distance(
    *,
    reference_value: StoredReturnValue | None,
    candidate_value: StoredReturnValue | None,
) -> float | bool | None:
    if reference_value is None or candidate_value is None:
        return None
    if isinstance(reference_value, bool) and isinstance(candidate_value, bool):
        return reference_value == candidate_value
    if isinstance(reference_value, str) and isinstance(candidate_value, str):
        return reference_value == candidate_value
    if _is_numeric_scalar(reference_value) and _is_numeric_scalar(candidate_value):
        return float(abs(candidate_value - reference_value))

    reference_vector = _as_numeric_vector(reference_value)
    candidate_vector = _as_numeric_vector(candidate_value)
    if reference_vector is None or candidate_vector is None:
        return None
    if len(reference_vector) != len(candidate_vector):
        return None
    squared_differences = (
        (candidate - reference) ** 2
        for reference, candidate in zip(reference_vector, candidate_vector, strict=True)
    )
    return sqrt(sum(squared_differences))
