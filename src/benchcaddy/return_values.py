from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from numbers import Real
from typing import Any

StoredReturnValue = bool | int | float | str | list[float]

_VECTOR_TYPES = (list, tuple)

try:
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - numpy is optional
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

    return _as_numeric_vector_without_numpy(value)


def _as_numeric_vector_without_numpy(value: Sequence[object]) -> list[float] | None:
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
    if isinstance(reference_value, _VECTOR_TYPES) != isinstance(candidate_value, _VECTOR_TYPES):
        return None

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


def return_relative_error(
    *,
    reference_value: StoredReturnValue | None,
    candidate_value: StoredReturnValue | None,
) -> float | bool | None:
    distance = return_distance(
        reference_value=reference_value,
        candidate_value=candidate_value,
    )
    if distance is None or isinstance(distance, bool):
        return distance

    if _is_numeric_scalar(reference_value):
        reference_magnitude = abs(float(reference_value))
    else:
        reference_vector = _as_numeric_vector(reference_value)
        if reference_vector is None:
            return None
        reference_magnitude = sqrt(sum(item * item for item in reference_vector))

    if reference_magnitude == 0.0:
        return 0.0 if distance == 0.0 else None

    return float(distance / reference_magnitude)
