from __future__ import annotations

from math import sqrt
from numbers import Real
from typing import Any

import numpy as np

StoredReturnValue = bool | int | float | str | list[float]


def _is_numeric_scalar(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _as_numeric_vector(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None

    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1:
        return None
    return [float(item) for item in array.tolist()]


def normalize_return_value(value: Any) -> StoredReturnValue:
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int)):
        return value
    if _is_numeric_scalar(value):
        return float(value)
    if (vector := _as_numeric_vector(value)) is not None:
        return vector
    raise TypeError(
        "Unsupported return value type. Expected one of: "
        "bool, int, float, str, or a one-dimensional numeric array/list/tuple."
    )


def _vector_magnitude(value: object) -> float | None:
    vector = _as_numeric_vector(value)
    if vector is None:
        return None
    return sqrt(sum(item * item for item in vector))


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
        reference_magnitude = _vector_magnitude(reference_value)
        if reference_magnitude is None:
            return None

    if reference_magnitude == 0.0:
        return 0.0 if distance == 0.0 else None

    return float(distance / reference_magnitude)
