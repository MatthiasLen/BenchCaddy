"""Observation hooks for isolated execution.

This module keeps the isolated-process observation flow local to the
isolation package. It provides a lightweight collector context and a
decorator that can record timing, normalized return values, or both for
decorated call sites that execute inside an isolated run.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter
from typing import Any, Literal

from benchcaddy.return_values import normalize_return_value

ObservationMode = Literal["time", "return"]


@dataclass
class ObservationCollector:
    records: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class IsolatedRunResult:
    elapsed_seconds: float
    return_value: Any
    observations: list[dict[str, Any]]


_ACTIVE_COLLECTOR: ContextVar[ObservationCollector | None] = ContextVar(
    "benchcaddy_isolation_active_collector",
    default=None,
)


@contextmanager
def collect_observations() -> Iterator[ObservationCollector]:
    collector = ObservationCollector()
    collector_token = _ACTIVE_COLLECTOR.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE_COLLECTOR.reset(collector_token)


class observe:
    """Decorator to record observations from call sites inside an isolated run."""

    def __init__(self, *modes: ObservationMode) -> None:
        self.normalized_modes = tuple(dict.fromkeys(modes))
        if not self.normalized_modes:
            raise ValueError("observe() requires at least one mode")

        invalid_modes = [mode for mode in self.normalized_modes if mode not in {"time", "return"}]
        if invalid_modes:
            invalid_text = ", ".join(sorted(set(str(mode) for mode in invalid_modes)))
            raise ValueError(f"Unsupported observe() modes: {invalid_text}")

    def __call__(self, function: Callable[..., Any]) -> Callable[..., Any]:
        label = function.__qualname__

        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            collector = _ACTIVE_COLLECTOR.get()
            if collector is None:
                return function(*args, **kwargs)

            start = perf_counter() if "time" in self.normalized_modes else None

            try:
                result = function(*args, **kwargs)
            except Exception:
                # If timing was requested, record the elapsed time even if the call raises.
                # This ensures that we capture timing information for failed calls, which can be crucial for diagnosing issues in isolated runs.
                if start is not None:
                    collector.records.append(
                        {
                            "label": label,
                            "kind": "time",
                            "duration_seconds": perf_counter() - start,
                        }
                    )
                raise

            # If return value normalization was requested, attempt to record the normalized return value.
            if "return" in self.normalized_modes:
                with suppress(TypeError):
                    collector.records.append(
                        {
                            "label": label,
                            "kind": "return",
                            "value": normalize_return_value(result),
                        }
                    )
            # If timing was requested, record the elapsed time for the successful call.
            if start is not None:
                collector.records.append(
                    {
                        "label": label,
                        "kind": "time",
                        "duration_seconds": perf_counter() - start,
                    }
                )
            return result

        return wrapper
