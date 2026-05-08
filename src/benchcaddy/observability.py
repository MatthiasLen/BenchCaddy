from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter
from typing import Any, Callable, Iterator


@dataclass
class ObservationCollector:
    records: list[dict[str, float | str]] = field(default_factory=list)

    def record(self, label: str, duration_seconds: float) -> None:
        self.records.append(
            {
                "label": label,
                "duration_seconds": duration_seconds,
            }
        )


_ACTIVE_COLLECTOR: ContextVar[ObservationCollector | None] = ContextVar(
    "benchcaddy_active_collector",
    default=None,
)
_BENCH_ACTIVE: ContextVar[bool] = ContextVar("benchcaddy_bench_active", default=False)


def get_active_collector() -> ObservationCollector | None:
    return _ACTIVE_COLLECTOR.get()


def is_bench_active() -> bool:
    return _BENCH_ACTIVE.get() or bool(os.getenv("BENCH_ACTIVE"))


@contextmanager
def collect_observations() -> Iterator[ObservationCollector]:
    collector = ObservationCollector()
    collector_token = _ACTIVE_COLLECTOR.set(collector)
    active_token = _BENCH_ACTIVE.set(True)
    try:
        yield collector
    finally:
        _BENCH_ACTIVE.reset(active_token)
        _ACTIVE_COLLECTOR.reset(collector_token)


def observe(label: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_bench_active():
                return function(*args, **kwargs)

            collector = get_active_collector()
            if collector is None:
                return function(*args, **kwargs)

            start = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                collector.record(label, perf_counter() - start)

        return wrapper

    return decorator
