from __future__ import annotations

import os
from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from statistics import fmean, stdev
from time import perf_counter
from typing import Any, Callable, Iterator


@dataclass
class ObservationCollector:
    records: list[dict[str, float | str]] = field(default_factory=list)

    def record(self, label: str, duration_seconds: float) -> None:
        self.records.append({"label": label, "duration_seconds": duration_seconds})


@dataclass(frozen=True)
class ObservationSummary:
    calls: int
    total_seconds: float
    mean_seconds: float
    std_seconds: float


_ACTIVE_COLLECTOR: ContextVar[ObservationCollector | None] = ContextVar(
    "benchcaddy_active_collector",
    default=None,
)
_BENCH_ACTIVE: ContextVar[bool] = ContextVar("benchcaddy_bench_active", default=False)


def summarize_observations(observations: Iterable[dict[str, Any]]) -> dict[str, ObservationSummary]:
    sample_totals: list[dict[str, float]] = []
    call_counts: dict[str, int] = {}

    for sample in observations:
        totals: dict[str, float] = {}
        for record in sample.get("records", []):
            label = str(record["label"])
            totals[label] = totals.get(label, 0.0) + float(record["duration_seconds"])
            call_counts[label] = call_counts.get(label, 0) + 1
        sample_totals.append(totals)

    return {
        label: ObservationSummary(
            calls=call_counts[label],
            total_seconds=sum(per_sample_totals),
            mean_seconds=float(fmean(per_sample_totals)),
            std_seconds=float(stdev(per_sample_totals)) if len(per_sample_totals) > 1 else 0.0,
        )
        for label in sorted(call_counts)
        if (per_sample_totals := [totals.get(label, 0.0) for totals in sample_totals])
    }


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
            if not (_BENCH_ACTIVE.get() or bool(os.getenv("BENCH_ACTIVE"))):
                return function(*args, **kwargs)

            collector = _ACTIVE_COLLECTOR.get()
            if collector is None:
                return function(*args, **kwargs)

            start = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                collector.record(label, perf_counter() - start)

        return wrapper

    return decorator
