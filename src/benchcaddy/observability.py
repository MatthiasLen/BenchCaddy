"""Observation summary helpers.

This module should encapsulate read-side aggregation of stored
observation records. It intentionally stays focused on summarizing
persisted timing observations for comparison and presentation rather than
collecting observations at runtime.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean, stdev
from typing import Any


@dataclass(frozen=True)
class ObservationSummary:
    calls: int
    total_seconds: float
    mean_seconds: float
    std_seconds: float


def summarize_observations(observations: Iterable[dict[str, Any]]) -> dict[str, ObservationSummary]:
    sample_totals: list[dict[str, float]] = []
    call_counts: dict[str, int] = {}

    for sample in observations:
        totals: dict[str, float] = {}
        for record in sample.get("records", []):
            if record.get("kind") == "return":
                continue
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
