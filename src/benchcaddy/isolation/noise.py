"""Timing-noise estimation for the current host.

This module should encapsulate quick, benchmark-adjacent probes that
estimate baseline timing jitter introduced by the runtime and operating
system. It should stay focused on producing simple noise classifications
that can inform result interpretation and reliability reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, stdev
from time import perf_counter

_DEFAULT_ITERATIONS = 200

# Relative jitter thresholds for classification.
_LOW_THRESHOLD = 0.02       # < 2 % → low
_MODERATE_THRESHOLD = 0.10  # 2–10 % → moderate, ≥ 10 % → high


@dataclass(frozen=True)
class NoiseEstimate:
    """Characterisation of timing noise measured on the current host."""

    relative_jitter: float
    """Coefficient of variation of empty-loop durations (std / mean).
    Lower is better."""

    level: str
    """Human-readable noise level: ``"low"``, ``"moderate"``, or
    ``"high"``."""


def _classify(relative_jitter: float) -> str:
    if relative_jitter < _LOW_THRESHOLD:
        return "low"
    if relative_jitter < _MODERATE_THRESHOLD:
        return "moderate"
    return "high"


def estimate_noise(iterations: int = _DEFAULT_ITERATIONS) -> NoiseEstimate:
    """Estimate timing noise by measuring *iterations* empty loops.

    Each loop body consists solely of a :func:`time.perf_counter` call
    pair; the resulting durations characterise the minimum achievable
    measurement overhead and its variability.
    """
    if iterations < 2:
        raise ValueError("iterations must be at least 2")

    durations: list[float] = []
    for _ in range(iterations):
        t0 = perf_counter()
        t1 = perf_counter()
        durations.append(t1 - t0)

    mean = fmean(durations)
    if mean <= 0:
        return NoiseEstimate(relative_jitter=0.0, level="low")

    cv = stdev(durations) / mean
    return NoiseEstimate(relative_jitter=cv, level=_classify(cv))
