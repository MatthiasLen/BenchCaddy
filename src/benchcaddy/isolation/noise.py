"""Timing-noise estimation for the current host.

This module implements a short synthetic probe that characterises how
stable repeated timings are on the active machine. The goal is not to
model application-specific performance, but to quantify the ambient
measurement noise that benchmarking code must compete with.

The estimator intentionally separates two failure modes:

* bulk jitter: the robust spread of ordinary probe durations
* tail jitter: slower pause-like events in the upper tail

Benchmark environments that look stable in the middle but occasionally
stall are still problematic, so the reported relative jitter is the
worse of those two views.
"""

from __future__ import annotations

from math import exp, log
from random import Random
from dataclasses import dataclass
from statistics import median
from time import get_clock_info, perf_counter

_DEFAULT_ITERATIONS = 500
_DEFAULT_CONSENSUS_REPEATS = 10
_DEFAULT_BOOTSTRAP_RESAMPLES = 750
_BOOTSTRAP_SEED = 0
_MIN_PROBE_SECONDS = 1e-03
_RESOLUTION_MULTIPLIER = 200
_MAX_WORK_UNITS = 1 << 20
_INITIAL_WORK_UNITS = 32
_WARMUP_ITERATIONS = 5
_CALIBRATION_REPEATS = 5
_MAD_SCALE_FACTOR = 1.4826
_TAIL_PERCENTILE = 0.95
_HIGH_CONFIDENCE_MIN_REPEATS = 5
_FAIR_CONFIDENCE_MIN_REPEATS = 3
_HIGH_CONFIDENCE_MAX_RELATIVE_MARGIN = 0.35
_FAIR_CONFIDENCE_MAX_RELATIVE_MARGIN = 0.75
_LOG_EPSILON = 1e-12
_CI_LOWER_QUANTILE = 0.025
_CI_UPPER_QUANTILE = 0.975

# Relative jitter thresholds for classification.
#
# These bands are expressed in terms of multiplicative timing distortion.
# A host whose p95 timing slowdowns stay below roughly 5% is usually good
# enough for ordinary microbenchmarks, while 5–20% means ambient jitter is
# material but still potentially workable with caution. Beyond 20%, the
# host noise is large enough to distort many short benchmark results.
_LOW_THRESHOLD = 0.05       # < 5 % → low
_MODERATE_THRESHOLD = 0.20  # 5–20 % → moderate, ≥ 20 % → high


@dataclass(frozen=True)
class NoiseEstimate:
    """Characterisation of timing noise measured on the current host."""

    relative_jitter: float
    """Estimated overall relative timing jitter.

    This is the larger of ``robust_jitter`` and ``tail_jitter`` so that a
    host with rare but meaningful pause events is not misclassified as
    clean just because its median behaviour is stable.
    """

    level: str
    """Human-readable noise level: ``"low"``, ``"moderate"``, or
    ``"high"``."""

    statistical_confidence: str = "LOW"
    """Confidence in the reported jitter estimate, not in the host quality."""

    iteration_count: int = 0
    """Number of probe-loop samples used per jitter estimate."""

    repeat_count: int = 1
    """Number of repeated jitter estimates aggregated into this value."""

    robust_jitter: float | None = None
    """Robust central jitter estimate based on median absolute deviation."""

    tail_jitter: float | None = None
    """Upper-tail jitter estimate defined as ``p95 / median - 1``."""

    median_sample_seconds: float | None = None
    """Median elapsed time of one probe sample in seconds."""

    ci_lower: float | None = None
    """Lower bound of the 95% confidence interval for ``relative_jitter``."""

    ci_upper: float | None = None
    """Upper bound of the 95% confidence interval for ``relative_jitter``."""

    relative_margin_of_error: float | None = None
    """Multiplicative 95% margin relative to the point estimate.
    For example, ``0.25`` means roughly ``estimate * (1 +- 25%)`` on a
    multiplicative scale. ``None`` when unavailable."""


def _classify(relative_jitter: float) -> str:
    """Map a relative jitter estimate onto the public severity bands."""
    if relative_jitter < _LOW_THRESHOLD:
        return "low"
    if relative_jitter < _MODERATE_THRESHOLD:
        return "moderate"
    return "high"


def _crossed_threshold_count(ci_lower: float | None, ci_upper: float | None) -> int:
    """Count how many severity boundaries are crossed by the interval."""
    if ci_lower is None or ci_upper is None:
        return 2
    return sum(ci_lower < threshold <= ci_upper for threshold in (_LOW_THRESHOLD, _MODERATE_THRESHOLD))


def _confidence_from_interval(
    *,
    repeats: int,
    ci_lower: float | None,
    ci_upper: float | None,
    relative_margin_of_error: float | None,
) -> str:
    if ci_lower is None or ci_upper is None:
        return "LOW"

    if relative_margin_of_error is None:
        return "LOW"

    crossed_thresholds = _crossed_threshold_count(ci_lower, ci_upper)

    if (
        repeats >= _HIGH_CONFIDENCE_MIN_REPEATS
        and crossed_thresholds == 0
        and relative_margin_of_error <= _HIGH_CONFIDENCE_MAX_RELATIVE_MARGIN
    ):
        return "HIGH"

    if (
        repeats >= _FAIR_CONFIDENCE_MIN_REPEATS
        and crossed_thresholds <= 1
        and relative_margin_of_error <= _FAIR_CONFIDENCE_MAX_RELATIVE_MARGIN
    ):
        return "FAIR"

    return "LOW"


def _probe_target_seconds() -> float:
    """Return the minimum useful per-sample probe duration."""
    resolution = get_clock_info("perf_counter").resolution
    return max(_MIN_PROBE_SECONDS, resolution * _RESOLUTION_MULTIPLIER)


def _probe_once(work_units: int) -> float:
    accumulator = 0
    t0 = perf_counter()
    for index in range(work_units):
        accumulator += index & 1
    elapsed = perf_counter() - t0
    if accumulator < 0:
        raise RuntimeError("unreachable")
    return elapsed


def _calibrate_work_units(target_seconds: float) -> int:
    """Choose a workload size whose median duration clears the target."""
    work_units = _INITIAL_WORK_UNITS
    while work_units < _MAX_WORK_UNITS:
        samples = [_probe_once(work_units) for _ in range(_CALIBRATION_REPEATS)]
        if median(samples) >= target_seconds:
            return work_units
        work_units *= 2
    return _MAX_WORK_UNITS


def _percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated sample quantile."""
    if not values:
        raise ValueError("values must not be empty")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * quantile
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = rank - lower_index
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return lower + (upper - lower) * fraction


def _robust_relative_jitter(durations: list[float], *, center: float | None = None) -> tuple[float, float]:
    """Estimate central timing dispersion via scaled median absolute deviation."""
    median_duration = median(durations) if center is None else center
    if median_duration <= 0.0:
        return 0.0, median_duration

    absolute_deviations = [abs(duration - median_duration) for duration in durations]
    mad = median(absolute_deviations)
    return (_MAD_SCALE_FACTOR * mad) / median_duration, median_duration


def _tail_relative_jitter(durations: list[float], *, center: float | None = None) -> tuple[float, float]:
    """Estimate bursty pause behaviour from the upper timing tail."""
    median_duration = median(durations) if center is None else center
    if median_duration <= 0.0:
        return 0.0, median_duration

    upper_tail = _percentile(durations, _TAIL_PERCENTILE)
    return max(0.0, (upper_tail / median_duration) - 1.0), median_duration


def _bootstrap_confidence_interval(
    values: list[float],
    *,
    resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
) -> tuple[float, float] | None:
    """Bootstrap a log-scale confidence interval for positive jitter values."""
    if len(values) < 2:
        return None

    rng = Random(_BOOTSTRAP_SEED)
    log_values = [log(max(value, _LOG_EPSILON)) for value in values]
    sample_size = len(values)
    bootstrap_log_medians = [
        median([log_values[rng.randrange(sample_size)] for _ in range(sample_size)])
        for _ in range(resamples)
    ]
    ordered = sorted(bootstrap_log_medians)
    lower_index = int((resamples - 1) * _CI_LOWER_QUANTILE)
    upper_index = int((resamples - 1) * _CI_UPPER_QUANTILE)
    return float(exp(ordered[lower_index])), float(exp(ordered[upper_index]))


def _relative_margin_of_error(point_estimate: float, ci_lower: float | None, ci_upper: float | None) -> float | None:
    """Return the larger multiplicative half-width implied by the interval."""
    if ci_lower is None or ci_upper is None:
        return None
    if point_estimate <= 0.0:
        return 0.0 if ci_upper <= 0.0 else None

    lower_factor = point_estimate / max(ci_lower, _LOG_EPSILON)
    upper_factor = ci_upper / point_estimate
    return float(max(lower_factor, upper_factor) - 1.0)


def _median_or_none(values: list[float]) -> float | None:
    """Return the median of *values* or ``None`` when no values exist."""
    if not values:
        return None
    return float(median(values))


def _estimate_noise_once(iterations: int, *, work_units: int | None = None) -> NoiseEstimate:
    """Run one calibrated probe batch and derive robust/tail jitter metrics."""
    if iterations < 2:
        raise ValueError("iterations must be at least 2")

    effective_work_units = work_units if work_units is not None else _calibrate_work_units(_probe_target_seconds())

    for _ in range(_WARMUP_ITERATIONS):
        _probe_once(effective_work_units)

    durations = [_probe_once(effective_work_units) for _ in range(iterations)]
    robust_jitter, median_duration = _robust_relative_jitter(durations)
    tail_jitter, _ = _tail_relative_jitter(durations, center=median_duration)
    if median_duration <= 0.0:
        return NoiseEstimate(
            relative_jitter=0.0,
            level="low",
            statistical_confidence="LOW",
            iteration_count=iterations,
            robust_jitter=0.0,
            tail_jitter=0.0,
            median_sample_seconds=0.0,
        )

    relative_jitter = max(robust_jitter, tail_jitter)
    return NoiseEstimate(
        relative_jitter=relative_jitter,
        level=_classify(relative_jitter),
        statistical_confidence="LOW",
        iteration_count=iterations,
        robust_jitter=robust_jitter,
        tail_jitter=tail_jitter,
        median_sample_seconds=median_duration,
    )


def estimate_noise(iterations: int = _DEFAULT_ITERATIONS) -> NoiseEstimate:
    """Estimate timing noise from a single calibrated probe run.

    The probe workload is a small deterministic Python loop whose size is
    calibrated so each sample is comfortably above the system timer
    resolution. This avoids classifying timer quantization itself as host
    jitter while still keeping the probe short enough to reflect runtime
    and scheduler noise.
    """
    estimate = _estimate_noise_once(iterations)
    return NoiseEstimate(
        relative_jitter=estimate.relative_jitter,
        level=estimate.level,
        statistical_confidence="LOW",
        iteration_count=estimate.iteration_count,
        repeat_count=1,
        robust_jitter=estimate.robust_jitter,
        tail_jitter=estimate.tail_jitter,
        median_sample_seconds=estimate.median_sample_seconds,
    )


def estimate_noise_consensus(
    iterations: int = _DEFAULT_ITERATIONS,
    *,
    repeats: int = _DEFAULT_CONSENSUS_REPEATS,
) -> NoiseEstimate:
    """Estimate timing noise from multiple probe runs and aggregate robustly.

    ``benchcaddy check`` is intended to summarize the current host, not one
    unlucky scheduler blip. This helper repeats the calibrated probe several
    times and reports the median relative jitter so adjacent invocations are
    materially more stable.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    if repeats == 1:
        return estimate_noise(iterations)

    work_units = _calibrate_work_units(_probe_target_seconds())
    estimates = [_estimate_noise_once(iterations, work_units=work_units) for _ in range(repeats)]
    relative_jitters = [estimate.relative_jitter for estimate in estimates]
    consensus_relative_jitter = float(median(relative_jitters))
    robust_jitters = [estimate.robust_jitter for estimate in estimates if estimate.robust_jitter is not None]
    tail_jitters = [estimate.tail_jitter for estimate in estimates if estimate.tail_jitter is not None]
    sample_medians = [estimate.median_sample_seconds for estimate in estimates if estimate.median_sample_seconds is not None]
    interval = _bootstrap_confidence_interval(relative_jitters)
    ci_lower = None if interval is None else interval[0]
    ci_upper = None if interval is None else interval[1]
    relative_margin_of_error = _relative_margin_of_error(consensus_relative_jitter, ci_lower, ci_upper)
    return NoiseEstimate(
        relative_jitter=consensus_relative_jitter,
        level=_classify(consensus_relative_jitter),
        statistical_confidence=_confidence_from_interval(
            repeats=repeats,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            relative_margin_of_error=relative_margin_of_error,
        ),
        iteration_count=iterations,
        repeat_count=repeats,
        robust_jitter=_median_or_none(robust_jitters),
        tail_jitter=_median_or_none(tail_jitters),
        median_sample_seconds=_median_or_none(sample_medians),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        relative_margin_of_error=relative_margin_of_error,
    )
