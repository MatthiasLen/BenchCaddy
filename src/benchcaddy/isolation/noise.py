"""Timing-noise estimation for the current host.

The active implementation keeps the capture flow explicit:

* calibrate a probe duration for the current host
* warm up once and collect one contiguous vector of samples
* estimate scatter noise and drift from that single capture
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from time import get_clock_info, perf_counter

from ..stats import robust_relative_jitter

# Probe defaults.
_DEFAULT_ITERATIONS = 500
_MIN_PROBE_SECONDS = 1e-03
_RESOLUTION_MULTIPLIER = 200
_INITIAL_WORK_UNITS = 32
_MAX_WORK_UNITS = 1 << 20
_CALIBRATION_REPEATS = 5
_WARMUP_ITERATIONS = 5

# Classification policy.
_LOW_THRESHOLD = 0.05
_MODERATE_THRESHOLD = 0.20


@dataclass(frozen=True)
class NoiseEstimate:
    """Characterisation of timing noise measured on the current host."""

    relative_jitter: float
    noise_level: str
    relative_drift: float
    drift_level: str
    iteration_count: int = 0
    median_sample_seconds: float | None = None


@dataclass(frozen=True)
class NoiseCapture:
    """Raw calibrated probe timings collected from one contiguous capture."""

    durations: tuple[float, ...]
    iteration_count: int
    work_units: int


class NoiseAnalyzer:
    """Lean orchestration entry point for noise capture and analysis."""

    def _probe_target_seconds(self) -> float:
        resolution = get_clock_info("perf_counter").resolution
        return max(_MIN_PROBE_SECONDS, resolution * _RESOLUTION_MULTIPLIER)

    def _probe_once(self, work_units: int) -> float:
        accumulator = 0
        started_at = perf_counter()
        for index in range(work_units):
            accumulator += index & 1
        elapsed = perf_counter() - started_at
        if accumulator < 0:
            raise RuntimeError("unreachable")
        return elapsed

    def _calibrate_work_units(self, target_seconds: float) -> int:
        work_units = _INITIAL_WORK_UNITS
        while work_units < _MAX_WORK_UNITS:
            samples = [self._probe_once(work_units) for _ in range(_CALIBRATION_REPEATS)]
            if median(samples) >= target_seconds:
                return work_units
            work_units *= 2
        return _MAX_WORK_UNITS

    def capture(
        self,
        iterations: int = _DEFAULT_ITERATIONS,
    ) -> NoiseCapture:
        """Capture calibrated probe durations from one contiguous run."""
        if iterations < 2:
            raise ValueError("iterations must be at least 2")

        # Calibrate probe work units to target a reasonable duration for the current host.
        work_units = self._calibrate_work_units(self._probe_target_seconds())
        for _ in range(_WARMUP_ITERATIONS):
            self._probe_once(work_units)
        durations = tuple(self._probe_once(work_units) for _ in range(iterations))

        return NoiseCapture(
            durations=durations,
            iteration_count=iterations,
            work_units=work_units,
        )

    def estimate_statistics(self, capture: NoiseCapture) -> NoiseEstimate:
        """Estimate scatter noise and drift from one captured timing vector."""
        if capture.iteration_count < 2:
            raise ValueError("iterations must be at least 2")
        if len(capture.durations) != capture.iteration_count:
            raise ValueError("capture iteration_count does not match recorded durations")

        sample_median = float(median(capture.durations))
        if sample_median <= 0.0:
            relative_jitter = 0.0
            relative_drift = 0.0
            sample_median = 0.0
        else:
            relative_jitter, _ = robust_relative_jitter(capture.durations, center=sample_median)
            quarter = max(1, len(capture.durations) // 4)
            first_quarter_median = float(median(capture.durations[:quarter]))
            last_quarter_median = float(median(capture.durations[-quarter:]))
            relative_drift = abs(last_quarter_median - first_quarter_median) / sample_median

        if relative_jitter < _LOW_THRESHOLD:
            noise_level = "low"
        elif relative_jitter < _MODERATE_THRESHOLD:
            noise_level = "moderate"
        else:
            noise_level = "high"

        if relative_drift < _LOW_THRESHOLD:
            drift_level = "low"
        elif relative_drift < _MODERATE_THRESHOLD:
            drift_level = "moderate"
        else:
            drift_level = "high"

        return NoiseEstimate(
            relative_jitter=relative_jitter,
            noise_level=noise_level,
            relative_drift=relative_drift,
            drift_level=drift_level,
            iteration_count=capture.iteration_count,
            median_sample_seconds=sample_median,
        )

    def analyze(
        self,
        iterations: int = _DEFAULT_ITERATIONS,
    ) -> NoiseEstimate:
        """Run capture and statistics end to end."""
        return self.estimate_statistics(self.capture(iterations=iterations))


def estimate_noise(iterations: int = _DEFAULT_ITERATIONS) -> NoiseEstimate:
    """Estimate timing noise from a single calibrated probe run."""
    return NoiseAnalyzer().analyze(iterations=iterations)
