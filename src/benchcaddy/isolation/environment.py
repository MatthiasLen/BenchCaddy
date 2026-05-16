"""Lightweight environment signal collection and scoring.

This module should encapsulate read-only system signals that affect the
trustworthiness of benchmark results, such as background CPU load,
battery state, thermal pressure, and frequency scaling. It gathers
environment state snapshots and derives an environment-only benchmark
risk score from those signals, without applying machine-level controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import sleep

import psutil

# Fraction of nominal max frequency below which we consider the CPU to
# be throttled or scaling (e.g. powersave governor active).
_FREQ_STABLE_THRESHOLD = 0.90
_FREQ_SPREAD_THRESHOLD = 0.10
_FREQ_SAMPLE_COUNT = 5
_FREQ_SAMPLE_INTERVAL_SECONDS = 0.02

# Temperature (°C) above which we flag potential thermal throttling.
_THERMAL_THROTTLE_CELSIUS = 90.0

# Thresholds used when deriving an environment-only benchmark risk score.
_LOAD_HIGH_THRESHOLD = 0.30
_LOAD_MODERATE_THRESHOLD = 0.15

# Risk points assigned for individual environment signals when present.
_BATTERY_RISK = 2
_THERMAL_RISK = 3
_FREQUENCY_SCALING_RISK = 2
_MODERATE_LOAD_RISK = 1
_HIGH_LOAD_RISK = 2
_UNKNOWN_ENVIRONMENT_RISK = 1


@dataclass(frozen=True)
class EnvironmentState:
    """Snapshot of lightweight environment signals at collection time."""

    cpu_load: float | None
    """Estimated CPU utilisation in [0.0, 1.0] or ``None`` if unavailable."""

    on_battery: bool | None
    """``True`` when the machine is running on battery power, ``False``
    when plugged in, ``None`` when the information is not available."""

    thermal_throttling: bool | None
    """``True`` when at least one CPU temperature sensor reports a reading
    at or above :data:`_THERMAL_THROTTLE_CELSIUS`.  ``None`` when no
    temperature sensors are accessible."""

    frequency_stable: bool | None
    """``True`` when the current CPU frequency is at least
    :data:`_FREQ_STABLE_THRESHOLD` of the nominal maximum.  ``None``
    when frequency information is unavailable."""


def all_environment_signals_missing(environment: EnvironmentState) -> bool:
    """Return ``True`` when no environment telemetry could be collected."""
    return (
        environment.cpu_load is None
        and environment.on_battery is None
        and environment.thermal_throttling is None
        and environment.frequency_stable is None
    )


def environment_risk_score(environment: EnvironmentState) -> int:
    """Return an additive environment-only benchmark risk score.

    The score summarizes host conditions that can degrade benchmark
    repeatability before any timing-noise analysis is considered.
    """
    risk = 0

    if environment.on_battery is True:
        risk += _BATTERY_RISK

    if environment.thermal_throttling is True:
        risk += _THERMAL_RISK

    if environment.frequency_stable is False:
        risk += _FREQUENCY_SCALING_RISK

    if environment.cpu_load is not None:
        if environment.cpu_load > _LOAD_HIGH_THRESHOLD:
            risk += _HIGH_LOAD_RISK
        elif environment.cpu_load > _LOAD_MODERATE_THRESHOLD:
            risk += _MODERATE_LOAD_RISK

    if all_environment_signals_missing(environment):
        risk += _UNKNOWN_ENVIRONMENT_RISK

    return risk


def environment_warnings(environment: EnvironmentState) -> list[str]:
    """Return actionable warnings derived from environment signals alone."""
    warnings: list[str] = []

    if environment.on_battery is True:
        warnings.append("Running on battery power — results may be throttled")

    if environment.thermal_throttling is True:
        warnings.append("Thermal throttling detected — CPU may be running below rated speed")

    if environment.frequency_stable is False:
        warnings.append("CPU frequency scaling active — consider disabling dynamic frequency scaling")

    if environment.cpu_load is not None:
        if environment.cpu_load > _LOAD_HIGH_THRESHOLD:
            warnings.append(
                f"Background load elevated ({environment.cpu_load:.0%}) — close competing processes for more reliable results"
            )
        elif environment.cpu_load > _LOAD_MODERATE_THRESHOLD:
            warnings.append(f"Minor background activity detected ({environment.cpu_load:.0%})")

    if all_environment_signals_missing(environment):
        warnings.append("Environment telemetry unavailable — quality estimate is conservative")

    return warnings


def _read_frequency_stable() -> bool | None:
    """Assess frequency stability from a short series of CPU frequency samples.

    A single frequency snapshot is too brittle on modern CPUs because
    turbo transitions and governor changes can occur between reads.
    BenchCaddy therefore samples a brief window and treats the CPU as
    stable only when the minimum observed ratio stays near nominal speed
    and the ratio does not swing sharply within that window.
    """
    ratios: list[float] = []
    for sample_index in range(_FREQ_SAMPLE_COUNT):
        try:
            freq = psutil.cpu_freq()
        except (AttributeError, OSError, NotImplementedError):
            return None

        if freq is not None and freq.max and freq.current:
            ratios.append(freq.current / freq.max)

        if sample_index + 1 < _FREQ_SAMPLE_COUNT:
            sleep(_FREQ_SAMPLE_INTERVAL_SECONDS)

    if not ratios:
        return None

    min_ratio = min(ratios)
    max_ratio = max(ratios)
    return min_ratio >= _FREQ_STABLE_THRESHOLD and (max_ratio - min_ratio) <= _FREQ_SPREAD_THRESHOLD


def _read_cpu_load() -> float | None:
    """Read short-window CPU utilisation as a fraction in ``[0, 1]``."""
    try:
        return psutil.cpu_percent(interval=0.05) / 100.0
    except (psutil.Error, AttributeError, OSError):
        return None


def _read_on_battery() -> bool | None:
    """Return whether the host is running on battery power."""
    try:
        battery = psutil.sensors_battery()
    except (AttributeError, NotImplementedError, OSError):
        return None
    if battery is None:
        return None
    return not battery.power_plugged


def _read_thermal_throttling() -> bool | None:
    """Inspect exposed temperature sensors for clear thermal pressure."""
    if not hasattr(psutil, "sensors_temperatures"):
        return None
    try:
        temps = psutil.sensors_temperatures()
    except (AttributeError, OSError, NotImplementedError):
        return None
    if not temps:
        return None

    for readings in temps.values():
        for entry in readings:
            current = getattr(entry, "current", None)
            if current is not None and current >= _THERMAL_THROTTLE_CELSIUS:
                return True
    return False


def collect_environment_state() -> EnvironmentState:
    """Collect a lightweight snapshot of the current benchmark environment."""
    return EnvironmentState(
        cpu_load=_read_cpu_load(),
        on_battery=_read_on_battery(),
        thermal_throttling=_read_thermal_throttling(),
        frequency_stable=_read_frequency_stable(),
    )
