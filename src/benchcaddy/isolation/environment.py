"""Lightweight environment signal collection.

This module should encapsulate read-only system signals that affect the
trustworthiness of benchmark results, such as background CPU load,
battery state, thermal pressure, and frequency scaling. It should gather
environment state snapshots, not apply machine-level controls.
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


def _sample_frequency_ratios() -> list[float]:
    """Collect a short run of current-to-max frequency ratios."""
    ratios: list[float] = []
    for sample_index in range(_FREQ_SAMPLE_COUNT):
        ratio = _frequency_ratio()
        if ratio is not None:
            ratios.append(ratio)
        if sample_index + 1 < _FREQ_SAMPLE_COUNT:
            sleep(_FREQ_SAMPLE_INTERVAL_SECONDS)
    return ratios


def _frequency_ratio() -> float | None:
    """Return the current-to-max CPU frequency ratio when available."""
    try:
        freq = psutil.cpu_freq()
    except (AttributeError, OSError, NotImplementedError):
        return None
    if freq is None:
        return None

    max_freq = freq.max
    current_freq = freq.current
    if not max_freq or not current_freq:
        return None

    return current_freq / max_freq


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





def _read_frequency_stable() -> bool | None:
    """Assess frequency stability from a short series of ratio samples.

    A single frequency snapshot is too brittle on modern CPUs because
    turbo transitions and governor changes can occur between reads.
    BenchCaddy therefore samples a brief window and treats the CPU as
    stable only when the minimum observed ratio stays near nominal speed
    and the ratio does not swing sharply within that window.
    """
    ratios = _sample_frequency_ratios()
    if not ratios:
        return None

    min_ratio = min(ratios)
    max_ratio = max(ratios)
    return min_ratio >= _FREQ_STABLE_THRESHOLD and (max_ratio - min_ratio) <= _FREQ_SPREAD_THRESHOLD


def collect_environment_state() -> EnvironmentState:
    """Collect a lightweight snapshot of the current benchmark environment."""
    return EnvironmentState(
        cpu_load=_read_cpu_load(),
        on_battery=_read_on_battery(),
        thermal_throttling=_read_thermal_throttling(),
        frequency_stable=_read_frequency_stable(),
    )
