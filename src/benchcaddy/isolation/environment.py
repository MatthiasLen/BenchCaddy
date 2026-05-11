"""Lightweight environment signal collection.

This module *observes* the system state — it does not attempt to
control or modify it.  The resulting :class:`EnvironmentState` is used
by :mod:`benchcaddy.isolation.report` to generate reliability warnings.
"""

from __future__ import annotations

from dataclasses import dataclass

import psutil

# Fraction of nominal max frequency below which we consider the CPU to
# be throttled or scaling (e.g. powersave governor active).
_FREQ_STABLE_THRESHOLD = 0.90

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


def _read_cpu_load() -> float | None:
    try:
        return psutil.cpu_percent(interval=0.05) / 100.0
    except (psutil.Error, AttributeError, OSError):
        return None


def _read_on_battery() -> bool | None:
    try:
        battery = psutil.sensors_battery()
    except (AttributeError, NotImplementedError, OSError):
        return None
    if battery is None:
        return None
    return not battery.power_plugged


def _read_thermal_throttling() -> bool | None:
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

    return (current_freq / max_freq) >= _FREQ_STABLE_THRESHOLD


def collect_environment_state() -> EnvironmentState:
    """Collect a lightweight snapshot of the current environment."""
    return EnvironmentState(
        cpu_load=_read_cpu_load(),
        on_battery=_read_on_battery(),
        thermal_throttling=_read_thermal_throttling(),
        frequency_stable=_read_frequency_stable(),
    )
