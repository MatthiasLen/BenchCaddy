"""Benchmark reliability reporting.

Combine an :class:`~benchcaddy.isolation.environment.EnvironmentState`
and a :class:`~benchcaddy.isolation.noise.NoiseEstimate` into a
human-readable :class:`ReliabilityReport` that surfaces actionable
warnings without overstating what BenchCaddy can guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

from .environment import EnvironmentState
from .noise import NoiseEstimate

# Thresholds used when classifying individual dimensions.
_LOAD_HIGH_THRESHOLD = 0.30       # > 30 % CPU load → background load elevated
_LOAD_MODERATE_THRESHOLD = 0.15   # > 15 % → minor background activity


@dataclass(frozen=True)
class ReliabilityReport:
    """A concise summary of benchmark environment reliability."""

    statistical_confidence: str
    """Overall statistical confidence level: ``"HIGH"``, ``"FAIR"``, or
    ``"LOW"``."""

    environmental_quality: str
    """Overall environmental quality level: ``"HIGH"``, ``"FAIR"``, or
    ``"LOW"``."""

    warnings: tuple[str, ...]
    """Ordered list of actionable warning messages."""

    def format(self) -> str:
        """Return a plain-text representation suitable for console output."""
        lines: list[str] = [
            "Benchmark Reliability",
            "---------------------",
            f"Statistical confidence: {self.statistical_confidence}",
            f"Environmental quality:  {self.environmental_quality}",
        ]
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        return "\n".join(lines)


def _statistical_confidence(noise: NoiseEstimate) -> str:
    mapping = {"low": "HIGH", "moderate": "FAIR", "high": "LOW"}
    return mapping.get(noise.level, "FAIR")


def _environmental_quality(env: EnvironmentState, warnings: list[str]) -> str:
    """Derive overall environmental quality from the collected warnings."""
    critical_keywords = ("thermal throttling", "battery power")
    fair_keywords = ("background load", "CPU scaling", "frequency")

    has_critical = any(any(kw in w.lower() for kw in critical_keywords) for w in warnings)
    has_fair = any(any(kw in w.lower() for kw in fair_keywords) for w in warnings)

    if has_critical:
        return "LOW"
    if has_fair:
        return "FAIR"
    if env.cpu_load is None and env.on_battery is None and env.thermal_throttling is None and env.frequency_stable is None:
        # No environment data available — be conservative.
        return "FAIR"
    return "HIGH"


def _collect_warnings(env: EnvironmentState, noise: NoiseEstimate) -> list[str]:
    warnings: list[str] = []

    # --- environment warnings ---
    if env.on_battery is True:
        warnings.append("Running on battery power — results may be throttled")

    if env.thermal_throttling is True:
        warnings.append("Thermal throttling detected — CPU may be running below rated speed")

    if env.frequency_stable is False:
        warnings.append("CPU frequency scaling active — consider disabling dynamic frequency scaling")

    if env.cpu_load is not None:
        if env.cpu_load > _LOAD_HIGH_THRESHOLD:
            warnings.append(f"Background load elevated ({env.cpu_load:.0%}) — close competing processes for more reliable results")
        elif env.cpu_load > _LOAD_MODERATE_THRESHOLD:
            warnings.append(f"Minor background activity detected ({env.cpu_load:.0%})")

    # --- noise warnings ---
    if noise.level == "high":
        warnings.append(f"High timing jitter detected (relative jitter {noise.relative_jitter:.1%}) — scheduler or system noise may distort results")
    elif noise.level == "moderate":
        warnings.append(f"Moderate timing jitter ({noise.relative_jitter:.1%}) — increase sample count to improve confidence")

    return warnings


def build_reliability_report(
    *,
    environment: EnvironmentState,
    noise: NoiseEstimate,
) -> ReliabilityReport:
    """Build a :class:`ReliabilityReport` from environment and noise data.

    Parameters
    ----------
    environment:
        An :class:`~benchcaddy.isolation.environment.EnvironmentState`
        collected via
        :func:`~benchcaddy.isolation.environment.collect_environment_state`.
    noise:
        A :class:`~benchcaddy.isolation.noise.NoiseEstimate` collected
        via :func:`~benchcaddy.isolation.noise.estimate_noise`.
    """
    warnings = _collect_warnings(environment, noise)
    statistical_confidence = _statistical_confidence(noise)
    environmental_quality = _environmental_quality(environment, warnings)

    return ReliabilityReport(
        statistical_confidence=statistical_confidence,
        environmental_quality=environmental_quality,
        warnings=tuple(warnings),
    )
