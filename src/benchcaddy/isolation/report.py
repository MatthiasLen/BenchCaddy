"""Reliability reporting for isolation diagnostics.

This module should encapsulate the policy that turns raw isolation
signals into human-readable reliability judgments and warnings. It is
where environment-derived risk and noise-derived classifications are
combined into a concise, actionable report for users.
"""

from __future__ import annotations

from dataclasses import dataclass

from .environment import EnvironmentState, environment_risk_score, environment_warnings
from .noise import NoiseEstimate

# Thresholds used when classifying individual dimensions.
_ENVIRONMENT_FAIR_RISK = 1
_ENVIRONMENT_LOW_RISK = 3
_MODERATE_NOISE_RISK = 1
_HIGH_NOISE_RISK = 3

_NOISE_RISK_BY_LEVEL = {
    "low": 0,
    "moderate": _MODERATE_NOISE_RISK,
    "high": _HIGH_NOISE_RISK,
}
_NOISE_LEVEL_ORDER = {"low": 0, "moderate": 1, "high": 2}


@dataclass(frozen=True)
class ReliabilityReport:
    """A concise summary of benchmark environment reliability."""

    timing_stability: str
    """Overall timing stability level: ``"HIGH"``, ``"FAIR"``, or
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
            f"Timing stability:      {self.timing_stability}",
            f"Environmental quality:  {self.environmental_quality}",
        ]
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        return "\n".join(lines)


def _noise_risk_score(noise: NoiseEstimate) -> int:
    """Return the additive timing-noise risk that feeds report quality."""
    return _NOISE_RISK_BY_LEVEL[noise.noise_level] + _NOISE_RISK_BY_LEVEL[noise.drift_level]


def _environmental_quality(env: EnvironmentState, noise: NoiseEstimate) -> str:
    """Derive overall environmental quality from cumulative benchmark risk.

    The quality label answers a host-level question: "How suitable is the
    current machine state for reproducible benchmarking?" It therefore uses
    additive risk scoring rather than the previous first-match heuristic,
    so multiple moderate degradations can collectively lower the result.
    """
    risk = environment_risk_score(env) + _noise_risk_score(noise)
    if risk >= _ENVIRONMENT_LOW_RISK:
        return "LOW"
    if risk >= _ENVIRONMENT_FAIR_RISK:
        return "FAIR"
    return "HIGH"


def _noise_warning(noise: NoiseEstimate) -> str | None:
    """Describe the measured timing jitter within the capture window."""
    if noise.noise_level == "low":
        return None

    severity = "High" if noise.noise_level == "high" else "Moderate"
    guidance = (
        "background scheduling or frequency changes may distort results"
        if noise.noise_level == "high"
        else "consider reducing background activity or increasing sample counts"
    )
    return f"{severity} timing jitter detected (MAD noise {noise.relative_jitter:.1%}) — {guidance}"


def _drift_warning(noise: NoiseEstimate) -> str | None:
    """Describe timing drift across the capture window."""
    if noise.drift_level == "low":
        return None

    severity = "High" if noise.drift_level == "high" else "Moderate"
    guidance = (
        "measurements changed materially during capture"
        if noise.drift_level == "high"
        else "results may still shift during longer benchmark runs"
    )
    return f"{severity} timing drift detected (early/late quartile drift {noise.relative_drift:.1%}) — {guidance}"


def _timing_stability(noise: NoiseEstimate) -> str:
    """Summarize overall timing stability from noise and drift levels."""
    worst_level = max(noise.noise_level, noise.drift_level, key=lambda level: _NOISE_LEVEL_ORDER[level])
    if worst_level == "high":
        return "LOW"
    if worst_level == "moderate":
        return "FAIR"
    return "HIGH"


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
        via :class:`~benchcaddy.isolation.noise.NoiseAnalyzer`.
    """
    warnings = environment_warnings(environment)
    for warning in (_noise_warning(noise), _drift_warning(noise)):
        if warning is not None:
            warnings.append(warning)

    return ReliabilityReport(
        timing_stability=_timing_stability(noise),
        environmental_quality=_environmental_quality(environment, noise),
        warnings=tuple(warnings),
    )
