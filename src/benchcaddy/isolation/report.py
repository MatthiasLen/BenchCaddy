"""Reliability reporting for isolation diagnostics.

This module should encapsulate the policy that turns raw isolation
signals into human-readable reliability judgments and warnings. It is
where environment state and noise estimates are combined into a concise,
actionable report for users.
"""

from __future__ import annotations

from dataclasses import dataclass

from .environment import EnvironmentState
from .noise import NoiseEstimate

# Thresholds used when classifying individual dimensions.
_LOAD_HIGH_THRESHOLD = 0.30       # > 30 % CPU load → background load elevated
_LOAD_MODERATE_THRESHOLD = 0.15   # > 15 % → minor background activity
_ENVIRONMENT_FAIR_RISK = 1
_ENVIRONMENT_LOW_RISK = 3
_BATTERY_RISK = 2
_THERMAL_RISK = 3
_FREQUENCY_SCALING_RISK = 2
_MODERATE_LOAD_RISK = 1
_HIGH_LOAD_RISK = 2
_MODERATE_NOISE_RISK = 1
_HIGH_NOISE_RISK = 3
_UNKNOWN_ENVIRONMENT_RISK = 1


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


def _all_environment_signals_missing(env: EnvironmentState) -> bool:
    """Return ``True`` when no environment telemetry could be collected."""
    return (
        env.cpu_load is None
        and env.on_battery is None
        and env.thermal_throttling is None
        and env.frequency_stable is None
    )


def _environmental_risk_score(env: EnvironmentState, noise: NoiseEstimate) -> int:
    """Combine environmental hazards into a monotone benchmark-risk score.

    Severe single hazards such as thermal throttling or very high timing
    jitter are enough to make the environment unsuitable on their own.
    Moderate hazards accumulate: for example, battery power plus notable
    background load should degrade quality even if each signal alone only
    warrants caution.
    """
    risk = 0

    if env.on_battery is True:
        risk += _BATTERY_RISK

    if env.thermal_throttling is True:
        risk += _THERMAL_RISK

    if env.frequency_stable is False:
        risk += _FREQUENCY_SCALING_RISK

    if env.cpu_load is not None:
        if env.cpu_load > _LOAD_HIGH_THRESHOLD:
            risk += _HIGH_LOAD_RISK
        elif env.cpu_load > _LOAD_MODERATE_THRESHOLD:
            risk += _MODERATE_LOAD_RISK

    if noise.level == "high":
        risk += _HIGH_NOISE_RISK
    elif noise.level == "moderate":
        risk += _MODERATE_NOISE_RISK

    if _all_environment_signals_missing(env):
        risk += _UNKNOWN_ENVIRONMENT_RISK

    return risk


def _environmental_quality(env: EnvironmentState, noise: NoiseEstimate) -> str:
    """Derive overall environmental quality from cumulative benchmark risk.

    The quality label answers a host-level question: "How suitable is the
    current machine state for reproducible benchmarking?" It therefore uses
    additive risk scoring rather than the previous first-match heuristic,
    so multiple moderate degradations can collectively lower the result.
    """
    risk = _environmental_risk_score(env, noise)
    if risk >= _ENVIRONMENT_LOW_RISK:
        return "LOW"
    if risk >= _ENVIRONMENT_FAIR_RISK:
        return "FAIR"
    return "HIGH"


def _noise_warning(noise: NoiseEstimate) -> str | None:
    """Describe the dominant form of timing instability."""
    if noise.level == "low":
        return None

    if (
        noise.tail_jitter is not None
        and noise.robust_jitter is not None
        and noise.tail_jitter > (noise.robust_jitter * 1.5)
    ):
        severity = "High" if noise.level == "high" else "Moderate"
        guidance = "intermittent scheduler pauses may distort results" if noise.level == "high" else "intermittent pauses may still affect short benchmarks"
        return (
            f"{severity} timing jitter detected (95th percentile tail {noise.tail_jitter:.1%} above median; overall {noise.relative_jitter:.1%}) — {guidance}"
        )

    severity = "High" if noise.level == "high" else "Moderate"
    guidance = "background scheduling or frequency changes may distort results" if noise.level == "high" else "consider reducing background activity or increasing sample counts"
    return f"{severity} timing jitter detected (overall {noise.relative_jitter:.1%}) — {guidance}"


def _confidence_warning(noise: NoiseEstimate) -> str | None:
    """Explain when the noise estimate itself is not yet stable enough."""
    if noise.statistical_confidence == "LOW":
        return "Noise estimate has low repeat stability — rerun check or increase --noise-iterations"
    if noise.statistical_confidence == "FAIR":
        return "Noise estimate is near a classification boundary — expect some run-to-run variability"
    return None


def _collect_warnings(env: EnvironmentState, noise: NoiseEstimate) -> list[str]:
    """Collect actionable warnings explaining the reliability verdict."""
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

    if _all_environment_signals_missing(env):
        warnings.append("Environment telemetry unavailable — quality estimate is conservative")

    # --- noise warnings ---
    noise_warning = _noise_warning(noise)
    if noise_warning is not None:
        warnings.append(noise_warning)

    confidence_warning = _confidence_warning(noise)
    if confidence_warning is not None:
        warnings.append(confidence_warning)

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
    statistical_confidence = noise.statistical_confidence
    environmental_quality = _environmental_quality(environment, noise)

    return ReliabilityReport(
        statistical_confidence=statistical_confidence,
        environmental_quality=environmental_quality,
        warnings=tuple(warnings),
    )
