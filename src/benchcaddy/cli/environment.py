from __future__ import annotations

from typing import Annotated

import typer

from ..isolation import build_reliability_report
from ..presentation import render_table, summary_panel
from ._rendering import _styled
from ._shared import _console, _emit_json_response, app


def _quality_style(level: str) -> str:
    return {"HIGH": "green", "FAIR": "yellow", "LOW": "red"}.get(level, "red")


@app.command("env", help="Check the current environment for benchmark reliability issues.")
def env_command(
    noise_iterations: Annotated[
        int,
        typer.Option(
            "--noise-iterations",
            min=2,
            help="Number of short calibrated probe loops used to estimate measurement jitter.",
        ),
    ] = 200,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Emit machine-readable JSON output.",
        ),
    ] = False,
) -> None:
    from . import NoiseAnalyzer, collect_environment_state, get_affinity

    env = collect_environment_state()
    noise = NoiseAnalyzer().analyze(iterations=noise_iterations)
    report = build_reliability_report(environment=env, noise=noise)
    affinity = get_affinity()
    if report.timing_stability == "LOW" or report.environmental_quality == "LOW":
        status = "fail"
        reason = "unreliable_environment"
        suggested_action = "Reduce system noise, stabilize power and thermal conditions, then rerun benchcaddy env -j."
        confidence = "low"
    elif report.warnings:
        status = "inconclusive"
        reason = "environment_warnings_detected"
        suggested_action = "Address the reported warnings before treating benchmark comparisons as reliable."
        confidence = "medium"
    elif report.timing_stability == "FAIR" or report.environmental_quality == "FAIR":
        status = "inconclusive"
        reason = "environment_marginal"
        suggested_action = "Use more samples or reduce background load before trusting small deltas."
        confidence = "medium"
    else:
        status = "pass"
        reason = "environment_ready"
        suggested_action = "Run a benchmark sweep or comparison."
        confidence = "high"

    if json_output:
        _emit_json_response(
            command="env",
            status=status,
            reason=reason,
            suggested_action=suggested_action,
            confidence=confidence,
            result={
                "timing_stability": report.timing_stability,
                "environmental_quality": report.environmental_quality,
                "warnings": list(report.warnings),
                "environment": {
                    "cpu_load": env.cpu_load,
                    "on_battery": env.on_battery,
                    "thermal_throttling": env.thermal_throttling,
                    "frequency_stable": env.frequency_stable,
                },
                "noise": {
                    "relative_jitter": noise.relative_jitter,
                    "noise_level": noise.noise_level,
                    "relative_drift": noise.relative_drift,
                    "drift_level": noise.drift_level,
                    "median_sample_seconds": noise.median_sample_seconds,
                    "iteration_count": noise.iteration_count,
                },
                "affinity": affinity,
            },
        )
        return

    stat_style = _quality_style(report.timing_stability)
    env_style = _quality_style(report.environmental_quality)
    _console().print(
        summary_panel(
            "Benchmark Reliability",
            [
                ("Timing Stability", _styled(report.timing_stability, stat_style)),
                ("Environmental Quality", _styled(report.environmental_quality, env_style)),
                ("Timing Noise", f"{noise.relative_jitter:.2%} ({noise.noise_level})"),
                ("Timing Drift", f"{noise.relative_drift:.2%} ({noise.drift_level})"),
                (
                    "Median Probe",
                    f"{noise.median_sample_seconds * 1_000_000:.0f} us" if noise.median_sample_seconds is not None else "unavailable",
                ),
                ("Probe Samples", str(noise.iteration_count) if noise.iteration_count else "unavailable"),
                ("CPU Affinity", ", ".join(str(c) for c in affinity) if affinity else "unavailable"),
                ("CPU Load", f"{env.cpu_load:.0%}" if env.cpu_load is not None else "unavailable"),
                ("On Battery", "yes" if env.on_battery else "no" if env.on_battery is False else "unknown"),
                ("Thermal Throttling", "yes" if env.thermal_throttling else "no" if env.thermal_throttling is False else "unknown"),
                ("Frequency Stable", "yes" if env.frequency_stable else "no" if env.frequency_stable is False else "unknown"),
            ],
        )
    )
    if report.warnings:
        _console().print(
            render_table(
                "Warnings",
                ["#", "Message"],
                [(i, msg) for i, msg in enumerate(report.warnings, start=1)],
            )
        )
