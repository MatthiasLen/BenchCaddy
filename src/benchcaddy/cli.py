"""Command-line interface for inspecting BenchCaddy benchmark data.

This module should encapsulate CLI concerns: command registration,
argument validation, orchestration of application services, and terminal
output wiring. Benchmark execution, persistence, formatting primitives,
and statistical calculations should stay in their dedicated modules and
only be coordinated from here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .db import (
    compare_runs,
    compare_suite_runs,
    get_all_run_details,
    get_database_path,
    get_run_details,
    get_selected_run_details,
    get_suite_details,
    get_suite_trend,
    list_suite_summaries,
    set_suite_baseline,
)
from .isolation import NoiseAnalyzer, build_reliability_report, collect_environment_state, get_affinity
from .observability import summarize_observations
from .presentation import (
    dump_json,
    format_interval,
    format_probability,
    format_ratio,
    format_return_error,
    format_return_value,
    format_time_summary,
    format_warning_list,
    json_panel,
    render_table,
    serialize_json,
    summary_panel,
)
from .stats import AnalysisOptions

app = typer.Typer(help="Inspect BenchCaddy benchmark suites.")
console = Console()
REGRESSION_EXIT_CODE = 3

DatabaseOption = Annotated[
    Path | None,
    typer.Option(
        "--database",
        "-d",
        exists=False,
        dir_okay=False,
        help="Path to the BenchCaddy SQLite database.",
    ),
]
CompareConfidenceLevelOption = Annotated[
    float,
    typer.Option(
        "--confidence-level",
        min=0.5,
        max=0.99,
        help="Bootstrap confidence level used for median and delta confidence intervals.",
    ),
]
CompareBootstrapResamplesOption = Annotated[
    int,
    typer.Option(
        "--bootstrap-resamples",
        min=100,
        help="Bootstrap and permutation resample count used for confidence intervals and significance estimates.",
    ),
]
NoiseThresholdOption = Annotated[
    float,
    typer.Option(
        "--noise-threshold",
        min=0.0,
        help="Coefficient-of-variation threshold (std/mean) used to flag noisy runs.",
    ),
]
SignificanceLevelOption = Annotated[
    float,
    typer.Option(
        "--significance-level",
        min=0.001,
        max=0.5,
        help="p-value threshold used when classifying regressions and improvements.",
    ),
]
RegressionThresholdOption = Annotated[
    float,
    typer.Option(
        "--regression-threshold",
        min=0.0,
        help="Practical regression threshold in percent relative to the baseline median.",
    ),
]


@dataclass
class CLIState:
    """Stores process-wide CLI flags shared across Typer commands."""

    verbose: bool = False


_STATE = CLIState()


def _parse_compare_operands(values: list[str], strict: bool) -> tuple[str | None, list[str]]:
    if not values:
        return None, []

    right, *extra = values
    if strict and _as_run_id(right) is None:
        return None, list(dict.fromkeys(values))
    if extra and not strict:
        console.print(f"Unexpected arguments: {' '.join(extra)}")
        raise typer.Exit(code=2)
    return right, list(dict.fromkeys(extra)) if strict else []


def _run_direct_compare(
    left_run_id: int | tuple[int, int],
    right_run_id: int | tuple[int, int],
    database_path: Path,
    analysis_options: AnalysisOptions,
    *,
    strict_keys: list[str],
    use_baseline: bool,
    pin_baseline: bool,
) -> dict[str, object]:
    if strict_keys or use_baseline or pin_baseline:
        console.print("--strict, --use-baseline, and --pin-baseline are only supported for suite comparisons.")
        raise typer.Exit(code=2)

    comparison = compare_runs(left_run_id, right_run_id, database_path, analysis_options=analysis_options)
    if comparison is None:
        console.print(f"Run comparison {left_run_id} vs {right_run_id} was not found in {database_path}.")
        raise typer.Exit(code=1)
    return comparison


def _resolve_compare_strict_keys(
    strict_keys: list[str],
    *,
    strict: bool,
    right: str | None,
    right_run_id: int | tuple[int, int] | None,
    database_path: Path,
) -> list[str]:
    if strict and right_run_id is None:
        console.print("--strict requires a suite comparison with a reference run ID.")
        raise typer.Exit(code=2)
    if strict and right_run_id is not None and not strict_keys:
        reference_run = get_run_details(right_run_id, database_path, include_analysis=False)
        if reference_run is None:
            console.print(f"Reference run '{right}' was not found in {database_path}.")
            raise typer.Exit(code=1)
        return list(reference_run["configuration"].keys())
    return strict_keys


def _validate_suite_compare_options(
    *,
    right_run_id: int | tuple[int, int] | None,
    use_baseline: bool,
    pin_baseline: bool,
) -> None:
    if use_baseline and right_run_id is not None:
        console.print("--use-baseline cannot be combined with an explicit reference run ID.")
        raise typer.Exit(code=2)
    if pin_baseline and right_run_id is None:
        console.print("--pin-baseline requires a suite comparison with a reference run ID.")
        raise typer.Exit(code=2)


def _raise_for_suite_compare_error(
    comparison: dict[str, object] | None,
    *,
    left: str,
    right: str | None,
    database_path: Path,
) -> dict[str, object]:
    if comparison is None:
        console.print(f"Suite '{left}' was not found in {database_path}.")
        raise typer.Exit(code=1)

    error = comparison.get("error")
    if error == "reference_run_not_found":
        console.print(f"Reference run '{right}' was not found in {database_path}.")
        raise typer.Exit(code=1)
    if error == "reference_run_wrong_suite":
        console.print(f"Reference run '{right}' belongs to suite '{comparison['reference_run_suite_name']}', not '{left}'.")
        raise typer.Exit(code=1)
    if error == "strict_requires_reference_run":
        console.print("--strict requires a suite comparison with a reference run ID.")
        raise typer.Exit(code=2)
    if error == "strict_keys_not_found":
        missing_keys = ", ".join(comparison["missing_strict_keys"])
        console.print(f"Strict key(s) {missing_keys} were not found on reference run {comparison['reference_run_display_id']}.")
        raise typer.Exit(code=1)
    if error == "baseline_not_found":
        console.print(f"Suite '{left}' does not have a pinned baseline in {database_path}.")
        raise typer.Exit(code=1)
    return comparison


def _pin_suite_baseline_if_requested(
    *,
    suite_name: str,
    right_run_id: int | tuple[int, int] | None,
    database_path: Path,
    analysis_options: AnalysisOptions,
    pin_baseline: bool,
    emit: bool = True,
) -> dict[str, object] | None:
    if not pin_baseline:
        return None

    pinned = set_suite_baseline(suite_name, right_run_id, database_path, analysis_options=analysis_options)
    if pinned is not None and not pinned.get("error") and emit:
        console.print(
            Panel.fit(
                f"Pinned baseline for {suite_name}: {pinned['display_id']} ({pinned['id']})",
                title="Baseline Updated",
            )
        )
    return pinned


def _parse_percent_option(value: str, *, option_name: str) -> float:
    normalized = value.strip()
    if normalized.endswith("%"):
        normalized = normalized[:-1].strip()
    if not normalized:
        console.print(f"{option_name} requires a numeric percent value.")
        raise typer.Exit(code=2)
    try:
        parsed = float(normalized)
    except ValueError as exc:
        console.print(f"{option_name} must be a number like 5 or 5%.")
        raise typer.Exit(code=2) from exc
    if parsed < 0.0:
        console.print(f"{option_name} must be zero or greater.")
        raise typer.Exit(code=2)
    return parsed


def _resolve_compare_thresholds(
    *,
    regression_threshold: float,
    fail_if_regression: str | None,
) -> tuple[float, float | None]:
    if fail_if_regression is None:
        return regression_threshold, None
    gate_threshold = _parse_percent_option(fail_if_regression, option_name="--fail-if-regression")
    return gate_threshold, gate_threshold


def _gate_run_payload(run: dict[str, object], comparison_analysis: dict[str, object]) -> dict[str, object]:
    return {
        "display_id": run["display_id"],
        "record_id": run.get("record_id", run["id"]),
        "classification": comparison_analysis.get("classification"),
        "percent_change": comparison_analysis.get("percent_change"),
        "regression_probability": comparison_analysis.get("regression_probability"),
        "significance_p_value": comparison_analysis.get("significance_p_value"),
        "warnings": list(comparison_analysis.get("warnings") or ()),
    }


def _direct_gate_payload(comparison: dict[str, object]) -> list[dict[str, object]]:
    comparison_analysis = comparison.get("comparison_analysis") or {}
    if not comparison_analysis.get("regression_detected"):
        return []
    return [_gate_run_payload(comparison["candidate"], comparison_analysis)]


def _suite_gate_payload(comparison: dict[str, object]) -> list[dict[str, object]]:
    basis_run = comparison.get("basis_run")
    basis_id = None if basis_run is None else basis_run.get("id")
    failing_runs: list[dict[str, object]] = []
    for run in comparison["runs"]:
        if run["id"] == basis_id:
            continue
        comparison_analysis = run.get("comparison_analysis") or {}
        if comparison_analysis.get("regression_detected"):
            failing_runs.append(_gate_run_payload(run, comparison_analysis))
    return failing_runs


def _evaluate_compare_gate(
    comparison: dict[str, object],
    *,
    mode: str,
    threshold_percent: float | None,
) -> dict[str, object] | None:
    if threshold_percent is None:
        return None

    failing_runs = _direct_gate_payload(comparison) if mode == "direct" else _suite_gate_payload(comparison)
    return {
        "enabled": True,
        "mode": mode,
        "threshold_percent": threshold_percent,
        "failed": bool(failing_runs),
        "failing_runs": failing_runs,
    }


def _print_compare_gate(gate: dict[str, object]) -> None:
    result = "failed" if gate["failed"] else "passed"
    failing_runs = ", ".join(run["display_id"] for run in gate["failing_runs"])
    console.print(
        summary_panel(
            "CI Gate",
            [
                ("Threshold", f"{float(gate['threshold_percent']):.2f}%"),
                ("Result", result),
                ("Regressing Runs", failing_runs or "-"),
            ],
        )
    )


def _emit_json(payload: dict[str, object]) -> None:
    typer.echo(serialize_json(payload))


def _comparison_title(comparison: dict[str, object]) -> str:
    strict_keys = comparison.get("strict_keys") or []
    if not strict_keys:
        return f"Comparison: {comparison['suite_name']}"
    return f"Comparison: {comparison['suite_name']} (strict: {', '.join(strict_keys)})"


def _best_run(runs: list[dict[str, object]]) -> dict[str, object]:
    return min(runs, key=lambda candidate: (candidate["median_seconds"], candidate["id"]))


def _as_run_id(value: str) -> int | tuple[int, int] | None:
    if "." in value:
        left, dot, right = value.partition(".")
        if dot and left.isdigit() and right.isdigit():
            return (int(left), int(right))
    try:
        return int(value)
    except ValueError:
        return None


def _style_delta(percent_change: float | None) -> Text:
    if percent_change is None:
        return Text("n/a")
    return Text(f"{percent_change:+.2f}%", style="green" if percent_change <= -5.0 else "red" if percent_change >= 5.0 else None)


def _combine_warning_lists(*values: object) -> tuple[str, ...]:
    combined: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            combined.extend(str(item) for item in value)
    return tuple(dict.fromkeys(combined))


def _analysis_options(
    *,
    confidence_level: float,
    bootstrap_resamples: int,
    noise_threshold: float = 0.05,
    significance_level: float = 0.05,
    regression_threshold: float = 5.0,
    window_size: int = 5,
) -> AnalysisOptions:
    return AnalysisOptions(
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        noise_cv_threshold=noise_threshold,
        significance_level=significance_level,
        regression_threshold_percent=regression_threshold,
        drift_window_size=window_size,
    )


def _has_analysis(run: dict[str, object]) -> bool:
    return run.get("analysis") is not None


def _styled(value: object, style: str | None = None) -> Text:
    return Text(str(value), style=style)


def _style_row(values: tuple[object, ...], style: str | None = None) -> tuple[object, ...]:
    return tuple(_styled(value, style) if style else value for value in values)


def _row_style(
    runs: list[dict[str, object]],
    run: dict[str, object],
    *,
    basis_run: dict[str, object] | None,
    highlight_basis: bool,
) -> str | None:
    if basis_run is None:
        return None

    if run["id"] == _best_run(runs)["id"]:
        return "green"

    if highlight_basis and run["id"] == basis_run["id"]:
        return "yellow"

    return None


def _suite_row_style(comparison: dict[str, object], run: dict[str, object]) -> str | None:
    return _row_style(
        comparison["runs"],
        run,
        basis_run=comparison.get("basis_run"),
        highlight_basis=comparison.get("basis_metric_label") == "Reference Median (s)",
    )


def _trend_row_style(trend: dict[str, object], run: dict[str, object]) -> str | None:
    return _row_style(
        trend["runs"],
        run,
        basis_run=trend.get("basis_run"),
        highlight_basis=True,
    )


def _format_optional_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def _styled_run_label(run: dict[str, object], style: str | None) -> Text:
    return _styled(f"{run['display_id']} ({run['id']})", style)


def _run_analysis_panel(run: dict[str, object], title: str = "Statistical Summary") -> Panel:
    analysis = run.get("analysis") or {}
    return summary_panel(
        title,
        [
            ("Median CI (s)", format_interval(run.get("ci_lower_seconds"), run.get("ci_upper_seconds"))),
            ("MAD (s)", _format_optional_seconds(run.get("mad_seconds"))),
            ("CV", format_ratio(run.get("coefficient_of_variation"))),
            ("Warnings", format_warning_list(run.get("noise_warnings"))),
            ("Sample Count", str(analysis.get("sample_count", len(run.get("samples", []))))),
        ],
    )


def _comparison_analysis_panel(comparison_analysis: dict[str, object], title: str = "Statistical Assessment") -> Panel:
    return summary_panel(
        title,
        [
            ("Delta CI (s)", format_interval(comparison_analysis.get("delta_ci_lower_seconds"), comparison_analysis.get("delta_ci_upper_seconds"))),
            ("Regression Probability", format_probability(comparison_analysis.get("regression_probability"))),
            ("Improvement Probability", format_probability(comparison_analysis.get("improvement_probability"))),
            ("p-value", f"{float(comparison_analysis.get('significance_p_value', 0.0)):.4f}"),
            ("Practical Threshold (s)", _format_optional_seconds(comparison_analysis.get("practical_threshold_seconds"))),
            ("Classification", str(comparison_analysis.get("classification", "-"))),
            ("Warnings", format_warning_list(comparison_analysis.get("warnings"))),
        ],
    )


def _best_vs_reference_panel(comparison: dict[str, object]) -> Panel | None:
    if comparison.get("basis_metric_label") != "Reference Median (s)":
        return None

    basis_run = comparison.get("basis_run")
    if basis_run is None:
        return None
    best_run = _best_run(comparison["runs"])

    scope = f"strict: {', '.join(comparison.get('strict_keys', []))}" if comparison.get("strict_keys") else "full suite"
    if best_run["id"] == basis_run["id"]:
        return summary_panel(
            "Best Run vs Reference",
            [
                ("Reference Run", _styled_run_label(basis_run, "green")),
                ("Status", "Reference is already the fastest run in this comparison scope."),
                ("Scope", scope),
            ],
        )

    best_analysis = best_run.get("comparison_analysis") or {}
    return summary_panel(
        "Best Run vs Reference",
        [
            ("Reference Run", _styled_run_label(basis_run, "yellow")),
            ("Best Run", _styled_run_label(best_run, "green")),
            ("Scope", scope),
            ("Delta CI (s)", format_interval(best_analysis.get("delta_ci_lower_seconds"), best_analysis.get("delta_ci_upper_seconds"))),
            ("Improvement Probability", format_probability(best_analysis.get("improvement_probability"))),
            ("Regression Probability", format_probability(best_analysis.get("regression_probability"))),
            ("p-value", f"{float(best_analysis.get('significance_p_value', 0.0)):.4f}"),
            ("Classification", str(best_analysis.get("classification", "-"))),
            ("Warnings", format_warning_list(best_analysis.get("warnings"))),
        ],
    )


def _trend_run_warnings(run: dict[str, object]) -> tuple[str, ...]:
    return _combine_warning_lists(
        run.get("noise_warnings"),
        (run.get("vs_baseline") or {}).get("warnings"),
        (run.get("drift_analysis") or {}).get("warnings"),
    )


def _trend_basis_panel(trend: dict[str, object]) -> Panel:
    basis_run = trend["basis_run"]
    return summary_panel(
        f"Trend Basis: {trend['suite_name']}",
        [
            ("Source", str(trend.get("basis_source", "latest"))),
            ("Run ID", _styled(basis_run["display_id"], "yellow")),
            ("Record ID", _styled(basis_run["id"], "yellow")),
            ("Configuration", dump_json(trend.get("config_filter"))),
            ("Median CI (s)", format_interval(basis_run.get("ci_lower_seconds"), basis_run.get("ci_upper_seconds"))),
        ],
    )


def _trend_delta_value(run: dict[str, object]) -> str:
    vs_baseline = run["vs_baseline"]
    delta_value = f"{vs_baseline['delta_seconds']:+.6f}"
    if vs_baseline.get("percent_change") is not None:
        return f"{delta_value} ({vs_baseline['percent_change']:+.2f}%)"
    return delta_value


def _trend_sparkline(values: list[float]) -> str:
    if not values:
        return "-"

    blocks = "▁▂▃▄▅▆▇█"
    if len(values) == 1:
        return blocks[0]

    low = min(values)
    high = max(values)
    if abs(high - low) <= 1e-12:
        return blocks[0] * len(values)

    scale = len(blocks) - 1
    return "".join(blocks[min(scale, max(0, round(((value - low) / (high - low)) * scale)))] for value in values)


def _trend_summary_signal(comparison: dict[str, object] | None) -> str:
    if comparison is None:
        return "n/a"

    classification = {
        "regressing": "reg",
        "improving": "imp",
        "stable": "stbl",
        "noisy": "noisy",
    }.get(str(comparison.get("classification", "-")), str(comparison.get("classification", "-")))
    percent_change = comparison.get("percent_change")
    if percent_change is not None:
        return f"{classification} {float(percent_change):+.1f}%"

    delta_seconds = comparison.get("delta_seconds")
    if delta_seconds is None:
        return classification
    return f"{classification} {float(delta_seconds):+.4f}s"


def _trend_headroom(summary: dict[str, object]) -> str:
    comparison = summary["latest_vs_best"]
    percent_change = comparison.get("percent_change")
    if percent_change is not None:
        return "at best" if abs(float(percent_change)) <= 0.005 else f"{float(percent_change):+.2f}%"

    delta_seconds = comparison.get("delta_seconds")
    return "-" if delta_seconds is None else f"{float(delta_seconds):+.6f}s"


def _trend_latest_value(summary: dict[str, object]) -> str:
    latest_run = summary["latest_run"]
    return f"{latest_run['median_seconds']:.6f}"


def _print_trend_summary(trend: dict[str, object]) -> None:
    limit = trend.get("limit")
    mode_detail = "Per-configuration summary"
    if limit is not None:
        mode_detail = f"Per-configuration summary (most recent {int(limit)} runs per configuration)"

    console.print(
        summary_panel(
            f"Trend Summary: {trend['suite_name']}",
            [
                ("Configurations", str(trend["configuration_count"])),
                ("Mode", mode_detail),
                ("Hint", "Pass a baseline run ID to inspect one configuration timeline."),
            ],
        )
    )

    table = Table(title=f"Trend Summary: {trend['suite_name']}", pad_edge=False, collapse_padding=True)
    table.add_column("Config", overflow="fold", max_width=14)
    table.add_column("Runs", justify="right", no_wrap=True, max_width=4)
    table.add_column("Trend", no_wrap=True, min_width=12, max_width=12)
    table.add_column("Vs 1st", no_wrap=True, min_width=11, max_width=11)
    table.add_column("Recent", no_wrap=True, min_width=11, max_width=11)
    table.add_column("Best", justify="right", no_wrap=True, max_width=8)
    table.add_column("Latest", justify="right", no_wrap=True, max_width=8)

    for summary in trend["config_summaries"]:
        run_count = summary["run_count"]
        total_run_count = summary.get("total_run_count", run_count)
        count_label = str(run_count) if run_count == total_run_count else f"{run_count}/{total_run_count}"
        configuration_label = ", ".join(
            f"{key}={summary['configuration'][key]}"
            for key in sorted(summary["configuration"])
        )
        table.add_row(
            configuration_label,
            count_label,
            _trend_sparkline(summary["median_series"]),
            _trend_summary_signal(summary["latest_vs_first"]),
            _trend_summary_signal(summary.get("recent_vs_window")),
            _trend_headroom(summary),
            _trend_latest_value(summary),
        )

    console.print(table)
    console.print(
        summary_panel(
            "Label Guide",
            [
                ("Trend", "sparkline of median timings from oldest to newest run for that configuration"),
                ("stbl", "no meaningful shift detected relative to the comparison basis"),
                ("noisy", "variance or confidence interval is too wide to call the direction cleanly"),
                ("reg", "meaningful slowdown detected"),
                ("imp", "meaningful speedup detected"),
                ("Vs 1st", "latest run compared with the first recorded run for that configuration"),
                ("Recent", "latest run compared with the recent trailing window for that configuration"),
                ("Best", "latest run compared with the best observed run for that configuration"),
            ],
        )
    )


def _trend_row(run: dict[str, object], *, verbose: bool) -> tuple[object, ...]:
    row: list[object] = [
        run["display_id"],
        f"{run['median_seconds']:.6f}",
        format_interval(run.get("ci_lower_seconds"), run.get("ci_upper_seconds")),
        _trend_delta_value(run),
        str(run.get("drift_status", "stable")),
        "basis" if run.get("is_basis") else str(run["vs_baseline"].get("classification", "stable")),
        run["created_at"],
    ]
    if verbose:
        row.extend(
            [
                run["id"],
                format_warning_list(_trend_run_warnings(run)),
            ]
        )
    return tuple(str(value) for value in row)


def _trend_warning_rows(runs: list[dict[str, object]]) -> list[tuple[object, object]]:
    return [(run["display_id"], format_warning_list(warnings)) for run in runs if (warnings := _trend_run_warnings(run))]


def _suite_findings_panel(comparison: dict[str, object]) -> Panel:
    regressing = [run["display_id"] for run in comparison["runs"] if (run.get("comparison_analysis") or {}).get("regression_detected")]
    noisy = [run["display_id"] for run in comparison["runs"] if run.get("noise_warnings")]
    return summary_panel(
        "Statistical Findings",
        [
            ("Regressing Runs", ", ".join(regressing) if regressing else "-"),
            ("Noisy Runs", ", ".join(noisy) if noisy else "-"),
            ("Basis Source", str(comparison.get("basis_source", "best"))),
        ],
    )


def _render_observation_table(observations: list[dict[str, object]], title: str) -> Table:
    summary = summarize_observations(observations)

    return render_table(
        title,
        ["Label", ("Calls", "right"), ("Mean +- Std (s)", "right"), ("Total (s)", "right")],
        [
            (
                label,
                stats.calls,
                format_time_summary(stats.mean_seconds, stats.std_seconds),
                f"{stats.total_seconds:.6f}",
            )
            for label, stats in summary.items()
        ],
    )


def _run_table_columns(*, include_suite: bool = False, include_target: bool = False) -> list[object]:
    columns: list[object] = [("Run ID", "right"), ("Record ID", "right")]
    if include_suite:
        columns.append("Suite")
    if include_target:
        columns.append("Target")
    columns.extend(
        [
            "Configuration",
            ("Mean +- Std (s)", "right"),
            "Return Value",
            ("Samples", "right"),
            "Recorded At",
        ]
    )
    return columns


def _run_table_row(
    run: dict[str, object],
    *,
    include_suite: bool = False,
    include_target: bool = False,
) -> tuple[object, ...]:
    row: list[object] = [run["display_id"], run.get("record_id", run["id"])]
    if include_suite:
        row.append(run["suite_name"])
    if include_target:
        row.append(run["target_name"])
    row.extend(
        [
            dump_json(run["configuration"]),
            format_time_summary(run.get("mean_seconds"), run.get("std_seconds")),
            format_return_value(run.get("target_return_value"), compact=True),
            len(run["samples"]),
            run["created_at"],
        ]
    )
    return tuple(row)


def _render_run_table(
    title: str,
    runs: list[dict[str, object]],
    *,
    include_suite: bool = False,
    include_target: bool = False,
) -> Table:
    return render_table(
        title,
        _run_table_columns(include_suite=include_suite, include_target=include_target),
        [_run_table_row(run, include_suite=include_suite, include_target=include_target) for run in runs],
    )


def _observed_timing_rows(
    runs: list[dict[str, object]],
    *,
    include_record_id: bool = False,
) -> list[tuple[object, ...]]:
    return [
        (
            run["display_id"],
            *([run.get("record_id", run["id"])] if include_record_id else []),
            label,
            stats.calls,
            format_time_summary(stats.mean_seconds, stats.std_seconds),
        )
        for run in runs
        for label, stats in summarize_observations(run["observations"]).items()
    ]


def _show_run(run: dict[str, object]) -> None:
    detail_rows: list[tuple[object, object]] = [
        ("Run ID", run["display_id"]),
        ("Record ID", run["id"]),
        ("Sweep ID", run["sweep_id"]),
        ("Run Index", run["run_index"]),
        ("Suite", run["suite_name"]),
        ("Target", run["target_name"]),
        ("Configuration", dump_json(run["configuration"])),
        ("Mean +- Std (s)", format_time_summary(run.get("mean_seconds"), run.get("std_seconds"))),
        ("Min (s)", _format_optional_seconds(run.get("min_seconds"))),
        ("Max (s)", _format_optional_seconds(run.get("max_seconds"))),
    ]
    if _has_analysis(run):
        detail_rows.extend(
            [
                ("Median CI (s)", format_interval(run.get("ci_lower_seconds"), run.get("ci_upper_seconds"))),
                ("MAD (s)", _format_optional_seconds(run.get("mad_seconds"))),
                ("CV", format_ratio(run.get("coefficient_of_variation"))),
                ("Warnings", format_warning_list(run.get("noise_warnings"))),
            ]
        )
    detail_rows.extend(
        [
            ("Return Value", format_return_value(run.get("target_return_value"), compact=True)),
            ("Samples", len(run["samples"])),
            ("Recorded At", run["created_at"]),
        ]
    )
    console.print(
        render_table(
            f"Run: {run['display_id']}",
            ["Field", "Value"],
            detail_rows,
        )
    )
    if _has_analysis(run):
        console.print(_run_analysis_panel(run))
    console.print(_render_observation_table(run["observations"], title="Observed Timings"))
    console.print(json_panel("Environment", run["environment"], indent=2))


def _show_suite(details: dict[str, object]) -> None:
    console.print(_render_run_table(f"Suite: {details['suite_name']}", details["runs"]))
    if details.get("baseline_run") is not None:
        baseline_run = details["baseline_run"]
        rows: list[tuple[object, object]] = [
            ("Run ID", _styled(baseline_run["display_id"], "yellow")),
            ("Record ID", _styled(baseline_run["id"], "yellow")),
        ]
        if _has_analysis(baseline_run):
            rows.append(("Median CI (s)", format_interval(baseline_run.get("ci_lower_seconds"), baseline_run.get("ci_upper_seconds"))))
        rows.append(("Configuration", dump_json(baseline_run["configuration"])))
        console.print(
            summary_panel(
                "Pinned Baseline",
                rows,
            )
        )
    console.print(
        render_table(
            f"Observed Timings: {details['suite_name']}",
            [("Run ID", "right"), "Label", ("Calls", "right"), ("Mean +- Std (s)", "right")],
            _observed_timing_rows(details["runs"]),
        )
    )
    if details["environment"] is not None:
        console.print(json_panel("Environment", details["environment"], indent=2))
    if _STATE.verbose:
        for run in details["runs"]:
            console.print(_render_observation_table(run["observations"], title=f"Observed Timings for Run {run['display_id']}"))


def _show_selected_runs(runs: list[dict[str, object]]) -> None:
    console.print(_render_run_table("Selected Runs", runs, include_suite=True, include_target=True))
    console.print(
        render_table(
            "Observed Timings: Selected Runs",
            [("Run ID", "right"), ("Record ID", "right"), "Label", ("Calls", "right"), ("Mean +- Std (s)", "right")],
            _observed_timing_rows(runs, include_record_id=True),
        )
    )


def _show_all_runs(runs: list[dict[str, object]]) -> None:
    console.print(_render_run_table("All Runs", runs, include_suite=True))


def _require_run_id(identifier: str) -> int | tuple[int, int]:
    run_id = _as_run_id(identifier)
    if run_id is None:
        console.print(f"'{identifier}' is not a valid run ID.")
        raise typer.Exit(code=1)
    return run_id


def _finalize_compare_result(
    comparison: dict[str, object],
    *,
    mode: str,
    threshold_percent: float | None,
    json_output: bool,
    render: Callable[[dict[str, object]], None],
) -> None:
    gate = _evaluate_compare_gate(comparison, mode=mode, threshold_percent=threshold_percent)
    if gate is not None:
        comparison["gate"] = gate
    if json_output:
        _emit_json(comparison)
    else:
        render(comparison)
        if gate is not None:
            _print_compare_gate(gate)
    if gate is not None and gate["failed"]:
        raise typer.Exit(code=REGRESSION_EXIT_CODE)


def _print_run_comparison(
    comparison: dict[str, object],
) -> None:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    baseline_style = "green" if baseline["median_seconds"] <= candidate["median_seconds"] else None
    candidate_style = "green" if candidate["median_seconds"] <= baseline["median_seconds"] else None
    console.print(
        render_table(
            f"Run Comparison: {baseline['display_id']} -> {candidate['display_id']}",
            ["Field", "Baseline", "Candidate"],
            [
                ("Suite", _styled(baseline["suite_name"], baseline_style), _styled(candidate["suite_name"], candidate_style)),
                ("Run ID", _styled(baseline["display_id"], baseline_style), _styled(candidate["display_id"], candidate_style)),
                ("Record ID", _styled(baseline["id"], baseline_style), _styled(candidate["id"], candidate_style)),
                *[
                    (
                        key,
                        _styled(dump_json(baseline["configuration"].get(key)), baseline_style),
                        _styled(dump_json(candidate["configuration"].get(key)), candidate_style),
                    )
                    for key in sorted(set(baseline["configuration"]) | set(candidate["configuration"]))
                ],
                ("Median (s)", _styled(f"{baseline['median_seconds']:.6f}", baseline_style), _styled(f"{candidate['median_seconds']:.6f}", candidate_style)),
                (
                    "Mean +- Std (s)",
                    _styled(format_time_summary(baseline.get("mean_seconds"), baseline.get("std_seconds")), baseline_style),
                    _styled(format_time_summary(candidate.get("mean_seconds"), candidate.get("std_seconds")), candidate_style),
                ),
                (
                    "Min (s)",
                    _styled(_format_optional_seconds(baseline.get("min_seconds")), baseline_style),
                    _styled(_format_optional_seconds(candidate.get("min_seconds")), candidate_style),
                ),
                (
                    "Max (s)",
                    _styled(_format_optional_seconds(baseline.get("max_seconds")), baseline_style),
                    _styled(_format_optional_seconds(candidate.get("max_seconds")), candidate_style),
                ),
                ("Median Delta (s)", "", f"{comparison['delta_seconds']:.6f}"),
                ("Median Percent Change", "", _style_delta(comparison["percent_change"])),
                (
                    "Return Value",
                    _styled(format_return_value(baseline.get("target_return_value"), compact=True), baseline_style),
                    _styled(format_return_value(candidate.get("target_return_value"), compact=True), candidate_style),
                ),
                ("Return Error", "", format_return_error(comparison.get("target_return_relative_error"))),
            ],
        )
    )
    console.print(_comparison_analysis_panel(comparison["comparison_analysis"]))

    if comparison["observation_rows"]:
        console.print(
            render_table(
                "Observed Timing Diff",
                ["Label", ("Baseline (s)", "right"), ("Candidate (s)", "right"), ("Delta (s)", "right")],
                [
                    (
                        row["label"],
                        "-" if row["baseline_mean_seconds"] is None else format_time_summary(row["baseline_mean_seconds"], row["baseline_std_seconds"]),
                        "-" if row["candidate_mean_seconds"] is None else format_time_summary(row["candidate_mean_seconds"], row["candidate_std_seconds"]),
                        _format_optional_seconds(row["delta_seconds"]),
                    )
                    for row in comparison["observation_rows"]
                ],
            )
        )


def _table_row(values: tuple[object, ...]) -> tuple[object, ...]:
    return tuple(value if isinstance(value, Text) else str(value) for value in values)


def _suite_comparison_row(run: dict[str, object], *, verbose: bool) -> tuple[object, ...]:
    comparison_analysis = run.get("comparison_analysis") or {}
    row: list[object] = [
        run["display_id"],
        run["id"],
        dump_json(run["configuration"]),
        format_time_summary(run.get("mean_seconds"), run.get("std_seconds")),
        f"{run['delta_seconds']:.6f}",
        "n/a" if run["slowdown_factor"] is None else f"{run['slowdown_factor']:.2f}x",
        format_return_value(run.get("target_return_value"), compact=True),
        format_return_error(run.get("target_return_relative_error")),
    ]
    if verbose:
        row.extend(
            [
                str(comparison_analysis.get("classification", "-")),
                format_interval(run.get("ci_lower_seconds"), run.get("ci_upper_seconds")),
                f"{float(comparison_analysis.get('significance_p_value', 0.0)):.4f}",
                run["sample_count"],
                run["created_at"],
            ]
        )
    return tuple(row)


def _suite_comparison_table(comparison: dict[str, object], *, verbose: bool) -> Table:
    table = Table(title=_comparison_title(comparison), pad_edge=False, collapse_padding=True)
    table.add_column("Run ID", justify="right", no_wrap=True, min_width=4, max_width=4)
    table.add_column("Record ID", justify="right", no_wrap=True, min_width=7, max_width=7)
    table.add_column("Configuration", overflow="ellipsis", max_width=16)
    table.add_column("Mean +- Std (s)", justify="right", no_wrap=True, max_width=18)
    table.add_column(str(comparison["delta_column_label"]), justify="right", no_wrap=True, max_width=12)
    table.add_column(str(comparison["ratio_column_label"]), justify="right", no_wrap=True, max_width=6)
    table.add_column("Return Value", overflow="ellipsis", no_wrap=True, max_width=16)
    table.add_column("Return Error", justify="right", no_wrap=True, max_width=12)
    if verbose:
        table.add_column("Status", no_wrap=True, max_width=10)
        table.add_column("Median CI (s)", justify="right", no_wrap=True, max_width=24)
        table.add_column("p-value", justify="right", no_wrap=True, max_width=8)
        table.add_column("Samples", justify="right", no_wrap=True)
        table.add_column("Recorded At", overflow="ellipsis", no_wrap=True, max_width=16)

    for run in comparison["runs"]:
        styled_row = _style_row(
            _suite_comparison_row(run, verbose=verbose),
            _suite_row_style(comparison, run),
        )
        table.add_row(*_table_row(styled_row))
    return table


def _stored_return_metrics_panel(comparison: dict[str, object]) -> Panel | None:
    if not any(run.get("target_return_value") is not None for run in comparison["runs"]):
        return None
    return summary_panel(
        "Stored Return Metrics",
        [
            ("Return Value", "shown per run"),
            ("Return Error", "relative to the comparison basis"),
        ],
    )


def _comparison_basis_panel(comparison: dict[str, object]) -> Panel | None:
    if comparison["basis_median_seconds"] is None:
        return None

    basis_run = comparison["basis_run"]
    basis_style = "yellow" if comparison.get("basis_metric_label") == "Reference Median (s)" else "green"
    return summary_panel(
        "Comparison Basis",
        [
            ("Run ID", _styled(basis_run["display_id"], basis_style)),
            ("Record ID", _styled(basis_run["id"], basis_style)),
            (str(comparison["basis_metric_label"]), f"{basis_run['median_seconds']:.6f}"),
            ("Mean +- Std (s)", format_time_summary(basis_run.get("mean_seconds"), basis_run.get("std_seconds"))),
            ("Median CI (s)", format_interval(basis_run.get("ci_lower_seconds"), basis_run.get("ci_upper_seconds"))),
            ("Return Value", format_return_value(basis_run.get("target_return_value"), compact=True)),
            ("Return Error", "relative to this basis run"),
        ],
    )


def _print_suite_comparison(comparison: dict[str, object]) -> None:
    console.print(_suite_comparison_table(comparison, verbose=_STATE.verbose))
    console.print(_suite_findings_panel(comparison))
    best_vs_reference = _best_vs_reference_panel(comparison)
    if best_vs_reference is not None:
        console.print(best_vs_reference)
    stored_return_metrics = _stored_return_metrics_panel(comparison)
    if stored_return_metrics is not None:
        console.print(stored_return_metrics)

    comparison_basis = _comparison_basis_panel(comparison)
    if comparison_basis is not None:
        console.print(comparison_basis)


def _print_trend(trend: dict[str, object]) -> None:
    console.print(_trend_basis_panel(trend))

    table = Table(title=f"Trend: {trend['suite_name']}", pad_edge=False, collapse_padding=True)
    table.add_column("Run ID", justify="right", no_wrap=True, min_width=4, max_width=4)
    table.add_column("Median (s)", justify="right", no_wrap=True, max_width=12)
    table.add_column("Median CI (s)", justify="right", no_wrap=True, max_width=24)
    table.add_column("Delta", justify="right", no_wrap=True, max_width=18)
    table.add_column("Drift", no_wrap=True, max_width=10)
    table.add_column("Status", no_wrap=True, max_width=10)
    table.add_column("Recorded At", overflow="ellipsis", no_wrap=True, max_width=16)
    if _STATE.verbose:
        table.add_column("Record ID", justify="right", no_wrap=True, min_width=7, max_width=7)
        table.add_column("Warnings", overflow="fold")

    for run in trend["runs"]:
        table.add_row(*_style_row(_trend_row(run, verbose=_STATE.verbose), _trend_row_style(trend, run)))

    console.print(table)
    if _STATE.verbose:
        warning_rows = _trend_warning_rows(trend["runs"])
        if warning_rows:
            console.print(render_table("Trend Warning Details", [("Run ID", "right"), "Warnings"], warning_rows))


@app.callback()
def callback(
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show additional detail in command output.",
        ),
    ] = False,
) -> None:
    _STATE.verbose = verbose


@app.command("list")
def list_command(
    database: DatabaseOption = None,
) -> None:
    database_path = get_database_path(database)
    summaries = list_suite_summaries(database_path)
    if not summaries:
        console.print(f"No suites found in {database_path}.")
        raise typer.Exit()

    console.print(
        render_table(
            f"BenchCaddy suites ({database_path})",
            ["Suite", "Target", "Observation Labels", ("Runs", "right"), "Last Run"],
            [
                (
                    summary["suite_name"],
                    summary["target_name"],
                    ", ".join(summary["observation_labels"]) or "-",
                    summary["run_count"],
                    summary["last_run_at"],
                )
                for summary in summaries
            ],
        )
    )
    if _STATE.verbose:
        console.print(Panel.fit(str(database_path), title="Database"))


@app.command("show", help="Inspect all recorded runs, a suite, or specific run IDs. When a suite has a pinned baseline, it is shown in the suite view.")
def show_command(
    identifiers: Annotated[
        list[str] | None,
        typer.Argument(
            help="Suite name or one or more run IDs to inspect (for example 3.2 5 7.1). Omit identifiers to list all recorded runs.",
        ),
    ] = None,
    skip_stats: Annotated[
        bool,
        typer.Option(
            "--no-stats",
            help="Skip per-run statistical analysis when showing run or suite details for a faster view.",
        ),
    ] = False,
    confidence_level: Annotated[
        float,
        typer.Option(
            "--confidence-level",
            min=0.5,
            max=0.99,
            help="Bootstrap confidence level used for per-run median confidence intervals.",
        ),
    ] = 0.95,
    bootstrap_resamples: Annotated[
        int,
        typer.Option(
            "--bootstrap-resamples",
            min=100,
            help="Bootstrap resample count used for per-run median confidence intervals.",
        ),
    ] = 2000,
    database: DatabaseOption = None,
) -> None:
    database_path = get_database_path(database)
    analysis_options = (
        None
        if skip_stats
        else _analysis_options(
            confidence_level=confidence_level,
            bootstrap_resamples=bootstrap_resamples,
        )
    )

    if not identifiers:
        _show_all_runs(get_all_run_details(database_path))
        return

    if len(identifiers) == 1:
        identifier = identifiers[0]
        run_id = _as_run_id(identifier)
        if run_id is not None:
            run = get_run_details(
                run_id,
                database_path,
                analysis_options=analysis_options,
                include_analysis=not skip_stats,
            )
            if run is None:
                console.print(f"Run '{identifier}' was not found in {database_path}.")
                raise typer.Exit(code=1)
            _show_run(run)
            return

        details = get_suite_details(
            identifier,
            database_path,
            analysis_options=analysis_options,
            include_analysis=not skip_stats,
        )
        if details is None:
            console.print(f"Suite '{identifier}' was not found in {database_path}.")
            raise typer.Exit(code=1)
        _show_suite(details)
        return

    run_ids = [_require_run_id(identifier) for identifier in identifiers]

    runs = get_selected_run_details(run_ids, database_path)
    if runs is None:
        console.print(f"One or more runs were not found in {database_path}.")
        raise typer.Exit(code=1)
    _show_selected_runs(runs)


@app.command("compare", help="Compare two runs directly, compare a suite to its best run, or compare a suite to a selected or pinned reference run.")
def compare_command(
    left: Annotated[
        str,
        typer.Argument(
            help="Suite name for suite comparison, or the baseline run ID for a direct run-to-run comparison.",
        ),
    ],
    operands: Annotated[
        list[str] | None,
        typer.Argument(
            help=(
                "For suite comparison: optional reference run ID, then strict config keys when "
                "--strict is used. With --strict and no trailing keys, BenchCaddy matches the "
                "reference run's full configuration. For direct run comparison: the candidate "
                "run ID."
            ),
        ),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            "-s",
            help=(
                "Restrict suite comparison to runs whose configuration matches the reference "
                "run for the given trailing config keys. If no keys are provided, all "
                "reference run configuration keys are used."
            ),
        ),
    ] = False,
    use_baseline: Annotated[
        bool,
        typer.Option(
            "--use-baseline",
            help="Use the pinned suite baseline as the comparison reference instead of the suite's best run.",
        ),
    ] = False,
    pin_baseline: Annotated[
        bool,
        typer.Option(
            "--pin-baseline",
            help="Persist the supplied suite reference run as the suite baseline for future show, compare, and trend commands.",
        ),
    ] = False,
    confidence_level: CompareConfidenceLevelOption = 0.95,
    bootstrap_resamples: CompareBootstrapResamplesOption = 2000,
    noise_threshold: NoiseThresholdOption = 0.05,
    significance_level: SignificanceLevelOption = 0.05,
    regression_threshold: RegressionThresholdOption = 5.0,
    fail_if_regression: Annotated[
        str | None,
        typer.Option(
            "--fail-if-regression",
            metavar="PERCENT",
            help="Fail with exit code 3 when compare detects a regression at the given practical threshold percent (for example 5 or 5%).",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit machine-readable JSON output for this comparison.",
        ),
    ] = False,
    database: DatabaseOption = None,
) -> None:
    right, strict_keys = _parse_compare_operands(operands, strict)
    database_path = get_database_path(database)
    effective_regression_threshold, gate_threshold = _resolve_compare_thresholds(
        regression_threshold=regression_threshold,
        fail_if_regression=fail_if_regression,
    )
    analysis_options = _analysis_options(
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        noise_threshold=noise_threshold,
        significance_level=significance_level,
        regression_threshold=effective_regression_threshold,
    )
    left_run_id = _as_run_id(left)
    right_run_id = _as_run_id(right) if right is not None else None
    comparison_mode = "direct" if left_run_id is not None and right_run_id is not None else "suite"
    if left_run_id is not None and right_run_id is not None:
        comparison = _run_direct_compare(
            left_run_id,
            right_run_id,
            database_path,
            analysis_options,
            strict_keys=strict_keys,
            use_baseline=use_baseline,
            pin_baseline=pin_baseline,
        )
        comparison["comparison_mode"] = comparison_mode
        _finalize_compare_result(
            comparison,
            mode=comparison_mode,
            threshold_percent=gate_threshold,
            json_output=json_output,
            render=_print_run_comparison,
        )
        return

    _validate_suite_compare_options(
        right_run_id=right_run_id,
        use_baseline=use_baseline,
        pin_baseline=pin_baseline,
    )
    strict_keys = _resolve_compare_strict_keys(
        strict_keys,
        strict=strict,
        right=right,
        right_run_id=right_run_id,
        database_path=database_path,
    )

    comparison = _raise_for_suite_compare_error(
        compare_suite_runs(
            left,
            right_run_id,
            strict_keys,
            database_path,
            analysis_options=analysis_options,
            use_pinned_baseline=use_baseline,
        ),
        left=left,
        right=right,
        database_path=database_path,
    )
    comparison["comparison_mode"] = comparison_mode
    pinned = _pin_suite_baseline_if_requested(
        suite_name=left,
        right_run_id=right_run_id,
        database_path=database_path,
        analysis_options=analysis_options,
        pin_baseline=pin_baseline,
        emit=not json_output,
    )
    if pinned is not None and json_output:
        comparison["baseline_update"] = pinned

    _finalize_compare_result(
        comparison,
        mode=comparison_mode,
        threshold_percent=gate_threshold,
        json_output=json_output,
        render=_print_suite_comparison,
    )


@app.command(
    "trend",
    help=(
        "Inspect one suite configuration over time. With a positional baseline run, trend shows the matching configuration timeline. "
        "Without a baseline, mixed suites show a compact per-configuration trend summary."
    ),
)
def trend_command(
    suite_name: Annotated[
        str,
        typer.Argument(help="Suite name to inspect as a time-series trend."),
    ],
    baseline: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Optional baseline run ID to anchor a single-configuration timeline. "
                "If omitted, trend shows a mixed-suite summary when multiple configurations are present."
            ),
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Limit the trend output to the most recent matching runs.",
        ),
    ] = None,
    window: Annotated[
        int,
        typer.Option(
            "--window",
            min=2,
            help="Rolling window size used for drift analysis.",
        ),
    ] = 5,
    confidence_level: CompareConfidenceLevelOption = 0.95,
    bootstrap_resamples: CompareBootstrapResamplesOption = 2000,
    noise_threshold: NoiseThresholdOption = 0.05,
    significance_level: SignificanceLevelOption = 0.05,
    regression_threshold: RegressionThresholdOption = 5.0,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit machine-readable JSON output for this trend report.",
        ),
    ] = False,
    database: DatabaseOption = None,
) -> None:
    database_path = get_database_path(database)
    baseline_run_id = None
    if baseline is not None:
        baseline_run_id = _require_run_id(baseline)

    analysis_options = _analysis_options(
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        noise_threshold=noise_threshold,
        significance_level=significance_level,
        regression_threshold=regression_threshold,
        window_size=window,
    )
    trend = get_suite_trend(
        suite_name,
        database_path,
        analysis_options=analysis_options,
        baseline_run_id=baseline_run_id,
        limit=limit,
    )
    if trend is None:
        console.print(f"Suite '{suite_name}' was not found in {database_path}.")
        raise typer.Exit(code=1)
    if trend.get("error") == "reference_run_not_found":
        console.print(f"Reference run '{baseline}' was not found in {database_path}.")
        raise typer.Exit(code=1)
    if trend.get("error") == "reference_run_wrong_suite":
        console.print(f"Reference run '{baseline}' belongs to suite '{trend['reference_run_suite_name']}', not '{suite_name}'.")
        raise typer.Exit(code=1)
    if trend.get("mode") != "summary" and trend.get("basis_run") is None:
        console.print(f"Suite '{suite_name}' does not have any recorded runs in {database_path}.")
        raise typer.Exit(code=1)

    if json_output:
        _emit_json(trend)
        return

    if trend.get("mode") == "summary":
        _print_trend_summary(trend)
        return

    _print_trend(trend)


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
            help="Emit machine-readable JSON output.",
        ),
    ] = False,
) -> None:
    env = collect_environment_state()
    noise = NoiseAnalyzer().analyze(iterations=noise_iterations)
    report = build_reliability_report(environment=env, noise=noise)
    affinity = get_affinity()

    if json_output:
        _emit_json(
            {
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
            }
        )
        return

    stat_style = _quality_style(report.timing_stability)
    env_style = _quality_style(report.environmental_quality)
    console.print(
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
        console.print(
            render_table(
                "Warnings",
                ["#", "Message"],
                [(i, msg) for i, msg in enumerate(report.warnings, start=1)],
            )
        )


main = app
