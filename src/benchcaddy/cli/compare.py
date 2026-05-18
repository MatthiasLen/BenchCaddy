from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..db import compare_runs, compare_suite_runs, get_database_path, get_run_details, set_suite_baseline
from ..presentation import (
    dump_json,
    format_interval,
    format_probability,
    format_return_error,
    format_return_value,
    format_time_summary,
    format_warning_list,
    render_table,
    summary_panel,
)
from ..stats import AnalysisOptions
from ._rendering import _best_run, _format_optional_seconds, _row_style, _style_row, _styled
from ._shared import (
    _STATE,
    REGRESSION_EXIT_CODE,
    CompareBootstrapResamplesOption,
    CompareConfidenceLevelOption,
    DatabaseOption,
    NoiseThresholdOption,
    RegressionThresholdOption,
    SignificanceLevelOption,
    _analysis_options,
    _as_run_id,
    _console,
    _emit_json,
    app,
)


def _parse_compare_operands(values: list[str] | None, strict: bool) -> tuple[str | None, list[str]]:
    if not values:
        return None, []

    right, *extra = values
    if strict and _as_run_id(right) is None:
        return None, list(dict.fromkeys(values))
    if extra and not strict:
        _console().print(f"Unexpected arguments: {' '.join(extra)}")
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
        _console().print("--strict, --use-baseline, and --pin-baseline are only supported for suite comparisons.")
        raise typer.Exit(code=2)

    comparison = compare_runs(left_run_id, right_run_id, database_path, analysis_options=analysis_options)
    if comparison is None:
        _console().print(f"Run comparison {left_run_id} vs {right_run_id} was not found in {database_path}.")
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
        _console().print("--strict requires a suite comparison with a reference run ID.")
        raise typer.Exit(code=2)
    if strict and right_run_id is not None and not strict_keys:
        reference_run = get_run_details(right_run_id, database_path, include_analysis=False)
        if reference_run is None:
            _console().print(f"Reference run '{right}' was not found in {database_path}.")
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
        _console().print("--use-baseline cannot be combined with an explicit reference run ID.")
        raise typer.Exit(code=2)
    if pin_baseline and right_run_id is None:
        _console().print("--pin-baseline requires a suite comparison with a reference run ID.")
        raise typer.Exit(code=2)


def _raise_for_suite_compare_error(
    comparison: dict[str, object] | None,
    *,
    left: str,
    right: str | None,
    database_path: Path,
) -> dict[str, object]:
    if comparison is None:
        _console().print(f"Suite '{left}' was not found in {database_path}.")
        raise typer.Exit(code=1)

    error = comparison.get("error")
    if error == "reference_run_not_found":
        _console().print(f"Reference run '{right}' was not found in {database_path}.")
        raise typer.Exit(code=1)
    if error == "reference_run_wrong_suite":
        _console().print(f"Reference run '{right}' belongs to suite '{comparison['reference_run_suite_name']}', not '{left}'.")
        raise typer.Exit(code=1)
    if error == "strict_requires_reference_run":
        _console().print("--strict requires a suite comparison with a reference run ID.")
        raise typer.Exit(code=2)
    if error == "strict_keys_not_found":
        missing_keys = ", ".join(comparison["missing_strict_keys"])
        _console().print(f"Strict key(s) {missing_keys} were not found on reference run {comparison['reference_run_display_id']}.")
        raise typer.Exit(code=1)
    if error == "baseline_not_found":
        _console().print(f"Suite '{left}' does not have a pinned baseline in {database_path}.")
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
        _console().print(
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
        _console().print(f"{option_name} requires a numeric percent value.")
        raise typer.Exit(code=2)
    try:
        parsed = float(normalized)
    except ValueError as exc:
        _console().print(f"{option_name} must be a number like 5 or 5%.")
        raise typer.Exit(code=2) from exc
    if parsed < 0.0:
        _console().print(f"{option_name} must be zero or greater.")
        raise typer.Exit(code=2)
    return parsed


def _evaluate_compare_gate(
    comparison: dict[str, object],
    *,
    mode: str,
    threshold_percent: float | None,
) -> dict[str, object] | None:
    if threshold_percent is None:
        return None

    if mode == "direct":
        candidate = comparison["candidate"]
        run_analyses = [(candidate, comparison.get("comparison_analysis") or {})]
    else:
        basis_run = comparison.get("basis_run")
        basis_id = None if basis_run is None else basis_run.get("id")
        run_analyses = [
            (run, run.get("comparison_analysis") or {})
            for run in comparison["runs"]
            if run["id"] != basis_id
        ]

    failing_runs: list[dict[str, object]] = []
    for run, comparison_analysis in run_analyses:
        if not comparison_analysis.get("regression_detected"):
            continue
        failing_runs.append(
            {
                "display_id": run["display_id"],
                "record_id": run.get("record_id", run["id"]),
                "classification": comparison_analysis.get("classification"),
                "percent_change": comparison_analysis.get("percent_change"),
                "regression_probability": comparison_analysis.get("regression_probability"),
                "significance_p_value": comparison_analysis.get("significance_p_value"),
                "warnings": list(comparison_analysis.get("warnings") or ()),
            }
        )

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
    _console().print(
        summary_panel(
            "CI Gate",
            [
                ("Threshold", f"{float(gate['threshold_percent']):.2f}%"),
                ("Result", result),
                ("Regressing Runs", failing_runs or "-"),
            ],
        )
    )


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
                ("Reference Run", _styled(f"{basis_run['display_id']} ({basis_run['id']})", "green")),
                ("Status", "Reference is already the fastest run in this comparison scope."),
                ("Scope", scope),
            ],
        )

    best_analysis = best_run.get("comparison_analysis") or {}
    return summary_panel(
        "Best Run vs Reference",
        [
            ("Reference Run", _styled(f"{basis_run['display_id']} ({basis_run['id']})", "yellow")),
            ("Best Run", _styled(f"{best_run['display_id']} ({best_run['id']})", "green")),
            ("Scope", scope),
            ("Delta CI (s)", format_interval(best_analysis.get("delta_ci_lower_seconds"), best_analysis.get("delta_ci_upper_seconds"))),
            ("Improvement Probability", format_probability(best_analysis.get("improvement_probability"))),
            ("Regression Probability", format_probability(best_analysis.get("regression_probability"))),
            ("p-value", f"{float(best_analysis.get('significance_p_value', 0.0)):.4f}"),
            ("Classification", str(best_analysis.get("classification", "-"))),
            ("Warnings", format_warning_list(best_analysis.get("warnings"))),
        ],
    )


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


def _suite_row_style(comparison: dict[str, object], run: dict[str, object]) -> str | None:
    return _row_style(
        comparison["runs"],
        run,
        basis_run=comparison.get("basis_run"),
        highlight_basis=comparison.get("basis_metric_label") == "Reference Median (s)",
    )


def _suite_comparison_table(comparison: dict[str, object], *, verbose: bool) -> Table:
    strict_keys = comparison.get("strict_keys") or []
    title = f"Comparison: {comparison['suite_name']}"
    if strict_keys:
        title = f"{title} (strict: {', '.join(strict_keys)})"

    table = Table(title=title, pad_edge=False, collapse_padding=True)
    table.add_column("Run ID", justify="right", no_wrap=True, min_width=4, max_width=4)
    table.add_column("Record ID", justify="right", no_wrap=True, min_width=7, max_width=7)
    table.add_column("Configuration", overflow="ellipsis", max_width=16)
    table.add_column("Mean +- Std (s)", justify="right", no_wrap=True, max_width=25)
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
        table.add_row(*(value if isinstance(value, Text) else str(value) for value in styled_row))
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


def _print_run_comparison(
    comparison: dict[str, object],
) -> None:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    baseline_style = "green" if baseline["median_seconds"] <= candidate["median_seconds"] else None
    candidate_style = "green" if candidate["median_seconds"] <= baseline["median_seconds"] else None
    _console().print(
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
                (
                    "Median Percent Change",
                    "",
                    Text(
                        "n/a"
                        if comparison["percent_change"] is None
                        else f"{comparison['percent_change']:+.2f}%",
                        style=(
                            None
                            if comparison["percent_change"] is None
                            else "green"
                            if comparison["percent_change"] <= -5.0
                            else "red"
                            if comparison["percent_change"] >= 5.0
                            else None
                        ),
                    ),
                ),
                (
                    "Return Value",
                    _styled(format_return_value(baseline.get("target_return_value"), compact=True), baseline_style),
                    _styled(format_return_value(candidate.get("target_return_value"), compact=True), candidate_style),
                ),
                ("Return Error", "", format_return_error(comparison.get("target_return_relative_error"))),
            ],
        )
    )
    _console().print(_comparison_analysis_panel(comparison["comparison_analysis"]))

    if comparison["observation_rows"]:
        _console().print(
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


def _print_suite_comparison(comparison: dict[str, object]) -> None:
    _console().print(_suite_comparison_table(comparison, verbose=_STATE.verbose))
    _console().print(_suite_findings_panel(comparison))
    best_vs_reference = _best_vs_reference_panel(comparison)
    if best_vs_reference is not None:
        _console().print(best_vs_reference)
    stored_return_metrics = _stored_return_metrics_panel(comparison)
    if stored_return_metrics is not None:
        _console().print(stored_return_metrics)

    comparison_basis = _comparison_basis_panel(comparison)
    if comparison_basis is not None:
        _console().print(comparison_basis)


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
    bootstrap_resamples: CompareBootstrapResamplesOption = 1000,
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
    effective_regression_threshold = regression_threshold
    gate_threshold: float | None = None
    if fail_if_regression is not None:
        gate_threshold = _parse_percent_option(fail_if_regression, option_name="--fail-if-regression")
        effective_regression_threshold = gate_threshold

    analysis_options = _analysis_options(
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        regression_threshold=effective_regression_threshold,
        noise_threshold=noise_threshold,
        significance_level=significance_level,
    )
    left_run_id = _as_run_id(left)
    right_run_id = _as_run_id(right)
    direct_run_compare = left_run_id is not None and right_run_id is not None

    if direct_run_compare:
        comparison = _run_direct_compare(
            left_run_id,
            right_run_id,
            database_path,
            analysis_options,
            strict_keys=strict_keys,
            use_baseline=use_baseline,
            pin_baseline=pin_baseline,
        )
        comparison_mode = "direct"
        render = _print_run_comparison
    else:
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
        comparison_mode = "suite"
        render = _print_suite_comparison

    comparison["comparison_mode"] = comparison_mode
    _finalize_compare_result(
        comparison,
        mode=comparison_mode,
        threshold_percent=gate_threshold,
        json_output=json_output,
        render=render,
    )
