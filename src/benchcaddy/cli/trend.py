from __future__ import annotations

from typing import Annotated

import typer
from rich.measure import Measurement
from rich.panel import Panel
from rich.table import Table

from ..db import get_database_path, get_suite_trend
from ..presentation import dump_json, format_interval, format_warning_list, render_table, summary_panel
from ._rendering import _best_run, _row_style, _style_row, _styled
from ._shared import (
    _STATE,
    CompareBootstrapResamplesOption,
    CompareConfidenceLevelOption,
    DatabaseOption,
    NoiseThresholdOption,
    RegressionThresholdOption,
    SignificanceLevelOption,
    _analysis_options,
    _console,
    _emit_json,
    _require_run_id,
    app,
)


def _combine_warning_lists(*values: object) -> tuple[str, ...]:
    combined: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            combined.extend(str(item) for item in value)
    return tuple(dict.fromkeys(combined))


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
        width=102,
    )


def _trend_delta_value(run: dict[str, object]) -> str:
    vs_baseline = run["vs_baseline"]
    delta_value = f"{vs_baseline['delta_seconds']:+.6f}"
    if vs_baseline.get("percent_change") is not None:
        return f"{delta_value} ({vs_baseline['percent_change']:+.2f}%)"
    return delta_value


def _trend_sparkline(values: list[float], *, max_points: int | None = None) -> str:
    if not values:
        return "-"

    spark_values = values
    if max_points is not None and max_points > 0 and len(values) > max_points:
        if max_points == 1:
            spark_values = [values[-1]]
        else:
            step = (len(values) - 1) / (max_points - 1)
            spark_values = [values[round(index * step)] for index in range(max_points)]

    blocks = "▁▂▃▄▅▆▇█"
    if len(spark_values) == 1:
        return blocks[0]

    low = min(spark_values)
    high = max(spark_values)
    if abs(high - low) <= 1e-12:
        return blocks[0] * len(spark_values)

    scale = len(blocks) - 1
    return "".join(blocks[min(scale, max(0, round(((value - low) / (high - low)) * scale)))] for value in spark_values)


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


def _print_trend_summary(trend: dict[str, object]) -> None:
    limit = trend.get("limit")
    mode_detail = "Per-configuration summary"
    if limit is not None:
        mode_detail = f"Per-configuration summary (most recent {int(limit)} runs per configuration)"

    _console().print(
        summary_panel(
            f"Trend Summary: {trend['suite_name']}",
            [
                ("Configurations", str(trend["configuration_count"])),
                ("Mode", mode_detail),
                ("Hint", "Pass a baseline run ID to inspect one configuration timeline."),
            ],
        )
    )

    table = Table(title=f"Trend Summary: {trend['suite_name']}", pad_edge=False, collapse_padding=True, min_width=100)
    table.add_column("Config", overflow="fold", max_width=14)
    table.add_column("Runs", justify="right", no_wrap=True, max_width=4)
    table.add_column("Trend", no_wrap=True, min_width=12, max_width=12)
    table.add_column("Vs 1st", no_wrap=True, min_width=11, max_width=11)
    table.add_column("Vs Recent", no_wrap=True, min_width=11, max_width=11)
    table.add_column("Vs Best", justify="right", no_wrap=True, max_width=8)
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
            _trend_sparkline(summary["median_series"], max_points=12),
            _trend_summary_signal(summary["latest_vs_first"]),
            _trend_summary_signal(summary.get("recent_vs_window")),
            _trend_headroom(summary),
            f"{summary['latest_run']['median_seconds']:.6f}",
        )

    _console().print(table)
    _console().print(
        summary_panel(
            "Label Guide",
            [
                ("Trend", "sparkline of median timings from oldest to newest run for that configuration"),
                ("stbl", "no meaningful shift detected relative to the comparison basis"),
                ("noisy", "variance or confidence interval is too wide to call the direction cleanly"),
                ("reg", "meaningful slowdown detected"),
                ("imp", "meaningful speedup detected"),
                ("Vs 1st", "latest run compared with the first recorded run for that configuration"),
                ("Vs Recent", "latest run compared with the recent trailing window for that configuration"),
                ("Vs Best", "latest run compared with the best observed run for that configuration"),
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


def _trend_row_style(trend: dict[str, object], run: dict[str, object]) -> str | None:
    return _row_style(
        trend["runs"],
        run,
        basis_run=trend.get("basis_run"),
        highlight_basis=True,
    )


def _print_trend(trend: dict[str, object]) -> None:
    table = Table(title=f"Trend: {trend['suite_name']}", pad_edge=False, collapse_padding=True)
    table.add_column("Run", justify="right", no_wrap=True, min_width=4, max_width=4)
    table.add_column("Median (s)", justify="right", no_wrap=True, max_width=10)
    table.add_column("Median CI (s)", justify="right", no_wrap=True, max_width=20)
    table.add_column("Delta", justify="right", no_wrap=True, max_width=18)
    table.add_column("Vs Recent", no_wrap=True, min_width=9, max_width=9)
    table.add_column("Vs Basis", no_wrap=True, min_width=8, max_width=8)
    table.add_column("Recorded At", overflow="ellipsis", no_wrap=True, max_width=20)
    if _STATE.verbose:
        table.add_column("Record ID", justify="right", no_wrap=True, min_width=7, max_width=7)
        table.add_column("Warnings", overflow="fold")

    for run in trend["runs"]:
        table.add_row(*_style_row(_trend_row(run, verbose=_STATE.verbose), _trend_row_style(trend, run)))

    current_console = _console()
    current_console.print(_trend_basis_panel(trend))
    median_series = [run["median_seconds"] for run in trend["runs"]]
    first_run = trend["runs"][0]
    latest_run = trend["runs"][-1]
    best_run = _best_run(trend["runs"])
    table_width = Measurement.get(current_console, current_console.options, table).maximum
    summary_label_width = max(len(label) for label in ("Graph", "First", "Best", "Latest"))
    graph_width = max(8, table_width - summary_label_width - 6)
    trend_summary = Table.grid(padding=(0, 2))
    trend_summary.add_row("Graph", _trend_sparkline(median_series, max_points=graph_width))
    trend_summary.add_row("First", f"{first_run['median_seconds']:.6f}")
    trend_summary.add_row("Best", f"{best_run['median_seconds']:.6f}")
    trend_summary.add_row("Latest", f"{latest_run['median_seconds']:.6f}")
    current_console.print(Panel(trend_summary, title="Median Trend", width=table_width))

    current_console.print(table)
    if _STATE.verbose:
        warning_rows = _trend_warning_rows(trend["runs"])
        if warning_rows:
            current_console.print(render_table("Trend Warning Details", [("Run ID", "right"), "Warnings"], warning_rows))


@app.command(
    "trend",
    help=(
        "Inspect one suite configuration over time. With a positional baseline run or --pinned, trend shows the matching configuration timeline. "
        "Without either, mixed suites show a compact per-configuration trend summary."
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
                "If omitted, trend shows a mixed-suite summary when multiple configurations are present unless --pinned is used."
            ),
        ),
    ] = None,
    use_pinned_baseline: Annotated[
        bool,
        typer.Option(
            "--pinned",
            "-p",
            help="Use the pinned suite baseline as the trend anchor instead of showing the mixed-suite summary.",
        ),
    ] = False,
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
        use_pinned_baseline=use_pinned_baseline,
        limit=limit,
    )
    if trend is None:
        _console().print(f"Suite '{suite_name}' was not found in {database_path}.")
        raise typer.Exit(code=1)
    if trend.get("error") == "reference_run_not_found":
        _console().print(f"Reference run '{baseline}' was not found in {database_path}.")
        raise typer.Exit(code=1)
    if trend.get("error") == "reference_run_wrong_suite":
        _console().print(f"Reference run '{baseline}' belongs to suite '{trend['reference_run_suite_name']}', not '{suite_name}'.")
        raise typer.Exit(code=1)
    if trend.get("error") == "baseline_not_found":
        _console().print(f"Suite '{suite_name}' does not have a pinned baseline in {database_path}.")
        raise typer.Exit(code=1)
    if trend.get("mode") != "summary" and trend.get("basis_run") is None:
        _console().print(f"Suite '{suite_name}' does not have any recorded runs in {database_path}.")
        raise typer.Exit(code=1)

    if json_output:
        _emit_json(trend)
        return

    if trend.get("mode") == "summary":
        _print_trend_summary(trend)
        return

    _print_trend(trend)