from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .db import compare_runs, compare_suite_runs, get_database_path, get_run_details, get_selected_run_details, get_suite_details, list_suite_summaries
from .observability import summarize_observations
from .presentation import dump_json, json_panel, render_table

app = typer.Typer(help="Inspect BenchCaddy benchmark suites.")
console = Console()


@dataclass
class CLIState:
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


def _comparison_title(comparison: dict[str, object]) -> str:
    strict_keys = comparison.get("strict_keys") or []
    if not strict_keys:
        return f"Comparison: {comparison['suite_name']}"
    return f"Comparison: {comparison['suite_name']} (strict: {', '.join(strict_keys)})"


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
    if percent_change is None: return Text("n/a")
    return Text(f"{percent_change:+.2f}%", style="green" if percent_change <= -5.0 else "red" if percent_change >= 5.0 else None)


def _format_time(mean_seconds: float | None, std_seconds: float | None) -> str:
    mean_value = 0.0 if mean_seconds is None else mean_seconds
    std_value = 0.0 if std_seconds is None else std_seconds
    return f"{mean_value:.6f} +- {std_value:.6f}"


def _styled(value: object, style: str | None = None) -> Text:
    return Text(str(value), style=style)


def _style_row(values: tuple[object, ...], style: str | None = None) -> tuple[object, ...]:
    return tuple(_styled(value, style) if style else value for value in values)


def _suite_row_style(comparison: dict[str, object], run: dict[str, object]) -> str | None:
    basis_run = comparison.get("basis_run")
    if basis_run is None:
        return None

    best_run = min(comparison["runs"], key=lambda candidate: (candidate["median_seconds"], candidate["id"]))
    if run["id"] == best_run["id"]:
        return "green"

    if comparison.get("basis_metric_label") == "Reference Median (s)" and run["id"] == basis_run["id"]:
        return "yellow"

    return None


def _format_optional_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def _format_return_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float)):
        return str(value)
    return dump_json(value)


def _format_return_distance(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "equal" if value else "different"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _render_observation_table(observations: list[dict[str, object]], title: str) -> Table:
    summary = summarize_observations(observations)

    return render_table(
        title,
        ["Label", ("Calls", "right"), ("Mean +- Std (s)", "right"), ("Total (s)", "right")],
        [
            (
                label,
                stats.calls,
                _format_time(stats.mean_seconds, stats.std_seconds),
                f"{stats.total_seconds:.6f}",
            )
            for label, stats in summary.items()
        ],
    )


def _show_run(run: dict[str, object]) -> None:
    console.print(
        render_table(
            f"Run: {run['display_id']}",
            ["Field", "Value"],
            [
                ("Run ID", run["display_id"]),
                ("Record ID", run["id"]),
                ("Sweep ID", run["sweep_id"]),
                ("Run Index", run["run_index"]),
                ("Suite", run["suite_name"]),
                ("Target", run["target_name"]),
                ("Configuration", dump_json(run["configuration"])),
                ("Mean +- Std (s)", _format_time(run.get("mean_seconds"), run.get("std_seconds"))),
                ("Min (s)", _format_optional_seconds(run.get("min_seconds"))),
                ("Max (s)", _format_optional_seconds(run.get("max_seconds"))),
                ("Return Value", _format_return_value(run.get("target_return_value"))),
                ("Samples", len(run["samples"])),
                ("Recorded At", run["created_at"]),
            ],
        )
    )
    console.print(_render_observation_table(run["observations"], title="Observed Timings"))
    console.print(json_panel("Environment", run["environment"], indent=2))


def _show_suite(details: dict[str, object]) -> None:
    console.print(render_table(
        f"Suite: {details['suite_name']}",
        [("Run ID", "right"), ("Record ID", "right"), "Configuration", ("Mean +- Std (s)", "right"), "Return Value", ("Samples", "right"), "Recorded At"],
        [
            (
                run["display_id"],
                run["id"],
                dump_json(run["configuration"]),
                _format_time(run.get("mean_seconds"), run.get("std_seconds")),
                _format_return_value(run.get("target_return_value")),
                len(run["samples"]),
                run["created_at"],
            )
            for run in details["runs"]
        ],
    ))
    console.print(
        render_table(
            f"Observed Timings: {details['suite_name']}",
            [("Run ID", "right"), "Label", ("Calls", "right"), ("Mean +- Std (s)", "right")],
            [
                (
                    run["display_id"],
                    label,
                    stats.calls,
                    _format_time(stats.mean_seconds, stats.std_seconds),
                )
                for run in details["runs"]
                for label, stats in summarize_observations(run["observations"]).items()
            ],
        )
    )
    if details["environment"] is not None:
        console.print(json_panel("Environment", details["environment"], indent=2))
    if _STATE.verbose:
        for run in details["runs"]:
            console.print(_render_observation_table(run["observations"], title=f"Observed Timings for Run {run['display_id']}"))


def _show_selected_runs(runs: list[dict[str, object]]) -> None:
    console.print(render_table(
        "Selected Runs",
        [
            ("Run ID", "right"),
            ("Record ID", "right"),
            "Suite",
            "Target",
            "Configuration",
            ("Mean +- Std (s)", "right"),
            "Return Value",
            ("Samples", "right"),
            "Recorded At",
        ],
        [
            (
                run["display_id"],
                run["id"],
                run["suite_name"],
                run["target_name"],
                dump_json(run["configuration"]),
                _format_time(run.get("mean_seconds"), run.get("std_seconds")),
                _format_return_value(run.get("target_return_value")),
                len(run["samples"]),
                run["created_at"],
            )
            for run in runs
        ],
    ))
    console.print(
        render_table(
            "Observed Timings: Selected Runs",
            [("Run ID", "right"), ("Record ID", "right"), "Label", ("Calls", "right"), ("Mean +- Std (s)", "right")],
            [
                (
                    run["display_id"],
                    run["id"],
                    label,
                    stats.calls,
                    _format_time(stats.mean_seconds, stats.std_seconds),
                )
                for run in runs
                for label, stats in summarize_observations(run["observations"]).items()
            ],
        )
    )


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
                ("Mean +- Std (s)", _styled(_format_time(baseline.get("mean_seconds"), baseline.get("std_seconds")), baseline_style), _styled(_format_time(candidate.get("mean_seconds"), candidate.get("std_seconds")), candidate_style)),
                ("Min (s)", _styled(_format_optional_seconds(baseline.get("min_seconds")), baseline_style), _styled(_format_optional_seconds(candidate.get("min_seconds")), candidate_style)),
                ("Max (s)", _styled(_format_optional_seconds(baseline.get("max_seconds")), baseline_style), _styled(_format_optional_seconds(candidate.get("max_seconds")), candidate_style)),
                ("Return Value", _styled(_format_return_value(baseline.get("target_return_value")), baseline_style), _styled(_format_return_value(candidate.get("target_return_value")), candidate_style)),
                ("Return Distance", "", _format_return_distance(comparison.get("target_return_distance"))),
                ("Median Delta (s)", "", f"{comparison['delta_seconds']:.6f}"),
                ("Median Percent Change", "", _style_delta(comparison["percent_change"])),
            ],
        )
    )

    if comparison["observation_rows"]:
        console.print(
            render_table(
                "Observed Timing Diff",
                ["Label", ("Baseline (s)", "right"), ("Candidate (s)", "right"), ("Delta (s)", "right")],
                [
                    (
                        row["label"],
                        "-" if row["baseline_mean_seconds"] is None else _format_time(row["baseline_mean_seconds"], row["baseline_std_seconds"]),
                        "-" if row["candidate_mean_seconds"] is None else _format_time(row["candidate_mean_seconds"], row["candidate_std_seconds"]),
                        _format_optional_seconds(row["delta_seconds"]),
                    )
                    for row in comparison["observation_rows"]
                ],
            )
        )


def _print_suite_comparison(comparison: dict[str, object]) -> None:
    console.print(
        render_table(
            _comparison_title(comparison),
            [("Run ID", "right"), ("Record ID", "right"), "Configuration", ("Mean +- Std (s)", "right"), (comparison["delta_column_label"], "right"), (comparison["ratio_column_label"], "right"), *([("Samples", "right"), "Recorded At"] if _STATE.verbose else [])],
            [
                _style_row(
                    (
                        run["display_id"],
                        run["id"],
                        dump_json(run["configuration"]),
                        _format_time(run.get("mean_seconds"), run.get("std_seconds")),
                        f"{run['delta_seconds']:.6f}",
                        "n/a" if run["slowdown_factor"] is None else f"{run['slowdown_factor']:.2f}x",
                        *([run["sample_count"], run["created_at"]] if _STATE.verbose else []),
                    ),
                    _suite_row_style(comparison, run),
                )
                for run in comparison["runs"]
            ],
        )
    )

    if comparison["basis_median_seconds"] is not None:
        best_run = comparison["basis_run"]
        console.print(
            Panel.fit(
                " | ".join(
                    [
                        f"Run ID: {best_run['display_id']}",
                        f"Record ID: {best_run['id']}",
                        f"{comparison['basis_metric_label']}: {best_run['median_seconds']:.6f}",
                        f"Mean +- Std (s): {_format_time(best_run.get('mean_seconds'), best_run.get('std_seconds'))}",
                    ]
                ),
                title="Comparison Basis",
            )
        )


@app.callback()
def callback(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show additional detail in command output.",
    ),
) -> None:
    _STATE.verbose = verbose


@app.command("list")
def list_command(
    database: Path = typer.Option(
        None,
        "--database",
        "-d",
        exists=False,
        dir_okay=False,
        help="Path to the BenchCaddy SQLite database.",
    ),
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


@app.command("show")
def show_command(
    identifiers: list[str] = typer.Argument(..., help="Suite name or one or more run IDs to inspect (for example 3.2 5 7.1)."),
    database: Path = typer.Option(
        None,
        "--database",
        "-d",
        exists=False,
        dir_okay=False,
        help="Path to the BenchCaddy SQLite database.",
    ),
) -> None:
    database_path = get_database_path(database)

    if len(identifiers) == 1:
        identifier = identifiers[0]
        run_id = _as_run_id(identifier)
        if run_id is not None:
            run = get_run_details(run_id, database_path)
            if run is None:
                console.print(f"Run '{identifier}' was not found in {database_path}.")
                raise typer.Exit(code=1)
            _show_run(run)
            return

        details = get_suite_details(identifier, database_path)
        if details is None:
            console.print(f"Suite '{identifier}' was not found in {database_path}.")
            raise typer.Exit(code=1)
        _show_suite(details)
        return

    run_ids: list[int | tuple[int, int]] = []
    for identifier in identifiers:
        run_id = _as_run_id(identifier)
        if run_id is None:
            console.print(f"'{identifier}' is not a valid run ID.")
            raise typer.Exit(code=1)
        run_ids.append(run_id)

    runs = get_selected_run_details(run_ids, database_path)
    if runs is None:
        console.print(f"One or more runs were not found in {database_path}.")
        raise typer.Exit(code=1)
    _show_selected_runs(runs)


@app.command("compare")
def compare_command(
    left: str = typer.Argument(..., help="Suite name or baseline run ID."),
    operands: list[str] = typer.Argument(None, help="Optional reference run ID followed by strict config keys."),
    strict: bool = typer.Option(
        False,
        "--strict",
        "-s",
        help="Restrict suite comparison to runs whose configuration matches the reference run for the given trailing config keys.",
    ),
    database: Path = typer.Option(
        None,
        "--database",
        "-d",
        exists=False,
        dir_okay=False,
        help="Path to the BenchCaddy SQLite database.",
    ),
) -> None:
    right, strict_keys = _parse_compare_operands(operands, strict)
    database_path = get_database_path(database)
    left_run_id = _as_run_id(left)
    right_run_id = _as_run_id(right) if right is not None else None
    if left_run_id is not None and right_run_id is not None:
        if strict_keys:
            console.print("--strict is only supported for suite comparisons with a reference run.")
            raise typer.Exit(code=2)
        comparison = compare_runs(left_run_id, right_run_id, database_path)
        if comparison is None:
            console.print(f"Run comparison {left_run_id} vs {right_run_id} was not found in {database_path}.")
            raise typer.Exit(code=1)
        _print_run_comparison(comparison)
        return

    if strict_keys and right_run_id is None:
        console.print("--strict requires a suite comparison with a reference run ID.")
        raise typer.Exit(code=2)

    comparison = compare_suite_runs(left, right_run_id, strict_keys, database_path)
    if comparison is None:
        console.print(f"Suite '{left}' was not found in {database_path}.")
        raise typer.Exit(code=1)
    if comparison.get("error") == "reference_run_not_found":
        console.print(f"Reference run '{right}' was not found in {database_path}.")
        raise typer.Exit(code=1)
    if comparison.get("error") == "reference_run_wrong_suite":
        console.print(
            f"Reference run '{right}' belongs to suite '{comparison['reference_run_suite_name']}', not '{left}'."
        )
        raise typer.Exit(code=1)
    if comparison.get("error") == "strict_requires_reference_run":
        console.print("--strict requires a suite comparison with a reference run ID.")
        raise typer.Exit(code=2)
    if comparison.get("error") == "strict_keys_not_found":
        missing_keys = ", ".join(comparison["missing_strict_keys"])
        console.print(
            f"Strict key(s) {missing_keys} were not found on reference run {comparison['reference_run_display_id']}."
        )
        raise typer.Exit(code=1)
    _print_suite_comparison(comparison)


main = app
