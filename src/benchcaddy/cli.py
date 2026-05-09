from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .db import compare_runs, compare_suite_runs, get_database_path, get_run_details, get_suite_details, list_suite_summaries
from .observability import summarize_observations
from .presentation import dump_json, json_panel, render_table

app = typer.Typer(help="Inspect BenchCaddy benchmark suites.")
console = Console()


@dataclass
class CLIState:
    verbose: bool = False


_STATE = CLIState()


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


def _render_observation_table(observations: list[dict[str, object]], title: str) -> Table:
    summary = summarize_observations(observations)

    return render_table(
        title,
        ["Label", ("Calls", "right"), ("Total (s)", "right"), ("Average (s)", "right")],
        [
            (
                label,
                calls,
                f"{total:.6f}",
                f"{total / calls:.6f}",
            )
            for label, (calls, total) in sorted(summary.items())
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
                ("Time (s)", _format_time(run.get("mean_seconds"), run.get("std_seconds"))),
                ("Min (s)", f"{run.get('min_seconds') or 0:.6f}"),
                ("Max (s)", f"{run.get('max_seconds') or 0:.6f}"),
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
        [("Run ID", "right"), ("Record ID", "right"), "Configuration", ("Time (s)", "right"), ("Samples", "right"), "Recorded At"],
        [
            (
                run["display_id"],
                run["id"],
                dump_json(run["configuration"]),
                _format_time(run.get("mean_seconds"), run.get("std_seconds")),
                len(run["samples"]),
                run["created_at"],
            )
            for run in details["runs"]
        ],
    ))
    console.print(
        render_table(
            f"Observed Timings: {details['suite_name']}",
            [("Run ID", "right"), "Label", ("Calls", "right"), ("Time (s)", "right")],
            [
                (
                    run["display_id"],
                    label,
                    calls,
                    _format_time(total / calls, 0.0),
                )
                for run in details["runs"]
                for label, (calls, total) in sorted(summarize_observations(run["observations"]).items())
            ],
        )
    )
    if details["environment"] is not None:
        console.print(json_panel("Environment", details["environment"], indent=2))
    if _STATE.verbose:
        for run in details["runs"]:
            console.print(_render_observation_table(run["observations"], title=f"Observed Timings for Run {run['display_id']}"))


def _filtered_keys(
    baseline: dict[str, object],
    candidate: dict[str, object],
    filter_keys: list[str] | None,
) -> list[str]:
    allowed = None if filter_keys is None else set(filter_keys)
    return [key for key in sorted(set(baseline["configuration"]) | set(candidate["configuration"])) if allowed is None or key in allowed]


def _print_run_comparison(
    comparison: dict[str, object],
    filter_keys: list[str] | None,
) -> None:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    console.print(
        render_table(
            f"Run Comparison: {baseline['display_id']} -> {candidate['display_id']}",
            ["Field", "Baseline", "Candidate"],
            [
                ("Run ID", baseline["display_id"], candidate["display_id"]),
                ("Record ID", baseline["id"], candidate["id"]),
                *[
                    (
                        key,
                        dump_json(baseline["configuration"].get(key)),
                        dump_json(candidate["configuration"].get(key)),
                    )
                    for key in _filtered_keys(baseline, candidate, filter_keys)
                ],
                ("Time (s)", _format_time(baseline.get("mean_seconds"), baseline.get("std_seconds")), _format_time(candidate.get("mean_seconds"), candidate.get("std_seconds"))),
                ("Min (s)", f"{baseline.get('min_seconds') or 0:.6f}", f"{candidate.get('min_seconds') or 0:.6f}"),
                ("Max (s)", f"{baseline.get('max_seconds') or 0:.6f}", f"{candidate.get('max_seconds') or 0:.6f}"),
                ("Delta (s)", "", f"{comparison['delta_seconds']:.6f}"),
                ("Percent Change", "", _style_delta(comparison["percent_change"])),
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
                        f"{row['baseline_seconds']:.6f}",
                        f"{row['candidate_seconds']:.6f}",
                        f"{row['delta_seconds']:.6f}",
                    )
                    for row in comparison["observation_rows"]
                ],
            )
        )


def _print_suite_comparison(comparison: dict[str, object]) -> None:
    console.print(
        render_table(
            f"Comparison: {comparison['suite_name']}",
            [("Run ID", "right"), ("Record ID", "right"), "Configuration", ("Time (s)", "right"), ("Delta vs Best (s)", "right"), ("Slowdown", "right"), *([("Samples", "right"), "Recorded At"] if _STATE.verbose else [])],
            [
                (
                    run["display_id"],
                    run["id"],
                    dump_json(run["configuration"]),
                    _format_time(run.get("mean_seconds"), run.get("std_seconds")),
                    f"{run['delta_seconds']:.6f}",
                    "n/a" if run["slowdown_factor"] is None else f"{run['slowdown_factor']:.2f}x",
                    *([run["sample_count"], run["created_at"]] if _STATE.verbose else []),
                )
                for run in comparison["runs"]
            ],
        )
    )

    if comparison["best_median_seconds"] is not None:
        console.print(Panel.fit(f"Best median: {comparison['best_median_seconds']:.6f}s", title="Comparison Basis"))


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
    identifier: str = typer.Argument(..., help="Suite name or run ID to inspect (for example 3.2)."),
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
    run_id = _as_run_id(identifier)
    if run_id is not None:
        run = get_run_details(run_id, database_path)
        if run is None:
            console.print(f"Run '{run_id}' was not found in {database_path}.")
            raise typer.Exit(code=1)
        _show_run(run)
        return

    details = get_suite_details(identifier, database_path)
    if details is None:
        console.print(f"Suite '{identifier}' was not found in {database_path}.")
        raise typer.Exit(code=1)
    _show_suite(details)


@app.command("compare")
def compare_command(
    left: str = typer.Argument(..., help="Suite name or baseline run ID."),
    right: str | None = typer.Argument(None, help="Candidate run ID for direct comparison."),
    database: Path = typer.Option(
        None,
        "--database",
        "-d",
        exists=False,
        dir_okay=False,
        help="Path to the BenchCaddy SQLite database.",
    ),
    filter_keys: list[str] = typer.Option(
        None,
        "--filter",
        help="Configuration keys to emphasize in direct run comparisons.",
    ),
) -> None:
    database_path = get_database_path(database)
    left_run_id = _as_run_id(left)
    right_run_id = _as_run_id(right) if right is not None else None
    if left_run_id is not None and right_run_id is not None:
        comparison = compare_runs(left_run_id, right_run_id, database_path)
        if comparison is None:
            console.print(f"Run comparison {left_run_id} vs {right_run_id} was not found in {database_path}.")
            raise typer.Exit(code=1)
        _print_run_comparison(comparison, filter_keys)
        return

    comparison = compare_suite_runs(left, database_path)
    if comparison is None:
        console.print(f"Suite '{left}' was not found in {database_path}.")
        raise typer.Exit(code=1)
    _print_suite_comparison(comparison)


main = app
