from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .db import compare_runs, compare_suite_runs, get_database_path, get_run_details, get_suite_details, list_suite_summaries

app = typer.Typer(help="Inspect BenchCaddy benchmark suites.")
console = Console()


@dataclass
class CLIState:
    verbose: bool = False


_STATE = CLIState()


def _as_run_id(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _style_delta(percent_change: float | None) -> Text:
    if percent_change is None:
        return Text("n/a")

    style = None
    if percent_change <= -5.0:
        style = "green"
    elif percent_change >= 5.0:
        style = "red"

    return Text(f"{percent_change:+.2f}%", style=style)


def _render_observation_table(observations: list[dict[str, object]], title: str) -> Table:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for sample in observations:
        for record in sample["records"]:
            label = str(record["label"])
            duration = float(record["duration_seconds"])
            totals[label] = totals.get(label, 0.0) + duration
            counts[label] = counts.get(label, 0) + 1

    table = Table(title=title)
    table.add_column("Label")
    table.add_column("Calls", justify="right")
    table.add_column("Total (s)", justify="right")
    table.add_column("Average (s)", justify="right")

    for label in sorted(totals):
        table.add_row(
            label,
            str(counts[label]),
            f"{totals[label]:.6f}",
            f"{totals[label] / counts[label]:.6f}",
        )

    return table


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

    table = Table(title=f"BenchCaddy suites ({database_path})")
    table.add_column("Suite")
    table.add_column("Target")
    table.add_column("Runs", justify="right")
    table.add_column("Last Run")

    for summary in summaries:
        table.add_row(
            summary["suite_name"],
            summary["target_name"],
            str(summary["run_count"]),
            str(summary["last_run_at"]),
        )

    console.print(table)
    if _STATE.verbose:
        console.print(Panel.fit(str(database_path), title="Database"))


@app.command("show")
def show_command(
    identifier: str = typer.Argument(..., help="Suite name or run ID to inspect."),
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

        summary = Table(title=f"Run: {run['id']}")
        summary.add_column("Field")
        summary.add_column("Value")
        summary.add_row("Suite", str(run["suite_name"]))
        summary.add_row("Target", str(run["target_name"]))
        summary.add_row("Configuration", json.dumps(run["configuration"], sort_keys=True))
        summary.add_row("Median (s)", f"{run['median_seconds']:.6f}")
        summary.add_row("Samples", str(len(run["samples"])))
        summary.add_row("Recorded At", str(run["created_at"]))
        console.print(summary)
        console.print(_render_observation_table(run["observations"], title="Observed Timings"))
        console.print(
            Panel(
                json.dumps(run["environment"], indent=2, sort_keys=True),
                title="Environment",
            )
        )
        return

    details = get_suite_details(identifier, database_path)
    if details is None:
        console.print(f"Suite '{identifier}' was not found in {database_path}.")
        raise typer.Exit(code=1)

    runs_table = Table(title=f"Suite: {details['suite_name']}")
    runs_table.add_column("Run ID", justify="right")
    runs_table.add_column("Configuration")
    runs_table.add_column("Median (s)", justify="right")
    runs_table.add_column("Samples", justify="right")
    runs_table.add_column("Recorded At")

    for run in details["runs"]:
        runs_table.add_row(
            str(run["id"]),
            json.dumps(run["configuration"], sort_keys=True),
            f"{run['median_seconds']:.6f}",
            str(len(run["samples"])),
            str(run["created_at"]),
        )

    console.print(runs_table)
    if details["environment"] is not None:
        console.print(
            Panel(
                json.dumps(details["environment"], indent=2, sort_keys=True),
                title="Environment",
            )
        )
    if _STATE.verbose:
        for run in details["runs"]:
            console.print(
                _render_observation_table(
                    run["observations"],
                    title=f"Observed Timings for Run {run['id']}",
                )
            )


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

        baseline = comparison["baseline"]
        candidate = comparison["candidate"]
        key_set = sorted(set(baseline["configuration"]) | set(candidate["configuration"]))
        if filter_keys:
            key_set = [key for key in key_set if key in set(filter_keys)]

        config_table = Table(title=f"Run Comparison: {left_run_id} -> {right_run_id}")
        config_table.add_column("Field")
        config_table.add_column("Baseline")
        config_table.add_column("Candidate")
        for key in key_set:
            config_table.add_row(
                key,
                json.dumps(baseline["configuration"].get(key), sort_keys=True),
                json.dumps(candidate["configuration"].get(key), sort_keys=True),
            )
        config_table.add_row("Median (s)", f"{baseline['median_seconds']:.6f}", f"{candidate['median_seconds']:.6f}")
        config_table.add_row("Delta (s)", "", f"{comparison['delta_seconds']:.6f}")
        config_table.add_row("Percent Change", "", _style_delta(comparison["percent_change"]))
        console.print(config_table)

        if comparison["observation_rows"]:
            observation_table = Table(title="Observed Timing Diff")
            observation_table.add_column("Label")
            observation_table.add_column("Baseline (s)", justify="right")
            observation_table.add_column("Candidate (s)", justify="right")
            observation_table.add_column("Delta (s)", justify="right")
            for row in comparison["observation_rows"]:
                observation_table.add_row(
                    row["label"],
                    f"{row['baseline_seconds']:.6f}",
                    f"{row['candidate_seconds']:.6f}",
                    f"{row['delta_seconds']:.6f}",
                )
            console.print(observation_table)
        return

    comparison = compare_suite_runs(left, database_path)
    if comparison is None:
        console.print(f"Suite '{left}' was not found in {database_path}.")
        raise typer.Exit(code=1)

    comparison_table = Table(title=f"Comparison: {comparison['suite_name']}")
    comparison_table.add_column("Run ID", justify="right")
    comparison_table.add_column("Configuration")
    comparison_table.add_column("Median (s)", justify="right")
    comparison_table.add_column("Delta vs Best (s)", justify="right")
    comparison_table.add_column("Slowdown", justify="right")

    if _STATE.verbose:
        comparison_table.add_column("Samples", justify="right")
        comparison_table.add_column("Recorded At")

    for run in comparison["runs"]:
        slowdown = "n/a"
        if run["slowdown_factor"] is not None:
            slowdown = f"{run['slowdown_factor']:.2f}x"

        row = [
            str(run["id"]),
            json.dumps(run["configuration"], sort_keys=True),
            f"{run['median_seconds']:.6f}",
            f"{run['delta_seconds']:.6f}",
            slowdown,
        ]
        if _STATE.verbose:
            row.extend([str(run["sample_count"]), str(run["created_at"])])

        comparison_table.add_row(*row)

    console.print(comparison_table)

    if comparison["best_median_seconds"] is not None:
        console.print(
            Panel.fit(
                f"Best median: {comparison['best_median_seconds']:.6f}s",
                title="Comparison Basis",
            )
        )


def main() -> None:
    app()
