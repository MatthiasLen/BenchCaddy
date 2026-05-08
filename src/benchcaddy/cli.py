from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .db import get_database_path, get_suite_details, list_suite_summaries

app = typer.Typer(help="Inspect BenchCaddy benchmark suites.")
console = Console()


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


@app.command("show")
def show_command(
    suite_name: str = typer.Argument(..., help="Suite name to inspect."),
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
    details = get_suite_details(suite_name, database_path)
    if details is None:
        console.print(f"Suite '{suite_name}' was not found in {database_path}.")
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


def main() -> None:
    app()
