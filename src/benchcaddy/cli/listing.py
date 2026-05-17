from __future__ import annotations

import typer

from ..db import get_database_path, list_suite_summaries
from ..presentation import render_table
from ._shared import _STATE, DatabaseOption, _console, app


@app.command("list")
def list_command(
    database: DatabaseOption = None,
) -> None:
    database_path = get_database_path(database)
    summaries = list_suite_summaries(database_path)
    if not summaries:
        _console().print(f"No suites found in {database_path}.")
        raise typer.Exit()

    _console().print(
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
        from rich.panel import Panel

        _console().print(Panel.fit(str(database_path), title="Database"))