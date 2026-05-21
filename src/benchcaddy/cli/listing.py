from __future__ import annotations

from typing import Annotated

import typer

from ..db import get_database_path, list_suite_summaries
from ..presentation import render_table
from ._shared import _STATE, DatabaseOption, _console, _emit_json_response, app


@app.command("list")
def list_command(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Emit machine-readable suite inventory as JSON.",
        ),
    ] = False,
    database: DatabaseOption = None,
) -> None:
    database_path = get_database_path(database)
    summaries = list_suite_summaries(database_path)
    if not summaries:
        if json_output:
            _emit_json_response(
                command="list",
                status="inconclusive",
                reason="no_suites_found",
                suggested_action="Record a benchmark sweep to create suite history.",
                confidence=None,
                result={
                    "database_path": database_path,
                    "suite_count": 0,
                    "suites": [],
                },
            )
            return
        _console().print(f"No suites found in {database_path}.")
        raise typer.Exit()

    if json_output:
        _emit_json_response(
            command="list",
            status="pass",
            reason="suite_inventory_available",
            suggested_action="Use benchcaddy show -j SUITE or benchcaddy compare -j SUITE for the next step.",
            confidence=None,
            result={
                "database_path": database_path,
                "suite_count": len(summaries),
                "suites": summaries,
            },
        )
        return

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
