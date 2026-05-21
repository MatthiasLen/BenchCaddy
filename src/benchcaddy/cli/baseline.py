from __future__ import annotations

from typing import Annotated

import typer
from rich.panel import Panel

from ..db import get_database_path, get_suite_baseline_history, set_suite_baseline
from ..presentation import dump_json, render_table, summary_panel
from ._rendering import _styled
from ._shared import DatabaseOption, _console, _emit_json, _require_run_id, app


@app.command("baseline", help="Inspect baseline history for a suite or pin a run as the latest suite baseline.")
def baseline_command(
    suite_name: Annotated[
        str,
        typer.Argument(help="Suite name whose baseline history should be inspected or updated."),
    ],
    pin: Annotated[
        str | None,
        typer.Option(
            "--pin",
            help="Append a new baseline event for the supplied run ID and make it the latest suite baseline.",
        ),
    ] = None,
    note: Annotated[
        str | None,
        typer.Option(
            "--note",
            help="Optional note stored alongside a baseline pin event.",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Limit the baseline history output to the most recent events.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit machine-readable JSON output for baseline history.",
        ),
    ] = False,
    database: DatabaseOption = None,
) -> None:
    database_path = get_database_path(database)
    if note is not None and pin is None:
        _console().print("--note requires --pin.")
        raise typer.Exit(code=2)

    pin_update = None
    if pin is not None:
        pin_update = set_suite_baseline(suite_name, _require_run_id(pin), database_path, note=note)
        if pin_update is None:
            _console().print(f"Suite '{suite_name}' was not found in {database_path}.")
            raise typer.Exit(code=1)
        if pin_update.get("error") == "reference_run_not_found":
            _console().print(f"Reference run '{pin}' was not found in {database_path}.")
            raise typer.Exit(code=1)
        if pin_update.get("error") == "reference_run_wrong_suite":
            _console().print(f"Reference run '{pin}' belongs to suite '{pin_update['reference_run_suite_name']}', not '{suite_name}'.")
            raise typer.Exit(code=1)

    history = get_suite_baseline_history(suite_name, database_path, limit=limit)
    if history is None:
        _console().print(f"Suite '{suite_name}' was not found in {database_path}.")
        raise typer.Exit(code=1)

    if json_output:
        payload = dict(history)
        if pin_update is not None:
            payload["pin_update"] = pin_update
        _emit_json(payload)
        return

    if pin_update is not None:
        _console().print(
            Panel.fit(
                f"Baseline for {suite_name}: {pin_update['display_id']} ({pin_update['id']})",
                title="Baseline Updated",
            )
        )

    current_baseline = history.get("current_baseline")
    if current_baseline is None:
        _console().print(
            summary_panel(
                f"Baseline: {history['suite_name']}",
                [
                    ("Target", history["target_name"]),
                    ("Database", database_path),
                    ("Status", "No baseline history."),
                ],
            )
        )
        return

    current_run = current_baseline["run"]
    _console().print(
        summary_panel(
            "Current Baseline",
            [
                ("Suite", history["suite_name"]),
                ("Run ID", _styled(current_run["display_id"], "yellow")),
                ("Record ID", _styled(current_run["id"], "yellow")),
                ("Median (s)", f"{current_run['median_seconds']:.6f}"),
                ("Pinned At", current_baseline["created_at"]),
                ("Note", current_baseline.get("note") or "-"),
            ],
        )
    )
    _console().print(
        render_table(
            f"Baseline History: {history['suite_name']}",
            [("Current", "right"), ("Event", "right"), ("Run ID", "right"), ("Record ID", "right"), "Configuration", ("Median (s)", "right"), "Note", "Pinned At"],
            [
                (
                    "yes" if entry["is_current"] else "",
                    entry["event_id"],
                    entry["run"]["display_id"],
                    entry["run"]["id"],
                    dump_json(entry["run"]["configuration"]),
                    f"{entry['run']['median_seconds']:.6f}",
                    entry.get("note") or "-",
                    entry["created_at"],
                )
                for entry in history["history"]
            ],
        )
    )
