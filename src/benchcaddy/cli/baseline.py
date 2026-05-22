from __future__ import annotations

from typing import Annotated

import typer
from rich.panel import Panel

from ..db import get_database_path, get_suite_baseline_history, set_suite_baseline
from ..presentation import dump_json, format_timestamp, render_table, summary_panel
from ._rendering import _styled
from ._shared import DatabaseOption, _console, _emit_json_response, _raise_cli_error, _require_run_id, app


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
            "-j",
            help="Emit machine-readable JSON output for baseline history.",
        ),
    ] = False,
    database: DatabaseOption = None,
) -> None:
    database_path = get_database_path(database)
    if note is not None and pin is None:
        _raise_cli_error(
            command="baseline",
            json_output=json_output,
            exit_code=2,
            message="--note requires --pin.",
            reason="missing_pin_for_note",
            error_code="missing_pin_for_note",
            suggested_action="Pass --pin RUN_ID when using --note.",
        )

    pin_update = None
    if pin is not None:
        pin_update = set_suite_baseline(
            suite_name,
            _require_run_id(pin, command="baseline", json_output=json_output),
            database_path,
            note=note,
        )
        if pin_update is None:
            _raise_cli_error(
                command="baseline",
                json_output=json_output,
                exit_code=1,
                message=f"Suite '{suite_name}' was not found in {database_path}.",
                reason="suite_not_found",
                error_code="suite_not_found",
                suggested_action="Use benchcaddy list -j to inspect available suites.",
            )
        if pin_update.get("error") == "reference_run_not_found":
            _raise_cli_error(
                command="baseline",
                json_output=json_output,
                exit_code=1,
                message=f"Reference run '{pin}' was not found in {database_path}.",
                reason="reference_run_not_found",
                error_code="reference_run_not_found",
                suggested_action="Use benchcaddy show -j to inspect available run IDs.",
            )
        if pin_update.get("error") == "reference_run_wrong_suite":
            _raise_cli_error(
                command="baseline",
                json_output=json_output,
                exit_code=1,
                message=f"Reference run '{pin}' belongs to suite '{pin_update['reference_run_suite_name']}', not '{suite_name}'.",
                reason="reference_run_wrong_suite",
                error_code="reference_run_wrong_suite",
                suggested_action="Choose a run ID from the requested suite before pinning a baseline.",
            )

    history = get_suite_baseline_history(suite_name, database_path, limit=limit)
    if history is None:
        _raise_cli_error(
            command="baseline",
            json_output=json_output,
            exit_code=1,
            message=f"Suite '{suite_name}' was not found in {database_path}.",
            reason="suite_not_found",
            error_code="suite_not_found",
            suggested_action="Use benchcaddy list -j to inspect available suites.",
        )

    if json_output:
        payload = dict(history)
        if pin_update is not None:
            payload["pin_update"] = pin_update
        current_baseline = payload.get("current_baseline")
        if pin_update is not None:
            status = "pass"
            reason = "baseline_pinned"
            suggested_action = "Use benchcaddy compare -j to compare new runs against the pinned baseline."
        elif current_baseline is None:
            status = "inconclusive"
            reason = "no_baseline_history"
            suggested_action = "Pin a known-good run with benchcaddy baseline SUITE --pin RUN_ID -j."
        else:
            status = "pass"
            reason = "baseline_available"
            suggested_action = "Use benchcaddy compare -j or benchcaddy trend -j with this suite."
        _emit_json_response(
            command="baseline",
            status=status,
            reason=reason,
            suggested_action=suggested_action,
            confidence=None,
            result=payload,
        )
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
                ("Pinned At", format_timestamp(current_baseline["created_at"])),
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
                    format_timestamp(entry["created_at"]),
                )
                for entry in history["history"]
            ],
        )
    )
