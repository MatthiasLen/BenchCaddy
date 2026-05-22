from __future__ import annotations

from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..db import get_all_run_count, get_all_run_details, get_database_path, get_run_details, get_selected_run_details, get_suite_details, get_suite_run_count
from ..observability import summarize_observations
from ..presentation import (
    dump_json,
    format_interval,
    format_ratio,
    format_return_value,
    format_time_summary,
    format_timestamp,
    format_warning_list,
    json_panel,
    render_table,
    summary_panel,
)
from ._rendering import _format_optional_seconds, _styled
from ._shared import _STATE, DatabaseOption, _as_run_id, _console, _emit_json_response, _parse_config_filter_entries, _raise_cli_error, _require_run_id, app


def _has_analysis(run: dict[str, object]) -> bool:
    return run.get("analysis") is not None


def _run_analysis_panel(run: dict[str, object], title: str = "Statistical Summary") -> Panel:
    analysis = run.get("analysis") or {}
    return summary_panel(
        title,
        [
            ("Median CI (s)", format_interval(run.get("ci_lower_seconds"), run.get("ci_upper_seconds"))),
            ("MAD (s)", _format_optional_seconds(run.get("mad_seconds"))),
            ("CV", format_ratio(run.get("coefficient_of_variation"))),
            ("Warnings", format_warning_list(run.get("noise_warnings"))),
            ("Sample Count", str(analysis.get("sample_count", len(run.get("samples", []))))),
        ],
    )


def _render_observation_table(observations: list[dict[str, object]], title: str, min_width: int | None = None) -> Table:
    summary = summarize_observations(observations)

    return render_table(
        title,
        ["Label", ("Calls", "right"), ("Mean +- Std (s)", "right"), ("Total (s)", "right")],
        [
            (
                label,
                stats.calls,
                format_time_summary(stats.mean_seconds, stats.std_seconds),
                f"{stats.total_seconds:.6f}",
            )
            for label, stats in summary.items()
        ],
        min_width=min_width,
    )


def _run_table_columns(*, include_suite: bool = False, include_target: bool = False) -> list[object]:
    columns: list[object] = [("Run ID", "right"), ("Record ID", "right")]
    if include_suite:
        columns.append("Suite")
    if include_target:
        columns.append("Target")
    columns.extend(
        [
            "Configuration",
            ("Mean +- Std (s)", "right"),
            "Return Value",
            ("Samples", "right"),
            "Recorded At",
        ]
    )
    return columns


def _run_table_row(
    run: dict[str, object],
    *,
    include_suite: bool = False,
    include_target: bool = False,
) -> tuple[object, ...]:
    row: list[object] = [run["display_id"], run.get("record_id", run["id"])]
    if include_suite:
        row.append(run["suite_name"])
    if include_target:
        row.append(run["target_name"])
    row.extend(
        [
            dump_json(run["configuration"]),
            format_time_summary(run.get("mean_seconds"), run.get("std_seconds")),
            format_return_value(run.get("target_return_value"), compact=True),
            len(run["samples"]),
            format_timestamp(run["created_at"]),
        ]
    )
    return tuple(row)


def _render_run_table(
    title: str,
    runs: list[dict[str, object]],
    *,
    include_suite: bool = False,
    include_target: bool = False,
) -> Table:
    return render_table(
        title,
        _run_table_columns(include_suite=include_suite, include_target=include_target),
        [_run_table_row(run, include_suite=include_suite, include_target=include_target) for run in runs],
    )


def _observed_timing_rows(
    runs: list[dict[str, object]],
    *,
    include_record_id: bool = False,
) -> list[tuple[object, ...]]:
    return [
        (
            run["display_id"],
            *([run.get("record_id", run["id"])] if include_record_id else []),
            label,
            stats.calls,
            format_time_summary(stats.mean_seconds, stats.std_seconds),
        )
        for run in runs
        for label, stats in summarize_observations(run["observations"]).items()
    ]


def _show_run(run: dict[str, object]) -> None:
    detail_rows: list[tuple[object, object]] = [
        ("Run ID", run["display_id"]),
        ("Record ID", run["id"]),
        ("Sweep ID", run["sweep_id"]),
        ("Run Index", run["run_index"]),
        ("Suite", run["suite_name"]),
        ("Target", run["target_name"]),
        ("Configuration", dump_json(run["configuration"])),
        ("Mean +- Std (s)", format_time_summary(run.get("mean_seconds"), run.get("std_seconds"))),
        ("Min (s)", _format_optional_seconds(run.get("min_seconds"))),
        ("Max (s)", _format_optional_seconds(run.get("max_seconds"))),
    ]
    if _has_analysis(run):
        detail_rows.extend(
            [
                ("Median CI (s)", format_interval(run.get("ci_lower_seconds"), run.get("ci_upper_seconds"))),
                ("MAD (s)", _format_optional_seconds(run.get("mad_seconds"))),
                ("CV", format_ratio(run.get("coefficient_of_variation"))),
                ("Warnings", format_warning_list(run.get("noise_warnings"))),
            ]
        )
    detail_rows.extend(
        [
            ("Return Value", format_return_value(run.get("target_return_value"), compact=True)),
            ("Samples", len(run["samples"])),
            ("Recorded At", format_timestamp(run["created_at"])),
        ]
    )
    _console().print(
        render_table(
            f"Run: {run['display_id']}",
            ["Field", "Value"],
            detail_rows,
            min_width=100,
        )
    )
    if _has_analysis(run):
        _console().print(_run_analysis_panel(run))
    _console().print(_render_observation_table(run["observations"], title="Observed Timings", min_width=100))
    _console().print(json_panel("Environment", run["environment"], indent=2, fit=False))


def _show_suite(details: dict[str, object]) -> None:
    title = f"Suite: {details['suite_name']}"
    config_filter = details.get("config_filter") or {}
    if config_filter:
        title = f"{title} (config: {', '.join(f'{key}={config_filter[key]}' for key in sorted(config_filter))})"

    _console().print(_render_run_table(title, details["runs"]))
    if details.get("baseline_run") is not None:
        baseline_run = details["baseline_run"]
        rows: list[tuple[object, object]] = [
            ("Run ID", _styled(baseline_run["display_id"], "yellow")),
            ("Record ID", _styled(baseline_run["id"], "yellow")),
        ]
        if _has_analysis(baseline_run):
            rows.append(("Median CI (s)", format_interval(baseline_run.get("ci_lower_seconds"), baseline_run.get("ci_upper_seconds"))))
        rows.append(("Configuration", dump_json(baseline_run["configuration"])))
        _console().print(summary_panel("Baseline", rows))
    _console().print(
        render_table(
            f"Observed Timings: {details['suite_name']}",
            [("Run ID", "right"), "Label", ("Calls", "right"), ("Mean +- Std (s)", "right")],
            _observed_timing_rows(details["runs"]),
        )
    )
    if details["environment"] is not None:
        _console().print(json_panel("Environment", details["environment"], indent=2))
    if _STATE.verbose:
        for run in details["runs"]:
            _console().print(_render_observation_table(run["observations"], title=f"Observed Timings for Run {run['display_id']}"))


def _show_selected_runs(runs: list[dict[str, object]]) -> None:
    _console().print(_render_run_table("Selected Runs", runs, include_suite=True, include_target=True))
    _console().print(
        render_table(
            "Observed Timings: Selected Runs",
            [("Run ID", "right"), ("Record ID", "right"), "Label", ("Calls", "right"), ("Mean +- Std (s)", "right")],
            _observed_timing_rows(runs, include_record_id=True),
        )
    )


def _show_all_runs(runs: list[dict[str, object]]) -> None:
    _console().print(_render_run_table("All Runs", runs, include_suite=True))


def _limit_runs(runs: list[dict[str, object]], numitems: int | None) -> tuple[list[dict[str, object]], bool]:
    if numitems is None or len(runs) <= numitems:
        return runs, False
    return runs[:numitems], True


def _print_numitems_notice(
    *,
    shown_count: int,
    total_count: int,
    identifiers: list[str] | None,
    database_path: str | None = None,
) -> None:
    command_parts = ["benchcaddy", "show", *(identifiers or []), "-n", str(total_count)]
    if database_path is not None:
        command_parts.extend(["--database", database_path])
    command = " ".join(command_parts)
    _console().print(
        Text.assemble(
            ("Output capped to the latest entries by record ID. ", "bold bright_cyan"),
            (f"Showing latest {shown_count} entries. ", "bright_black"),
            ("Run ", "bright_black"),
            (command, "bold yellow"),
            (" to show all entries.", "bright_black"),
        )
    )


@app.command("show", help="Inspect all recorded runs, a suite, or specific run IDs. When a suite has a recorded baseline, it is shown in the suite view.")
def show_command(
    identifiers: Annotated[
        list[str] | None,
        typer.Argument(
            help="Suite name or one or more run IDs to inspect (for example 3.2 5 7.1). Omit identifiers to list all recorded runs.",
        ),
    ] = None,
    numitems: Annotated[
        int | None,
        typer.Option(
            "--numitems",
            "-n",
            min=1,
            help="Limit list-style show output to the latest N entries by record ID.",
        ),
    ] = 100,
    config: Annotated[
        bool,
        typer.Option(
            "--config",
            "-c",
            help="Restrict a suite view to runs whose configuration contains the trailing key=value pairs.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Emit machine-readable run or suite details as JSON.",
        ),
    ] = False,
    database: DatabaseOption = None,
) -> None:
    database_path = get_database_path(database)
    if config:
        if not identifiers or len(identifiers) < 2:
            _raise_cli_error(
                command="show",
                json_output=json_output,
                exit_code=2,
                message="--config/-c requires a suite name followed by one or more key=value entries.",
                reason="missing_config_filter_scope",
                error_code="missing_config_filter_scope",
                suggested_action="Pass a suite name followed by one or more -c key=value entries.",
            )
        if _as_run_id(identifiers[0]) is not None:
            _raise_cli_error(
                command="show",
                json_output=json_output,
                exit_code=2,
                message="--config/-c only supports suite views, not run IDs.",
                reason="config_filter_requires_suite",
                error_code="config_filter_requires_suite",
                suggested_action="Use a suite name, not a run ID, when combining show with --config/-c.",
            )

        config_filter = _parse_config_filter_entries(
            identifiers[1:],
            option_name="-c",
            command="show",
            json_output=json_output,
        )
        details = get_suite_details(
            identifiers[0],
            database_path,
            include_analysis=True,
            limit=numitems,
            config_filter=config_filter,
        )
        if details is None:
            _raise_cli_error(
                command="show",
                json_output=json_output,
                exit_code=1,
                message=f"Suite '{identifiers[0]}' was not found in {database_path}.",
                reason="suite_not_found",
                error_code="suite_not_found",
                suggested_action="Use benchcaddy list -j to inspect available suites.",
            )
        total_count = None
        truncated = False
        if numitems is not None and len(details["runs"]) == numitems:
            total_count = get_suite_run_count(identifiers[0], database_path, config_filter=config_filter)
            truncated = total_count is not None and total_count > numitems
        if json_output:
            _emit_json_response(
                command="show",
                status="pass" if details["runs"] else "inconclusive",
                reason="suite_details_available" if details["runs"] else "no_runs_matched_scope",
                suggested_action=("Use benchcaddy compare -j or trend -j on this suite." if details["runs"] else "Relax the filter or record more runs for this suite."),
                confidence=None,
                result={
                    "mode": "suite",
                    "database_path": database_path,
                    "truncated": truncated,
                    "total_run_count": total_count,
                    **details,
                },
            )
            return
        _show_suite(details)

        if truncated and total_count is not None:
            _print_numitems_notice(
                shown_count=len(details["runs"]),
                total_count=total_count,
                identifiers=[identifiers[0], "-c", *identifiers[1:]],
                database_path=None if database is None else str(database_path),
            )
        return

    if not identifiers:
        runs = get_all_run_details(database_path, limit=numitems)
        total_count = None
        truncated = False
        if numitems is not None and len(runs) == numitems:
            total_count = get_all_run_count(database_path)
            truncated = total_count > numitems
        if json_output:
            _emit_json_response(
                command="show",
                status="pass" if runs else "inconclusive",
                reason="run_list_available" if runs else "no_runs_found",
                suggested_action=(
                    "Use benchcaddy show -j RUN_ID or benchcaddy compare -j SUITE for a narrower view." if runs else "Record a benchmark sweep before requesting run details."
                ),
                confidence=None,
                result={
                    "mode": "all_runs",
                    "database_path": database_path,
                    "run_count": len(runs),
                    "truncated": truncated,
                    "total_run_count": total_count,
                    "runs": runs,
                },
            )
            return
        _show_all_runs(runs)
        if truncated and total_count is not None:
            _print_numitems_notice(
                shown_count=numitems,
                total_count=total_count,
                identifiers=None,
                database_path=None if database is None else str(database_path),
            )
        return

    if len(identifiers) == 1:
        identifier = identifiers[0]
        run_id = _as_run_id(identifier)
        if run_id is not None:
            run = get_run_details(
                run_id,
                database_path,
                include_analysis=True,
            )
            if run is None:
                _raise_cli_error(
                    command="show",
                    json_output=json_output,
                    exit_code=1,
                    message=f"Run '{identifier}' was not found in {database_path}.",
                    reason="run_not_found",
                    error_code="run_not_found",
                    suggested_action="Use benchcaddy show -j without a run ID to inspect available runs.",
                )
            if json_output:
                _emit_json_response(
                    command="show",
                    status="pass",
                    reason="run_details_available",
                    suggested_action="Use benchcaddy compare -j with this run ID and a candidate run for analysis.",
                    confidence=None,
                    result={
                        "mode": "run",
                        "database_path": database_path,
                        "run": run,
                    },
                )
                return
            _show_run(run)
            return

        details = get_suite_details(
            identifier,
            database_path,
            include_analysis=True,
            limit=numitems,
        )
        if details is None:
            _raise_cli_error(
                command="show",
                json_output=json_output,
                exit_code=1,
                message=f"Suite '{identifier}' was not found in {database_path}.",
                reason="suite_not_found",
                error_code="suite_not_found",
                suggested_action="Use benchcaddy list -j to inspect available suites.",
            )
        total_count = None
        truncated = False
        if numitems is not None and len(details["runs"]) == numitems:
            total_count = get_suite_run_count(identifier, database_path)
            truncated = total_count is not None and total_count > numitems
        if json_output:
            _emit_json_response(
                command="show",
                status="pass" if details["runs"] else "inconclusive",
                reason="suite_details_available" if details["runs"] else "suite_has_no_runs",
                suggested_action=("Use benchcaddy compare -j or trend -j on this suite." if details["runs"] else "Record new runs for this suite before comparing or trending it."),
                confidence=None,
                result={
                    "mode": "suite",
                    "database_path": database_path,
                    "truncated": truncated,
                    "total_run_count": total_count,
                    **details,
                },
            )
            return
        _show_suite(details)

        if truncated and total_count is not None:
            _print_numitems_notice(
                shown_count=len(details["runs"]),
                total_count=total_count,
                identifiers=identifiers,
                database_path=None if database is None else str(database_path),
            )
        return

    run_ids = [_require_run_id(identifier, command="show", json_output=json_output) for identifier in identifiers]
    runs = get_selected_run_details(run_ids, database_path)

    if runs is None:
        _raise_cli_error(
            command="show",
            json_output=json_output,
            exit_code=1,
            message=f"One or more runs were not found in {database_path}.",
            reason="selected_run_not_found",
            error_code="selected_run_not_found",
            suggested_action="Use benchcaddy show -j without explicit run IDs to inspect available runs.",
        )

    visible_runs, was_limited = _limit_runs(runs, numitems)
    if json_output:
        _emit_json_response(
            command="show",
            status="pass" if visible_runs else "inconclusive",
            reason="selected_runs_available" if visible_runs else "no_selected_runs_visible",
            suggested_action="Use benchcaddy compare -j with these run IDs for direct analysis.",
            confidence=None,
            result={
                "mode": "selected_runs",
                "database_path": database_path,
                "requested_run_ids": identifiers,
                "run_count": len(visible_runs),
                "truncated": was_limited,
                "total_run_count": len(runs),
                "runs": visible_runs,
            },
        )
        return
    _show_selected_runs(visible_runs)

    if was_limited:
        _print_numitems_notice(
            shown_count=len(visible_runs),
            total_count=len(runs),
            identifiers=identifiers,
            database_path=None if database is None else str(database_path),
        )
