from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.text import Text

from ..core import Sweep
from ..db import get_database_path
from ..isolation import validate_isolated_target
from ..presentation import dump_json, format_return_value, render_table
from ._shared import _STATE, DatabaseOption, _console, _emit_json_response, _raise_cli_error, app

_JSON_SCALAR_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def _raise_sweep_usage_error(
    message: str,
    *,
    json_output: bool,
    reason: str = "invalid_sweep_request",
    error_code: str = "invalid_sweep_request",
    suggested_action: str | None = None,
) -> None:
    _raise_cli_error(
        command="sweep",
        json_output=json_output,
        exit_code=2,
        message=message,
        reason=reason,
        error_code=error_code,
        suggested_action=suggested_action,
    )


def _resolve_sweep_target(target_reference: str, *, json_output: bool) -> Any:
    module_name, separator, qualname = target_reference.partition(":")
    if not separator or not module_name or not qualname or ":" in qualname:
        _raise_sweep_usage_error(
            "Target must use 'module:qualname' format.",
            json_output=json_output,
            reason="invalid_target_reference",
            error_code="invalid_target_reference",
            suggested_action="Pass a target like package.module:function or package.module:Class.method.",
        )

    cwd = str(Path.cwd())
    if "" not in sys.path and cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        target: Any = importlib.import_module(module_name)
    except Exception as error:
        _raise_sweep_usage_error(
            f"Could not import module '{module_name}': {error}",
            json_output=json_output,
            reason="target_module_import_failed",
            error_code="target_module_import_failed",
            suggested_action="Verify the module path and make sure it is importable from the current working directory.",
        )

    try:
        for attribute in qualname.split("."):
            target = getattr(target, attribute)
    except AttributeError:
        _raise_sweep_usage_error(
            f"Target '{target_reference}' was not found.",
            json_output=json_output,
            reason="target_not_found",
            error_code="target_not_found",
            suggested_action="Verify the target qualname after the module path.",
        )

    if not callable(target):
        _raise_sweep_usage_error(
            f"Target '{target_reference}' does not resolve to a callable.",
            json_output=json_output,
            reason="target_not_callable",
            error_code="target_not_callable",
            suggested_action="Point sweep at a callable function or method.",
        )

    try:
        validate_isolated_target(target)
    except TypeError as error:
        _raise_sweep_usage_error(
            str(error),
            json_output=json_output,
            reason="invalid_isolated_target",
            error_code="invalid_isolated_target",
            suggested_action="Use a target compatible with isolated benchmark execution.",
        )

    return target


def _parse_param_values(raw_values: str, *, json_output: bool) -> list[object]:
    if not raw_values:
        _raise_sweep_usage_error(
            "--param requires at least one value.",
            json_output=json_output,
            reason="missing_param_values",
            error_code="missing_param_values",
            suggested_action="Pass --param name=value1,value2 or --param name=[...].",
        )

    stripped_values = raw_values.strip()
    if stripped_values.startswith("["):
        try:
            parsed_values = json.loads(stripped_values)
        except json.JSONDecodeError as error:
            _raise_sweep_usage_error(
                f"--param JSON arrays must be valid JSON: {error.msg}.",
                json_output=json_output,
                reason="invalid_param_json",
                error_code="invalid_param_json",
                suggested_action="Use a valid JSON array for complex parameter values.",
            )

        if not isinstance(parsed_values, list):
            _raise_sweep_usage_error(
                "--param JSON input must be an array.",
                json_output=json_output,
                reason="invalid_param_json_type",
                error_code="invalid_param_json_type",
                suggested_action="Use --param name=[...] for JSON parameter lists.",
            )
        if not parsed_values:
            _raise_sweep_usage_error(
                "--param JSON arrays must not be empty.",
                json_output=json_output,
                reason="empty_param_values",
                error_code="empty_param_values",
                suggested_action="Provide at least one parameter value in the JSON array.",
            )
        return parsed_values

    if stripped_values.startswith("{"):
        _raise_sweep_usage_error(
            "--param object values must use a JSON array like name=[{...}].",
            json_output=json_output,
            reason="invalid_param_object_value",
            error_code="invalid_param_object_value",
            suggested_action="Wrap object values in a JSON array like name=[{...}].",
        )

    values: list[object] = []
    for raw_value in stripped_values.split(","):
        value_text = raw_value.strip()
        if not value_text:
            _raise_sweep_usage_error(
                "--param values must not be empty.",
                json_output=json_output,
                reason="empty_param_values",
                error_code="empty_param_values",
                suggested_action="Remove empty comma-separated values from --param.",
            )
        if value_text[0] in {"[", "{"}:
            _raise_sweep_usage_error(
                "Complex --param values must use a JSON array like name=[...].",
                json_output=json_output,
                reason="invalid_param_complex_value",
                error_code="invalid_param_complex_value",
                suggested_action="Use --param name=[...] for complex values.",
            )
        if value_text[0] == '"':
            try:
                values.append(json.loads(value_text))
            except json.JSONDecodeError as error:
                _raise_sweep_usage_error(
                    f"Quoted --param values must be valid JSON strings: {error.msg}.",
                    json_output=json_output,
                    reason="invalid_param_json_string",
                    error_code="invalid_param_json_string",
                    suggested_action="Wrap quoted parameter values in valid JSON string syntax.",
                )
            continue
        if value_text in {"true", "false", "null"} or _JSON_SCALAR_RE.fullmatch(value_text):
            try:
                values.append(json.loads(value_text))
            except json.JSONDecodeError as error:
                _raise_sweep_usage_error(
                    f"Invalid JSON scalar in --param value '{value_text}': {error.msg}.",
                    json_output=json_output,
                    reason="invalid_param_scalar",
                    error_code="invalid_param_scalar",
                    suggested_action="Pass scalar parameter values as valid JSON scalars or plain text.",
                )
            continue
        values.append(value_text)

    return values


def _parse_params(entries: list[str], *, json_output: bool) -> dict[str, list[object]]:
    params: dict[str, list[object]] = {}
    for entry in entries:
        key, separator, raw_values = entry.partition("=")
        key = key.strip()
        if not separator or not key:
            _raise_sweep_usage_error(
                "--param entries must use 'name=value1,value2' or 'name=[...]' format.",
                json_output=json_output,
                reason="invalid_param_entry",
                error_code="invalid_param_entry",
                suggested_action="Pass each --param as name=value1,value2 or name=[...].",
            )
        if key in params:
            _raise_sweep_usage_error(
                f"Duplicate --param key '{key}'.",
                json_output=json_output,
                reason="duplicate_param_key",
                error_code="duplicate_param_key",
                suggested_action=f"Remove the duplicate --param key '{key}'.",
            )
        params[key] = _parse_param_values(raw_values.strip(), json_output=json_output)
    return params


@app.command("sweep", help="Run a benchmark sweep for an importable target reference like 'package.module:function' or 'package.module:Class.method'.")
def sweep_command(
    target_reference: Annotated[str, typer.Argument(metavar="MODULE:QUALNAME")],
    suite_name: Annotated[
        str,
        typer.Option(
            "--suite-name",
            help="Suite name recorded for the sweep.",
        ),
    ],
    param: Annotated[
        list[str],
        typer.Option(
            "--param",
            help="Parameter grid entry as name=value1,value2 or name=[...]",
        ),
    ]
    | None = None,
    samples: Annotated[
        int,
        typer.Option(
            "--samples",
            min=1,
            help="Number of measured samples per configuration.",
        ),
    ] = 7,
    warmup_iterations: Annotated[
        int,
        typer.Option(
            "--warmup-iterations",
            min=0,
            help="Warmup runs executed before measurement begins.",
        ),
    ] = 1,
    lock_cpu_affinity: Annotated[
        bool,
        typer.Option(
            "--lock-cpu-affinity/--no-lock-cpu-affinity",
            help="Preserve the current CPU affinity set before benchmarking.",
        ),
    ] = True,
    store_target_return_value: Annotated[
        bool,
        typer.Option(
            "--store-target-return-value",
            help="Store one supported target return value per run.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            "-j",
            help="Emit machine-readable sweep results as JSON.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show structured progress output during the sweep run.",
        ),
    ] = False,
    database: DatabaseOption = None,
) -> None:
    effective_verbose = _STATE.verbose or verbose

    if json_output and effective_verbose:
        _raise_sweep_usage_error(
            "--json cannot be combined with --verbose.",
            json_output=json_output,
            reason="json_conflicts_with_verbose",
            error_code="json_conflicts_with_verbose",
            suggested_action="Drop --verbose when using -j/--json.",
        )

    target = _resolve_sweep_target(target_reference, json_output=json_output)
    params = _parse_params([] if param is None else param, json_output=json_output)
    database_path = get_database_path(database)

    results = Sweep(
        target=target,
        params=params,
        suite_name=suite_name,
        samples=samples,
        warmup_iterations=warmup_iterations,
        lock_cpu_affinity=lock_cpu_affinity,
        database_path=database,
        store_target_return_value=store_target_return_value,
        verbose=effective_verbose,
    ).run()

    if json_output:
        _emit_json_response(
            command="sweep",
            status="pass",
            reason="runs_recorded",
            suggested_action="Use benchcaddy show -j, compare -j, or trend -j on the recorded suite.",
            confidence=None,
            result={
                "target_reference": target_reference,
                "suite_name": suite_name,
                "database_path": database_path,
                "params": params,
                "samples": samples,
                "warmup_iterations": warmup_iterations,
                "lock_cpu_affinity": lock_cpu_affinity,
                "store_target_return_value": store_target_return_value,
                "run_count": len(results),
                "runs": [
                    {
                        "display_id": result.run_id,
                        "record_id": result.record_id,
                        "configuration": result.configuration,
                        "samples": result.samples,
                        "observations": result.observations,
                        "median_seconds": result.median_seconds,
                        "min_seconds": result.min_seconds,
                        "max_seconds": result.max_seconds,
                        "std_seconds": result.std_seconds,
                        "target_return_value": result.target_return_value,
                    }
                    for result in results
                ],
            },
        )
        return

    if not effective_verbose:
        _console().print(
            render_table(
                f"Recorded Runs: {suite_name}",
                ["Run ID", ("Record ID", "right"), "Configuration", ("Median (s)", "right"), ("Samples", "right"), "Return Value"],
                [
                    (
                        result.run_id,
                        result.record_id,
                        dump_json(result.configuration),
                        f"{result.median_seconds:.6f}",
                        len(result.samples),
                        format_return_value(result.target_return_value, compact=True),
                    )
                    for result in results
                ],
            )
        )

    run_ids = " ".join(result.run_id for result in results)
    _console().print(f"Completed {len(results)} runs. Follow-up commands:")
    _console().print(
        Text.assemble(
            ("Inspect details:  ", "bold green"),
            (f"benchcaddy show {run_ids} --database {database_path}",),
        )
    )
    _console().print(
        Text.assemble(
            ("Compare runs:     ", "bold yellow"),
            (f"benchcaddy compare {suite_name} --database {database_path}",),
        )
    )
    _console().print(
        Text.assemble(
            ("Trend history:    ", "bold magenta"),
            (f"benchcaddy trend {suite_name} --database {database_path}",),
        )
    )
