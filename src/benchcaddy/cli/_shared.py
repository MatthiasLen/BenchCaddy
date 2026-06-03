from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from rich.console import Console

from ..presentation import serialize_json
from ..stats import AnalysisOptions

app = typer.Typer(
    help="BenchCaddy\n\n"
    "Record, manage, and statistically analyze benchmarks. "
    "Compare runs and track performance trends.",
    no_args_is_help=True
)
console = Console()
REGRESSION_EXIT_CODE = 3
JSON_SCHEMA_VERSION = "1.0"
_JSON_SCALAR_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$")

DatabaseOption = Annotated[
    Path | None,
    typer.Option(
        "--database",
        "-d",
        exists=False,
        dir_okay=False,
        help="Path to the BenchCaddy SQLite database.",
    ),
]
CompareConfidenceLevelOption = Annotated[
    float,
    typer.Option(
        "--confidence-level",
        min=0.5,
        max=0.99,
        help="Bootstrap confidence level used for median and delta confidence intervals.",
    ),
]
CompareBootstrapResamplesOption = Annotated[
    int,
    typer.Option(
        "--bootstrap-resamples",
        min=100,
        help="Bootstrap and permutation resample count used for confidence intervals and significance estimates.",
    ),
]
NoiseThresholdOption = Annotated[
    float,
    typer.Option(
        "--noise-threshold",
        min=0.0,
        help="Coefficient-of-variation threshold (std/mean) used to flag noisy runs.",
    ),
]
SignificanceLevelOption = Annotated[
    float,
    typer.Option(
        "--significance-level",
        min=0.001,
        max=0.5,
        help="p-value threshold used when classifying regressions and improvements.",
    ),
]
RegressionThresholdOption = Annotated[
    float,
    typer.Option(
        "--regression-threshold",
        min=0.0,
        help="Practical regression threshold in percent relative to the baseline median.",
    ),
]


@dataclass
class CLIState:
    """Stores process-wide CLI flags shared across Typer commands."""

    verbose: bool = False


_STATE = CLIState()


@app.callback()
def callback(
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show additional detail in command output.",
        ),
    ] = False,
) -> None:
    _STATE.verbose = verbose


@app.command("help")
def help_command(ctx: typer.Context) -> None:
    """Show help information for BenchCaddy."""
    root_ctx = ctx.find_root()
    typer.echo(root_ctx.command.get_help(root_ctx))
    raise typer.Exit(0)


def _emit_json_response(
    *,
    command: str,
    status: str,
    reason: str,
    result: dict[str, object] | None = None,
    error_code: str | None = None,
    suggested_action: str | None = None,
    confidence: str | None = None,
    exit_code: int = 0,
) -> None:
    payload: dict[str, object] = {
        "schema_version": JSON_SCHEMA_VERSION,
        "command": command,
        "status": status,
        "reason": reason,
        "error_code": error_code,
        "suggested_action": suggested_action,
        "confidence": confidence,
        "exit_code": exit_code,
    }
    if result is not None:
        payload["result"] = result
    typer.echo(serialize_json(payload))


def _confidence_label(confidence_level: float | None) -> str | None:
    if confidence_level is None:
        return None
    if confidence_level >= 0.95:
        return "high"
    if confidence_level >= 0.8:
        return "medium"
    return "low"


def _raise_cli_error(
    *,
    command: str,
    json_output: bool,
    exit_code: int,
    message: str,
    reason: str,
    error_code: str,
    suggested_action: str | None = None,
    confidence: str | None = None,
    result: dict[str, object] | None = None,
) -> NoReturn:
    if json_output:
        _emit_json_response(
            command=command,
            status="fail",
            reason=reason,
            result=result,
            error_code=error_code,
            suggested_action=suggested_action,
            confidence=confidence,
            exit_code=exit_code,
        )
    else:
        _console().print(message)
    raise typer.Exit(code=exit_code)


def _console() -> Console:
    from . import console as root_console

    return root_console


def _analysis_options(
    *,
    confidence_level: float,
    bootstrap_resamples: int,
    noise_threshold: float = 0.05,
    significance_level: float = 0.05,
    regression_threshold: float = 5.0,
    window_size: int = 5,
) -> AnalysisOptions:
    return AnalysisOptions(
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        noise_cv_threshold=noise_threshold,
        significance_level=significance_level,
        regression_threshold_percent=regression_threshold,
        drift_window_size=window_size,
    )


def _as_run_id(value: str | None) -> int | tuple[int, int] | None:
    if value is None:
        return None
    if "." in value:
        left, dot, right = value.partition(".")
        if dot and left.isdigit() and right.isdigit():
            return (int(left), int(right))
    try:
        return int(value)
    except ValueError:
        return None


def _require_run_id(
    identifier: str,
    *,
    command: str | None = None,
    json_output: bool = False,
) -> int | tuple[int, int]:
    run_id = _as_run_id(identifier)
    if run_id is None:
        if command is None:
            _console().print(f"'{identifier}' is not a valid run ID.")
            raise typer.Exit(code=2)
        _raise_cli_error(
            command=command,
            json_output=json_output,
            exit_code=2,
            message=f"'{identifier}' is not a valid run ID.",
            reason="invalid_run_id",
            error_code="invalid_run_id",
            suggested_action="Use a run ID like 3 or 3.2.",
        )
    return run_id


def _parse_config_filter_value(
    value_text: str,
    *,
    option_name: str,
    command: str | None = None,
    json_output: bool = False,
) -> Any:
    normalized = value_text.strip()
    if not normalized:
        if command is None:
            _console().print(f"{option_name} entries must not use empty values.")
            raise typer.Exit(code=2)
        _raise_cli_error(
            command=command,
            json_output=json_output,
            exit_code=2,
            message=f"{option_name} entries must not use empty values.",
            reason="empty_config_filter_value",
            error_code="empty_config_filter_value",
            suggested_action=f"Pass {option_name} entries as key=value.",
        )

    if normalized[0] == '"':
        try:
            return json.loads(normalized)
        except json.JSONDecodeError as exc:
            if command is None:
                _console().print(f"Quoted {option_name} values must be valid JSON strings: {exc.msg}.")
                raise typer.Exit(code=2) from exc
            _raise_cli_error(
                command=command,
                json_output=json_output,
                exit_code=2,
                message=f"Quoted {option_name} values must be valid JSON strings: {exc.msg}.",
                reason="invalid_config_filter_value",
                error_code="invalid_config_filter_json_string",
                suggested_action=f"Wrap string values for {option_name} in valid JSON quotes.",
            )

    if normalized in {"true", "false", "null"} or _JSON_SCALAR_RE.fullmatch(normalized):
        try:
            return json.loads(normalized)
        except json.JSONDecodeError as exc:
            if command is None:
                _console().print(f"Invalid scalar value '{normalized}' for {option_name}: {exc.msg}.")
                raise typer.Exit(code=2) from exc
            _raise_cli_error(
                command=command,
                json_output=json_output,
                exit_code=2,
                message=f"Invalid scalar value '{normalized}' for {option_name}: {exc.msg}.",
                reason="invalid_config_filter_value",
                error_code="invalid_config_filter_scalar",
                suggested_action=f"Pass scalar {option_name} values as valid JSON scalars or plain text.",
            )

    return normalized


def _parse_config_filter_entries(
    entries: list[str],
    *,
    option_name: str = "-c",
    command: str | None = None,
    json_output: bool = False,
) -> dict[str, Any]:
    if not entries:
        if command is None:
            _console().print(f"{option_name} requires one or more key=value entries.")
            raise typer.Exit(code=2)
        _raise_cli_error(
            command=command,
            json_output=json_output,
            exit_code=2,
            message=f"{option_name} requires one or more key=value entries.",
            reason="missing_config_filter_entries",
            error_code="missing_config_filter_entries",
            suggested_action=f"Pass one or more {option_name} key=value entries.",
        )

    config_filter: dict[str, Any] = {}
    for entry in entries:
        key, separator, raw_value = entry.partition("=")
        key = key.strip()
        if not separator or not key:
            if command is None:
                _console().print(f"{option_name} entries must use 'key=value' format.")
                raise typer.Exit(code=2)
            _raise_cli_error(
                command=command,
                json_output=json_output,
                exit_code=2,
                message=f"{option_name} entries must use 'key=value' format.",
                reason="invalid_config_filter_entry",
                error_code="invalid_config_filter_entry",
                suggested_action=f"Pass each {option_name} entry as key=value.",
            )
        if key in config_filter:
            if command is None:
                _console().print(f"Duplicate {option_name} key '{key}'.")
                raise typer.Exit(code=2)
            _raise_cli_error(
                command=command,
                json_output=json_output,
                exit_code=2,
                message=f"Duplicate {option_name} key '{key}'.",
                reason="duplicate_config_filter_key",
                error_code="duplicate_config_filter_key",
                suggested_action=f"Remove the duplicate {option_name} key '{key}'.",
            )
        config_filter[key] = _parse_config_filter_value(
            raw_value,
            option_name=option_name,
            command=command,
            json_output=json_output,
        )
    return config_filter
