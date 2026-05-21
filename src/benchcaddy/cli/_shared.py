from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from ..presentation import serialize_json
from ..stats import AnalysisOptions

app = typer.Typer(help="Inspect BenchCaddy benchmark suites.")
console = Console()
REGRESSION_EXIT_CODE = 3
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


def _emit_json(payload: dict[str, object]) -> None:
    typer.echo(serialize_json(payload))


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


def _require_run_id(identifier: str) -> int | tuple[int, int]:
    run_id = _as_run_id(identifier)
    if run_id is None:
        _console().print(f"'{identifier}' is not a valid run ID.")
        raise typer.Exit(code=1)
    return run_id


def _parse_config_filter_value(value_text: str, *, option_name: str) -> Any:
    normalized = value_text.strip()
    if not normalized:
        _console().print(f"{option_name} entries must not use empty values.")
        raise typer.Exit(code=2)

    if normalized[0] == '"':
        try:
            return json.loads(normalized)
        except json.JSONDecodeError as exc:
            _console().print(f"Quoted {option_name} values must be valid JSON strings: {exc.msg}.")
            raise typer.Exit(code=2) from exc

    if normalized in {"true", "false", "null"} or _JSON_SCALAR_RE.fullmatch(normalized):
        try:
            return json.loads(normalized)
        except json.JSONDecodeError as exc:
            _console().print(f"Invalid scalar value '{normalized}' for {option_name}: {exc.msg}.")
            raise typer.Exit(code=2) from exc

    return normalized


def _parse_config_filter_entries(entries: list[str], *, option_name: str = "-c") -> dict[str, Any]:
    if not entries:
        _console().print(f"{option_name} requires one or more key=value entries.")
        raise typer.Exit(code=2)

    config_filter: dict[str, Any] = {}
    for entry in entries:
        key, separator, raw_value = entry.partition("=")
        key = key.strip()
        if not separator or not key:
            _console().print(f"{option_name} entries must use 'key=value' format.")
            raise typer.Exit(code=2)
        if key in config_filter:
            _console().print(f"Duplicate {option_name} key '{key}'.")
            raise typer.Exit(code=2)
        config_filter[key] = _parse_config_filter_value(raw_value, option_name=option_name)
    return config_filter
