from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ..presentation import serialize_json
from ..stats import AnalysisOptions

app = typer.Typer(help="Inspect BenchCaddy benchmark suites.")
console = Console()
REGRESSION_EXIT_CODE = 3

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