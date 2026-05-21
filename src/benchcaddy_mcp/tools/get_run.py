from __future__ import annotations

from typing import Any

from benchcaddy.db import get_run_details

from .._app import app
from .._shared import _analysis_options, _invalid_run_id_response, _resolved_database_path, _response, _run_id


@app.tool(description="Inspect one benchmark run, including environment metadata and stored observations.")
def get_run(
    run_id: int | str,
    database_path: str | None = None,
    include_analysis: bool = False,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 0,
    noise_cv_threshold: float = 0.05,
    noise_ci_ratio_threshold: float = 0.10,
    outlier_z_threshold: float = 3.5,
    significance_level: float = 0.05,
    regression_threshold_percent: float = 5.0,
    drift_window_size: int = 5,
) -> dict[str, Any]:
    try:
        normalized_run_id = _run_id(run_id)
    except ValueError:
        return _invalid_run_id_response(tool_name="get_run", run_id=run_id, suggested_action="Use a run ID like 3 or 3.2.")

    resolved_database_path = _resolved_database_path(database_path)
    run = get_run_details(
        normalized_run_id,
        resolved_database_path,
        analysis_options=_analysis_options(
            confidence_level=confidence_level,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            noise_cv_threshold=noise_cv_threshold,
            noise_ci_ratio_threshold=noise_ci_ratio_threshold,
            outlier_z_threshold=outlier_z_threshold,
            significance_level=significance_level,
            regression_threshold_percent=regression_threshold_percent,
            drift_window_size=drift_window_size,
        ),
        include_analysis=include_analysis,
    )
    if run is None:
        return _response(
            tool_name="get_run",
            status="fail",
            reason="run_not_found",
            error_code="run_not_found",
            suggested_action="Use get_suite or list_suites to inspect available runs.",
        )
    return _response(
        tool_name="get_run",
        status="pass",
        reason="run_details_available",
        result={
            "mode": "run",
            "database_path": resolved_database_path,
            "run": run,
        },
        suggested_action="Use compare_runs with this run ID and a candidate run for analysis.",
        confidence=None,
    )