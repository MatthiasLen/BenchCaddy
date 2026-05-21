from __future__ import annotations

from typing import Any

from benchcaddy.db import compare_runs as _compare_runs

from .._app import app
from .._shared import _analysis_options, _comparison_response, _invalid_run_id_response, _response, _run_id


@app.tool(description="Compare two specific runs head-to-head.")
def compare_runs(
    left_run_id: int | str,
    right_run_id: int | str,
    database_path: str | None = None,
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
        normalized_left = _run_id(left_run_id)
    except ValueError:
        return _invalid_run_id_response(tool_name="compare_runs", run_id=left_run_id, suggested_action="Use run IDs like 3 or 3.2.")
    try:
        normalized_right = _run_id(right_run_id)
    except ValueError:
        return _invalid_run_id_response(tool_name="compare_runs", run_id=right_run_id, suggested_action="Use run IDs like 3 or 3.2.")

    comparison = _compare_runs(
        normalized_left,
        normalized_right,
        database_path,
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
    )
    if comparison is None:
        return _response(
            tool_name="compare_runs",
            status="fail",
            reason="run_not_found",
            error_code="run_not_found",
            suggested_action="Use get_run or get_suite to inspect available run IDs.",
        )

    comparison["comparison_mode"] = "direct"
    return _comparison_response(tool_name="compare_runs", comparison=comparison, confidence_level=confidence_level, mode="direct")
