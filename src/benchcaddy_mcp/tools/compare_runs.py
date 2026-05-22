from __future__ import annotations

from typing import Any

from benchcaddy.db import compare_runs as _compare_runs

from .._app import app
from .._shared import (
    DEFAULT_RESPONSE_DETAIL,
    ResponseDetail,
    _analysis_options,
    _comparison_response,
    _invalid_response_detail_response,
    _invalid_run_id_response,
    _normalized_response_detail,
    _response,
    _run_id,
)


@app.tool(
    description=(
        "Compare two specific benchmark runs directly. Use this when the "
        "user asks to compare run 4.1 against run 4.2 or wants a "
        "head-to-head run comparison instead of a suite-wide comparison."
    )
)
def compare_runs(
    left_run_id: int | str,
    right_run_id: int | str,
    database_path: str | None = None,
    response_detail: ResponseDetail = DEFAULT_RESPONSE_DETAIL,
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
        normalized_response_detail = _normalized_response_detail(response_detail)
    except ValueError:
        return _invalid_response_detail_response(tool_name="compare_runs", response_detail=response_detail)

    try:
        normalized_left = _run_id(left_run_id)
    except ValueError:
        return _invalid_run_id_response(
            tool_name="compare_runs",
            run_id=left_run_id,
            suggested_action="Use run IDs like 3 or 3.2.",
            response_detail=normalized_response_detail,
        )
    try:
        normalized_right = _run_id(right_run_id)
    except ValueError:
        return _invalid_run_id_response(
            tool_name="compare_runs",
            run_id=right_run_id,
            suggested_action="Use run IDs like 3 or 3.2.",
            response_detail=normalized_response_detail,
        )

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
        include_samples=normalized_response_detail == "full",
        include_observations=normalized_response_detail == "full",
        include_environment=normalized_response_detail == "full",
    )
    if comparison is None:
        return _response(
            tool_name="compare_runs",
            status="fail",
            reason="run_not_found",
            error_code="run_not_found",
            suggested_action="Use get_run to inspect that run ID or get_suite to browse valid run IDs in the target suite.",
            response_detail=normalized_response_detail,
        )

    comparison["comparison_mode"] = "direct"
    return _comparison_response(
        tool_name="compare_runs",
        comparison=comparison,
        confidence_level=confidence_level,
        mode="direct",
        response_detail=normalized_response_detail,
    )
