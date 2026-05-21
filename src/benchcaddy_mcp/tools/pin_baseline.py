from __future__ import annotations

from typing import Any

from benchcaddy.db import set_suite_baseline

from .._app import app
from .._shared import (
    ResponseDetail,
    _analysis_options,
    _invalid_response_detail_response,
    _invalid_run_id_response,
    _normalized_response_detail,
    _response,
    _run_id,
)


@app.tool(description="Pin one run as the suite baseline.")
def pin_baseline(
    suite_name: str,
    run_id: int | str,
    note: str | None = None,
    database_path: str | None = None,
    response_detail: ResponseDetail = "summary",
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
        return _invalid_response_detail_response(tool_name="pin_baseline", response_detail=response_detail)

    try:
        normalized_run_id = _run_id(run_id)
    except ValueError:
        return _invalid_run_id_response(
            tool_name="pin_baseline",
            run_id=run_id,
            suggested_action="Use a run ID like 3 or 3.2.",
            response_detail=normalized_response_detail,
        )

    pin_update = set_suite_baseline(
        suite_name,
        normalized_run_id,
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
        note=note,
        include_samples=normalized_response_detail == "full",
        include_observations=normalized_response_detail == "full",
        include_environment=normalized_response_detail == "full",
    )
    if pin_update is None:
        return _response(
            tool_name="pin_baseline",
            status="fail",
            reason="suite_not_found",
            error_code="suite_not_found",
            suggested_action="Use list_suites to inspect available suites.",
            response_detail=normalized_response_detail,
        )

    error_code = pin_update.get("error")
    if error_code == "reference_run_not_found":
        return _response(
            tool_name="pin_baseline",
            status="fail",
            reason="reference_run_not_found",
            error_code="reference_run_not_found",
            result=pin_update,
            suggested_action="Use get_run or get_suite to inspect available run IDs.",
            response_detail=normalized_response_detail,
        )
    if error_code == "reference_run_wrong_suite":
        return _response(
            tool_name="pin_baseline",
            status="fail",
            reason="reference_run_wrong_suite",
            error_code="reference_run_wrong_suite",
            result=pin_update,
            suggested_action="Choose a run ID from the requested suite before pinning a baseline.",
            response_detail=normalized_response_detail,
        )

    result = {"pin_update": pin_update}
    return _response(
        tool_name="pin_baseline",
        status="pass",
        reason="baseline_pinned",
        result=result,
        suggested_action="Use compare_suite or trend_suite with use_pinned_baseline=True.",
        confidence=None,
        response_detail=normalized_response_detail,
    )
