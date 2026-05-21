from __future__ import annotations

from typing import Any

from benchcaddy.db import get_suite_trend

from .._app import app
from .._shared import (
    DEFAULT_LIMIT,
    ResponseDetail,
    _analysis_options,
    _capped_rows,
    _confidence_label,
    _invalid_response_detail_response,
    _invalid_run_id_response,
    _normalized_response_detail,
    _response,
    _run_id,
)


@app.tool(description="Inspect how a benchmark suite or one configuration changes over time. Use this when the user asks about drift, history, regressions over time, or long-term trends for a suite.")
def trend_suite(
    suite_name: str,
    baseline_run_id: int | str | None = None,
    use_pinned_baseline: bool = False,
    limit: int | None = DEFAULT_LIMIT,
    config_filter: dict[str, Any] | None = None,
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
        return _invalid_response_detail_response(tool_name="trend_suite", response_detail=response_detail)

    try:
        normalized_baseline = None if baseline_run_id is None else _run_id(baseline_run_id)
    except ValueError:
        return _invalid_run_id_response(
            tool_name="trend_suite",
            run_id=baseline_run_id,
            suggested_action="Use a run ID like 3 or 3.2.",
            response_detail=normalized_response_detail,
        )

    trend = get_suite_trend(
        suite_name,
        database_path=database_path,
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
        baseline_run_id=normalized_baseline,
        use_pinned_baseline=use_pinned_baseline,
        limit=None,
        config_filter=config_filter,
        include_samples=normalized_response_detail == "full",
        include_observations=normalized_response_detail == "full",
        include_environment=normalized_response_detail == "full",
    )
    if trend is None:
        return _response(
            tool_name="trend_suite",
            status="fail",
            reason="suite_not_found",
            error_code="suite_not_found",
            suggested_action="Use list_suites to inspect available suites.",
            response_detail=normalized_response_detail,
        )

    error_code = trend.get("error")

    if error_code == "reference_run_not_found":
        return _response(
            tool_name="trend_suite",
            status="fail",
            reason="reference_run_not_found",
            error_code="reference_run_not_found",
            result=trend,
            suggested_action="Use get_run or get_suite to inspect available run IDs.",
            response_detail=normalized_response_detail,
        )
    if error_code == "reference_run_wrong_suite":
        return _response(
            tool_name="trend_suite",
            status="fail",
            reason="reference_run_wrong_suite",
            error_code="reference_run_wrong_suite",
            result=trend,
            suggested_action="Choose a baseline run from the requested suite.",
            response_detail=normalized_response_detail,
        )
    if error_code == "baseline_not_found":
        return _response(
            tool_name="trend_suite",
            status="fail",
            reason="baseline_not_found",
            error_code="baseline_not_found",
            result=trend,
            suggested_action="Call pin_baseline before using use_pinned_baseline.",
            response_detail=normalized_response_detail,
        )
    if error_code == "config_filter_no_matches":
        return _response(
            tool_name="trend_suite",
            status="inconclusive",
            reason="config_filter_no_matches",
            result={
                "suite_name": suite_name,
                "config_filter": config_filter,
            },
            suggested_action="Relax the filter or record more runs for that configuration.",
            confidence=None,
            response_detail=normalized_response_detail,
        )
    if error_code == "config_filter_conflicts_with_basis":
        return _response(
            tool_name="trend_suite",
            status="fail",
            reason="config_filter_conflicts_with_basis",
            error_code="config_filter_conflicts_with_basis",
            result=trend,
            suggested_action="Use either config_filter or an explicit or pinned basis, not both.",
            response_detail=normalized_response_detail,
        )
    if trend.get("mode") != "summary" and trend.get("basis_run") is None:
        return _response(
            tool_name="trend_suite",
            status="fail",
            reason="no_runs_found",
            error_code="no_runs_found",
            result=trend,
            suggested_action="Record one or more runs for this suite before trending it.",
            response_detail=normalized_response_detail,
        )

    if trend.get("mode") == "summary":
        return _response(
            tool_name="trend_suite",
            status="inconclusive",
            reason="multiple_configurations_summary",
            result=trend,
            suggested_action="Use config_filter or an explicit baseline run ID to inspect one configuration timeline.",
            confidence=_confidence_label(confidence_level),
            response_detail=normalized_response_detail,
        )

    _capped_rows(trend, "runs", limit)
    confidence = _confidence_label(confidence_level)
    runs = trend.get("runs") or []
    
    if any((run.get("vs_baseline") or {}).get("regression_detected") for run in runs):
        return _response(
            tool_name="trend_suite",
            status="fail",
            reason="regression_detected",
            result=trend,
            suggested_action="Inspect the regressing runs in result.runs before accepting the change.",
            confidence=confidence,
            response_detail=normalized_response_detail,
        )
    if any((run.get("vs_baseline") or {}).get("warnings") or run.get("drift_status") == "noisy" for run in runs):
        return _response(
            tool_name="trend_suite",
            status="inconclusive",
            reason="noisy_samples",
            result=trend,
            suggested_action="Increase samples or reduce environmental noise, then rerun trend_suite.",
            confidence=confidence,
            response_detail=normalized_response_detail,
        )
    return _response(
        tool_name="trend_suite",
        status="pass",
        reason="trend_timeline_available",
        result=trend,
        suggested_action="Use the timeline payload to inspect drift and baseline deltas.",
        confidence=confidence,
        response_detail=normalized_response_detail,
    )
