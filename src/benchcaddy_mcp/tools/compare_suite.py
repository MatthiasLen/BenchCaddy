from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from benchcaddy.db import compare_suite_runs as _compare_suite_runs

from .._app import app
from .._shared import (
    DEFAULT_LIMIT,
    ResponseDetail,
    _analysis_options,
    _capped_rows,
    _comparison_response,
    _invalid_response_detail_response,
    _invalid_run_id_response,
    _normalized_response_detail,
    _response,
    _run_id,
)


@app.tool(description="Compare an entire suite against a chosen reference run, best run, or pinned baseline. Use this when the user asks to compare a suite against run 4.1, a baseline, or the best run.")
def compare_suite(
    suite_name: str,
    reference_run_id: int | str | None = None,
    use_pinned_baseline: bool = False,
    strict_keys: Sequence[str] = (),
    config_filter: dict[str, Any] | None = None,
    limit: int | None = DEFAULT_LIMIT,
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
        return _invalid_response_detail_response(tool_name="compare_suite", response_detail=response_detail)

    try:
        normalized_reference = None if reference_run_id is None else _run_id(reference_run_id)
    except ValueError:
        return _invalid_run_id_response(
            tool_name="compare_suite",
            run_id=reference_run_id,
            suggested_action="Use a run ID like 3 or 3.2.",
            response_detail=normalized_response_detail,
        )

    comparison = _compare_suite_runs(
        suite_name,
        reference_run_id=normalized_reference,
        strict_keys=strict_keys,
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
        use_pinned_baseline=use_pinned_baseline,
        config_filter=config_filter,
    )
    if comparison is None:
        return _response(
            tool_name="compare_suite",
            status="fail",
            reason="suite_not_found",
            error_code="suite_not_found",
            suggested_action="Use list_suites to inspect available suites.",
            response_detail=normalized_response_detail,
        )

    error_code = comparison.get("error")
    if error_code == "reference_run_not_found":
        return _response(
            tool_name="compare_suite",
            status="fail",
            reason="reference_run_not_found",
            error_code="reference_run_not_found",
            result=comparison,
            suggested_action="Use get_run or get_suite to inspect available run IDs.",
            response_detail=normalized_response_detail,
        )
    if error_code == "reference_run_wrong_suite":
        return _response(
            tool_name="compare_suite",
            status="fail",
            reason="reference_run_wrong_suite",
            error_code="reference_run_wrong_suite",
            result=comparison,
            suggested_action="Choose a reference run from the requested suite.",
            response_detail=normalized_response_detail,
        )
    if error_code == "strict_requires_reference_run":
        return _response(
            tool_name="compare_suite",
            status="fail",
            reason="strict_requires_reference_run",
            error_code="strict_requires_reference_run",
            result=comparison,
            suggested_action="Pass a suite name and reference run ID before using strict_keys.",
            response_detail=normalized_response_detail,
        )
    if error_code == "strict_keys_not_found":
        return _response(
            tool_name="compare_suite",
            status="fail",
            reason="strict_keys_not_found",
            error_code="strict_keys_not_found",
            result=comparison,
            suggested_action="Use keys that exist on the reference run configuration.",
            response_detail=normalized_response_detail,
        )
    if error_code == "reference_run_does_not_match_config_filter":
        return _response(
            tool_name="compare_suite",
            status="fail",
            reason="reference_run_does_not_match_config_filter",
            error_code="reference_run_does_not_match_config_filter",
            result=comparison,
            suggested_action="Choose a reference run that matches the config filter or relax the filter.",
            response_detail=normalized_response_detail,
        )
    if error_code == "baseline_not_found":
        return _response(
            tool_name="compare_suite",
            status="fail",
            reason="baseline_not_found",
            error_code="baseline_not_found",
            result=comparison,
            suggested_action="Call pin_baseline before using use_pinned_baseline.",
            response_detail=normalized_response_detail,
        )

    comparison["comparison_mode"] = "suite"
    _capped_rows(comparison, "runs", limit)
    return _comparison_response(
        tool_name="compare_suite",
        comparison=comparison,
        confidence_level=confidence_level,
        mode="suite",
        response_detail=normalized_response_detail,
    )
