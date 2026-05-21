from __future__ import annotations

from typing import Any

from benchcaddy.db import get_suite_details

from .._app import app
from .._shared import DEFAULT_LIMIT, _analysis_options, _capped_rows, _resolved_database_path, _response


@app.tool(description="Inspect one benchmark suite and return the newest runs, environment, and baseline context.")
def get_suite(
    suite_name: str,
    database_path: str | None = None,
    limit: int | None = DEFAULT_LIMIT,
    config_filter: dict[str, Any] | None = None,
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
    resolved_database_path = _resolved_database_path(database_path)
    suite_details = get_suite_details(
        suite_name,
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
        limit=None,
        include_analysis=include_analysis,
        config_filter=config_filter,
    )
    if suite_details is None:
        return _response(
            tool_name="get_suite",
            status="fail",
            reason="suite_not_found",
            error_code="suite_not_found",
            suggested_action="Use list_suites to inspect available suites.",
        )

    result = {
        "mode": "suite",
        "database_path": resolved_database_path,
        **suite_details,
    }
    _capped_rows(result, "runs", limit)
    if result["runs"]:
        return _response(
            tool_name="get_suite",
            status="pass",
            reason="suite_details_available",
            result=result,
            suggested_action="Use compare_suite or trend_suite on this suite.",
            confidence=None,
        )
    return _response(
        tool_name="get_suite",
        status="inconclusive",
        reason="no_runs_matched_scope" if config_filter else "suite_has_no_runs",
        result=result,
        suggested_action=("Relax the filter or record more runs for this suite." if config_filter else "Record new runs for this suite before comparing or trending it."),
        confidence=None,
    )
