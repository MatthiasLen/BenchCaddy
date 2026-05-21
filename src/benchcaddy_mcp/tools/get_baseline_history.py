from __future__ import annotations

from typing import Any

from benchcaddy.db import get_suite_baseline_history as _get_suite_baseline_history

from .._app import app
from .._shared import DEFAULT_LIMIT, _analysis_options, _capped_rows, _response


@app.tool(description="Show the baseline pin history for one suite.")
def get_baseline_history(
    suite_name: str,
    database_path: str | None = None,
    limit: int | None = DEFAULT_LIMIT,
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
    history = _get_suite_baseline_history(
        suite_name,
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
        limit=None,
        include_analysis=include_analysis,
    )
    if history is None:
        return _response(
            tool_name="get_baseline_history",
            status="fail",
            reason="suite_not_found",
            error_code="suite_not_found",
            suggested_action="Use list_suites to inspect available suites.",
        )

    _capped_rows(history, "history", limit)
    history["current_baseline"] = None if not history["history"] else history["history"][0]
    if history["current_baseline"] is None:
        return _response(
            tool_name="get_baseline_history",
            status="inconclusive",
            reason="no_baseline_history",
            result=history,
            suggested_action="Call pin_baseline with a known-good run before relying on baseline history.",
            confidence=None,
        )
    return _response(
        tool_name="get_baseline_history",
        status="pass",
        reason="baseline_available",
        result=history,
        suggested_action="Use compare_suite or trend_suite with this suite.",
        confidence=None,
    )