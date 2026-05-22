from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from benchcaddy.db import get_database_path
from benchcaddy.presentation import utc_timestamp
from benchcaddy.stats import AnalysisOptions

from ._summary import ResponseSummaryBuilder

JSON_SCHEMA_VERSION = "2.0"
DEFAULT_LIMIT = 20
ResponseDetail = Literal["summary", "full"]
DEFAULT_RESPONSE_DETAIL: ResponseDetail = "summary"
ALLOWED_RESPONSE_DETAILS: tuple[ResponseDetail, ResponseDetail] = ("summary", "full")


def _analysis_options(
    *,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 0,
    noise_cv_threshold: float = 0.05,
    noise_ci_ratio_threshold: float = 0.10,
    outlier_z_threshold: float = 3.5,
    significance_level: float = 0.05,
    regression_threshold_percent: float = 5.0,
    drift_window_size: int = 5,
) -> AnalysisOptions:
    return AnalysisOptions(
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        noise_cv_threshold=noise_cv_threshold,
        noise_ci_ratio_threshold=noise_ci_ratio_threshold,
        outlier_z_threshold=outlier_z_threshold,
        significance_level=significance_level,
        regression_threshold_percent=regression_threshold_percent,
        drift_window_size=drift_window_size,
    )


def _confidence_label(confidence_level: float | None) -> str | None:
    if confidence_level is None:
        return None
    if confidence_level >= 0.95:
        return "high"
    if confidence_level >= 0.8:
        return "medium"
    return "low"


def _normalized_response_detail(response_detail: str) -> ResponseDetail:
    if response_detail in ALLOWED_RESPONSE_DETAILS:
        return response_detail
    raise ValueError(f"'{response_detail}' is not a valid response detail.")


def _response(
    *,
    tool_name: str,
    status: str,
    reason: str,
    summary: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    error_code: str | None = None,
    suggested_action: str | None = None,
    confidence: str | None = None,
    response_detail: ResponseDetail = DEFAULT_RESPONSE_DETAIL,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": JSON_SCHEMA_VERSION,
        "command": tool_name,
        "status": status,
        "reason": reason,
        "error_code": error_code,
        "suggested_action": suggested_action,
        "confidence": confidence,
        "response_detail": response_detail,
    }
    if summary is None and result is not None:
        summary = _SUMMARY_BUILDER.build(tool_name, result)
    if summary is not None:
        payload["summary"] = _normalize_datetimes(summary)
    if response_detail == "full" and result is not None:
        payload["result"] = _normalize_datetimes(result)
    return payload


_SUMMARY_BUILDER = ResponseSummaryBuilder()


def _normalize_datetimes(value: Any) -> Any:
    if isinstance(value, datetime):
        return utc_timestamp(value).isoformat(timespec="seconds").replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: _normalize_datetimes(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_normalize_datetimes(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_datetimes(item) for item in value]
    return value


def _run_id(value: int | str | tuple[int, int]) -> int | tuple[int, int]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, int):
        return value
    if "." in value:
        left, dot, right = value.partition(".")
        if dot and left.isdigit() and right.isdigit():
            return (int(left), int(right))
    if value.isdigit():
        return int(value)
    raise ValueError(f"'{value}' is not a valid run ID.")


def _capped_rows(result: dict[str, Any], key: str, limit: int | None) -> dict[str, Any]:
    rows = list(result.get(key, []))
    if limit is None:
        result["truncated"] = False
        result["total_run_count"] = len(rows)
        return result
    result[key] = rows[:limit]
    result["truncated"] = len(rows) > limit
    result["total_run_count"] = len(rows)
    result["limit"] = limit
    return result


def _invalid_response_detail_response(*, tool_name: str, response_detail: object) -> dict[str, Any]:
    return _response(
        tool_name=tool_name,
        status="fail",
        reason="invalid_response_detail",
        error_code="invalid_response_detail",
        suggested_action="Use response_detail='summary' or response_detail='full'.",
        result={
            "message": f"'{response_detail}' is not a valid response detail.",
            "requested_response_detail": response_detail,
            "allowed_values": list(ALLOWED_RESPONSE_DETAILS),
        },
    )


def _invalid_run_id_response(
    *,
    tool_name: str,
    run_id: object,
    suggested_action: str,
    response_detail: ResponseDetail = "full",
) -> dict[str, Any]:
    return _response(
        tool_name=tool_name,
        status="fail",
        reason="invalid_run_id",
        error_code="invalid_run_id",
        suggested_action=suggested_action,
        result={"message": f"'{run_id}' is not a valid run ID.", "requested_run_id": run_id},
        response_detail=response_detail,
    )


def _comparison_response(
    *,
    tool_name: str,
    comparison: dict[str, Any],
    confidence_level: float,
    mode: str,
    response_detail: ResponseDetail = "full",
) -> dict[str, Any]:
    confidence = _confidence_label(confidence_level)
    if mode == "direct":
        comparison_analysis = comparison.get("comparison_analysis") or {}
        status = "pass"
        reason = "comparison_complete"
        suggested_action = "Use the result payload to inspect classifications and candidate deltas."
        if comparison_analysis.get("regression_detected"):
            status = "fail"
            reason = str(comparison_analysis.get("classification") or "regression_detected")
            suggested_action = "Increase samples or inspect the candidate run before accepting the regression."
        elif comparison_analysis.get("classification") == "noisy" or comparison_analysis.get("warnings"):
            status = "inconclusive"
            reason = "noisy_samples"
            suggested_action = "Increase samples or reduce environmental noise, then rerun compare_runs."
        return _response(
            tool_name=tool_name,
            status=status,
            reason=reason,
            result=comparison,
            suggested_action=suggested_action,
            confidence=confidence,
            response_detail=response_detail,
        )

    analyses = [run.get("comparison_analysis") or {} for run in comparison.get("runs", [])]
    if not comparison.get("runs"):
        return _response(
            tool_name=tool_name,
            status="inconclusive",
            reason="no_runs_matched_scope",
            result=comparison,
            suggested_action="Relax the comparison scope or record more runs before treating the result as decisive.",
            confidence=None,
            response_detail=response_detail,
        )
    if any(analysis.get("regression_detected") for analysis in analyses):
        return _response(
            tool_name=tool_name,
            status="fail",
            reason="regression_detected",
            result=comparison,
            suggested_action="Inspect regressing runs in result.runs before accepting the change.",
            confidence=confidence,
            response_detail=response_detail,
        )
    if any(analysis.get("classification") == "noisy" or analysis.get("warnings") for analysis in analyses):
        return _response(
            tool_name=tool_name,
            status="inconclusive",
            reason="noisy_samples",
            result=comparison,
            suggested_action="Increase samples or narrow the comparison scope, then rerun compare_suite.",
            confidence=confidence,
            response_detail=response_detail,
        )
    return _response(
        tool_name=tool_name,
        status="pass",
        reason="comparison_complete",
        result=comparison,
        suggested_action="Use the result payload to inspect classifications and candidate deltas.",
        confidence=confidence,
        response_detail=response_detail,
    )


def _resolved_database_path(database_path: str | None) -> str:
    return str(get_database_path(database_path))
