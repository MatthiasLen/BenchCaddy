from __future__ import annotations

from typing import Any

from benchcaddy.db import list_suite_summaries

from .._app import app
from .._shared import DEFAULT_RESPONSE_DETAIL, ResponseDetail, _invalid_response_detail_response, _normalized_response_detail, _resolved_database_path, _response


@app.tool(description="List the benchmark suites available in a BenchCaddy database.")
def list_suites(database_path: str | None = None, response_detail: ResponseDetail = DEFAULT_RESPONSE_DETAIL) -> dict[str, Any]:
    try:
        normalized_response_detail = _normalized_response_detail(response_detail)
    except ValueError:
        return _invalid_response_detail_response(tool_name="list_suites", response_detail=response_detail)

    resolved_database_path = _resolved_database_path(database_path)
    suites = list_suite_summaries(resolved_database_path)
    result = {
        "database_path": resolved_database_path,
        "suite_count": len(suites),
        "suites": suites,
    }
    if not suites:
        return _response(
            tool_name="list_suites",
            status="inconclusive",
            reason="no_suites_found",
            suggested_action="Record a benchmark sweep to create suite history, then call list_suites again or use get_capabilities to review the analysis tools.",
            confidence=None,
            result=result,
            response_detail=normalized_response_detail,
        )
    return _response(
        tool_name="list_suites",
        status="pass",
        reason="suite_inventory_available",
        suggested_action="Use get_suite to inspect one suite, compare_suite to compare a suite against a run or baseline, or trend_suite to inspect drift over time.",
        result=result,
        confidence=None,
        response_detail=normalized_response_detail,
    )
