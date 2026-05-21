from __future__ import annotations

from typing import Any

from benchcaddy.db import list_suite_summaries

from .._app import app
from .._shared import _resolved_database_path, _response


@app.tool(description="List the benchmark suites available in a BenchCaddy database.")
def list_suites(database_path: str | None = None) -> dict[str, Any]:
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
            suggested_action="Record a benchmark sweep to create suite history.",
            confidence=None,
            result=result,
        )
    return _response(
        tool_name="list_suites",
        status="pass",
        reason="suite_inventory_available",
        suggested_action="Use get_suite or compare_suite for the next step.",
        result=result,
        confidence=None,
    )