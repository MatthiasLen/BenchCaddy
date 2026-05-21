from __future__ import annotations

from typing import Any

from .._app import SERVER_NAME, app
from .._capabilities import SERVER_TRANSPORT, TOOL_SPECS, _database_diagnostics, _server_version, _tool_names
from .._shared import (
    ALLOWED_RESPONSE_DETAILS,
    DEFAULT_RESPONSE_DETAIL,
    JSON_SCHEMA_VERSION,
    ResponseDetail,
    _invalid_response_detail_response,
    _normalized_response_detail,
    _response,
)


@app.tool(description="Check whether the BenchCaddy MCP server is reachable and how it resolves the database path.")
def server_status(database_path: str | None = None, response_detail: ResponseDetail = DEFAULT_RESPONSE_DETAIL) -> dict[str, Any]:
    try:
        normalized_response_detail = _normalized_response_detail(response_detail)
    except ValueError:
        return _invalid_response_detail_response(tool_name="server_status", response_detail=response_detail)

    database = _database_diagnostics(database_path)
    result = {
        "server_name": SERVER_NAME,
        "server_version": _server_version(),
        "schema_version": JSON_SCHEMA_VERSION,
        "transport": SERVER_TRANSPORT,
        "default_response_detail": DEFAULT_RESPONSE_DETAIL,
        "allowed_response_details": list(ALLOWED_RESPONSE_DETAILS),
        "tool_count": len(TOOL_SPECS),
        "tool_names": _tool_names(),
        "database": database,
    }
    suggested_action = "Call get_capabilities for the full contract or list_suites to inspect benchmark data."
    if not result["database"]["exists"]:
        suggested_action = "Provide database_path or create ./benchcaddy.db before calling data inspection tools."
    return _response(
        tool_name="server_status",
        status="pass",
        reason="server_ready",
        summary={
            "server_name": result["server_name"],
            "server_version": result["server_version"],
            "schema_version": result["schema_version"],
            "default_response_detail": result["default_response_detail"],
            "tool_count": result["tool_count"],
            "tool_names": result["tool_names"],
            "database": result["database"],
        },
        result=result,
        suggested_action=suggested_action,
        confidence="high",
        response_detail=normalized_response_detail,
    )
