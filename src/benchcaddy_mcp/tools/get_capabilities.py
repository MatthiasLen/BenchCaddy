from __future__ import annotations

from typing import Any

from .._app import SERVER_NAME, app
from .._capabilities import RESPONSE_ENVELOPE_FIELDS, SERVER_TRANSPORT, TOOL_SPECS, _database_diagnostics, _server_version, _tool_names
from .._shared import (
    ALLOWED_RESPONSE_DETAILS,
    DEFAULT_RESPONSE_DETAIL,
    JSON_SCHEMA_VERSION,
    ResponseDetail,
    _invalid_response_detail_response,
    _normalized_response_detail,
    _response,
)


@app.tool(description="Inspect BenchCaddy MCP version, response contract, and available tools.")
def get_capabilities(database_path: str | None = None, response_detail: ResponseDetail = DEFAULT_RESPONSE_DETAIL) -> dict[str, Any]:
    try:
        normalized_response_detail = _normalized_response_detail(response_detail)
    except ValueError:
        return _invalid_response_detail_response(tool_name="get_capabilities", response_detail=response_detail)

    database = _database_diagnostics(database_path)
    result = {
        "server_name": SERVER_NAME,
        "server_version": _server_version(),
        "schema_version": JSON_SCHEMA_VERSION,
        "transport": SERVER_TRANSPORT,
        "default_response_detail": DEFAULT_RESPONSE_DETAIL,
        "allowed_response_details": list(ALLOWED_RESPONSE_DETAILS),
        "response_envelope_fields": RESPONSE_ENVELOPE_FIELDS,
        "tool_argument_conventions": {
            "database_path": "Optional path to a BenchCaddy database file. Defaults to ./benchcaddy.db.",
            "response_detail": "Optional response detail mode. Use 'summary' by default or 'full' for nested payloads.",
        },
        "tool_count": len(TOOL_SPECS),
        "tools": list(TOOL_SPECS),
        "database": database,
    }
    return _response(
        tool_name="get_capabilities",
        status="pass",
        reason="capabilities_available",
        summary={
            "server_name": result["server_name"],
            "server_version": result["server_version"],
            "schema_version": result["schema_version"],
            "default_response_detail": result["default_response_detail"],
            "allowed_response_details": result["allowed_response_details"],
            "tool_count": result["tool_count"],
            "tool_names": _tool_names(),
            "database": result["database"],
        },
        result=result,
        suggested_action="Use the per-tool when_to_use and example_queries fields to choose the next tool, then call server_status or list_suites if you still need orientation.",
        confidence="high",
        response_detail=normalized_response_detail,
    )
