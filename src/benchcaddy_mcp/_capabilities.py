from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from benchcaddy.db import get_database_path

SERVER_TRANSPORT = "stdio"
RESPONSE_ENVELOPE_FIELDS = [
    "schema_version",
    "command",
    "status",
    "reason",
    "error_code",
    "suggested_action",
    "confidence",
    "response_detail",
    "summary",
    "result",
]
TOOL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "server_status",
        "description": "Check whether the BenchCaddy MCP server is reachable and how it resolves the database path.",
        "category": "diagnostics",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
    },
    {
        "name": "get_capabilities",
        "description": "Inspect BenchCaddy MCP version, response contract, and available tools.",
        "category": "diagnostics",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
    },
    {
        "name": "list_suites",
        "description": "List the benchmark suites available in a BenchCaddy database.",
        "category": "inspection",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
    },
    {
        "name": "get_suite",
        "description": "Inspect one benchmark suite, including recent runs and baseline context.",
        "category": "inspection",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
    },
    {
        "name": "get_run",
        "description": "Inspect one recorded benchmark run in detail.",
        "category": "inspection",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
    },
    {
        "name": "compare_suite",
        "description": "Compare runs within a suite against a baseline, best run, or reference run.",
        "category": "analysis",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
    },
    {
        "name": "compare_runs",
        "description": "Compare two specific benchmark runs head-to-head.",
        "category": "analysis",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
    },
    {
        "name": "trend_suite",
        "description": "Inspect trend and drift information for a benchmark suite.",
        "category": "analysis",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
    },
    {
        "name": "get_baseline_history",
        "description": "Inspect the pinned baseline history for a benchmark suite.",
        "category": "baseline",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
    },
    {
        "name": "pin_baseline",
        "description": "Update the pinned baseline for a benchmark suite.",
        "category": "baseline",
        "mutates_state": True,
        "supports_database_path": True,
        "supports_response_detail": True,
    },
)


def _server_version() -> str | None:
    try:
        return version("benchcaddy")
    except PackageNotFoundError:
        return None


def _tool_names() -> list[str]:
    return [tool["name"] for tool in TOOL_SPECS]


def _database_diagnostics(database_path: str | None) -> dict[str, Any]:
    resolved_path = str(get_database_path(database_path))
    return {
        "requested_path": database_path,
        "resolved_path": resolved_path,
        "exists": Path(resolved_path).exists(),
        "uses_default_path": database_path is None,
    }
