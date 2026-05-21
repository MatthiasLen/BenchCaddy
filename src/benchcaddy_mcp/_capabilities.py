from __future__ import annotations

from pathlib import Path
from typing import Any

from benchcaddy.db import get_database_path

from ._app import SERVER_VERSION

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
        "when_to_use": "Use this first when you want a quick health check, need to confirm the database path, or suspect the MCP server is not ready.",
        "example_queries": [
            "is the BenchCaddy MCP server reachable",
            "what database path is the server using",
            "check MCP server status",
        ],
    },
    {
        "name": "get_capabilities",
        "description": "Inspect BenchCaddy MCP version, response contract, and available tools.",
        "category": "diagnostics",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
        "when_to_use": "Use this when you need the tool inventory, response contract, server version, or examples of which tool matches a user request.",
        "example_queries": [
            "what tools does BenchCaddy expose",
            "show the MCP contract",
            "which tool should I use to compare a suite against run 4.1",
        ],
    },
    {
        "name": "list_suites",
        "description": "List the benchmark suites available in a BenchCaddy database.",
        "category": "inspection",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
        "when_to_use": "Use this when the user wants to discover available suite names before inspecting, comparing, or trending them.",
        "example_queries": [
            "what suites are in the database",
            "list benchmark suites",
            "show available suite names",
        ],
    },
    {
        "name": "get_suite",
        "description": "Inspect one benchmark suite, including recent runs and baseline context.",
        "category": "inspection",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
        "when_to_use": "Use this when the user wants to inspect one suite, see recent runs, find run IDs, or understand baseline context before deeper analysis.",
        "example_queries": [
            "show details for suite nonlinear-transform",
            "what runs are in this suite",
            "inspect the nonlinear-transform suite",
        ],
    },
    {
        "name": "get_run",
        "description": "Inspect one recorded benchmark run in detail.",
        "category": "inspection",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
        "when_to_use": "Use this when the user names one run ID and wants its configuration, environment, or detailed measurements before comparing it.",
        "example_queries": [
            "show run 4.1",
            "inspect benchmark run 3.2",
            "what configuration did run 4.1 use",
        ],
    },
    {
        "name": "compare_suite",
        "description": (
            "Compare an entire suite against a chosen reference run, best run, "
            "or pinned baseline. Use this when the user asks to compare a "
            "suite against run 4.1, a baseline, or the best run."
        ),
        "category": "analysis",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
        "when_to_use": "Use this when the user wants a suite-wide comparison against one reference run, the pinned baseline, or the best run in the suite.",
        "example_queries": [
            "compare nonlinear-transform against run 4.1",
            "compare this suite to its baseline",
            "show regressions in a suite",
        ],
    },
    {
        "name": "compare_runs",
        "description": (
            "Compare two specific benchmark runs directly. Use this when the "
            "user asks to compare run 4.1 against run 4.2 or wants a "
            "head-to-head run comparison instead of a suite-wide comparison."
        ),
        "category": "analysis",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
        "when_to_use": "Use this when the user wants a direct run-vs-run comparison rather than a suite-wide comparison.",
        "example_queries": [
            "compare run 4.1 against run 4.2",
            "show the difference between runs 2.1 and 3.1",
            "compare two benchmark runs directly",
        ],
    },
    {
        "name": "trend_suite",
        "description": (
            "Inspect how a benchmark suite or one configuration changes over "
            "time. Use this when the user asks about drift, history, "
            "regressions over time, or long-term trends for a suite."
        ),
        "category": "analysis",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
        "when_to_use": "Use this when the user asks about history, drift, regressions over time, or how one suite configuration is trending.",
        "example_queries": [
            "show the trend for nonlinear-transform",
            "has this suite regressed over time",
            "inspect drift for one suite",
        ],
    },
    {
        "name": "get_baseline_history",
        "description": "Inspect the pinned baseline history for a benchmark suite.",
        "category": "baseline",
        "mutates_state": False,
        "supports_database_path": True,
        "supports_response_detail": True,
        "when_to_use": "Use this when the user wants to see which runs were pinned as baselines over time.",
        "example_queries": [
            "show baseline history for nonlinear-transform",
            "which run is pinned as the baseline",
            "list baseline changes for this suite",
        ],
    },
    {
        "name": "pin_baseline",
        "description": "Update the pinned baseline for a benchmark suite.",
        "category": "baseline",
        "mutates_state": True,
        "supports_database_path": True,
        "supports_response_detail": True,
        "when_to_use": "Use this when the user explicitly wants to pin or change the baseline run for a suite.",
        "example_queries": [
            "pin run 4.2 as the nonlinear-transform baseline",
            "set the suite baseline to run 3.1",
            "update the pinned baseline",
        ],
    },
)


def _server_version() -> str:
    return SERVER_VERSION


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
