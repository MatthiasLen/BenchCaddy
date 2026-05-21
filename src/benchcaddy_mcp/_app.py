from __future__ import annotations

from fastmcp import FastMCP

SERVER_NAME = "BenchCaddy MCP"
SERVER_INSTRUCTIONS = "Inspect benchmark suites, compare runs, review trends, manage suite baselines, and diagnose MCP connectivity."

app = FastMCP(
    name=SERVER_NAME,
    instructions=SERVER_INSTRUCTIONS,
)


def apply_tool_schema_compatibility_workarounds() -> None:
    provider = getattr(app, "_local_provider", None)
    components = getattr(provider, "_components", None)
    if not isinstance(components, dict):
        return

    # Some external schema consumers cache or partially project these tool schemas.
    # Allowing extra top-level properties keeps optional arguments from being rejected
    # before the request reaches the BenchCaddy server implementation.
    for tool_key in ("tool:compare_runs@", "tool:trend_suite@"):
        tool = components.get(tool_key)
        if tool is None:
            continue
        parameters = dict(tool.parameters)
        parameters["additionalProperties"] = True
        tool.parameters = parameters
