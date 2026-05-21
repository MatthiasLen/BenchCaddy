from __future__ import annotations

from fastmcp import FastMCP

SERVER_NAME = "BenchCaddy MCP"
SERVER_INSTRUCTIONS = "Inspect benchmark suites, compare runs, review trends, manage suite baselines, and diagnose MCP connectivity."

app = FastMCP(
    name=SERVER_NAME,
    instructions=SERVER_INSTRUCTIONS,
)
