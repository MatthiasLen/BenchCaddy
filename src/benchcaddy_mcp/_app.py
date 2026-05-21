from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fastmcp import FastMCP

SERVER_NAME = "BenchCaddy MCP"
SERVER_INSTRUCTIONS = "Inspect benchmark suites, compare runs, review trends, manage suite baselines, and diagnose MCP connectivity."


def _package_version() -> str:
    try:
        return version("benchcaddy")
    except PackageNotFoundError:
        return "0+unknown"


SERVER_VERSION = _package_version()

app = FastMCP(
    name=SERVER_NAME,
    instructions=SERVER_INSTRUCTIONS,
    version=SERVER_VERSION,
)
