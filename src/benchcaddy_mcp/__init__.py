"""Standalone FastMCP server for BenchCaddy benchmark inspection."""

from .server import app, main
from .tools import compare_runs, compare_suite, get_baseline_history, get_capabilities, get_run, get_suite, list_suites, pin_baseline, server_status, trend_suite

__all__ = [
    "app",
    "compare_runs",
    "compare_suite",
    "get_capabilities",
    "get_baseline_history",
    "get_run",
    "get_suite",
    "list_suites",
    "main",
    "pin_baseline",
    "server_status",
    "trend_suite",
]
