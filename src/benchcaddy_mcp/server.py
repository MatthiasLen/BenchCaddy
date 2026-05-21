"""Standalone FastMCP server for BenchCaddy benchmark inspection."""

from ._app import app
from .tools import compare_runs, compare_suite, get_baseline_history, get_run, get_suite, list_suites, pin_baseline, trend_suite

__all__ = [
    "app",
    "compare_runs",
    "compare_suite",
    "get_baseline_history",
    "get_run",
    "get_suite",
    "list_suites",
    "main",
    "pin_baseline",
    "trend_suite",
]


def main() -> None:
    app.run()
