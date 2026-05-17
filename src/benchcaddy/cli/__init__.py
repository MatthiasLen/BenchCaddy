"""Command-line interface package for inspecting BenchCaddy benchmark data.

This package keeps the public ``benchcaddy.cli`` import surface stable while
splitting the implementation into command-oriented modules with narrowly shared
support code.
"""

from __future__ import annotations

from ..isolation import NoiseAnalyzer, collect_environment_state, get_affinity
from . import compare as _compare
from . import environment as _environment  # noqa: F401
from . import listing as _listing  # noqa: F401
from . import show as _show  # noqa: F401
from . import trend as _trend
from ._shared import REGRESSION_EXIT_CODE, app, console

_suite_row_style = _compare._suite_row_style
_trend_row_style = _trend._trend_row_style
_trend_sparkline = _trend._trend_sparkline

main = app

__all__ = [
    "NoiseAnalyzer",
    "REGRESSION_EXIT_CODE",
    "_suite_row_style",
    "_trend_row_style",
    "_trend_sparkline",
    "app",
    "collect_environment_state",
    "console",
    "get_affinity",
    "main",
]