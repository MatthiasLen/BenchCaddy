"""Public persistence surface for BenchCaddy.

Import persistence workflows from here. Statistical analysis helpers,
return-value helpers, and SQLAlchemy internals live in their own modules.
"""

from ._sqlalchemy.session import get_database_path
from .analysis import compare_runs, compare_suite_runs, get_suite_trend
from .read import (
    get_all_run_count,
    get_all_run_details,
    get_run_details,
    get_selected_run_details,
    get_suite_details,
    list_suite_summaries,
)
from .write import benchmark_run_payload, create_sweep_execution, record_benchmark_run, set_suite_baseline

__all__ = [
    "benchmark_run_payload",
    "compare_runs",
    "compare_suite_runs",
    "create_sweep_execution",
    "get_all_run_count",
    "get_all_run_details",
    "get_database_path",
    "get_run_details",
    "get_selected_run_details",
    "get_suite_details",
    "get_suite_trend",
    "list_suite_summaries",
    "record_benchmark_run",
    "set_suite_baseline",
]