"""Public persistence surface for BenchCaddy.

Prefer the read, write, and analysis functions re-exported here for normal
callers. The SQLAlchemy-specific engine, session, and model exports remain
available for backward compatibility and live in ``benchcaddy.db.compat``.
"""

from ..observability import summarize_observations
from ..return_values import StoredReturnValue, normalize_return_value, return_relative_error
from ..stats import AnalysisOptions, analyze_samples, compare_sample_sets
from .analysis import compare_runs, compare_suite_runs, get_suite_trend
from .compat import (
    Base,
    BenchmarkRun,
    BenchmarkSuite,
    BenchmarkSuiteBaseline,
    BenchmarkSweepExecution,
    EnvironmentInfo,
    db_session,
    get_database_path,
    get_engine,
    initialize_database,
)
from .read import (
    get_all_run_details,
    get_run_details,
    get_selected_run_details,
    get_suite_details,
    list_suite_summaries,
)
from .write import benchmark_run_payload, create_sweep_execution, record_benchmark_run, set_suite_baseline

__all__ = [
    "AnalysisOptions",
    "StoredReturnValue",
    "analyze_samples",
    "benchmark_run_payload",
    "compare_runs",
    "compare_sample_sets",
    "compare_suite_runs",
    "create_sweep_execution",
    "get_all_run_details",
    "get_database_path",
    "get_run_details",
    "get_selected_run_details",
    "get_suite_details",
    "get_suite_trend",
    "list_suite_summaries",
    "normalize_return_value",
    "record_benchmark_run",
    "return_relative_error",
    "set_suite_baseline",
    "summarize_observations",
    "Base",
    "BenchmarkRun",
    "BenchmarkSuite",
    "BenchmarkSuiteBaseline",
    "BenchmarkSweepExecution",
    "EnvironmentInfo",
    "db_session",
    "get_engine",
    "initialize_database",
]