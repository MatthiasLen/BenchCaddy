"""Internal backend binding for persistence services.

The read, write, and analysis modules depend on this module instead of the
concrete SQLAlchemy package so a future backend swap has one internal seam.
"""

from ._sqlalchemy.models import BenchmarkRun, BenchmarkSuite, BenchmarkSuiteBaseline, BenchmarkSweepExecution, EnvironmentInfo
from ._sqlalchemy.session import db_session
from ._sqlalchemy.store import (
    _collect_observation_labels,
    _get_or_create_suite,
    _get_suite,
    _get_suite_baseline_record,
    _list_all_runs_latest_first,
    _list_all_suites,
    _list_suite_runs_created_desc,
    _list_suite_runs_latest_first,
    _list_suite_runs_oldest_first,
    _resolve_run,
    _resolve_suite_baseline_run,
)

__all__ = [
    "BenchmarkRun",
    "BenchmarkSuite",
    "BenchmarkSuiteBaseline",
    "BenchmarkSweepExecution",
    "EnvironmentInfo",
    "db_session",
    "_collect_observation_labels",
    "_get_or_create_suite",
    "_get_suite",
    "_get_suite_baseline_record",
    "_list_all_runs_latest_first",
    "_list_all_suites",
    "_list_suite_runs_created_desc",
    "_list_suite_runs_latest_first",
    "_list_suite_runs_oldest_first",
    "_resolve_run",
    "_resolve_suite_baseline_run",
]