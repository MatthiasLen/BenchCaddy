"""Compatibility exports that still expose SQLAlchemy-specific details.

Normal callers should prefer the read, write, and analysis functions from
``benchcaddy.db``. This module exists to keep older imports working while the
package moves toward a narrower, backend-agnostic public API.
"""

from ._sqlalchemy.models import (
    Base,
    BenchmarkRun,
    BenchmarkSuite,
    BenchmarkSuiteBaseline,
    BenchmarkSweepExecution,
    EnvironmentInfo,
)
from ._sqlalchemy.session import db_session, get_database_path, get_engine, initialize_database

__all__ = [
    "Base",
    "BenchmarkRun",
    "BenchmarkSuite",
    "BenchmarkSuiteBaseline",
    "BenchmarkSweepExecution",
    "EnvironmentInfo",
    "db_session",
    "get_database_path",
    "get_engine",
    "initialize_database",
]