from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .models import Base

_ENGINES: dict[Path, Engine] = {}
_INITIALIZED_DATABASES: set[Path] = set()
_ENGINE_LOCK = threading.Lock()
_INITIALIZATION_LOCKS: dict[Path, threading.Lock] = {}


class DatabaseSchemaError(RuntimeError):
    pass


def get_database_path(database_path: str | Path | None = None) -> Path:
    if database_path is None:
        return (Path.cwd() / "benchcaddy.db").resolve()
    return Path(database_path).resolve()


def get_engine(database_path: str | Path | None = None) -> Engine:
    path = get_database_path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ENGINE_LOCK:
        # Reuse one engine per resolved database path so SQLite connection state stays consistent.
        engine = _ENGINES.get(path)
        if engine is None:
            engine = create_engine(f"sqlite:///{path}", future=True)
            _ENGINES[path] = engine
        return engine


def initialize_database(database_path: str | Path | None = None) -> Engine:
    path = get_database_path(database_path)
    engine = get_engine(path)
    with _get_initialization_lock(path):
        if path not in _INITIALIZED_DATABASES:
            # Existing databases are validated before any create_all call so stale files are rejected, not half-upgraded.
            if inspect(engine).get_table_names():
                _validate_schema(engine, path)
            else:
                Base.metadata.create_all(engine)
            _validate_schema(engine, path)
            _INITIALIZED_DATABASES.add(path)
    return engine


@contextmanager
def db_session(database_path: str | Path | None = None):
    engine = initialize_database(database_path)
    # Sessions are provided without an implicit transaction so callers control begin/commit boundaries explicitly.
    with Session(engine, expire_on_commit=False) as session:
        yield session


def _get_initialization_lock(path: Path) -> threading.Lock:
    with _ENGINE_LOCK:
        lock = _INITIALIZATION_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _INITIALIZATION_LOCKS[path] = lock
        return lock


def _validate_schema(engine: Engine, path: Path) -> None:
    inspector = inspect(engine)
    missing_tables: list[str] = []
    missing_columns: dict[str, list[str]] = {}
    missing_indexes: dict[str, list[str]] = {}
    missing_unique_indexes: dict[str, list[str]] = {}

    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            missing_tables.append(table.name)
            continue
        actual_columns = {column["name"] for column in inspector.get_columns(table.name)}
        expected_columns = {column.name for column in table.columns}
        table_missing_columns = sorted(expected_columns - actual_columns)
        if table_missing_columns:
            missing_columns[table.name] = table_missing_columns

    benchmark_run_indexes = inspector.get_indexes("benchmark_runs") if inspector.has_table("benchmark_runs") else []
    # Tuple run ids are part of current read/write semantics, so missing uniqueness is a hard schema mismatch.
    has_unique_sweep_run_index = any(index.get("unique") and index.get("column_names") == ["sweep_execution_id", "run_index"] for index in benchmark_run_indexes)
    if not has_unique_sweep_run_index:
        missing_unique_indexes["benchmark_runs"] = ["sweep_execution_id", "run_index"]

    baseline_event_indexes = inspector.get_indexes("benchmark_suite_baseline_events") if inspector.has_table("benchmark_suite_baseline_events") else []
    has_suite_history_index = any(index.get("column_names") == ["suite_id", "created_at", "id"] for index in baseline_event_indexes)
    if inspector.has_table("benchmark_suite_baseline_events") and not has_suite_history_index:
        missing_indexes["benchmark_suite_baseline_events"] = ["suite_id", "created_at", "id"]

    if not missing_tables and not missing_columns and not missing_indexes and not missing_unique_indexes:
        return

    message_parts: list[str] = []
    if missing_tables:
        message_parts.append(f"missing tables: {', '.join(sorted(missing_tables))}")
    if missing_columns:
        column_parts = [f"{table}({', '.join(columns)})" for table, columns in sorted(missing_columns.items())]
        message_parts.append(f"missing columns: {', '.join(column_parts)}")
    if missing_indexes:
        index_parts = [f"{table}({', '.join(columns)})" for table, columns in sorted(missing_indexes.items())]
        message_parts.append(f"missing indexes: {', '.join(index_parts)}")
    if missing_unique_indexes:
        index_parts = [f"{table}({', '.join(columns)})" for table, columns in sorted(missing_unique_indexes.items())]
        message_parts.append(f"missing unique indexes: {', '.join(index_parts)}")

    details = "; ".join(message_parts)
    raise DatabaseSchemaError(f"Unsupported BenchCaddy database schema at {path}. {details}. Recreate the SQLite database with the current BenchCaddy version.")
