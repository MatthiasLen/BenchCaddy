from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .models import Base

_ENGINES: dict[Path, Engine] = {}
_INITIALIZED_DATABASES: set[Path] = set()
_ENGINE_LOCK = threading.Lock()
_INITIALIZATION_LOCKS: dict[Path, threading.Lock] = {}


def get_database_path(database_path: str | Path | None = None) -> Path:
    if database_path is None:
        return (Path.cwd() / "benchcaddy.db").resolve()
    return Path(database_path).resolve()


def get_engine(database_path: str | Path | None = None) -> Engine:
    path = get_database_path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ENGINE_LOCK:
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
            Base.metadata.create_all(engine)
            _migrate_legacy_schema(engine)
            _INITIALIZED_DATABASES.add(path)
    return engine


@contextmanager
def db_session(database_path: str | Path | None = None):
    engine = initialize_database(database_path)
    with Session(engine, expire_on_commit=False) as session:
        yield session


def _get_initialization_lock(path: Path) -> threading.Lock:
    with _ENGINE_LOCK:
        lock = _INITIALIZATION_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _INITIALIZATION_LOCKS[path] = lock
        return lock


def _migrate_legacy_schema(engine: Engine) -> None:
    inspector = inspect(engine)
    if not inspector.has_table("benchmark_runs"):
        return
    columns = {column["name"] for column in inspector.get_columns("benchmark_runs")}
    statements: list[str] = []
    if "sweep_execution_id" not in columns:
        statements.append("ALTER TABLE benchmark_runs ADD COLUMN sweep_execution_id INTEGER")
    if "run_index" not in columns:
        statements.append("ALTER TABLE benchmark_runs ADD COLUMN run_index INTEGER")
    if "target_return_value" not in columns:
        statements.append("ALTER TABLE benchmark_runs ADD COLUMN target_return_value JSON")

    if inspector.has_table("environment_info"):
        environment_columns = {column["name"] for column in inspector.get_columns("environment_info")}
        if "total_memory_bytes" not in environment_columns:
            statements.append("ALTER TABLE environment_info ADD COLUMN total_memory_bytes INTEGER")

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
