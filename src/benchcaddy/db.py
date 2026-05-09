from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from statistics import fmean
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, String, create_engine, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from sqlalchemy.sql.functions import now
from sqlalchemy.types import JSON

from .observability import summarize_observations

_ENGINES: dict[Path, Engine] = {}
_INITIALIZED_DATABASES: set[Path] = set()


class Base(DeclarativeBase):
    pass


class BenchmarkSuite(Base):
    __tablename__ = "benchmark_suites"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    target_name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=now())

    sweep_executions: Mapped[list["BenchmarkSweepExecution"]] = relationship(back_populates="suite")
    runs: Mapped[list["BenchmarkRun"]] = relationship(back_populates="suite")


class BenchmarkSweepExecution(Base):
    __tablename__ = "benchmark_sweep_executions"

    id: Mapped[int] = mapped_column(primary_key=True)
    suite_id: Mapped[int] = mapped_column(ForeignKey("benchmark_suites.id"), index=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=now())

    suite: Mapped[BenchmarkSuite] = relationship(back_populates="sweep_executions")
    runs: Mapped[list["BenchmarkRun"]] = relationship(back_populates="sweep_execution")


class EnvironmentInfo(Base):
    __tablename__ = "environment_info"

    id: Mapped[int] = mapped_column(primary_key=True)
    python_version: Mapped[str] = mapped_column(String(64))
    operating_system: Mapped[str] = mapped_column(String(255))
    cpu_model: Mapped[str] = mapped_column(String(255))
    gpu_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    git_branch: Mapped[str | None] = mapped_column(String(255), nullable=True)
    git_commit_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    git_dirty: Mapped[bool | None] = mapped_column(nullable=True)
    process_state: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=now())

    runs: Mapped[list["BenchmarkRun"]] = relationship(back_populates="environment")

    @classmethod
    def from_payload(cls, environment: dict[str, Any]) -> "EnvironmentInfo":
        return cls(
            python_version=environment["python_version"],
            operating_system=environment["operating_system"],
            cpu_model=environment["cpu_model"],
            gpu_model=environment.get("gpu_model"),
            git_branch=environment["git"]["branch"],
            git_commit_hash=environment["git"]["commit_hash"],
            git_dirty=environment["git"]["dirty"],
            process_state=environment["process"],
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "operating_system": self.operating_system,
            "cpu_model": self.cpu_model,
            "gpu_model": self.gpu_model,
            "git_branch": self.git_branch,
            "git_commit_hash": self.git_commit_hash,
            "git_dirty": self.git_dirty,
            "process_state": self.process_state,
        }


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    suite_id: Mapped[int] = mapped_column(ForeignKey("benchmark_suites.id"), index=True)
    sweep_execution_id: Mapped[int | None] = mapped_column(ForeignKey("benchmark_sweep_executions.id"), index=True, nullable=True)
    run_index: Mapped[int | None] = mapped_column(nullable=True)
    environment_id: Mapped[int] = mapped_column(ForeignKey("environment_info.id"))
    configuration: Mapped[dict[str, Any]] = mapped_column(JSON)
    samples: Mapped[list[float]] = mapped_column(JSON)
    observations: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    median_seconds: Mapped[float] = mapped_column(Float)
    min_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    std_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=now())

    suite: Mapped[BenchmarkSuite] = relationship(back_populates="runs")
    sweep_execution: Mapped[BenchmarkSweepExecution | None] = relationship(back_populates="runs")
    environment: Mapped[EnvironmentInfo] = relationship(back_populates="runs")

    @property
    def display_id(self) -> str:
        sweep_id = self.sweep_execution_id or self.id
        run_index = self.run_index or 1
        return f"{sweep_id}.{run_index}"

    @property
    def mean_seconds(self) -> float:
        return float(fmean(self.samples)) if self.samples else 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "record_id": self.id,
            "display_id": self.display_id,
            "sweep_id": self.sweep_execution_id or self.id,
            "run_index": self.run_index or 1,
            "configuration": self.configuration,
            "samples": self.samples,
            "observations": self.observations,
            "mean_seconds": self.mean_seconds,
            "median_seconds": self.median_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "std_seconds": self.std_seconds,
            "created_at": self.created_at,
        }

    def to_suite_comparison_row(self, best_median_seconds: float) -> dict[str, Any]:
        return {
            "id": self.id,
            "record_id": self.id,
            "display_id": self.display_id,
            "sweep_id": self.sweep_execution_id or self.id,
            "run_index": self.run_index or 1,
            "configuration": self.configuration,
            "mean_seconds": self.mean_seconds,
            "median_seconds": self.median_seconds,
            "delta_seconds": self.median_seconds - best_median_seconds,
            "slowdown_factor": None if best_median_seconds <= 0 else self.median_seconds / best_median_seconds,
            "sample_count": len(self.samples),
            "created_at": self.created_at,
        }


def _suite_query(suite_name: str):
    return select(BenchmarkSuite).where(BenchmarkSuite.name == suite_name)


def get_database_path(database_path: str | Path | None = None) -> Path:
    if database_path is None:
        return (Path.cwd() / "benchcaddy.db").resolve()
    return Path(database_path).resolve()

def get_engine(database_path: str | Path | None = None) -> Engine:
    path = get_database_path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = _ENGINES.get(path)
    if engine is None:
        engine = create_engine(f"sqlite:///{path}", future=True)
        _ENGINES[path] = engine
    return engine


def initialize_database(database_path: str | Path | None = None) -> Engine:
    path = get_database_path(database_path)
    engine = get_engine(path)
    if path not in _INITIALIZED_DATABASES:
        Base.metadata.create_all(engine)
        _migrate_legacy_schema(engine)
        _INITIALIZED_DATABASES.add(path)
    return engine


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

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


@contextmanager
def db_session(database_path: str | Path | None = None):
    engine = initialize_database(database_path)
    with Session(engine, expire_on_commit=False) as session:
        yield session


def _get_or_create_suite(session: Session, suite_name: str, target_name: str) -> BenchmarkSuite:
    suite = session.scalar(_suite_query(suite_name))
    if suite is None:
        suite = BenchmarkSuite(name=suite_name, target_name=target_name)
        session.add(suite)
        session.flush()
    return suite


def create_sweep_execution(
    *,
    suite_name: str,
    target_name: str,
    database_path: str | Path | None = None,
) -> BenchmarkSweepExecution:
    with db_session(database_path) as session:
        suite = _get_or_create_suite(session, suite_name, target_name)
        sweep_execution = BenchmarkSweepExecution(suite_id=suite.id)
        session.add(sweep_execution)
        session.commit()
        session.refresh(sweep_execution)
        return sweep_execution


def record_benchmark_run(
    *,
    suite_name: str,
    target_name: str,
    configuration: dict[str, Any],
    samples: list[float],
    observations: list[dict[str, Any]],
    median_seconds: float,
    min_seconds: float,
    max_seconds: float,
    std_seconds: float,
    environment: dict[str, Any],
    sweep_execution_id: int | None = None,
    run_index: int | None = None,
    database_path: str | Path | None = None,
) -> BenchmarkRun:
    with db_session(database_path) as session:
        suite = _get_or_create_suite(session, suite_name, target_name)

        if sweep_execution_id is None:
            sweep_execution = BenchmarkSweepExecution(suite_id=suite.id)
            session.add(sweep_execution)
            session.flush()
            sweep_execution_id = sweep_execution.id
        if run_index is None:
            run_index = 1

        environment_info = EnvironmentInfo.from_payload(environment)
        session.add(environment_info)
        session.flush()

        benchmark_run = BenchmarkRun(
            suite_id=suite.id,
            sweep_execution_id=sweep_execution_id,
            run_index=run_index,
            environment_id=environment_info.id,
            configuration=configuration,
            samples=samples,
            observations=observations,
            median_seconds=median_seconds,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            std_seconds=std_seconds,
        )
        session.add(benchmark_run)
        session.commit()
        session.refresh(benchmark_run)
        return benchmark_run


def list_suite_summaries(database_path: str | Path | None = None) -> list[dict[str, Any]]:
    with db_session(database_path) as session:
        suites = session.execute(
            select(BenchmarkSuite).order_by(BenchmarkSuite.name)
        ).scalars().all()
        summaries: list[dict[str, Any]] = []
        for suite in suites:
            runs = session.execute(
                select(BenchmarkRun).where(BenchmarkRun.suite_id == suite.id)
            ).scalars().all()
            if not runs:
                continue
            summaries.append(
                {
                    "suite_name": suite.name,
                    "target_name": suite.target_name,
                    "run_count": len(runs),
                    "last_run_at": max((run.created_at for run in runs), default=None),
                    "observation_labels": _collect_observation_labels([run.observations for run in runs]),
                }
            )

    return summaries


def get_suite_details(
    suite_name: str,
    database_path: str | Path | None = None,
) -> dict[str, Any] | None:
    with db_session(database_path) as session:
        suite = session.scalar(_suite_query(suite_name))
        if suite is None:
            return None

        runs = session.execute(
            select(BenchmarkRun)
            .where(BenchmarkRun.suite_id == suite.id)
            .order_by(BenchmarkRun.created_at.desc())
        ).scalars().all()
        environment = None
        if runs:
            environment = runs[0].environment

    run_payloads = [run.to_payload() for run in runs]
    run_payloads.sort(key=lambda run: (-(run["sweep_id"]), -(run["run_index"]), -run["record_id"]))

    return {
        "suite_name": suite.name,
        "target_name": suite.target_name,
        "runs": run_payloads,
        "environment": None if environment is None else environment.to_payload(),
    }


def compare_suite_runs(
    suite_name: str,
    database_path: str | Path | None = None,
) -> dict[str, Any] | None:
    with db_session(database_path) as session:
        suite = session.scalar(_suite_query(suite_name))
        if suite is None:
            return None

        runs = session.execute(
            select(BenchmarkRun)
            .where(BenchmarkRun.suite_id == suite.id)
            .order_by(BenchmarkRun.sweep_execution_id.desc(), BenchmarkRun.run_index.desc(), BenchmarkRun.id.desc())
        ).scalars().all()

    if not runs:
        return {
            "suite_name": suite.name,
            "target_name": suite.target_name,
            "best_median_seconds": None,
            "runs": [],
        }

    best_median_seconds = min(run.median_seconds for run in runs)
    return {
        "suite_name": suite.name,
        "target_name": suite.target_name,
        "best_median_seconds": best_median_seconds,
        "runs": [run.to_suite_comparison_row(best_median_seconds) for run in runs],
    }


def get_run_details(
    run_id: int | tuple[int, int],
    database_path: str | Path | None = None,
) -> dict[str, Any] | None:
    with db_session(database_path) as session:
        run = _resolve_run(session, run_id)
        if run is None:
            return None

        suite = run.suite
        environment = run.environment

    return {
        "id": run.id,
        "suite_name": suite.name,
        "target_name": suite.target_name,
        "display_id": run.display_id,
        "sweep_id": run.sweep_execution_id or run.id,
        "run_index": run.run_index or 1,
        "configuration": run.configuration,
        "samples": run.samples,
        "observations": run.observations,
        "median_seconds": run.median_seconds,
        "created_at": run.created_at,
        "environment": environment.to_payload(),
    }


def compare_runs(
    baseline_run_id: int | tuple[int, int],
    candidate_run_id: int | tuple[int, int],
    database_path: str | Path | None = None,
) -> dict[str, Any] | None:
    baseline = get_run_details(baseline_run_id, database_path)
    candidate = get_run_details(candidate_run_id, database_path)
    if baseline is None or candidate is None:
        return None

    percent_change = None
    if baseline["median_seconds"] > 0:
        percent_change = (
            (candidate["median_seconds"] - baseline["median_seconds"])
            / baseline["median_seconds"]
        ) * 100.0

    baseline_observations = summarize_observations(baseline["observations"])
    candidate_observations = summarize_observations(candidate["observations"])
    labels = sorted(set(baseline_observations) | set(candidate_observations))

    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta_seconds": candidate["median_seconds"] - baseline["median_seconds"],
        "percent_change": percent_change,
        "observation_rows": [
            {
                "label": label,
                "baseline_seconds": baseline_observations.get(label, (0, 0.0))[1],
                "candidate_seconds": candidate_observations.get(label, (0, 0.0))[1],
                "delta_seconds": candidate_observations.get(label, (0, 0.0))[1] - baseline_observations.get(label, (0, 0.0))[1],
            }
            for label in labels
        ],
    }


def _collect_observation_labels(observation_groups: list[list[dict[str, Any]]] | Any) -> list[str]:
    labels: set[str] = set()
    for observations in observation_groups:
        for sample in observations:
            for record in sample.get("records", []):
                labels.add(str(record.get("label")))
    return sorted(labels)


def _resolve_run(session: Session, run_id: int | tuple[int, int]) -> BenchmarkRun | None:
    if isinstance(run_id, int):
        return session.get(BenchmarkRun, run_id)

    sweep_id, run_index = run_id
    run = session.scalar(
        select(BenchmarkRun).where(
            BenchmarkRun.sweep_execution_id == sweep_id,
            BenchmarkRun.run_index == run_index,
        )
    )
    if run is None and run_index == 1:
        return session.get(BenchmarkRun, sweep_id)
    return run
