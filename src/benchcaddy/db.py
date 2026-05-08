from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class BenchmarkSuite(Base):
    __tablename__ = "benchmark_suites"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    target_name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())

    runs: Mapped[list["BenchmarkRun"]] = relationship(back_populates="suite")


class EnvironmentInfo(Base):
    __tablename__ = "environment_info"

    id: Mapped[int] = mapped_column(primary_key=True)
    python_version: Mapped[str] = mapped_column(String(64))
    operating_system: Mapped[str] = mapped_column(String(255))
    cpu_model: Mapped[str] = mapped_column(String(255))
    gpu_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    git_branch: Mapped[str | None] = mapped_column(String(255), nullable=True)
    git_commit_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    git_dirty: Mapped[bool] = mapped_column(default=False)
    process_state: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())

    runs: Mapped[list["BenchmarkRun"]] = relationship(back_populates="environment")


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    suite_id: Mapped[int] = mapped_column(ForeignKey("benchmark_suites.id"), index=True)
    environment_id: Mapped[int] = mapped_column(ForeignKey("environment_info.id"))
    configuration: Mapped[dict[str, Any]] = mapped_column(JSON)
    samples: Mapped[list[float]] = mapped_column(JSON)
    observations: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    median_seconds: Mapped[float] = mapped_column(Float)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())

    suite: Mapped[BenchmarkSuite] = relationship(back_populates="runs")
    environment: Mapped[EnvironmentInfo] = relationship(back_populates="runs")


def get_database_path(database_path: str | Path | None = None) -> Path:
    if database_path is None:
        return (Path.cwd() / "benchcaddy.db").resolve()
    return Path(database_path).resolve()


def get_engine(database_path: str | Path | None = None):
    path = get_database_path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", future=True)


def get_session_factory(database_path: str | Path | None = None) -> sessionmaker[Session]:
    engine = get_engine(database_path)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def record_benchmark_run(
    *,
    suite_name: str,
    target_name: str,
    configuration: dict[str, Any],
    samples: list[float],
    observations: list[dict[str, Any]],
    median_seconds: float,
    environment: dict[str, Any],
    database_path: str | Path | None = None,
) -> BenchmarkRun:
    session_factory = get_session_factory(database_path)

    with session_factory() as session:
        suite = session.scalar(
            select(BenchmarkSuite).where(BenchmarkSuite.name == suite_name)
        )
        if suite is None:
            suite = BenchmarkSuite(name=suite_name, target_name=target_name)
            session.add(suite)
            session.flush()

        environment_info = EnvironmentInfo(
            python_version=environment["python_version"],
            operating_system=environment["operating_system"],
            cpu_model=environment["cpu_model"],
            gpu_model=environment.get("gpu_model"),
            git_branch=environment["git"]["branch"],
            git_commit_hash=environment["git"]["commit_hash"],
            git_dirty=environment["git"]["dirty"],
            process_state=environment["process"],
        )
        session.add(environment_info)
        session.flush()

        benchmark_run = BenchmarkRun(
            suite_id=suite.id,
            environment_id=environment_info.id,
            configuration=configuration,
            samples=samples,
            observations=observations,
            median_seconds=median_seconds,
        )
        session.add(benchmark_run)
        session.commit()
        session.refresh(benchmark_run)
        return benchmark_run


def list_suite_summaries(database_path: str | Path | None = None) -> list[dict[str, Any]]:
    session_factory = get_session_factory(database_path)
    with session_factory() as session:
        rows = session.execute(
            select(
                BenchmarkSuite.name,
                BenchmarkSuite.target_name,
                func.count(BenchmarkRun.id),
                func.max(BenchmarkRun.created_at),
            )
            .join(BenchmarkRun, BenchmarkRun.suite_id == BenchmarkSuite.id)
            .group_by(BenchmarkSuite.id)
            .order_by(BenchmarkSuite.name)
        ).all()

    return [
        {
            "suite_name": name,
            "target_name": target_name,
            "run_count": run_count,
            "last_run_at": last_run_at,
        }
        for name, target_name, run_count, last_run_at in rows
    ]


def get_suite_details(
    suite_name: str,
    database_path: str | Path | None = None,
) -> dict[str, Any] | None:
    session_factory = get_session_factory(database_path)
    with session_factory() as session:
        suite = session.scalar(
            select(BenchmarkSuite).where(BenchmarkSuite.name == suite_name)
        )
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

    return {
        "suite_name": suite.name,
        "target_name": suite.target_name,
        "runs": [
            {
                "id": run.id,
                "configuration": run.configuration,
                "samples": run.samples,
                "observations": run.observations,
                "median_seconds": run.median_seconds,
                "created_at": run.created_at,
            }
            for run in runs
        ],
        "environment": None
        if environment is None
        else {
            "python_version": environment.python_version,
            "operating_system": environment.operating_system,
            "cpu_model": environment.cpu_model,
            "gpu_model": environment.gpu_model,
            "git_branch": environment.git_branch,
            "git_commit_hash": environment.git_commit_hash,
            "git_dirty": environment.git_dirty,
            "process_state": environment.process_state,
        },
    }
