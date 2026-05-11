from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean
from typing import Any

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, String, create_engine, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from sqlalchemy.sql.functions import now
from sqlalchemy.types import JSON

from .observability import summarize_observations
from .return_values import StoredReturnValue, normalize_return_value, return_relative_error
from .stats import AnalysisOptions, analyze_samples, compare_sample_sets

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

    baseline: Mapped[BenchmarkSuiteBaseline | None] = relationship(back_populates="suite")
    sweep_executions: Mapped[list[BenchmarkSweepExecution]] = relationship(back_populates="suite")
    runs: Mapped[list[BenchmarkRun]] = relationship(back_populates="suite")


class BenchmarkSuiteBaseline(Base):
    __tablename__ = "benchmark_suite_baselines"

    id: Mapped[int] = mapped_column(primary_key=True)
    suite_id: Mapped[int] = mapped_column(ForeignKey("benchmark_suites.id"), unique=True, index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("benchmark_runs.id"), index=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=now())

    suite: Mapped[BenchmarkSuite] = relationship(back_populates="baseline")
    run: Mapped[BenchmarkRun] = relationship()


class BenchmarkSweepExecution(Base):
    __tablename__ = "benchmark_sweep_executions"

    id: Mapped[int] = mapped_column(primary_key=True)
    suite_id: Mapped[int] = mapped_column(ForeignKey("benchmark_suites.id"), index=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=now())

    suite: Mapped[BenchmarkSuite] = relationship(back_populates="sweep_executions")
    runs: Mapped[list[BenchmarkRun]] = relationship(back_populates="sweep_execution")


class EnvironmentInfo(Base):
    __tablename__ = "environment_info"

    id: Mapped[int] = mapped_column(primary_key=True)
    python_version: Mapped[str] = mapped_column(String(64))
    operating_system: Mapped[str] = mapped_column(String(255))
    cpu_model: Mapped[str] = mapped_column(String(255))
    total_memory_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    gpu_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    git_branch: Mapped[str | None] = mapped_column(String(255), nullable=True)
    git_commit_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    git_dirty: Mapped[bool | None] = mapped_column(nullable=True)
    process_state: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=now())

    runs: Mapped[list[BenchmarkRun]] = relationship(back_populates="environment")

    @classmethod
    def from_payload(cls, environment: dict[str, Any]) -> EnvironmentInfo:
        git_payload = environment.get("git") or {}
        return cls(
            python_version=environment["python_version"],
            operating_system=environment["operating_system"],
            cpu_model=environment["cpu_model"],
            total_memory_bytes=environment.get("total_memory_bytes"),
            gpu_model=environment.get("gpu_model"),
            git_branch=git_payload.get("branch", environment.get("git_branch")),
            git_commit_hash=git_payload.get("commit_hash", environment.get("git_commit_hash")),
            git_dirty=git_payload.get("dirty", environment.get("git_dirty")),
            process_state=environment.get("process", environment.get("process_state", {})),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "python_version": self.python_version,
            "operating_system": self.operating_system,
            "cpu_model": self.cpu_model,
            "total_memory_bytes": self.total_memory_bytes,
            "gpu_model": self.gpu_model,
            "process": self.process_state,
        }
        if any(value is not None for value in (self.git_branch, self.git_commit_hash, self.git_dirty)):
            payload["git"] = {
                "branch": self.git_branch,
                "commit_hash": self.git_commit_hash,
                "dirty": self.git_dirty,
            }
        return payload


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
    target_return_value: Mapped[StoredReturnValue | None] = mapped_column(JSON, nullable=True)
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

    def analysis_payload(self, analysis_options: AnalysisOptions | None = None) -> dict[str, object]:
        analysis = analyze_samples(self.samples, analysis_options)
        return analysis.to_payload()

    def to_payload(
        self,
        analysis_options: AnalysisOptions | None = None,
        *,
        include_analysis: bool = False,
    ) -> dict[str, Any]:
        analysis = self.analysis_payload(analysis_options) if include_analysis or analysis_options is not None else None
        payload = {
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
            "target_return_value": self.target_return_value,
            "created_at": self.created_at,
        }
        payload.update(
            {
                "analysis": analysis,
                "mad_seconds": None if analysis is None else analysis["mad_seconds"],
                "coefficient_of_variation": None if analysis is None else analysis["coefficient_of_variation"],
                "ci_lower_seconds": None if analysis is None else analysis["ci_lower_seconds"],
                "ci_upper_seconds": None if analysis is None else analysis["ci_upper_seconds"],
                "noise_warnings": [] if analysis is None else list(analysis["warnings"]),
                "is_noisy": False if analysis is None else analysis["is_noisy"],
            }
        )
        return payload

    def to_detail_payload(
        self,
        analysis_options: AnalysisOptions | None = None,
        *,
        include_analysis: bool = False,
    ) -> dict[str, Any]:
        return {
            **self.to_payload(analysis_options, include_analysis=include_analysis),
            "suite_name": self.suite.name,
            "target_name": self.suite.target_name,
            "environment": self.environment.to_payload(),
        }

    def to_suite_comparison_row(
        self,
        reference_median_seconds: float,
        reference_run_target_value: StoredReturnValue | None,
        reference_samples: Sequence[float],
        analysis_options: AnalysisOptions | None = None,
    ) -> dict[str, Any]:
        analysis = self.analysis_payload(analysis_options)
        comparison_analysis = compare_sample_sets(reference_samples, self.samples, analysis_options).to_payload()
        return {
            "id": self.id,
            "record_id": self.id,
            "display_id": self.display_id,
            "sweep_id": self.sweep_execution_id or self.id,
            "run_index": self.run_index or 1,
            "configuration": self.configuration,
            "mean_seconds": self.mean_seconds,
            "median_seconds": self.median_seconds,
            "std_seconds": self.std_seconds,
            "target_return_value": self.target_return_value,
            "target_return_relative_error": return_relative_error(reference_value=reference_run_target_value, candidate_value=self.target_return_value),
            "delta_seconds": self.median_seconds - reference_median_seconds,
            "slowdown_factor": None if reference_median_seconds <= 0 else self.median_seconds / reference_median_seconds,
            "sample_count": len(self.samples),
            "created_at": self.created_at,
            "analysis": analysis,
            "mad_seconds": analysis["mad_seconds"],
            "coefficient_of_variation": analysis["coefficient_of_variation"],
            "ci_lower_seconds": analysis["ci_lower_seconds"],
            "ci_upper_seconds": analysis["ci_upper_seconds"],
            "comparison_analysis": comparison_analysis,
            "status": comparison_analysis["classification"],
            "noise_warnings": list(analysis["warnings"]),
            "is_noisy": analysis["is_noisy"],
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


def _get_suite_baseline_record(session: Session, suite_id: int) -> BenchmarkSuiteBaseline | None:
    return session.scalar(select(BenchmarkSuiteBaseline).where(BenchmarkSuiteBaseline.suite_id == suite_id))


def _resolve_suite_baseline_run(session: Session, suite: BenchmarkSuite) -> BenchmarkRun | None:
    baseline_record = _get_suite_baseline_record(session, suite.id)
    if baseline_record is None:
        return None
    return session.get(BenchmarkRun, baseline_record.run_id)


def _list_suite_runs(session: Session, suite_id: int) -> list[BenchmarkRun]:
    return (
        session.execute(
            select(BenchmarkRun)
            .where(BenchmarkRun.suite_id == suite_id)
            .order_by(BenchmarkRun.sweep_execution_id.desc(), BenchmarkRun.run_index.desc(), BenchmarkRun.id.desc())
        )
        .scalars()
        .all()
    )


def _resolve_suite_reference(
    session: Session,
    suite: BenchmarkSuite,
    reference_run_id: int | tuple[int, int] | None,
    *,
    use_pinned_baseline: bool,
) -> tuple[BenchmarkRun | None, BenchmarkRun | None, int | tuple[int, int] | None, str | None, bool] | dict[str, Any]:
    reference_run = None
    reference_from_pinned = False
    reference_run_suite_name = None
    pinned_baseline_run = _resolve_suite_baseline_run(session, suite)

    if reference_run_id is not None:
        reference_run = _resolve_run(session, reference_run_id)
        if reference_run is not None:
            reference_run_suite_name = reference_run.suite.name
    elif use_pinned_baseline:
        reference_run = pinned_baseline_run
        if reference_run is None:
            return {
                "error": "baseline_not_found",
                "suite_name": suite.name,
            }
        reference_from_pinned = True
        reference_run_id = reference_run.id
        reference_run_suite_name = reference_run.suite.name

    return (
        reference_run,
        pinned_baseline_run,
        reference_run_id,
        reference_run_suite_name,
        reference_from_pinned,
    )


def _empty_suite_comparison_payload(suite: BenchmarkSuite) -> dict[str, Any]:
    return {
        "suite_name": suite.name,
        "target_name": suite.target_name,
        "basis_median_seconds": None,
        "basis_run": None,
        "basis_metric_label": "Best Median (s)",
        "delta_column_label": "Delta vs Best (s)",
        "ratio_column_label": "Slowdown",
        "runs": [],
        "pinned_baseline": None,
    }


def _resolve_suite_basis(
    suite: BenchmarkSuite,
    runs: Sequence[BenchmarkRun],
    reference_run: BenchmarkRun | None,
    reference_run_id: int | tuple[int, int] | None,
    reference_run_suite_name: str | None,
) -> tuple[BenchmarkRun, str, str, str] | dict[str, Any]:
    if reference_run_id is None:
        return (
            min(runs, key=lambda run: (run.median_seconds, run.id)),
            "Best Median (s)",
            "Delta vs Best (s)",
            "Slowdown",
        )
    if reference_run is None:
        return {
            "error": "reference_run_not_found",
            "suite_name": suite.name,
        }
    if reference_run.suite_id != suite.id:
        return {
            "error": "reference_run_wrong_suite",
            "suite_name": suite.name,
            "reference_run_display_id": reference_run.display_id,
            "reference_run_record_id": reference_run.id,
            "reference_run_suite_name": reference_run_suite_name,
        }
    return (
        reference_run,
        "Reference Median (s)",
        "Delta vs Reference (s)",
        "Relative",
    )


def _apply_strict_comparison_filter(
    suite: BenchmarkSuite,
    runs: Sequence[BenchmarkRun],
    basis_run: BenchmarkRun,
    strict_keys: Sequence[str],
    reference_run_id: int | tuple[int, int] | None,
) -> tuple[list[BenchmarkRun], tuple[str, ...], dict[str, Any] | None] | dict[str, Any]:
    normalized_keys = tuple(dict.fromkeys(strict_keys))
    if not normalized_keys:
        return list(runs), normalized_keys, None
    if reference_run_id is None:
        return {
            "error": "strict_requires_reference_run",
            "suite_name": suite.name,
        }
    missing_keys = [key for key in normalized_keys if key not in basis_run.configuration]
    if missing_keys:
        return {
            "error": "strict_keys_not_found",
            "suite_name": suite.name,
            "strict_keys": list(normalized_keys),
            "missing_strict_keys": missing_keys,
            "reference_run_display_id": basis_run.display_id,
        }
    strict_config = {key: basis_run.configuration[key] for key in normalized_keys}
    filtered_runs = [run for run in runs if all(run.configuration.get(key) == value for key, value in strict_config.items())]
    return filtered_runs, normalized_keys, strict_config


def _suite_comparison_payload(
    *,
    suite: BenchmarkSuite,
    runs: Sequence[BenchmarkRun],
    basis_run: BenchmarkRun,
    basis_metric_label: str,
    delta_column_label: str,
    ratio_column_label: str,
    strict_keys: Sequence[str],
    strict_config: dict[str, Any] | None,
    reference_from_pinned: bool,
    reference_run_id: int | tuple[int, int] | None,
    pinned_baseline_run: BenchmarkRun | None,
    analysis_options: AnalysisOptions | None,
) -> dict[str, Any]:
    basis_median_seconds = basis_run.median_seconds
    return {
        "suite_name": suite.name,
        "target_name": suite.target_name,
        "basis_median_seconds": basis_median_seconds,
        "basis_run": basis_run.to_suite_comparison_row(
            basis_median_seconds,
            basis_run.target_return_value,
            basis_run.samples,
            analysis_options,
        ),
        "basis_metric_label": basis_metric_label,
        "delta_column_label": delta_column_label,
        "ratio_column_label": ratio_column_label,
        "strict_keys": list(strict_keys),
        "strict_config": strict_config,
        "basis_source": "pinned" if reference_from_pinned else "reference" if reference_run_id is not None else "best",
        "pinned_baseline": None if pinned_baseline_run is None else pinned_baseline_run.to_payload(analysis_options),
        "runs": [
            run.to_suite_comparison_row(
                basis_median_seconds,
                basis_run.target_return_value,
                basis_run.samples,
                analysis_options,
            )
            for run in runs
        ],
    }


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


def benchmark_run_payload(
    *,
    configuration: dict[str, Any],
    samples: list[float],
    observations: list[dict[str, Any]],
    median_seconds: float,
    min_seconds: float,
    max_seconds: float,
    std_seconds: float,
    target_return_value: StoredReturnValue | None = None,
) -> dict[str, Any]:
    return {
        "configuration": configuration,
        "samples": samples,
        "observations": observations,
        "median_seconds": median_seconds,
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
        "std_seconds": std_seconds,
        "target_return_value": target_return_value,
    }


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
    target_return_value: StoredReturnValue | None = None,
    environment: dict[str, Any],
    sweep_execution_id: int | None = None,
    run_index: int | None = None,
    database_path: str | Path | None = None,
) -> BenchmarkRun:
    with db_session(database_path) as session:
        suite = _get_or_create_suite(session, suite_name, target_name)
        stored_return_value = None if target_return_value is None else normalize_return_value(target_return_value)

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
            **benchmark_run_payload(
                configuration=configuration,
                samples=samples,
                observations=observations,
                median_seconds=median_seconds,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                std_seconds=std_seconds,
                target_return_value=stored_return_value,
            ),
        )
        session.add(benchmark_run)
        session.commit()
        session.refresh(benchmark_run)
        return benchmark_run


def list_suite_summaries(database_path: str | Path | None = None) -> list[dict[str, Any]]:
    with db_session(database_path) as session:
        suites = session.execute(select(BenchmarkSuite).order_by(BenchmarkSuite.name)).scalars().all()
        summaries: list[dict[str, Any]] = []
        for suite in suites:
            runs = session.execute(select(BenchmarkRun).where(BenchmarkRun.suite_id == suite.id)).scalars().all()
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
    analysis_options: AnalysisOptions | None = None,
    *,
    include_analysis: bool = False,
) -> dict[str, Any] | None:
    with db_session(database_path) as session:
        suite = session.scalar(_suite_query(suite_name))
        if suite is None:
            return None

        runs = session.execute(select(BenchmarkRun).where(BenchmarkRun.suite_id == suite.id).order_by(BenchmarkRun.created_at.desc())).scalars().all()
        environment = None
        if runs:
            environment = runs[0].environment
        baseline_run = _resolve_suite_baseline_run(session, suite)

    run_payloads = [run.to_payload(analysis_options, include_analysis=include_analysis) for run in runs]
    run_payloads.sort(key=lambda run: (-(run["sweep_id"]), -(run["run_index"]), -run["record_id"]))

    return {
        "suite_name": suite.name,
        "target_name": suite.target_name,
        "runs": run_payloads,
        "environment": None if environment is None else environment.to_payload(),
        "baseline_run": None if baseline_run is None else baseline_run.to_payload(analysis_options, include_analysis=include_analysis),
    }


def set_suite_baseline(
    suite_name: str,
    run_id: int | tuple[int, int],
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
) -> dict[str, Any] | None:
    with db_session(database_path) as session:
        suite = session.scalar(_suite_query(suite_name))
        if suite is None:
            return None
        run = _resolve_run(session, run_id)
        if run is None:
            return {"error": "reference_run_not_found", "suite_name": suite.name}
        if run.suite_id != suite.id:
            return {
                "error": "reference_run_wrong_suite",
                "suite_name": suite.name,
                "reference_run_display_id": run.display_id,
                "reference_run_record_id": run.id,
                "reference_run_suite_name": run.suite.name,
            }

        baseline = _get_suite_baseline_record(session, suite.id)
        if baseline is None:
            baseline = BenchmarkSuiteBaseline(suite_id=suite.id, run_id=run.id)
            session.add(baseline)
        else:
            baseline.run_id = run.id
        session.commit()
        session.refresh(run)
        return run.to_detail_payload(analysis_options)


def get_suite_baseline(
    suite_name: str,
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
) -> dict[str, Any] | None:
    with db_session(database_path) as session:
        suite = session.scalar(_suite_query(suite_name))
        if suite is None:
            return None
        run = _resolve_suite_baseline_run(session, suite)
        if run is None:
            return None
        return run.to_detail_payload(analysis_options)


def clear_suite_baseline(
    suite_name: str,
    database_path: str | Path | None = None,
) -> bool | None:
    with db_session(database_path) as session:
        suite = session.scalar(_suite_query(suite_name))
        if suite is None:
            return None
        baseline = _get_suite_baseline_record(session, suite.id)
        if baseline is None:
            return False
        session.delete(baseline)
        session.commit()
        return True


def compare_suite_runs(
    suite_name: str,
    reference_run_id: int | tuple[int, int] | None = None,
    strict_keys: Sequence[str] = (),
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
    use_pinned_baseline: bool = False,
) -> dict[str, Any] | None:
    with db_session(database_path) as session:
        suite = session.scalar(_suite_query(suite_name))
        if suite is None:
            return None

        reference_context = _resolve_suite_reference(
            session,
            suite,
            reference_run_id,
            use_pinned_baseline=use_pinned_baseline,
        )
        if isinstance(reference_context, dict):
            return reference_context
        reference_run, pinned_baseline_run, reference_run_id, reference_run_suite_name, reference_from_pinned = reference_context
        runs = _list_suite_runs(session, suite.id)

    if not runs:
        return _empty_suite_comparison_payload(suite)

    basis = _resolve_suite_basis(
        suite,
        runs,
        reference_run,
        reference_run_id,
        reference_run_suite_name,
    )
    if isinstance(basis, dict):
        return basis
    basis_run, basis_metric_label, delta_column_label, ratio_column_label = basis

    strict_result = _apply_strict_comparison_filter(
        suite,
        runs,
        basis_run,
        strict_keys,
        reference_run_id,
    )
    if isinstance(strict_result, dict):
        return strict_result
    filtered_runs, strict_keys, strict_config = strict_result

    return _suite_comparison_payload(
        suite=suite,
        runs=filtered_runs,
        basis_run=basis_run,
        basis_metric_label=basis_metric_label,
        delta_column_label=delta_column_label,
        ratio_column_label=ratio_column_label,
        strict_keys=strict_keys,
        strict_config=strict_config,
        reference_from_pinned=reference_from_pinned,
        reference_run_id=reference_run_id,
        pinned_baseline_run=pinned_baseline_run,
        analysis_options=analysis_options,
    )


def get_run_details(
    run_id: int | tuple[int, int],
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
    *,
    include_analysis: bool = False,
) -> dict[str, Any] | None:
    with db_session(database_path) as session:
        run = _resolve_run(session, run_id)
        if run is None:
            return None
        return run.to_detail_payload(analysis_options, include_analysis=include_analysis)


def get_selected_run_details(
    run_ids: Sequence[int | tuple[int, int]],
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
    *,
    include_analysis: bool = False,
) -> list[dict[str, Any]] | None:
    unique_run_ids = list(dict.fromkeys(run_ids))
    runs = [
        get_run_details(
            run_id,
            database_path,
            analysis_options=analysis_options,
            include_analysis=include_analysis,
        )
        for run_id in unique_run_ids
    ]
    if any(run is None for run in runs):
        return None

    return sorted(
        [run for run in runs if run is not None],
        key=lambda run: -int(run["id"]),
    )


def get_all_run_details(
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
    *,
    include_analysis: bool = False,
) -> list[dict[str, Any]]:
    with db_session(database_path) as session:
        runs = session.execute(select(BenchmarkRun).order_by(BenchmarkRun.sweep_execution_id.desc(), BenchmarkRun.run_index.desc(), BenchmarkRun.id.desc())).scalars().all()

        return [
            {
                **run.to_payload(analysis_options, include_analysis=include_analysis),
                "suite_name": run.suite.name,
                "target_name": run.suite.target_name,
            }
            for run in runs
        ]


def compare_runs(
    baseline_run_id: int | tuple[int, int],
    candidate_run_id: int | tuple[int, int],
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
) -> dict[str, Any] | None:
    baseline = get_run_details(
        baseline_run_id,
        database_path,
        analysis_options=analysis_options,
        include_analysis=True,
    )
    candidate = get_run_details(
        candidate_run_id,
        database_path,
        analysis_options=analysis_options,
        include_analysis=True,
    )
    if baseline is None or candidate is None:
        return None

    percent_change = None
    if baseline["median_seconds"] > 0:
        percent_change = ((candidate["median_seconds"] - baseline["median_seconds"]) / baseline["median_seconds"]) * 100.0

    baseline_observations = summarize_observations(baseline["observations"])
    candidate_observations = summarize_observations(candidate["observations"])
    labels = sorted(set(baseline_observations) | set(candidate_observations))
    target_return_relative_error = return_relative_error(
        reference_value=baseline["target_return_value"],
        candidate_value=candidate["target_return_value"],
    )

    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta_seconds": candidate["median_seconds"] - baseline["median_seconds"],
        "percent_change": percent_change,
        "target_return_relative_error": target_return_relative_error,
        "comparison_analysis": compare_sample_sets(
            baseline["samples"],
            candidate["samples"],
            analysis_options,
        ).to_payload(),
        "observation_rows": [
            {
                "label": label,
                "baseline_mean_seconds": baseline_observations[label].mean_seconds if label in baseline_observations else None,
                "baseline_std_seconds": baseline_observations[label].std_seconds if label in baseline_observations else None,
                "candidate_mean_seconds": candidate_observations[label].mean_seconds if label in candidate_observations else None,
                "candidate_std_seconds": candidate_observations[label].std_seconds if label in candidate_observations else None,
                "delta_seconds": (
                    candidate_observations[label].mean_seconds - baseline_observations[label].mean_seconds
                    if label in baseline_observations and label in candidate_observations
                    else None
                ),
            }
            for label in labels
        ],
    }


def get_suite_trend(
    suite_name: str,
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
    baseline_run_id: int | tuple[int, int] | None = None,
    limit: int | None = None,
) -> dict[str, Any] | None:
    chosen_options = analysis_options or AnalysisOptions()
    with db_session(database_path) as session:
        suite = session.scalar(_suite_query(suite_name))
        if suite is None:
            return None

        runs = (
            session.execute(
                select(BenchmarkRun).where(BenchmarkRun.suite_id == suite.id).order_by(BenchmarkRun.sweep_execution_id.asc(), BenchmarkRun.run_index.asc(), BenchmarkRun.id.asc())
            )
            .scalars()
            .all()
        )
        if not runs:
            return {
                "suite_name": suite.name,
                "target_name": suite.target_name,
                "basis_run": None,
                "basis_source": "empty",
                "config_filter": None,
                "runs": [],
            }

        if baseline_run_id is not None:
            basis_run = _resolve_run(session, baseline_run_id)
            if basis_run is None:
                return {"error": "reference_run_not_found", "suite_name": suite.name}
            if basis_run.suite_id != suite.id:
                return {
                    "error": "reference_run_wrong_suite",
                    "suite_name": suite.name,
                    "reference_run_display_id": basis_run.display_id,
                    "reference_run_record_id": basis_run.id,
                    "reference_run_suite_name": basis_run.suite.name,
                }
            basis_source = "explicit"
        else:
            basis_run = _resolve_suite_baseline_run(session, suite)
            if basis_run is not None:
                basis_source = "pinned"
            else:
                basis_run = runs[-1]
                basis_source = "latest"

        config_filter = dict(basis_run.configuration)
        filtered_runs = [run for run in runs if run.configuration == config_filter]
        if limit is not None:
            filtered_runs = filtered_runs[-limit:]
        basis_payload = basis_run.to_detail_payload(chosen_options)
        basis_samples = list(basis_run.samples)
        basis_run_id = basis_run.id
        filtered_payloads = [run.to_payload(chosen_options) for run in filtered_runs]
        filtered_samples = [list(run.samples) for run in filtered_runs]
        filtered_ids = [run.id for run in filtered_runs]

    trend_rows: list[dict[str, Any]] = []
    for index, payload in enumerate(filtered_payloads):
        run_samples = filtered_samples[index]
        vs_basis = compare_sample_sets(basis_samples, run_samples, chosen_options).to_payload()
        trailing_samples = [sample for prior_samples in filtered_samples[max(0, index - chosen_options.drift_window_size) : index] for sample in prior_samples]
        drift_analysis = None
        drift_status_label = "baseline" if payload["id"] == basis_run_id else "stable"
        if trailing_samples:
            drift_analysis = compare_sample_sets(trailing_samples, run_samples, chosen_options).to_payload()
            drift_status_label = str(drift_analysis["classification"])

        trend_rows.append(
            {
                **payload,
                "vs_baseline": vs_basis,
                "drift_analysis": drift_analysis,
                "drift_status": drift_status_label,
                "is_basis": filtered_ids[index] == basis_run_id,
            }
        )

    return {
        "suite_name": suite_name,
        "target_name": basis_payload["target_name"],
        "basis_run": basis_payload,
        "basis_source": basis_source,
        "config_filter": config_filter,
        "runs": trend_rows,
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
