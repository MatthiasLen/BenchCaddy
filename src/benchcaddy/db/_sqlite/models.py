from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean
from typing import Any

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Index, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql.functions import now
from sqlalchemy.types import JSON

from ...return_values import StoredReturnValue, return_relative_error
from ...stats import AnalysisOptions, analyze_samples, compare_sample_sets


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
    __table_args__ = (
        # Tuple run ids back the public sweep.run identifier format, so they must be unique when present.
        Index("ix_benchmark_runs_sweep_run", "sweep_execution_id", "run_index", unique=True),
        Index("ix_benchmark_runs_suite_history", "suite_id", "sweep_execution_id", "run_index", "id"),
    )

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
        # Standalone historical rows fall back to their record id so every run still has a stable display id.
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
        # Reuse one computed analysis payload for both the nested block and the convenience summary fields.
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
            # Detail views inline suite and environment data so callers do not need follow-up lookups.
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
        # Comparison rows bundle per-run analysis with deltas relative to the chosen basis run.
        analysis = analyze_samples(self.samples, analysis_options).to_payload()
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
            "target_return_relative_error": return_relative_error(
                reference_value=reference_run_target_value,
                candidate_value=self.target_return_value,
            ),
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
