from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..return_values import StoredReturnValue, normalize_return_value
from ..stats import AnalysisOptions
from ._sqlite.models import BenchmarkRun, BenchmarkSuiteBaselineEvent, BenchmarkSweepExecution, EnvironmentInfo
from ._sqlite.session import db_session
from ._sqlite.store import (
    _get_or_create_suite,
    _get_suite,
    _resolve_run,
)


@dataclass(frozen=True)
class SweepExecutionRecord:
    id: int


@dataclass(frozen=True)
class BenchmarkRunRecord:
    id: int
    display_id: str


def create_sweep_execution(
    *,
    suite_name: str,
    target_name: str,
    database_path: str | Path | None = None,
) -> SweepExecutionRecord:
    with db_session(database_path) as session:
        with session.begin():
            suite = _get_or_create_suite(session, suite_name, target_name)
            sweep_execution = BenchmarkSweepExecution(suite_id=suite.id)
            session.add(sweep_execution)
            # Flush assigns the sweep id before the transaction commits so callers can reuse it immediately.
            session.flush()
            sweep_execution_id = sweep_execution.id
        return SweepExecutionRecord(id=sweep_execution_id)


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
) -> BenchmarkRunRecord:
    with db_session(database_path) as session:
        with session.begin():
            suite = _get_or_create_suite(session, suite_name, target_name)
            # Persist a normalized return-value shape so later comparisons do not need per-call coercion.
            stored_return_value = None if target_return_value is None else normalize_return_value(target_return_value)

            if sweep_execution_id is None:
                # Standalone writes create their own sweep so display ids still use the sweep.run format.
                sweep_execution = BenchmarkSweepExecution(suite_id=suite.id)
                session.add(sweep_execution)
                session.flush()
                sweep_execution_id = sweep_execution.id
            if run_index is None:
                run_index = 1

            # Environment metadata is normalized into its own row and linked from each benchmark run.
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
            # Flush exposes generated ids for the API payload without committing early.
            session.flush()
            benchmark_run_id = benchmark_run.id
            benchmark_run_display_id = benchmark_run.display_id
        return BenchmarkRunRecord(id=benchmark_run_id, display_id=benchmark_run_display_id)


def set_suite_baseline(
    suite_name: str,
    run_id: int | tuple[int, int],
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
    note: str | None = None,
    *,
    include_samples: bool = True,
    include_observations: bool = True,
    include_environment: bool = True,
) -> dict[str, Any] | None:
    with db_session(database_path) as session:
        with session.begin():
            suite = _get_suite(session, suite_name)
            if suite is None:
                return None
            run = _resolve_run(session, run_id)
            if run is None:
                return {"error": "reference_run_not_found", "suite_name": suite.name}
            # Baselines are suite-local; pointing at another suite would break comparisons and trend views.
            if run.suite_id != suite.id:
                return {
                    "error": "reference_run_wrong_suite",
                    "suite_name": suite.name,
                    "reference_run_display_id": run.display_id,
                    "reference_run_record_id": run.id,
                    "reference_run_suite_name": run.suite.name,
                }

            # Baseline pins are append-only events so the full baseline history remains auditable.
            session.add(BenchmarkSuiteBaselineEvent(suite_id=suite.id, run_id=run.id, note=note))
        return run.to_detail_payload(
            analysis_options,
            include_samples=include_samples,
            include_observations=include_observations,
            include_environment=include_environment,
        )
