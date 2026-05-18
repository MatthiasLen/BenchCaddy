from __future__ import annotations

from pathlib import Path
from typing import Any

from ..return_values import StoredReturnValue, normalize_return_value
from ..stats import AnalysisOptions
from ._backend import (
    BenchmarkRun,
    BenchmarkSuiteBaseline,
    BenchmarkSweepExecution,
    EnvironmentInfo,
    _get_or_create_suite,
    _get_suite,
    _get_suite_baseline_record,
    _resolve_run,
    db_session,
)


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


def set_suite_baseline(
    suite_name: str,
    run_id: int | tuple[int, int],
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
) -> dict[str, Any] | None:
    with db_session(database_path) as session:
        suite = _get_suite(session, suite_name)
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