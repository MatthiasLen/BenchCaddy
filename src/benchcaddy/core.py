"""Benchmark sweep execution engine.

This module should encapsulate the mechanics of running benchmark suites:
expanding parameter grids, invoking targets, collecting samples and
observations, normalizing optional return values, and persisting run
results. It is the operational core of BenchCaddy and should avoid
taking on presentation or long-term storage responsibilities beyond the
minimal coordination needed to record completed runs.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import median, stdev
from typing import Any

from .db import benchmark_run_payload, create_sweep_execution, get_database_path, record_benchmark_run
from .isolation import prepare_system, run_isolated, validate_isolated_target
from .metadata import collect_environment_metadata, metadata_to_dict
from .reporting import RichSweepReporter, SweepReporter
from .return_values import StoredReturnValue, normalize_return_value

_DEFAULT_SAMPLE_COUNT = 7


def _target_name(target: Callable[..., Any]) -> str:
    if target_name := getattr(target, "__name__", None):
        return target_name
    if callable(target) and (call_name := getattr(target.__call__, "__name__", None)):
        return call_name
    return "callable_instance"


def _report(reporter: SweepReporter | None, event: str, **payload: Any) -> None:
    if reporter is not None:
        getattr(reporter, event)(**payload)


@dataclass
class BenchmarkResult:
    run_id: str
    record_id: int
    configuration: dict[str, Any]
    samples: list[float]
    observations: list[dict[str, Any]]
    median_seconds: float
    min_seconds: float
    max_seconds: float
    std_seconds: float
    target_return_value: StoredReturnValue | None = None


@dataclass
class Sweep:
    target: Callable[..., Any]
    params: Mapping[str, Iterable[Any]]
    suite_name: str
    samples: int = _DEFAULT_SAMPLE_COUNT
    warmup_iterations: int = 1
    lock_cpu_affinity: bool = True
    database_path: str | Path | None = None
    store_target_return_value: bool = False
    return_value_postprocessor: Callable[[Any], Any] | None = None
    verbose: bool = False
    reporter: SweepReporter | None = None

    def _prepare_return_value(self, result: Any) -> StoredReturnValue | None:
        if not self.store_target_return_value:
            return None

        transformed = self.return_value_postprocessor(result) if self.return_value_postprocessor is not None else result
        try:
            return normalize_return_value(transformed)
        except TypeError as error:
            if self.return_value_postprocessor is None:
                raise TypeError(
                    "Target returned an unsupported value type. "
                    "Provide return_value_postprocessor to map it to one of: "
                    "bool, int, float, str, or a one-dimensional numeric array/list/tuple."
                ) from error
            raise TypeError("return_value_postprocessor must return one of: bool, int, float, str, or a one-dimensional numeric array/list/tuple.") from error

    def _configurations(self) -> list[dict[str, Any]]:
        if not self.params:
            return [{}]

        param_names = list(self.params.keys())
        param_values = [list(values) for values in self.params.values()]
        return [dict(zip(param_names, combination, strict=True)) for combination in product(*param_values)]

    def _run_sample(
        self,
        configuration: Mapping[str, Any],
        reporter: SweepReporter | None,
        sample_index: int,
        sample_total: int,
    ) -> tuple[float, dict[str, Any], StoredReturnValue | None]:
        gc.collect()
        isolated_result = run_isolated(
            self.target,
            kwargs=dict(configuration),
            warmup_runs=self.warmup_iterations,
            lock_cpu_affinity=self.lock_cpu_affinity,
        )
        stored_return_value = self._prepare_return_value(isolated_result.return_value)

        _report(
            reporter,
            "on_sample_completed",
            sample_index=sample_index,
            sample_total=sample_total,
            elapsed_seconds=isolated_result.elapsed_seconds,
            observation_count=len(isolated_result.observations),
        )
        return (
            isolated_result.elapsed_seconds,
            {"sample": sample_index, "records": isolated_result.observations},
            stored_return_value,
        )

    def run(self) -> list[BenchmarkResult]:
        prepare_system(lock_cpu_affinity=self.lock_cpu_affinity)
        environment = metadata_to_dict(collect_environment_metadata())
        configurations = self._configurations()
        reporter = self.reporter or (RichSweepReporter() if self.verbose else None)
        results: list[BenchmarkResult] = []
        sample_count = self.samples
        validate_isolated_target(self.target)

        _report(
            reporter,
            "on_sweep_started",
            suite_name=self.suite_name,
            total_configurations=len(configurations),
            samples=sample_count,
            warmup_iterations=self.warmup_iterations,
            database_path=get_database_path(self.database_path),
        )

        sweep_execution = create_sweep_execution(
            suite_name=self.suite_name,
            target_name=_target_name(self.target),
            database_path=self.database_path,
        )

        for configuration_index, configuration in enumerate(configurations, start=1):
            _report(
                reporter,
                "on_configuration_started",
                index=configuration_index,
                total=len(configurations),
                configuration=configuration,
            )

            samples: list[float] = []
            observations: list[dict[str, Any]] = []
            target_return_value: StoredReturnValue | None = None

            # Run the configured number of samples for this configuration, collecting timings, observations, and the target return value from each sample.
            for sample_index in range(1, sample_count + 1):
                elapsed, observation, sample_return_value = self._run_sample(configuration, reporter, sample_index, sample_count)
                samples.append(elapsed)
                observations.append(observation)
                if target_return_value is None:
                    target_return_value = sample_return_value

            median_seconds = float(median(samples))
            min_seconds, max_seconds = float(min(samples)), float(max(samples))
            std_seconds = float(stdev(samples)) if len(samples) > 1 else 0.0
            run_payload = benchmark_run_payload(
                configuration=configuration,
                samples=samples,
                observations=observations,
                target_return_value=target_return_value,
                median_seconds=median_seconds,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                std_seconds=std_seconds,
            )

            benchmark_run = record_benchmark_run(
                suite_name=self.suite_name,
                target_name=_target_name(self.target),
                **run_payload,
                environment=environment,
                sweep_execution_id=sweep_execution.id,
                run_index=configuration_index,
                database_path=self.database_path,
            )
            results.append(
                BenchmarkResult(
                    run_id=benchmark_run.display_id,
                    record_id=benchmark_run.id,
                    **run_payload,
                )
            )

            _report(
                reporter,
                "on_configuration_completed",
                index=configuration_index,
                total=len(configurations),
                configuration=configuration,
                median_seconds=median_seconds,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                std_seconds=std_seconds,
                sample_count=len(samples),
                target_return_value=target_return_value,
            )

        _report(reporter, "on_sweep_completed", results=results)

        return results

    __call__ = run
