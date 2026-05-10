from __future__ import annotations

import gc
import os
import subprocess
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import median, stdev
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

import psutil

from .db import create_sweep_execution, get_database_path, record_benchmark_run
from .metadata import collect_environment_metadata, metadata_to_dict
from .observability import collect_observations
from .return_values import StoredReturnValue, normalize_return_value
from .reporting import RichSweepReporter, SweepReporter

_MAX_UNIX_PRIORITY = -20
_DEFAULT_SAMPLE_COUNT = 7


def _as_script_path(target: Callable[..., Any] | str | Path) -> Path | None:
    if isinstance(target, (str, Path)):
        return Path(target)
    return None


def prepare_system(lock_cpu_affinity: bool = True) -> None:
    """Raise process priority, optionally preserve the current affinity set, and freeze GC state."""
    process = psutil.Process()

    try:
        if os.name == "nt" and hasattr(psutil, "HIGH_PRIORITY_CLASS"):
            process.nice(psutil.HIGH_PRIORITY_CLASS)
        elif hasattr(os, "nice"):
            # set to highest priority
            os.nice(_MAX_UNIX_PRIORITY - os.nice(0))
    except (PermissionError, psutil.AccessDenied, AttributeError, OSError):
        pass

    if lock_cpu_affinity and hasattr(process, "cpu_affinity"):
        try:
            affinity = list(process.cpu_affinity())
            if affinity:
                # Self-Affinity Refresh: Re-apply the current affinity to potentially lock 
                # the process to the current CPU set and avoid migrations.
                process.cpu_affinity(affinity)
        except (psutil.AccessDenied, NotImplementedError, ValueError):
            pass

    gc.collect()
    if hasattr(gc, "freeze"):
        gc.freeze()


def _target_name(target: Callable[..., Any] | str | Path) -> str:
    if (script := _as_script_path(target)): return script.name
    return getattr(target, "__name__", None) or getattr(getattr(target, "__call__", None), "__name__", None) or "callable_instance"


def _argument_tokens(configuration: Mapping[str, Any]) -> list[str]:
    tokens: list[str] = []
    for k, v in configuration.items():
        flag = f"--{k.replace('_', '-')}"
        if v is True: tokens.append(flag)
        elif v is False: tokens.extend([flag, "false"])
        elif v is not None: tokens.extend([flag, str(v)])
    return tokens


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
    target: Callable[..., Any] | str | Path
    params: Mapping[str, Iterable[Any]]
    suite_name: str
    samples: int = _DEFAULT_SAMPLE_COUNT
    iterations: int | None = None
    warmup_iterations: int = 1
    warmup_runs: int | None = None
    lock_cpu_affinity: bool = True
    database_path: str | Path | None = None
    sync: Callable[[], None] | None = None
    store_target_return_value: bool = False
    return_value_postprocessor: Callable[[Any], Any] | None = None
    verbose: bool = False
    reporter: SweepReporter | None = None

    def _prepare_return_value(self, result: Any) -> StoredReturnValue | None:
        if not self.store_target_return_value:
            return None

        transformed = (
            self.return_value_postprocessor(result)
            if self.return_value_postprocessor is not None
            else result
        )
        try:
            return normalize_return_value(transformed)
        except TypeError as error:
            if self.return_value_postprocessor is None:
                raise TypeError(
                    "Target returned an unsupported value type. "
                    "Provide return_value_postprocessor to map it to one of: "
                    "bool, int, float, str, or a one-dimensional numeric array/list/tuple."
                ) from error
            raise TypeError(
                "return_value_postprocessor must return one of: "
                "bool, int, float, str, or a one-dimensional numeric array/list/tuple."
            ) from error

    def _configurations(self) -> list[dict[str, Any]]:
        if not self.params:
            return [{}]

        param_names = list(self.params.keys())
        param_values = [list(values) for values in self.params.values()]
        return [
            dict(zip(param_names, combination, strict=True))
            for combination in product(*param_values)
        ]

    def _sync_if_needed(self, result: Any) -> None:
        if self.sync is not None:
            self.sync()
        elif callable(synchronize := getattr(result, "synchronize", None)):
            synchronize()

    def _invoke_target(self, configuration: Mapping[str, Any]) -> Any:
        script_path = _as_script_path(self.target)
        if script_path is not None:
            command = [sys.executable, str(script_path), *_argument_tokens(configuration)]
            return subprocess.run(command, check=True)

        return self.target(**configuration)

    def _run_sample(
        self,
        configuration: Mapping[str, Any],
        reporter: SweepReporter | None,
        sample_index: int,
        sample_total: int,
    ) -> tuple[float, dict[str, Any], StoredReturnValue | None]:
        gc.collect()
        with collect_observations() as collector:
            start = perf_counter()
            result = self._invoke_target(configuration)
            self._sync_if_needed(result)
            elapsed = perf_counter() - start
            stored_return_value = self._prepare_return_value(result)

        _report(
            reporter,
            "on_sample_completed",
            sample_index=sample_index,
            sample_total=sample_total,
            elapsed_seconds=elapsed,
            observation_count=len(collector.records),
        )
        return elapsed, {"sample": sample_index, "records": collector.records}, stored_return_value

    def run(self, sync: Callable[[], None] | None = None) -> list[BenchmarkResult]:
        prepare_system(lock_cpu_affinity=self.lock_cpu_affinity)
        environment = metadata_to_dict(collect_environment_metadata())
        configurations = self._configurations()
        reporter = self.reporter or (RichSweepReporter() if self.verbose else None)
        results: list[BenchmarkResult] = []
        sample_count = self.iterations if self.iterations is not None else self.samples
        warmup_count = self.warmup_runs if self.warmup_runs is not None else self.warmup_iterations

        if sync is not None:
            self.sync = sync

        _report(
            reporter,
            "on_sweep_started",
            suite_name=self.suite_name,
            total_configurations=len(configurations),
            samples=sample_count,
            warmup_iterations=warmup_count,
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

            for _ in range(warmup_count):
                self._sync_if_needed(self._invoke_target(configuration))

            samples: list[float] = []
            observations: list[dict[str, Any]] = []
            target_return_value: StoredReturnValue | None = None
            for sample_index in range(1, sample_count + 1):
                elapsed, observation, sample_return_value = self._run_sample(
                    configuration, reporter, sample_index, sample_count
                )
                samples.append(elapsed)
                observations.append(observation)
                if target_return_value is None:
                    target_return_value = sample_return_value

            median_seconds = float(median(samples))
            min_seconds, max_seconds = float(min(samples)), float(max(samples))
            std_seconds = float(stdev(samples)) if len(samples) > 1 else 0.0

            benchmark_run = record_benchmark_run(
                suite_name=self.suite_name,
                target_name=_target_name(self.target),
                configuration=configuration,
                samples=samples,
                observations=observations,
                median_seconds=median_seconds,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                std_seconds=std_seconds,
                target_return_value=target_return_value,
                environment=environment,
                sweep_execution_id=sweep_execution.id,
                run_index=configuration_index,
                database_path=self.database_path,
            )
            results.append(
                BenchmarkResult(
                    run_id=benchmark_run.display_id,
                    record_id=benchmark_run.id,
                    configuration=configuration,
                    samples=samples,
                    observations=observations,
                    median_seconds=median_seconds,
                    min_seconds=min_seconds,
                    max_seconds=max_seconds,
                    std_seconds=std_seconds,
                    target_return_value=target_return_value,
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
