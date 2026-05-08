from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

import psutil

from .db import record_benchmark_run
from .metadata import collect_environment_metadata, metadata_to_dict
from .observability import collect_observations


def prepare_system(lock_cpu_affinity: bool = True) -> None:
    """Raise process priority, optionally pin the current affinity set, and freeze GC state."""
    process = psutil.Process()

    try:
        if os.name == "nt" and hasattr(psutil, "HIGH_PRIORITY_CLASS"):
            process.nice(psutil.HIGH_PRIORITY_CLASS)
        elif hasattr(os, "nice"):
            os.nice(-20)
    except (PermissionError, psutil.AccessDenied, AttributeError, OSError):
        pass

    if lock_cpu_affinity and hasattr(process, "cpu_affinity"):
        try:
            affinity = list(process.cpu_affinity())
            if affinity:
                process.cpu_affinity([affinity[0]])
        except (psutil.AccessDenied, NotImplementedError, ValueError):
            pass

    gc.collect()
    if hasattr(gc, "freeze"):
        gc.freeze()


def _target_name(target: Callable[..., Any]) -> str:
    name = getattr(target, "__name__", None)
    if isinstance(name, str) and name:
        return name

    call_name = getattr(getattr(target, "__call__", None), "__name__", None)
    if isinstance(call_name, str) and call_name != "__call__":
        return call_name

    return "callable_instance"


@dataclass
class BenchmarkResult:
    configuration: dict[str, Any]
    samples: list[float]
    observations: list[dict[str, Any]]
    median_seconds: float


@dataclass
class Sweep:
    target: Callable[..., Any]
    params: Mapping[str, Iterable[Any]]
    suite_name: str
    samples: int = 7
    warmup_iterations: int = 1
    lock_cpu_affinity: bool = True
    database_path: str | Path | None = None
    sync: Callable[[], None] | None = None

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
            return

        synchronize = getattr(result, "synchronize", None)
        if callable(synchronize):
            synchronize()

    def run(self) -> list[BenchmarkResult]:
        prepare_system(lock_cpu_affinity=self.lock_cpu_affinity)
        environment = metadata_to_dict(collect_environment_metadata())
        results: list[BenchmarkResult] = []

        for configuration in self._configurations():
            for _ in range(self.warmup_iterations):
                warmup_result = self.target(**configuration)
                self._sync_if_needed(warmup_result)

            samples: list[float] = []
            observations: list[dict[str, Any]] = []
            for sample_index in range(self.samples):
                with collect_observations() as collector:
                    start = perf_counter()
                    result = self.target(**configuration)
                    self._sync_if_needed(result)
                    elapsed = perf_counter() - start

                samples.append(elapsed)
                observations.append(
                    {
                        "sample": sample_index + 1,
                        "records": collector.records,
                    }
                )

            median_seconds = float(median(samples))
            record_benchmark_run(
                suite_name=self.suite_name,
                target_name=_target_name(self.target),
                configuration=configuration,
                samples=samples,
                observations=observations,
                median_seconds=median_seconds,
                environment=environment,
                database_path=self.database_path,
            )
            results.append(
                BenchmarkResult(
                    configuration=configuration,
                    samples=samples,
                    observations=observations,
                    median_seconds=median_seconds,
                )
            )

        return results

    def __call__(self) -> list[BenchmarkResult]:
        return self.run()
