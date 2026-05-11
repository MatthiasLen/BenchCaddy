from __future__ import annotations

import pytest

from benchcaddy.db import record_benchmark_run


@pytest.fixture
def environment_payload() -> dict[str, object]:
    return {
        "python_version": "3.12.0",
        "operating_system": "TestOS-1.0",
        "cpu_model": "Test CPU",
        "total_memory_bytes": 17179869184,
        "gpu_model": None,
        "git": {
            "branch": "main",
            "commit_hash": "deadbeef",
            "dirty": False,
        },
        "process": {
            "pid": 123,
            "priority": 0,
            "affinity": [0],
            "rss_bytes": 4096,
        },
    }


@pytest.fixture
def record_simple_run(environment_payload: dict[str, object]):
    def _record_simple_run(
        *,
        database_path,
        suite_name: str,
        configuration: dict[str, object],
        target_name: str = "benchmark_target",
        median_seconds: float = 0.1,
        target_return_value: bool | int | float | str | list[float] | tuple[float, ...] | dict[str, object] | None = None,
        environment: dict[str, object] | None = None,
    ):
        active_environment = environment_payload if environment is None else environment
        return record_benchmark_run(
            suite_name=suite_name,
            target_name=target_name,
            configuration=configuration,
            samples=[median_seconds, median_seconds],
            observations=[],
            median_seconds=median_seconds,
            min_seconds=median_seconds,
            max_seconds=median_seconds,
            std_seconds=0.0,
            target_return_value=target_return_value,
            environment=active_environment,
            database_path=database_path,
        )

    return _record_simple_run
