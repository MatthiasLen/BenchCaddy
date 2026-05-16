from __future__ import annotations

import threading
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

import benchcaddy.core as core_module
import benchcaddy.isolation.process as isolation_process_module
import benchcaddy.observability as observability_module
from benchcaddy import Sweep, observe
from benchcaddy.db import (
    compare_runs,
    compare_suite_runs,
    db_session,
    get_run_details,
    get_suite_details,
    get_suite_trend,
    initialize_database,
    list_suite_summaries,
    record_benchmark_run,
    set_suite_baseline,
)
from benchcaddy.observability import collect_observations, summarize_observations
from benchcaddy.reporting import RichSweepReporter


def test_sweep_records_results_and_observations(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    metadata_marker = object()

    def stub_collect_environment_metadata() -> object:
        return metadata_marker

    def stub_prepare_system(lock_cpu_affinity: bool = True) -> None:
        del lock_cpu_affinity

    monkeypatch.setattr(core_module, "prepare_system", stub_prepare_system)
    monkeypatch.setattr(core_module, "collect_environment_metadata", stub_collect_environment_metadata)
    monkeypatch.setattr(core_module, "metadata_to_dict", lambda metadata: environment_payload)

    @observe("inner-step")
    def measured_step(size: int, bias: float) -> float:
        total = 0.0
        for index in range(size * 200):
            total += ((index % 11) * 0.01) + bias
        return total

    def benchmark_target(size: int, variant: str) -> float:
        bias = 0.01 if variant == "baseline" else 0.015
        return measured_step(size, bias)

    sweep = Sweep(
        target=benchmark_target,
        params={"size": [8], "variant": ["baseline", "stabilized"]},
        suite_name="core-test-suite",
        samples=2,
        warmup_iterations=1,
        lock_cpu_affinity=False,
        database_path=database_path,
    )

    results = sweep.run()

    assert len(results) == 2
    assert all(len(result.samples) == 2 for result in results)
    assert all(len(result.observations) == 2 for result in results)
    assert all(result.observations[0]["records"][0]["label"] == "inner-step" for result in results)

    summaries = list_suite_summaries(database_path)
    assert summaries == [
        {
            "suite_name": "core-test-suite",
            "target_name": "benchmark_target",
            "run_count": 2,
            "last_run_at": summaries[0]["last_run_at"],
            "observation_labels": ["inner-step"],
        }
    ]

    details = get_suite_details("core-test-suite", database_path)
    assert details is not None
    assert details["target_name"] == "benchmark_target"
    assert len(details["runs"]) == 2
    assert [run["display_id"] for run in details["runs"]] == ["1.2", "1.1"]
    assert details["environment"]["python_version"] == "3.12.0"
    assert details["environment"]["total_memory_bytes"] == 17179869184


def _stringifiable_return_value(size: int) -> dict[str, object]:
    return {"size": size, "ok": True}


def _vector_return_value() -> tuple[int, float, int]:
    return (1, 2.5, 3)


def _numpy_vector_payload() -> dict[str, list[int]]:
    return {"values": [1, 2, 3]}


def _postprocess_stringifiable_return_value(payload: dict[str, object]) -> str:
    return f"{payload['size']}:{payload['ok']}"


def _postprocess_numpy_vector_payload(payload: dict[str, list[int]]):
    import numpy as np

    return np.asarray(payload["values"], dtype=float)


@pytest.mark.parametrize(
    ("target", "params", "suite_name", "return_value_postprocessor", "expected_return_value"),
    [
        pytest.param(
            _stringifiable_return_value,
            {"size": [3]},
            "return-value-suite",
            _postprocess_stringifiable_return_value,
            "3:True",
            id="postprocessed-string",
        ),
        pytest.param(
            _vector_return_value,
            {},
            "vector-return-value-suite",
            None,
            [1.0, 2.5, 3.0],
            id="numeric-vector",
        ),
        pytest.param(
            _numpy_vector_payload,
            {},
            "numpy-return-value-suite",
            _postprocess_numpy_vector_payload,
            [1.0, 2.0, 3.0],
            id="numpy-postprocessor",
        ),
    ],
)
def test_sweep_persists_supported_return_value_shapes(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
    build_single_sample_sweep,
    target,
    params: dict[str, list[int] | list[str]],
    suite_name: str,
    return_value_postprocessor,
    expected_return_value: str | list[float],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    metadata_marker = object()

    monkeypatch.setattr(core_module, "prepare_system", lambda lock_cpu_affinity=True: None)
    monkeypatch.setattr(core_module, "collect_environment_metadata", lambda: metadata_marker)
    monkeypatch.setattr(core_module, "metadata_to_dict", lambda metadata: environment_payload)

    sweep = build_single_sample_sweep(
        target=target,
        params=params,
        suite_name=suite_name,
        database_path=database_path,
        store_target_return_value=True,
        return_value_postprocessor=return_value_postprocessor,
    )

    results = sweep.run()
    assert results[0].target_return_value == expected_return_value

    run = get_run_details((1, 1), database_path)
    assert run is not None
    assert run["target_return_value"] == expected_return_value


def test_suite_baseline_persistence_and_trend_filtering(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"

    record_benchmark_run(
        suite_name="trend-suite",
        target_name="benchmark_target",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        observations=[],
        median_seconds=0.100,
        min_seconds=0.099,
        max_seconds=0.102,
        std_seconds=0.0008944272,
        environment=environment_payload,
        database_path=database_path,
    )
    record_benchmark_run(
        suite_name="trend-suite",
        target_name="benchmark_target",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.109, 0.110, 0.111, 0.110, 0.112, 0.109, 0.111],
        observations=[],
        median_seconds=0.110,
        min_seconds=0.109,
        max_seconds=0.112,
        std_seconds=0.0008944272,
        environment=environment_payload,
        database_path=database_path,
    )
    record_benchmark_run(
        suite_name="trend-suite",
        target_name="benchmark_target",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.139, 0.140, 0.141, 0.142, 0.140, 0.139, 0.141],
        observations=[],
        median_seconds=0.140,
        min_seconds=0.139,
        max_seconds=0.142,
        std_seconds=0.0008944272,
        environment=environment_payload,
        database_path=database_path,
    )
    record_benchmark_run(
        suite_name="trend-suite",
        target_name="benchmark_target",
        configuration={"size": 1024, "variant": "baseline"},
        samples=[0.199, 0.200, 0.201, 0.200, 0.199],
        observations=[],
        median_seconds=0.200,
        min_seconds=0.199,
        max_seconds=0.201,
        std_seconds=0.0008944272,
        environment=environment_payload,
        database_path=database_path,
    )

    pinned = set_suite_baseline("trend-suite", 1, database_path)
    assert pinned is not None
    assert pinned["display_id"] == "1.1"

    comparison = compare_suite_runs("trend-suite", database_path=database_path, use_pinned_baseline=True)
    assert comparison is not None
    assert comparison["basis_source"] == "pinned"
    assert comparison["basis_run"]["display_id"] == "1.1"

    trend = get_suite_trend("trend-suite", database_path, baseline_run_id=1)
    assert trend is not None
    assert trend["basis_source"] == "explicit"
    assert trend["config_filter"] == {"size": 512, "variant": "baseline"}
    assert [run["display_id"] for run in trend["runs"]] == ["1.1", "2.1", "3.1"]
    assert trend["runs"][-1]["vs_baseline"]["classification"] == "regressing"


def test_sweep_requires_supported_target_return_types_when_enabled(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
    build_single_sample_sweep,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    metadata_marker = object()

    monkeypatch.setattr(core_module, "prepare_system", lambda lock_cpu_affinity=True: None)
    monkeypatch.setattr(core_module, "collect_environment_metadata", lambda: metadata_marker)
    monkeypatch.setattr(core_module, "metadata_to_dict", lambda metadata: environment_payload)

    sweep = build_single_sample_sweep(
        target=lambda: {"complex": "payload"},
        suite_name="unsupported-return-value-suite",
        database_path=database_path,
        store_target_return_value=True,
    )

    with pytest.raises(TypeError, match="one-dimensional numeric array/list/tuple"):
        sweep.run()


def test_sweep_rejects_non_vector_array_shapes(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
    build_single_sample_sweep,
) -> None:
    import numpy as np

    database_path = tmp_path / "benchcaddy.db"
    metadata_marker = object()

    monkeypatch.setattr(core_module, "prepare_system", lambda lock_cpu_affinity=True: None)
    monkeypatch.setattr(core_module, "collect_environment_metadata", lambda: metadata_marker)
    monkeypatch.setattr(core_module, "metadata_to_dict", lambda metadata: environment_payload)

    sweep = build_single_sample_sweep(
        target=lambda: np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        suite_name="invalid-vector-return-suite",
        database_path=database_path,
        store_target_return_value=True,
    )

    with pytest.raises(TypeError, match="one-dimensional numeric array/list/tuple"):
        sweep.run()


def test_record_benchmark_run_normalizes_supported_return_values(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"

    record_simple_run(
        suite_name="direct-recording-suite",
        database_path=database_path,
        configuration={"variant": "vector"},
        target_return_value=(1, 2.5, 3),
    )

    run = get_run_details((1, 1), database_path)
    assert run is not None
    assert run["target_return_value"] == [1.0, 2.5, 3.0]


def test_record_benchmark_run_rejects_unsupported_return_values(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"

    with pytest.raises(TypeError, match="one-dimensional numeric array/list/tuple"):
        record_simple_run(
            suite_name="direct-recording-suite",
            database_path=database_path,
            configuration={"variant": "invalid"},
            target_return_value={"complex": "payload"},
        )


def test_compare_runs_computes_vector_distance_and_handles_mismatched_lengths(
    tmp_path: Path,
    environment_payload: dict[str, object],
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"

    record_simple_run(
        suite_name="vector-distance-suite",
        database_path=database_path,
        configuration={"variant": "baseline"},
        target_return_value=[1.0, 2.0, 3.0],
    )
    record_simple_run(
        suite_name="vector-distance-suite",
        database_path=database_path,
        configuration={"variant": "candidate"},
        median_seconds=0.11,
        target_return_value=[4.0, 6.0, 3.0],
    )

    comparison = compare_runs((1, 1), (2, 1), database_path)
    assert comparison is not None
    expected_distance = ((4.0 - 1.0) ** 2 + (6.0 - 2.0) ** 2 + (3.0 - 3.0) ** 2) ** 0.5
    expected_relative_error = expected_distance / ((1.0**2 + 2.0**2 + 3.0**2) ** 0.5)
    assert comparison["target_return_relative_error"] == expected_relative_error

    record_simple_run(
        suite_name="vector-distance-suite",
        database_path=database_path,
        configuration={"variant": "mismatched"},
        median_seconds=0.12,
        target_return_value=[1.0, 2.0],
    )

    mismatched_comparison = compare_runs((1, 1), (3, 1), database_path)
    assert mismatched_comparison is not None
    assert mismatched_comparison["target_return_relative_error"] is None


def test_verbose_sweep_uses_reporter(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    events: list[tuple[str, object]] = []
    metadata_marker = object()

    class RecordingReporter:
        def on_sweep_started(self, **kwargs) -> None:
            events.append(("sweep-started", kwargs["suite_name"]))

        def on_configuration_started(self, **kwargs) -> None:
            events.append(("configuration-started", kwargs["index"]))

        def on_sample_completed(self, **kwargs) -> None:
            events.append(("sample-completed", kwargs["sample_index"]))

        def on_configuration_completed(self, **kwargs) -> None:
            events.append(("configuration-completed", kwargs["median_seconds"]))

        def on_sweep_completed(self, **kwargs) -> None:
            events.append(("sweep-completed", len(kwargs["results"])))

    def stub_collect_environment_metadata() -> object:
        return metadata_marker

    def stub_prepare_system(lock_cpu_affinity: bool = True) -> None:
        del lock_cpu_affinity

    def reporter_factory() -> RecordingReporter:
        return RecordingReporter()

    monkeypatch.setattr(core_module, "prepare_system", stub_prepare_system)
    monkeypatch.setattr(core_module, "collect_environment_metadata", stub_collect_environment_metadata)
    monkeypatch.setattr(core_module, "metadata_to_dict", lambda metadata: environment_payload)
    monkeypatch.setattr(core_module, "RichSweepReporter", reporter_factory)

    sweep = Sweep(
        target=lambda: 1.0,
        params={},
        suite_name="verbose-test-suite",
        samples=2,
        warmup_iterations=0,
        lock_cpu_affinity=False,
        database_path=database_path,
        verbose=True,
    )

    sweep.run()

    event_names = [event[0] for event in events]
    assert event_names.count("sweep-started") == 1
    assert event_names.count("configuration-started") == 1
    assert event_names.count("sample-completed") == 2
    assert event_names.count("configuration-completed") == 1
    assert event_names.count("sweep-completed") == 1


def test_verbose_sweep_prints_scientific_return_values(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    output = StringIO()
    metadata_marker = object()

    monkeypatch.setattr(core_module, "prepare_system", lambda lock_cpu_affinity=True: None)
    monkeypatch.setattr(core_module, "collect_environment_metadata", lambda: metadata_marker)
    monkeypatch.setattr(core_module, "metadata_to_dict", lambda metadata: environment_payload)
    monkeypatch.setattr(
        core_module,
        "RichSweepReporter",
        lambda: RichSweepReporter(console=Console(file=output, force_terminal=False, color_system=None, width=120)),
    )

    sweep = Sweep(
        target=lambda: (1.0, 2.5, 3.0),
        params={},
        suite_name="verbose-return-suite",
        samples=1,
        warmup_iterations=0,
        lock_cpu_affinity=False,
        database_path=database_path,
        store_target_return_value=True,
        verbose=True,
    )

    sweep.run()

    reporter_output = output.getvalue()
    assert "Run ID" in reporter_output
    assert "Record ID" in reporter_output
    assert "1.1" in reporter_output
    assert "Return Value" in reporter_output
    assert "[1.000000e+00, 2.500000e+00, 3.000000e+00]" in reporter_output


def test_sweep_supports_script_targets(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    script_path = tmp_path / "benchmark_script.py"
    marker_path = tmp_path / "invocations.txt"
    metadata_marker = object()

    script_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import argparse",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--size', type=int, required=True)",
                "parser.add_argument('--variant', required=True)",
                "args = parser.parse_args()",
                "Path(" + repr(str(marker_path)) + ").open('a', encoding='utf-8').write(f'{args.size}:{args.variant}\\n')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def stub_collect_environment_metadata() -> object:
        return metadata_marker

    def stub_prepare_system(lock_cpu_affinity: bool = True) -> None:
        del lock_cpu_affinity

    monkeypatch.setattr(core_module, "prepare_system", stub_prepare_system)
    monkeypatch.setattr(core_module, "collect_environment_metadata", stub_collect_environment_metadata)
    monkeypatch.setattr(core_module, "metadata_to_dict", lambda metadata: environment_payload)

    sweep = Sweep(
        target=script_path,
        params={"size": [8], "variant": ["baseline", "stabilized"]},
        suite_name="script-target-suite",
        iterations=1,
        warmup_runs=0,
        lock_cpu_affinity=False,
        database_path=database_path,
    )

    results = sweep.run()

    assert len(results) == 2
    assert marker_path.read_text(encoding="utf-8").splitlines() == [
        "8:baseline",
        "8:stabilized",
    ]


def test_script_argument_tokens_preserve_false_values() -> None:
    assert core_module._argument_tokens(
        {
            "use_cache": False,
            "dry_run": True,
            "size": 512,
            "label": None,
        }
    ) == ["--use-cache", "false", "--dry-run", "--size", "512"]


def test_separate_sweeps_get_distinct_sweep_ids(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    metadata_marker = object()

    monkeypatch.setattr(core_module, "prepare_system", lambda lock_cpu_affinity=True: None)
    monkeypatch.setattr(core_module, "collect_environment_metadata", lambda: metadata_marker)
    monkeypatch.setattr(core_module, "metadata_to_dict", lambda metadata: environment_payload)

    sweep = Sweep(
        target=lambda variant: 1.0 if variant == "baseline" else 2.0,
        params={"variant": ["baseline", "candidate"]},
        suite_name="grouped-suite",
        samples=1,
        warmup_iterations=0,
        lock_cpu_affinity=False,
        database_path=database_path,
    )

    sweep.run()
    sweep.run()

    details = get_suite_details("grouped-suite", database_path)
    assert details is not None
    assert [run["display_id"] for run in details["runs"]] == ["2.2", "2.1", "1.2", "1.1"]


def test_multiple_observe_labels_are_recorded_and_compared(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    metadata_marker = object()

    monkeypatch.setattr(core_module, "prepare_system", lambda lock_cpu_affinity=True: None)
    monkeypatch.setattr(core_module, "collect_environment_metadata", lambda: metadata_marker)
    monkeypatch.setattr(core_module, "metadata_to_dict", lambda metadata: environment_payload)

    @observe("outer")
    def outer_step(scale: int) -> int:
        return inner_step(scale) + scale

    @observe("inner")
    def inner_step(scale: int) -> int:
        return scale * 2

    sweep = Sweep(
        target=outer_step,
        params={"scale": [1, 2]},
        suite_name="observe-suite",
        samples=2,
        warmup_iterations=0,
        lock_cpu_affinity=False,
        database_path=database_path,
    )

    sweep.run()

    first_run = get_run_details((1, 1), database_path)
    second_run = get_run_details((1, 2), database_path)
    assert first_run is not None
    assert second_run is not None
    assert all({record["label"] for record in sample["records"]} == {"inner", "outer"} for sample in first_run["observations"])

    comparison = compare_runs((1, 1), (1, 2), database_path)
    assert comparison is not None
    assert [row["label"] for row in comparison["observation_rows"]] == ["inner", "outer"]


def test_observation_summary_tracks_mean_and_std_per_sample_total() -> None:
    summary = summarize_observations(
        [
            {"sample": 1, "records": [{"label": "inner", "duration_seconds": 0.01}, {"label": "inner", "duration_seconds": 0.02}]},
            {"sample": 2, "records": [{"label": "inner", "duration_seconds": 0.06}]},
        ]
    )

    assert summary["inner"].calls == 3
    assert summary["inner"].total_seconds == 0.09
    assert summary["inner"].mean_seconds == 0.045
    assert round(summary["inner"].std_seconds, 6) == 0.021213


def test_prepare_system_keeps_current_affinity_set(monkeypatch) -> None:
    recorded_affinity: list[int] = []

    class DummyProcess:
        def nice(self, *_args, **_kwargs) -> None:
            return None

        def cpu_affinity(self, cpus=None):
            nonlocal recorded_affinity
            if cpus is None:
                return [2, 4]
            recorded_affinity = list(cpus)
            return recorded_affinity

    def process_factory() -> DummyProcess:
        return DummyProcess()

    monkeypatch.setattr(isolation_process_module.psutil, "Process", process_factory)
    monkeypatch.setattr(isolation_process_module.gc, "collect", lambda: None)
    monkeypatch.setattr(isolation_process_module.gc, "freeze", lambda: None)
    monkeypatch.setattr(isolation_process_module.os, "name", "nt")
    monkeypatch.setattr(isolation_process_module.psutil, "HIGH_PRIORITY_CLASS", 128, raising=False)

    core_module.prepare_system(lock_cpu_affinity=True)

    assert recorded_affinity == [2, 4]


def test_prepare_system_skips_affinity_refresh_when_disabled(monkeypatch) -> None:
    affinity_set_calls: list[list[int]] = []

    class DummyProcess:
        def nice(self, *_args, **_kwargs) -> None:
            return None

        def cpu_affinity(self, cpus=None):
            if cpus is None:
                return [1, 3]
            affinity_set_calls.append(list(cpus))
            return list(cpus)

    monkeypatch.setattr(isolation_process_module.psutil, "Process", DummyProcess)
    monkeypatch.setattr(isolation_process_module.gc, "collect", lambda: None)
    monkeypatch.setattr(isolation_process_module.gc, "freeze", lambda: None)
    monkeypatch.setattr(isolation_process_module.os, "name", "nt")
    monkeypatch.setattr(isolation_process_module.psutil, "HIGH_PRIORITY_CLASS", 128, raising=False)

    core_module.prepare_system(lock_cpu_affinity=False)

    assert affinity_set_calls == []


def test_run_sample_measures_target_and_sync_with_gc_outside_timing(monkeypatch) -> None:
    events: list[str] = []
    timer_values = iter([10.0, 10.125])
    sweep = Sweep(target=lambda: None, params={}, suite_name="timing-boundary-suite", warmup_iterations=0)

    monkeypatch.setattr(core_module.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(core_module, "perf_counter", lambda: events.append("timer") or next(timer_values))
    monkeypatch.setattr(sweep, "_invoke_target", lambda configuration: events.append("target") or object())
    sweep.sync = lambda: events.append("sync")

    elapsed, observation, stored_return_value = sweep._run_sample({}, None, 1, 1)

    assert events == ["gc", "timer", "target", "sync", "timer"]
    assert elapsed == pytest.approx(0.125)
    assert observation == {"sample": 1, "records": []}
    assert stored_return_value is None


def test_run_sample_uses_result_synchronize_when_sync_callback_is_absent(monkeypatch) -> None:
    events: list[str] = []
    timer_values = iter([20.0, 20.05])
    sweep = Sweep(target=lambda: None, params={}, suite_name="result-sync-suite", warmup_iterations=0)

    class ResultWithSynchronize:
        def synchronize(self) -> None:
            events.append("result-sync")

    monkeypatch.setattr(core_module.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(core_module, "perf_counter", lambda: events.append("timer") or next(timer_values))
    monkeypatch.setattr(sweep, "_invoke_target", lambda configuration: events.append("target") or ResultWithSynchronize())

    elapsed, observation, stored_return_value = sweep._run_sample({}, None, 1, 1)

    assert events == ["gc", "timer", "target", "result-sync", "timer"]
    assert elapsed == pytest.approx(0.05)
    assert observation == {"sample": 1, "records": []}
    assert stored_return_value is None


def test_sweep_run_computes_summary_metrics_from_sample_times(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    metadata_marker = object()
    sweep = Sweep(
        target=lambda: 99.0,
        params={},
        suite_name="timing-summary-suite",
        samples=3,
        warmup_iterations=0,
        lock_cpu_affinity=False,
        database_path=database_path,
        store_target_return_value=True,
    )
    sample_results = iter(
        [
            (0.30, {"sample": 1, "records": []}, 11.0),
            (0.10, {"sample": 2, "records": []}, 12.0),
            (0.20, {"sample": 3, "records": []}, 13.0),
        ]
    )

    monkeypatch.setattr(core_module, "prepare_system", lambda lock_cpu_affinity=True: None)
    monkeypatch.setattr(core_module, "collect_environment_metadata", lambda: metadata_marker)
    monkeypatch.setattr(core_module, "metadata_to_dict", lambda metadata: environment_payload)
    monkeypatch.setattr(sweep, "_run_sample", lambda configuration, reporter, sample_index, sample_total: next(sample_results))

    results = sweep.run()
    run = get_run_details((1, 1), database_path)

    assert len(results) == 1
    assert results[0].samples == [0.3, 0.1, 0.2]
    assert results[0].median_seconds == pytest.approx(0.2)
    assert results[0].min_seconds == pytest.approx(0.1)
    assert results[0].max_seconds == pytest.approx(0.3)
    assert results[0].std_seconds == pytest.approx(0.1)
    assert results[0].target_return_value == pytest.approx(11.0)

    assert run is not None
    assert run["samples"] == [0.3, 0.1, 0.2]
    assert run["median_seconds"] == pytest.approx(0.2)
    assert run["min_seconds"] == pytest.approx(0.1)
    assert run["max_seconds"] == pytest.approx(0.3)
    assert run["std_seconds"] == pytest.approx(0.1)
    assert run["target_return_value"] == pytest.approx(11.0)


def test_sweep_warmup_calls_are_not_persisted_as_samples(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    metadata_marker = object()
    sweep = Sweep(
        target=lambda: 0.0,
        params={},
        suite_name="warmup-isolation-suite",
        samples=2,
        warmup_iterations=2,
        lock_cpu_affinity=False,
        database_path=database_path,
        store_target_return_value=True,
    )
    events: list[str] = []
    sample_results = iter(
        [
            (0.21, {"sample": 1, "records": [{"label": "measured", "duration_seconds": 0.05}]}, 11.0),
            (0.19, {"sample": 2, "records": [{"label": "measured", "duration_seconds": 0.04}]}, 12.0),
        ]
    )

    monkeypatch.setattr(core_module, "prepare_system", lambda lock_cpu_affinity=True: None)
    monkeypatch.setattr(core_module, "collect_environment_metadata", lambda: metadata_marker)
    monkeypatch.setattr(core_module, "metadata_to_dict", lambda metadata: environment_payload)
    monkeypatch.setattr(sweep, "_invoke_target", lambda configuration: events.append("warmup-target") or object())
    monkeypatch.setattr(sweep, "_sync_if_needed", lambda result: events.append("warmup-sync"))
    monkeypatch.setattr(
        sweep,
        "_run_sample",
        lambda configuration, reporter, sample_index, sample_total: events.append(f"sample-{sample_index}") or next(sample_results),
    )

    results = sweep.run()
    run = get_run_details((1, 1), database_path)

    assert events == ["warmup-target", "warmup-sync", "warmup-target", "warmup-sync", "sample-1", "sample-2"]
    assert len(results) == 1
    assert results[0].samples == [0.21, 0.19]
    assert results[0].target_return_value == pytest.approx(11.0)
    assert results[0].observations == [
        {"sample": 1, "records": [{"label": "measured", "duration_seconds": 0.05}]},
        {"sample": 2, "records": [{"label": "measured", "duration_seconds": 0.04}]},
    ]

    assert run is not None
    assert run["samples"] == [0.21, 0.19]
    assert run["target_return_value"] == pytest.approx(11.0)
    assert run["observations"] == [
        {"sample": 1, "records": [{"label": "measured", "duration_seconds": 0.05}]},
        {"sample": 2, "records": [{"label": "measured", "duration_seconds": 0.04}]},
    ]


def test_collect_observations_records_exceptions_and_resets_context(monkeypatch) -> None:
    timer_values = iter([1.0, 1.25, 2.0, 2.1])

    @observe("fragile")
    def fragile_step() -> None:
        raise RuntimeError("boom")

    @observe("healthy")
    def healthy_step() -> str:
        return "ok"

    monkeypatch.setattr(observability_module, "perf_counter", lambda: next(timer_values))

    with pytest.raises(RuntimeError, match="boom"), collect_observations() as collector:
        fragile_step()

    assert len(collector.records) == 1
    assert collector.records[0]["label"] == "fragile"
    assert collector.records[0]["duration_seconds"] == pytest.approx(0.25)
    assert observability_module._ACTIVE_COLLECTOR.get() is None
    assert observability_module._BENCH_ACTIVE.get() is False

    with collect_observations() as next_collector:
        assert healthy_step() == "ok"

    assert next_collector.records == [{"label": "healthy", "duration_seconds": pytest.approx(0.1)}]


def test_observe_does_not_leak_records_across_threads() -> None:
    @observe("threaded")
    def observed_step() -> None:
        return None

    with collect_observations() as collector:
        observed_step()
        worker = threading.Thread(target=observed_step)
        worker.start()
        worker.join()

    assert [record["label"] for record in collector.records] == ["threaded"]


def test_database_initialization_runs_once(tmp_path: Path, monkeypatch) -> None:
    database_path = tmp_path / "benchcaddy.db"
    create_all_calls: list[object] = []

    def record_create_all(engine) -> None:
        create_all_calls.append(engine)

    monkeypatch.setattr(
        "benchcaddy.db.Base.metadata.create_all",
        record_create_all,
    )

    initialize_database(database_path)
    initialize_database(database_path)

    with db_session(database_path) as session:
        assert session is not None

    assert len(create_all_calls) == 1


def test_compare_suite_runs_can_filter_to_matching_reference_config(
    tmp_path: Path,
    environment_payload: dict[str, object],
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"

    for configuration, median_seconds in [
        ({"size": 33, "variant": "baseline"}, 0.10),
        ({"size": 33, "variant": "candidate"}, 0.12),
        ({"size": 34, "variant": "candidate"}, 0.09),
        ({"size": 33, "variant": "candidate", "mode": "extra"}, 0.11),
    ]:
        record_simple_run(
            suite_name="strict-suite",
            database_path=database_path,
            configuration=configuration,
            median_seconds=median_seconds,
            environment=environment_payload,
        )

    comparison = compare_suite_runs("strict-suite", (2, 1), ["size"], database_path)

    assert comparison is not None
    assert comparison["strict_keys"] == ["size"]
    assert comparison["strict_config"] == {"size": 33}
    assert [run["display_id"] for run in comparison["runs"]] == ["4.1", "2.1", "1.1"]

    stricter_comparison = compare_suite_runs("strict-suite", (2, 1), ["size", "variant"], database_path)

    assert stricter_comparison is not None
    assert stricter_comparison["strict_config"] == {"size": 33, "variant": "candidate"}
    assert [run["display_id"] for run in stricter_comparison["runs"]] == ["4.1", "2.1"]


def test_compare_suite_runs_uses_explicit_reference_for_basis_and_relative_metrics(
    tmp_path: Path,
    environment_payload: dict[str, object],
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"

    record_simple_run(
        suite_name="reference-suite",
        database_path=database_path,
        configuration={"variant": "best"},
        median_seconds=0.10,
        target_return_value=10.0,
        environment=environment_payload,
    )
    record_simple_run(
        suite_name="reference-suite",
        database_path=database_path,
        configuration={"variant": "reference"},
        median_seconds=0.15,
        target_return_value=20.0,
        environment=environment_payload,
    )
    record_simple_run(
        suite_name="reference-suite",
        database_path=database_path,
        configuration={"variant": "slower"},
        median_seconds=0.30,
        target_return_value=25.0,
        environment=environment_payload,
    )

    comparison = compare_suite_runs("reference-suite", (2, 1), database_path=database_path)

    assert comparison is not None
    assert comparison["basis_source"] == "reference"
    assert comparison["basis_metric_label"] == "Reference Median (s)"
    assert comparison["delta_column_label"] == "Delta vs Reference (s)"
    assert comparison["ratio_column_label"] == "Relative"
    assert comparison["basis_run"]["display_id"] == "2.1"
    assert comparison["basis_run"]["target_return_value"] == pytest.approx(20.0)

    rows_by_id = {row["display_id"]: row for row in comparison["runs"]}
    assert set(rows_by_id) == {"1.1", "2.1", "3.1"}

    assert rows_by_id["2.1"]["delta_seconds"] == pytest.approx(0.0)
    assert rows_by_id["2.1"]["slowdown_factor"] == pytest.approx(1.0)
    assert rows_by_id["2.1"]["target_return_relative_error"] == pytest.approx(0.0)

    assert rows_by_id["1.1"]["delta_seconds"] == pytest.approx(-0.05)
    assert rows_by_id["1.1"]["slowdown_factor"] == pytest.approx(2.0 / 3.0)
    assert rows_by_id["1.1"]["target_return_relative_error"] == pytest.approx(0.5)

    assert rows_by_id["3.1"]["delta_seconds"] == pytest.approx(0.15)
    assert rows_by_id["3.1"]["slowdown_factor"] == pytest.approx(2.0)
    assert rows_by_id["3.1"]["target_return_relative_error"] == pytest.approx(0.25)


def test_concurrent_first_writes_share_one_suite_and_keep_compare_consistent(
    tmp_path: Path,
    environment_payload: dict[str, object],
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    barrier = threading.Barrier(3)
    errors: list[Exception] = []
    recorded_display_ids: list[str] = []
    lock = threading.Lock()

    def worker(*, variant: str, median_seconds: float, target_return_value: float) -> None:
        try:
            barrier.wait()
            run = record_simple_run(
                suite_name="concurrent-suite",
                database_path=database_path,
                configuration={"variant": variant},
                median_seconds=median_seconds,
                target_return_value=target_return_value,
                environment=environment_payload,
            )
            with lock:
                recorded_display_ids.append(run.display_id)
        except Exception as error:  # pragma: no cover - exercised only on failure
            with lock:
                errors.append(error)

    threads = [
        threading.Thread(target=worker, kwargs={"variant": "baseline", "median_seconds": 0.10, "target_return_value": 10.0}),
        threading.Thread(target=worker, kwargs={"variant": "candidate", "median_seconds": 0.12, "target_return_value": 12.0}),
    ]

    for thread in threads:
        thread.start()

    barrier.wait()

    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert errors == []
    assert len(recorded_display_ids) == 2
    assert len(set(recorded_display_ids)) == 2

    details = get_suite_details("concurrent-suite", database_path)
    comparison = compare_suite_runs("concurrent-suite", database_path=database_path)

    assert details is not None
    assert len(details["runs"]) == 2
    assert {run["configuration"]["variant"] for run in details["runs"]} == {"baseline", "candidate"}
    assert {run["target_return_value"] for run in details["runs"]} == {10.0, 12.0}

    assert comparison is not None
    assert comparison["basis_source"] == "best"
    assert comparison["basis_run"]["median_seconds"] == pytest.approx(0.10)
    assert len(comparison["runs"]) == 2


def test_concurrent_writes_update_single_suite_summary_consistently(
    tmp_path: Path,
    environment_payload: dict[str, object],
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    barrier = threading.Barrier(3)
    errors: list[Exception] = []
    lock = threading.Lock()

    record_simple_run(
        suite_name="summary-concurrent-suite",
        database_path=database_path,
        configuration={"variant": "seed"},
        median_seconds=0.09,
        target_return_value=9.0,
        environment=environment_payload,
    )

    def worker(*, variant: str, median_seconds: float) -> None:
        try:
            barrier.wait()
            record_simple_run(
                suite_name="summary-concurrent-suite",
                database_path=database_path,
                configuration={"variant": variant},
                median_seconds=median_seconds,
                target_return_value=median_seconds * 100.0,
                environment=environment_payload,
            )
        except Exception as error:  # pragma: no cover - exercised only on failure
            with lock:
                errors.append(error)

    threads = [
        threading.Thread(target=worker, kwargs={"variant": "baseline", "median_seconds": 0.10}),
        threading.Thread(target=worker, kwargs={"variant": "candidate", "median_seconds": 0.12}),
    ]

    for thread in threads:
        thread.start()

    barrier.wait()

    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert errors == []

    summaries = list_suite_summaries(database_path)
    details = get_suite_details("summary-concurrent-suite", database_path)

    assert summaries == [
        {
            "suite_name": "summary-concurrent-suite",
            "target_name": "benchmark_target",
            "run_count": 3,
            "last_run_at": summaries[0]["last_run_at"],
            "observation_labels": [],
        }
    ]
    assert details is not None
    assert len(details["runs"]) == 3
    assert {run["configuration"]["variant"] for run in details["runs"]} == {"seed", "baseline", "candidate"}


def test_run_prefers_iterations_and_warmup_runs_over_default_counts(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    metadata_marker = object()
    sweep = Sweep(
        target=lambda: object(),
        params={},
        suite_name="count-override-suite",
        samples=5,
        iterations=2,
        warmup_iterations=4,
        warmup_runs=1,
        lock_cpu_affinity=False,
        database_path=database_path,
    )
    events: list[str] = []
    sample_results = iter(
        [
            (0.11, {"sample": 1, "records": []}, None),
            (0.12, {"sample": 2, "records": []}, None),
        ]
    )

    monkeypatch.setattr(core_module, "prepare_system", lambda lock_cpu_affinity=True: None)
    monkeypatch.setattr(core_module, "collect_environment_metadata", lambda: metadata_marker)
    monkeypatch.setattr(core_module, "metadata_to_dict", lambda metadata: environment_payload)
    monkeypatch.setattr(sweep, "_invoke_target", lambda configuration: events.append("warmup-target") or object())
    monkeypatch.setattr(sweep, "_sync_if_needed", lambda result: events.append("warmup-sync"))
    monkeypatch.setattr(
        sweep,
        "_run_sample",
        lambda configuration, reporter, sample_index, sample_total: events.append(f"sample-{sample_index}-of-{sample_total}") or next(sample_results),
    )

    results = sweep.run()

    assert events == ["warmup-target", "warmup-sync", "sample-1-of-2", "sample-2-of-2"]
    assert len(results) == 1
    assert results[0].samples == [0.11, 0.12]


def test_run_sync_argument_overrides_result_synchronize(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    metadata_marker = object()
    events: list[str] = []

    class ResultWithSynchronize:
        def synchronize(self) -> None:
            events.append("result-sync")

    def target() -> ResultWithSynchronize:
        events.append("target")
        return ResultWithSynchronize()

    monkeypatch.setattr(core_module, "prepare_system", lambda lock_cpu_affinity=True: None)
    monkeypatch.setattr(core_module, "collect_environment_metadata", lambda: metadata_marker)
    monkeypatch.setattr(core_module, "metadata_to_dict", lambda metadata: environment_payload)

    results = Sweep(
        target=target,
        params={},
        suite_name="sync-override-suite",
        samples=1,
        warmup_iterations=0,
        lock_cpu_affinity=False,
        database_path=database_path,
    ).run(sync=lambda: events.append("explicit-sync"))

    assert len(results) == 1
    assert events.count("explicit-sync") == 1
    assert "result-sync" not in events


def test_sweep_does_not_persist_partial_run_when_sample_fails(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    metadata_marker = object()
    sweep = Sweep(
        target=lambda: 1.0,
        params={"variant": ["broken"]},
        suite_name="failure-isolation-suite",
        samples=2,
        warmup_iterations=0,
        lock_cpu_affinity=False,
        database_path=database_path,
    )

    monkeypatch.setattr(core_module, "prepare_system", lambda lock_cpu_affinity=True: None)
    monkeypatch.setattr(core_module, "collect_environment_metadata", lambda: metadata_marker)
    monkeypatch.setattr(core_module, "metadata_to_dict", lambda metadata: environment_payload)

    def failing_run_sample(configuration, reporter, sample_index, sample_total):
        if sample_index == 1:
            return (0.10, {"sample": 1, "records": []}, None)
        raise RuntimeError("sample failed")

    monkeypatch.setattr(sweep, "_run_sample", failing_run_sample)

    with pytest.raises(RuntimeError, match="sample failed"):
        sweep.run()

    assert get_suite_details("failure-isolation-suite", database_path) is not None
    assert get_suite_details("failure-isolation-suite", database_path)["runs"] == []
    assert list_suite_summaries(database_path) == []


def test_sweep_keeps_first_return_value_when_later_samples_differ(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    metadata_marker = object()
    sweep = Sweep(
        target=lambda: 0.0,
        params={},
        suite_name="first-return-suite",
        samples=3,
        warmup_iterations=0,
        lock_cpu_affinity=False,
        database_path=database_path,
        store_target_return_value=True,
    )
    sample_results = iter(
        [
            (0.10, {"sample": 1, "records": []}, 11.0),
            (0.11, {"sample": 2, "records": []}, 22.0),
            (0.12, {"sample": 3, "records": []}, 33.0),
        ]
    )

    monkeypatch.setattr(core_module, "prepare_system", lambda lock_cpu_affinity=True: None)
    monkeypatch.setattr(core_module, "collect_environment_metadata", lambda: metadata_marker)
    monkeypatch.setattr(core_module, "metadata_to_dict", lambda metadata: environment_payload)
    monkeypatch.setattr(sweep, "_run_sample", lambda configuration, reporter, sample_index, sample_total: next(sample_results))

    results = sweep.run()
    run = get_run_details((1, 1), database_path)

    assert results[0].target_return_value == pytest.approx(11.0)
    assert run is not None
    assert run["target_return_value"] == pytest.approx(11.0)
