from __future__ import annotations

from pathlib import Path

import benchcaddy.core as core_module
from benchcaddy import Sweep, observe
from benchcaddy.db import compare_runs, db_session, get_run_details, initialize_database
from benchcaddy.db import get_suite_details, list_suite_summaries


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

    monkeypatch.setattr(core_module.psutil, "Process", process_factory)
    monkeypatch.setattr(core_module.gc, "collect", lambda: None)
    monkeypatch.setattr(core_module.gc, "freeze", lambda: None)
    monkeypatch.setattr(core_module.os, "name", "nt")
    monkeypatch.setattr(core_module.psutil, "HIGH_PRIORITY_CLASS", 128, raising=False)

    core_module.prepare_system(lock_cpu_affinity=True)

    assert recorded_affinity == [2, 4]


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