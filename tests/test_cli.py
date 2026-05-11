from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import benchcaddy.cli as cli_module
import benchcaddy.core as core_module
import benchcaddy.db as db_module
from benchcaddy import Sweep
from benchcaddy.cli import _suite_row_style, _trend_row_style, app
from benchcaddy.db import compare_runs, get_run_details, get_suite_details, record_benchmark_run
from benchcaddy.presentation import format_return_error, format_return_value


def _uniform_run_kwargs(
    median_seconds: float,
    *,
    target_return_value: bool | int | float | str | list[float] | None = None,
) -> dict[str, object]:
    return {
        "median_seconds": median_seconds,
        "min_seconds": median_seconds,
        "max_seconds": median_seconds,
        "std_seconds": 0.0,
        "target_return_value": target_return_value,
    }


def _seed_run(
    *,
    database_path: Path,
    suite_name: str,
    configuration: dict[str, object],
    median_seconds: float,
    environment_payload: dict[str, object],
    target_return_value: bool | int | float | str | list[float] | None = None,
) -> None:
    record_benchmark_run(
        suite_name=suite_name,
        target_name="benchmark_target",
        configuration=configuration,
        samples=[median_seconds, median_seconds],
        observations=[
            {
                "sample": 1,
                "records": [
                    {"label": "inner", "duration_seconds": median_seconds / 4},
                    {"label": "outer", "duration_seconds": median_seconds / 2},
                ],
            },
            {
                "sample": 2,
                "records": [
                    {"label": "inner", "duration_seconds": median_seconds / 5},
                    {"label": "outer", "duration_seconds": median_seconds / 3},
                ],
            },
        ],
        **_uniform_run_kwargs(median_seconds, target_return_value=target_return_value),
        environment=environment_payload,
        database_path=database_path,
    )


def _seed_custom_run(
    *,
    database_path: Path,
    suite_name: str,
    configuration: dict[str, object],
    median_seconds: float,
    observations: list[dict[str, object]],
    environment_payload: dict[str, object],
    target_return_value: bool | int | float | str | list[float] | None = None,
) -> None:
    record_benchmark_run(
        suite_name=suite_name,
        target_name="benchmark_target",
        configuration=configuration,
        samples=[median_seconds, median_seconds],
        observations=observations,
        **_uniform_run_kwargs(median_seconds, target_return_value=target_return_value),
        environment=environment_payload,
        database_path=database_path,
    )


def _seed_sampled_run(
    *,
    database_path: Path,
    suite_name: str,
    configuration: dict[str, object],
    samples: list[float],
    environment_payload: dict[str, object],
) -> None:
    record_benchmark_run(
        suite_name=suite_name,
        target_name="benchmark_target",
        configuration=configuration,
        samples=samples,
        observations=[],
        median_seconds=sorted(samples)[len(samples) // 2],
        min_seconds=min(samples),
        max_seconds=max(samples),
        std_seconds=0.0,
        environment=environment_payload,
        database_path=database_path,
    )


def _compact_output(output: str) -> str:
    return "".join(output.split())


def _raise_if_analysis_called(*args, **kwargs):
    raise AssertionError("unexpected analysis")


def _stub_sweep_runtime(monkeypatch, environment_payload: dict[str, object]) -> None:
    metadata_marker = object()
    monkeypatch.setattr(core_module, "prepare_system", lambda lock_cpu_affinity=True: None)
    monkeypatch.setattr(core_module, "collect_environment_metadata", lambda: metadata_marker)
    monkeypatch.setattr(core_module, "metadata_to_dict", lambda metadata: environment_payload)


def test_cli_lists_shows_and_compares_runs(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="nonlinear-transform",
        configuration={"size": 512, "variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="nonlinear-transform",
        configuration={"size": 512, "variant": "stabilized"},
        median_seconds=0.150,
        environment_payload=environment_payload,
    )

    list_result = runner.invoke(app, ["list", "--database", str(database_path)])
    assert list_result.exit_code == 0
    assert "nonlinear" in list_result.stdout
    assert "benchmark" in list_result.stdout
    assert "Observation" in list_result.stdout
    assert "Labels" in list_result.stdout
    assert "inner, outer" in list_result.stdout

    show_result = runner.invoke(
        app,
        ["show", "nonlinear-transform", "--database", str(database_path)],
    )
    assert show_result.exit_code == 0
    assert "Suite: nonlinear-transform" in show_result.stdout
    assert "Mean +- Std (s)" in show_result.stdout
    assert "1.1" in show_result.stdout
    assert "2.1" in show_result.stdout
    assert "size: 512" in show_result.stdout
    assert "variant:" in show_result.stdout
    assert "Observed Timings: nonlinear-transform" in show_result.stdout
    assert "inner" in show_result.stdout
    assert "outer" in show_result.stdout
    assert "Environment" in show_result.stdout

    compare_result = runner.invoke(
        app,
        ["compare", "nonlinear-transform", "--database", str(database_path)],
    )
    assert compare_result.exit_code == 0
    assert "Comparison: nonlinear-transform" in compare_result.stdout
    assert "Delta" in compare_result.stdout
    assert "Best" in compare_result.stdout
    assert "Mean +- Std (s)" in compare_result.stdout
    assert "1.1" in compare_result.stdout
    assert "2.1" in compare_result.stdout

    suite_reference_compare_result = runner.invoke(
        app,
        ["compare", "nonlinear-transform", "2.1", "--database", str(database_path)],
    )
    assert suite_reference_compare_result.exit_code == 0
    assert "Comparison: nonlinear-transform" in suite_reference_compare_result.stdout
    assert "Comparison Basis" in suite_reference_compare_result.stdout
    assert "Reference Median (s)" in suite_reference_compare_result.stdout
    assert "Reference" in suite_reference_compare_result.stdout
    assert "2.1" in suite_reference_compare_result.stdout
    assert "Best Run vs Reference" in suite_reference_compare_result.stdout
    assert "Scope" in suite_reference_compare_result.stdout
    assert "Improvement Probability" in suite_reference_compare_result.stdout
    assert "p-value" in suite_reference_compare_result.stdout

    suite_fastest_reference_result = runner.invoke(
        app,
        ["compare", "nonlinear-transform", "1.1", "--database", str(database_path)],
    )
    assert suite_fastest_reference_result.exit_code == 0
    assert "Best Run vs Reference" in suite_fastest_reference_result.stdout
    assert "already the fastest run" in suite_fastest_reference_result.stdout
    assert "comparison" in suite_fastest_reference_result.stdout
    assert "scope" in suite_fastest_reference_result.stdout
    assert "Scope" in suite_fastest_reference_result.stdout
    assert "full suite" in suite_fastest_reference_result.stdout

    verbose_compare_result = runner.invoke(
        app,
        ["--verbose", "compare", "nonlinear-transform", "--database", str(database_path)],
    )
    assert verbose_compare_result.exit_code == 0
    assert verbose_compare_result.stdout != compare_result.stdout
    assert "Comparison Basis" in verbose_compare_result.stdout
    assert "Best Median (s)" in verbose_compare_result.stdout
    assert "Run ID" in verbose_compare_result.stdout
    assert "Record ID" in verbose_compare_result.stdout
    assert "1.1" in verbose_compare_result.stdout
    assert "Mean +- Std (s)" in verbose_compare_result.stdout

    run_show_result = runner.invoke(app, ["show", "1.1", "--database", str(database_path)])
    assert run_show_result.exit_code == 0
    assert "Run: 1.1" in run_show_result.stdout
    assert "Mean +- Std (s)" in run_show_result.stdout
    assert "Min (s)" in run_show_result.stdout
    assert "Max (s)" in run_show_result.stdout
    assert "Observed Timings" in run_show_result.stdout
    assert "inner" in run_show_result.stdout
    assert "outer" in run_show_result.stdout

    run_compare_result = runner.invoke(
        app,
        ["compare", "1.1", "2.1", "--database", str(database_path)],
    )
    assert run_compare_result.exit_code == 0
    assert "Run Comparison: 1.1 -> 2.1" in run_compare_result.stdout
    assert "Median (s)" in run_compare_result.stdout
    assert "Median Delta (s)" in run_compare_result.stdout
    assert "Median Percent Change" in run_compare_result.stdout
    assert "inner" in run_compare_result.stdout
    assert "outer" in run_compare_result.stdout

    multi_show_result = runner.invoke(
        app,
        ["show", "1", "2.1", "--database", str(database_path)],
    )
    assert multi_show_result.exit_code == 0
    assert "Selected Runs" in multi_show_result.stdout
    assert "Observed Timings: Selected Runs" in multi_show_result.stdout
    assert multi_show_result.stdout.index("2.1") < multi_show_result.stdout.index("1.1")

    duplicate_multi_show_result = runner.invoke(
        app,
        ["show", "1", "2.1", "1", "--database", str(database_path)],
    )
    assert duplicate_multi_show_result.exit_code == 0
    assert "Selected Runs" in duplicate_multi_show_result.stdout
    assert "Observed Timings: Selected Runs" in duplicate_multi_show_result.stdout
    assert duplicate_multi_show_result.stdout.count("1.1") == multi_show_result.stdout.count("1.1")
    assert duplicate_multi_show_result.stdout.count("2.1") == multi_show_result.stdout.count("2.1")


def test_cli_show_and_compare_include_return_values_and_distance(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="return-compare",
        configuration={"variant": "baseline"},
        median_seconds=0.1,
        environment_payload=environment_payload,
        target_return_value=10,
    )
    _seed_run(
        database_path=database_path,
        suite_name="return-compare",
        configuration={"variant": "candidate"},
        median_seconds=0.11,
        environment_payload=environment_payload,
        target_return_value=13.5,
    )
    comparison = compare_runs((1, 1), (2, 1), database_path)

    assert comparison is not None
    expected_baseline_value = format_return_value(10, compact=True)
    expected_candidate_value = format_return_value(13.5, compact=True)
    expected_return_error = format_return_error(comparison["target_return_relative_error"])

    show_result = runner.invoke(app, ["show", "1.1", "--database", str(database_path)])
    assert show_result.exit_code == 0
    assert "Return Value" in show_result.stdout
    assert expected_baseline_value in show_result.stdout

    compare_result = runner.invoke(app, ["compare", "1.1", "2.1", "--database", str(database_path)])
    assert compare_result.exit_code == 0
    assert "Return Value" in compare_result.stdout
    assert "Return Error" in compare_result.stdout
    assert expected_baseline_value in compare_result.stdout
    assert expected_candidate_value in compare_result.stdout
    assert expected_return_error in compare_result.stdout

    suite_compare_result = runner.invoke(app, ["compare", "return-compare", "1.1", "--database", str(database_path)])
    assert suite_compare_result.exit_code == 0
    assert "Return Value" in suite_compare_result.stdout
    assert "Return Error" in suite_compare_result.stdout
    assert "Stored Return Metrics" in suite_compare_result.stdout
    assert "Comparison Basis" in suite_compare_result.stdout
    assert expected_baseline_value in suite_compare_result.stdout

    suite_best_compare_result = runner.invoke(app, ["compare", "return-compare", "--database", str(database_path)])
    assert suite_best_compare_result.exit_code == 0
    assert "Return Value" in suite_best_compare_result.stdout
    assert "Return Error" in suite_best_compare_result.stdout
    assert "Stored Return Metrics" in suite_best_compare_result.stdout
    assert "Comparison Basis" in suite_best_compare_result.stdout
    assert expected_baseline_value in suite_best_compare_result.stdout


def test_cli_compare_formats_vector_return_error(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="vector-return-compare",
        configuration={"variant": "baseline"},
        median_seconds=0.1,
        environment_payload=environment_payload,
        target_return_value=[1.0, 2.0, 3.0],
    )
    _seed_run(
        database_path=database_path,
        suite_name="vector-return-compare",
        configuration={"variant": "candidate"},
        median_seconds=0.11,
        environment_payload=environment_payload,
        target_return_value=[4.0, 6.0, 3.0],
    )
    comparison = compare_runs((1, 1), (2, 1), database_path)

    assert comparison is not None
    expected_baseline_value = _compact_output(format_return_value([1.0, 2.0, 3.0], compact=True))
    expected_candidate_value = _compact_output(format_return_value([4.0, 6.0, 3.0], compact=True))
    expected_return_error = _compact_output(format_return_error(comparison["target_return_relative_error"]))

    compare_result = runner.invoke(app, ["compare", "1.1", "2.1", "--database", str(database_path)])
    assert compare_result.exit_code == 0
    assert "Return Value" in compare_result.stdout
    compact_compare_output = _compact_output(compare_result.stdout)
    assert expected_baseline_value in compact_compare_output
    assert expected_candidate_value in compact_compare_output
    assert "ReturnError" in compact_compare_output
    assert expected_return_error in compact_compare_output

    suite_compare_result = runner.invoke(app, ["compare", "vector-return-compare", "--database", str(database_path)])
    assert suite_compare_result.exit_code == 0
    assert "Return Value" in suite_compare_result.stdout
    assert "Return Error" in suite_compare_result.stdout
    compact_suite_compare_output = _compact_output(suite_compare_result.stdout)
    assert "StoredReturnMetrics" in compact_suite_compare_output
    assert "ComparisonBasis" in compact_suite_compare_output
    assert expected_baseline_value in compact_suite_compare_output


def test_direct_sweep_persists_and_displays_return_values(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "example.db"
    runner = CliRunner()

    _stub_sweep_runtime(monkeypatch, environment_payload)

    def benchmark_target(variant: str) -> float:
        return 10.0 if variant == "baseline" else 13.5

    sweep = Sweep(
        target=benchmark_target,
        params={"variant": ["baseline", "candidate"]},
        suite_name="direct-return-value-suite",
        samples=1,
        warmup_iterations=0,
        lock_cpu_affinity=False,
        database_path=database_path,
        store_target_return_value=True,
    )
    sweep.run()

    baseline_run = get_run_details((1, 1), database_path)
    candidate_run = get_run_details((1, 2), database_path)
    comparison = compare_runs((1, 1), (1, 2), database_path)

    assert baseline_run is not None
    assert candidate_run is not None
    assert baseline_run["target_return_value"] == pytest.approx(10.0)
    assert candidate_run["target_return_value"] == pytest.approx(13.5)
    assert comparison is not None
    assert comparison["target_return_relative_error"] == pytest.approx(0.35)

    show_result = runner.invoke(app, ["show", "1.1", "--database", str(database_path)])
    assert show_result.exit_code == 0
    assert "Run: 1.1" in show_result.stdout
    assert "Return Value" in show_result.stdout

    compare_result = runner.invoke(app, ["compare", "1.1", "1.2", "--database", str(database_path)])
    assert compare_result.exit_code == 0
    assert "Return Value" in compare_result.stdout
    assert "Return Error" in compare_result.stdout
    assert "1.1" in compare_result.stdout
    assert "1.2" in compare_result.stdout


def test_show_without_arguments_lists_all_runs(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    monkeypatch.setattr(
        cli_module,
        "console",
        Console(force_terminal=False, color_system=None, width=160),
    )

    _seed_run(
        database_path=database_path,
        suite_name="suite-a",
        configuration={"size": 33, "variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
        target_return_value=1.0,
    )
    _seed_run(
        database_path=database_path,
        suite_name="suite-b",
        configuration={"size": 34, "variant": "candidate"},
        median_seconds=0.120,
        environment_payload=environment_payload,
        target_return_value=2.0,
    )

    show_result = runner.invoke(app, ["show", "--database", str(database_path)])

    assert show_result.exit_code == 0
    assert "All Runs" in show_result.stdout
    assert "Run ID" in show_result.stdout
    assert "Record ID" in show_result.stdout
    assert "Suite" in show_result.stdout
    assert "Configuration" in show_result.stdout
    assert "Mean +- Std (s)" in show_result.stdout
    assert "Return Value" in show_result.stdout
    assert "Samples" in show_result.stdout
    assert "Recorded At" in show_result.stdout
    assert "2.1" in show_result.stdout
    assert "1.1" in show_result.stdout
    assert "suite-a" in show_result.stdout
    assert "suite-b" in show_result.stdout
    assert "candidate" in show_result.stdout
    assert "baseline" in show_result.stdout


def test_show_without_arguments_skips_statistical_analysis(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="suite-a",
        configuration={"size": 33, "variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )

    monkeypatch.setattr(
        db_module,
        "analyze_samples",
        _raise_if_analysis_called,
    )

    show_result = runner.invoke(app, ["show", "--database", str(database_path)])

    assert show_result.exit_code == 0
    assert "All Runs" in show_result.stdout


def test_show_run_supports_no_stats_fast_path(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="suite-a",
        configuration={"size": 33, "variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )

    monkeypatch.setattr(
        db_module,
        "analyze_samples",
        _raise_if_analysis_called,
    )

    show_result = runner.invoke(app, ["show", "1.1", "--no-stats", "--database", str(database_path)])

    assert show_result.exit_code == 0
    assert "Run: 1.1" in show_result.stdout
    assert "Statistical Summary" not in show_result.stdout
    assert "Median CI (s)" not in show_result.stdout


@pytest.mark.parametrize(
    ("suite_name", "target", "expected_return_value"),
    [
        ("return-type-bool", lambda: True, True),
        ("return-type-float-vector", lambda: [2.0, 1.0, 0.5], [2.0, 1.0, 0.5]),
        ("return-type-bool-vector", lambda: (True, False, True), [1.0, 0.0, 1.0]),
    ],
)
def test_direct_sweep_supports_return_value_types(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
    suite_name: str,
    target,
    expected_return_value: bool | list[float],
) -> None:
    database_path = tmp_path / "return-types.db"

    _stub_sweep_runtime(monkeypatch, environment_payload)

    Sweep(
        target=target,
        params={},
        suite_name=suite_name,
        samples=1,
        warmup_iterations=0,
        lock_cpu_affinity=False,
        database_path=database_path,
        store_target_return_value=True,
    ).run()

    details = get_suite_details(suite_name, database_path)

    assert details is not None
    assert details["runs"][-1]["target_return_value"] == expected_return_value


def test_cli_show_renders_partial_git_environment(tmp_path: Path, environment_payload: dict[str, object]) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()
    partial_git_environment = {
        **environment_payload,
        "git": {
            "branch": None,
            "commit_hash": "cafebabe1234",
            "dirty": False,
        },
    }

    _seed_custom_run(
        suite_name="partial-git-suite",
        database_path=database_path,
        configuration={"variant": "detached-head"},
        median_seconds=0.1,
        observations=[],
        environment_payload=partial_git_environment,
    )

    show_result = runner.invoke(app, ["show", "1.1", "--database", str(database_path)])

    assert show_result.exit_code == 0
    assert "Environment" in show_result.stdout
    assert "git" in show_result.stdout
    assert "branch" in show_result.stdout
    assert "commit_hash" in show_result.stdout
    assert "cafebabe1234" in show_result.stdout
    assert "dirty" in show_result.stdout
    assert "False" in show_result.stdout


def test_cli_renders_missing_min_max_as_dash(tmp_path: Path, environment_payload: dict[str, object]) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    record_benchmark_run(
        suite_name="nullable-times",
        target_name="benchmark_target",
        configuration={"variant": "baseline"},
        samples=[0.1, 0.2],
        observations=[],
        median_seconds=0.15,
        min_seconds=None,
        max_seconds=None,
        std_seconds=0.01,
        environment=environment_payload,
        database_path=database_path,
    )
    record_benchmark_run(
        suite_name="nullable-times",
        target_name="benchmark_target",
        configuration={"variant": "candidate"},
        samples=[0.3, 0.4],
        observations=[],
        median_seconds=0.35,
        min_seconds=None,
        max_seconds=None,
        std_seconds=0.01,
        environment=environment_payload,
        database_path=database_path,
    )

    show_result = runner.invoke(app, ["show", "1.1", "--database", str(database_path)])
    assert show_result.exit_code == 0
    assert "Min (s)" in show_result.stdout
    assert "Max (s)" in show_result.stdout
    assert "-" in show_result.stdout

    compare_result = runner.invoke(app, ["compare", "1.1", "2.1", "--database", str(database_path)])
    assert compare_result.exit_code == 0
    assert "Run Comparison: 1.1 -> 2.1" in compare_result.stdout
    assert "Suite" in compare_result.stdout
    assert "nullable-times" in compare_result.stdout
    assert "Min (s)" in compare_result.stdout
    assert "Max (s)" in compare_result.stdout
    assert "-" in compare_result.stdout


def test_run_compare_marks_missing_observations(tmp_path: Path, environment_payload: dict[str, object]) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_custom_run(
        database_path=database_path,
        suite_name="obs-diff-suite",
        configuration={"variant": "baseline"},
        median_seconds=0.1,
        observations=[
            {"sample": 1, "records": [{"label": "shared", "duration_seconds": 0.02}, {"label": "baseline_only", "duration_seconds": 0.03}]},
            {"sample": 2, "records": [{"label": "shared", "duration_seconds": 0.02}, {"label": "baseline_only", "duration_seconds": 0.03}]},
        ],
        environment_payload=environment_payload,
    )
    _seed_custom_run(
        database_path=database_path,
        suite_name="obs-diff-suite",
        configuration={"variant": "candidate"},
        median_seconds=0.12,
        observations=[
            {"sample": 1, "records": [{"label": "shared", "duration_seconds": 0.025}, {"label": "candidate_only", "duration_seconds": 0.04}]},
            {"sample": 2, "records": [{"label": "shared", "duration_seconds": 0.025}, {"label": "candidate_only", "duration_seconds": 0.04}]},
        ],
        environment_payload=environment_payload,
    )

    compare_result = runner.invoke(app, ["compare", "1.1", "2.1", "--database", str(database_path)])

    assert compare_result.exit_code == 0
    assert "baseline_only" in compare_result.stdout
    assert "candidate_only" in compare_result.stdout
    assert "-" in compare_result.stdout


def test_cli_shows_observation_std_for_selected_runs_and_compare(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_custom_run(
        database_path=database_path,
        suite_name="obs-std-suite",
        configuration={"variant": "baseline"},
        median_seconds=0.1,
        observations=[
            {"sample": 1, "records": [{"label": "shared", "duration_seconds": 0.01}]},
            {"sample": 2, "records": [{"label": "shared", "duration_seconds": 0.03}]},
        ],
        environment_payload=environment_payload,
    )
    _seed_custom_run(
        database_path=database_path,
        suite_name="obs-std-suite",
        configuration={"variant": "candidate"},
        median_seconds=0.12,
        observations=[
            {"sample": 1, "records": [{"label": "shared", "duration_seconds": 0.02}]},
            {"sample": 2, "records": [{"label": "shared", "duration_seconds": 0.04}]},
        ],
        environment_payload=environment_payload,
    )

    show_result = runner.invoke(app, ["show", "1.1", "2.1", "--database", str(database_path)])
    assert show_result.exit_code == 0
    assert "Selected Runs" in show_result.stdout
    assert "Observed Timings: Selected Runs" in show_result.stdout
    assert "shared" in show_result.stdout
    assert "1.1" in show_result.stdout
    assert "2.1" in show_result.stdout

    compare_result = runner.invoke(app, ["compare", "1.1", "2.1", "--database", str(database_path)])
    assert compare_result.exit_code == 0
    assert "Run Comparison: 1.1 -> 2.1" in compare_result.stdout
    assert "shared" in compare_result.stdout
    assert "Baseline" in compare_result.stdout
    assert "Candidate" in compare_result.stdout


def test_suite_compare_rejects_reference_run_from_other_suite(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="suite-a",
        configuration={"variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="suite-b",
        configuration={"variant": "candidate"},
        median_seconds=0.120,
        environment_payload=environment_payload,
    )

    result = runner.invoke(app, ["compare", "suite-b", "1", "--database", str(database_path)])

    assert result.exit_code == 1
    assert "belongs to suite 'suite-a', not 'suite-b'" in result.stdout


@pytest.mark.parametrize(
    ("extra_args", "expected_title"),
    [
        (["--strict", "size", "variant"], "Comparison: nonlinear-transform (strict: size, variant)"),
        (["--strict"], "Comparison: nonlinear-transform (strict: size, variant)"),
    ],
)
def test_cli_strict_suite_compare_filters_reference_configuration(
    tmp_path: Path,
    environment_payload: dict[str, object],
    extra_args: list[str],
    expected_title: str,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="nonlinear-transform",
        configuration={"size": 33, "variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="nonlinear-transform",
        configuration={"size": 33, "variant": "candidate"},
        median_seconds=0.120,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="nonlinear-transform",
        configuration={"size": 34, "variant": "candidate"},
        median_seconds=0.090,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="nonlinear-transform",
        configuration={"size": 33, "variant": "candidate", "mode": "extra"},
        median_seconds=0.110,
        environment_payload=environment_payload,
    )

    result = runner.invoke(
        app,
        ["compare", "nonlinear-transform", "2.1", *extra_args, "--database", str(database_path)],
    )

    assert result.exit_code == 0
    assert expected_title in result.stdout
    assert "2.1" in result.stdout
    assert "4.1" in result.stdout
    assert "1.1" not in result.stdout
    assert "3.1" not in result.stdout


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--strict", "size"],
        ["--strict"],
    ],
)
def test_cli_strict_suite_compare_requires_reference_run(
    tmp_path: Path,
    environment_payload: dict[str, object],
    extra_args: list[str],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="nonlinear-transform",
        configuration={"size": 33, "variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )

    result = runner.invoke(
        app,
        ["compare", "nonlinear-transform", *extra_args, "--database", str(database_path)],
    )

    assert result.exit_code == 2
    assert "--strict requires a suite comparison with a reference run ID." in result.stdout


def test_suite_compare_basis_matches_best_run_time_and_std(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    record_benchmark_run(
        suite_name="basis-suite",
        target_name="benchmark_target",
        configuration={"variant": "best"},
        samples=[0.08, 0.10, 0.20],
        observations=[],
        median_seconds=0.10,
        min_seconds=0.08,
        max_seconds=0.20,
        std_seconds=0.0642910059,
        environment=environment_payload,
        database_path=database_path,
    )
    record_benchmark_run(
        suite_name="basis-suite",
        target_name="benchmark_target",
        configuration={"variant": "slower"},
        samples=[0.20, 0.21, 0.22],
        observations=[],
        median_seconds=0.21,
        min_seconds=0.20,
        max_seconds=0.22,
        std_seconds=0.01,
        environment=environment_payload,
        database_path=database_path,
    )

    compare_result = runner.invoke(app, ["compare", "basis-suite", "--database", str(database_path)])

    assert compare_result.exit_code == 0
    assert "Run ID" in compare_result.stdout
    assert "Record ID" in compare_result.stdout
    assert "1.1" in compare_result.stdout
    assert "Best Median (s)" in compare_result.stdout
    assert "Mean +- Std (s)" in compare_result.stdout


def test_suite_row_style_uses_green_for_best_and_yellow_for_reference() -> None:
    comparison = {
        "basis_metric_label": "Reference Median (s)",
        "basis_run": {"id": 2, "median_seconds": 0.20},
        "runs": [
            {"id": 1, "median_seconds": 0.10},
            {"id": 2, "median_seconds": 0.20},
        ],
    }

    assert _suite_row_style(comparison, comparison["runs"][0]) == "green"
    assert _suite_row_style(comparison, comparison["runs"][1]) == "yellow"


def test_suite_row_style_keeps_reference_green_when_it_is_best() -> None:
    comparison = {
        "basis_metric_label": "Reference Median (s)",
        "basis_run": {"id": 2, "median_seconds": 0.10},
        "runs": [
            {"id": 1, "median_seconds": 0.12},
            {"id": 2, "median_seconds": 0.10},
        ],
    }

    assert _suite_row_style(comparison, comparison["runs"][1]) == "green"
    assert _suite_row_style(comparison, comparison["runs"][0]) is None


def test_trend_row_style_uses_green_for_best_and_yellow_for_anchor() -> None:
    trend = {
        "basis_run": {"id": 2, "median_seconds": 0.20},
        "runs": [
            {"id": 1, "median_seconds": 0.10},
            {"id": 2, "median_seconds": 0.20},
        ],
    }

    assert _trend_row_style(trend, trend["runs"][0]) == "green"
    assert _trend_row_style(trend, trend["runs"][1]) == "yellow"


def test_trend_row_style_keeps_anchor_green_when_it_is_best() -> None:
    trend = {
        "basis_run": {"id": 2, "median_seconds": 0.10},
        "runs": [
            {"id": 1, "median_seconds": 0.12},
            {"id": 2, "median_seconds": 0.10},
        ],
    }

    assert _trend_row_style(trend, trend["runs"][1]) == "green"
    assert _trend_row_style(trend, trend["runs"][0]) is None


def test_cli_compare_can_pin_and_use_baseline(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="baseline-suite",
        configuration={"variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="baseline-suite",
        configuration={"variant": "candidate"},
        median_seconds=0.120,
        environment_payload=environment_payload,
    )

    pin_result = runner.invoke(
        app,
        ["compare", "baseline-suite", "2.1", "--pin-baseline", "--database", str(database_path)],
    )

    assert pin_result.exit_code == 0
    assert "Baseline Updated" in pin_result.stdout
    assert "Pinned baseline for baseline-suite:" in pin_result.stdout
    assert "2.1" in pin_result.stdout

    use_result = runner.invoke(
        app,
        ["compare", "baseline-suite", "--use-baseline", "--database", str(database_path)],
    )

    assert use_result.exit_code == 0
    assert "Statistical Findings" in use_result.stdout
    assert "Basis Source" in use_result.stdout
    assert "pinned" in use_result.stdout
    assert "2.1" in use_result.stdout


def test_cli_verbose_trend_preserves_warning_categories(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_sampled_run(
        database_path=database_path,
        suite_name="warning-trend-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.100, 0.105, 0.110, 0.115, 0.500],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="warning-trend-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.200, 0.210, 0.205],
        environment_payload=environment_payload,
    )

    result = runner.invoke(
        app,
        ["--verbose", "trend", "warning-trend-suite", "1.1", "--database", str(database_path)],
    )

    assert result.exit_code == 0
    assert "baseline outliers detected" in result.stdout
    assert "candidate low sample count" in result.stdout


def test_cli_help_mentions_show_defaults_and_compare_modes() -> None:
    test_runner = CliRunner()

    show_result = test_runner.invoke(app, ["show", "--help"])
    compare_result = test_runner.invoke(app, ["compare", "--help"])

    assert show_result.exit_code == 0
    assert "Inspect all recorded runs, a suite, or specific run IDs." in show_result.stdout
    assert "Omit" in show_result.stdout
    assert "identifiers to list all recorded runs." in show_result.stdout
    assert "pinned baseline" in show_result.stdout

    assert compare_result.exit_code == 0
    assert "Compare two runs directly" in compare_result.stdout
    assert "suite comparison" in compare_result.stdout
    assert "direct run-to-run" in compare_result.stdout
    assert "show," in compare_result.stdout
    assert "compare, and trend" in compare_result.stdout


def test_cli_trend_shows_time_series_for_matching_configuration(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_sampled_run(
        database_path=database_path,
        suite_name="trend-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="trend-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.109, 0.110, 0.111, 0.110, 0.112, 0.109, 0.111],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="trend-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.139, 0.140, 0.141, 0.142, 0.140, 0.139, 0.141],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="trend-suite",
        configuration={"size": 1024, "variant": "baseline"},
        samples=[0.199, 0.200, 0.201, 0.200, 0.199],
        environment_payload=environment_payload,
    )

    result = runner.invoke(
        app,
        ["trend", "trend-suite", "1.1", "--database", str(database_path)],
    )

    assert result.exit_code == 0
    assert "Trend Basis: trend-suite" in result.stdout
    assert "Trend: trend-suite" in result.stdout
    assert "Delta" in result.stdout
    assert "Drift" in result.stdout
    assert "Status" in result.stdout
    assert "1.1" in result.stdout
    assert "2.1" in result.stdout
    assert "3.1" in result.stdout
    assert "4.1" not in result.stdout
