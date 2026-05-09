from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from benchcaddy.cli import app
from benchcaddy.db import record_benchmark_run


def _seed_run(
    *,
    database_path: Path,
    suite_name: str,
    configuration: dict[str, object],
    median_seconds: float,
    environment_payload: dict[str, object],
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
        median_seconds=median_seconds,
        min_seconds=median_seconds,
        max_seconds=median_seconds,
        std_seconds=0.0,
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
) -> None:
    record_benchmark_run(
        suite_name=suite_name,
        target_name="benchmark_target",
        configuration=configuration,
        samples=[median_seconds, median_seconds],
        observations=observations,
        median_seconds=median_seconds,
        min_seconds=median_seconds,
        max_seconds=median_seconds,
        std_seconds=0.0,
        environment=environment_payload,
        database_path=database_path,
    )

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
    assert "Time (s)" in show_result.stdout
    assert "0.100000" in show_result.stdout
    assert "+-" in show_result.stdout
    assert "0.000000" in show_result.stdout
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
    assert "Time (s)" in compare_result.stdout
    assert "1.1" in compare_result.stdout
    assert "2.1" in compare_result.stdout
    assert "1.50x" in compare_result.stdout

    verbose_compare_result = runner.invoke(
        app,
        ["--verbose", "compare", "nonlinear-transform", "--database", str(database_path)],
    )
    assert verbose_compare_result.exit_code == 0
    assert "Samples" in verbose_compare_result.stdout
    assert verbose_compare_result.stdout != compare_result.stdout
    assert "Comparison Basis" in verbose_compare_result.stdout

    run_show_result = runner.invoke(app, ["show", "1.1", "--database", str(database_path)])
    assert run_show_result.exit_code == 0
    assert "Run: 1.1" in run_show_result.stdout
    assert "Time (s)" in run_show_result.stdout
    assert "0.100000 +- 0.000000" in run_show_result.stdout
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
    assert "inner" in run_compare_result.stdout
    assert "outer" in run_compare_result.stdout
    assert "+50.00%" in run_compare_result.stdout


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