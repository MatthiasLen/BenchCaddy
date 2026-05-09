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
        observations=[{"sample": 1, "records": []}, {"sample": 2, "records": []}],
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
    assert "nonlinear-transform" in list_result.stdout
    assert "benchmark_target" in list_result.stdout

    show_result = runner.invoke(
        app,
        ["show", "nonlinear-transform", "--database", str(database_path)],
    )
    assert show_result.exit_code == 0
    assert "Suite: nonlinear-transform" in show_result.stdout
    assert "Environment" in show_result.stdout

    compare_result = runner.invoke(
        app,
        ["compare", "nonlinear-transform", "--database", str(database_path)],
    )
    assert compare_result.exit_code == 0
    assert "Comparison: nonlinear-transform" in compare_result.stdout
    assert "Delta vs Best" in compare_result.stdout
    assert "1.50x" in compare_result.stdout

    verbose_compare_result = runner.invoke(
        app,
        ["--verbose", "compare", "nonlinear-transform", "--database", str(database_path)],
    )
    assert verbose_compare_result.exit_code == 0
    assert "Samples" in verbose_compare_result.stdout
    assert verbose_compare_result.stdout != compare_result.stdout
    assert "Comparison Basis" in verbose_compare_result.stdout

    run_show_result = runner.invoke(app, ["show", "1", "--database", str(database_path)])
    assert run_show_result.exit_code == 0
    assert "Run: 1" in run_show_result.stdout
    assert "Observed Timings" in run_show_result.stdout

    run_compare_result = runner.invoke(
        app,
        ["compare", "1", "2", "--database", str(database_path)],
    )
    assert run_compare_result.exit_code == 0
    assert "Run Comparison: 1 -> 2" in run_compare_result.stdout
    assert "+50.00%" in run_compare_result.stdout