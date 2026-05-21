from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

import benchcaddy.cli as cli_module
import benchcaddy.cli.environment as environment_cli_module
import benchcaddy.cli.show as show_module
import benchcaddy.core as core_module
import benchcaddy.db._sqlite.models as models_module
from benchcaddy.cli import _suite_row_style, _trend_row_style, app
from benchcaddy.db import compare_runs, get_run_details, get_suite_details, record_benchmark_run
from benchcaddy.isolation import IsolatedRunResult
from benchcaddy.presentation import format_return_error, format_return_value

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def cli_variant_benchmark_target(variant: str) -> float:
    return 10.0 if variant == "baseline" else 13.5


def cli_e2e_benchmark_target() -> float:
    return 42.0


def cli_bool_target() -> bool:
    return True


def cli_float_vector_target() -> list[float]:
    return [2.0, 1.0, 0.5]


def cli_bool_vector_target() -> tuple[bool, bool, bool]:
    return (True, False, True)


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


def _plain_output(output: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)


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
    build_single_sample_sweep,
) -> None:
    database_path = tmp_path / "example.db"
    runner = CliRunner()

    _stub_sweep_runtime(monkeypatch, environment_payload)

    sweep = build_single_sample_sweep(
        target=cli_variant_benchmark_target,
        params={"variant": ["baseline", "candidate"]},
        suite_name="direct-return-value-suite",
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


def test_cli_sweep_runs_importable_target_and_records_results(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _stub_sweep_runtime(monkeypatch, environment_payload)
    monkeypatch.setattr(
        core_module,
        "run_isolated",
        lambda target, kwargs, **_ignored_kwargs: IsolatedRunResult(
            0.100 if kwargs["variant"] == "baseline" else 0.150,
            target(**kwargs),
            [],
        ),
    )

    result = runner.invoke(
        app,
        [
            "sweep",
            "tests.test_cli:cli_variant_benchmark_target",
            "--suite-name",
            "cli-sweep-suite",
            "--param",
            "variant=baseline,candidate",
            "--samples",
            "1",
            "--warmup-iterations",
            "0",
            "--store-target-return-value",
            "--database",
            str(database_path),
        ],
    )

    assert result.exit_code == 0
    assert "Recorded Runs: cli-sweep-suite" in result.stdout
    assert "1.1" in result.stdout
    assert "1.2" in result.stdout
    assert "baseline" in result.stdout
    assert "candidate" in result.stdout
    normalized_output = " ".join(result.stdout.split())
    assert "Inspect details: benchcaddy show 1.1 1.2 --database" in normalized_output
    assert "Compare runs: benchcaddy compare cli-sweep-suite --database" in normalized_output
    assert "Trend history: benchcaddy trend cli-sweep-suite --database" in normalized_output
    assert database_path.name in normalized_output

    show_result = runner.invoke(app, ["show", "cli-sweep-suite", "--database", str(database_path)])
    run_show_result = runner.invoke(app, ["show", "1.1", "--database", str(database_path)])

    assert show_result.exit_code == 0
    assert "Suite: cli-sweep-suite" in show_result.stdout
    assert "1.1" in show_result.stdout
    assert "1.2" in show_result.stdout

    assert run_show_result.exit_code == 0
    assert "Run: 1.1" in run_show_result.stdout
    assert "Return Value" in run_show_result.stdout


def test_cli_sweep_rejects_invalid_target_reference_syntax() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["sweep", "tests.test_cli.cli_variant_benchmark_target", "--suite-name", "cli-sweep-suite"])

    assert result.exit_code == 2
    assert "module:qualname" in result.stdout


def test_cli_sweep_rejects_duplicate_param_keys() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "sweep",
            "tests.test_cli:cli_variant_benchmark_target",
            "--suite-name",
            "cli-sweep-suite",
            "--param",
            "variant=baseline",
            "--param",
            "variant=candidate",
        ],
    )

    assert result.exit_code == 2
    assert "Duplicate --param key 'variant'." in result.stdout


def test_cli_sweep_rejects_malformed_json_array_param() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "sweep",
            "tests.test_cli:cli_variant_benchmark_target",
            "--suite-name",
            "cli-sweep-suite",
            "--param",
            "size=[1,",
        ],
    )

    assert result.exit_code == 2
    assert "valid JSON" in result.stdout


def test_cli_sweep_json_output_reports_recorded_runs(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _stub_sweep_runtime(monkeypatch, environment_payload)
    monkeypatch.setattr(
        core_module,
        "run_isolated",
        lambda target, kwargs, **_ignored_kwargs: IsolatedRunResult(
            0.100 if kwargs["variant"] == "baseline" else 0.200,
            target(**kwargs),
            [{"label": "inner", "duration_seconds": 0.01}],
        ),
    )

    result = runner.invoke(
        app,
        [
            "sweep",
            "tests.test_cli:cli_variant_benchmark_target",
            "--suite-name",
            "cli-sweep-json-suite",
            "--param",
            'variant=["baseline", "candidate"]',
            "--samples",
            "1",
            "--warmup-iterations",
            "0",
            "--store-target-return-value",
            "--json",
            "--database",
            str(database_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "sweep"
    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "pass"
    assert payload["reason"] == "runs_recorded"
    result_payload = payload["result"]
    assert result_payload["target_reference"] == "tests.test_cli:cli_variant_benchmark_target"
    assert result_payload["suite_name"] == "cli-sweep-json-suite"
    assert result_payload["run_count"] == 2
    assert result_payload["params"] == {"variant": ["baseline", "candidate"]}
    assert result_payload["runs"][0]["display_id"] == "1.1"
    assert result_payload["runs"][1]["display_id"] == "1.2"
    assert result_payload["runs"][0]["target_return_value"] == pytest.approx(10.0)


def test_cli_sweep_rejects_json_with_verbose() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--verbose",
            "sweep",
            "tests.test_cli:cli_variant_benchmark_target",
            "--suite-name",
            "cli-sweep-suite",
            "--json",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["command"] == "sweep"
    assert payload["status"] == "fail"
    assert payload["error_code"] == "json_conflicts_with_verbose"
    assert payload["reason"] == "json_conflicts_with_verbose"


def test_cli_sweep_subcommand_verbose_forwards_to_sweep_runtime(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _stub_sweep_runtime(monkeypatch, environment_payload)
    monkeypatch.setattr(
        core_module,
        "run_isolated",
        lambda target, kwargs, **_ignored_kwargs: IsolatedRunResult(
            0.100 if kwargs["variant"] == "baseline" else 0.150,
            target(**kwargs),
            [],
        ),
    )

    result = runner.invoke(
        app,
        [
            "sweep",
            "tests.test_cli:cli_variant_benchmark_target",
            "--suite-name",
            "cli-sweep-verbose-suite",
            "--param",
            'variant=["baseline", "candidate"]',
            "--samples",
            "1",
            "--warmup-iterations",
            "0",
            "--verbose",
            "--database",
            str(database_path),
        ],
    )

    assert result.exit_code == 0
    assert "BenchCaddy Run" in result.stdout
    assert "Configuration 1/2" in result.stdout
    assert "BenchCaddy Summary" in result.stdout


def test_cli_sweep_imports_target_from_current_working_directory(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    package_dir = tmp_path / "examples"
    package_dir.mkdir()
    (package_dir / "benchmark_local.py").write_text(
        "def benchmark_case(size: int, variant: str) -> float:\n    return float(size) if variant == 'baseline' else float(size) * 1.5\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in sys.path if entry not in {"", str(tmp_path)}],
    )
    _stub_sweep_runtime(monkeypatch, environment_payload)
    monkeypatch.setattr(
        core_module,
        "run_isolated",
        lambda target, kwargs, **_ignored_kwargs: IsolatedRunResult(
            0.100 if kwargs["variant"] == "baseline" else 0.150,
            target(**kwargs),
            [],
        ),
    )

    result = runner.invoke(
        app,
        [
            "sweep",
            "examples.benchmark_local:benchmark_case",
            "--suite-name",
            "cwd-import-suite",
            "--param",
            "size=[512]",
            "--param",
            'variant=["baseline", "candidate"]',
            "--samples",
            "1",
            "--warmup-iterations",
            "0",
            "--json",
            "--database",
            str(tmp_path / "benchcaddy.db"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    result_payload = payload["result"]
    assert result_payload["target_reference"] == "examples.benchmark_local:benchmark_case"
    assert result_payload["run_count"] == 2
    assert result_payload["runs"][0]["display_id"] == "1.1"
    assert result_payload["runs"][1]["display_id"] == "1.2"


def test_cli_env_json_output_uses_versioned_envelope(monkeypatch) -> None:
    runner = CliRunner()

    fake_environment = SimpleNamespace(
        cpu_load=0.12,
        on_battery=False,
        thermal_throttling=False,
        frequency_stable=True,
    )
    fake_noise = SimpleNamespace(
        relative_jitter=0.01,
        noise_level="low",
        relative_drift=0.01,
        drift_level="low",
        median_sample_seconds=0.000001,
        iteration_count=5,
    )
    fake_report = SimpleNamespace(
        timing_stability="HIGH",
        environmental_quality="HIGH",
        warnings=(),
    )

    monkeypatch.setattr(cli_module, "collect_environment_state", lambda: fake_environment)
    monkeypatch.setattr(cli_module, "NoiseAnalyzer", lambda: SimpleNamespace(analyze=lambda iterations: fake_noise))
    monkeypatch.setattr(cli_module, "get_affinity", lambda: [0, 1])
    monkeypatch.setattr(environment_cli_module, "build_reliability_report", lambda environment, noise: fake_report)

    result = runner.invoke(app, ["env", "-j"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "env"
    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "pass"
    assert payload["reason"] == "environment_ready"
    assert payload["confidence"] == "high"
    assert payload["result"]["affinity"] == [0, 1]
    assert payload["result"]["noise"]["iteration_count"] == 5


def test_cli_list_json_reports_empty_database(tmp_path: Path) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    result = runner.invoke(app, ["list", "-j", "--database", str(database_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "list"
    assert payload["status"] == "inconclusive"
    assert payload["reason"] == "no_suites_found"
    assert payload["result"]["suite_count"] == 0
    assert payload["result"]["suites"] == []


def test_cli_list_json_reports_suite_inventory(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="list-json-suite-a",
        configuration={"size": 512, "variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="list-json-suite-b",
        configuration={"size": 1024, "variant": "candidate"},
        median_seconds=0.120,
        environment_payload=environment_payload,
    )

    result = runner.invoke(app, ["list", "-j", "--database", str(database_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "list"
    assert payload["status"] == "pass"
    assert payload["reason"] == "suite_inventory_available"
    assert payload["result"]["database_path"] == str(database_path)
    assert payload["result"]["suite_count"] == 2
    suites_by_name = {suite["suite_name"]: suite for suite in payload["result"]["suites"]}
    assert set(suites_by_name) == {"list-json-suite-a", "list-json-suite-b"}
    assert suites_by_name["list-json-suite-a"]["run_count"] == 1
    assert suites_by_name["list-json-suite-b"]["run_count"] == 1
    assert suites_by_name["list-json-suite-a"]["observation_labels"] == ["inner", "outer"]
    assert suites_by_name["list-json-suite-b"]["observation_labels"] == ["inner", "outer"]


def test_cli_show_json_output_reports_suite_details(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="show-json-suite",
        configuration={"size": 512, "variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="show-json-suite",
        configuration={"size": 512, "variant": "candidate"},
        median_seconds=0.120,
        environment_payload=environment_payload,
    )

    result = runner.invoke(app, ["show", "show-json-suite", "-j", "--database", str(database_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "show"
    assert payload["status"] == "pass"
    assert payload["reason"] == "suite_details_available"
    assert payload["result"]["mode"] == "suite"
    assert payload["result"]["suite_name"] == "show-json-suite"
    assert len(payload["result"]["runs"]) == 2


def test_cli_show_json_output_reports_all_runs(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="show-all-suite-a",
        configuration={"size": 512, "variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
        target_return_value=1.0,
    )
    _seed_run(
        database_path=database_path,
        suite_name="show-all-suite-b",
        configuration={"size": 1024, "variant": "candidate"},
        median_seconds=0.120,
        environment_payload=environment_payload,
        target_return_value=2.0,
    )

    result = runner.invoke(app, ["show", "-j", "--database", str(database_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "show"
    assert payload["status"] == "pass"
    assert payload["reason"] == "run_list_available"
    assert payload["result"]["mode"] == "all_runs"
    assert payload["result"]["database_path"] == str(database_path)
    assert payload["result"]["run_count"] == 2
    assert payload["result"]["truncated"] is False
    assert payload["result"]["runs"][0]["suite_name"] == "show-all-suite-b"
    assert payload["result"]["runs"][1]["suite_name"] == "show-all-suite-a"
    assert payload["result"]["runs"][0]["target_return_value"] == pytest.approx(2.0)


def test_cli_show_json_output_reports_single_run_details(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="show-run-suite",
        configuration={"size": 512, "variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
        target_return_value=10.0,
    )

    result = runner.invoke(app, ["show", "1.1", "-j", "--database", str(database_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "show"
    assert payload["status"] == "pass"
    assert payload["reason"] == "run_details_available"
    assert payload["result"]["mode"] == "run"
    assert payload["result"]["database_path"] == str(database_path)
    assert payload["result"]["run"]["display_id"] == "1.1"
    assert payload["result"]["run"]["suite_name"] == "show-run-suite"
    assert payload["result"]["run"]["target_return_value"] == pytest.approx(10.0)
    assert payload["result"]["run"]["environment"]["cpu_model"] == environment_payload["cpu_model"]


def test_cli_show_json_output_reports_selected_runs(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="show-selected-suite",
        configuration={"variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="show-selected-suite",
        configuration={"variant": "candidate"},
        median_seconds=0.120,
        environment_payload=environment_payload,
    )

    result = runner.invoke(app, ["show", "1", "2.1", "-j", "--database", str(database_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "show"
    assert payload["status"] == "pass"
    assert payload["reason"] == "selected_runs_available"
    assert payload["result"]["mode"] == "selected_runs"
    assert payload["result"]["database_path"] == str(database_path)
    assert payload["result"]["requested_run_ids"] == ["1", "2.1"]
    assert payload["result"]["run_count"] == 2
    assert payload["result"]["total_run_count"] == 2
    assert payload["result"]["runs"][0]["display_id"] == "2.1"
    assert payload["result"]["runs"][1]["display_id"] == "1.1"


def test_cli_show_json_reports_config_usage_errors(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="show-json-suite",
        configuration={"size": 512, "variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )

    result = runner.invoke(app, ["show", "show-json-suite", "-j", "--config", "--database", str(database_path)])

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["command"] == "show"
    assert payload["status"] == "fail"
    assert payload["error_code"] == "missing_config_filter_scope"


def test_cli_show_json_reports_invalid_run_id_as_usage_error() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["show", "1.1", "not-a-run-id", "-j"])

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["command"] == "show"
    assert payload["status"] == "fail"
    assert payload["error_code"] == "invalid_run_id"


@pytest.mark.parametrize(
    (
        "case_name",
        "expected_exit_code",
        "expected_status",
        "expected_reason",
        "expected_error_code",
    ),
    [
        ("show_invalid_run_id", 2, "fail", "invalid_run_id", "invalid_run_id"),
        ("show_missing_config_scope", 2, "fail", "missing_config_filter_scope", "missing_config_filter_scope"),
        ("compare_strict_requires_reference_run", 2, "fail", "strict_requires_reference_run", "strict_requires_reference_run"),
        ("compare_empty_scope", 0, "inconclusive", "no_runs_matched_scope", None),
        ("trend_config_filter_no_matches", 0, "inconclusive", "config_filter_no_matches", None),
        ("trend_missing_baseline", 1, "fail", "baseline_not_found", "baseline_not_found"),
        ("sweep_json_conflicts_with_verbose", 2, "fail", "json_conflicts_with_verbose", "json_conflicts_with_verbose"),
    ],
)
def test_cli_json_contract_matrix(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
    case_name: str,
    expected_exit_code: int,
    expected_status: str,
    expected_reason: str,
    expected_error_code: str | None,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    if case_name == "show_invalid_run_id":
        args = ["show", "1.1", "not-a-run-id", "-j"]
    elif case_name == "show_missing_config_scope":
        _seed_run(
            database_path=database_path,
            suite_name="matrix-show-suite",
            configuration={"size": 512, "variant": "baseline"},
            median_seconds=0.100,
            environment_payload=environment_payload,
        )
        args = ["show", "matrix-show-suite", "-j", "--config", "--database", str(database_path)]
    elif case_name == "compare_strict_requires_reference_run":
        _seed_run(
            database_path=database_path,
            suite_name="matrix-compare-suite",
            configuration={"size": 33, "variant": "baseline"},
            median_seconds=0.100,
            environment_payload=environment_payload,
        )
        args = ["compare", "matrix-compare-suite", "--strict", "-j", "--database", str(database_path)]
    elif case_name == "compare_empty_scope":
        _seed_run(
            database_path=database_path,
            suite_name="matrix-empty-scope-suite",
            configuration={"size": 33, "variant": "baseline"},
            median_seconds=0.100,
            environment_payload=environment_payload,
        )
        args = ["compare", "matrix-empty-scope-suite", "-c", "size=99", "-j", "--database", str(database_path)]
    elif case_name == "trend_missing_baseline":
        _seed_sampled_run(
            database_path=database_path,
            suite_name="matrix-trend-suite",
            configuration={"size": 512, "variant": "baseline"},
            samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
            environment_payload=environment_payload,
        )
        args = ["trend", "matrix-trend-suite", "--baseline", "-j", "--database", str(database_path)]
    elif case_name == "trend_config_filter_no_matches":
        _seed_sampled_run(
            database_path=database_path,
            suite_name="matrix-trend-suite",
            configuration={"size": 512, "variant": "baseline"},
            samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
            environment_payload=environment_payload,
        )
        args = ["trend", "matrix-trend-suite", "-c", "size=999", "-j", "--database", str(database_path)]
    elif case_name == "sweep_json_conflicts_with_verbose":
        _stub_sweep_runtime(monkeypatch, environment_payload)
        args = [
            "--verbose",
            "sweep",
            "tests.test_cli:cli_variant_benchmark_target",
            "--suite-name",
            "matrix-sweep-suite",
            "-j",
        ]
    else:
        raise AssertionError(f"Unhandled matrix case: {case_name}")

    result = runner.invoke(app, args)

    assert result.exit_code == expected_exit_code
    payload = json.loads(result.stdout)
    assert payload["status"] == expected_status
    assert payload["reason"] == expected_reason
    assert payload["error_code"] == expected_error_code


def test_compare_does_not_alias_missing_dotted_run_id_to_record_id(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="alias-guard-suite",
        configuration={"size": 512, "variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )
    record_benchmark_run(
        suite_name="alias-guard-suite",
        target_name="benchmark_target",
        configuration={"size": 768, "variant": "baseline"},
        samples=[0.110, 0.110],
        observations=[],
        **_uniform_run_kwargs(0.110),
        environment=environment_payload,
        sweep_execution_id=1,
        run_index=2,
        database_path=database_path,
    )
    record_benchmark_run(
        suite_name="alias-guard-suite",
        target_name="benchmark_target",
        configuration={"size": 1024, "variant": "baseline"},
        samples=[0.120, 0.120],
        observations=[],
        **_uniform_run_kwargs(0.120),
        environment=environment_payload,
        sweep_execution_id=1,
        run_index=3,
        database_path=database_path,
    )
    _seed_run(
        database_path=database_path,
        suite_name="alias-guard-suite",
        configuration={"size": 2048, "variant": "baseline"},
        median_seconds=0.130,
        environment_payload=environment_payload,
    )

    result = runner.invoke(app, ["compare", "2.1", "3.1", "--database", str(database_path)])

    assert result.exit_code == 1
    assert "Run comparison" in result.stdout
    assert "1.3" not in result.stdout


def test_cli_end_to_end_compare_and_trend_json_support_regression_gate(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _stub_sweep_runtime(monkeypatch, environment_payload)

    run_samples = [
        [0.1090, 0.1100, 0.1110, 0.1105, 0.1095],
        [0.1020, 0.1010, 0.1030, 0.1025, 0.1015],
        [0.1060, 0.1070, 0.1050, 0.1065, 0.1055],
        [0.1120, 0.1110, 0.1130, 0.1125, 0.1115],
        [0.1000, 0.0990, 0.1010, 0.1005, 0.0995],
        [0.1140, 0.1150, 0.1130, 0.1145, 0.1135],
        [0.1180, 0.1170, 0.1190, 0.1185, 0.1175],
        [0.1210, 0.1200, 0.1220, 0.1215, 0.1205],
        [0.1250, 0.1240, 0.1260, 0.1255, 0.1245],
        [0.1290, 0.1280, 0.1300, 0.1295, 0.1285],
        [0.1330, 0.1320, 0.1340, 0.1335, 0.1325],
        [0.1370, 0.1360, 0.1380, 0.1375, 0.1365],
        [0.1410, 0.1400, 0.1420, 0.1415, 0.1405],
        [0.1450, 0.1440, 0.1460, 0.1455, 0.1445],
        [0.1490, 0.1480, 0.1500, 0.1495, 0.1485],
        [0.1530, 0.1520, 0.1540, 0.1535, 0.1525],
        [0.1600, 0.1590, 0.1610, 0.1605, 0.1595],
        [0.1680, 0.1670, 0.1690, 0.1685, 0.1675],
        [0.1810, 0.1800, 0.1820, 0.1815, 0.1805],
        [0.1950, 0.1940, 0.1960, 0.1955, 0.1945],
    ]

    perf_counter_values: list[float] = []
    current_time = 0.0
    for samples in run_samples:
        for duration in samples:
            perf_counter_values.append(duration)
            current_time += duration + 1.0
    perf_counter_iter = iter(perf_counter_values)
    monkeypatch.setattr(
        core_module,
        "run_isolated",
        lambda target, **kwargs: IsolatedRunResult(next(perf_counter_iter), cli_e2e_benchmark_target(), []),
    )

    sweep = core_module.Sweep(
        target=cli_e2e_benchmark_target,
        params={},
        suite_name="e2e-json-suite",
        samples=5,
        warmup_iterations=0,
        lock_cpu_affinity=False,
        database_path=database_path,
    )

    for _ in range(20):
        results = sweep.run()
        assert len(results) == 1

    compare_result = runner.invoke(app, ["compare", "e2e-json-suite", "--database", str(database_path)])
    assert compare_result.exit_code == 0
    assert "Comparison: e2e-json-suite" in compare_result.stdout
    assert "Best" in compare_result.stdout

    compare_json_result = runner.invoke(app, ["compare", "e2e-json-suite", "--json", "--database", str(database_path)])
    assert compare_json_result.exit_code == 0
    compare_payload = json.loads(compare_json_result.stdout)
    compare_result_payload = compare_payload["result"]
    assert compare_payload["command"] == "compare"
    assert compare_result_payload["comparison_mode"] == "suite"
    assert len(compare_result_payload["runs"]) == 20

    best_run = compare_result_payload["basis_run"]
    trend_result = runner.invoke(app, ["trend", "e2e-json-suite", best_run["display_id"], "--json", "--database", str(database_path)])
    assert trend_result.exit_code == 0
    trend_payload = json.loads(trend_result.stdout)
    trend_result_payload = trend_payload["result"]
    assert trend_result_payload["suite_name"] == "e2e-json-suite"
    assert trend_result_payload["basis_run"]["display_id"] == best_run["display_id"]
    assert len(trend_result_payload["runs"]) == 20

    trend_runs = trend_result_payload["runs"]
    worst_run = max(trend_runs, key=lambda run: (run["median_seconds"], run["id"]))
    expected_best_run = min(trend_runs, key=lambda run: (run["median_seconds"], run["id"]))
    assert best_run["display_id"] == expected_best_run["display_id"]

    percent_change = ((worst_run["median_seconds"] - best_run["median_seconds"]) / best_run["median_seconds"]) * 100.0
    assert percent_change > 0.0

    narrow_gap = 0.01
    failing_threshold = max(percent_change * 0.98, narrow_gap)
    passing_threshold = percent_change + narrow_gap

    failing_gate_result = runner.invoke(
        app,
        [
            "compare",
            best_run["display_id"],
            worst_run["display_id"],
            "--json",
            "--fail-if-regression",
            f"{failing_threshold:.4f}",
            "--database",
            str(database_path),
        ],
    )

    assert failing_gate_result.exit_code == cli_module.REGRESSION_EXIT_CODE == 3
    failing_payload = json.loads(failing_gate_result.stdout)
    failing_result_payload = failing_payload["result"]
    assert failing_payload["status"] == "fail"
    assert failing_result_payload["comparison_mode"] == "direct"
    assert failing_result_payload["candidate"]["display_id"] == worst_run["display_id"]
    assert failing_result_payload["comparison_analysis"]["regression_detected"] is True
    assert failing_result_payload["gate"]["failed"] is True
    assert failing_result_payload["gate"]["failing_runs"][0]["display_id"] == worst_run["display_id"]

    passing_gate_result = runner.invoke(
        app,
        [
            "compare",
            best_run["display_id"],
            worst_run["display_id"],
            "--json",
            "--fail-if-regression",
            f"{passing_threshold:.4f}",
            "--database",
            str(database_path),
        ],
    )

    assert passing_gate_result.exit_code == 0
    passing_payload = json.loads(passing_gate_result.stdout)
    passing_result_payload = passing_payload["result"]
    assert passing_payload["status"] == "pass"
    assert passing_result_payload["comparison_mode"] == "direct"
    assert passing_result_payload["comparison_analysis"]["regression_detected"] is False
    assert passing_result_payload["gate"]["failed"] is False
    assert passing_result_payload["gate"]["failing_runs"] == []


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

    monkeypatch.setattr(models_module, "analyze_samples", _raise_if_analysis_called)

    show_result = runner.invoke(app, ["show", "--database", str(database_path)])

    assert show_result.exit_code == 0
    assert "All Runs" in show_result.stdout


def test_show_suite_numitems_limits_to_latest_runs_and_prints_notice(
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
        suite_name="limited-suite",
        configuration={"variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="limited-suite",
        configuration={"variant": "candidate-a"},
        median_seconds=0.110,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="limited-suite",
        configuration={"variant": "candidate-b"},
        median_seconds=0.120,
        environment_payload=environment_payload,
    )

    show_result = runner.invoke(app, ["show", "limited-suite", "--numitems", "2", "--database", str(database_path)])

    assert show_result.exit_code == 0
    assert "Suite: limited-suite" in show_result.stdout
    assert "3.1" in show_result.stdout
    assert "2.1" in show_result.stdout
    assert "1.1" not in show_result.stdout
    assert "Output capped to the latest entries by record ID." in show_result.stdout
    assert _compact_output(f"benchcaddy show limited-suite -n 3 --database {database_path}") in _compact_output(show_result.stdout)


def test_show_suite_can_filter_by_config(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="filtered-show-suite",
        configuration={"size": 512, "variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="filtered-show-suite",
        configuration={"size": 1024, "variant": "candidate-a"},
        median_seconds=0.120,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="filtered-show-suite",
        configuration={"size": 1024, "variant": "candidate-b"},
        median_seconds=0.110,
        environment_payload=environment_payload,
    )

    result = runner.invoke(
        app,
        ["show", "filtered-show-suite", "-c", "size=1024", "--database", str(database_path)],
    )

    assert result.exit_code == 0
    assert "Suite: filtered-show-suite (config: size=1024)" in result.stdout
    assert "3.1" in result.stdout
    assert "2.1" in result.stdout
    assert "1.1" not in result.stdout


def test_show_filtered_suite_numitems_notice_preserves_config_flag(
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
        suite_name="filtered-show-suite",
        configuration={"size": 1024, "variant": "candidate-a"},
        median_seconds=0.120,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="filtered-show-suite",
        configuration={"size": 1024, "variant": "candidate-b"},
        median_seconds=0.110,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="filtered-show-suite",
        configuration={"size": 1024, "variant": "candidate-c"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )

    result = runner.invoke(
        app,
        ["show", "filtered-show-suite", "-c", "size=1024", "--numitems", "2", "--database", str(database_path)],
    )

    assert result.exit_code == 0
    assert "Output capped to the latest entries by record ID." in result.stdout
    assert _compact_output(f"benchcaddy show filtered-show-suite -c size=1024 -n 3 --database {database_path}") in _compact_output(result.stdout)


def test_show_config_requires_suite_name_and_entries(tmp_path: Path) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    result = runner.invoke(app, ["show", "-c", "size=1024", "--database", str(database_path)])

    assert result.exit_code == 2
    assert "--config/-c requires a suite name followed by one or more key=value entries." in result.stdout


def test_show_without_arguments_numitems_limits_output_and_prints_notice(
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
        configuration={"variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="suite-b",
        configuration={"variant": "candidate-a"},
        median_seconds=0.110,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="suite-c",
        configuration={"variant": "candidate-b"},
        median_seconds=0.120,
        environment_payload=environment_payload,
    )

    show_result = runner.invoke(app, ["show", "--numitems", "2", "--database", str(database_path)])

    assert show_result.exit_code == 0
    assert "All Runs" in show_result.stdout
    assert "3.1" in show_result.stdout
    assert "2.1" in show_result.stdout
    assert "1.1" not in show_result.stdout
    assert "Output capped to the latest entries by record ID." in show_result.stdout
    assert _compact_output(f"benchcaddy show -n 3 --database {database_path}") in _compact_output(show_result.stdout)


def test_show_selected_runs_numitems_limits_output_and_prints_notice(
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
        suite_name="selected-suite",
        configuration={"variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="selected-suite",
        configuration={"variant": "candidate"},
        median_seconds=0.110,
        environment_payload=environment_payload,
    )

    show_result = runner.invoke(app, ["show", "1", "2.1", "--numitems", "1", "--database", str(database_path)])

    assert show_result.exit_code == 0
    assert "Selected Runs" in show_result.stdout
    assert "2.1" in show_result.stdout
    assert "1.1" not in show_result.stdout
    assert "Output capped to the latest entries by record ID." in show_result.stdout
    assert _compact_output(f"benchcaddy show 1 2.1 -n 2 --database {database_path}") in _compact_output(show_result.stdout)


def test_show_defaults_to_100_items_and_suggests_total_count(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    monkeypatch.setattr(
        cli_module,
        "console",
        Console(force_terminal=False, color_system=None, width=200),
    )

    for index in range(101):
        _seed_run(
            database_path=database_path,
            suite_name="default-cap-suite",
            configuration={"variant": f"candidate-{index}"},
            median_seconds=0.100 + index * 0.001,
            environment_payload=environment_payload,
        )

    show_result = runner.invoke(app, ["show", "default-cap-suite", "--database", str(database_path)])

    assert show_result.exit_code == 0
    assert "Suite: default-cap-suite" in show_result.stdout
    assert "101.1" in show_result.stdout
    assert "2.1" in show_result.stdout
    assert "candidate-0" not in show_result.stdout
    assert _compact_output(f"benchcaddy show default-cap-suite -n 101 --database {database_path}") in _compact_output(show_result.stdout)


def test_show_suite_requests_limited_details_from_db(tmp_path: Path, monkeypatch) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        show_module,
        "get_suite_details",
        lambda suite_name, database_path_arg, include_analysis=False, limit=None: (
            calls.append(("details", limit)),
            {
                "suite_name": suite_name,
                "target_name": "benchmark_target",
                "runs": [
                    {
                        "id": 2,
                        "display_id": "2.1",
                        "record_id": 2,
                        "configuration": {"variant": "candidate"},
                        "samples": [0.2, 0.2],
                        "observations": [],
                        "median_seconds": 0.2,
                        "min_seconds": 0.2,
                        "max_seconds": 0.2,
                        "std_seconds": 0.0,
                        "target_return_value": None,
                        "suite_name": suite_name,
                        "target_name": "benchmark_target",
                        "analysis": None,
                        "ci_lower_seconds": None,
                        "ci_upper_seconds": None,
                        "mad_seconds": None,
                        "coefficient_of_variation": None,
                        "noise_warnings": [],
                        "created_at": "2026-05-19T00:00:00Z",
                    },
                    {
                        "id": 1,
                        "display_id": "1.1",
                        "record_id": 1,
                        "configuration": {"variant": "baseline"},
                        "samples": [0.1, 0.1],
                        "observations": [],
                        "median_seconds": 0.1,
                        "min_seconds": 0.1,
                        "max_seconds": 0.1,
                        "std_seconds": 0.0,
                        "target_return_value": None,
                        "suite_name": suite_name,
                        "target_name": "benchmark_target",
                        "analysis": None,
                        "ci_lower_seconds": None,
                        "ci_upper_seconds": None,
                        "mad_seconds": None,
                        "coefficient_of_variation": None,
                        "noise_warnings": [],
                        "created_at": "2026-05-18T00:00:00Z",
                    },
                ],
                "environment": None,
                "baseline_run": None,
            },
        )[1],
    )
    monkeypatch.setattr(
        show_module,
        "get_suite_run_count",
        lambda suite_name, database_path_arg: calls.append(("count", suite_name)) or 3,
    )
    monkeypatch.setattr(
        cli_module,
        "console",
        Console(force_terminal=False, color_system=None, width=160),
    )

    show_result = runner.invoke(app, ["show", "limited-suite", "--numitems", "2", "--database", str(database_path)])

    assert show_result.exit_code == 0
    assert calls == [("details", 2), ("count", "limited-suite")]
    assert _compact_output(f"benchcaddy show limited-suite -n 3 --database {database_path}") in _compact_output(show_result.stdout)


def test_show_rejects_removed_stats_flags(
    tmp_path: Path,
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

    show_result = runner.invoke(app, ["show", "1.1", "--no-stats", "--database", str(database_path)])
    normalized_output = _strip_ansi(show_result.output)

    assert show_result.exit_code == 2
    assert "--no-stats" in normalized_output
    assert "No such option" in normalized_output


@pytest.mark.parametrize(
    ("suite_name", "target", "expected_return_value"),
    [
        ("return-type-bool", cli_bool_target, True),
        ("return-type-float-vector", cli_float_vector_target, [2.0, 1.0, 0.5]),
        ("return-type-bool-vector", cli_bool_vector_target, [1.0, 0.0, 1.0]),
    ],
)
def test_direct_sweep_supports_return_value_types(
    tmp_path: Path,
    monkeypatch,
    environment_payload: dict[str, object],
    build_single_sample_sweep,
    suite_name: str,
    target,
    expected_return_value: bool | list[float],
) -> None:
    database_path = tmp_path / "return-types.db"

    _stub_sweep_runtime(monkeypatch, environment_payload)

    build_single_sample_sweep(
        target=target,
        suite_name=suite_name,
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


def test_cli_compare_can_filter_suite_by_config(
    tmp_path: Path,
    environment_payload: dict[str, object],
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
        ["compare", "nonlinear-transform", "-c", "size=33", "--database", str(database_path)],
    )

    assert result.exit_code == 0
    assert "Comparison: nonlinear-transform (config: size=33)" in result.stdout
    assert "4.1" in result.stdout
    assert "2.1" in result.stdout
    assert "1.1" in result.stdout
    assert "3.1" not in result.stdout


def test_cli_compare_with_pinned_baseline_and_config_filter_warns_when_baseline_is_outside_filter(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="baseline-filter-suite",
        configuration={"size": 512, "variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="baseline-filter-suite",
        configuration={"size": 1024, "variant": "candidate-a"},
        median_seconds=0.120,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="baseline-filter-suite",
        configuration={"size": 1024, "variant": "candidate-b"},
        median_seconds=0.110,
        environment_payload=environment_payload,
    )

    pin_result = runner.invoke(app, ["baseline", "baseline-filter-suite", "--pin", "1.1", "--database", str(database_path)])

    assert pin_result.exit_code == 0

    result = runner.invoke(
        app,
        ["compare", "baseline-filter-suite", "--baseline", "-c", "size=1024", "--database", str(database_path)],
    )

    assert result.exit_code == 0
    assert "Filter Warning" in result.stdout
    assert "Baseline 1.1 does not match the requested config filter." in result.stdout
    assert "3.1" in result.stdout
    assert "2.1" in result.stdout


def test_cli_compare_rejects_config_filter_with_strict(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="config-strict-suite",
        configuration={"size": 33, "variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )

    result = runner.invoke(
        app,
        ["compare", "config-strict-suite", "-c", "size=33", "--strict", "size", "--database", str(database_path)],
    )

    assert result.exit_code == 2
    assert "--strict cannot be combined with --config/-c." in result.stdout


def test_cli_compare_json_reports_empty_scope_as_inconclusive(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="empty-scope-suite",
        configuration={"size": 33, "variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )

    result = runner.invoke(
        app,
        ["compare", "empty-scope-suite", "-c", "size=99", "-j", "--database", str(database_path)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "compare"
    assert payload["status"] == "inconclusive"
    assert payload["reason"] == "no_runs_matched_scope"
    assert payload["confidence"] is None
    assert payload["result"]["runs"] == []
    assert payload["result"]["basis_run"] is None


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


def test_trend_sparkline_compacts_to_requested_width() -> None:
    series = [float(index) for index in range(1, 21)]

    compact = cli_module._trend_sparkline(series, max_points=6)

    assert len(compact) == 6
    assert "…" not in compact
    assert compact[-1] == cli_module._trend_sparkline(series)[-1]


def test_cli_baseline_can_pin_and_compare_can_use_baseline(
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

    pin_result = runner.invoke(app, ["baseline", "baseline-suite", "--pin", "2.1", "--database", str(database_path)])

    assert pin_result.exit_code == 0
    assert "Baseline Updated" in pin_result.stdout
    assert "baseline-suite" in pin_result.stdout
    assert "2.1" in pin_result.stdout

    use_result = runner.invoke(
        app,
        ["compare", "baseline-suite", "--baseline", "--database", str(database_path)],
    )

    assert use_result.exit_code == 0
    assert "Statistical Findings" in use_result.stdout
    assert "Basis Source" in use_result.stdout
    assert "baseline" in use_result.stdout
    assert "2.1" in use_result.stdout


def test_cli_baseline_uses_latest_pinned_baseline(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="baseline-history-suite",
        configuration={"variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="baseline-history-suite",
        configuration={"variant": "candidate"},
        median_seconds=0.120,
        environment_payload=environment_payload,
    )

    first_pin = runner.invoke(app, ["baseline", "baseline-history-suite", "--pin", "1.1", "--database", str(database_path)])
    second_pin = runner.invoke(app, ["baseline", "baseline-history-suite", "--pin", "2.1", "--database", str(database_path)])
    use_result = runner.invoke(
        app,
        ["compare", "baseline-history-suite", "--baseline", "--json", "--database", str(database_path)],
    )

    assert first_pin.exit_code == 0
    assert second_pin.exit_code == 0
    assert use_result.exit_code == 0

    payload = json.loads(use_result.stdout)
    assert payload["result"]["basis_source"] == "pinned"
    assert payload["result"]["basis_run"]["display_id"] == "2.1"


def test_cli_baseline_shows_history_and_note(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="baseline-notes-suite",
        configuration={"variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="baseline-notes-suite",
        configuration={"variant": "candidate"},
        median_seconds=0.120,
        environment_payload=environment_payload,
    )

    pin_result = runner.invoke(
        app,
        ["baseline", "baseline-notes-suite", "--pin", "2.1", "--note", "release candidate", "--database", str(database_path)],
    )
    show_result = runner.invoke(app, ["baseline", "baseline-notes-suite", "--database", str(database_path)])

    assert pin_result.exit_code == 0
    assert show_result.exit_code == 0
    assert "Current Baseline" in show_result.stdout
    assert "Baseline History: baseline-notes-suite" in show_result.stdout
    assert "release candidate" in show_result.stdout
    assert "2.1" in show_result.stdout


def test_cli_compare_rejects_removed_pin_baseline_option(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="removed-pin-suite",
        configuration={"variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )

    result = runner.invoke(app, ["compare", "removed-pin-suite", "1.1", "--pin-baseline", "--database", str(database_path)])
    normalized_output = _strip_ansi(result.output)

    assert result.exit_code == 2
    assert "No such option" in normalized_output
    assert "--pin-baseline" in normalized_output


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

    baseline_result = test_runner.invoke(app, ["baseline", "--help"])
    show_result = test_runner.invoke(app, ["show", "--help"])
    compare_result = test_runner.invoke(app, ["compare", "--help"])
    sweep_result = test_runner.invoke(app, ["sweep", "--help"])
    trend_result = test_runner.invoke(app, ["trend", "--help"])
    baseline_output = _plain_output(baseline_result.stdout)
    show_output = _plain_output(show_result.stdout)
    compare_output = _plain_output(compare_result.stdout)
    sweep_output = _plain_output(sweep_result.stdout)
    trend_output = _plain_output(trend_result.stdout)

    assert baseline_result.exit_code == 0
    assert "Inspect baseline history" in baseline_output
    assert "--pin" in baseline_output
    assert "--note" in baseline_output
    assert "--json" in baseline_output

    assert show_result.exit_code == 0
    assert "Inspect all recorded runs, a suite, or specific run IDs." in show_output
    assert "Omit" in show_output
    assert "identifiers to list all recorded runs." in show_output
    assert "recorded baseline" in show_output
    assert "--no-stats" not in show_output
    assert "--confidence-level" not in show_output
    assert "--bootstrap-resamples" not in show_output

    assert compare_result.exit_code == 0
    assert "Compare two runs directly" in compare_output
    assert "suite comparison" in compare_output
    assert "direct run-to-run" in compare_output
    assert "--baseline" in compare_output
    assert "-b" in compare_output
    assert "--json" in compare_output
    assert "--fail-if-regression" in compare_output
    assert "--pin-baseline" not in compare_output

    assert sweep_result.exit_code == 0
    assert "importable target reference" in sweep_output
    assert "module:function" in sweep_output
    assert "module:Class.method" in sweep_output
    assert "--suite-name" in sweep_output
    assert "--param" in sweep_output
    assert "--json" in sweep_output
    assert "--verbose" in sweep_output

    assert trend_result.exit_code == 0
    assert "--baseline" in trend_output
    assert "-b" in trend_output
    assert "--json" in trend_output
    assert "--pinned" not in trend_output


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
    assert "Median Trend" in result.stdout
    assert "Trend: trend-suite" in result.stdout
    assert "Delta" in result.stdout
    assert "Vs Recent" in result.stdout
    assert "Vs Basis" in result.stdout
    assert "1.1" in result.stdout
    assert "2.1" in result.stdout
    assert "3.1" in result.stdout
    assert "4.1" not in result.stdout
    assert "Selected configuration" in result.stdout
    assert "Available suite configurations" in result.stdout
    assert "size=512, variant=baseline" in result.stdout
    assert "size=1024, variant=baseline" in result.stdout
    assert "▁▃█" in result.stdout


def test_cli_trend_shows_summary_for_mixed_suite_without_baseline(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_sampled_run(
        database_path=database_path,
        suite_name="summary-trend-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="summary-trend-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.109, 0.110, 0.111, 0.110, 0.112, 0.109, 0.111],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="summary-trend-suite",
        configuration={"size": 1024, "variant": "baseline"},
        samples=[0.199, 0.200, 0.201, 0.200, 0.199],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="summary-trend-suite",
        configuration={"size": 1024, "variant": "baseline"},
        samples=[0.219, 0.220, 0.221, 0.220, 0.219],
        environment_payload=environment_payload,
    )

    result = runner.invoke(
        app,
        ["trend", "summary-trend-suite", "--database", str(database_path)],
    )

    output = _plain_output(result.stdout)
    assert result.exit_code == 0
    assert "Trend Summary: summary-trend-suite" in output
    assert "Per-configuration summary" in output
    assert "Vs 1st" in output
    assert "Vs Recent" in output
    assert "Vs Best" in output
    assert "Trend Basis:" not in output
    assert "Vs Basis" not in output
    assert "512" in output
    assert "1024" in output
    assert "▁█" in output
    assert "([" not in output
    assert "Label Guide" in output
    assert "reg" in output
    assert "meaningful slowdown detected" in output
    assert "noisy" in output


def test_cli_trend_uses_baseline_for_mixed_suite_when_requested(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_sampled_run(
        database_path=database_path,
        suite_name="pinned-trend-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="pinned-trend-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.109, 0.110, 0.111, 0.110, 0.112, 0.109, 0.111],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="pinned-trend-suite",
        configuration={"size": 1024, "variant": "baseline"},
        samples=[0.199, 0.200, 0.201, 0.200, 0.199],
        environment_payload=environment_payload,
    )

    pin_result = runner.invoke(app, ["baseline", "pinned-trend-suite", "--pin", "1.1", "--database", str(database_path)])

    assert pin_result.exit_code == 0

    result = runner.invoke(
        app,
        ["trend", "pinned-trend-suite", "--baseline", "--database", str(database_path)],
    )

    assert result.exit_code == 0
    assert "Trend Basis: pinned-trend-suite" in result.stdout
    assert "Trend Summary:" not in result.stdout
    assert "1.1" in result.stdout
    assert "2.1" in result.stdout
    assert "3.1" not in result.stdout
    assert "baseline" in result.stdout


def test_cli_trend_can_filter_by_config_and_use_best_filtered_run(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_sampled_run(
        database_path=database_path,
        suite_name="filtered-trend-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="filtered-trend-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.109, 0.110, 0.111, 0.110, 0.112],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="filtered-trend-suite",
        configuration={"size": 1024, "variant": "baseline"},
        samples=[0.079, 0.080, 0.081, 0.080, 0.082],
        environment_payload=environment_payload,
    )

    result = runner.invoke(
        app,
        ["trend", "filtered-trend-suite", "-c", "size=512", "--database", str(database_path)],
    )

    assert result.exit_code == 0
    assert "Trend Basis: filtered-trend-suite" in result.stdout
    assert "best" in result.stdout
    assert "Selected configuration" in result.stdout
    assert "Available suite configurations" in result.stdout
    assert "size: 512" in result.stdout
    assert "size=512, variant=baseline" in result.stdout
    assert "size=1024, variant=baseline" in result.stdout
    assert "1.1" in result.stdout
    assert "2.1" in result.stdout
    assert "3.1" not in result.stdout


def test_cli_trend_rejects_config_with_baseline_or_explicit_baseline(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_sampled_run(
        database_path=database_path,
        suite_name="invalid-filter-trend-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.099, 0.100, 0.101],
        environment_payload=environment_payload,
    )

    explicit_result = runner.invoke(
        app,
        ["trend", "invalid-filter-trend-suite", "1.1", "-c", "size=512", "--database", str(database_path)],
    )
    baseline_result = runner.invoke(
        app,
        ["trend", "invalid-filter-trend-suite", "-c", "size=512", "--baseline", "--database", str(database_path)],
    )

    assert explicit_result.exit_code == 2
    assert "--config/-c cannot be combined with an explicit baseline run ID." in explicit_result.stdout
    assert baseline_result.exit_code == 2
    assert "--config/-c cannot be combined with --baseline/-b." in baseline_result.stdout


def test_cli_trend_ignores_baseline_without_flag(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_sampled_run(
        database_path=database_path,
        suite_name="default-trend-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="default-trend-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.109, 0.110, 0.111, 0.110, 0.112, 0.109, 0.111],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="default-trend-suite",
        configuration={"size": 1024, "variant": "baseline"},
        samples=[0.199, 0.200, 0.201, 0.200, 0.199],
        environment_payload=environment_payload,
    )

    pin_result = runner.invoke(app, ["baseline", "default-trend-suite", "--pin", "1.1", "--database", str(database_path)])

    assert pin_result.exit_code == 0

    result = runner.invoke(
        app,
        ["trend", "default-trend-suite", "--database", str(database_path)],
    )

    assert result.exit_code == 0
    assert "Trend Summary:" in result.stdout
    assert "Trend Basis: default-trend-suite" not in result.stdout
    assert "512" in result.stdout
    assert "1024" in result.stdout


def test_cli_trend_baseline_requires_existing_baseline(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_sampled_run(
        database_path=database_path,
        suite_name="unpinned-cli-trend-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )

    result = runner.invoke(
        app,
        ["trend", "unpinned-cli-trend-suite", "--baseline", "--database", str(database_path)],
    )

    assert result.exit_code == 1
    assert "does not have a recorded baseline" in result.stdout


def test_cli_trend_compacts_long_median_graph_without_right_ellipsis(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    for index in range(1, 81):
        baseline = 0.100 + ((index % 9) * 0.003) + (index * 0.0002)
        _seed_sampled_run(
            database_path=database_path,
            suite_name="wide-graph-suite",
            configuration={"size": 1, "variant": "baseline"},
            samples=[baseline - 0.001, baseline, baseline + 0.001, baseline, baseline],
            environment_payload=environment_payload,
        )

    result = runner.invoke(
        app,
        ["trend", "wide-graph-suite", "1.1", "--database", str(database_path)],
    )

    assert result.exit_code == 0
    graph_line = next(line for line in result.stdout.splitlines() if "Graph" in line)
    assert "…" not in graph_line


def test_cli_compare_json_output_reports_direct_regression_gate_failure(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_sampled_run(
        database_path=database_path,
        suite_name="direct-gate-suite",
        configuration={"variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="direct-gate-suite",
        configuration={"variant": "candidate"},
        samples=[0.129, 0.130, 0.131, 0.130, 0.132, 0.129, 0.131],
        environment_payload=environment_payload,
    )

    result = runner.invoke(
        app,
        ["compare", "1.1", "2.1", "--json", "--fail-if-regression", "5%", "--database", str(database_path)],
    )

    assert result.exit_code == cli_module.REGRESSION_EXIT_CODE
    payload = json.loads(result.stdout)
    result_payload = payload["result"]
    assert payload["status"] == "fail"
    assert result_payload["comparison_mode"] == "direct"
    assert result_payload["comparison_analysis"]["regression_detected"] is True
    assert result_payload["gate"]["enabled"] is True
    assert result_payload["gate"]["failed"] is True
    assert result_payload["gate"]["threshold_percent"] == pytest.approx(5.0)
    assert result_payload["gate"]["failing_runs"][0]["display_id"] == "2.1"


def test_cli_compare_json_output_reports_direct_gate_pass(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_sampled_run(
        database_path=database_path,
        suite_name="direct-gate-pass-suite",
        configuration={"variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="direct-gate-pass-suite",
        configuration={"variant": "candidate"},
        samples=[0.101, 0.102, 0.103, 0.102, 0.104, 0.101, 0.103],
        environment_payload=environment_payload,
    )

    result = runner.invoke(
        app,
        ["compare", "1.1", "2.1", "--json", "--fail-if-regression", "5", "--database", str(database_path)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    result_payload = payload["result"]
    assert payload["status"] == "pass"
    assert result_payload["comparison_mode"] == "direct"
    assert result_payload["gate"]["failed"] is False
    assert result_payload["gate"]["failing_runs"] == []


def test_cli_compare_json_output_reports_suite_regression_gate_failure(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_sampled_run(
        database_path=database_path,
        suite_name="suite-gate-suite",
        configuration={"variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="suite-gate-suite",
        configuration={"variant": "candidate"},
        samples=[0.129, 0.130, 0.131, 0.130, 0.132, 0.129, 0.131],
        environment_payload=environment_payload,
    )

    result = runner.invoke(
        app,
        ["compare", "suite-gate-suite", "--json", "--fail-if-regression", "5%", "--database", str(database_path)],
    )

    assert result.exit_code == cli_module.REGRESSION_EXIT_CODE
    payload = json.loads(result.stdout)
    result_payload = payload["result"]
    assert payload["status"] == "fail"
    assert result_payload["comparison_mode"] == "suite"
    assert result_payload["basis_run"]["display_id"] == "1.1"
    assert result_payload["gate"]["failed"] is True
    assert result_payload["gate"]["failing_runs"][0]["display_id"] == "2.1"


def test_cli_baseline_json_output_includes_pin_update(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_run(
        database_path=database_path,
        suite_name="json-pin-suite",
        configuration={"variant": "baseline"},
        median_seconds=0.100,
        environment_payload=environment_payload,
    )
    _seed_run(
        database_path=database_path,
        suite_name="json-pin-suite",
        configuration={"variant": "candidate"},
        median_seconds=0.120,
        environment_payload=environment_payload,
    )

    result = runner.invoke(app, ["baseline", "json-pin-suite", "--pin", "2.1", "--note", "manual", "--json", "--database", str(database_path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    result_payload = payload["result"]
    assert payload["status"] == "pass"
    assert payload["reason"] == "baseline_pinned"
    assert result_payload["pin_update"]["display_id"] == "2.1"
    assert result_payload["current_baseline"]["run"]["display_id"] == "2.1"
    assert result_payload["current_baseline"]["note"] == "manual"


def test_cli_trend_json_output_is_machine_readable(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_sampled_run(
        database_path=database_path,
        suite_name="trend-json-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="trend-json-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.109, 0.110, 0.111, 0.110, 0.112, 0.109, 0.111],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="trend-json-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.139, 0.140, 0.141, 0.142, 0.140, 0.139, 0.141],
        environment_payload=environment_payload,
    )

    result = runner.invoke(
        app,
        ["trend", "trend-json-suite", "1.1", "--json", "--database", str(database_path)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    result_payload = payload["result"]
    assert payload["status"] == "fail"
    assert payload["reason"] == "regression_detected"
    assert result_payload["suite_name"] == "trend-json-suite"
    assert result_payload["basis_run"]["display_id"] == "1.1"
    assert len(result_payload["runs"]) == 3
    assert "Trend Basis:" not in result.stdout


def test_cli_trend_json_output_summarizes_mixed_configurations(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    runner = CliRunner()

    _seed_sampled_run(
        database_path=database_path,
        suite_name="summary-json-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="summary-json-suite",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.109, 0.110, 0.111, 0.110, 0.112, 0.109, 0.111],
        environment_payload=environment_payload,
    )
    _seed_sampled_run(
        database_path=database_path,
        suite_name="summary-json-suite",
        configuration={"size": 1024, "variant": "baseline"},
        samples=[0.199, 0.200, 0.201, 0.200, 0.199],
        environment_payload=environment_payload,
    )

    result = runner.invoke(
        app,
        ["trend", "summary-json-suite", "--json", "--database", str(database_path)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    result_payload = payload["result"]
    assert payload["status"] == "inconclusive"
    assert payload["reason"] == "multiple_configurations_summary"
    assert result_payload["mode"] == "summary"
    assert result_payload["suite_name"] == "summary-json-suite"
    assert result_payload["configuration_count"] == 2
    assert len(result_payload["config_summaries"]) == 2
    assert result_payload["config_summaries"][0]["latest_vs_first"]["classification"] == "regressing"
    assert "Trend Summary:" not in result.stdout
