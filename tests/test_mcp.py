from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from benchcaddy.db import record_benchmark_run
from benchcaddy_mcp.tools import (
    DEFAULT_LIMIT,
    compare_runs,
    compare_suite,
    get_baseline_history,
    get_run,
    get_suite,
    list_suites,
    pin_baseline,
    trend_suite,
)


def _record_sampled_run(
    *,
    database_path: Path,
    suite_name: str,
    configuration: dict[str, object],
    samples: list[float],
    environment_payload: dict[str, object],
) -> None:
    median_seconds = sorted(samples)[len(samples) // 2]
    record_benchmark_run(
        suite_name=suite_name,
        target_name="benchmark_target",
        configuration=configuration,
        samples=samples,
        observations=[],
        median_seconds=median_seconds,
        min_seconds=min(samples),
        max_seconds=max(samples),
        std_seconds=0.0,
        environment=environment_payload,
        database_path=database_path,
    )


def test_benchcaddy_mcp_entrypoint_invokes_app_run(monkeypatch) -> None:
    server_module = importlib.import_module("benchcaddy_mcp.server")
    calls: list[str] = []

    monkeypatch.setattr(server_module.app, "run", lambda: calls.append("run"))

    server_module.main()

    assert calls == ["run"]


def test_list_suites_returns_machine_readable_inventory(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(database_path=database_path, suite_name="suite-a", configuration={"variant": "baseline"})
    record_simple_run(database_path=database_path, suite_name="suite-b", configuration={"variant": "candidate"})

    payload = list_suites(str(database_path))

    assert payload["status"] == "pass"
    assert payload["reason"] == "suite_inventory_available"
    assert payload["result"]["database_path"] == str(database_path)
    assert payload["result"]["suite_count"] == 2


def test_list_suites_reports_empty_inventory_as_inconclusive(tmp_path: Path) -> None:
    payload = list_suites(str(tmp_path / "benchcaddy.db"))

    assert payload["status"] == "inconclusive"
    assert payload["reason"] == "no_suites_found"
    assert payload["result"]["suite_count"] == 0


def test_get_suite_caps_runs_by_default(
    tmp_path: Path,
    environment_payload: dict[str, object],
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    for index in range(DEFAULT_LIMIT + 5):
        record_simple_run(
            database_path=database_path,
            suite_name="suite-capped",
            configuration={"variant": index},
            median_seconds=0.100 + (index / 1000),
            environment=environment_payload,
        )

    payload = get_suite("suite-capped", str(database_path))

    assert payload["status"] == "pass"
    assert payload["reason"] == "suite_details_available"
    assert payload["result"]["mode"] == "suite"
    assert payload["result"]["database_path"] == str(database_path)
    assert payload["result"]["truncated"] is True
    assert payload["result"]["total_run_count"] == DEFAULT_LIMIT + 5
    assert len(payload["result"]["runs"]) == DEFAULT_LIMIT


def test_get_run_accepts_display_ids(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(database_path=database_path, suite_name="suite-run", configuration={"variant": "baseline"})

    payload = get_run("1.1", str(database_path))

    assert payload["status"] == "pass"
    assert payload["reason"] == "run_details_available"
    assert payload["result"]["mode"] == "run"
    assert payload["result"]["database_path"] == str(database_path)
    assert payload["result"]["run"]["display_id"] == "1.1"


def test_get_suite_reports_config_filter_miss_as_inconclusive(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(database_path=database_path, suite_name="suite-filter", configuration={"variant": "baseline"})

    payload = get_suite("suite-filter", str(database_path), config_filter={"variant": "candidate"})

    assert payload["status"] == "inconclusive"
    assert payload["reason"] == "no_runs_matched_scope"


def test_get_suite_payload_matches_requested_filter_and_limit(
    tmp_path: Path,
    environment_payload: dict[str, object],
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(
        database_path=database_path,
        suite_name="suite-filtered-payload",
        configuration={"size": 512, "variant": "baseline"},
        median_seconds=0.100,
        environment=environment_payload,
    )
    record_simple_run(
        database_path=database_path,
        suite_name="suite-filtered-payload",
        configuration={"size": 512, "variant": "baseline"},
        median_seconds=0.101,
        environment=environment_payload,
    )
    record_simple_run(
        database_path=database_path,
        suite_name="suite-filtered-payload",
        configuration={"size": 1024, "variant": "candidate"},
        median_seconds=0.130,
        environment=environment_payload,
    )

    payload = get_suite(
        "suite-filtered-payload",
        str(database_path),
        limit=1,
        config_filter={"size": 512, "variant": "baseline"},
    )

    assert payload["status"] == "pass"
    assert payload["result"]["suite_name"] == "suite-filtered-payload"
    assert payload["result"]["config_filter"] == {"size": 512, "variant": "baseline"}
    assert payload["result"]["total_run_count"] == 2
    assert payload["result"]["truncated"] is True
    assert len(payload["result"]["runs"]) == 1
    assert all(run["configuration"] == {"size": 512, "variant": "baseline"} for run in payload["result"]["runs"])


def test_compare_suite_caps_runs_by_default(
    tmp_path: Path,
    environment_payload: dict[str, object],
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    samples = [0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101]
    for index in range(DEFAULT_LIMIT + 3):
        _record_sampled_run(
            database_path=database_path,
            suite_name="suite-compare",
            configuration={"variant": index},
            samples=samples,
            environment_payload=environment_payload,
        )

    payload = compare_suite("suite-compare", database_path=str(database_path))

    assert payload["status"] == "pass"
    assert payload["reason"] == "comparison_complete"
    assert payload["result"]["comparison_mode"] == "suite"
    assert payload["result"]["truncated"] is True
    assert payload["result"]["total_run_count"] == DEFAULT_LIMIT + 3
    assert len(payload["result"]["runs"]) == DEFAULT_LIMIT


def test_compare_runs_returns_head_to_head_payload(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-pair",
        configuration={"variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-pair",
        configuration={"variant": "candidate"},
        samples=[0.101, 0.102, 0.103, 0.102, 0.104, 0.101, 0.103],
        environment_payload=environment_payload,
    )

    payload = compare_runs("1.1", "2.1", str(database_path))

    assert payload["status"] == "pass"
    assert payload["reason"] == "comparison_complete"
    assert payload["result"]["comparison_mode"] == "direct"
    assert payload["result"]["percent_change"] == pytest.approx(2.0)


def test_compare_suite_reports_invalid_reference_run_id(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(database_path=database_path, suite_name="suite-invalid-ref", configuration={"variant": "baseline"})

    payload = compare_suite("suite-invalid-ref", reference_run_id="bad-id", database_path=str(database_path))

    assert payload["status"] == "fail"
    assert payload["reason"] == "invalid_run_id"


def test_compare_suite_payload_matches_requested_reference_and_scope(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    stable_samples = [0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101]
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-compare-payload",
        configuration={"size": 512, "variant": "baseline"},
        samples=stable_samples,
        environment_payload=environment_payload,
    )
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-compare-payload",
        configuration={"size": 512, "variant": "candidate"},
        samples=stable_samples,
        environment_payload=environment_payload,
    )
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-compare-payload",
        configuration={"size": 1024, "variant": "candidate"},
        samples=stable_samples,
        environment_payload=environment_payload,
    )

    payload = compare_suite(
        "suite-compare-payload",
        reference_run_id="1.1",
        config_filter={"size": 512},
        database_path=str(database_path),
    )

    assert payload["status"] == "pass"
    assert payload["result"]["basis_source"] == "reference"
    assert payload["result"]["basis_run"]["display_id"] == "1.1"
    assert payload["result"]["config_filter"] == {"size": 512}
    assert {run["configuration"]["size"] for run in payload["result"]["runs"]} == {512}
    assert {run["display_id"] for run in payload["result"]["runs"]} == {"1.1", "2.1"}


def test_trend_suite_caps_timeline_runs_by_default(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    samples = [0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101]
    for _index in range(DEFAULT_LIMIT + 4):
        _record_sampled_run(
            database_path=database_path,
            suite_name="suite-trend",
            configuration={"size": 512},
            samples=samples,
            environment_payload=environment_payload,
        )

    payload = trend_suite("suite-trend", database_path=str(database_path))

    assert payload["status"] == "pass"
    assert payload["reason"] == "trend_timeline_available"
    assert payload["result"]["mode"] == "timeline"
    assert payload["result"]["truncated"] is True
    assert payload["result"]["total_run_count"] == DEFAULT_LIMIT + 4
    assert len(payload["result"]["runs"]) == DEFAULT_LIMIT


def test_trend_suite_payload_matches_requested_filter_context(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    stable_samples = [0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101]
    for _index in range(3):
        _record_sampled_run(
            database_path=database_path,
            suite_name="suite-trend-filtered",
            configuration={"size": 512, "variant": "baseline"},
            samples=stable_samples,
            environment_payload=environment_payload,
        )
    for _index in range(2):
        _record_sampled_run(
            database_path=database_path,
            suite_name="suite-trend-filtered",
            configuration={"size": 1024, "variant": "baseline"},
            samples=stable_samples,
            environment_payload=environment_payload,
        )

    payload = trend_suite(
        "suite-trend-filtered",
        config_filter={"size": 512, "variant": "baseline"},
        limit=2,
        database_path=str(database_path),
    )

    assert payload["status"] == "pass"
    assert payload["result"]["mode"] == "timeline"
    assert payload["result"]["config_filter"] == {"size": 512, "variant": "baseline"}
    assert payload["result"]["basis_source"] == "best"
    assert payload["result"]["total_run_count"] == 3
    assert payload["result"]["truncated"] is True
    assert len(payload["result"]["runs"]) == 2
    assert all(run["configuration"] == {"size": 512, "variant": "baseline"} for run in payload["result"]["runs"])


def test_pin_baseline_and_history_are_available(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    for index in range(DEFAULT_LIMIT + 2):
        record_simple_run(
            database_path=database_path,
            suite_name="suite-baseline",
            configuration={"variant": index},
            median_seconds=0.100 + (index / 1000),
        )
        pin_payload = pin_baseline("suite-baseline", f"{index + 1}.1", database_path=str(database_path), note=f"pin-{index}")
        assert pin_payload["status"] == "pass"
        assert pin_payload["reason"] == "baseline_pinned"
        assert pin_payload["result"]["pin_update"]["display_id"] == f"{index + 1}.1"

    payload = get_baseline_history("suite-baseline", str(database_path))

    assert payload["status"] == "pass"
    assert payload["reason"] == "baseline_available"
    assert payload["result"]["truncated"] is True
    assert payload["result"]["total_run_count"] == DEFAULT_LIMIT + 2
    assert len(payload["result"]["history"]) == DEFAULT_LIMIT
    assert payload["result"]["current_baseline"]["note"] == f"pin-{DEFAULT_LIMIT + 1}"


def test_get_baseline_history_reports_missing_history_as_inconclusive(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(database_path=database_path, suite_name="suite-no-baseline", configuration={"variant": "baseline"})

    payload = get_baseline_history("suite-no-baseline", str(database_path))

    assert payload["status"] == "inconclusive"
    assert payload["reason"] == "no_baseline_history"