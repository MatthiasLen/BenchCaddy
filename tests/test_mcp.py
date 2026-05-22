from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from benchcaddy.db import compare_runs as db_compare_runs
from benchcaddy.db import get_run_details as db_get_run_details
from benchcaddy.db import get_suite_baseline_history as db_get_suite_baseline_history
from benchcaddy.db import get_suite_details as db_get_suite_details
from benchcaddy.db import get_suite_trend as db_get_suite_trend
from benchcaddy.db import record_benchmark_run
from benchcaddy.db import set_suite_baseline as db_set_suite_baseline
from benchcaddy_mcp.tools import (
    DEFAULT_LIMIT,
    compare_runs,
    compare_suite,
    get_baseline_history,
    get_capabilities,
    get_run,
    get_suite,
    list_suites,
    pin_baseline,
    server_status,
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


def test_mcp_responses_use_consistent_envelope_fields(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(database_path=database_path, suite_name="suite-envelope", configuration={"variant": "baseline"})

    payloads = [
        ("server_status", server_status(str(database_path))),
        ("get_capabilities", get_capabilities(str(database_path))),
        ("list_suites", list_suites(str(database_path))),
        ("get_suite", get_suite("missing-suite", str(database_path))),
        ("get_run", get_run("not-a-run-id", str(database_path))),
        ("get_baseline_history", get_baseline_history("suite-envelope", str(database_path))),
    ]

    for command, payload in payloads:
        assert payload["schema_version"] == "2.0"
        assert payload["command"] == command
        assert payload["status"] in {"pass", "fail", "inconclusive"}
        assert isinstance(payload["reason"], str)
        assert payload["suggested_action"] is None or isinstance(payload["suggested_action"], str)
        assert "confidence" in payload
        assert payload["response_detail"] in {"summary", "full"}


def test_server_status_returns_ping_friendly_summary(tmp_path: Path) -> None:
    database_path = tmp_path / "missing.db"

    payload = server_status(str(database_path))

    assert payload["status"] == "pass"
    assert payload["reason"] == "server_ready"
    assert payload["response_detail"] == "summary"
    assert payload["summary"]["server_name"] == "BenchCaddy MCP"
    assert payload["summary"]["database"]["resolved_path"] == str(database_path)
    assert payload["summary"]["database"]["exists"] is False
    assert "result" not in payload


def test_get_capabilities_returns_predictable_contract(tmp_path: Path) -> None:
    database_path = tmp_path / "benchcaddy.db"

    payload = get_capabilities(str(database_path), response_detail="full")

    assert payload["status"] == "pass"
    assert payload["reason"] == "capabilities_available"
    assert payload["response_detail"] == "full"
    assert payload["summary"]["tool_count"] == 10
    assert payload["summary"]["tool_names"] == [
        "server_status",
        "get_capabilities",
        "list_suites",
        "get_suite",
        "get_run",
        "compare_suite",
        "compare_runs",
        "trend_suite",
        "get_baseline_history",
        "pin_baseline",
    ]
    assert payload["result"]["response_envelope_fields"] == [
        "schema_version",
        "command",
        "status",
        "reason",
        "error_code",
        "suggested_action",
        "confidence",
        "response_detail",
        "summary",
        "result",
    ]
    assert payload["result"]["tool_argument_conventions"]["response_detail"].startswith("Optional response detail mode")
    compare_suite_spec = next(tool for tool in payload["result"]["tools"] if tool["name"] == "compare_suite")
    assert compare_suite_spec["description"].startswith("Compare an entire suite against a chosen reference run")
    assert compare_suite_spec["when_to_use"].startswith("Use this when the user wants a suite-wide comparison")
    assert "compare nonlinear-transform against run 4.1" in compare_suite_spec["example_queries"]
    compare_runs_spec = next(tool for tool in payload["result"]["tools"] if tool["name"] == "compare_runs")
    assert compare_runs_spec["description"].startswith("Compare two specific benchmark runs directly")
    assert compare_runs_spec["when_to_use"].startswith("Use this when the user wants a direct run-vs-run comparison")
    assert "compare run 4.1 against run 4.2" in compare_runs_spec["example_queries"]
    trend_suite_spec = next(tool for tool in payload["result"]["tools"] if tool["name"] == "trend_suite")
    assert trend_suite_spec["description"].startswith("Inspect how a benchmark suite or one configuration changes over time")
    assert trend_suite_spec["when_to_use"].startswith("Use this when the user asks about history, drift")
    assert "show the trend for nonlinear-transform" in trend_suite_spec["example_queries"]
    assert payload["result"]["database"]["resolved_path"] == str(database_path)


def test_server_status_reports_invalid_response_detail(tmp_path: Path) -> None:
    payload = server_status(str(tmp_path / "benchcaddy.db"), response_detail="verbose")

    assert payload["status"] == "fail"
    assert payload["reason"] == "invalid_response_detail"
    assert payload["error_code"] == "invalid_response_detail"
    assert payload["summary"]["requested_response_detail"] == "verbose"


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
    assert payload["response_detail"] == "summary"
    assert payload["summary"]["database_path"] == str(database_path)
    assert payload["summary"]["suite_count"] == 2
    assert "result" not in payload


def test_list_suites_reports_empty_inventory_as_inconclusive(tmp_path: Path) -> None:
    payload = list_suites(str(tmp_path / "benchcaddy.db"))

    assert payload["status"] == "inconclusive"
    assert payload["reason"] == "no_suites_found"
    assert payload["suggested_action"] == "Record a benchmark sweep to create suite history, then call list_suites again or use get_capabilities to review the analysis tools."
    assert payload["summary"]["suites"] == []
    assert payload["summary"]["suite_count"] == 0


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
    assert payload["response_detail"] == "summary"
    assert payload["summary"]["mode"] == "suite"
    assert payload["summary"]["database_path"] == str(database_path)
    assert payload["summary"]["truncated"] is True
    assert payload["summary"]["total_run_count"] == DEFAULT_LIMIT + 5
    assert payload["summary"]["configuration_count"] == DEFAULT_LIMIT + 5
    assert len(payload["summary"]["available_configurations"]) == DEFAULT_LIMIT + 5
    assert len(payload["summary"]["latest_runs"]) == DEFAULT_LIMIT
    assert payload["summary"]["latest_runs"][0]["created_at"].endswith("Z")


def test_mcp_summary_and_full_payload_timestamps_are_explicit_utc(
    tmp_path: Path,
    environment_payload: dict[str, object],
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(
        database_path=database_path,
        suite_name="suite-timestamps",
        configuration={"variant": "baseline"},
        environment=environment_payload,
    )

    list_payload = list_suites(str(database_path))
    assert list_payload["summary"]["suites"][0]["last_run_at"].endswith("Z")

    suite_payload = get_suite("suite-timestamps", str(database_path))
    assert suite_payload["summary"]["latest_runs"][0]["created_at"].endswith("Z")
    assert suite_payload["summary"]["baseline_run"] is None or suite_payload["summary"]["baseline_run"]["created_at"].endswith("Z")

    run_payload = get_run("1.1", str(database_path), response_detail="full")
    assert run_payload["summary"]["run"]["created_at"].endswith("Z")
    assert run_payload["result"]["run"]["created_at"].endswith("Z")


def test_get_run_accepts_display_ids(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(database_path=database_path, suite_name="suite-run", configuration={"variant": "baseline"})

    payload = get_run("1.1", str(database_path))

    assert payload["status"] == "pass"
    assert payload["reason"] == "run_details_available"
    assert payload["response_detail"] == "summary"
    assert payload["summary"]["mode"] == "run"
    assert payload["summary"]["database_path"] == str(database_path)
    assert payload["summary"]["run"]["display_id"] == "1.1"


def test_db_get_run_details_can_build_lean_payload(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-run-lean",
        configuration={"variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )

    payload = db_get_run_details(
        (1, 1),
        str(database_path),
        include_samples=False,
        include_observations=False,
        include_environment=False,
    )

    assert payload is not None
    assert "samples" not in payload
    assert "observations" not in payload
    assert "environment" not in payload
    assert payload["observation_labels"] == []


def test_get_suite_reports_missing_suite_with_error_payload(tmp_path: Path) -> None:
    payload = get_suite("missing-suite", str(tmp_path / "benchcaddy.db"))

    assert payload["status"] == "fail"
    assert payload["reason"] == "suite_not_found"
    assert payload["error_code"] == "suite_not_found"
    assert payload["suggested_action"] == "Use list_suites to discover valid suite names, then call get_suite, compare_suite, or trend_suite for the suite you want."
    assert "result" not in payload


def test_get_run_reports_invalid_run_id_payload_fields(tmp_path: Path) -> None:
    payload = get_run("not-a-run-id", str(tmp_path / "benchcaddy.db"))

    assert payload["status"] == "fail"
    assert payload["reason"] == "invalid_run_id"
    assert payload["error_code"] == "invalid_run_id"
    assert payload["summary"]["requested_run_id"] == "not-a-run-id"
    assert payload["summary"]["message"] == "'not-a-run-id' is not a valid run ID."


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
    assert payload["summary"]["suite_name"] == "suite-filtered-payload"
    assert payload["summary"]["config_filter"] == {"size": 512, "variant": "baseline"}
    assert payload["summary"]["total_run_count"] == 2
    assert payload["summary"]["truncated"] is True
    assert payload["summary"]["configuration_count"] == 2
    assert payload["summary"]["available_configurations"] == [
        {"size": 512, "variant": "baseline"},
        {"size": 1024, "variant": "candidate"},
    ]
    assert len(payload["summary"]["latest_runs"]) == 1
    assert all(run["configuration"] == {"size": 512, "variant": "baseline"} for run in payload["summary"]["latest_runs"])


def test_db_get_suite_details_can_build_lean_payloads(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-details-lean",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-details-lean",
        configuration={"size": 1024, "variant": "candidate"},
        samples=[0.109, 0.110, 0.111, 0.110, 0.112, 0.109, 0.111],
        environment_payload=environment_payload,
    )

    payload = db_get_suite_details(
        "suite-details-lean",
        str(database_path),
        include_samples=False,
        include_observations=False,
    )

    assert payload is not None
    assert all("samples" not in run for run in payload["runs"])
    assert all("observations" not in run for run in payload["runs"])
    assert payload["environment"] is not None
    assert payload["baseline_run"] is None or "samples" not in payload["baseline_run"]


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
    assert payload["summary"]["comparison_mode"] == "suite"
    assert payload["summary"]["truncated"] is True
    assert payload["summary"]["total_run_count"] == DEFAULT_LIMIT + 3
    assert len(payload["summary"]["comparison_runs"]) == DEFAULT_LIMIT


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
    assert payload["response_detail"] == "summary"
    assert payload["summary"]["comparison_mode"] == "direct"
    assert payload["summary"]["percent_change"] == pytest.approx(2.0)
    assert payload["summary"]["same_configuration"] is False
    assert "result" not in payload


def test_compare_runs_full_detail_preserves_existing_payload(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-pair-full",
        configuration={"variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-pair-full",
        configuration={"variant": "candidate"},
        samples=[0.101, 0.102, 0.103, 0.102, 0.104, 0.101, 0.103],
        environment_payload=environment_payload,
    )

    payload = compare_runs("1.1", "2.1", str(database_path), response_detail="full")

    assert payload["response_detail"] == "full"
    assert payload["summary"]["comparison_mode"] == "direct"
    assert payload["result"]["comparison_mode"] == "direct"
    assert payload["result"]["percent_change"] == pytest.approx(2.0)


def test_db_compare_runs_can_build_lean_payloads(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-pair-lean",
        configuration={"variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-pair-lean",
        configuration={"variant": "candidate"},
        samples=[0.101, 0.102, 0.103, 0.102, 0.104, 0.101, 0.103],
        environment_payload=environment_payload,
    )

    payload = db_compare_runs(
        (1, 1),
        (2, 1),
        str(database_path),
        include_samples=False,
        include_observations=False,
        include_environment=False,
    )

    assert payload is not None
    assert "samples" not in payload["baseline"]
    assert "observations" not in payload["baseline"]
    assert "environment" not in payload["baseline"]
    assert "samples" not in payload["candidate"]
    assert "observations" not in payload["candidate"]
    assert "environment" not in payload["candidate"]
    assert isinstance(payload["comparison_analysis"]["percent_change"], float)
    assert isinstance(payload["observation_rows"], list)


def test_compare_runs_reports_missing_candidate_without_result_payload(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(database_path=database_path, suite_name="suite-missing-run", configuration={"variant": "baseline"})

    payload = compare_runs("1.1", "2.1", str(database_path))

    assert payload["status"] == "fail"
    assert payload["reason"] == "run_not_found"
    assert payload["error_code"] == "run_not_found"
    assert payload["suggested_action"] == "Use get_run to inspect that run ID or get_suite to browse valid run IDs in the target suite."
    assert "result" not in payload


def test_compare_runs_reports_regressing_payload_with_analysis(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-regressing-direct",
        configuration={"variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-regressing-direct",
        configuration={"variant": "candidate"},
        samples=[0.129, 0.130, 0.131, 0.130, 0.132, 0.129, 0.131],
        environment_payload=environment_payload,
    )

    payload = compare_runs("1.1", "2.1", str(database_path))

    assert payload["status"] == "fail"
    assert payload["reason"] == "regressing"
    assert payload["summary"]["comparison_mode"] == "direct"
    assert payload["summary"]["regression_detected"] is True
    assert payload["summary"]["classification"] == "regressing"


def test_compare_runs_reports_invalid_response_detail(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(database_path=database_path, suite_name="suite-invalid-detail", configuration={"variant": "baseline"})

    payload = compare_runs("1.1", "1.1", str(database_path), response_detail="verbose")

    assert payload["status"] == "fail"
    assert payload["reason"] == "invalid_response_detail"
    assert payload["error_code"] == "invalid_response_detail"
    assert payload["summary"]["requested_response_detail"] == "verbose"


def test_compare_suite_reports_invalid_reference_run_id(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(database_path=database_path, suite_name="suite-invalid-ref", configuration={"variant": "baseline"})

    payload = compare_suite("suite-invalid-ref", reference_run_id="bad-id", database_path=str(database_path))

    assert payload["status"] == "fail"
    assert payload["reason"] == "invalid_run_id"


def test_compare_suite_reports_wrong_suite_reference_payload_fields(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(database_path=database_path, suite_name="suite-left", configuration={"variant": "baseline"})
    record_simple_run(database_path=database_path, suite_name="suite-right", configuration={"variant": "candidate"})

    payload = compare_suite("suite-left", reference_run_id="2.1", database_path=str(database_path))

    assert payload["status"] == "fail"
    assert payload["reason"] == "reference_run_wrong_suite"
    assert payload["error_code"] == "reference_run_wrong_suite"
    assert payload["summary"]["reference_run_display_id"] == "2.1"
    assert payload["summary"]["reference_run_suite_name"] == "suite-right"


def test_compare_suite_reports_empty_scope_payload_context(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-empty-scope",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )

    payload = compare_suite("suite-empty-scope", config_filter={"size": 2048}, database_path=str(database_path))

    assert payload["status"] == "inconclusive"
    assert payload["reason"] == "no_runs_matched_scope"
    assert payload["summary"]["suite_name"] == "suite-empty-scope"
    assert payload["summary"]["config_filter"] == {"size": 2048}
    assert payload["summary"]["comparison_runs"] == []


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
    assert payload["summary"]["basis_source"] == "reference"
    assert payload["summary"]["basis_run"]["display_id"] == "1.1"
    assert payload["summary"]["config_filter"] == {"size": 512}
    assert payload["summary"]["comparison_verdict"] == "stable"
    assert {run["configuration"]["size"] for run in payload["summary"]["comparison_runs"]} == {512}
    assert {run["display_id"] for run in payload["summary"]["comparison_runs"]} == {"1.1", "2.1"}
    assert "target_return_relative_error" not in payload["summary"]["comparison_runs"][0]
    assert "slowdown_factor" not in payload["summary"]["comparison_runs"][0]


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
    assert payload["summary"]["mode"] == "timeline"
    assert payload["summary"]["truncated"] is True
    assert payload["summary"]["total_run_count"] == DEFAULT_LIMIT + 4
    assert payload["summary"]["trend_verdict"] == "stable"
    assert len(payload["summary"]["timeline_runs"]) == DEFAULT_LIMIT


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
    assert payload["summary"]["mode"] == "timeline"
    assert payload["summary"]["config_filter"] == {"size": 512, "variant": "baseline"}
    assert payload["summary"]["basis_source"] == "best"
    assert payload["summary"]["trend_verdict"] == "stable"
    assert payload["summary"]["total_run_count"] == 3
    assert payload["summary"]["truncated"] is True
    assert payload["summary"]["configuration_count"] == 2
    assert len(payload["summary"]["timeline_runs"]) == 2
    assert all(run["configuration"] == {"size": 512, "variant": "baseline"} for run in payload["summary"]["timeline_runs"])
    assert payload["summary"]["available_configurations"] == [
        {"size": 512, "variant": "baseline"},
        {"size": 1024, "variant": "baseline"},
    ]


def test_db_trend_suite_can_build_lean_filtered_timeline(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    stable_samples = [0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101]
    for _index in range(3):
        _record_sampled_run(
            database_path=database_path,
            suite_name="suite-trend-lean",
            configuration={"size": 512, "variant": "baseline"},
            samples=stable_samples,
            environment_payload=environment_payload,
        )

    payload = db_get_suite_trend(
        "suite-trend-lean",
        str(database_path),
        config_filter={"size": 512, "variant": "baseline"},
        include_samples=False,
        include_observations=False,
        include_environment=False,
    )

    assert payload is not None
    assert payload["mode"] == "timeline"
    assert "samples" not in payload["basis_run"]
    assert "observations" not in payload["basis_run"]
    assert "environment" not in payload["basis_run"]
    assert all("samples" not in run for run in payload["runs"])
    assert all("observations" not in run for run in payload["runs"])
    assert payload["runs"][0]["sample_count"] == 7


def test_trend_suite_reports_conflicting_basis_payload_fields(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-trend-conflict",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )

    payload = trend_suite(
        "suite-trend-conflict",
        baseline_run_id="1.1",
        config_filter={"size": 512},
        database_path=str(database_path),
    )

    assert payload["status"] == "fail"
    assert payload["reason"] == "config_filter_conflicts_with_basis"
    assert payload["error_code"] == "config_filter_conflicts_with_basis"
    assert payload["summary"]["suite_name"] == "suite-trend-conflict"
    assert payload["summary"]["config_filter"] == {"size": 512}


def test_trend_suite_reports_summary_mode_for_mixed_configurations(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-trend-summary",
        configuration={"size": 512, "variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-trend-summary",
        configuration={"size": 1024, "variant": "baseline"},
        samples=[0.109, 0.110, 0.111, 0.110, 0.112, 0.109, 0.111],
        environment_payload=environment_payload,
    )

    payload = trend_suite("suite-trend-summary", database_path=str(database_path))

    assert payload["status"] == "inconclusive"
    assert payload["reason"] == "multiple_configurations_summary"
    assert payload["summary"]["mode"] == "summary"
    assert payload["summary"]["configuration_count"] == 2
    assert payload["summary"]["trend_verdict"] == "stable"
    assert len(payload["summary"]["configuration_summaries"]) == 2
    assert payload["summary"]["available_configurations"] == [
        {"size": 512, "variant": "baseline"},
        {"size": 1024, "variant": "baseline"},
    ]
    assert "configuration" not in payload["summary"]["configuration_summaries"][0]["first_run"]
    assert "suite_name" not in payload["summary"]["configuration_summaries"][0]["first_run"]


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
        assert pin_payload["summary"]["pin_update"]["display_id"] == f"{index + 1}.1"

    payload = get_baseline_history("suite-baseline", str(database_path))

    assert payload["status"] == "pass"
    assert payload["reason"] == "baseline_available"
    assert payload["summary"]["truncated"] is True
    assert payload["summary"]["total_run_count"] == DEFAULT_LIMIT + 2
    assert len(payload["summary"]["baseline_history"]) == DEFAULT_LIMIT
    assert payload["summary"]["current_baseline"]["note"] == f"pin-{DEFAULT_LIMIT + 1}"


def test_db_baseline_history_and_pin_update_can_build_lean_payloads(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-baseline-lean",
        configuration={"variant": "baseline"},
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )

    pin_update = db_set_suite_baseline(
        "suite-baseline-lean",
        (1, 1),
        str(database_path),
        include_samples=False,
        include_observations=False,
        include_environment=False,
    )
    history = db_get_suite_baseline_history(
        "suite-baseline-lean",
        str(database_path),
        include_samples=False,
        include_observations=False,
        include_environment=False,
    )

    assert pin_update is not None
    assert "samples" not in pin_update
    assert "observations" not in pin_update
    assert "environment" not in pin_update
    assert history is not None
    assert "samples" not in history["history"][0]["run"]
    assert "observations" not in history["history"][0]["run"]
    assert "environment" not in history["history"][0]["run"]


def test_pin_baseline_reports_wrong_suite_payload_fields(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(database_path=database_path, suite_name="suite-pin-left", configuration={"variant": "baseline"})
    record_simple_run(database_path=database_path, suite_name="suite-pin-right", configuration={"variant": "candidate"})

    payload = pin_baseline("suite-pin-left", "2.1", database_path=str(database_path))

    assert payload["status"] == "fail"
    assert payload["reason"] == "reference_run_wrong_suite"
    assert payload["error_code"] == "reference_run_wrong_suite"
    assert payload["summary"]["reference_run_display_id"] == "2.1"
    assert payload["summary"]["reference_run_suite_name"] == "suite-pin-right"


def test_get_run_full_detail_preserves_existing_payload(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(database_path=database_path, suite_name="suite-run-full", configuration={"variant": "baseline"})

    payload = get_run("1.1", str(database_path), response_detail="full")

    assert payload["response_detail"] == "full"
    assert payload["summary"]["run"]["display_id"] == "1.1"
    assert payload["result"]["run"]["display_id"] == "1.1"


def test_get_suite_full_detail_preserves_existing_payload(
    tmp_path: Path,
    environment_payload: dict[str, object],
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(
        database_path=database_path,
        suite_name="suite-full",
        configuration={"variant": "baseline"},
        environment=environment_payload,
    )

    payload = get_suite("suite-full", str(database_path), response_detail="full")

    assert payload["response_detail"] == "full"
    assert payload["summary"]["suite_name"] == "suite-full"
    assert payload["result"]["suite_name"] == "suite-full"
    assert payload["result"]["runs"][0]["display_id"] == "1.1"


def test_compare_suite_full_detail_preserves_existing_payload(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    samples = [0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101]
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-compare-full",
        configuration={"variant": "baseline"},
        samples=samples,
        environment_payload=environment_payload,
    )
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-compare-full",
        configuration={"variant": "candidate"},
        samples=samples,
        environment_payload=environment_payload,
    )

    payload = compare_suite("suite-compare-full", database_path=str(database_path), response_detail="full")

    assert payload["response_detail"] == "full"
    assert payload["summary"]["comparison_mode"] == "suite"
    assert payload["result"]["comparison_mode"] == "suite"


def test_trend_suite_full_detail_preserves_existing_payload(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    samples = [0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101]
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-trend-full",
        configuration={"size": 512},
        samples=samples,
        environment_payload=environment_payload,
    )
    _record_sampled_run(
        database_path=database_path,
        suite_name="suite-trend-full",
        configuration={"size": 512},
        samples=samples,
        environment_payload=environment_payload,
    )

    payload = trend_suite("suite-trend-full", database_path=str(database_path), response_detail="full")

    assert payload["response_detail"] == "full"
    assert payload["summary"]["mode"] == "timeline"
    assert payload["result"]["mode"] == "timeline"


def test_pin_baseline_full_detail_preserves_existing_payload(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(database_path=database_path, suite_name="suite-pin-full", configuration={"variant": "baseline"})

    payload = pin_baseline("suite-pin-full", "1.1", database_path=str(database_path), response_detail="full")

    assert payload["response_detail"] == "full"
    assert payload["summary"]["pin_update"]["display_id"] == "1.1"
    assert payload["result"]["pin_update"]["display_id"] == "1.1"


def test_get_baseline_history_reports_missing_history_as_inconclusive(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(database_path=database_path, suite_name="suite-no-baseline", configuration={"variant": "baseline"})

    payload = get_baseline_history("suite-no-baseline", str(database_path))

    assert payload["status"] == "inconclusive"
    assert payload["reason"] == "no_baseline_history"


def test_get_baseline_history_reports_missing_suite_without_result_payload(tmp_path: Path) -> None:
    payload = get_baseline_history("missing-suite", str(tmp_path / "benchcaddy.db"))

    assert payload["status"] == "fail"
    assert payload["reason"] == "suite_not_found"
    assert payload["error_code"] == "suite_not_found"
    assert "result" not in payload