from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from fastmcp.client import Client, StdioTransport

from benchcaddy.db import record_benchmark_run

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


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


def _transport() -> StdioTransport:
    script_name = "benchcaddy-mcp.exe" if os.name == "nt" else "benchcaddy-mcp"
    script_path = Path(sys.executable).with_name(script_name)
    if script_path.exists():
        return StdioTransport(
            command=str(script_path),
            args=[],
            cwd=str(WORKSPACE_ROOT),
        )
    return StdioTransport(
        command=sys.executable,
        args=["-m", "benchcaddy_mcp.server"],
        cwd=str(WORKSPACE_ROOT),
    )


async def _call_tool(client: Client, name: str, arguments: dict[str, object]) -> dict[str, object]:
    result = await client.call_tool(name, arguments)
    payload = result.structured_content or result.data
    assert isinstance(payload, dict)
    return payload


def test_live_mcp_stdio_smoke(
    tmp_path: Path,
    environment_payload: dict[str, object],
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    suite_name = "suite-live"
    trend_filter = {"size": 512, "variant": "baseline"}

    _record_sampled_run(
        database_path=database_path,
        suite_name=suite_name,
        configuration=trend_filter,
        samples=[0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        environment_payload=environment_payload,
    )
    _record_sampled_run(
        database_path=database_path,
        suite_name=suite_name,
        configuration=trend_filter,
        samples=[0.098, 0.099, 0.100, 0.099, 0.101, 0.098, 0.100],
        environment_payload=environment_payload,
    )
    _record_sampled_run(
        database_path=database_path,
        suite_name=suite_name,
        configuration={"size": 512, "variant": "candidate"},
        samples=[0.109, 0.110, 0.111, 0.110, 0.112, 0.109, 0.111],
        environment_payload=environment_payload,
    )
    _record_sampled_run(
        database_path=database_path,
        suite_name=suite_name,
        configuration={"size": 1024, "variant": "baseline"},
        samples=[0.129, 0.130, 0.131, 0.130, 0.132, 0.129, 0.131],
        environment_payload=environment_payload,
    )

    async def run_smoke() -> None:
        async with Client(_transport()) as client:
            tools = await client.list_tools()
            assert {tool.name for tool in tools} == {
                "compare_runs",
                "compare_suite",
                "get_capabilities",
                "get_baseline_history",
                "get_run",
                "get_suite",
                "list_suites",
                "pin_baseline",
                "server_status",
                "trend_suite",
            }

            status_payload = await _call_tool(client, "server_status", {"database_path": str(database_path)})
            assert status_payload["status"] == "pass"
            assert status_payload["reason"] == "server_ready"
            assert status_payload["summary"]["database"]["exists"] is True

            capabilities_payload = await _call_tool(client, "get_capabilities", {"database_path": str(database_path)})
            assert capabilities_payload["status"] == "pass"
            assert capabilities_payload["reason"] == "capabilities_available"
            assert capabilities_payload["summary"]["tool_count"] == 10
            assert "compare_runs" in capabilities_payload["summary"]["tool_names"]

            list_payload = await _call_tool(client, "list_suites", {"database_path": str(database_path)})
            assert list_payload["status"] == "pass"
            assert list_payload["reason"] == "suite_inventory_available"
            assert list_payload["response_detail"] == "summary"
            assert list_payload["summary"]["suite_count"] == 1
            assert list_payload["summary"]["suites"][0]["suite_name"] == suite_name

            suite_payload = await _call_tool(client, "get_suite", {"suite_name": suite_name, "database_path": str(database_path)})
            assert suite_payload["status"] == "pass"
            assert suite_payload["reason"] == "suite_details_available"
            assert suite_payload["summary"]["total_run_count"] == 4

            run_a = suite_payload["summary"]["runs"][0]["display_id"]
            run_b = suite_payload["summary"]["runs"][1]["display_id"]

            run_payload = await _call_tool(client, "get_run", {"run_id": run_a, "database_path": str(database_path)})
            assert run_payload["status"] == "pass"
            assert run_payload["reason"] == "run_details_available"
            assert run_payload["summary"]["run"]["display_id"] == run_a

            compare_runs_payload = await _call_tool(
                client,
                "compare_runs",
                {"left_run_id": run_a, "right_run_id": run_b, "database_path": str(database_path)},
            )
            assert compare_runs_payload["status"] == "pass"
            assert compare_runs_payload["reason"] == "comparison_complete"
            assert isinstance(compare_runs_payload["summary"]["percent_change"], float)
            assert isinstance(compare_runs_payload["summary"]["classification"], str)

            compare_runs_full_payload = await _call_tool(
                client,
                "compare_runs",
                {
                    "left_run_id": run_a,
                    "right_run_id": run_b,
                    "database_path": str(database_path),
                    "response_detail": "full",
                },
            )
            assert compare_runs_full_payload["response_detail"] == "full"
            assert isinstance(compare_runs_full_payload["result"]["percent_change"], float)

            compare_suite_payload = await _call_tool(
                client,
                "compare_suite",
                {
                    "suite_name": suite_name,
                    "database_path": str(database_path),
                    "config_filter": trend_filter,
                    "limit": 2,
                },
            )
            assert compare_suite_payload["status"] == "pass"
            assert compare_suite_payload["reason"] == "comparison_complete"
            assert compare_suite_payload["summary"]["comparison_mode"] == "suite"
            assert compare_suite_payload["summary"]["total_run_count"] == 2
            assert compare_suite_payload["summary"]["truncated"] is False

            trend_payload = await _call_tool(
                client,
                "trend_suite",
                {
                    "suite_name": suite_name,
                    "database_path": str(database_path),
                    "config_filter": trend_filter,
                    "limit": 2,
                },
            )
            assert trend_payload["status"] == "pass"
            assert trend_payload["reason"] == "trend_timeline_available"
            assert trend_payload["summary"]["mode"] == "timeline"
            assert trend_payload["summary"]["total_run_count"] == 2
            assert trend_payload["summary"]["truncated"] is False

            baseline_history_payload = await _call_tool(
                client,
                "get_baseline_history",
                {"suite_name": suite_name, "database_path": str(database_path)},
            )
            assert baseline_history_payload["status"] == "inconclusive"
            assert baseline_history_payload["reason"] == "no_baseline_history"

            pin_payload = await _call_tool(
                client,
                "pin_baseline",
                {
                    "suite_name": suite_name,
                    "run_id": run_a,
                    "database_path": str(database_path),
                    "note": "live-smoke",
                },
            )
            assert pin_payload["status"] == "pass"
            assert pin_payload["reason"] == "baseline_pinned"
            assert "pin_update" in pin_payload["summary"]

            baseline_history_after_pin = await _call_tool(
                client,
                "get_baseline_history",
                {"suite_name": suite_name, "database_path": str(database_path)},
            )
            assert baseline_history_after_pin["status"] == "pass"
            assert baseline_history_after_pin["reason"] == "baseline_available"

    asyncio.run(run_smoke())


def test_live_mcp_reports_invalid_run_id_over_stdio(
    tmp_path: Path,
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"
    record_simple_run(database_path=database_path, suite_name="suite-invalid", configuration={"variant": "baseline"})

    async def run_invalid_check() -> None:
        async with Client(_transport()) as client:
            payload = await _call_tool(
                client,
                "compare_runs",
                {
                    "left_run_id": "bad-id",
                    "right_run_id": "1.1",
                    "database_path": str(database_path),
                },
            )

            assert payload["status"] == "fail"
            assert payload["reason"] == "invalid_run_id"
            assert payload["error_code"] == "invalid_run_id"
            assert payload["summary"]["requested_run_id"] == "bad-id"
            assert payload["summary"]["message"] == "'bad-id' is not a valid run ID."

    asyncio.run(run_invalid_check())