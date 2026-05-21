from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..stats import AnalysisOptions
from ._sqlite.session import db_session
from ._sqlite.store import (
    _collect_observation_labels,
    _count_all_runs,
    _count_suite_runs,
    _get_suite,
    _list_all_runs_latest_first,
    _list_all_suites,
    _list_suite_runs_created_desc,
    _list_suite_runs_latest_first,
    _resolve_run,
    _resolve_suite_baseline_run,
)


def list_suite_summaries(database_path: str | Path | None = None) -> list[dict[str, Any]]:
    with db_session(database_path) as session:
        suites = _list_all_suites(session)
        summaries: list[dict[str, Any]] = []
        for suite in suites:
            # Summaries derive labels from the full stored run history because observed probes can change over time.
            runs = _list_suite_runs_created_desc(session, suite.id)
            if not runs:
                continue
            summaries.append(
                {
                    "suite_name": suite.name,
                    "target_name": suite.target_name,
                    "run_count": len(runs),
                    "last_run_at": runs[0].created_at,
                    "observation_labels": _collect_observation_labels(run.observations for run in runs),
                }
            )

    return summaries


def get_suite_details(
    suite_name: str,
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
    *,
    limit: int | None = None,
    include_analysis: bool = False,
) -> dict[str, Any] | None:
    with db_session(database_path) as session:
        suite = _get_suite(session, suite_name)
        if suite is None:
            return None

        runs = _list_suite_runs_latest_first(session, suite.id, limit=limit)
        # Suite views show the newest recorded environment snapshot alongside the selected run slice.
        environment = runs[0].environment if runs else None
        baseline_run = _resolve_suite_baseline_run(session, suite)

    run_payloads = [run.to_payload(analysis_options, include_analysis=include_analysis) for run in runs]

    return {
        "suite_name": suite.name,
        "target_name": suite.target_name,
        "runs": run_payloads,
        "environment": None if environment is None else environment.to_payload(),
        "baseline_run": None if baseline_run is None else baseline_run.to_payload(analysis_options, include_analysis=include_analysis),
    }


def get_run_details(
    run_id: int | tuple[int, int],
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
    *,
    include_analysis: bool = False,
) -> dict[str, Any] | None:
    with db_session(database_path) as session:
        run = _resolve_run(session, run_id)
        if run is None:
            return None
        return run.to_detail_payload(analysis_options, include_analysis=include_analysis)


def get_selected_run_details(
    run_ids: Sequence[int | tuple[int, int]],
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
    *,
    include_analysis: bool = False,
) -> list[dict[str, Any]] | None:
    # Drop duplicate requests without disturbing the caller's original run-id ordering.
    unique_run_ids = list(dict.fromkeys(run_ids))
    with db_session(database_path) as session:
        runs = [_resolve_run(session, run_id) for run_id in unique_run_ids]
        if any(run is None for run in runs):
            return None

        run_payloads = [run.to_detail_payload(analysis_options, include_analysis=include_analysis) for run in runs if run is not None]

    # Normalize the final payload to newest-first regardless of the input order.
    return sorted(run_payloads, key=lambda run: -int(run["id"]))


def get_all_run_details(
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
    *,
    limit: int | None = None,
    include_analysis: bool = False,
) -> list[dict[str, Any]]:
    with db_session(database_path) as session:
        runs = _list_all_runs_latest_first(session, limit=limit)
        return [
            {
                **run.to_payload(analysis_options, include_analysis=include_analysis),
                "suite_name": run.suite.name,
                "target_name": run.suite.target_name,
            }
            for run in runs
        ]


def get_all_run_count(database_path: str | Path | None = None) -> int:
    with db_session(database_path) as session:
        return _count_all_runs(session)


def get_suite_run_count(suite_name: str, database_path: str | Path | None = None) -> int | None:
    with db_session(database_path) as session:
        suite = _get_suite(session, suite_name)
        if suite is None:
            return None
        return _count_suite_runs(session, suite.id)
