from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..observability import summarize_observations
from ..return_values import return_relative_error
from ..stats import AnalysisOptions, compare_sample_sets
from ._sqlalchemy.models import BenchmarkRun, BenchmarkSuite
from ._sqlalchemy.session import db_session
from ._sqlalchemy.store import (
    _get_suite,
    _list_suite_runs_latest_first,
    _list_suite_runs_oldest_first,
    _resolve_run,
    _resolve_suite_baseline_run,
)
from .read import get_run_details


def compare_suite_runs(
    suite_name: str,
    reference_run_id: int | tuple[int, int] | None = None,
    strict_keys: Sequence[str] = (),
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
    use_pinned_baseline: bool = False,
) -> dict[str, Any] | None:
    with db_session(database_path) as session:
        suite = _get_suite(session, suite_name)
        if suite is None:
            return None

        reference_context = _resolve_suite_reference(
            session,
            suite,
            reference_run_id,
            use_pinned_baseline=use_pinned_baseline,
        )
        if isinstance(reference_context, dict):
            return reference_context
        reference_run, pinned_baseline_run, reference_run_id, reference_run_suite_name, reference_from_pinned = reference_context
        runs = _list_suite_runs_latest_first(session, suite.id)

    if not runs:
        return _empty_suite_comparison_payload(suite)

    basis = _resolve_suite_basis(
        suite,
        runs,
        reference_run,
        reference_run_id,
        reference_run_suite_name,
    )
    if isinstance(basis, dict):
        return basis
    basis_run, basis_metric_label, delta_column_label, ratio_column_label = basis

    strict_result = _apply_strict_comparison_filter(
        suite,
        runs,
        basis_run,
        strict_keys,
        reference_run_id,
    )
    if isinstance(strict_result, dict):
        return strict_result
    filtered_runs, strict_keys, strict_config = strict_result

    return _suite_comparison_payload(
        suite=suite,
        runs=filtered_runs,
        basis_run=basis_run,
        basis_metric_label=basis_metric_label,
        delta_column_label=delta_column_label,
        ratio_column_label=ratio_column_label,
        strict_keys=strict_keys,
        strict_config=strict_config,
        reference_from_pinned=reference_from_pinned,
        reference_run_id=reference_run_id,
        pinned_baseline_run=pinned_baseline_run,
        analysis_options=analysis_options,
    )


def compare_runs(
    baseline_run_id: int | tuple[int, int],
    candidate_run_id: int | tuple[int, int],
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
) -> dict[str, Any] | None:
    baseline = get_run_details(
        baseline_run_id,
        database_path,
        analysis_options=analysis_options,
        include_analysis=True,
    )
    candidate = get_run_details(
        candidate_run_id,
        database_path,
        analysis_options=analysis_options,
        include_analysis=True,
    )
    if baseline is None or candidate is None:
        return None

    percent_change = None
    if baseline["median_seconds"] > 0:
        percent_change = ((candidate["median_seconds"] - baseline["median_seconds"]) / baseline["median_seconds"]) * 100.0

    baseline_observations = summarize_observations(baseline["observations"])
    candidate_observations = summarize_observations(candidate["observations"])
    labels = sorted(set(baseline_observations) | set(candidate_observations))
    target_return_relative_error = return_relative_error(
        reference_value=baseline["target_return_value"],
        candidate_value=candidate["target_return_value"],
    )

    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta_seconds": candidate["median_seconds"] - baseline["median_seconds"],
        "percent_change": percent_change,
        "target_return_relative_error": target_return_relative_error,
        "comparison_analysis": compare_sample_sets(
            baseline["samples"],
            candidate["samples"],
            analysis_options,
        ).to_payload(),
        "observation_rows": [
            {
                "label": label,
                "baseline_mean_seconds": baseline_observations[label].mean_seconds if label in baseline_observations else None,
                "baseline_std_seconds": baseline_observations[label].std_seconds if label in baseline_observations else None,
                "candidate_mean_seconds": candidate_observations[label].mean_seconds if label in candidate_observations else None,
                "candidate_std_seconds": candidate_observations[label].std_seconds if label in candidate_observations else None,
                "delta_seconds": (
                    candidate_observations[label].mean_seconds - baseline_observations[label].mean_seconds
                    if label in baseline_observations and label in candidate_observations
                    else None
                ),
            }
            for label in labels
        ],
    }


def get_suite_trend(
    suite_name: str,
    database_path: str | Path | None = None,
    analysis_options: AnalysisOptions | None = None,
    baseline_run_id: int | tuple[int, int] | None = None,
    use_pinned_baseline: bool = False,
    limit: int | None = None,
) -> dict[str, Any] | None:
    chosen_options = analysis_options or AnalysisOptions()
    with db_session(database_path) as session:
        suite = _get_suite(session, suite_name)
        if suite is None:
            return None

        runs = _list_suite_runs_oldest_first(session, suite.id)
        if not runs:
            return {
                "mode": "timeline",
                "suite_name": suite.name,
                "target_name": suite.target_name,
                "basis_run": None,
                "basis_source": "empty",
                "config_filter": None,
                "runs": [],
            }

        if baseline_run_id is not None:
            basis_run = _resolve_run(session, baseline_run_id)
            if basis_run is None:
                return {"error": "reference_run_not_found", "suite_name": suite.name}
            if basis_run.suite_id != suite.id:
                return {
                    "error": "reference_run_wrong_suite",
                    "suite_name": suite.name,
                    "reference_run_display_id": basis_run.display_id,
                    "reference_run_record_id": basis_run.id,
                    "reference_run_suite_name": basis_run.suite.name,
                }
            basis_source = "explicit"
        else:
            basis_run = _resolve_suite_baseline_run(session, suite)
            if use_pinned_baseline:
                if basis_run is None:
                    return {"error": "baseline_not_found", "suite_name": suite.name}
                basis_source = "pinned"
            else:
                grouped_runs: dict[str, list[BenchmarkRun]] = {}
                for run in runs:
                    grouped_runs.setdefault(_configuration_group_key(run.configuration), []).append(run)

                if basis_run is None and len(grouped_runs) > 1:
                    return {
                        "mode": "summary",
                        "suite_name": suite.name,
                        "target_name": suite.target_name,
                        "configuration_count": len(grouped_runs),
                        "limit": limit,
                        "config_summaries": [_configuration_trend_summary_payload(grouped_runs[key], chosen_options, limit=limit) for key in grouped_runs],
                    }

                if basis_run is not None:
                    basis_source = "pinned"
                else:
                    basis_run = runs[-1]
                    basis_source = "latest"

        config_filter = dict(basis_run.configuration)
        filtered_runs = [run for run in runs if run.configuration == config_filter]
        if limit is not None:
            filtered_runs = filtered_runs[-limit:]
        basis_payload = basis_run.to_detail_payload(chosen_options)
        basis_samples = list(basis_run.samples)
        basis_run_id = basis_run.id
        filtered_payloads = [run.to_payload(chosen_options) for run in filtered_runs]
        filtered_samples = [list(run.samples) for run in filtered_runs]
        filtered_ids = [run.id for run in filtered_runs]

    trend_rows: list[dict[str, Any]] = []
    for index, payload in enumerate(filtered_payloads):
        run_samples = filtered_samples[index]
        vs_basis = compare_sample_sets(basis_samples, run_samples, chosen_options).to_payload()
        trailing_samples = [sample for prior_samples in filtered_samples[max(0, index - chosen_options.drift_window_size) : index] for sample in prior_samples]
        drift_analysis = None
        drift_status_label = "baseline" if payload["id"] == basis_run_id else "stable"
        if trailing_samples:
            drift_analysis = compare_sample_sets(trailing_samples, run_samples, chosen_options).to_payload()
            drift_status_label = str(drift_analysis["classification"])

        trend_rows.append(
            {
                **payload,
                "vs_baseline": vs_basis,
                "drift_analysis": drift_analysis,
                "drift_status": drift_status_label,
                "is_basis": filtered_ids[index] == basis_run_id,
            }
        )

    return {
        "mode": "timeline",
        "suite_name": suite_name,
        "target_name": basis_payload["target_name"],
        "basis_run": basis_payload,
        "basis_source": basis_source,
        "config_filter": config_filter,
        "runs": trend_rows,
    }


def _resolve_suite_reference(
    session,
    suite: BenchmarkSuite,
    reference_run_id: int | tuple[int, int] | None,
    *,
    use_pinned_baseline: bool,
) -> tuple[BenchmarkRun | None, BenchmarkRun | None, int | tuple[int, int] | None, str | None, bool] | dict[str, Any]:
    reference_run = None
    reference_from_pinned = False
    reference_run_suite_name = None
    pinned_baseline_run = _resolve_suite_baseline_run(session, suite)

    if reference_run_id is not None:
        reference_run = _resolve_run(session, reference_run_id)
        if reference_run is not None:
            reference_run_suite_name = reference_run.suite.name
    elif use_pinned_baseline:
        reference_run = pinned_baseline_run
        if reference_run is None:
            return {
                "error": "baseline_not_found",
                "suite_name": suite.name,
            }
        reference_from_pinned = True
        reference_run_id = reference_run.id
        reference_run_suite_name = reference_run.suite.name

    return (
        reference_run,
        pinned_baseline_run,
        reference_run_id,
        reference_run_suite_name,
        reference_from_pinned,
    )


def _empty_suite_comparison_payload(suite: BenchmarkSuite) -> dict[str, Any]:
    return {
        "suite_name": suite.name,
        "target_name": suite.target_name,
        "basis_median_seconds": None,
        "basis_run": None,
        "basis_metric_label": "Best Median (s)",
        "delta_column_label": "Delta vs Best (s)",
        "ratio_column_label": "Slowdown",
        "runs": [],
        "pinned_baseline": None,
    }


def _resolve_suite_basis(
    suite: BenchmarkSuite,
    runs: Sequence[BenchmarkRun],
    reference_run: BenchmarkRun | None,
    reference_run_id: int | tuple[int, int] | None,
    reference_run_suite_name: str | None,
) -> tuple[BenchmarkRun, str, str, str] | dict[str, Any]:
    if reference_run_id is None:
        return (
            min(runs, key=lambda run: (run.median_seconds, run.id)),
            "Best Median (s)",
            "Delta vs Best (s)",
            "Slowdown",
        )
    if reference_run is None:
        return {
            "error": "reference_run_not_found",
            "suite_name": suite.name,
        }
    if reference_run.suite_id != suite.id:
        return {
            "error": "reference_run_wrong_suite",
            "suite_name": suite.name,
            "reference_run_display_id": reference_run.display_id,
            "reference_run_record_id": reference_run.id,
            "reference_run_suite_name": reference_run_suite_name,
        }
    return (
        reference_run,
        "Reference Median (s)",
        "Delta vs Reference (s)",
        "Relative",
    )


def _apply_strict_comparison_filter(
    suite: BenchmarkSuite,
    runs: Sequence[BenchmarkRun],
    basis_run: BenchmarkRun,
    strict_keys: Sequence[str],
    reference_run_id: int | tuple[int, int] | None,
) -> tuple[list[BenchmarkRun], tuple[str, ...], dict[str, Any] | None] | dict[str, Any]:
    normalized_keys = tuple(dict.fromkeys(strict_keys))
    if not normalized_keys:
        return list(runs), normalized_keys, None
    if reference_run_id is None:
        return {
            "error": "strict_requires_reference_run",
            "suite_name": suite.name,
        }
    missing_keys = [key for key in normalized_keys if key not in basis_run.configuration]
    if missing_keys:
        return {
            "error": "strict_keys_not_found",
            "suite_name": suite.name,
            "strict_keys": list(normalized_keys),
            "missing_strict_keys": missing_keys,
            "reference_run_display_id": basis_run.display_id,
        }
    strict_config = {key: basis_run.configuration[key] for key in normalized_keys}
    filtered_runs = [run for run in runs if all(run.configuration.get(key) == value for key, value in strict_config.items())]
    return filtered_runs, normalized_keys, strict_config


def _suite_comparison_payload(
    *,
    suite: BenchmarkSuite,
    runs: Sequence[BenchmarkRun],
    basis_run: BenchmarkRun,
    basis_metric_label: str,
    delta_column_label: str,
    ratio_column_label: str,
    strict_keys: Sequence[str],
    strict_config: dict[str, Any] | None,
    reference_from_pinned: bool,
    reference_run_id: int | tuple[int, int] | None,
    pinned_baseline_run: BenchmarkRun | None,
    analysis_options: AnalysisOptions | None,
) -> dict[str, Any]:
    basis_median_seconds = basis_run.median_seconds
    return {
        "suite_name": suite.name,
        "target_name": suite.target_name,
        "basis_median_seconds": basis_median_seconds,
        "basis_run": basis_run.to_suite_comparison_row(
            basis_median_seconds,
            basis_run.target_return_value,
            basis_run.samples,
            analysis_options,
        ),
        "basis_metric_label": basis_metric_label,
        "delta_column_label": delta_column_label,
        "ratio_column_label": ratio_column_label,
        "strict_keys": list(strict_keys),
        "strict_config": strict_config,
        "basis_source": "pinned" if reference_from_pinned else "reference" if reference_run_id is not None else "best",
        "pinned_baseline": None if pinned_baseline_run is None else pinned_baseline_run.to_payload(analysis_options),
        "runs": [
            run.to_suite_comparison_row(
                basis_median_seconds,
                basis_run.target_return_value,
                basis_run.samples,
                analysis_options,
            )
            for run in runs
        ],
    }


def _configuration_group_key(configuration: dict[str, Any]) -> str:
    return json.dumps(configuration, sort_keys=True, separators=(",", ":"))


def _configuration_trend_summary_payload(
    runs: Sequence[BenchmarkRun],
    analysis_options: AnalysisOptions,
    *,
    limit: int | None,
) -> dict[str, Any]:
    visible_runs = list(runs[-limit:] if limit is not None else runs)
    first_run = visible_runs[0]
    latest_run = visible_runs[-1]
    best_run = min(visible_runs, key=lambda candidate: (candidate.median_seconds, candidate.id))
    prior_runs = visible_runs[:-1]
    recent_vs_window = None
    if prior_runs:
        recent_window = prior_runs[-analysis_options.drift_window_size :]
        recent_samples = [sample for run in recent_window for sample in run.samples]
        if recent_samples:
            recent_vs_window = compare_sample_sets(recent_samples, latest_run.samples, analysis_options).to_payload()

    return {
        "configuration": dict(first_run.configuration),
        "run_count": len(visible_runs),
        "total_run_count": len(runs),
        "median_series": [run.median_seconds for run in visible_runs],
        "first_run": first_run.to_payload(analysis_options),
        "latest_run": latest_run.to_payload(analysis_options),
        "best_run": best_run.to_payload(analysis_options),
        "latest_vs_first": compare_sample_sets(first_run.samples, latest_run.samples, analysis_options).to_payload(),
        "recent_vs_window": recent_vs_window,
        "latest_vs_best": compare_sample_sets(best_run.samples, latest_run.samples, analysis_options).to_payload(),
    }
