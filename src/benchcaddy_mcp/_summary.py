from __future__ import annotations

from typing import Any


class ResponseSummaryBuilder:
    def build(self, tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "compare_runs" and "baseline" in result and "candidate" in result:
            return self._direct_comparison_summary(result)
        if tool_name == "compare_suite" and "basis_run" in result and "runs" in result:
            return self._suite_comparison_summary(result)
        if tool_name == "trend_suite" and result.get("mode") in {"summary", "timeline"}:
            return self._trend_summary(result)
        if tool_name == "get_suite" and result.get("mode") == "suite":
            return self._suite_details_summary(result)
        if tool_name == "get_run" and result.get("mode") == "run":
            return self._run_details_summary(result)
        if tool_name == "list_suites":
            return self._suite_inventory_summary(result)
        if tool_name == "get_baseline_history" and "history" in result:
            return self._baseline_history_summary(result)
        if tool_name == "pin_baseline" and "pin_update" in result:
            return self._pin_update_summary(result)
        return result

    def _configuration_differences(
        self,
        left: dict[str, Any] | None,
        right: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        left = left or {}
        right = right or {}
        differences: dict[str, dict[str, Any]] = {}
        for key in sorted(set(left) | set(right)):
            if left.get(key) != right.get(key):
                differences[key] = {
                    "left": left.get(key),
                    "right": right.get(key),
                }
        return differences

    def _analysis_summary(self, analysis: dict[str, Any] | None) -> dict[str, Any] | None:
        if analysis is None:
            return None
        return {
            "classification": analysis.get("classification"),
            "regression_detected": analysis.get("regression_detected"),
            "statistically_significant": analysis.get("statistically_significant"),
            "exceeds_practical_threshold": analysis.get("exceeds_practical_threshold"),
            "regression_probability": analysis.get("regression_probability"),
            "improvement_probability": analysis.get("improvement_probability"),
            "percent_change": analysis.get("percent_change"),
            "delta_seconds": analysis.get("delta_seconds"),
            "warnings": analysis.get("warnings") or [],
        }

    def _compact_analysis_summary(self, analysis: dict[str, Any] | None) -> dict[str, Any] | None:
        if analysis is None:
            return None
        return {
            "classification": analysis.get("classification"),
            "regression_detected": analysis.get("regression_detected"),
            "percent_change": analysis.get("percent_change"),
            "delta_seconds": analysis.get("delta_seconds"),
            "warnings": analysis.get("warnings") or [],
        }

    def _environment_summary(self, environment: dict[str, Any] | None) -> dict[str, Any] | None:
        if environment is None:
            return None
        return {
            "python_version": environment.get("python_version"),
            "operating_system": environment.get("operating_system"),
            "cpu_model": environment.get("cpu_model"),
            "gpu_model": environment.get("gpu_model"),
            "git": environment.get("git"),
        }

    def _observation_labels(self, observations: list[dict[str, Any]] | None) -> list[str]:
        if not observations:
            return []
        labels: set[str] = set()
        for observation in observations:
            for record in observation.get("records", []):
                label = record.get("label")
                if isinstance(label, str):
                    labels.add(label)
        return sorted(labels)

    def _run_summary(self, run: dict[str, Any] | None) -> dict[str, Any] | None:
        if run is None:
            return None
        analysis = run.get("analysis") or {}
        return {
            "display_id": run.get("display_id"),
            "suite_name": run.get("suite_name"),
            "target_name": run.get("target_name"),
            "configuration": run.get("configuration"),
            "created_at": run.get("created_at"),
            "median_seconds": run.get("median_seconds"),
            "mean_seconds": run.get("mean_seconds"),
            "sample_count": analysis.get("sample_count") or len(run.get("samples") or []),
            "target_return_value": run.get("target_return_value"),
            "coefficient_of_variation": run.get("coefficient_of_variation"),
            "is_noisy": run.get("is_noisy"),
        }

    def _compact_run_summary(
        self,
        run: dict[str, Any] | None,
        *,
        include_configuration: bool = True,
    ) -> dict[str, Any] | None:
        if run is None:
            return None
        analysis = run.get("analysis") or {}
        payload = {
            "display_id": run.get("display_id"),
            "created_at": run.get("created_at"),
            "median_seconds": run.get("median_seconds"),
            "sample_count": analysis.get("sample_count") or run.get("sample_count") or len(run.get("samples") or []),
            "is_noisy": run.get("is_noisy"),
        }
        if include_configuration:
            payload["configuration"] = run.get("configuration")
        return payload

    def _suite_comparison_row_summary(self, run: dict[str, Any]) -> dict[str, Any]:
        comparison_analysis = run.get("comparison_analysis") or {}
        return {
            **(self._compact_run_summary(run) or {}),
            "delta_seconds": run.get("delta_seconds"),
            "status": run.get("status"),
            "comparison": self._compact_analysis_summary(comparison_analysis),
        }

    def _trend_run_summary(self, run: dict[str, Any]) -> dict[str, Any]:
        return {
            **(self._compact_run_summary(run) or {}),
            "is_basis": run.get("is_basis"),
            "drift_status": run.get("drift_status"),
            "vs_baseline": self._compact_analysis_summary(run.get("vs_baseline")),
            "drift_analysis": self._compact_analysis_summary(run.get("drift_analysis")),
        }

    def _trend_verdict_from_analysis(self, analysis: dict[str, Any] | None) -> str:
        analysis = analysis or {}
        if analysis.get("regression_detected"):
            return "regressing"
        if analysis.get("classification") == "noisy" or analysis.get("warnings"):
            return "inconclusive"
        if analysis.get("classification") == "improving":
            return "improving"
        return "stable"

    def _configuration_trend_verdict(self, summary: dict[str, Any]) -> str:
        verdicts = [
            self._trend_verdict_from_analysis(summary.get("recent_vs_window")),
            self._trend_verdict_from_analysis(summary.get("latest_vs_best")),
            self._trend_verdict_from_analysis(summary.get("latest_vs_first")),
        ]
        if "regressing" in verdicts:
            return "regressing"
        if "inconclusive" in verdicts:
            return "inconclusive"
        if verdicts and all(verdict == "improving" for verdict in verdicts):
            return "improving"
        return "stable"

    def _timeline_trend_verdict(self, runs: list[dict[str, Any]]) -> str:
        if any((run.get("vs_baseline") or {}).get("regression_detected") for run in runs):
            return "regressing"
        if any((run.get("vs_baseline") or {}).get("warnings") or run.get("drift_status") == "noisy" for run in runs):
            return "inconclusive"
        latest_run = runs[-1] if runs else None
        latest_analysis = None if latest_run is None else (latest_run.get("drift_analysis") or latest_run.get("vs_baseline"))
        return self._trend_verdict_from_analysis(latest_analysis)

    def _direct_comparison_summary(self, comparison: dict[str, Any]) -> dict[str, Any]:
        baseline = comparison.get("baseline") or {}
        candidate = comparison.get("candidate") or {}
        comparison_analysis = comparison.get("comparison_analysis") or {}
        configuration_differences = self._configuration_differences(
            baseline.get("configuration"),
            candidate.get("configuration"),
        )
        return {
            "comparison_mode": comparison.get("comparison_mode"),
            "baseline": self._run_summary(baseline),
            "candidate": self._run_summary(candidate),
            "same_configuration": not configuration_differences,
            "configuration_differences": configuration_differences,
            "delta_seconds": comparison.get("delta_seconds"),
            "percent_change": comparison.get("percent_change"),
            "target_return_relative_error": comparison.get("target_return_relative_error"),
            **(self._analysis_summary(comparison_analysis) or {}),
        }

    def _suite_details_summary(self, result: dict[str, Any]) -> dict[str, Any]:
        runs = result.get("runs") or []
        return {
            "mode": result.get("mode"),
            "database_path": result.get("database_path"),
            "suite_name": result.get("suite_name"),
            "target_name": result.get("target_name"),
            "config_filter": result.get("config_filter"),
            "total_run_count": result.get("total_run_count", len(runs)),
            "truncated": result.get("truncated", False),
            "limit": result.get("limit"),
            "run_count": len(runs),
            "configuration_count": result.get("configuration_count"),
            "available_configurations": result.get("available_configurations") or [],
            "latest_runs": [self._compact_run_summary(run) for run in runs],
            "baseline_run": self._compact_run_summary(result.get("baseline_run")),
            "environment": self._environment_summary(result.get("environment")),
        }

    def _run_details_summary(self, result: dict[str, Any]) -> dict[str, Any]:
        run = result.get("run") or {}
        return {
            "mode": result.get("mode"),
            "database_path": result.get("database_path"),
            "run": self._run_summary(run),
            "environment": self._environment_summary(run.get("environment")),
            "observation_labels": run.get("observation_labels") or self._observation_labels(run.get("observations")),
        }

    def _suite_inventory_summary(self, result: dict[str, Any]) -> dict[str, Any]:
        suites = result.get("suites") or []
        return {
            "database_path": result.get("database_path"),
            "suite_count": result.get("suite_count", len(suites)),
            "suites": suites,
        }

    def _suite_comparison_summary(self, comparison: dict[str, Any]) -> dict[str, Any]:
        runs = comparison.get("runs") or []
        analyses = [run.get("comparison_analysis") or {} for run in runs]
        comparison_verdict = "stable"
        if any(analysis.get("regression_detected") for analysis in analyses):
            comparison_verdict = "regressing"
        elif any(analysis.get("classification") == "noisy" or analysis.get("warnings") for analysis in analyses):
            comparison_verdict = "inconclusive"
        return {
            "comparison_mode": comparison.get("comparison_mode"),
            "suite_name": comparison.get("suite_name"),
            "target_name": comparison.get("target_name"),
            "basis_source": comparison.get("basis_source"),
            "basis_run": self._suite_comparison_row_summary(comparison.get("basis_run") or {}) if comparison.get("basis_run") else None,
            "pinned_baseline": self._compact_run_summary(comparison.get("pinned_baseline")),
            "config_filter": comparison.get("config_filter"),
            "strict_keys": comparison.get("strict_keys") or [],
            "strict_config": comparison.get("strict_config"),
            "config_filter_warning": comparison.get("config_filter_warning"),
            "total_run_count": comparison.get("total_run_count", len(runs)),
            "truncated": comparison.get("truncated", False),
            "limit": comparison.get("limit"),
            "run_count": len(runs),
            "regression_count": sum(1 for analysis in analyses if analysis.get("regression_detected")),
            "noisy_count": sum(1 for analysis in analyses if analysis.get("classification") == "noisy" or analysis.get("warnings")),
            "comparison_verdict": comparison_verdict,
            "comparison_runs": [self._suite_comparison_row_summary(run) for run in runs],
        }

    def _trend_summary(self, trend: dict[str, Any]) -> dict[str, Any]:
        if trend.get("mode") == "summary":
            config_summaries = trend.get("config_summaries") or []
            summarized_configurations = [summary.get("configuration") for summary in config_summaries if summary.get("configuration") is not None]
            summarized_rows = [
                {
                    "configuration": summary.get("configuration"),
                    "trend_verdict": self._configuration_trend_verdict(summary),
                    "run_count": summary.get("run_count"),
                    "total_run_count": summary.get("total_run_count"),
                    "first_run": self._compact_run_summary(summary.get("first_run"), include_configuration=False),
                    "latest_run": self._compact_run_summary(summary.get("latest_run"), include_configuration=False),
                    "best_run": self._compact_run_summary(summary.get("best_run"), include_configuration=False),
                    "latest_vs_first": self._compact_analysis_summary(summary.get("latest_vs_first")),
                    "recent_vs_window": self._compact_analysis_summary(summary.get("recent_vs_window")),
                    "latest_vs_best": self._compact_analysis_summary(summary.get("latest_vs_best")),
                }
                for summary in config_summaries
            ]
            regression_count = sum(1 for summary in summarized_rows if summary["trend_verdict"] == "regressing")
            noisy_count = sum(1 for summary in summarized_rows if summary["trend_verdict"] == "inconclusive")
            trend_verdict = "stable"
            if regression_count:
                trend_verdict = "regressing"
            elif noisy_count:
                trend_verdict = "inconclusive"
            elif summarized_rows and all(summary["trend_verdict"] == "improving" for summary in summarized_rows):
                trend_verdict = "improving"
            return {
                "mode": "summary",
                "suite_name": trend.get("suite_name"),
                "target_name": trend.get("target_name"),
                "configuration_count": trend.get("configuration_count", len(config_summaries)),
                "available_configurations": summarized_configurations,
                "trend_verdict": trend_verdict,
                "regression_count": regression_count,
                "noisy_count": noisy_count,
                "limit": trend.get("limit"),
                "configuration_summaries": summarized_rows,
            }

        runs = trend.get("runs") or []
        available_configurations = trend.get("available_suite_configurations") or []
        timeline_rows = [self._trend_run_summary(run) for run in runs]
        return {
            "mode": trend.get("mode"),
            "suite_name": trend.get("suite_name"),
            "target_name": trend.get("target_name"),
            "basis_source": trend.get("basis_source"),
            "basis_run": self._compact_run_summary(trend.get("basis_run")),
            "config_filter": trend.get("config_filter"),
            "available_configurations": available_configurations,
            "configuration_count": len(available_configurations),
            "total_run_count": trend.get("total_run_count", len(runs)),
            "truncated": trend.get("truncated", False),
            "limit": trend.get("limit"),
            "run_count": len(runs),
            "regression_count": sum(1 for run in runs if (run.get("vs_baseline") or {}).get("regression_detected")),
            "noisy_count": sum(1 for run in runs if (run.get("vs_baseline") or {}).get("warnings") or run.get("drift_status") == "noisy"),
            "trend_verdict": self._timeline_trend_verdict(runs),
            "latest_run": None if not timeline_rows else timeline_rows[-1],
            "timeline_runs": timeline_rows,
        }

    def _baseline_history_summary(self, history: dict[str, Any]) -> dict[str, Any]:
        entries = history.get("history") or []
        return {
            "suite_name": history.get("suite_name"),
            "target_name": history.get("target_name"),
            "total_run_count": history.get("total_run_count", len(entries)),
            "truncated": history.get("truncated", False),
            "limit": history.get("limit"),
            "history_count": len(entries),
            "current_baseline": self._baseline_event_summary(history.get("current_baseline")),
            "baseline_history": [self._baseline_event_summary(entry) for entry in entries],
        }

    def _baseline_event_summary(self, event: dict[str, Any] | None) -> dict[str, Any] | None:
        if event is None:
            return None
        return {
            "event_id": event.get("event_id"),
            "created_at": event.get("created_at"),
            "note": event.get("note"),
            "is_current": event.get("is_current"),
            "run": self._compact_run_summary(event.get("run")),
        }

    def _pin_update_summary(self, result: dict[str, Any]) -> dict[str, Any]:
        pin_update = result.get("pin_update") or {}
        return {
            "pin_update": self._run_summary(pin_update),
        }
