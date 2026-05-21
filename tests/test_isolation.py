"""Tests for the benchcaddy.isolation package."""

from __future__ import annotations

import gc
import json
import operator
import pickle
import runpy
import sys
import textwrap
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import benchcaddy.cli as cli_module
import benchcaddy.isolation.environment as isolation_environment_module
import benchcaddy.isolation.process as isolation_process_module
from benchcaddy.cli import app
from benchcaddy.isolation import (
    EnvironmentState,
    IsolatedRunResult,
    NoiseAnalyzer,
    NoiseCapture,
    NoiseEstimate,
    build_reliability_report,
    collect_environment_state,
    collect_observations,
    get_affinity,
    observe,
    run_isolated,
)
from tests import subprocess_observed_targets as observed_targets

# ---------------------------------------------------------------------------
# affinity
# ---------------------------------------------------------------------------


class TestGetAffinity:
    def test_success_returns_cpu_list(self):
        mock_process = MagicMock()
        mock_process.cpu_affinity.return_value = [0, 2]
        with patch("benchcaddy.isolation.process.psutil.Process", return_value=mock_process):
            assert get_affinity() == [0, 2]

    def test_access_denied_returns_none(self):
        import psutil

        mock_process = MagicMock()
        mock_process.cpu_affinity.side_effect = psutil.AccessDenied(0)
        with patch("benchcaddy.isolation.process.psutil.Process", return_value=mock_process):
            assert get_affinity() is None

    def test_no_cpu_affinity_attribute(self):
        mock_process = MagicMock(spec=[])  # no cpu_affinity attribute
        with patch("benchcaddy.isolation.process.psutil.Process", return_value=mock_process):
            assert get_affinity() is None


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


class TestEnvironmentState:
    def test_frozen(self):
        state = EnvironmentState(cpu_load=0.1, on_battery=None, thermal_throttling=None, frequency_stable=None)
        with pytest.raises(AttributeError):
            state.cpu_load = 0.5  # type: ignore[misc]


class TestCollectEnvironmentState:
    def test_collects_signal_snapshot_from_reader_functions(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(isolation_environment_module, "_read_cpu_load", lambda: 0.12)
        monkeypatch.setattr(isolation_environment_module, "_read_on_battery", lambda: False)
        monkeypatch.setattr(isolation_environment_module, "_read_thermal_throttling", lambda: True)
        monkeypatch.setattr(isolation_environment_module, "_read_frequency_stable", lambda: False)

        assert collect_environment_state() == EnvironmentState(
            cpu_load=0.12,
            on_battery=False,
            thermal_throttling=True,
            frequency_stable=False,
        )

    def test_frequency_stability_uses_windowed_samples(self):
        samples = [
            SimpleNamespace(current=3600.0, max=4000.0),
            SimpleNamespace(current=3950.0, max=4000.0),
            SimpleNamespace(current=3980.0, max=4000.0),
            SimpleNamespace(current=3970.0, max=4000.0),
            SimpleNamespace(current=3960.0, max=4000.0),
        ]

        with (
            patch("benchcaddy.isolation.environment.psutil.cpu_freq", side_effect=samples),
            patch("benchcaddy.isolation.environment.sleep", return_value=None),
        ):
            state = collect_environment_state()

        assert state.frequency_stable is True

    def test_frequency_instability_detects_short_window_dips(self):
        samples = [
            SimpleNamespace(current=4000.0, max=4000.0),
            SimpleNamespace(current=3980.0, max=4000.0),
            SimpleNamespace(current=3200.0, max=4000.0),
            SimpleNamespace(current=3990.0, max=4000.0),
            SimpleNamespace(current=4000.0, max=4000.0),
        ]

        with (
            patch("benchcaddy.isolation.environment.psutil.cpu_freq", side_effect=samples),
            patch("benchcaddy.isolation.environment.sleep", return_value=None),
        ):
            state = collect_environment_state()

        assert state.frequency_stable is False

    def test_frequency_stability_returns_none_when_frequency_api_gives_no_samples(self):
        with (
            patch("benchcaddy.isolation.environment.psutil.cpu_freq", return_value=None),
            patch("benchcaddy.isolation.environment.sleep", return_value=None),
        ):
            assert isolation_environment_module._read_frequency_stable() is None


class TestEnvironmentPolicy:
    def test_missing_telemetry_adds_conservative_risk_and_warning(self):
        environment = EnvironmentState(
            cpu_load=None,
            on_battery=None,
            thermal_throttling=None,
            frequency_stable=None,
        )

        assert isolation_environment_module.environment_risk_score(environment) == 1
        assert isolation_environment_module.environment_warnings(environment) == ["Environment telemetry unavailable — quality estimate is conservative"]

    def test_environment_risk_score_accumulates_signal_risks(self):
        environment = EnvironmentState(
            cpu_load=0.50,
            on_battery=True,
            thermal_throttling=True,
            frequency_stable=False,
        )

        assert isolation_environment_module.environment_risk_score(environment) == 9


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------


class TestNoiseEstimate:
    def test_frozen(self):
        est = NoiseEstimate(relative_jitter=0.018, noise_level="low", relative_drift=0.004, drift_level="low")
        with pytest.raises(AttributeError):
            est.noise_level = "high"  # type: ignore[misc]


class TestEstimateNoise:
    def test_analyze_forwards_iterations_and_returns_statistics(self):
        recorded_iterations: list[int] = []

        class StubNoiseAnalyzer(NoiseAnalyzer):
            def capture(self, iterations: int = 500) -> NoiseCapture:
                recorded_iterations.append(iterations)
                return _capture(1.0, 1.0, 1.0, 1.0)

        result = StubNoiseAnalyzer().analyze(iterations=4)

        assert recorded_iterations == [4]
        assert result == NoiseEstimate(
            relative_jitter=0.0,
            noise_level="low",
            relative_drift=0.0,
            drift_level="low",
            iteration_count=4,
            median_sample_seconds=1.0,
        )

    def test_raises_on_fewer_than_two_iterations(self):
        with pytest.raises(ValueError, match="iterations must be at least 2"):
            NoiseAnalyzer().analyze(iterations=1)


def _capture(*durations: float) -> NoiseCapture:
    return NoiseCapture(
        durations=tuple(durations),
        iteration_count=len(durations),
        work_units=512,
    )


@observe("time")
def _timed_increment(value: int) -> int:
    return value + 1


@observe("return")
def _observed_identity(value: object) -> object:
    return value


@observe("time", "return")
def _observed_add(a: int, b: int) -> int:
    return a + b


@observe("time")
def _observed_failure() -> None:
    raise RuntimeError("boom")


class TestIsolationObservability:
    def test_time_mode_records_duration(self):
        with collect_observations() as collector:
            assert _timed_increment(2) == 3

        assert len(collector.records) == 1
        assert collector.records[0]["label"] == "_timed_increment"
        assert collector.records[0]["kind"] == "time"
        assert collector.records[0]["duration_seconds"] >= 0.0

    def test_return_mode_records_normalized_values_and_skips_unsupported_values(self):
        with collect_observations() as collector:
            assert _observed_identity([1, 2, 3]) == [1, 2, 3]
            unsupported = object()
            assert _observed_identity(unsupported) is unsupported

        assert collector.records == [
            {
                "label": "_observed_identity",
                "kind": "return",
                "value": [1.0, 2.0, 3.0],
            }
        ]

    def test_combined_modes_record_both_kinds(self):
        with collect_observations() as collector:
            assert _observed_add(2, 3) == 5

        assert len(collector.records) == 2
        assert collector.records[0] == {
            "label": "_observed_add",
            "kind": "return",
            "value": 5,
        }
        assert collector.records[1]["label"] == "_observed_add"
        assert collector.records[1]["kind"] == "time"
        assert collector.records[1]["duration_seconds"] >= 0.0

    def test_time_mode_records_exceptions(self):
        with pytest.raises(RuntimeError, match="boom"), collect_observations() as collector:
            _observed_failure()

        assert len(collector.records) == 1
        assert collector.records[0]["label"] == "_observed_failure"
        assert collector.records[0]["kind"] == "time"

    def test_observe_rejects_missing_or_invalid_modes(self):
        with pytest.raises(ValueError, match="at least one mode"):
            observe()

        with pytest.raises(ValueError, match="Unsupported"):
            observe("bogus")


class TestNoiseAnalyzer:
    def test_capture_returns_noise_capture(self):
        class ConstantProbeNoiseAnalyzer(NoiseAnalyzer):
            def _probe_target_seconds(self) -> float:
                return 0.0

            def _probe_once(self, work_units: int) -> float:
                return 1.0

        analyzer = ConstantProbeNoiseAnalyzer()

        capture = analyzer.capture(iterations=4)

        assert isinstance(capture, NoiseCapture)
        assert capture.iteration_count == 4
        assert len(capture.durations) == 4

    def test_capture_reuses_one_calibration_for_the_single_run(self):
        recorded_work_units: list[int] = []

        class RecordingNoiseAnalyzer(NoiseAnalyzer):
            def _probe_target_seconds(self) -> float:
                return 5e-05

            def _probe_once(self, work_units: int) -> float:
                recorded_work_units.append(work_units)
                if work_units < 512:
                    return 1e-06
                return 8e-05

        analyzer = RecordingNoiseAnalyzer()

        capture = analyzer.capture(iterations=4)

        assert capture.work_units == 512
        assert recorded_work_units[:25] == [
            32,
            32,
            32,
            32,
            32,
            64,
            64,
            64,
            64,
            64,
            128,
            128,
            128,
            128,
            128,
            256,
            256,
            256,
            256,
            256,
            512,
            512,
            512,
            512,
            512,
        ]
        assert set(recorded_work_units[25:]) == {512}

    def test_estimate_statistics_low_noise_classification(self):
        analyzer = NoiseAnalyzer()
        capture = _capture(1.0, 1.0, 1.0, 1.0)

        result = analyzer.estimate_statistics(capture)

        assert result.noise_level == "low"
        assert result.relative_jitter == pytest.approx(0.0)
        assert result.drift_level == "low"
        assert result.relative_drift == pytest.approx(0.0)

    def test_estimate_statistics_high_noise_classification(self):
        analyzer = NoiseAnalyzer()
        capture = _capture(0.70, 0.85, 0.95, 1.0, 1.10, 1.20, 1.30, 1.45)

        result = analyzer.estimate_statistics(capture)

        assert result.noise_level == "high"
        assert result.relative_jitter >= 0.20

    def test_single_isolated_outlier_does_not_dominate_noise_metric(self):
        analyzer = NoiseAnalyzer()
        capture = _capture(*([1.0] * 39 + [6.0]))

        result = analyzer.estimate_statistics(capture)

        assert result.relative_jitter == pytest.approx(0.0)
        assert result.noise_level == "low"
        assert result.relative_drift == pytest.approx(0.0)
        assert result.drift_level == "low"

    def test_drift_detects_change_between_early_and_late_samples(self):
        analyzer = NoiseAnalyzer()
        capture = _capture(*([1.0] * 10 + [1.05] * 20 + [1.25] * 10))

        result = analyzer.estimate_statistics(capture)

        assert result.relative_drift >= 0.20
        assert result.drift_level == "high"

    def test_zero_median_is_handled(self):
        analyzer = NoiseAnalyzer()
        capture = _capture(0.0, 0.0, 0.0, 0.0)

        result = analyzer.estimate_statistics(capture)

        assert result.relative_jitter == pytest.approx(0.0)
        assert result.relative_drift == pytest.approx(0.0)
        assert result.median_sample_seconds == pytest.approx(0.0)

    def test_estimate_statistics_rejects_duration_count_mismatch(self):
        analyzer = NoiseAnalyzer()
        capture = NoiseCapture(
            durations=(1.0, 1.0),
            iteration_count=3,
            work_units=512,
        )

        with pytest.raises(ValueError, match="capture iteration_count does not match recorded durations"):
            analyzer.estimate_statistics(capture)


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------


_worker_call_log: list[tuple[int, int]] = []


def _record_call(a: int, b: int) -> int:
    _worker_call_log.append((a, b))
    return len(_worker_call_log)


def _return_unpickleable_value():
    return lambda: None


def _assert_nested_observed_result(result: IsolatedRunResult) -> None:
    assert result.return_value == 6
    assert result.elapsed_seconds >= 0.0
    assert len(result.observations) == 4

    assert result.observations[0]["label"] == "nested_observed_target.<locals>.inner_time"
    assert result.observations[0]["kind"] == "time"
    assert result.observations[0]["duration_seconds"] >= 0.0

    assert result.observations[1] == {
        "label": "nested_observed_target.<locals>.inner_return",
        "kind": "return",
        "value": 6,
    }

    assert result.observations[2] == {
        "label": "nested_observed_target.<locals>.inner_both",
        "kind": "return",
        "value": 6,
    }

    assert result.observations[3]["label"] == "nested_observed_target.<locals>.inner_both"
    assert result.observations[3]["kind"] == "time"
    assert result.observations[3]["duration_seconds"] >= 0.0


def _assert_realistic_workflow_result(result: IsolatedRunResult) -> None:
    assert result.return_value == 17
    assert result.elapsed_seconds >= 0.0
    assert len(result.observations) == 5

    assert result.observations[0]["label"] == "module_time_helper"
    assert result.observations[0]["kind"] == "time"
    assert result.observations[0]["duration_seconds"] >= 0.0

    assert result.observations[1]["label"] == "ObservableService.static_time_helper"
    assert result.observations[1]["kind"] == "time"
    assert result.observations[1]["duration_seconds"] >= 0.0

    assert result.observations[2] == {
        "label": "ObservableService.class_return_helper",
        "kind": "return",
        "value": 10,
    }

    assert result.observations[3] == {
        "label": "ObservableService.instance_both_helper",
        "kind": "return",
        "value": 17,
    }

    assert result.observations[4]["label"] == "ObservableService.instance_both_helper"
    assert result.observations[4]["kind"] == "time"
    assert result.observations[4]["duration_seconds"] >= 0.0


class TestRunIsolated:
    def test_direct_call(self):
        result = run_isolated(operator.add, args=(2, 3), fresh_process=False)
        assert result.return_value == 5
        assert result.elapsed_seconds >= 0.0
        assert result.observations == []

    def test_direct_call_collects_nested_observations(self):
        result = run_isolated(observed_targets.nested_observed_target, args=(2,), fresh_process=False)
        _assert_nested_observed_result(result)

    def test_fresh_process(self):
        result = run_isolated(operator.add, args=(2, 3), fresh_process=True, timeout=30)
        assert result.return_value == 5
        assert result.elapsed_seconds >= 0.0
        assert result.observations == []

    def test_fresh_process_collects_nested_observations(self):
        result = run_isolated(observed_targets.nested_observed_target, args=(2,), fresh_process=True, timeout=30)
        _assert_nested_observed_result(result)

    def test_fresh_process_collects_realistic_helper_and_submethod_observations(self):
        result = run_isolated(observed_targets.realistic_observed_workflow, args=(1,), fresh_process=True, timeout=30)
        _assert_realistic_workflow_result(result)

    def test_fresh_process_supports_top_level_module_function_target(self):
        result = run_isolated(observed_targets.top_level_module_target, args=(2,), fresh_process=True, timeout=30)

        assert result.return_value == 6
        assert result.elapsed_seconds >= 0.0
        assert result.observations == [
            {
                "label": "module_time_helper",
                "kind": "time",
                "duration_seconds": pytest.approx(result.observations[0]["duration_seconds"], abs=0.0),
            },
            {
                "label": "module_return_helper",
                "kind": "return",
                "value": 6,
            },
        ]

    def test_fresh_process_supports_top_level_script_target_loaded_as_main(self, tmp_path, monkeypatch):
        script_path = tmp_path / "script_target_example.py"
        script_path.write_text(
            textwrap.dedent(
                """
                from benchcaddy import observe

                @observe("time")
                def script_main_target(value: int) -> int:
                    return value + 4
                """
            ),
            encoding="utf-8",
        )

        monkeypatch.syspath_prepend(str(tmp_path))

        namespace = runpy.run_path(str(script_path), run_name="__main__")
        script_main_target = namespace["script_main_target"]

        assert script_main_target.__module__ == "__main__"

        result = run_isolated(script_main_target, args=(2,), fresh_process=True, timeout=30)

        assert result.return_value == 6
        assert result.elapsed_seconds >= 0.0
        assert len(result.observations) == 1
        assert result.observations[0]["label"] == "script_main_target"
        assert result.observations[0]["kind"] == "time"
        assert result.observations[0]["duration_seconds"] >= 0.0

    def test_fresh_process_supports_importable_static_method_target(self):
        result = run_isolated(observed_targets.ObservableService.static_time_helper, args=(2,), fresh_process=True, timeout=30)

        assert result.return_value == 5
        assert result.elapsed_seconds >= 0.0
        assert len(result.observations) == 1
        assert result.observations[0]["label"] == "ObservableService.static_time_helper"
        assert result.observations[0]["kind"] == "time"
        assert result.observations[0]["duration_seconds"] >= 0.0

    def test_fresh_process_supports_importable_class_method_target(self):
        result = run_isolated(observed_targets.ObservableService.class_return_helper, args=(2,), fresh_process=True, timeout=30)

        assert result.return_value == 7
        assert result.elapsed_seconds >= 0.0
        assert result.observations == [
            {
                "label": "ObservableService.class_return_helper",
                "kind": "return",
                "value": 7,
            }
        ]

    def test_fresh_process_warmups_do_not_leak_nested_observations(self):
        observed_targets.reset_warmup_sensitive_state()

        result = run_isolated(
            observed_targets.warmup_sensitive_observed_target,
            fresh_process=True,
            warmup_runs=2,
            timeout=30,
        )

        assert result.return_value == 3
        assert result.elapsed_seconds >= 0.0
        assert result.observations == [
            {
                "label": "warmup_sensitive_observed_target.<locals>.inner_return",
                "kind": "return",
                "value": 3,
            }
        ]

    def test_fresh_process_skips_unsupported_nested_return_observations(self):
        result = run_isolated(
            observed_targets.unsupported_nested_return_target,
            fresh_process=True,
            timeout=30,
        )

        assert result.return_value == "done"
        assert result.elapsed_seconds >= 0.0
        assert len(result.observations) == 1
        assert result.observations[0]["label"] == "unsupported_nested_return_target.<locals>.inner_both"
        assert result.observations[0]["kind"] == "time"
        assert result.observations[0]["duration_seconds"] >= 0.0

    def test_kwargs_forwarded(self):
        result = run_isolated(
            json.dumps,
            kwargs={"obj": {"b": 1, "a": 2}, "sort_keys": True},
            fresh_process=True,
            timeout=30,
        )
        assert result.return_value == '{"a": 2, "b": 1}'
        assert result.elapsed_seconds >= 0.0
        assert result.observations == []

    def test_timeout_raises(self):
        import time

        with pytest.raises(TimeoutError):
            run_isolated(time.sleep, args=(10,), fresh_process=True, timeout=0.1)

    def test_exception_in_child_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="JSONDecodeError"):
            run_isolated(json.loads, args=("{",), fresh_process=True, timeout=30)

    def test_disable_gc(self):
        result = run_isolated(gc.isenabled, fresh_process=True, disable_gc=True, timeout=30)
        assert result.return_value is False
        assert result.elapsed_seconds >= 0.0
        assert result.observations == []

    def test_rejects_non_importable_callables_for_fresh_process(self):
        with pytest.raises(
            TypeError,
            match="lambdas are unsupported because they do not provide a stable import path",
        ):
            run_isolated(lambda: None, fresh_process=True)

    def test_rejects_nested_local_functions_for_fresh_process(self):
        def local_target() -> None:
            return None

        with pytest.raises(
            TypeError,
            match="nested or local functions are unsupported because they are scoped to a parent frame",
        ):
            run_isolated(local_target, fresh_process=True)

    def test_rejects_bound_instance_methods_for_fresh_process(self):
        service = observed_targets.ObservableService()

        with pytest.raises(
            TypeError,
            match="bound instance methods are unsupported because the worker reconstructs call targets from module and qualname, not from a live instance",
        ):
            run_isolated(service.instance_both_helper, args=(1,), fresh_process=True)

    def test_rejects_arbitrary_callable_instances_for_fresh_process(self):
        with pytest.raises(
            TypeError,
            match="arbitrary callable instances are unsupported because the worker cannot reconstruct a live __call__ object",
        ):
            run_isolated(observed_targets.CallableTarget(), fresh_process=True)

    def test_rejects_unresolvable_module_symbol_for_fresh_process(self):
        isolation_process_module._validated_target_reference.cache_clear()
        original_qualname = observed_targets.top_level_module_target.__qualname__
        observed_targets.top_level_module_target.__qualname__ = "missing_symbol"

        try:
            with pytest.raises(
                TypeError,
                match=(
                    r"could not resolve tests\.subprocess_observed_targets\.missing_symbol\. "
                    r"Ensure the symbol is importable in the child process and exposed at that module path"
                ),
            ):
                run_isolated(observed_targets.top_level_module_target, args=(1,), fresh_process=True)
        finally:
            observed_targets.top_level_module_target.__qualname__ = original_qualname
            isolation_process_module._validated_target_reference.cache_clear()

    def test_execute_worker_request_applies_warmup_and_preparation(self, monkeypatch: pytest.MonkeyPatch):
        prepare_calls: list[bool] = []
        gc_disable_calls: list[str] = []
        _worker_call_log.clear()

        monkeypatch.setattr(
            isolation_process_module,
            "prepare_system",
            lambda lock_cpu_affinity=True: prepare_calls.append(lock_cpu_affinity),
        )
        monkeypatch.setattr(isolation_process_module.gc, "disable", lambda: gc_disable_calls.append("disabled"))

        response = isolation_process_module._execute_worker_request(
            {
                "module_name": __name__,
                "qualname": "_record_call",
                "args": (2, 3),
                "kwargs": {},
                "disable_gc": True,
                "warmup_runs": 2,
                "lock_cpu_affinity": False,
            }
        )

        assert response["ok"] is True
        assert isinstance(response["payload"], IsolatedRunResult)
        assert response["payload"].return_value == 3
        assert response["payload"].elapsed_seconds >= 0.0
        assert response["payload"].observations == []
        assert _worker_call_log == [(2, 3), (2, 3), (2, 3)]
        assert prepare_calls == [False]
        assert gc_disable_calls == ["disabled"]

    def test_execute_worker_request_collects_observations_only_for_measured_call(self):
        response = isolation_process_module._execute_worker_request(
            {
                "module_name": __name__,
                "qualname": "_observed_add",
                "args": (2, 3),
                "kwargs": {},
                "disable_gc": False,
                "warmup_runs": 2,
                "lock_cpu_affinity": True,
            }
        )

        assert response["ok"] is True
        payload = response["payload"]
        assert isinstance(payload, IsolatedRunResult)
        assert payload.return_value == 5
        assert payload.elapsed_seconds >= 0.0
        assert payload.observations[0] == {
            "label": "_observed_add",
            "kind": "return",
            "value": 5,
        }
        assert payload.observations[1]["label"] == "_observed_add"
        assert payload.observations[1]["kind"] == "time"
        assert len(payload.observations) == 2

    def test_run_isolated_reuses_cached_target_validation(self, monkeypatch: pytest.MonkeyPatch):
        isolation_process_module._validated_target_reference.cache_clear()
        validation_calls: list[Callable[..., object]] = []
        original_reference_lookup = isolation_process_module._importable_target_reference

        monkeypatch.setattr(
            isolation_process_module,
            "_importable_target_reference",
            lambda fn: validation_calls.append(fn) or original_reference_lookup(fn),
        )
        monkeypatch.setattr(
            isolation_process_module,
            "_run_subprocess_worker",
            lambda request, timeout: {
                "ok": True,
                "payload": IsolatedRunResult(
                    elapsed_seconds=0.1,
                    return_value=5,
                    observations=[],
                ),
            },
        )

        isolation_process_module.validate_isolated_target(operator.add)
        result = run_isolated(operator.add, args=(2, 3), fresh_process=True, timeout=1.0)

        assert result == IsolatedRunResult(elapsed_seconds=0.1, return_value=5, observations=[])
        assert validation_calls == [operator.add]

    def test_execute_worker_request_collects_gc_before_warmups_when_gc_stays_enabled(self, monkeypatch: pytest.MonkeyPatch):
        events: list[str] = []

        monkeypatch.setattr(isolation_process_module, "prepare_system", lambda lock_cpu_affinity=True: events.append("prepare"))
        monkeypatch.setattr(isolation_process_module.gc, "collect", lambda: events.append("gc"))

        response = isolation_process_module._execute_worker_request(
            {
                "module_name": __name__,
                "qualname": "_record_call",
                "args": (2, 3),
                "kwargs": {},
                "disable_gc": False,
                "warmup_runs": 2,
                "lock_cpu_affinity": True,
            }
        )

        assert response["ok"] is True
        assert events == ["prepare", "gc"]

    def test_fresh_process_uses_package_worker_entrypoint(self, monkeypatch: pytest.MonkeyPatch):
        commands: list[list[str]] = []

        def fake_popen(cmd, **kwargs):
            commands.append(cmd)

            class _FakeProc:
                def __init__(self, cmd):
                    self.returncode = 0

                def communicate(self, timeout=None):
                    with isolation_process_module.Path(cmd[4]).open("rb") as handle:
                        request = pickle.load(handle)
                    assert request["qualname"] == "add"
                    assert isolation_process_module._resolve_callable(request["module_name"], request["qualname"])(2, 3) == 5
                    with isolation_process_module.Path(cmd[5]).open("wb") as handle:
                        pickle.dump(
                            {
                                "ok": True,
                                "payload": IsolatedRunResult(
                                    elapsed_seconds=0.125,
                                    return_value=5,
                                    observations=[],
                                ),
                            },
                            handle,
                        )
                    return ("", "")

            return _FakeProc(cmd)

        monkeypatch.setattr(isolation_process_module.subprocess, "Popen", fake_popen)

        result = run_isolated(operator.add, args=(2, 3), fresh_process=True, timeout=1.0)

        assert result == IsolatedRunResult(elapsed_seconds=0.125, return_value=5, observations=[])
        assert commands == [
            [
                sys.executable,
                "-m",
                "benchcaddy.isolation.process",
                isolation_process_module._WORKER_FLAG,
                commands[0][4],
                commands[0][5],
            ]
        ]

    def test_child_exit_before_result_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch):
        # Simulate a child process that exits with a non-zero return code and
        # does not write a response file. Patch Popen (used by the product code)
        # to return a proc-like object with the expected attributes.
        monkeypatch.setattr(
            isolation_process_module.subprocess,
            "Popen",
            lambda command, **kwargs: SimpleNamespace(returncode=3, communicate=lambda timeout=None: ("", "")),
        )

        with pytest.raises(RuntimeError, match="failed before sending a result"):
            run_isolated(operator.add, args=(2, 3), fresh_process=True, timeout=1.0)

    def test_unpickleable_worker_result_raises_structured_runtime_error(self):
        with pytest.raises(RuntimeError, match="could not serialize the isolated result payload"):
            run_isolated(_return_unpickleable_value, fresh_process=True, timeout=30)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _make_env(**kwargs) -> EnvironmentState:
    defaults = {
        "cpu_load": 0.05,
        "on_battery": False,
        "thermal_throttling": False,
        "frequency_stable": True,
    }
    defaults.update(kwargs)
    return EnvironmentState(**defaults)


def _make_noise(
    noise_level="low",
    jitter=0.01,
    *,
    drift_level: str = "low",
    drift: float = 0.01,
    iteration_count: int = 500,
) -> NoiseEstimate:
    return NoiseEstimate(
        relative_jitter=jitter,
        noise_level=noise_level,
        relative_drift=drift,
        drift_level=drift_level,
        iteration_count=iteration_count,
    )


class TestReliabilityReport:
    def test_clean_environment_is_high(self):
        report = build_reliability_report(
            environment=_make_env(),
            noise=_make_noise(jitter=0.01),
        )
        assert report.timing_stability == "HIGH"
        assert report.environmental_quality == "HIGH"
        assert report.warnings == ()

    def test_high_jitter_degrades_timing_stability(self):
        report = build_reliability_report(
            environment=_make_env(),
            noise=_make_noise(noise_level="high", jitter=0.18),
        )
        assert report.timing_stability == "LOW"
        assert report.environmental_quality == "LOW"

    def test_battery_warning(self):
        report = build_reliability_report(environment=_make_env(on_battery=True), noise=_make_noise())
        assert any("battery" in w.lower() for w in report.warnings)
        assert report.environmental_quality == "FAIR"

    def test_thermal_throttling_warning(self):
        report = build_reliability_report(environment=_make_env(thermal_throttling=True), noise=_make_noise())
        assert any("thermal" in w.lower() for w in report.warnings)
        assert report.environmental_quality == "LOW"

    def test_frequency_scaling_warning(self):
        report = build_reliability_report(environment=_make_env(frequency_stable=False), noise=_make_noise())
        assert any("frequency" in w.lower() or "scaling" in w.lower() for w in report.warnings)

    def test_high_cpu_load_warning(self):
        report = build_reliability_report(environment=_make_env(cpu_load=0.50), noise=_make_noise())
        assert any("background load" in w.lower() for w in report.warnings)

    def test_moderate_cpu_load_warning(self):
        report = build_reliability_report(environment=_make_env(cpu_load=0.20), noise=_make_noise())
        assert any("background activity" in w.lower() for w in report.warnings)

    def test_high_noise_warning(self):
        report = build_reliability_report(environment=_make_env(), noise=_make_noise(noise_level="high", jitter=0.15))
        assert report.timing_stability == "LOW"
        assert report.environmental_quality == "LOW"
        assert any("jitter" in w.lower() for w in report.warnings)

    def test_moderate_noise_warning(self):
        report = build_reliability_report(environment=_make_env(), noise=_make_noise(noise_level="moderate", jitter=0.05))
        assert report.timing_stability == "FAIR"
        assert report.environmental_quality == "FAIR"
        assert any("jitter" in w.lower() for w in report.warnings)

    def test_high_drift_warning(self):
        report = build_reliability_report(environment=_make_env(), noise=_make_noise(drift_level="high", drift=0.22))
        assert report.timing_stability == "LOW"
        assert report.environmental_quality == "LOW"
        assert any("drift" in w.lower() for w in report.warnings)

    def test_combined_moderate_risks_degrade_environment_to_low(self):
        report = build_reliability_report(
            environment=_make_env(on_battery=True, cpu_load=0.20),
            noise=_make_noise(noise_level="moderate", jitter=0.05),
        )
        assert report.environmental_quality == "LOW"

    def test_warnings_is_tuple(self):
        report = build_reliability_report(environment=_make_env(), noise=_make_noise())
        assert isinstance(report.warnings, tuple)

    def test_format_no_warnings(self):
        report = build_reliability_report(environment=_make_env(), noise=_make_noise())
        text = report.format()
        assert "Timing stability:      HIGH" in text
        assert "Environmental quality:  HIGH" in text
        assert "Warnings:" not in text

    def test_format_with_warnings(self):
        report = build_reliability_report(environment=_make_env(on_battery=True), noise=_make_noise())
        text = report.format()
        assert "Warnings:" in text


# ---------------------------------------------------------------------------
# CLI env command
# ---------------------------------------------------------------------------


runner = CliRunner()


@pytest.fixture
def stubbed_check_dependencies(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    analyze_calls: list[int] = []

    monkeypatch.setattr(
        cli_module,
        "collect_environment_state",
        lambda: _make_env(cpu_load=0.12, on_battery=False, thermal_throttling=False, frequency_stable=True),
    )
    monkeypatch.setattr(cli_module, "get_affinity", lambda: [0, 1])

    def fake_analyze(self, iterations: int = 500) -> NoiseEstimate:
        analyze_calls.append(iterations)
        return _make_noise(
            noise_level="moderate",
            jitter=0.05,
            drift_level="low",
            drift=0.01,
            iteration_count=iterations,
        )

    monkeypatch.setattr(cli_module.NoiseAnalyzer, "analyze", fake_analyze)
    return analyze_calls


class TestEnvCommand:
    def test_env_renders_summary_and_warnings(self, stubbed_check_dependencies: list[int]):
        result = runner.invoke(app, ["env"])
        assert result.exit_code == 0
        assert "Benchmark Reliability" in result.output
        assert "Warnings" in result.output
        assert stubbed_check_dependencies == [200]

    def test_env_json_output(self, stubbed_check_dependencies: list[int]):
        result = runner.invoke(app, ["env", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["command"] == "env"
        assert payload["schema_version"] == "1.0"
        assert payload["status"] == "inconclusive"
        assert payload["reason"] == "environment_warnings_detected"

        result_payload = payload["result"]
        assert "timing_stability" in result_payload
        assert "environmental_quality" in result_payload
        assert "warnings" in result_payload
        assert "noise_level" in result_payload["noise"]
        assert "drift_level" in result_payload["noise"]
        assert "environment" in result_payload
        assert "noise" in result_payload
        assert "affinity" in result_payload
        assert result_payload["timing_stability"] in {"HIGH", "FAIR", "LOW"}
        assert result_payload["environmental_quality"] in {"HIGH", "FAIR", "LOW"}
        assert result_payload["noise"]["noise_level"] in {"low", "moderate", "high"}
        assert result_payload["noise"]["drift_level"] in {"low", "moderate", "high"}
        assert result_payload["environment"]["cpu_load"] == pytest.approx(0.12)
        assert result_payload["environment"]["on_battery"] is False
        assert result_payload["environment"]["thermal_throttling"] is False
        assert result_payload["environment"]["frequency_stable"] is True
        assert result_payload["noise"]["relative_jitter"] == pytest.approx(0.05)
        assert result_payload["noise"]["noise_level"] == "moderate"
        assert result_payload["noise"]["relative_drift"] == pytest.approx(0.01)
        assert result_payload["noise"]["drift_level"] == "low"
        assert result_payload["noise"]["iteration_count"] == 200
        assert result_payload["affinity"] == [0, 1]
        assert stubbed_check_dependencies == [200]

    def test_env_with_noise_iterations(self, stubbed_check_dependencies: list[int]):
        result = runner.invoke(app, ["env", "--noise-iterations", "10"])
        assert result.exit_code == 0
        assert stubbed_check_dependencies == [10]
