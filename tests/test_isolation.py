"""Tests for the benchcaddy.isolation package."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import benchcaddy.cli as cli_module
from typer.testing import CliRunner

from benchcaddy.cli import app
from benchcaddy.isolation import (
    EnvironmentState,
    NoiseAnalyzer,
    NoiseCapture,
    NoiseEstimate,
    ReliabilityReport,
    build_reliability_report,
    collect_environment_state,
    get_affinity,
    run_isolated,
    set_affinity,
)

# ---------------------------------------------------------------------------
# affinity
# ---------------------------------------------------------------------------


class TestGetAffinity:
    def test_returns_list_or_none(self):
        result = get_affinity()
        assert result is None or isinstance(result, list)

    def test_returns_ints(self):
        result = get_affinity()
        if result is not None:
            assert all(isinstance(cpu, int) for cpu in result)

    def test_no_cpu_affinity_attribute(self):
        mock_process = MagicMock(spec=[])  # no cpu_affinity attribute
        with patch("benchcaddy.isolation.process.psutil.Process", return_value=mock_process):
            assert get_affinity() is None


class TestSetAffinity:
    def test_empty_list_returns_false(self):
        assert set_affinity([]) is False

    def test_no_cpu_affinity_attribute(self):
        mock_process = MagicMock(spec=[])  # no cpu_affinity attribute
        with patch("benchcaddy.isolation.process.psutil.Process", return_value=mock_process):
            assert set_affinity([0]) is False

    def test_access_denied_returns_false(self):
        import psutil

        mock_process = MagicMock()
        mock_process.cpu_affinity.side_effect = psutil.AccessDenied(0)
        with patch("benchcaddy.isolation.process.psutil.Process", return_value=mock_process):
            assert set_affinity([0]) is False

    def test_success_returns_true(self):
        mock_process = MagicMock()
        mock_process.cpu_affinity.return_value = None
        with patch("benchcaddy.isolation.process.psutil.Process", return_value=mock_process):
            assert set_affinity([0]) is True
            mock_process.cpu_affinity.assert_called_once_with([0])


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


class TestEnvironmentState:
    def test_fields(self):
        state = EnvironmentState(
            cpu_load=0.12,
            on_battery=False,
            thermal_throttling=False,
            frequency_stable=True,
        )
        assert state.cpu_load == pytest.approx(0.12)
        assert state.on_battery is False
        assert state.thermal_throttling is False
        assert state.frequency_stable is True

    def test_frozen(self):
        state = EnvironmentState(cpu_load=0.1, on_battery=None, thermal_throttling=None, frequency_stable=None)
        with pytest.raises(AttributeError):
            state.cpu_load = 0.5  # type: ignore[misc]


class TestCollectEnvironmentState:
    def test_returns_environment_state(self):
        state = collect_environment_state()
        assert isinstance(state, EnvironmentState)

    def test_cpu_load_in_range_or_none(self):
        state = collect_environment_state()
        if state.cpu_load is not None:
            assert 0.0 <= state.cpu_load <= 1.0

    def test_on_battery_bool_or_none(self):
        state = collect_environment_state()
        assert state.on_battery is None or isinstance(state.on_battery, bool)

    def test_thermal_throttling_bool_or_none(self):
        state = collect_environment_state()
        assert state.thermal_throttling is None or isinstance(state.thermal_throttling, bool)

    def test_frequency_stable_bool_or_none(self):
        state = collect_environment_state()
        assert state.frequency_stable is None or isinstance(state.frequency_stable, bool)

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


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------


class TestNoiseEstimate:
    def test_fields(self):
        est = NoiseEstimate(
            relative_jitter=0.018,
            noise_level="low",
            relative_drift=0.004,
            drift_level="low",
        )
        assert est.relative_jitter == pytest.approx(0.018)
        assert est.noise_level == "low"
        assert est.relative_drift == pytest.approx(0.004)
        assert est.drift_level == "low"
        assert est.iteration_count == 0
        assert est.median_sample_seconds is None

    def test_frozen(self):
        est = NoiseEstimate(relative_jitter=0.018, noise_level="low", relative_drift=0.004, drift_level="low")
        with pytest.raises(AttributeError):
            est.noise_level = "high"  # type: ignore[misc]


class TestEstimateNoise:
    def test_returns_noise_estimate(self):
        result = NoiseAnalyzer().analyze(iterations=10)
        assert isinstance(result, NoiseEstimate)

    def test_level_is_valid(self):
        result = NoiseAnalyzer().analyze(iterations=10)
        assert result.noise_level in {"low", "moderate", "high"}
        assert result.drift_level in {"low", "moderate", "high"}

    def test_relative_jitter_non_negative(self):
        result = NoiseAnalyzer().analyze(iterations=10)
        assert result.relative_jitter >= 0.0
        assert result.relative_drift >= 0.0

    def test_raises_on_fewer_than_two_iterations(self):
        with pytest.raises(ValueError, match="iterations must be at least 2"):
            NoiseAnalyzer().analyze(iterations=1)

def _capture(*durations: float) -> NoiseCapture:
    return NoiseCapture(
        durations=tuple(durations),
        iteration_count=len(durations),
        work_units=512,
    )


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


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------


def _add(a, b):
    return a + b


def _raise_error():
    raise ValueError("boom")


class TestRunIsolated:
    def test_direct_call(self):
        result = run_isolated(_add, args=(2, 3), fresh_process=False)
        assert result == 5

    def test_fresh_process(self):
        result = run_isolated(_add, args=(2, 3), fresh_process=True, timeout=30)
        assert result == 5

    def test_kwargs_forwarded(self):
        result = run_isolated(_add, kwargs={"a": 10, "b": 20}, fresh_process=False)
        assert result == 30

    def test_timeout_raises(self):
        import time

        with pytest.raises(TimeoutError):
            run_isolated(time.sleep, args=(10,), fresh_process=True, timeout=0.1)

    def test_exception_in_child_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="ValueError: boom"):
            run_isolated(_raise_error, fresh_process=True, timeout=30)

    def test_disable_gc(self):
        result = run_isolated(_add, args=(1, 1), fresh_process=True, disable_gc=True, timeout=30)
        assert result == 2

    def test_fresh_process_uses_spawn_context(self, monkeypatch: pytest.MonkeyPatch):
        calls: list[str] = []

        class DummyConnection:
            def __init__(self, payload=None):
                self._payload = payload

            def poll(self, timeout=None):
                return True

            def recv(self):
                return self._payload

            def close(self):
                return None

        class DummyProcess:
            def __init__(self, *, target, args, daemon):
                self.target = target
                self.args = args
                self.daemon = daemon
                self.exitcode = 0

            def start(self):
                return None

            def join(self, timeout=None):
                return None

            def terminate(self):
                self.exitcode = -15

            def kill(self):
                self.exitcode = -9

        class DummyContext:
            def Pipe(self, duplex=False):
                return DummyConnection((True, 5)), DummyConnection()

            def Process(self, *, target, args, daemon):
                return DummyProcess(target=target, args=args, daemon=daemon)

        monkeypatch.setattr(
            "benchcaddy.isolation.process.multiprocessing.get_context",
            lambda method: calls.append(method) or DummyContext(),
        )

        assert run_isolated(_add, args=(2, 3), fresh_process=True, timeout=1.0) == 5
        assert calls == ["spawn"]


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


class TestReliabilityReportFrozen:
    def test_frozen(self):
        report = ReliabilityReport(timing_stability="HIGH", environmental_quality="HIGH", warnings=())
        with pytest.raises(AttributeError):
            report.timing_stability = "LOW"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CLI check command
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


class TestCheckCommand:
    def test_check_runs_without_error(self, stubbed_check_dependencies: list[int]):
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0
        assert stubbed_check_dependencies == [50]

    def test_check_json_output(self, stubbed_check_dependencies: list[int]):
        result = runner.invoke(app, ["check", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "timing_stability" in payload
        assert "environmental_quality" in payload
        assert "warnings" in payload
        assert "noise_level" in payload["noise"]
        assert "drift_level" in payload["noise"]
        assert "environment" in payload
        assert "noise" in payload
        assert "affinity" in payload
        assert payload["timing_stability"] in {"HIGH", "FAIR", "LOW"}
        assert payload["environmental_quality"] in {"HIGH", "FAIR", "LOW"}
        assert payload["noise"]["noise_level"] in {"low", "moderate", "high"}
        assert payload["noise"]["drift_level"] in {"low", "moderate", "high"}
        assert stubbed_check_dependencies == [50]

    def test_check_with_noise_iterations(self, stubbed_check_dependencies: list[int]):
        result = runner.invoke(app, ["check", "--noise-iterations", "10"])
        assert result.exit_code == 0
        assert stubbed_check_dependencies == [10]

    def test_check_json_environment_keys(self, stubbed_check_dependencies: list[int]):
        result = runner.invoke(app, ["check", "--json"])
        payload = json.loads(result.output)
        env = payload["environment"]
        assert "cpu_load" in env
        assert "on_battery" in env
        assert "thermal_throttling" in env
        assert "frequency_stable" in env
        assert stubbed_check_dependencies == [50]

    def test_check_json_noise_keys(self, stubbed_check_dependencies: list[int]):
        result = runner.invoke(app, ["check", "--json"])
        payload = json.loads(result.output)
        noise = payload["noise"]
        assert "relative_jitter" in noise
        assert "noise_level" in noise
        assert "relative_drift" in noise
        assert "drift_level" in noise
        assert "iteration_count" in noise
        assert "median_sample_seconds" in noise
        assert noise["relative_jitter"] >= 0.0
        assert noise["relative_drift"] >= 0.0
        assert stubbed_check_dependencies == [50]
