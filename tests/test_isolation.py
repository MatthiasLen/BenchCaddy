"""Tests for the benchcaddy.isolation package."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from benchcaddy.cli import app
from benchcaddy.isolation import (
    EnvironmentState,
    NoiseEstimate,
    ReliabilityReport,
    build_reliability_report,
    collect_environment_state,
    estimate_noise,
    estimate_noise_consensus,
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
        with patch("benchcaddy.isolation.affinity.psutil.Process", return_value=mock_process):
            assert get_affinity() is None


class TestSetAffinity:
    def test_empty_list_returns_false(self):
        assert set_affinity([]) is False

    def test_no_cpu_affinity_attribute(self):
        mock_process = MagicMock(spec=[])  # no cpu_affinity attribute
        with patch("benchcaddy.isolation.affinity.psutil.Process", return_value=mock_process):
            assert set_affinity([0]) is False

    def test_access_denied_returns_false(self):
        import psutil

        mock_process = MagicMock()
        mock_process.cpu_affinity.side_effect = psutil.AccessDenied(0)
        with patch("benchcaddy.isolation.affinity.psutil.Process", return_value=mock_process):
            assert set_affinity([0]) is False

    def test_success_returns_true(self):
        mock_process = MagicMock()
        mock_process.cpu_affinity.return_value = None
        with patch("benchcaddy.isolation.affinity.psutil.Process", return_value=mock_process):
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
        est = NoiseEstimate(relative_jitter=0.018, level="low")
        assert est.relative_jitter == pytest.approx(0.018)
        assert est.level == "low"
        assert est.statistical_confidence == "LOW"
        assert est.repeat_count == 1
        assert est.robust_jitter is None
        assert est.tail_jitter is None
        assert est.median_sample_seconds is None
        assert est.ci_lower is None
        assert est.ci_upper is None
        assert est.relative_margin_of_error is None

    def test_frozen(self):
        est = NoiseEstimate(relative_jitter=0.018, level="low")
        with pytest.raises(AttributeError):
            est.level = "high"  # type: ignore[misc]


class TestEstimateNoise:
    def test_returns_noise_estimate(self):
        result = estimate_noise(iterations=10)
        assert isinstance(result, NoiseEstimate)

    def test_level_is_valid(self):
        result = estimate_noise(iterations=10)
        assert result.level in {"low", "moderate", "high"}

    def test_relative_jitter_non_negative(self):
        result = estimate_noise(iterations=10)
        assert result.relative_jitter >= 0.0

    def test_raises_on_fewer_than_two_iterations(self):
        with pytest.raises(ValueError, match="iterations must be at least 2"):
            estimate_noise(iterations=1)

    def test_low_noise_classification(self):
        durations = [1.0, 1.0, 1.0, 1.0]

        with (
            patch("benchcaddy.isolation.noise._calibrate_work_units", return_value=512),
            patch("benchcaddy.isolation.noise._probe_once", side_effect=durations * 2 + durations),
        ):
            result = estimate_noise(iterations=len(durations))

        assert result.level == "low"
        assert result.robust_jitter == pytest.approx(0.0)
        assert result.tail_jitter == pytest.approx(0.0)

    def test_high_noise_classification(self):
        durations = [1.0, 1.0, 1.0, 1.30]

        with (
            patch("benchcaddy.isolation.noise._calibrate_work_units", return_value=512),
            patch("benchcaddy.isolation.noise._probe_once", side_effect=durations * 2 + durations),
        ):
            result = estimate_noise(iterations=len(durations))

        assert result.level == "high"
        assert result.tail_jitter is not None
        assert result.tail_jitter >= 0.20

    def test_calibrates_probe_above_timer_resolution(self):
        from benchcaddy.isolation import noise as noise_module

        recorded_work_units: list[int] = []

        def fake_probe_once(work_units: int) -> float:
            recorded_work_units.append(work_units)
            if work_units < 512:
                return 1e-06
            return 8e-05

        with (
            patch.object(noise_module, "_probe_target_seconds", return_value=5e-05),
            patch.object(noise_module, "_probe_once", side_effect=fake_probe_once),
        ):
            result = estimate_noise(iterations=4)

        assert result.relative_jitter == pytest.approx(0.0)
        assert recorded_work_units[:25] == [32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512]
        assert set(recorded_work_units[25:]) == {512}

    def test_single_isolated_outlier_does_not_dominate_tail_metric(self):
        from benchcaddy.isolation import noise as noise_module

        durations = [1.0] * 39 + [6.0]

        with (
            patch.object(noise_module, "_calibrate_work_units", return_value=512),
            patch.object(noise_module, "_probe_once", side_effect=durations[:5] + durations),
        ):
            result = estimate_noise(iterations=len(durations))

        assert result.relative_jitter == pytest.approx(0.0)
        assert result.level == "low"

    def test_consensus_uses_median_relative_jitter(self):
        from benchcaddy.isolation import noise as noise_module

        estimates = [
            NoiseEstimate(relative_jitter=0.02, level="low"),
            NoiseEstimate(relative_jitter=0.25, level="high"),
            NoiseEstimate(relative_jitter=0.06, level="moderate"),
            NoiseEstimate(relative_jitter=0.05, level="moderate"),
            NoiseEstimate(relative_jitter=0.50, level="high"),
            NoiseEstimate(relative_jitter=0.08, level="moderate"),
            NoiseEstimate(relative_jitter=0.03, level="low"),
        ]

        with (
            patch.object(noise_module, "_calibrate_work_units", return_value=512),
            patch.object(noise_module, "_estimate_noise_once", side_effect=estimates),
        ):
            result = estimate_noise_consensus(iterations=10, repeats=len(estimates))

        assert result.relative_jitter == pytest.approx(0.06)
        assert result.level == "moderate"
        assert result.iteration_count == 10
        assert result.repeat_count == len(estimates)
        assert result.robust_jitter is None
        assert result.tail_jitter is None
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.ci_lower <= result.relative_jitter <= result.ci_upper
        assert result.relative_margin_of_error is not None

    def test_bootstrap_interval_uses_log_scale_for_positive_values(self):
        estimates = [
            NoiseEstimate(relative_jitter=0.01, level="low"),
            NoiseEstimate(relative_jitter=0.02, level="moderate"),
            NoiseEstimate(relative_jitter=0.04, level="moderate"),
            NoiseEstimate(relative_jitter=0.08, level="moderate"),
        ]

        with (
            patch("benchcaddy.isolation.noise._calibrate_work_units", return_value=512),
            patch("benchcaddy.isolation.noise._estimate_noise_once", side_effect=estimates),
        ):
            result = estimate_noise_consensus(iterations=10, repeats=len(estimates))

        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.ci_lower > 0.0
        assert result.ci_upper >= result.ci_lower

    def test_confidence_is_high_when_interval_stays_within_one_bucket(self):
        estimates = [
            NoiseEstimate(relative_jitter=0.012, level="low"),
            NoiseEstimate(relative_jitter=0.013, level="low"),
            NoiseEstimate(relative_jitter=0.014, level="low"),
            NoiseEstimate(relative_jitter=0.015, level="low"),
            NoiseEstimate(relative_jitter=0.016, level="low"),
            NoiseEstimate(relative_jitter=0.017, level="low"),
            NoiseEstimate(relative_jitter=0.018, level="low"),
        ]

        with (
            patch("benchcaddy.isolation.noise._calibrate_work_units", return_value=512),
            patch("benchcaddy.isolation.noise._estimate_noise_once", side_effect=estimates),
        ):
            result = estimate_noise_consensus(iterations=10, repeats=len(estimates))

        assert result.statistical_confidence == "HIGH"

    def test_confidence_is_fair_when_interval_crosses_one_threshold(self):
        estimates = [
            NoiseEstimate(relative_jitter=0.040, level="low"),
            NoiseEstimate(relative_jitter=0.045, level="low"),
            NoiseEstimate(relative_jitter=0.048, level="low"),
            NoiseEstimate(relative_jitter=0.052, level="moderate"),
            NoiseEstimate(relative_jitter=0.058, level="moderate"),
            NoiseEstimate(relative_jitter=0.064, level="moderate"),
            NoiseEstimate(relative_jitter=0.070, level="moderate"),
        ]

        with (
            patch("benchcaddy.isolation.noise._calibrate_work_units", return_value=512),
            patch("benchcaddy.isolation.noise._estimate_noise_once", side_effect=estimates),
        ):
            result = estimate_noise_consensus(iterations=10, repeats=len(estimates))

        assert result.statistical_confidence == "FAIR"

    def test_confidence_is_low_when_interval_spans_multiple_buckets(self):
        estimates = [
            NoiseEstimate(relative_jitter=0.030, level="low"),
            NoiseEstimate(relative_jitter=0.045, level="low"),
            NoiseEstimate(relative_jitter=0.060, level="moderate"),
            NoiseEstimate(relative_jitter=0.120, level="moderate"),
            NoiseEstimate(relative_jitter=0.220, level="high"),
            NoiseEstimate(relative_jitter=0.180, level="high"),
            NoiseEstimate(relative_jitter=0.240, level="high"),
        ]

        with (
            patch("benchcaddy.isolation.noise._calibrate_work_units", return_value=512),
            patch("benchcaddy.isolation.noise._estimate_noise_once", side_effect=estimates),
        ):
            result = estimate_noise_consensus(iterations=10, repeats=len(estimates))

        assert result.statistical_confidence == "LOW"

    def test_consensus_reuses_one_calibration(self):
        from benchcaddy.isolation import noise as noise_module

        calibration_calls: list[float] = []

        def fake_calibrate(target_seconds: float) -> int:
            calibration_calls.append(target_seconds)
            return 512

        with (
            patch.object(noise_module, "_calibrate_work_units", side_effect=fake_calibrate),
            patch.object(noise_module, "_probe_once", return_value=1.0),
        ):
            result = estimate_noise_consensus(iterations=5, repeats=3)

        assert result.repeat_count == 3
        assert len(calibration_calls) == 1


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
        with pytest.raises(RuntimeError, match="Isolated process raised"):
            run_isolated(_raise_error, fresh_process=True, timeout=30)

    def test_disable_gc(self):
        result = run_isolated(_add, args=(1, 1), fresh_process=True, disable_gc=True, timeout=30)
        assert result == 2


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
    level="low",
    jitter=0.01,
    *,
    statistical_confidence: str = "HIGH",
    iteration_count: int = 500,
    repeat_count: int = 7,
    ci_lower: float | None = 0.008,
    ci_upper: float | None = 0.012,
    relative_margin_of_error: float | None = 0.20,
) -> NoiseEstimate:
    return NoiseEstimate(
        relative_jitter=jitter,
        level=level,
        statistical_confidence=statistical_confidence,
        iteration_count=iteration_count,
        repeat_count=repeat_count,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        relative_margin_of_error=relative_margin_of_error,
    )


class TestReliabilityReport:
    def test_clean_environment_is_high(self):
        report = build_reliability_report(
            environment=_make_env(),
            noise=_make_noise(jitter=0.01),
        )
        assert report.statistical_confidence == "HIGH"
        assert report.environmental_quality == "HIGH"
        assert report.warnings == ()

    def test_high_jitter_can_still_have_high_statistical_confidence(self):
        report = build_reliability_report(
            environment=_make_env(),
            noise=NoiseEstimate(
                relative_jitter=0.18,
                level="high",
                statistical_confidence="HIGH",
                iteration_count=500,
                repeat_count=7,
                ci_lower=0.16,
                ci_upper=0.20,
                relative_margin_of_error=0.1111111111111111,
            ),
        )
        assert report.statistical_confidence == "HIGH"
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
        report = build_reliability_report(environment=_make_env(), noise=_make_noise(level="high", jitter=0.15))
        assert report.statistical_confidence == "HIGH"
        assert report.environmental_quality == "LOW"
        assert any("jitter" in w.lower() for w in report.warnings)

    def test_moderate_noise_warning(self):
        report = build_reliability_report(environment=_make_env(), noise=_make_noise(level="moderate", jitter=0.05))
        assert report.statistical_confidence == "HIGH"
        assert report.environmental_quality == "FAIR"
        assert any("jitter" in w.lower() for w in report.warnings)

    def test_combined_moderate_risks_degrade_environment_to_low(self):
        report = build_reliability_report(
            environment=_make_env(on_battery=True, cpu_load=0.20),
            noise=_make_noise(level="moderate", jitter=0.05),
        )
        assert report.environmental_quality == "LOW"

    def test_low_statistical_confidence_adds_warning(self):
        report = build_reliability_report(
            environment=_make_env(),
            noise=_make_noise(statistical_confidence="LOW"),
        )
        assert any("repeat stability" in w.lower() for w in report.warnings)

    def test_warnings_is_tuple(self):
        report = build_reliability_report(environment=_make_env(), noise=_make_noise())
        assert isinstance(report.warnings, tuple)

    def test_format_no_warnings(self):
        report = build_reliability_report(environment=_make_env(), noise=_make_noise())
        text = report.format()
        assert "Statistical confidence: HIGH" in text
        assert "Environmental quality:  HIGH" in text
        assert "Warnings:" not in text

    def test_format_with_warnings(self):
        report = build_reliability_report(environment=_make_env(on_battery=True), noise=_make_noise())
        text = report.format()
        assert "Warnings:" in text


class TestReliabilityReportFrozen:
    def test_frozen(self):
        report = ReliabilityReport(statistical_confidence="HIGH", environmental_quality="HIGH", warnings=())
        with pytest.raises(AttributeError):
            report.statistical_confidence = "LOW"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CLI check command
# ---------------------------------------------------------------------------


runner = CliRunner()


class TestCheckCommand:
    def test_check_runs_without_error(self):
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0

    def test_check_json_output(self):
        result = runner.invoke(app, ["check", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "statistical_confidence" in payload
        assert "environmental_quality" in payload
        assert "warnings" in payload
        assert "robust_jitter" in payload["noise"]
        assert "tail_jitter" in payload["noise"]
        assert "environment" in payload
        assert "noise" in payload
        assert "affinity" in payload
        assert payload["statistical_confidence"] in {"HIGH", "FAIR", "LOW"}
        assert payload["environmental_quality"] in {"HIGH", "FAIR", "LOW"}
        assert payload["noise"]["level"] in {"low", "moderate", "high"}

    def test_check_with_noise_iterations(self):
        result = runner.invoke(app, ["check", "--noise-iterations", "10"])
        assert result.exit_code == 0

    def test_check_json_environment_keys(self):
        result = runner.invoke(app, ["check", "--json"])
        payload = json.loads(result.output)
        env = payload["environment"]
        assert "cpu_load" in env
        assert "on_battery" in env
        assert "thermal_throttling" in env
        assert "frequency_stable" in env

    def test_check_json_noise_keys(self):
        result = runner.invoke(app, ["check", "--json"])
        payload = json.loads(result.output)
        noise = payload["noise"]
        assert "relative_jitter" in noise
        assert "level" in noise
        assert "iteration_count" in noise
        assert "repeat_count" in noise
        assert "ci_lower" in noise
        assert "ci_upper" in noise
        assert "relative_margin_of_error" in noise
        assert noise["relative_jitter"] >= 0.0
