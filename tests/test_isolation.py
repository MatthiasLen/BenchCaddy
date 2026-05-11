"""Tests for the benchcaddy.isolation package."""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------


class TestNoiseEstimate:
    def test_fields(self):
        est = NoiseEstimate(relative_jitter=0.018, level="low")
        assert est.relative_jitter == pytest.approx(0.018)
        assert est.level == "low"

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
        from benchcaddy.isolation import noise as noise_module

        # Force a low jitter scenario via monkeypatching stdev/fmean
        with (
            patch.object(noise_module, "stdev", return_value=0.001),
            patch.object(noise_module, "fmean", return_value=1.0),
        ):
            result = estimate_noise(iterations=2)
        assert result.level == "low"

    def test_high_noise_classification(self):
        from benchcaddy.isolation import noise as noise_module

        with (
            patch.object(noise_module, "stdev", return_value=0.5),
            patch.object(noise_module, "fmean", return_value=1.0),
        ):
            result = estimate_noise(iterations=2)
        assert result.level == "high"


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


def _make_noise(level="low", jitter=0.01) -> NoiseEstimate:
    return NoiseEstimate(relative_jitter=jitter, level=level)


class TestReliabilityReport:
    def test_clean_environment_is_high(self):
        report = build_reliability_report(environment=_make_env(), noise=_make_noise())
        assert report.statistical_confidence == "HIGH"
        assert report.environmental_quality == "HIGH"
        assert report.warnings == ()

    def test_battery_warning(self):
        report = build_reliability_report(environment=_make_env(on_battery=True), noise=_make_noise())
        assert any("battery" in w.lower() for w in report.warnings)
        assert report.environmental_quality in {"FAIR", "LOW"}

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
        assert report.statistical_confidence == "LOW"
        assert any("jitter" in w.lower() for w in report.warnings)

    def test_moderate_noise_warning(self):
        report = build_reliability_report(environment=_make_env(), noise=_make_noise(level="moderate", jitter=0.05))
        assert report.statistical_confidence == "FAIR"
        assert any("jitter" in w.lower() for w in report.warnings)

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
        assert noise["relative_jitter"] >= 0.0
