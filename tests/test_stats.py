from __future__ import annotations

from benchcaddy.stats import (
    AnalysisOptions,
    analyze_samples,
    bootstrap_log_interval,
    compare_sample_sets,
    median_or_none,
    robust_relative_jitter,
    tail_relative_jitter,
)


def test_median_or_none_handles_empty_and_populated_values() -> None:
    assert median_or_none([]) is None
    assert median_or_none([1.0, 3.0, 2.0]) == 2.0


def test_robust_relative_jitter_uses_scaled_mad() -> None:
    jitter, median_value = robust_relative_jitter([1.0, 1.0, 1.0, 1.3])

    assert median_value == 1.0
    assert jitter == 0.0


def test_tail_relative_jitter_uses_requested_upper_quantile() -> None:
    jitter, median_value = tail_relative_jitter([1.0, 1.0, 1.0, 1.3])

    assert median_value == 1.0
    assert jitter >= 0.20


def test_bootstrap_log_interval_returns_positive_ordered_bounds() -> None:
    interval = bootstrap_log_interval([0.01, 0.02, 0.04, 0.08], resamples=500, seed=7)

    assert interval is not None
    lower, upper = interval
    assert lower > 0.0
    assert upper >= lower


def test_analyze_samples_reports_noise_and_outliers() -> None:
    stats = analyze_samples(
        [0.100, 0.101, 0.099, 0.102, 0.210],
        AnalysisOptions(bootstrap_resamples=500, bootstrap_seed=7),
    )

    assert stats.sample_count == 5
    assert round(stats.median_seconds, 6) == 0.101
    assert round(stats.mad_seconds, 6) == 0.001
    assert stats.coefficient_of_variation is not None
    assert stats.outlier_count == 1
    assert stats.outlier_indices == (4,)
    assert "high_variance" in stats.warnings
    assert "outliers_detected" in stats.warnings
    assert stats.ci_lower_seconds <= stats.median_seconds <= stats.ci_upper_seconds


def test_compare_sample_sets_detects_regression() -> None:
    comparison = compare_sample_sets(
        [0.099, 0.100, 0.101, 0.100, 0.102, 0.099, 0.101],
        [0.139, 0.140, 0.141, 0.142, 0.140, 0.139, 0.141],
        AnalysisOptions(bootstrap_resamples=600, bootstrap_seed=11),
    )

    assert round(comparison.delta_seconds, 6) == 0.040
    assert comparison.percent_change is not None
    assert round(comparison.percent_change, 2) == 40.00
    assert comparison.delta_ci_lower_seconds > 0.0
    assert comparison.regression_probability > 0.95
    assert comparison.statistically_significant is True
    assert comparison.regression_detected is True
    assert comparison.classification == "regressing"


def test_compare_sample_sets_marks_noisy_when_small_and_overlapping() -> None:
    comparison = compare_sample_sets(
        [0.100, 0.160, 0.120],
        [0.110, 0.150, 0.130],
        AnalysisOptions(bootstrap_resamples=500, bootstrap_seed=3),
    )

    assert comparison.regression_detected is False
    assert comparison.classification == "noisy"
    assert "low_sample_count" in comparison.warnings


def test_compare_sample_sets_handles_empty_samples_without_crashing() -> None:
    comparison = compare_sample_sets(
        [],
        [0.110, 0.150, 0.130],
        AnalysisOptions(bootstrap_resamples=500, bootstrap_seed=3),
    )

    assert comparison.regression_detected is False
    assert comparison.statistically_significant is False
    assert comparison.classification == "noisy"
    assert comparison.significance_p_value == 1.0
    assert comparison.delta_ci_lower_seconds == 0.0
    assert comparison.delta_ci_upper_seconds == 0.0
    assert "baseline_empty_samples" in comparison.warnings


def test_analyze_samples_handles_zero_baseline_without_false_noise() -> None:
    stats = analyze_samples(
        [0.0, 0.0, 0.0, 0.0, 0.0],
        AnalysisOptions(bootstrap_resamples=500, bootstrap_seed=5),
    )

    assert stats.sample_count == 5
    assert stats.mean_seconds == 0.0
    assert stats.median_seconds == 0.0
    assert stats.std_seconds == 0.0
    assert stats.coefficient_of_variation == 0.0
    assert stats.ci_width_ratio is None
    assert stats.warnings == ()
    assert stats.is_noisy is False


def test_compare_sample_sets_propagates_noisy_warnings_from_both_sides() -> None:
    comparison = compare_sample_sets(
        [0.100, 0.160, 0.120],
        [0.110, 0.170, 0.130],
        AnalysisOptions(bootstrap_resamples=500, bootstrap_seed=13),
    )

    assert comparison.classification == "noisy"
    assert "low_sample_count" in comparison.warnings
    assert "baseline_low_sample_count" in comparison.warnings
    assert "candidate_low_sample_count" in comparison.warnings
    assert "baseline_noisy" in comparison.warnings
    assert "candidate_noisy" in comparison.warnings
