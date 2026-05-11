from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose

import numpy as np


@dataclass(frozen=True)
class AnalysisOptions:
    confidence_level: float = 0.95
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 0
    noise_cv_threshold: float = 0.05
    noise_ci_ratio_threshold: float = 0.10
    outlier_z_threshold: float = 3.5
    significance_level: float = 0.05
    regression_threshold_percent: float = 5.0
    drift_window_size: int = 5


@dataclass(frozen=True)
class RunStatistics:
    sample_count: int
    mean_seconds: float
    median_seconds: float
    std_seconds: float
    min_seconds: float | None
    max_seconds: float | None
    mad_seconds: float
    coefficient_of_variation: float | None
    ci_lower_seconds: float
    ci_upper_seconds: float
    ci_width_ratio: float | None
    outlier_count: int
    outlier_indices: tuple[int, ...]
    warnings: tuple[str, ...]
    is_noisy: bool

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonStatistics:
    delta_seconds: float
    percent_change: float | None
    delta_ci_lower_seconds: float
    delta_ci_upper_seconds: float
    regression_probability: float
    improvement_probability: float
    significance_p_value: float
    statistically_significant: bool
    practical_threshold_seconds: float
    exceeds_practical_threshold: bool
    regression_detected: bool
    classification: str
    warnings: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def _as_array(samples: list[float] | tuple[float, ...]) -> np.ndarray:
    return np.asarray(samples, dtype=float)


def _validate_options(options: AnalysisOptions) -> None:
    if not 0.0 < options.confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    if options.bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100.")
    if not 0.0 < options.significance_level < 1.0:
        raise ValueError("significance_level must be between 0 and 1.")
    if options.drift_window_size < 2:
        raise ValueError("drift_window_size must be at least 2.")


def _median_abs_deviation(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _sample_std(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _bootstrap_estimates(
    values: np.ndarray,
    *,
    estimator,
    resamples: int,
    seed: int,
) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, values.size, size=(resamples, values.size))
    sampled = values[sample_indices]
    return estimator(sampled, axis=1)


def _bootstrap_interval(
    values: np.ndarray,
    *,
    estimator,
    options: AnalysisOptions,
    seed_offset: int = 0,
) -> tuple[float, float]:
    if values.size == 0:
        return (0.0, 0.0)
    estimates = _bootstrap_estimates(
        values,
        estimator=estimator,
        resamples=options.bootstrap_resamples,
        seed=options.bootstrap_seed + seed_offset,
    )
    alpha = (1.0 - options.confidence_level) / 2.0
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def _coefficient_of_variation(mean_seconds: float, std_seconds: float) -> float | None:
    if isclose(mean_seconds, 0.0, abs_tol=1e-12):
        return 0.0 if isclose(std_seconds, 0.0, abs_tol=1e-12) else None
    return float(std_seconds / abs(mean_seconds))


def _outlier_indices(values: np.ndarray, *, threshold: float) -> tuple[int, ...]:
    if values.size < 3:
        return ()
    mad = _median_abs_deviation(values)
    if isclose(mad, 0.0, abs_tol=1e-12):
        return ()
    med = float(np.median(values))
    modified_z = 0.6745 * (values - med) / mad
    return tuple(int(index) for index in np.nonzero(np.abs(modified_z) > threshold)[0].tolist())


def analyze_samples(
    samples: list[float] | tuple[float, ...],
    options: AnalysisOptions | None = None,
) -> RunStatistics:
    chosen_options = options or AnalysisOptions()
    _validate_options(chosen_options)
    values = _as_array(samples)
    mean_seconds = float(np.mean(values)) if values.size else 0.0
    median_seconds = float(np.median(values)) if values.size else 0.0
    std_seconds = _sample_std(values)
    mad_seconds = _median_abs_deviation(values)
    coefficient_of_variation = _coefficient_of_variation(mean_seconds, std_seconds)
    ci_lower_seconds, ci_upper_seconds = _bootstrap_interval(
        values,
        estimator=np.median,
        options=chosen_options,
    )
    ci_width_ratio = None
    if not isclose(median_seconds, 0.0, abs_tol=1e-12):
        ci_width_ratio = float((ci_upper_seconds - ci_lower_seconds) / abs(median_seconds))

    outlier_indices = _outlier_indices(values, threshold=chosen_options.outlier_z_threshold)
    warnings: list[str] = []
    if values.size < 5:
        warnings.append("low_sample_count")
    if coefficient_of_variation is not None and coefficient_of_variation >= chosen_options.noise_cv_threshold:
        warnings.append("high_variance")
    if ci_width_ratio is not None and ci_width_ratio >= chosen_options.noise_ci_ratio_threshold:
        warnings.append("wide_confidence_interval")
    if outlier_indices:
        warnings.append("outliers_detected")

    return RunStatistics(
        sample_count=int(values.size),
        mean_seconds=mean_seconds,
        median_seconds=median_seconds,
        std_seconds=std_seconds,
        min_seconds=None if values.size == 0 else float(np.min(values)),
        max_seconds=None if values.size == 0 else float(np.max(values)),
        mad_seconds=mad_seconds,
        coefficient_of_variation=coefficient_of_variation,
        ci_lower_seconds=ci_lower_seconds,
        ci_upper_seconds=ci_upper_seconds,
        ci_width_ratio=ci_width_ratio,
        outlier_count=len(outlier_indices),
        outlier_indices=outlier_indices,
        warnings=tuple(warnings),
        is_noisy=bool(warnings),
    )


def compare_sample_sets(
    baseline_samples: list[float] | tuple[float, ...],
    candidate_samples: list[float] | tuple[float, ...],
    options: AnalysisOptions | None = None,
) -> ComparisonStatistics:
    chosen_options = options or AnalysisOptions()
    _validate_options(chosen_options)
    baseline_values = _as_array(baseline_samples)
    candidate_values = _as_array(candidate_samples)

    baseline_stats = analyze_samples(list(baseline_values), chosen_options)
    candidate_stats = analyze_samples(list(candidate_values), chosen_options)
    delta_seconds = candidate_stats.median_seconds - baseline_stats.median_seconds
    percent_change = None
    if not isclose(baseline_stats.median_seconds, 0.0, abs_tol=1e-12):
        percent_change = float((delta_seconds / baseline_stats.median_seconds) * 100.0)

    if baseline_values.size == 0 or candidate_values.size == 0:
        warnings = [
            *(["low_sample_count"] if baseline_stats.sample_count < 5 or candidate_stats.sample_count < 5 else []),
            *(["baseline_empty_samples"] if baseline_values.size == 0 else []),
            *(["candidate_empty_samples"] if candidate_values.size == 0 else []),
            *(f"baseline_{warning}" for warning in baseline_stats.warnings),
            *(f"candidate_{warning}" for warning in candidate_stats.warnings),
        ]
        return ComparisonStatistics(
            delta_seconds=delta_seconds,
            percent_change=percent_change,
            delta_ci_lower_seconds=0.0,
            delta_ci_upper_seconds=0.0,
            regression_probability=0.0,
            improvement_probability=0.0,
            significance_p_value=1.0,
            statistically_significant=False,
            practical_threshold_seconds=abs(baseline_stats.median_seconds) * (chosen_options.regression_threshold_percent / 100.0),
            exceeds_practical_threshold=False,
            regression_detected=False,
            classification="noisy",
            warnings=tuple(dict.fromkeys(warnings)),
        )

    rng = np.random.default_rng(chosen_options.bootstrap_seed + 101)
    baseline_indices = rng.integers(0, baseline_values.size, size=(chosen_options.bootstrap_resamples, baseline_values.size))
    candidate_indices = rng.integers(0, candidate_values.size, size=(chosen_options.bootstrap_resamples, candidate_values.size))
    baseline_bootstrap = np.median(baseline_values[baseline_indices], axis=1)
    candidate_bootstrap = np.median(candidate_values[candidate_indices], axis=1)
    delta_distribution = candidate_bootstrap - baseline_bootstrap
    alpha = (1.0 - chosen_options.confidence_level) / 2.0
    delta_ci_lower_seconds = float(np.quantile(delta_distribution, alpha))
    delta_ci_upper_seconds = float(np.quantile(delta_distribution, 1.0 - alpha))

    practical_threshold_seconds = abs(baseline_stats.median_seconds) * (chosen_options.regression_threshold_percent / 100.0)
    regression_probability = float(np.mean(delta_distribution >= practical_threshold_seconds))
    improvement_probability = float(np.mean(delta_distribution <= -practical_threshold_seconds))

    pooled = np.concatenate([baseline_values, candidate_values])
    observed_abs_delta = abs(delta_seconds)
    permutation_deltas: list[float] = []
    for _ in range(chosen_options.bootstrap_resamples):
        permutation = rng.permutation(pooled)
        baseline_perm = permutation[: baseline_values.size]
        candidate_perm = permutation[baseline_values.size :]
        permutation_deltas.append(abs(float(np.median(candidate_perm) - np.median(baseline_perm))))
    significance_p_value = float(np.mean(np.asarray(permutation_deltas) >= observed_abs_delta))

    warnings = [
        *( ["low_sample_count"] if baseline_stats.sample_count < 5 or candidate_stats.sample_count < 5 else [] ),
        *(f"baseline_{warning}" for warning in baseline_stats.warnings),
        *(f"candidate_{warning}" for warning in candidate_stats.warnings),
    ]
    if baseline_stats.is_noisy:
        warnings.append("baseline_noisy")
    if candidate_stats.is_noisy:
        warnings.append("candidate_noisy")

    statistically_significant = bool(significance_p_value <= chosen_options.significance_level)
    exceeds_practical_threshold = bool(delta_seconds >= practical_threshold_seconds)
    regression_detected = bool(
        statistically_significant
        and exceeds_practical_threshold
        and regression_probability >= 1.0 - chosen_options.significance_level
    )

    if regression_detected:
        classification = "regressing"
    elif statistically_significant and improvement_probability >= 1.0 - chosen_options.significance_level:
        classification = "improving"
    elif warnings:
        classification = "noisy"
    else:
        classification = "stable"

    return ComparisonStatistics(
        delta_seconds=delta_seconds,
        percent_change=percent_change,
        delta_ci_lower_seconds=delta_ci_lower_seconds,
        delta_ci_upper_seconds=delta_ci_upper_seconds,
        regression_probability=regression_probability,
        improvement_probability=improvement_probability,
        significance_p_value=significance_p_value,
        statistically_significant=statistically_significant,
        practical_threshold_seconds=practical_threshold_seconds,
        exceeds_practical_threshold=exceeds_practical_threshold,
        regression_detected=regression_detected,
        classification=classification,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def drift_status(
    samples: list[float] | tuple[float, ...],
    reference_samples: list[float] | tuple[float, ...],
    options: AnalysisOptions | None = None,
) -> str:
    comparison = compare_sample_sets(reference_samples, samples, options)
    return comparison.classification
