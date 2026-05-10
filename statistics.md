# Statistical Model

BenchCaddy's statistical output is meant to help interpret repeated benchmark
samples. It is **decision support**, not proof. The labels (`stable`,
`improving`, `regressing`, `noisy`) are heuristics built on a small set of
robust summary statistics and resampling methods.

## Assumptions

BenchCaddy's conclusions are most trustworthy when:

- each sample measures the same workload and code path
- hardware, affinity, thermals, and background load are reasonably stable
- warmup effects are handled before measurement
- samples are independent enough that resampling is meaningful
- the median is an appropriate center estimate for the runtime distribution

If those assumptions are violated, the warnings may still be useful, but the
classification labels should be treated cautiously.

## Per-run analysis

For a single recorded run, BenchCaddy computes:

- **mean** and **sample standard deviation**
- **median runtime**
- **MAD** (median absolute deviation), a more outlier-resistant spread estimate
- **bootstrap confidence interval for the median**
- **coefficient of variation** (`std / mean`)
- **outlier diagnostics** using a modified z-score around the median

### Default thresholds

These defaults come from `AnalysisOptions` in `src/benchcaddy/stats.py`:

- confidence level: **0.95**
- bootstrap resamples: **2000**
- low sample count warning: **fewer than 5 samples**
- high variance warning: **CV >= 0.05**
- wide confidence interval warning: **CI width / |median| >= 0.10**
- outlier warning: **modified z-score > 3.5**

### Why these thresholds exist

- **`< 5` samples**: with very small sample sets, medians and resampling-based
  intervals become unstable enough that BenchCaddy prefers to warn instead of
  projecting confidence.
- **`CV >= 0.05`**: a 5% relative spread is a practical "this may be noisy"
  threshold for microbenchmarks and short-running code, where scheduler jitter,
  cache effects, and thermal drift often dominate.
- **`CI width / median >= 0.10`**: if the median confidence interval spans 10%
  or more of the estimate, BenchCaddy treats the run as too imprecise for
  strong claims.
- **modified z-score `> 3.5`**: this is a common robust outlier heuristic that
  works better than mean/std-based cutoffs on skewed runtime samples.

These are intentionally conservative defaults. They aim to catch suspicious
measurements early, not to certify that a run is "clean."

## Run-to-run comparisons

When comparing a candidate run to a baseline run, BenchCaddy reports:

- **delta seconds**: `candidate median - baseline median`
- **percent change** relative to the baseline median
- **bootstrap delta confidence interval**
- **regression probability**
- **improvement probability**
- **permutation-based significance estimate**

### Methodology

1. BenchCaddy bootstraps both sample sets independently.
2. It computes a median delta distribution from those resamples.
3. It derives:
   - a percentile confidence interval for the median delta
   - the share of bootstrap deltas above or below a practical threshold
4. It runs a permutation test on the pooled samples to estimate how surprising
   the observed median gap would be under a no-difference assumption.

### Default comparison thresholds

- significance level: **0.05**
- practical regression threshold: **5% of the baseline median**

### Interpretation of comparison fields

- **Regression probability** is the fraction of bootstrap deltas at or above the
  practical regression threshold. It is a stability-oriented heuristic, not a
  calibrated Bayesian posterior.
- **Improvement probability** is the symmetric heuristic for speedups.
- **Significance estimate / p-value** comes from the permutation test and is
  best read as "how often would a gap this large appear by chance if both runs
  came from the same distribution?"

### Classification rules

BenchCaddy labels a comparison as:

- **`regressing`** when all of the following are true:
  - permutation `p <= 0.05`
  - the observed median slowdown exceeds the 5% practical threshold
  - bootstrap regression probability is at least `0.95`
- **`improving`** when the result is statistically significant in the opposite
  direction and bootstrap improvement probability is at least `0.95`
- **`noisy`** when warning signals are present but the evidence is not strong
  enough to call a regression or improvement
- **`stable`** otherwise

This design intentionally combines **statistical evidence** with a **minimum
effect size** so that tiny but repeatable differences are less likely to be
overstated.

## Trend analysis

`benchcaddy trend` applies the same comparison logic in two ways:

- each matching run is compared against the selected baseline run
- each run is also compared against trailing samples from the previous matching
  runs inside the rolling drift window (default: **5**)

The baseline view answers "how far are we from the chosen reference?" The drift
view answers "does the recent local history look stable, noisy, improving, or
regressing?"

## False positives and false negatives

BenchCaddy can still misclassify benchmark data:

- **false positives** can happen when samples are autocorrelated, system load
  changes during the sweep, or a few structured outliers survive the robust
  checks
- **false negatives** can happen when sample counts are small, variance is high,
  or the slowdown is real but below the 5% practical threshold

In practice:

- treat `regressing` as a strong signal to investigate
- treat `noisy` as "collect more samples or stabilize the environment"
- treat `stable` as "no meaningful evidence of change under the current setup,"
  not as proof that the two configurations are identical

## Benchmark interpretation guide

Use the statistical output as a checklist:

1. **Check sample count first.** Fewer than 5 samples means the rest of the
   output is low-confidence.
2. **Check warnings next.** High variance, wide CIs, and outliers often explain
   surprising labels.
3. **Prefer median-based fields** over the mean when comparing variants.
4. **Look for agreement across signals.** The strongest cases have a clear delta,
   a confidence interval away from zero, low warnings, and a stable trend.
5. **Re-run before acting** when the change is operationally important.

If you need stricter or looser behavior, the CLI exposes the main comparison
knobs (`--confidence-level`, `--bootstrap-resamples`, `--noise-threshold`,
`--significance-level`, `--regression-threshold`) for `compare` and `trend`.
