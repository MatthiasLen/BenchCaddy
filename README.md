<img src="https://raw.githubusercontent.com/MatthiasLen/BenchCaddy/main/benchcaddy_logo.png" alt="BenchCaddy logo" width="240"></img>

[![CI](https://github.com/MatthiasLen/BenchCaddy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MatthiasLen/BenchCaddy/actions/workflows/ci.yml)

We all tell ourselves we’re going to use Scalene,PyInstrument or TorchProfile - tools that produce traces so complex and beautiful they belong in a modern art gallery. But let’s be real: most days, "benchmarking" is just us sprinkling time.time() across our code like frantic seasoning on a failing dish. You’re staring at the terminal, trying to remember if the last run was actually faster or if you just happen to be in a better mood, only to realize you’ve already lost the thread. *"Wait, when did I change the naming convention of the log files? Is 'results_v2_final' newer than 'results_new_test'?"*


**BenchCaddy** is the humble sidekick for those of us living in that chaotic middle ground. It replaces "vibes-based" timing with stabilized sweeps and environment metadata, tucking everything into a neat database before your brain can wander. It won’t give you a call-graph of every thread’s inner life, but it will save you from your own memory and provide a summary clean enough to make you look like the organized professional your friends think you are. No traces to decipher, no lost logs, and no more gaslighting yourself.

## Installation

You can install BenchCaddy using [uv](https://github.com/astral-sh/uv), or standard `pip`.

**Using `uv`** Add to your current project dependencies
```bash
uv add benchcaddy
```

**Using `pip`**
```bash
pip install benchcaddy
```

## Quick start

BenchCaddy is designed around two steps:

1. Run a benchmark sweep over one or more configurations.
2. Inspect or compare the recorded results from the database (e.g. using the CLI).

This example stays self-contained and benchmarks a nonlinear iterative transform
with two variants and two input sizes.

```python
import math

from benchcaddy import Sweep, observe


def initial_signal(size: int) -> list[float]:
    return [
        math.sin(index * 0.013) + 0.5 * math.cos(index * 0.007)
        for index in range(size)
    ]


@observe("time")
def nonlinear_iteration(values: list[float], variant: str) -> list[float]:
    next_values: list[float] = []
    for value in values:
        transformed = (
            math.tanh(value * 1.4)
            + 0.75 * math.sin(value * value + 0.2)
            + 0.25 * math.cos(value - 0.1)
        )
        if variant == "stabilized":
            transformed += 0.05 * value * value
        else:
            transformed += 0.03 * math.exp(-(value * value))
        next_values.append(transformed)
    return next_values


@observe("time")
def benchmark_case(size: int, variant: str) -> float:
    values = initial_signal(size)
    for _ in range(8):
        values = nonlinear_iteration(values, variant)
    return sum(abs(value) for value in values)


Sweep(
    target=benchmark_case,
    params={
        "size": [512, 2048],
        "variant": ["baseline", "stabilized"],
    },
    suite_name="nonlinear-transform",
    samples=5,
    warmup_iterations=1,
    verbose=True,
).run()
```

BenchCaddy writes samples, medians, observations, and environment metadata to
`benchcaddy.db` in the current working directory. Those persisted raw samples also drive richer analysis during inspection,
including bootstrap confidence intervals, 
outlier diagnostics, noise warnings, and regression classification.
Those statistics are intended as decision support rather than proof, and they
should be interpreted alongside sample count, variance, outliers, and overall
benchmark-environment stability.

The full runnable example lives in the repository and source distribution at
[`examples/benchmark_nonlinear_transform.py`](https://github.com/MatthiasLen/BenchCaddy/blob/main/examples/benchmark_nonlinear_transform.py)
and supports `--verbose`, `--database`, `--samples`, and `--warmup-iterations`.

## Sweep options

The main public `Sweep(...)` options are:

- `samples`: number of measured samples per configuration
- `warmup_iterations`: warmup runs before sampling begins
- `database_path`: store results in a specific SQLite file instead of `./benchcaddy.db`
- `lock_cpu_affinity`: preserve the current CPU affinity set before benchmarking
- `store_target_return_value=True`: store one accepted target return value per run (`bool`, `int`, `float`, `str`, or 1D numeric vectors from list/tuple/numpy arrays)
- `return_value_postprocessor`: map complex target return values to a supported type before storage
  - when multiple samples are collected, the first measured sample return value is stored for the run
- `reporter`: custom reporter implementing the `SweepReporter` protocol
- `verbose=True`: use the built-in Rich reporter during execution

## Benchmark target contract

A benchmark target must be synchronous from BenchCaddy's perspective: it should
return only after the measured work is complete.

If your workload schedules asynchronous device or background work, make the
target wait for completion before it returns. For example, GPU benchmarks should
perform any required device synchronization inside the benchmarked function so
that BenchCaddy measures the full workload rather than only the launch overhead.

`Sweep` executes targets in a fresh worker process. That means the target must
be importable in the child process: use a module-level function, static method,
or class method. Lambdas, nested or local functions, bound instance methods,
arbitrary callable instances, and script-path targets are not supported by
`Sweep`.

The public `observe(...)` decorator records isolated observations by mode:

- `@observe("time")` records call duration
- `@observe("return")` records a normalized return value when supported
- `@observe("time", "return")` records both

Observation labels come from the decorated function name or qualname.

## CLI and inspect results

List all recorded suites:

```bash
benchcaddy list
```

`list` also shows the observation labels seen across runs in each suite.

Show all recorded runs across the database:

```bash
benchcaddy show
```

Show the recorded runs and environment for a suite:

```bash
benchcaddy show nonlinear-transform
```

Show the detailed timings for a single recorded run:

```bash
benchcaddy show 12
benchcaddy show 2.3
```

Composite run IDs use `SWEEP_ID.RUN_INDEX`, so `2.3` means the third run in
the second recorded sweep.

Show multiple runs side by side in a suite-style view:

```bash
benchcaddy show 4 2.3 1.2
```

When stored, `show` includes a **Return Value** field/column and displays `-` for missing values.

Compare configurations within a suite by median runtime:

```bash
benchcaddy compare nonlinear-transform
```

Compare a suite against a selected recorded run instead of the best run:

```bash
benchcaddy compare nonlinear-transform 2.4
```

Pin a suite baseline and reuse it later without repeating the run ID:

```bash
benchcaddy compare nonlinear-transform 2.4 --pin-baseline
benchcaddy compare nonlinear-transform --use-baseline
```

Restrict a suite comparison to runs that match selected configuration keys from
the reference run:

```bash
benchcaddy compare nonlinear-transform 2.4 --strict size
benchcaddy compare nonlinear-transform 2.4 --strict size variant
benchcaddy compare nonlinear-transform 2.4 --strict variant
```

Compare two specific runs directly:


```bash
benchcaddy compare 12 15
benchcaddy compare 2.3 3
```

Direct run comparisons include **Return Value** and **Return Error**:
- numbers: relative error percentage (`abs(candidate - reference) / abs(reference) * 100`)
- 1D numeric vectors (`list` / `tuple` / `numpy.ndarray`): relative error percentage based on Euclidean distance (`||candidate - reference|| / ||reference|| * 100`)
- strings / booleans: equality (`equal` / `different`)

In other words, numeric return errors are reported relative to the reference run's return value (or reference vector magnitude), not as a raw absolute distance.

`compare` now also prints an additive statistical assessment panel for direct
run comparisons and a compact findings panel for suite comparisons. These are
derived from the stored samples and include bootstrap delta confidence
intervals, significance estimates, and regression probabilities.

Inspect the historical drift of a suite configuration over time:

```bash
benchcaddy trend nonlinear-transform
benchcaddy trend nonlinear-transform 2.4
benchcaddy trend nonlinear-transform --limit 8 --window 4
```

`trend` follows the selected baseline configuration over time, shows median
confidence intervals, compares each run to the baseline, and labels rolling
drift as stable, noisy, improving, or regressing.

Inspect the current machine's benchmark reliability signals before recording or comparing runs:

```bash
benchcaddy env
benchcaddy env --json
```

`env` reports timing noise, drift, affinity, and a few environment risk signals
such as CPU load, battery state, thermal throttling, and frequency stability.

## CI/CD integration

BenchCaddy can support CI-oriented benchmark checks without introducing a
separate command surface.

Use `compare --json` for machine-readable output:

```bash
benchcaddy compare nonlinear-transform --json
benchcaddy compare nonlinear-transform 2.4 --json
benchcaddy compare 2.3 3 --json
benchcaddy trend nonlinear-transform --json
```

Use `compare --fail-if-regression PERCENT` to turn the existing regression
classification into a CI gate. The supplied percent is used as the practical
regression threshold for that invocation, so the reported classification and
the exit condition stay aligned.

```bash
benchcaddy compare nonlinear-transform --use-baseline --fail-if-regression 5%
benchcaddy compare 2.3 3 --json --fail-if-regression 5
```

Exit codes for gated compares:

- `0`: comparison completed and the regression gate passed
- `1`: requested suite or run could not be resolved
- `2`: CLI usage error
- `3`: comparison completed and the regression gate failed

When `--fail-if-regression` is enabled, the JSON payload includes a `gate`
object with the threshold, pass/fail state, and any failing runs.

Example GitHub Actions job:

```yaml
jobs:
    benchmark-gate:
        runs-on: ubuntu-latest
        steps:
            - uses: actions/checkout@v4
            - uses: actions/setup-python@v5
              with:
                python-version: '3.12'
            - name: Install BenchCaddy
              run: python -m pip install -e .
            - name: Record benchmark run
              run: python examples/benchmark_nonlinear_transform.py --database benchcaddy.db
            - name: Enforce regression gate
              run: benchcaddy compare nonlinear-transform --json --fail-if-regression 5% --database benchcaddy.db
```

For a baseline-driven workflow, pin the reference run once and reuse it in CI:

```bash
benchcaddy compare nonlinear-transform 2.4 --pin-baseline
benchcaddy compare nonlinear-transform --use-baseline --json --fail-if-regression 5%
```

For more detail in the inspection output, add `--verbose`:

```bash
benchcaddy --verbose show nonlinear-transform
benchcaddy --verbose compare nonlinear-transform
benchcaddy --verbose trend nonlinear-transform
```

## How to read the output

- `Mean +- Std (s)` is the arithmetic mean and sample standard deviation across benchmark samples
- suite comparisons are ranked by median runtime, not by the mean column
- `Best Median (s)`, `Delta vs Best`, and direct-run `Median Delta` / `Median Percent Change` all use median runtime
- `Median CI (s)` is a bootstrap confidence interval around the median runtime
- `MAD (s)` is the median absolute deviation, a robust spread estimate less sensitive to outliers than standard deviation
- `CV` is the coefficient of variation (`std / mean`) and is used as one of the noise-warning signals
- `Warnings` surface low sample counts, wide confidence intervals, high relative variance, and detected outliers
- direct and trend comparisons combine practical thresholds with significance estimates before labeling a run as regressing
- observation tables report per-label timing aggregated across samples
- `Total (s)` in observation tables is the sum across all samples for that label

These signals are conservative heuristics rather than guarantees. Treat
`regressing` as a strong prompt to investigate, `noisy` as a sign to collect
more samples or stabilize the environment, and `stable` as "no meaningful
evidence of change under the current setup" rather than proof that nothing
changed.

## Environment metadata

Every recorded run stores environment details alongside the timing data, including:

- Python version and operating system string
- CPU model and total system memory
- GPU model when it can be detected
- Git branch, commit hash, and dirty state when run inside a Git repository
- process metadata such as PID, priority, affinity, and RSS memory

# Something missing ?

BenchCaddy is intentionally lean. I built it to curb my own occasional "log-file-chaos," but I’m curious how you manage yours. If you’ve got a feature idea, a bug that’s getting on your nerves, or a suggestion for an export format that actually belongs in this decade, open an issue. I’m not trying to build a bloated enterprise behemoth; I just want this to be the best way to track performance without ever having to name a file timings_final_v4_fixed_REALLY.log again.
