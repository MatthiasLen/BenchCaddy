<img src="https://raw.githubusercontent.com/MatthiasLen/BenchCaddy/main/benchcaddy_logo.png" alt="BenchCaddy logo" width="240"></img>

[![CI](https://github.com/MatthiasLen/BenchCaddy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MatthiasLen/BenchCaddy/actions/workflows/ci.yml)

BenchCaddy is a lightweight Python benchmarking toolkit for software developers, AI engineers, and data scientists who need repeatable performance measurements without building custom log-file workflows.
It runs parameter sweeps in an isolated worker process, stores raw samples and environment metadata in SQLite, and gives you a CLI to inspect runs, compare configurations, pin baselines, and track drift over time.
It is for the gap between full profilers and a directory full of `timings_final_v4_really.csv`.


<img src="https://raw.githubusercontent.com/MatthiasLen/BenchCaddy/main/bc_trends.png" alt="BenchCaddy trend summary overview" width="640"></img>

## Why BenchCaddy

- Run repeatable benchmark sweeps across parameter grids
- Persist raw samples, observations, and machine metadata in `benchcaddy.db`
- Compare runs with median-based summaries, confidence intervals, and noise warnings
- Pin suite baselines and reuse them for local checks or CI gates
- Capture supported return values to validate correctness alongside runtime

BenchCaddy is intentionally narrow: it helps you answer "is this actually faster, and under what environment?" It is not a profiler or tracing system.

## Installation

Install with [uv](https://github.com/astral-sh/uv) or `pip`.

```bash
uv add benchcaddy
```

```bash
pip install benchcaddy
```

## Quick Start

BenchCaddy workflows have two steps:

1. Run a benchmark sweep over one or more configurations.
2. Inspect, compare, or trend the recorded results from the database.

This example benchmarks a nonlinear transform with two variants and two input sizes.

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
        transformed = math.tanh(value * 1.4) + math.sin(value * 0.8)
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

BenchCaddy writes results to `./benchcaddy.db` relative to the directory where you run the example. Stored raw samples power richer analysis during inspection, including bootstrap confidence intervals, outlier diagnostics, noise warnings, and regression classification.

The full runnable example lives at [`examples/benchmark_nonlinear_transform.py`](https://github.com/MatthiasLen/BenchCaddy/blob/main/examples/benchmark_nonlinear_transform.py).

## Core Concepts

### `Sweep`

`Sweep(...)` is the main entry point for benchmark execution.

Common options:

- `samples`: measured samples per configuration
- `warmup_iterations`: warmup runs before sampling
- `database_path`: store results in a specific SQLite file instead of `./benchcaddy.db`
- `lock_cpu_affinity`: preserve the current CPU affinity set before benchmarking
- `reporter`: custom reporter implementing the `SweepReporter` protocol
- `verbose=True`: use the built-in Rich reporter during execution
- `store_target_return_value=True`: persist one supported return value per run
- `return_value_postprocessor`: convert complex return values before storage

Supported stored return values are `bool`, `int`, `float`, `str`, and 1D numeric vectors from `list`, `tuple`, or NumPy arrays.

### `observe(...)`

The public `observe(...)` decorator records isolated observations:

- `@observe("time")`: record call duration
- `@observe("return")`: record a normalized return value when supported
- `@observe("time", "return")`: record both

Observation labels come from the decorated function name or qualname.

### Benchmark target contract

`Sweep` executes targets in a fresh worker process. Your target must therefore be importable by the child process: use a module-level function, static method, or class method.

Unsupported targets include lambdas, nested or local functions, bound instance methods, arbitrary callable instances, and script-path targets.

BenchCaddy measures synchronous completion from its point of view. If your workload schedules asynchronous device or background work, the benchmarked function must wait for completion before returning.

## CLI Workflow

### Run a sweep from the CLI

If the target is importable, you can launch a sweep without writing a separate driver script.

```bash
benchcaddy sweep examples.benchmark_nonlinear_transform:benchmark_case \
    --suite-name nonlinear-transform \
    --param 'size=[512, 1024]' \
    --param 'variant=["baseline", "stabilized"]' \
    --samples 20 \
    --warmup-iterations 5 \
    --store-target-return-value \
    --verbose
```

Use repeated `--param` flags for parameter grids. Each flag accepts either a JSON array such as `size=[512, 2048]` or a compact scalar list such as `variant=baseline,stabilized`.

For machine-readable output:

```bash
benchcaddy sweep examples.benchmark_nonlinear_transform:benchmark_case \
    --suite-name nonlinear-transform \
    --param 'size=[512, 1024]' \
    --param 'variant=["baseline", "stabilized"]' \
    --json
```

`--json` and `--verbose` cannot be combined.

### Inspect recorded results

List recorded suites:

```bash
benchcaddy list
```

Show recent runs across the database:

```bash
benchcaddy show
benchcaddy show --numitems 10
```

Show runs for a suite:

```bash
benchcaddy show nonlinear-transform
benchcaddy show nonlinear-transform --numitems 5
```

Show one or more specific runs:

```bash
benchcaddy show 12
benchcaddy show 2.3
benchcaddy show 4 2.3 1.2
```

Composite run IDs use `SWEEP_ID.RUN_INDEX`, so `2.3` means the third run in the second recorded sweep.

### Compare, baseline, and trend

Compare configurations within a suite by median runtime:

```bash
benchcaddy compare nonlinear-transform
benchcaddy compare nonlinear-transform 2.4
```

Restrict a suite comparison to runs matching selected configuration keys from the reference run:

```bash
benchcaddy compare nonlinear-transform 2.4 --strict size
benchcaddy compare nonlinear-transform 2.4 --strict size variant
```

Compare two specific runs directly:

```bash
benchcaddy compare 12 15
benchcaddy compare 2.3 3
```

Pin a suite baseline and reuse it later:

```bash
benchcaddy baseline nonlinear-transform --pin 2.4 --note "post-optimization"
benchcaddy baseline nonlinear-transform
benchcaddy compare nonlinear-transform --baseline
benchcaddy trend nonlinear-transform --baseline
```

Trend a suite or a specific configuration over time:

```bash
benchcaddy trend nonlinear-transform
benchcaddy trend nonlinear-transform 2.4
benchcaddy trend nonlinear-transform --limit 8 --window 4
```

Direct run comparisons include return-value validation when values were stored:

- numbers: relative error percentage
- 1D numeric vectors: relative error percentage based on Euclidean distance
- strings and booleans: equality (`equal` or `different`)

### Check environment stability

Inspect current machine reliability signals before recording or comparing runs:

```bash
benchcaddy env
benchcaddy env --json
```

`env` reports timing noise, drift, affinity, CPU load, battery state, thermal throttling, and frequency stability signals when available.

<img src="https://raw.githubusercontent.com/MatthiasLen/BenchCaddy/main/bc_environment.png" alt="BenchCaddy environment check" width="640"></img>

### JSON envelope for agents and MCP consumers

All top-level CLI commands support `-j` / `--json`: `env`, `baseline`, `compare`, `list`, `show`, `sweep`, and `trend`.

Each JSON response uses the same top-level envelope:

```json
{
    "schema_version": "1.0",
    "command": "compare",
    "status": "pass|fail|inconclusive",
    "reason": "machine_readable_reason",
    "error_code": null,
    "suggested_action": "next step for the caller",
    "confidence": "high|medium|low|null",
    "exit_code": 0,
    "result": {
        "...": "command-specific payload"
    }
}
```

Use `status` as the primary control signal:

- `pass`: the command completed and the result is actionable as-is
- `fail`: the command found a blocking problem, regression, or invalid request
- `inconclusive`: the command completed, but the result needs more data, a narrower scope, or a cleaner environment before automation should treat it as decisive

`reason` is a stable snake_case classifier that adds more detail without replacing `status`. Typical reasons include `runs_recorded`, `suite_details_available`, `regression_detected`, `noisy_samples`, `environment_warnings_detected`, and `suite_not_found`.

For automation, branch on `status` first, then use `reason`, `error_code`, and `suggested_action` to decide the next step. Treat `result` as the command-specific data payload and keep callers tolerant of additional keys in future schema versions.

## BenchCaddy MCP

BenchCaddy also ships an MCP server so coding agents can inspect benchmark data through tools instead of shelling out to `benchcaddy ... --json`.
For setup, client examples, tool overview, and a sample transcript, see [README_MCP.md](README_MCP.md).

## CI And Automation

Use `compare --json` for machine-readable output:

```bash
benchcaddy compare nonlinear-transform --json
benchcaddy compare nonlinear-transform 2.4 --json
benchcaddy compare 2.3 3 --json
benchcaddy trend nonlinear-transform --json
```

Use `--fail-if-regression PERCENT` to turn the existing regression classification into a CI gate.

```bash
benchcaddy compare nonlinear-transform --baseline --fail-if-regression 5%
benchcaddy compare 2.3 3 --json --fail-if-regression 5
```

Exit codes for gated compares:

- `0`: comparison completed and the regression gate passed
- `1`: requested suite or run could not be resolved
- `2`: CLI usage error
- `3`: comparison completed and the regression gate failed

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

For more detail in inspection output, add `--verbose`:

```bash
benchcaddy --verbose show nonlinear-transform
benchcaddy --verbose compare nonlinear-transform
benchcaddy --verbose trend nonlinear-transform
```

## How To Read The Output

- `Mean +- Std (s)`: arithmetic mean and sample standard deviation across benchmark samples
- suite comparisons are ranked by median runtime, not by the mean column
- `Best Median (s)`, `Delta vs Best`, and direct-run median deltas use median runtime
- `Median CI (s)`: bootstrap confidence interval around the median runtime
- `MAD (s)`: median absolute deviation, a robust spread estimate
- `CV`: coefficient of variation (`std / mean`), used as one noise signal
- `Warnings`: low sample counts, wide confidence intervals, high variance, and detected outliers

These signals are heuristics, not proof. Treat `regressing` as a prompt to investigate and `noisy` as a sign to collect more samples or stabilize the environment.

## Recorded Environment Metadata

Each recorded run stores environment details alongside timing data, including:

- Python version and operating system string
- CPU model and total system memory
- GPU model when detectable
- Git branch, commit hash, and dirty state when run inside a Git repository
- process metadata such as PID, priority, affinity, and RSS memory

## Feedback

BenchCaddy is intentionally lean. If a workflow is missing, open an issue with the benchmark shape you are trying to support. The goal is to make performance tracking less chaotic, not to create another excuse for `results_new_final_fixed.csv`.
