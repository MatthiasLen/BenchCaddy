<img src="./benchcaddy_logo.png" alt="BenchCaddy logo" width="240"></img>

We all tell ourselves we’re going to use Scalene,PyInstrument or TorchProfile - tools that produce traces so complex and beautiful they belong in a modern art gallery. But let’s be real: most days, "benchmarking" is just us sprinkling time.time() across our code like frantic seasoning on a failing dish. You’re staring at the terminal, trying to remember if the last run was actually faster or if you just happen to be in a better mood, only to realize you’ve already lost the thread. *"Wait, when did I change the naming convention of the log files? Is 'results_v2_final' newer than 'results_new_test'?"*


**BenchCaddy** is the humble sidekick for those of us living in that chaotic middle ground. It replaces "vibes-based" timing with stabilized sweeps and environment metadata, tucking everything into a neat database before your brain can wander. It won’t map your entire soul, but it will save you from your own memory and provide a summary clean enough to make you look like the organized professional your friends think you are. No traces to decipher, no lost logs, and no more gaslighting yourself - just actual proof your code is getting faster.

# Something missing ?

BenchCaddy is intentionally lean - a sidekick, not a supervisor. I built it to curb my own "log-file-chaos," but I’m curious how you manage yours. If you’ve got a feature idea, a bug that’s getting on your nerves, or a suggestion for an export format that actually belongs in this decade, open an issue. I’m not trying to build a bloated enterprise behemoth; I just want this to be the best way to track performance without ever having to name a file timings_final_v4_fixed_REALLY.log again.


## Quick start

BenchCaddy is designed around two steps:

1. Run a benchmark sweep over one or more configurations.
2. Inspect or compare the recorded results from the SQLite database.

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


@observe("nonlinear_iteration")
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
`benchcaddy.db` in the current working directory.

The full runnable example lives in the repository and source distribution at
[`examples/benchmark_nonlinear_transform.py`](https://github.com/MatthiasLen/BenchCaddy/blob/main/examples/benchmark_nonlinear_transform.py)
and supports `--verbose`, `--database`, `--samples`, and `--warmup-iterations`.

`Sweep` also accepts a script path as the target. In that mode, parameter keys
are mapped to CLI flags such as `size -> --size` and `warmup_runs` / `iterations`
can be used as aliases for `warmup_iterations` / `samples`.

## Sweep options

The main public `Sweep(...)` options are:

- `samples`: number of measured samples per configuration
- `iterations`: alias for `samples`
- `warmup_iterations`: warmup runs before sampling begins
- `warmup_runs`: alias for `warmup_iterations`
- `database_path`: store results in a specific SQLite file instead of `./benchcaddy.db`
- `lock_cpu_affinity`: preserve the current CPU affinity set before benchmarking
- `sync`: callable used to synchronize async device work after each invocation
- `reporter`: custom reporter implementing the `SweepReporter` protocol
- `verbose=True`: use the built-in Rich reporter during execution

## Script targets

You can benchmark a standalone script instead of a Python callable:

```python
from benchcaddy import Sweep


Sweep(
    target="./train_step.py",
    params={
        "size": [512, 2048],
        "variant": ["baseline", "stabilized"],
        "use_cache": [True, False],
    },
    suite_name="train-step",
    samples=5,
).run()
```

BenchCaddy converts configuration keys to CLI flags:

- `size=512` becomes `--size 512`
- `use_cache=True` becomes `--use-cache`
- `use_cache=False` becomes `--use-cache false`

That mode works best with scripts that parse explicit values for non-presence
flags and exit with status code `0` on success.

## Inspect results

List all recorded suites:

```bash
benchcaddy list
```

`list` also shows the observation labels seen across runs in each suite.

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

Compare configurations within a suite by median runtime:

```bash
benchcaddy compare nonlinear-transform
```

Compare a suite against a selected recorded run instead of the best run:

```bash
benchcaddy compare nonlinear-transform 2.4
```

Restrict a suite comparison to runs that match selected configuration keys from
the reference run:

```bash
benchcaddy compare nonlinear-transform 2.4 --strict size
benchcaddy compare nonlinear-transform 2.4 --strict size variant
benchcaddy compare nonlinear-transform 2.4 --strict variant
```

Compare two specific runs directly. Improvements greater than 5% are shown in
green and regressions greater than 5% are shown in red:

```bash
benchcaddy compare 12 15
benchcaddy compare 2.3 3
```

For more detail in the inspection output, add `--verbose`:

```bash
benchcaddy --verbose show nonlinear-transform
benchcaddy --verbose compare nonlinear-transform
```

## How to read the output

- `Mean +- Std (s)` is the arithmetic mean and sample standard deviation across benchmark samples
- suite comparisons are ranked by median runtime, not by the mean column
- `Best Median (s)`, `Delta vs Best`, and direct-run `Median Delta` / `Median Percent Change` all use median runtime
- observation tables report per-label timing aggregated across samples
- `Total (s)` in observation tables is the sum across all samples for that label

## Environment metadata

Every recorded run stores environment details alongside the timing data, including:

- Python version and operating system string
- CPU model and total system memory
- GPU model when it can be detected
- Git branch, commit hash, and dirty state when run inside a Git repository
- process metadata such as PID, priority, affinity, and RSS memory

## What comparisons can I do?

BenchCaddy currently compares runs within a suite. This works best when the
suite name represents one benchmark target and the parameters represent the
variants you want to evaluate.

For each recorded run, `benchcaddy compare` shows:

- the configuration that was executed
- the median runtime across samples
- the absolute delta versus the fastest recorded run
- the slowdown factor relative to the fastest recorded run

That gives you a practical baseline for answering questions like:

- Is the stabilized implementation faster than the baseline?
- How does runtime scale with input size?
- Did a new run outperform the previous best configuration?
