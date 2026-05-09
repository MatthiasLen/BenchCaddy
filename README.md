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

The full runnable example is in `examples/benchmark_nonlinear_transform.py` and
supports `--verbose`, `--database`, `--samples`, and `--warmup-iterations`.

`Sweep` also accepts a script path as the target. In that mode, parameter keys
are mapped to CLI flags such as `size -> --size` and `warmup_runs` / `iterations`
can be used as aliases for `warmup_iterations` / `samples`.

## Inspect results

List all recorded suites:

```bash
benchcaddy list
```

Show the recorded runs and environment for a suite:

```bash
benchcaddy show nonlinear-transform
```

Show the detailed timings for a single recorded run:

```bash
benchcaddy show 12
```

Compare configurations within a suite by median runtime:

```bash
benchcaddy compare nonlinear-transform
```

Compare two specific runs directly. Improvements greater than 5% are shown in
green and regressions greater than 5% are shown in red:

```bash
benchcaddy compare 12 15
```

For more detail in the inspection output, add `--verbose`:

```bash
benchcaddy --verbose show nonlinear-transform
benchcaddy --verbose compare nonlinear-transform
```

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
