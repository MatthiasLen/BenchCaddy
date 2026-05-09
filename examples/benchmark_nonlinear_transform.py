from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a self-contained BenchCaddy benchmark example.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Optional path to the benchmark SQLite database.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show structured progress output during the benchmark run.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of timed samples per configuration.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=1,
        help="Number of warmup iterations per configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep = Sweep(
        target=benchmark_case,
        params={
            "size": [512, 2048],
            "variant": ["baseline", "stabilized"],
        },
        suite_name="nonlinear-transform",
        samples=args.samples,
        warmup_iterations=args.warmup_iterations,
        database_path=args.database,
        verbose=args.verbose,
    )
    sweep.run()


if __name__ == "__main__":
    main()