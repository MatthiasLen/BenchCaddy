from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchcaddy import Sweep, observe


@observe("time")
def benchmark_case(iterations: int, variant: str) -> float:
    if variant == "leibniz":
        total = 0.0
        sign = 1.0
        denominator = 1.0

        for _ in range(iterations):
            total += sign * (4.0 / denominator)
            sign = -sign
            denominator += 2.0
        return total

    if variant == "nilakantha":
        total = 3.0
        sign = 1.0
        denominator = 2.0

        for _ in range(iterations):
            total += sign * (
                4.0 / (denominator * (denominator + 1.0) * (denominator + 2.0))
            )
            sign = -sign
            denominator += 2.0
        return total

    raise ValueError(f"Unsupported variant: {variant}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark two pi approximation strategies with BenchCaddy.",
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
        default=20,
        help="Number of timed samples per configuration.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=2,
        help="Number of warmup iterations per configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep = Sweep(
        target=benchmark_case,
        params={
            "iterations": [25_000, 100_000],
            "variant": ["leibniz", "nilakantha"],
        },
        suite_name="pi-approximations",
        samples=args.samples,
        warmup_iterations=args.warmup_iterations,
        database_path=args.database,
        store_target_return_value=True,
        verbose=args.verbose,
    )
    sweep.run()


if __name__ == "__main__":
    main()
