from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchcaddy import Sweep


def bool_target(size: int) -> bool:
    return size % 2 == 0


def float_vector_target(size: int) -> list[float]:
    return [float(size), float(size) / 2.0, float(size) / 4.0]


def bool_vector_target(size: int) -> list[bool]:
    return [size % 2 == 0, size > 2, size < 10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compact BenchCaddy return-value examples.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    Sweep(
        target=bool_target,
        params={"size": [2, 3]},
        suite_name="return-type-bool",
        samples=10,
        warmup_iterations=2,
        database_path=args.database,
        store_target_return_value=True,
        verbose=args.verbose,
    ).run()

    Sweep(
        target=float_vector_target,
        params={"size": [2, 4]},
        suite_name="return-type-float-vector",
        samples=10,
        warmup_iterations=2,
        database_path=args.database,
        store_target_return_value=True,
        verbose=args.verbose,
    ).run()

    Sweep(
        target=bool_vector_target,
        params={"size": [2]},
        suite_name="return-type-bool-vector",
        samples=10,
        warmup_iterations=2,
        database_path=args.database,
        store_target_return_value=True,
        verbose=args.verbose,
    ).run()


if __name__ == "__main__":
    main()
