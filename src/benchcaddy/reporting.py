from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from rich.console import Console

from .presentation import dump_json, format_return_value, json_panel, render_table, summary_panel

if TYPE_CHECKING:
    from .core import BenchmarkResult


class SweepReporter(Protocol):
    """Defines the progress callback contract emitted during a benchmark sweep."""

    def on_sweep_started(
        self,
        *,
        suite_name: str,
        total_configurations: int,
        samples: int,
        warmup_iterations: int,
        database_path: Path,
    ) -> None: ...

    def on_configuration_started(
        self,
        *,
        index: int,
        total: int,
        configuration: dict[str, object],
    ) -> None: ...

    def on_sample_completed(
        self,
        *,
        sample_index: int,
        sample_total: int,
        elapsed_seconds: float,
        observation_count: int,
    ) -> None: ...

    def on_configuration_completed(
        self,
        *,
        index: int,
        total: int,
        configuration: dict[str, object],
        median_seconds: float,
        min_seconds: float,
        max_seconds: float,
        std_seconds: float,
        sample_count: int,
        target_return_value: object | None,
    ) -> None: ...

    def on_sweep_completed(self, *, results: Sequence[BenchmarkResult]) -> None: ...


class RichSweepReporter:
    """Renders sweep progress and summary events to a Rich console."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def on_sweep_started(
        self,
        *,
        suite_name: str,
        total_configurations: int,
        samples: int,
        warmup_iterations: int,
        database_path: Path,
    ) -> None:
        self.console.print(
            summary_panel(
                "BenchCaddy Run",
                [
                    ("Suite", suite_name),
                    ("Configurations", str(total_configurations)),
                    ("Samples per configuration", str(samples)),
                    ("Warmup iterations", str(warmup_iterations)),
                    ("Database", str(database_path)),
                ],
            )
        )

    def on_configuration_started(
        self,
        *,
        index: int,
        total: int,
        configuration: dict[str, object],
    ) -> None:
        self.console.print(json_panel(f"Configuration {index}/{total}", configuration, fit=True))

    def on_sample_completed(
        self,
        *,
        sample_index: int,
        sample_total: int,
        elapsed_seconds: float,
        observation_count: int,
    ) -> None:
        self.console.print(f"Sample {sample_index}/{sample_total}: {elapsed_seconds:.6f}s | observations={observation_count}")

    def on_configuration_completed(
        self,
        *,
        index: int,
        total: int,
        configuration: dict[str, object],
        median_seconds: float,
        min_seconds: float,
        max_seconds: float,
        std_seconds: float,
        sample_count: int,
        target_return_value: object | None,
    ) -> None:
        self.console.print(
            summary_panel(
                f"Completed {index}/{total}",
                [
                    ("Configuration", dump_json(configuration)),
                    ("Samples", str(sample_count)),
                    ("Median", f"{median_seconds:.6f}s"),
                    ("Min / Max", f"{min_seconds:.6f}s / {max_seconds:.6f}s"),
                    ("Std Dev", f"{std_seconds:.6f}s"),
                    ("Return Value", format_return_value(target_return_value)),
                ],
            )
        )

    def on_sweep_completed(self, *, results: Sequence[BenchmarkResult]) -> None:
        self.console.print(
            render_table(
                "BenchCaddy Summary",
                [
                    "Run ID",
                    ("Record ID", "right"),
                    "Configuration",
                    ("Median (s)", "right"),
                    ("Min (s)", "right"),
                    ("Max (s)", "right"),
                    ("Std (s)", "right"),
                    "Return Value",
                ],
                [
                    (
                        result.run_id,
                        result.record_id,
                        dump_json(result.configuration),
                        f"{result.median_seconds:.6f}",
                        f"{result.min_seconds:.6f}",
                        f"{result.max_seconds:.6f}",
                        f"{result.std_seconds:.6f}",
                        format_return_value(result.target_return_value),
                    )
                    for result in results
                ],
            )
        )
