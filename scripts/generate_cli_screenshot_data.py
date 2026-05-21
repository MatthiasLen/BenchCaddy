from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median, stdev

from benchcaddy.db import create_sweep_execution, record_benchmark_run, set_suite_baseline
from benchcaddy.db._sqlite.models import BenchmarkRun, BenchmarkSuiteBaselineEvent, BenchmarkSweepExecution
from benchcaddy.db._sqlite.session import db_session


@dataclass(frozen=True)
class SeriesSpec:
    configuration: dict[str, object]
    medians: tuple[float, ...]
    spread: float


SUITE_NAME = "cli-trend-showcase"
TARGET_NAME = "synthetic_inference_step"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "examples" / "generated" / "cli_screenshot_trends.db"
SERIES: tuple[SeriesSpec, ...] = (
    SeriesSpec(
        configuration={"model": "tiny", "batch": 1},
        medians=(0.0800, 0.0795, 0.0805, 0.0820, 0.0830, 0.0845, 0.0855, 0.0870, 0.0885, 0.0900, 0.0910, 0.0925, 0.0940, 0.0955, 0.0970, 0.0985),
        spread=0.0008,
    ),
    SeriesSpec(
        configuration={"model": "tiny", "batch": 8},
        medians=(0.1200, 0.1190, 0.1215, 0.1240, 0.1260, 0.1300, 0.1330, 0.1370, 0.1415, 0.1450, 0.1495, 0.1550, 0.1590, 0.1645, 0.1700, 0.1765),
        spread=0.0011,
    ),
    SeriesSpec(
        configuration={"model": "base", "batch": 1},
        medians=(0.1450, 0.1435, 0.1420, 0.1410, 0.1395, 0.1380, 0.1360, 0.1345, 0.1325, 0.1310, 0.1295, 0.1280, 0.1265, 0.1250, 0.1235, 0.1225),
        spread=0.0012,
    ),
    SeriesSpec(
        configuration={"model": "base", "batch": 8},
        medians=(0.2050, 0.2040, 0.2030, 0.2050, 0.2040, 0.2060, 0.2035, 0.2055, 0.2045, 0.2065, 0.2050, 0.2070, 0.2185, 0.2220, 0.2265, 0.2310),
        spread=0.0010,
    ),
)


def _build_samples(center: float, spread: float, *, sweep_index: int, run_index: int) -> list[float]:
    offsets = [-1.45, -1.00, -0.62, -0.28, 0.0, 0.24, 0.58, 0.95, 1.38]
    samples: list[float] = []

    for sample_index, offset in enumerate(offsets):
        if offset == 0.0:
            samples.append(round(center, 6))
            continue

        wave = math.sin((sweep_index + 1) * 0.85 + (run_index + 1) * 0.55 + sample_index * 0.9)
        ripple = math.cos((sweep_index + 1) * 0.4 + sample_index * 0.7)
        jitter = (wave * 0.10 + ripple * 0.05) * spread
        samples.append(round(center + (offset * spread) + jitter, 6))

    return samples


def _environment_payload(sweep_number: int) -> dict[str, object]:
    return {
        "python_version": "3.12.4",
        "operating_system": "Ubuntu 24.04",
        "cpu_model": "AMD Ryzen 9 7950X",
        "total_memory_bytes": 68719476736,
        "gpu_model": "NVIDIA RTX 4090",
        "git": {
            "branch": "main",
            "commit_hash": f"demo{sweep_number:04x}",
            "dirty": False,
        },
        "process": {
            "pid": 4100 + sweep_number,
            "priority": 0,
            "affinity": [0, 1, 2, 3],
            "rss_bytes": 73400320 + (sweep_number * 1048576),
        },
    }


def _set_sweep_timestamp(database_path: Path, sweep_id: int, run_ids: list[int], when: datetime) -> None:
    with db_session(database_path) as session, session.begin():
        sweep = session.get(BenchmarkSweepExecution, sweep_id)
        if sweep is not None:
            sweep.created_at = when

        for offset_minutes, run_id in enumerate(run_ids):
            run = session.get(BenchmarkRun, run_id)
            if run is not None:
                run.created_at = when + timedelta(minutes=offset_minutes)


def _set_baseline_timestamp(database_path: Path, when: datetime) -> None:
    with db_session(database_path) as session, session.begin():
        event = session.query(BenchmarkSuiteBaselineEvent).order_by(BenchmarkSuiteBaselineEvent.id.desc()).first()
        if event is not None:
            event.created_at = when


def generate_database(database_path: Path) -> Path:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()

    start = datetime(2026, 4, 3, 9, 30, tzinfo=timezone.utc)

    for sweep_index in range(len(SERIES[0].medians)):
        sweep_timestamp = start + timedelta(days=sweep_index * 4)
        sweep = create_sweep_execution(
            suite_name=SUITE_NAME,
            target_name=TARGET_NAME,
            database_path=database_path,
        )
        run_ids: list[int] = []

        for run_index, spec in enumerate(SERIES, start=1):
            samples = _build_samples(spec.medians[sweep_index], spec.spread, sweep_index=sweep_index, run_index=run_index)
            recorded = record_benchmark_run(
                suite_name=SUITE_NAME,
                target_name=TARGET_NAME,
                configuration=spec.configuration,
                samples=samples,
                observations=[],
                median_seconds=float(median(samples)),
                min_seconds=min(samples),
                max_seconds=max(samples),
                std_seconds=float(stdev(samples)),
                environment=_environment_payload(sweep_index + 1),
                sweep_execution_id=sweep.id,
                run_index=run_index,
                database_path=database_path,
            )
            run_ids.append(recorded.id)

        _set_sweep_timestamp(database_path, sweep.id, run_ids, sweep_timestamp)

    set_suite_baseline(SUITE_NAME, 1, database_path, note="release-0.1 reference")
    _set_baseline_timestamp(database_path, start + timedelta(hours=2))
    return database_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic BenchCaddy trend data for CLI screenshots.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to the generated SQLite database (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    database_path = generate_database(args.output.resolve())
    print(f"Generated synthetic trend database at {database_path}")
    print(f"Summary view:  .\\.venv\\Scripts\\benchcaddy.exe trend {SUITE_NAME} --database {database_path}")
    print(f"Baseline view: .\\.venv\\Scripts\\benchcaddy.exe trend {SUITE_NAME} --baseline --database {database_path}")
    print(f"History view:  .\\.venv\\Scripts\\benchcaddy.exe baseline {SUITE_NAME} --database {database_path}")


if __name__ == "__main__":
    main()