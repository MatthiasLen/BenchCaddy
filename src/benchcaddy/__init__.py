"""Public package surface for BenchCaddy.

This module should encapsulate the stable, user-facing imports that make
up the primary BenchCaddy API. Re-exports that belong in the default
package namespace should stay here, while internal helpers should remain
in their implementation modules.
"""

from .core import Sweep
from .isolation import (
    EnvironmentState,
    NoiseAnalyzer,
    NoiseCapture,
    NoiseEstimate,
    ProcessState,
    ReliabilityReport,
    build_reliability_report,
    collect_environment_state,
    collect_process_state,
    get_affinity,
    prepare_system,
    run_isolated,
)
from .observability import observe
from .reporting import RichSweepReporter, SweepReporter

__all__ = [
    "EnvironmentState",
    "NoiseAnalyzer",
    "NoiseCapture",
    "NoiseEstimate",
    "ProcessState",
    "ReliabilityReport",
    "RichSweepReporter",
    "Sweep",
    "SweepReporter",
    "build_reliability_report",
    "collect_environment_state",
    "collect_process_state",
    "get_affinity",
    "observe",
    "prepare_system",
    "run_isolated",
]
