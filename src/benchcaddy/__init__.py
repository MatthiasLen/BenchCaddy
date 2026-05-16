"""Public package surface for BenchCaddy.

This module should encapsulate the stable, user-facing imports that make
up the primary BenchCaddy API. Re-exports that belong in the default
package namespace should stay here, while internal helpers should remain
in their implementation modules.
"""

from .core import Sweep
from .isolation import (
    EnvironmentState,
    NoiseEstimate,
    ReliabilityReport,
    build_reliability_report,
    collect_environment_state,
    get_affinity,
    run_isolated,
    set_affinity,
)
from .observability import observe
from .reporting import RichSweepReporter, SweepReporter

__all__ = [
    "EnvironmentState",
    "NoiseEstimate",
    "ReliabilityReport",
    "RichSweepReporter",
    "Sweep",
    "SweepReporter",
    "build_reliability_report",
    "collect_environment_state",
    "get_affinity",
    "observe",
    "run_isolated",
    "set_affinity",
]
