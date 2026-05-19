"""Benchmark isolation and environment-reliability utilities.

This package should encapsulate the optional controls and diagnostics
that help BenchCaddy run in a cleaner measurement environment. Affinity
management, lightweight environment inspection, noise estimation,
reliability reporting, and subprocess isolation belong here as a cohesive
support layer around the core sweep engine.
"""

from .environment import EnvironmentState as EnvironmentState
from .environment import collect_environment_state as collect_environment_state
from .noise import NoiseAnalyzer as NoiseAnalyzer
from .noise import NoiseCapture as NoiseCapture
from .noise import NoiseEstimate as NoiseEstimate
from .observability import IsolatedRunResult, ObservationCollector, collect_observations
from .observability import observe as observe
from .process import (
    ProcessState as ProcessState,
)
from .process import (
    collect_process_state as collect_process_state,
)
from .process import (
    get_affinity as get_affinity,
)
from .process import (
    prepare_system as prepare_system,
)
from .process import (
    run_isolated as run_isolated,
)
from .process import (
    validate_isolated_target,
)
from .report import ReliabilityReport as ReliabilityReport
from .report import build_reliability_report as build_reliability_report

_CORE_EXPORTS = (
    "EnvironmentState",
    "NoiseAnalyzer",
    "NoiseCapture",
    "NoiseEstimate",
    "ProcessState",
    "ReliabilityReport",
    "build_reliability_report",
    "collect_environment_state",
    "collect_process_state",
    "get_affinity",
    "observe",
    "prepare_system",
    "run_isolated",
)

__all__ = [
    "IsolatedRunResult",
    "ObservationCollector",
    "collect_observations",
    "validate_isolated_target",
    *_CORE_EXPORTS,
]
