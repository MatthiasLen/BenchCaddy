"""Benchmark isolation and environment-reliability utilities.

This package should encapsulate the optional controls and diagnostics
that help BenchCaddy run in a cleaner measurement environment. Affinity
management, lightweight environment inspection, noise estimation,
reliability reporting, and subprocess isolation belong here as a cohesive
support layer around the core sweep engine.
"""

from .environment import EnvironmentState, collect_environment_state
from .noise import NoiseAnalyzer, NoiseCapture, NoiseEstimate
from .process import ProcessState, collect_process_state, get_affinity, prepare_system, run_isolated
from .report import ReliabilityReport, build_reliability_report

__all__ = [
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
    "prepare_system",
    "run_isolated",
]
