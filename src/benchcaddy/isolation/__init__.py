"""benchcaddy.isolation — practical benchmark environment awareness.

This package provides lightweight, portable utilities for detecting
unreliable benchmark environments.  It deliberately avoids complex
isolation mechanisms (cgroups, NUMA, turbo-boost control) in favour of
simple observation and clear reporting.

Public API
----------
.. autosummary::

   set_affinity
   get_affinity
   EnvironmentState
   collect_environment_state
   NoiseEstimate
   estimate_noise
   ReliabilityReport
   build_reliability_report
   run_isolated
"""

from .affinity import get_affinity, set_affinity
from .environment import EnvironmentState, collect_environment_state
from .noise import NoiseEstimate, estimate_noise
from .process import run_isolated
from .report import ReliabilityReport, build_reliability_report

__all__ = [
    "EnvironmentState",
    "NoiseEstimate",
    "ReliabilityReport",
    "build_reliability_report",
    "collect_environment_state",
    "estimate_noise",
    "get_affinity",
    "run_isolated",
    "set_affinity",
]
