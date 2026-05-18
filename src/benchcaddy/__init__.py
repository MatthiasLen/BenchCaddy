"""Public package surface for BenchCaddy.

This module should encapsulate the stable, user-facing imports that make
up the primary BenchCaddy API. Re-exports that belong in the default
package namespace should stay here, while internal helpers should remain
in their implementation modules.
"""

from .core import Sweep
from .isolation import (
    _CORE_EXPORTS as _ISOLATION_CORE_EXPORTS,
)
from .isolation import (
    EnvironmentState as EnvironmentState,
)
from .isolation import (
    NoiseAnalyzer as NoiseAnalyzer,
)
from .isolation import (
    NoiseCapture as NoiseCapture,
)
from .isolation import (
    NoiseEstimate as NoiseEstimate,
)
from .isolation import (
    ProcessState as ProcessState,
)
from .isolation import (
    ReliabilityReport as ReliabilityReport,
)
from .isolation import (
    build_reliability_report as build_reliability_report,
)
from .isolation import (
    collect_environment_state as collect_environment_state,
)
from .isolation import (
    collect_process_state as collect_process_state,
)
from .isolation import (
    get_affinity as get_affinity,
)
from .isolation import (
    observe as observe,
)
from .isolation import (
    prepare_system as prepare_system,
)
from .isolation import (
    run_isolated as run_isolated,
)
from .reporting import RichSweepReporter, SweepReporter

__all__ = [
    "RichSweepReporter",
    "Sweep",
    "SweepReporter",
    *_ISOLATION_CORE_EXPORTS,
]
