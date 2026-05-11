"""CPU affinity management.

Pin the current process to specific CPU cores when supported, and
detect the current affinity set.  Both operations fail gracefully on
platforms that do not expose the API (e.g. macOS, Windows without
admin, some container environments).
"""

from __future__ import annotations

import psutil


def set_affinity(cpus: list[int]) -> bool:
    """Pin the current process to *cpus*.

    Returns ``True`` on success, ``False`` when the platform does not
    support CPU affinity or the caller lacks the required permissions.
    """
    if not cpus:
        return False

    process = psutil.Process()
    if not hasattr(process, "cpu_affinity"):
        return False

    try:
        process.cpu_affinity(cpus)
        return True
    except (psutil.AccessDenied, NotImplementedError, ValueError, OSError):
        return False


def get_affinity() -> list[int] | None:
    """Return the CPU cores the current process is allowed to run on.

    Returns ``None`` when the platform does not expose affinity
    information.
    """
    process = psutil.Process()
    if not hasattr(process, "cpu_affinity"):
        return None

    try:
        return list(process.cpu_affinity())
    except (psutil.AccessDenied, NotImplementedError, AttributeError, OSError):
        return None
