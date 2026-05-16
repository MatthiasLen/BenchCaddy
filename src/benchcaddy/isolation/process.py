"""Process-level isolation helpers.

This module should encapsulate process-scoped utilities used by the
isolation layer, including subprocess execution and snapshots of the
current process state. Process spawning, timeout handling, child result
propagation, and process-level introspection belong here.
"""

from __future__ import annotations

import gc
import multiprocessing
import multiprocessing.connection
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psutil

_MAX_UNIX_PRIORITY = -20


@dataclass
class ProcessState:
    """Snapshot of process-local settings relevant to benchmarking."""

    pid: int
    """Operating-system process identifier."""

    priority: int | str | None
    """Current scheduling priority, or ``None`` when unavailable."""

    affinity: list[int]
    """CPU cores the process may run on, or an empty list when unavailable."""

    rss_bytes: int | None
    """Resident set size in bytes, or ``None`` when unavailable."""


def _read_process_affinity(process: psutil.Process) -> list[int]:
    """Return the current affinity mask, falling back to an empty list."""
    try:
        return list(process.cpu_affinity())
    except (psutil.AccessDenied, NotImplementedError, AttributeError):
        return []


def _read_process_priority(process: psutil.Process) -> int | str | None:
    """Return the process priority when the platform exposes it."""
    try:
        return process.nice()
    except (psutil.AccessDenied, AttributeError):
        return None


def _read_process_rss(process: psutil.Process) -> int | None:
    """Return the process resident set size in bytes."""
    try:
        return process.memory_info().rss
    except (psutil.AccessDenied, AttributeError, OSError):
        return None


def collect_process_state() -> ProcessState:
    """Collect a lightweight snapshot of the current Python process."""
    process = psutil.Process()

    return ProcessState(
        pid=process.pid,
        priority=_read_process_priority(process),
        affinity=_read_process_affinity(process),
        rss_bytes=_read_process_rss(process),
    )


def prepare_system(lock_cpu_affinity: bool = True) -> None:
    """Raise process priority and reduce avoidable runtime interference."""
    process = psutil.Process()

    try:
        if os.name == "nt" and hasattr(psutil, "HIGH_PRIORITY_CLASS"):
            process.nice(psutil.HIGH_PRIORITY_CLASS)
        elif hasattr(os, "nice"):
            os.nice(_MAX_UNIX_PRIORITY - os.nice(0))
    except (PermissionError, psutil.AccessDenied, AttributeError, OSError):
        pass

    if lock_cpu_affinity and hasattr(process, "cpu_affinity"):
        try:
            affinity = list(process.cpu_affinity())
            if affinity:
                process.cpu_affinity(affinity)
        except (psutil.AccessDenied, NotImplementedError, ValueError):
            pass

    gc.collect()
    if hasattr(gc, "freeze"):
        gc.freeze()


def _worker(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    conn: multiprocessing.connection.Connection,
    disable_gc: bool,
) -> None:
    """Entry point executed inside the child process."""
    if disable_gc:
        gc.disable()
    else:
        gc.collect()

    try:
        result = fn(*args, **kwargs)
        conn.send((True, result))
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        conn.send((False, exc))
    finally:
        conn.close()


def run_isolated(
    fn: Callable[..., Any],
    *,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    fresh_process: bool = True,
    disable_gc: bool = False,
    timeout: float | None = None,
) -> Any:
    """Call *fn* with optional subprocess isolation.

    Parameters
    ----------
    fn:
        The callable to invoke.
    args:
        Positional arguments forwarded to *fn*.
    kwargs:
        Keyword arguments forwarded to *fn*.
    fresh_process:
        When ``True`` (default), *fn* is executed in a freshly spawned
        subprocess so that GC state, import side effects, and JIT
        warm-up from the host process cannot influence timings.
        When ``False``, *fn* is called directly in the current process.
    disable_gc:
        When ``True``, the Python garbage collector is disabled inside
        the child process before *fn* is invoked.  Has no effect when
        *fresh_process* is ``False``.
    timeout:
        Optional wall-clock timeout in seconds.  A :exc:`TimeoutError`
        is raised if the child process does not finish within this
        duration.

    Returns
    -------
    Any
        The return value of *fn*.

    Raises
    ------
    TimeoutError
        When *timeout* elapses before the child process finishes.
    RuntimeError
        When the child process raises an exception.
    """
    kw = kwargs or {}

    if not fresh_process:
        return fn(*args, **kw)

    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=_worker,
        args=(fn, args, kw, child_conn, disable_gc),
        daemon=True,
    )
    process.start()
    child_conn.close()

    try:
        if parent_conn.poll(timeout):
            success, payload = parent_conn.recv()
        else:
            process.kill()
            process.join()
            raise TimeoutError(f"run_isolated timed out after {timeout}s")
    finally:
        parent_conn.close()

    process.join()

    if not success:
        exc = payload if isinstance(payload, Exception) else RuntimeError(str(payload))
        raise RuntimeError(f"Isolated process raised an exception: {payload}") from exc

    return payload
