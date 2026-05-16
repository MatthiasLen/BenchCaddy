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
import traceback
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import psutil

_MAX_UNIX_PRIORITY = -20
_PROCESS_JOIN_GRACE_SECONDS = 0.2


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


def set_affinity(cpus: list[int]) -> bool:
    """Pin the current process to *cpus* when the platform supports it."""
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
    """Return the CPU cores the current process may run on, if exposed."""
    process = psutil.Process()
    if not hasattr(process, "cpu_affinity"):
        return None

    try:
        return list(process.cpu_affinity())
    except (psutil.AccessDenied, NotImplementedError, AttributeError, OSError):
        return None


def collect_process_state() -> ProcessState:
    """Collect a lightweight snapshot of the current Python process."""
    process = psutil.Process()

    try:
        affinity = list(process.cpu_affinity())
    except (psutil.AccessDenied, NotImplementedError, AttributeError):
        affinity = []

    try:
        priority = process.nice()
    except (psutil.AccessDenied, AttributeError):
        priority = None

    try:
        rss_bytes = process.memory_info().rss
    except (psutil.AccessDenied, AttributeError, OSError):
        rss_bytes = None

    return ProcessState(
        pid=process.pid,
        priority=priority,
        affinity=affinity,
        rss_bytes=rss_bytes,
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
        conn.send((True, fn(*args, **kwargs)))
    except Exception as exc:
        with suppress(BrokenPipeError, EOFError, OSError):
            conn.send(
                (
                    False,
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
    finally:
        conn.close()


def _join_process(process: multiprocessing.Process, timeout: float) -> bool:
    """Wait briefly for *process* to exit and report whether it stopped."""
    process.join(timeout)
    return process.exitcode is not None


def _stop_process(process: multiprocessing.Process, timeout: float) -> None:
    """Terminate a child process without blocking indefinitely."""
    if process.exitcode is not None:
        return

    process.terminate()
    if _join_process(process, timeout):
        return

    if hasattr(process, "kill"):
        process.kill()
        _join_process(process, timeout)


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
        subprocess created with the ``spawn`` start method so that GC
        state, import side effects, and JIT warm-up from the host
        process cannot influence timings. This requires *fn*, *args*,
        and *kwargs* to be picklable, and the calling module must be
        safe to import under ``spawn``.
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

    context = multiprocessing.get_context("spawn")
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(fn, args, kw, child_conn, disable_gc),
        daemon=True,
    )
    process.start()
    child_conn.close()

    try:
        if parent_conn.poll(timeout):
            try:
                success, payload = parent_conn.recv()
            except EOFError as exc:
                raise RuntimeError("Isolated process exited before sending a result.") from exc
        else:
            _stop_process(process, _PROCESS_JOIN_GRACE_SECONDS)
            raise TimeoutError(f"run_isolated timed out after {timeout}s")
    finally:
        parent_conn.close()

    if not _join_process(process, _PROCESS_JOIN_GRACE_SECONDS):
        _stop_process(process, _PROCESS_JOIN_GRACE_SECONDS)

    if not success:
        if isinstance(payload, dict):
            exception_type = payload.get("type", "Exception")
            exception_message = payload.get("message", "")
            child_traceback = payload.get("traceback", "")
            details = f"{exception_type}: {exception_message}" if exception_message else exception_type
            message = f"Isolated process raised an exception: {details}"
            if child_traceback:
                message = f"{message}\n{child_traceback.rstrip()}"
            raise RuntimeError(message)
        raise RuntimeError(f"Isolated process raised an exception: {payload}")

    return payload
