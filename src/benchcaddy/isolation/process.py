"""Process-level isolation helpers.

This module should encapsulate process-scoped utilities used by the
isolation layer, including subprocess execution and snapshots of the
current process state. Process spawning, timeout handling, child result
propagation, and process-level introspection belong here.
"""

from __future__ import annotations

import gc
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import psutil

from .observability import IsolatedRunResult, collect_observations

_MAX_UNIX_PRIORITY = -20
_WORKER_FLAG = "--benchcaddy-worker"


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

    # On Windows, set the highest possible priority class. On Unix, set the lowest nice value.
    try:
        if os.name == "nt" and hasattr(psutil, "HIGH_PRIORITY_CLASS"):
            process.nice(psutil.HIGH_PRIORITY_CLASS)
        elif hasattr(os, "nice"):
            os.nice(_MAX_UNIX_PRIORITY - os.nice(0))
    except (PermissionError, psutil.AccessDenied, AttributeError, OSError):
        pass

    # Re-apply CPU affinity to reduce the chance of being migrated to a different set of cores after preparation. 
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


def _resolve_callable(module_name: str, qualname: str) -> Callable[..., Any]:
    """Import and resolve a callable from a module/qualname reference."""
    target: Any = importlib.import_module(module_name)
    for attribute in qualname.split("."):
        target = getattr(target, attribute)

    if not callable(target):
        raise TypeError(f"{module_name}.{qualname} does not resolve to a callable")
    return target


def _child_error_payload(exc: Exception) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _unsupported_target_error(reason: str) -> TypeError:
    return TypeError(
        "fresh isolated execution requires an importable module-level function, static method, or class method; "
        f"{reason}"
    )


def _run_observed_call(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> IsolatedRunResult:
    with collect_observations() as collector:
        start = perf_counter()
        result = fn(*args, **kwargs)
        elapsed = perf_counter() - start
    return IsolatedRunResult(
        elapsed_seconds=elapsed,
        return_value=result,
        observations=list(collector.records),
    )


def _execute_worker_request(request: dict[str, Any]) -> dict[str, Any]:
    """Execute one isolated request inside the worker process."""
    fn = _resolve_callable(request["module_name"], request["qualname"])
    args = tuple(request.get("args", ()))
    kwargs = dict(request.get("kwargs", {}))
    warmup_runs = int(request.get("warmup_runs", 0))

    # 1. Prepare the system to reduce avoidable interference, applying CPU affinity if requested.
    prepare_system(lock_cpu_affinity=bool(request.get("lock_cpu_affinity", True)))

    # 2. Optionally disable GC to avoid unpredictable collection during timing.
    if bool(request.get("disable_gc", False)):
        gc.disable()

    # 3. Optionally perform untimed warmup runs to allow JIT optimizations and other one-time effects to occur before the measured execution.
    try:
        for _ in range(warmup_runs):
            fn(*args, **kwargs)
        return {"ok": True, "payload": _run_observed_call(fn, args, kwargs)}
    except Exception as exc:
        return {"ok": False, "payload": _child_error_payload(exc)}


def _run_subprocess_worker(request: dict[str, Any], timeout: float | None) -> dict[str, Any]:
    """Run a worker subprocess to execute one isolated request and return its structured response."""

    with tempfile.TemporaryDirectory(prefix="benchcaddy-isolated-") as temp_dir:
        request_path = Path(temp_dir) / "request.pkl"
        response_path = Path(temp_dir) / "response.pkl"

        # Write the request payload to a temporary file for the worker to read.
        with request_path.open("wb") as handle:
            pickle.dump(request, handle)

        try:
            # Spawn a worker process to read the request, execute it, and write back a structured response.
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchcaddy.isolation.process",
                    _WORKER_FLAG,
                    str(request_path),
                    str(response_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"run_isolated timed out after {timeout}s") from exc

        if response_path.exists():
            with response_path.open("rb") as handle:
                return pickle.load(handle)

        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        details = stderr or stdout or f"worker exited with code {completed.returncode}"
        raise RuntimeError(f"Isolated worker failed before sending a result: {details}")


def run_isolated(
    fn: Callable[..., Any],
    *,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    fresh_process: bool = True,
    disable_gc: bool = False,
    warmup_runs: int = 0,
    lock_cpu_affinity: bool = True,
    timeout: float | None = None,
) -> IsolatedRunResult:
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
        Python subprocess so that GC state, import side effects, and
        JIT warm-up from the host process cannot influence timings.
        This requires *fn* to be an importable top-level callable and
        *args* / *kwargs* to be pickle-serializable.
        When ``False``, *fn* is called directly in the current process.
    disable_gc:
        When ``True``, the Python garbage collector is disabled inside
        the child process before *fn* is invoked.  Has no effect when
        *fresh_process* is ``False``.
    warmup_runs:
        Number of untimed throwaway invocations to perform inside the
        isolated child before the measured execution. Has no effect when
        *fresh_process* is ``False``.
    lock_cpu_affinity:
        When ``True`` (default), the child process re-applies its current
        CPU affinity mask as part of :func:`prepare_system`. Has no effect
        when *fresh_process* is ``False``.
    timeout:
        Optional wall-clock timeout in seconds.  A :exc:`TimeoutError`
        is raised if the child process does not finish within this
        duration.

    Returns
    -------
    IsolatedRunResult
        Structured result containing the measured elapsed time, the
        callable return value, and observations recorded by isolated
        decorators during the measured call.

    Raises
    ------
    TimeoutError
        When *timeout* elapses before the child process finishes.
    RuntimeError
        When the child process raises an exception or exits before it can
        send back a structured result.
    """
    kw = kwargs or {}
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")

    if not fresh_process:
        return _run_observed_call(fn, args, kw)

    if inspect.ismethod(fn) and getattr(fn, "__self__", None) is not None and not isinstance(fn.__self__, type):
        raise _unsupported_target_error(
            "bound instance methods are unsupported because the worker reconstructs call targets from module and qualname, "
            "not from a live instance. Wrap the method call in a module-level benchmark function that creates or receives the instance explicitly."
        )

    if not (inspect.isfunction(fn) or inspect.ismethod(fn) or inspect.isbuiltin(fn)):
        raise _unsupported_target_error(
            "arbitrary callable instances are unsupported because the worker cannot reconstruct a live __call__ object from module and qualname alone. "
            "Expose a module-level function, static method, or class method instead."
        )

    module_name = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)

    if not module_name or not qualname:
        raise _unsupported_target_error(
            "the target is missing module or qualname metadata, so the worker cannot re-import it in a fresh process."
        )

    if "<lambda>" in qualname:
        raise _unsupported_target_error(
            "lambdas are unsupported because they do not provide a stable import path for the worker. Define a named module-level function instead."
        )

    if "<locals>" in qualname:
        raise _unsupported_target_error(
            "nested or local functions are unsupported because they are scoped to a parent frame and cannot be re-imported by qualname in the worker. "
            "Move the benchmark target to module scope instead."
        )

    # Validate that the target is importable and callable before spawning a process, so that we can raise
    # errors synchronously instead of via the subprocess channel.
    try:
        _resolve_callable(module_name, qualname)
    except Exception as exc:
        raise _unsupported_target_error(
            f"could not resolve {module_name}.{qualname}. Ensure the symbol is importable in the child process and exposed at that module path."
        ) from exc

    # The worker process will re-import the target function and execute it with the provided arguments,
    # then send back a structured result or error payload. We use temporary files and pickle for
    # inter-process communication (IPC) to keep the implementation simple and robust across platforms.
    response = _run_subprocess_worker(
        {
            "module_name": module_name,
            "qualname": qualname,
            "args": args,
            "kwargs": kw,
            "disable_gc": disable_gc,
            "warmup_runs": warmup_runs,
            "lock_cpu_affinity": lock_cpu_affinity,
        },
        timeout=timeout,
    )

    payload = response.get("payload")
    if response.get("ok"):
        if not isinstance(payload, IsolatedRunResult):
            raise RuntimeError("Isolated worker returned an invalid result payload")
        return payload

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


def _main(argv: list[str]) -> int:
    """Entry point for the worker subprocess. This should never be called directly; it's invoked via subprocess in :func:`run_isolated`."""
    if len(argv) != 3 or argv[0] != _WORKER_FLAG:
        return 2

    request_path = Path(argv[1])
    response_path = Path(argv[2])
    response: dict[str, Any]

    try:
        # Read the request, execute it, and write back a structured response.
        with request_path.open("rb") as handle:
            response = _execute_worker_request(pickle.load(handle))
    except Exception as exc:
        response = {"ok": False, "payload": _child_error_payload(exc)}

    with suppress(OSError), response_path.open("wb") as handle:
        pickle.dump(response, handle)

    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
