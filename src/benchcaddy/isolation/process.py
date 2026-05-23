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
import json
import logging
import os
import signal
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

from ._protocol import (
    request_from_json_payload,
    request_to_json_payload,
    response_from_json_payload,
    response_to_json_payload,
)
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


def _callable_source_path(fn: Callable[..., Any]) -> str | None:
    """Return the canonical source path for a callable when Python can resolve one."""
    resolved_target = inspect.unwrap(fn)
    with suppress(OSError, TypeError, ValueError):
        source_path = inspect.getsourcefile(resolved_target) or inspect.getfile(resolved_target)
        if source_path:
            return str(Path(source_path).resolve())
    return None


def _child_error_payload(exc: Exception) -> dict[str, str]:
    """Convert a worker exception into the plain-string payload sent across IPC."""
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _unsupported_target_error(reason: str) -> TypeError:
    """Build the shared validation error for unsupported fresh-process call targets."""
    return TypeError(f"fresh isolated execution requires an importable module-level function, static method, or class method; {reason}")


def _parent_import_path_snapshot() -> list[str]:
    """Capture the parent's effective import roots in deterministic order for child replay."""
    import_roots: list[str] = []
    for entry in sys.path:
        try:
            root = (Path.cwd() if not entry else Path(entry)).resolve()
        except OSError:
            continue

        root_text = str(root)
        if root_text not in import_roots:
            import_roots.append(root_text)

    return import_roots


def _module_name_from_source_path(source_path: str | None, import_paths: list[str]) -> str | None:
    """Attempt to map a source file path back to an importable module name using one parent import snapshot."""
    if not source_path:
        return None

    resolved_source = Path(source_path).resolve()

    for import_path in import_paths:
        root = Path(import_path)
        with suppress(ValueError):
            relative_path = resolved_source.relative_to(root)
            module_parts = relative_path.with_suffix("").parts
            if not module_parts:
                continue
            if module_parts[-1] == "__init__":
                module_parts = module_parts[:-1]
            if module_parts:
                return ".".join(module_parts)

    return None


def _importable_target_reference(fn: Callable[..., Any], import_paths: list[str]) -> tuple[str, str, str | None]:
    """Extract the module name, qualname, and source path for a callable target under one parent import snapshot."""

    resolved_target = inspect.unwrap(fn)
    module_name = getattr(resolved_target, "__module__", None)
    qualname = getattr(resolved_target, "__qualname__", None)
    source_path = _callable_source_path(resolved_target)

    if not module_name or not qualname:
        raise _unsupported_target_error("the target is missing module or qualname metadata, so the worker cannot re-import it in a fresh process.")

    if module_name == "__main__":
        if resolved_module_name := _module_name_from_source_path(source_path, import_paths):
            module_name = resolved_module_name
        else:
            raise _unsupported_target_error(
                "the target is defined in __main__ and could not be mapped back to an importable module path. "
                "Run the benchmark from a directory where the script is importable, or move the target into an importable module."
            )

    return module_name, qualname, source_path


def _validated_target_reference(fn: Callable[..., Any], import_paths: list[str]) -> tuple[str, str, str | None]:
    """Validate that one callable still resolves to the same source file under one parent import snapshot."""
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

    module_name, qualname, source_path = _importable_target_reference(fn, import_paths)

    if "<lambda>" in qualname:
        raise _unsupported_target_error("lambdas are unsupported because they do not provide a stable import path for the worker. Define a named module-level function instead.")

    if "<locals>" in qualname:
        raise _unsupported_target_error(
            "nested or local functions are unsupported because they are scoped to a parent frame and cannot be re-imported by qualname in the worker. "
            "Move the benchmark target to module scope instead."
        )

    try:
        resolved_callable = _resolve_callable(module_name, qualname)
    except Exception as exc:
        raise _unsupported_target_error(
            f"could not resolve {module_name}.{qualname}. Ensure the symbol is importable in the child process and exposed at that module path."
        ) from exc

    if source_path:
        resolved_source_path = _callable_source_path(resolved_callable)
        if resolved_source_path != source_path:
            raise _unsupported_target_error(
                f"could not resolve {module_name}.{qualname} back to the same source file that defined the original target. "
                f"Expected {source_path}, got {resolved_source_path or 'unresolved'}."
            )

    return module_name, qualname, source_path


def validate_isolated_target(fn: Callable[..., Any]) -> tuple[str, str, list[str]]:
    """Validate that the provided callable is a supported target for fresh subprocess isolation and return its module and qualname for later import."""
    import_paths = _parent_import_path_snapshot()
    module_name, qualname, _source_path = _validated_target_reference(fn, import_paths)
    return module_name, qualname, import_paths


def _run_observed_call(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> IsolatedRunResult:
    """Run one measured call while collecting observation records for the result."""
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

    import_paths = [str(path) for path in request.get("import_paths", ()) if path]
    if import_paths:
        sys.path[:] = import_paths

    fn = _resolve_callable(request["module_name"], request["qualname"])
    expected_source_path = request.get("source_path")
    if expected_source_path:
        resolved_source_path = _callable_source_path(fn)
        if resolved_source_path != expected_source_path:
            raise RuntimeError(
                f"Worker resolved the target from a different source file than the parent validated. Expected {expected_source_path}, got {resolved_source_path or 'unresolved'}."
            )
    args = tuple(request.get("args", ()))
    kwargs = dict(request.get("kwargs", {}))
    warmup_runs = int(request.get("warmup_runs", 0))

    # Prepare the worker process before any warmups or measured execution.
    prepare_system(lock_cpu_affinity=bool(request.get("lock_cpu_affinity", True)))

    # When GC stays enabled, force a collection up front so warmups and measurement start from a stable point.
    if bool(request.get("disable_gc", False)):
        gc.disable()
    else:
        gc.collect()

    try:
        # Warmups happen outside the measured call so one-time effects do not contaminate timing.
        for _ in range(warmup_runs):
            fn(*args, **kwargs)

        # Execute measured call and return structured result or error payload.
        return {"ok": True, "payload": _run_observed_call(fn, args, kwargs)}
    except Exception as exc:
        return {"ok": False, "payload": _child_error_payload(exc)}


def _run_subprocess_worker(request: dict[str, Any], timeout: float | None) -> dict[str, Any]:
    """Run a worker subprocess to execute one isolated request and return its structured response."""
    with tempfile.TemporaryDirectory(prefix="benchcaddy-isolated-") as temp_dir:
        response_path = Path(temp_dir) / "response.json"
        request_bytes = json.dumps(request_to_json_payload(request), ensure_ascii=True, allow_nan=False).encode("utf-8")
        worker_script = str(Path(__file__).with_name("worker.py").resolve())

        cmd = [
            sys.executable,
            "-I",
            worker_script,
            _WORKER_FLAG,
            str(response_path),
        ]

        worker_env = os.environ.copy()
        for variable_name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
            worker_env.pop(variable_name, None)

        popen_kwargs = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "close_fds": True,
            "env": worker_env,
        }

        # Windows: avoid creating a visible console for the child. POSIX: create
        # a new session so we can terminate the whole group if the child spawns children.
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            popen_kwargs["preexec_fn"] = os.setsid

        logger = logging.getLogger(__name__)
        logger.debug("Spawning worker: %s", cmd)

        # Spawn the worker process. If this fails, there's no subprocess to clean up, so we can just raise.
        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)
        except Exception as exc:
            raise RuntimeError(f"failed to spawn worker process: {exc}") from exc

        def _terminate_gracefully(p: subprocess.Popen) -> None:
            if os.name != "nt":
                with suppress(Exception):
                    os.killpg(p.pid, signal.SIGTERM)
            else:
                with suppress(Exception):
                    p.terminate()

        def _kill_forcefully(p: subprocess.Popen) -> None:
            if os.name != "nt":
                with suppress(Exception):
                    os.killpg(p.pid, signal.SIGKILL)
            else:
                with suppress(Exception):
                    p.kill()

        # Wait for the worker to finish or time out. If the timeout expires, attempt a graceful
        # termination first, then escalate to a forceful kill if the worker does not exit.
        try:
            stdout, stderr = proc.communicate(input=request_bytes, timeout=timeout)
            completed_returncode = proc.returncode
            completed_stdout = (stdout or b"").decode("utf-8", errors="replace")
            completed_stderr = (stderr or b"").decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired as e:
            logger.debug("Worker timed out after %s seconds", timeout)
            _terminate_gracefully(proc)
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                logger.debug("Worker did not exit after terminate; killing")
                _kill_forcefully(proc)
                proc.communicate()

            raise TimeoutError(f"run_isolated timed out after {timeout}s") from e

        # If the worker wrote a response, return it. Otherwise surface
        # the child's stderr/stdout as an error.
        if response_path.exists():
            try:
                with response_path.open("r", encoding="utf-8") as handle:
                    return response_from_json_payload(json.load(handle))
            except Exception as exc:
                # The worker wrote a response file but it could not be
                # deserialized. Surface a clear error including stderr/stdout
                # to aid diagnosis.
                details = completed_stderr.strip() or completed_stdout.strip()
                raise RuntimeError(f"Could not deserialize worker response: {exc}. Child output: {details}") from exc

        stderr_text = completed_stderr.strip()
        stdout_text = completed_stdout.strip()
        details = stderr_text or stdout_text or f"worker exited with code {completed_returncode}"
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
        *args* / *kwargs* to be JSON-serializable, while the isolated
        return value and observations must be JSON-serializable.
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

    import_paths = _parent_import_path_snapshot()
    module_name, qualname, source_path = _validated_target_reference(fn, import_paths)

    # Pass the resolved import reference into the worker request so the child can replay the parent's import roots for the target.
    # The worker process will re-import the target function and execute it with the provided arguments,
    # then send back a structured result or error payload. Both directions now use strictly validated JSON,
    # so the parent never deserializes executable worker-controlled objects and the request no longer touches pickle either.
    response = _run_subprocess_worker(
        {
            "module_name": module_name,
            "qualname": qualname,
            "args": args,
            "kwargs": kw,
            "import_paths": list(import_paths),
            "source_path": source_path,
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
    if len(argv) != 2 or argv[0] != _WORKER_FLAG:
        return 2

    response_path = Path(argv[1])
    response: dict[str, Any]

    try:
        # The worker reads one request, executes it, and writes back one structured response.
        response = _execute_worker_request(request_from_json_payload(json.loads(sys.stdin.buffer.read().decode("utf-8"))))
    except Exception as exc:
        response = {"ok": False, "payload": _child_error_payload(exc)}

    try:
        serializable_response = response_to_json_payload(response)
    except (AttributeError, TypeError, ValueError) as exc:
        response = {
            "ok": False,
            "payload": {
                "type": "SerializationError",
                "message": (
                    "Worker could not serialize the isolated result payload. "
                    "Ensure the benchmark return value and recorded observations are JSON-serializable. "
                    f"Original error: {exc}"
                ),
                "traceback": traceback.format_exc(),
            },
        }

        serializable_response = response_to_json_payload(response)

    try:
        with response_path.open("w", encoding="utf-8") as handle:
            json.dump(serializable_response, handle, ensure_ascii=True, allow_nan=False)
    except OSError:
        return 1

    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
