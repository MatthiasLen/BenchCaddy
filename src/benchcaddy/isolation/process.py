"""Run a callable in an isolated subprocess.

Executing benchmarks in a freshly-spawned process eliminates
interpreter warm-up effects, module-level side effects, and GC state
inherited from the host process — providing cleaner, more reproducible
measurements.
"""

from __future__ import annotations

import gc
import multiprocessing
import multiprocessing.connection
from collections.abc import Callable
from typing import Any


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
    except Exception as exc:  # noqa: BLE001
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
        raise RuntimeError(f"Isolated process raised an exception: {payload}") from payload if isinstance(payload, Exception) else RuntimeError(str(payload))

    return payload
