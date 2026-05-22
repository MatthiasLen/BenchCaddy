"""Internal JSON IPC schema policy for subprocess isolation.

This module owns the security-sensitive normalization and validation rules
for data crossing the parent/worker boundary. It keeps transport schema
policy separate from process orchestration so the protocol can be reviewed
and evolved in one place.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .observability import IsolatedRunResult


def json_transport_value(value: Any) -> Any:
    """Normalize Python and NumPy values into the plain JSON shapes used by the IPC layer."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, np.generic):
        return json_transport_value(value.item())
    if isinstance(value, np.ndarray):
        return json_transport_value(value.tolist())
    if isinstance(value, float):
        return value
    if isinstance(value, (list, tuple)):
        return [json_transport_value(item) for item in value]
    if isinstance(value, dict):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("worker payload dictionaries must use string keys for JSON IPC")
            converted[key] = json_transport_value(item)
        return converted
    raise TypeError(f"worker payload must be JSON-serializable; unsupported value type {type(value).__name__}")


def ensure_finite_json_value(value: Any, *, field_name: str) -> Any:
    """Reject NaN and +/-inf anywhere in a normalized JSON payload to keep numeric types stable."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{field_name} cannot contain non-finite floats for JSON IPC")
        return value
    if isinstance(value, list):
        for item in value:
            ensure_finite_json_value(item, field_name=field_name)
        return value
    if isinstance(value, dict):
        for item in value.values():
            ensure_finite_json_value(item, field_name=field_name)
        return value
    return value


def json_compatible_value(value: Any, *, field_name: str = "worker payload") -> Any:
    """Normalize a value for JSON IPC and reject non-finite floats at the transport boundary."""
    return ensure_finite_json_value(json_transport_value(value), field_name=field_name)


def finite_json_number(value: Any, *, field_name: str) -> float:
    """Validate a field that must remain a finite numeric scalar across the IPC boundary."""
    normalized = json_compatible_value(value, field_name=field_name)
    if not isinstance(normalized, (int, float)) or isinstance(normalized, bool):
        raise TypeError(f"{field_name} must be a finite number")
    return float(normalized)


def json_observation_record(record: Any) -> dict[str, Any]:
    """Validate one observation record against the compact IPC schema used by isolated runs."""
    if not isinstance(record, dict):
        raise TypeError("worker observation payloads must be JSON objects")

    label = record.get("label")
    kind = record.get("kind")
    if not isinstance(label, str) or not isinstance(kind, str):
        raise TypeError("worker observation payloads must contain string 'label' and 'kind' fields")

    if kind == "time":
        if set(record) != {"label", "kind", "duration_seconds"}:
            raise TypeError("time observations must contain exactly 'label', 'kind', and 'duration_seconds'")
        return {
            "label": label,
            "kind": kind,
            "duration_seconds": finite_json_number(record["duration_seconds"], field_name="time observations field 'duration_seconds'"),
        }

    if kind == "return":
        if set(record) != {"label", "kind", "value"}:
            raise TypeError("return observations must contain exactly 'label', 'kind', and 'value'")
        return {
            "label": label,
            "kind": kind,
            "value": json_compatible_value(record["value"], field_name="worker observation return values"),
        }

    raise TypeError(f"worker observation payload kind '{kind}' is unsupported")


def request_to_json_payload(request: dict[str, Any]) -> dict[str, Any]:
    """Serialize the parent request into the strict JSON object sent over worker stdin."""
    module_name = request.get("module_name")
    qualname = request.get("qualname")
    if not isinstance(module_name, str) or not isinstance(qualname, str):
        raise TypeError("worker request must contain string 'module_name' and 'qualname' fields")

    args = json_compatible_value(request.get("args", ()), field_name="worker request arguments")
    kwargs = json_compatible_value(request.get("kwargs", {}), field_name="worker request keyword arguments")
    import_paths = json_compatible_value(request.get("import_paths", []), field_name="worker request import paths")
    source_path = request.get("source_path")
    if not isinstance(args, list):
        raise TypeError("worker request field 'args' must serialize to a JSON array")
    if not isinstance(kwargs, dict):
        raise TypeError("worker request field 'kwargs' must serialize to a JSON object")
    if not isinstance(import_paths, list) or any(not isinstance(path, str) for path in import_paths):
        raise TypeError("worker request field 'import_paths' must be a list of strings")
    if source_path is not None and not isinstance(source_path, str):
        raise TypeError("worker request field 'source_path' must be a string when provided")

    disable_gc = request.get("disable_gc", False)
    lock_cpu_affinity = request.get("lock_cpu_affinity", True)
    warmup_runs = request.get("warmup_runs", 0)
    if not isinstance(disable_gc, bool) or not isinstance(lock_cpu_affinity, bool):
        raise TypeError("worker request boolean fields must stay booleans for JSON IPC")
    if not isinstance(warmup_runs, int) or isinstance(warmup_runs, bool):
        raise TypeError("worker request field 'warmup_runs' must be an integer")

    return {
        "module_name": module_name,
        "qualname": qualname,
        "args": args,
        "kwargs": kwargs,
        "import_paths": import_paths,
        "source_path": source_path,
        "disable_gc": disable_gc,
        "warmup_runs": warmup_runs,
        "lock_cpu_affinity": lock_cpu_affinity,
    }


def request_from_json_payload(request: Any) -> dict[str, Any]:
    """Validate a JSON-decoded worker request and rebuild the child execution payload."""
    if not isinstance(request, dict):
        raise TypeError("worker request must be a JSON object")
    if set(request) != {
        "module_name",
        "qualname",
        "args",
        "kwargs",
        "import_paths",
        "source_path",
        "disable_gc",
        "warmup_runs",
        "lock_cpu_affinity",
    }:
        raise TypeError(
            "worker request must contain exactly 'module_name', 'qualname', 'args', 'kwargs', 'import_paths', 'source_path', 'disable_gc', 'warmup_runs', and 'lock_cpu_affinity'"
        )

    serialized_request = request_to_json_payload(request)
    return {
        "module_name": serialized_request["module_name"],
        "qualname": serialized_request["qualname"],
        "args": tuple(serialized_request["args"]),
        "kwargs": dict(serialized_request["kwargs"]),
        "import_paths": list(serialized_request["import_paths"]),
        "source_path": serialized_request["source_path"],
        "disable_gc": serialized_request["disable_gc"],
        "warmup_runs": serialized_request["warmup_runs"],
        "lock_cpu_affinity": serialized_request["lock_cpu_affinity"],
    }


def response_to_json_payload(response: dict[str, Any]) -> dict[str, Any]:
    """Serialize the worker response into the strict JSON object written back to the parent."""
    ok = response.get("ok")
    if not isinstance(ok, bool):
        raise TypeError("worker response must contain a boolean 'ok' field before JSON encoding")

    payload = response.get("payload")
    if ok:
        if not isinstance(payload, IsolatedRunResult):
            raise TypeError("worker success payload must be an IsolatedRunResult before JSON encoding")
        return {
            "ok": True,
            "payload": {
                "elapsed_seconds": finite_json_number(payload.elapsed_seconds, field_name="worker success payload field 'elapsed_seconds'"),
                "return_value": json_compatible_value(payload.return_value, field_name="worker return values"),
                "observations": [json_observation_record(record) for record in payload.observations],
            },
        }

    error_payload = json_compatible_value(payload, field_name="worker error payload")
    if not isinstance(error_payload, dict):
        raise TypeError("worker error payload must be a JSON object")
    return {"ok": False, "payload": error_payload}


def response_from_json_payload(response: Any) -> dict[str, Any]:
    """Validate a JSON-decoded worker response and rebuild the in-process result object."""
    if not isinstance(response, dict) or set(response) != {"ok", "payload"}:
        raise TypeError("worker response must be a JSON object with exactly 'ok' and 'payload'")

    ok = response["ok"]
    if not isinstance(ok, bool):
        raise TypeError("worker response field 'ok' must be a boolean")

    payload = response["payload"]
    if not ok:
        error_payload = json_compatible_value(payload, field_name="worker error payload")
        if not isinstance(error_payload, dict):
            raise TypeError("worker error payload must be a JSON object")
        return {"ok": False, "payload": error_payload}

    if not isinstance(payload, dict) or set(payload) != {"elapsed_seconds", "return_value", "observations"}:
        raise TypeError("worker success payload must be a JSON object with exactly 'elapsed_seconds', 'return_value', and 'observations'")

    elapsed_seconds = finite_json_number(payload["elapsed_seconds"], field_name="worker success payload field 'elapsed_seconds'")
    return_value = json_compatible_value(payload["return_value"], field_name="worker return values")
    if not isinstance(payload["observations"], list):
        raise TypeError("worker success payload field 'observations' must be a list of JSON objects")
    observations = [json_observation_record(record) for record in payload["observations"]]

    return {
        "ok": True,
        "payload": IsolatedRunResult(
            elapsed_seconds=elapsed_seconds,
            return_value=return_value,
            observations=observations,
        ),
    }
