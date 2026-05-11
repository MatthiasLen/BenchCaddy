from __future__ import annotations

import pytest


@pytest.fixture
def environment_payload() -> dict[str, object]:
    return {
        "python_version": "3.12.0",
        "operating_system": "TestOS-1.0",
        "cpu_model": "Test CPU",
        "total_memory_bytes": 17179869184,
        "gpu_model": None,
        "git": {
            "branch": "main",
            "commit_hash": "deadbeef",
            "dirty": False,
        },
        "process": {
            "pid": 123,
            "priority": 0,
            "affinity": [0],
            "rss_bytes": 4096,
        },
    }
