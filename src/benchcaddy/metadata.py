from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil
from git import InvalidGitRepositoryError, NoSuchPathError, Repo

_COMMAND_TIMEOUT_SECONDS = 2


@dataclass
class GitState:
    branch: str | None
    commit_hash: str | None
    dirty: bool | None


@dataclass
class ProcessState:
    pid: int
    priority: int | str | None
    affinity: list[int]


@dataclass
class EnvironmentMetadata:
    python_version: str
    operating_system: str
    cpu_model: str
    gpu_model: str | None
    git: GitState
    process: ProcessState


def _run_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    output = completed.stdout.strip()
    return output or None


def _read_cpu_model() -> str:
    processor = platform.processor().strip()
    if processor:
        return processor

    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                _, _, value = line.partition(":")
                return value.strip()

    sysctl_value = _run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
    if sysctl_value:
        return sysctl_value

    return "Unknown CPU"


def _read_gpu_model() -> str | None:
    nvidia_output = _run_command(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    if nvidia_output:
        gpu_names = [line.strip() for line in nvidia_output.splitlines() if line.strip()]
        if gpu_names:
            return ", ".join(gpu_names)

    system = platform.system().lower()
    if system == "darwin":
        system_profiler = _run_command(["system_profiler", "SPDisplaysDataType"])
        if system_profiler:
            for line in system_profiler.splitlines():
                stripped = line.strip()
                if stripped.startswith(("Chipset Model:", "Model:")):
                    _, _, value = stripped.partition(":")
                    gpu_name = value.strip()
                    if gpu_name:
                        return gpu_name
        return None
    if system == "windows":
        wmic_output = _run_command(
            ["wmic", "path", "win32_VideoController", "get", "name"]
        )
        if wmic_output:
            for line in wmic_output.splitlines():
                stripped = line.strip()
                if stripped and stripped.lower() != "name":
                    return stripped
    return None


def collect_git_state(cwd: Path | None = None) -> GitState:
    repository_path = cwd or Path.cwd()
    try:
        repo = Repo(repository_path, search_parent_directories=True)
    except (InvalidGitRepositoryError, NoSuchPathError):
        return GitState(branch=None, commit_hash=None, dirty=None)

    branch = None if repo.head.is_detached else repo.active_branch.name

    return GitState(
        branch=branch,
        commit_hash=repo.head.commit.hexsha,
        dirty=repo.is_dirty(untracked_files=True),
    )


def collect_process_state() -> ProcessState:
    process = psutil.Process()
    affinity: list[int] = []
    if hasattr(process, "cpu_affinity"):
        try:
            affinity = list(process.cpu_affinity())
        except (psutil.AccessDenied, NotImplementedError):
            affinity = []

    try:
        priority: int | str | None = process.nice()
    except (psutil.AccessDenied, AttributeError):
        priority = None

    return ProcessState(pid=process.pid, priority=priority, affinity=affinity)


def collect_environment_metadata(cwd: Path | None = None) -> EnvironmentMetadata:
    return EnvironmentMetadata(
        python_version=sys.version.split()[0],
        operating_system=platform.platform(),
        cpu_model=_read_cpu_model(),
        gpu_model=_read_gpu_model(),
        git=collect_git_state(cwd),
        process=collect_process_state(),
    )


def metadata_to_dict(metadata: EnvironmentMetadata) -> dict[str, Any]:
    return asdict(metadata)
