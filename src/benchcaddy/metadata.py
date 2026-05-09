from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
from git import Repo
from git.exc import GitError

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
    rss_bytes: int | None


@dataclass
class EnvironmentMetadata:
    python_version: str
    operating_system: str
    cpu_model: str
    total_memory_bytes: int | None
    gpu_model: str | None
    git: GitState
    process: ProcessState


def _empty_git_state() -> GitState:
    return GitState(branch=None, commit_hash=None, dirty=None)


def _read_git_branch(repo: Repo) -> str | None:
    try:
        return None if repo.head.is_detached else repo.active_branch.name
    except (GitError, OSError, TypeError, ValueError):
        return None


def _read_git_commit_hash(repo: Repo) -> str | None:
    try:
        return repo.head.commit.hexsha
    except (GitError, OSError, TypeError, ValueError):
        return None


def _read_git_dirty(repo: Repo) -> bool | None:
    if repo.bare:
        return None

    try:
        return repo.is_dirty(untracked_files=True)
    except (GitError, OSError, ValueError):
        return None


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

    return completed.stdout.strip() or None


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
    if nvidia_output := _run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]):
        return ", ".join(line.strip() for line in nvidia_output.splitlines() if line.strip()) or None

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
        if wmic_output := _run_command(["wmic", "path", "win32_VideoController", "get", "name"]):
            for line in wmic_output.splitlines():
                stripped = line.strip()
                if stripped and stripped.lower() != "name":
                    return stripped
    return None


def collect_git_state(cwd: Path | None = None) -> GitState:
    repository_path = cwd or Path.cwd()
    try:
        repo = Repo(repository_path, search_parent_directories=True)
    except (GitError, OSError, ValueError):
        return _empty_git_state()

    commit_hash = _read_git_commit_hash(repo)
    if commit_hash is None:
        return _empty_git_state()

    return GitState(
        branch=_read_git_branch(repo),
        commit_hash=commit_hash,
        dirty=_read_git_dirty(repo),
    )


def collect_process_state() -> ProcessState:
    process = psutil.Process()
    try: affinity = list(process.cpu_affinity())
    except (psutil.AccessDenied, NotImplementedError, AttributeError): affinity = []
    try: priority = process.nice()
    except (psutil.AccessDenied, AttributeError): priority = None
    try: rss_bytes = process.memory_info().rss
    except (psutil.AccessDenied, AttributeError, OSError): rss_bytes = None

    return ProcessState(
        pid=process.pid,
        priority=priority,
        affinity=affinity,
        rss_bytes=rss_bytes,
    )


def collect_environment_metadata(cwd: Path | None = None) -> EnvironmentMetadata:
    try:
        total_memory_bytes = psutil.virtual_memory().total
    except (psutil.Error, AttributeError):
        total_memory_bytes = None

    return EnvironmentMetadata(
        python_version=sys.version.split()[0],
        operating_system=platform.platform(),
        cpu_model=_read_cpu_model(),
        total_memory_bytes=total_memory_bytes,
        gpu_model=_read_gpu_model(),
        git=collect_git_state(cwd),
        process=collect_process_state(),
    )


def metadata_to_dict(metadata: EnvironmentMetadata) -> dict[str, object]:
    payload = asdict(metadata)
    git_state = payload.get("git")
    if isinstance(git_state, dict) and not any(value is not None for value in git_state.values()):
        payload.pop("git", None)
    return payload
