from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from git.exc import GitCommandError, InvalidGitRepositoryError

import benchcaddy.metadata as metadata_module
from benchcaddy.db import get_run_details
from benchcaddy.metadata import collect_environment_metadata, collect_git_state, metadata_to_dict


def test_metadata_to_dict_omits_git_outside_repository(tmp_path: Path) -> None:
    payload = metadata_to_dict(collect_environment_metadata(tmp_path))

    assert "git" not in payload
    assert payload["process"]["pid"] > 0


def test_collect_git_state_handles_unborn_head(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)

    git_state = collect_git_state(tmp_path)
    payload = metadata_to_dict(collect_environment_metadata(tmp_path))

    assert git_state.branch is None
    assert git_state.commit_hash is None
    assert git_state.dirty is None
    assert "git" not in payload


def test_collect_git_state_handles_detached_head(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "BenchCaddy Tests"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "benchcaddy@example.com"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "tracked.txt").write_text("content\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True, text=True)

    head_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "checkout", "--detach"], cwd=tmp_path, check=True, capture_output=True, text=True)

    git_state = collect_git_state(tmp_path)
    payload = metadata_to_dict(collect_environment_metadata(tmp_path))

    assert git_state.branch is None
    assert git_state.commit_hash == head_commit
    assert git_state.dirty is False
    assert payload["git"] == {
        "branch": None,
        "commit_hash": head_commit,
        "dirty": False,
    }


def test_collect_git_state_handles_repo_discovery_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def failing_repo(*args: object, **kwargs: object) -> object:
        raise InvalidGitRepositoryError("broken repo")

    monkeypatch.setattr(metadata_module, "Repo", failing_repo)

    assert collect_git_state(tmp_path).commit_hash is None
    assert "git" not in metadata_to_dict(collect_environment_metadata(tmp_path))


def test_collect_git_state_preserves_commit_when_branch_lookup_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeCommit:
        hexsha = "deadbeef"

    class FakeHead:
        is_detached = False
        commit = FakeCommit()

    class FakeActiveBranch:
        @property
        def name(self) -> str:
            raise TypeError("branch unavailable")

    class FakeRepo:
        bare = False
        head = FakeHead()
        active_branch = FakeActiveBranch()

        def is_dirty(self, *, untracked_files: bool) -> bool:
            assert untracked_files is True
            return False

    monkeypatch.setattr(metadata_module, "Repo", lambda *args, **kwargs: FakeRepo())

    assert collect_git_state(tmp_path) == metadata_module.GitState(branch=None, commit_hash="deadbeef", dirty=False)


def test_collect_git_state_preserves_commit_when_dirty_lookup_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeCommit:
        hexsha = "deadbeef"

    class FakeHead:
        is_detached = False
        commit = FakeCommit()

    class FakeActiveBranch:
        name = "main"

    class FakeRepo:
        bare = False
        head = FakeHead()
        active_branch = FakeActiveBranch()

        def is_dirty(self, *, untracked_files: bool) -> bool:
            assert untracked_files is True
            raise GitCommandError("status", 1)

    monkeypatch.setattr(metadata_module, "Repo", lambda *args, **kwargs: FakeRepo())

    git_state = collect_git_state(tmp_path)

    assert git_state == metadata_module.GitState(branch="main", commit_hash="deadbeef", dirty=None)


def test_collect_git_state_omits_git_when_commit_lookup_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class MissingCommit:
        @property
        def hexsha(self) -> str:
            raise ValueError("missing commit")

    class FakeHead:
        is_detached = False
        commit = MissingCommit()

    class FakeActiveBranch:
        name = "main"

    class FakeRepo:
        bare = False
        head = FakeHead()
        active_branch = FakeActiveBranch()

        def is_dirty(self, *, untracked_files: bool) -> bool:
            return False

    monkeypatch.setattr(metadata_module, "Repo", lambda *args, **kwargs: FakeRepo())

    assert collect_git_state(tmp_path) == metadata_module.GitState(branch=None, commit_hash=None, dirty=None)
    assert "git" not in metadata_to_dict(collect_environment_metadata(tmp_path))


def test_environment_round_trip_preserves_git_payload(
    tmp_path: Path,
    environment_payload: dict[str, object],
    record_simple_run,
) -> None:
    database_path = tmp_path / "benchcaddy.db"

    record_simple_run(
        suite_name="git-aware-suite",
        database_path=database_path,
        configuration={"variant": "baseline"},
        environment=environment_payload,
    )

    run = get_run_details((1, 1), database_path)

    assert run is not None
    assert run["environment"]["git"] == environment_payload["git"]
    assert run["environment"]["process"] == environment_payload["process"]


def test_environment_round_trip_omits_git_payload_when_missing(tmp_path: Path, record_simple_run) -> None:
    database_path = tmp_path / "benchcaddy.db"
    environment_payload = metadata_to_dict(collect_environment_metadata(tmp_path))

    record_simple_run(
        suite_name="non-git-suite",
        database_path=database_path,
        configuration={"variant": "baseline"},
        environment=environment_payload,
    )

    run = get_run_details((1, 1), database_path)

    assert run is not None
    assert "git" not in run["environment"]
    assert run["environment"]["process"] == environment_payload["process"]


def test_run_command_returns_none_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def timed_out(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=["fake"], timeout=1)

    monkeypatch.setattr(metadata_module.subprocess, "run", timed_out)

    assert metadata_module._run_command(["fake"]) is None


def test_read_gpu_model_uses_first_windows_wmic_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        metadata_module,
        "_run_command",
        lambda command: "Name\nNVIDIA RTX 6000\nIntegrated GPU\n" if command[:2] == ["wmic", "path"] else None,
    )

    assert metadata_module._read_gpu_model() == "NVIDIA RTX 6000"


def test_collect_process_state_handles_access_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        pid = 321

        def cpu_affinity(self):
            raise metadata_module.psutil.AccessDenied()

        def nice(self):
            raise metadata_module.psutil.AccessDenied()

        def memory_info(self):
            raise metadata_module.psutil.AccessDenied()

    monkeypatch.setattr(metadata_module.psutil, "Process", lambda: FakeProcess())

    process_state = metadata_module.collect_process_state()

    assert process_state.pid == 321
    assert process_state.affinity == []
    assert process_state.priority is None
    assert process_state.rss_bytes is None
