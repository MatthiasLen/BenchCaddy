from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from benchcaddy.db import get_run_details, record_benchmark_run
from git.exc import GitCommandError, InvalidGitRepositoryError

import benchcaddy.metadata as metadata_module
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
) -> None:
    database_path = tmp_path / "benchcaddy.db"

    record_benchmark_run(
        suite_name="git-aware-suite",
        target_name="benchmark_target",
        configuration={"variant": "baseline"},
        samples=[0.1, 0.1],
        observations=[],
        median_seconds=0.1,
        min_seconds=0.1,
        max_seconds=0.1,
        std_seconds=0.0,
        environment=environment_payload,
        database_path=database_path,
    )

    run = get_run_details((1, 1), database_path)

    assert run is not None
    assert run["environment"]["git"] == environment_payload["git"]
    assert run["environment"]["process"] == environment_payload["process"]


def test_environment_round_trip_omits_git_payload_when_missing(tmp_path: Path) -> None:
    database_path = tmp_path / "benchcaddy.db"
    environment_payload = metadata_to_dict(collect_environment_metadata(tmp_path))

    record_benchmark_run(
        suite_name="non-git-suite",
        target_name="benchmark_target",
        configuration={"variant": "baseline"},
        samples=[0.1, 0.1],
        observations=[],
        median_seconds=0.1,
        min_seconds=0.1,
        max_seconds=0.1,
        std_seconds=0.0,
        environment=environment_payload,
        database_path=database_path,
    )

    run = get_run_details((1, 1), database_path)

    assert run is not None
    assert "git" not in run["environment"]
    assert run["environment"]["process"] == environment_payload["process"]