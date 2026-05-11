from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TARGETS = ("src", "tests", "scripts")
DEFAULT_EXCLUDES = (
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
)


@dataclass(slots=True)
class CheckResult:
    name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run code-structure checks that are useful for humans and coding agents. The default targets stay scoped to the repository source tree.")
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_TARGETS),
        help="Files or directories to scan. Defaults to src tests scripts.",
    )
    parser.add_argument(
        "--ruff-only",
        action="store_true",
        help="Run only Ruff checks and skip the deeper redundancy and complexity tools.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="Output format. Use json or markdown for agent workflows.",
    )
    parser.add_argument(
        "--ruff-fix",
        action="store_true",
        help="Apply all Ruff fixes, including formatter changes. Default behavior is check-only.",
    )
    parser.add_argument(
        "--min-similarity-lines",
        type=int,
        default=4,
        help="Minimum duplicate block size for Pylint duplicate-code.",
    )
    parser.add_argument(
        "--min-vulture-confidence",
        type=int,
        default=80,
        help="Minimum confidence for dead-code findings from Vulture.",
    )
    parser.add_argument(
        "--max-complexity",
        type=int,
        default=10,
        help="Cyclomatic complexity threshold for Radon.",
    )
    parser.add_argument(
        "--min-maintainability",
        type=str,
        default="B",
        help="Lowest acceptable maintainability grade for Radon MI (A-F).",
    )
    return parser.parse_args()


def existing_paths(paths: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    for path in paths:
        if Path(path).exists():
            resolved.append(path)
        else:
            print(f"Skipping missing path: {path}", file=sys.stderr)
    if not resolved:
        raise SystemExit("No valid paths to scan.")
    return resolved


def ruff_command(paths: Sequence[str], fix: bool) -> list[str]:
    command = ["ruff", "check"]
    if fix:
        command.append("--fix")
    command.extend(paths)
    return command


def ruff_format_command(paths: Sequence[str], check_only: bool) -> list[str]:
    command = ["ruff", "format"]
    if check_only:
        command.append("--check")
    command.extend(paths)
    return command


def pylint_command(paths: Sequence[str], min_similarity_lines: int) -> list[str]:
    return [
        "pylint",
        "--disable=all",
        "--enable=duplicate-code",
        f"--min-similarity-lines={min_similarity_lines}",
        *paths,
    ]


def vulture_command(paths: Sequence[str], min_confidence: int) -> list[str]:
    return [
        "vulture",
        *paths,
        f"--min-confidence={min_confidence}",
        *[f"--exclude={pattern}" for pattern in DEFAULT_EXCLUDES],
    ]


def min_rank_for_complexity(max_complexity: int) -> str:
    if max_complexity < 5:
        return "B"
    if max_complexity < 10:
        return "C"
    if max_complexity < 20:
        return "D"
    if max_complexity < 30:
        return "E"
    return "F"


def radon_cc_command(paths: Sequence[str], max_complexity: int) -> list[str]:
    return [
        "radon",
        "cc",
        *paths,
        "--min",
        min_rank_for_complexity(max_complexity),
        "--show-complexity",
        "--no-assert",
        "--total-average",
    ]


def radon_mi_command(paths: Sequence[str], min_maintainability: str) -> list[str]:
    return [
        "radon",
        "mi",
        *paths,
        "--min",
        min_maintainability.upper(),
        "--show",
    ]


def format_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def missing_tools(commands: Sequence[Sequence[str]]) -> list[str]:
    missing: list[str] = []
    for command in commands:
        tool_name = command[0]
        if shutil.which(tool_name) is None and tool_name not in missing:
            missing.append(tool_name)
    return missing


def run_check(name: str, command: Sequence[str]) -> CheckResult:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return CheckResult(
        name=name,
        command=list(command),
        returncode=result.returncode,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
    )


def summarize_agent_guidance(results: Sequence[CheckResult], max_complexity: int) -> list[str]:
    guidance: list[str] = []
    for result in results:
        if result.ok:
            continue
        if result.name == "pylint-duplicate-code":
            guidance.append("Consolidate duplicated control flow or payload assembly into one helper before adding features.")
        elif result.name == "vulture-dead-code":
            guidance.append("Remove unused functions, imports, and stale branches before expanding adjacent code.")
        elif result.name == "radon-complexity":
            guidance.append(f"Break functions above complexity {max_complexity} into smaller units with single responsibilities.")
        elif result.name == "radon-maintainability":
            guidance.append("Refactor low-maintainability modules before layering new behavior onto them.")
        elif result.name.startswith("ruff"):
            guidance.append("Apply style and formatting fixes first so later structural findings are easier to read.")
    return guidance


def emit_text(results: Sequence[CheckResult], guidance: Sequence[str]) -> None:
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {format_command(result.command)}")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print()

    if guidance:
        print("Agent guidance:")
        for item in guidance:
            print(f"- {item}")


def emit_markdown(results: Sequence[CheckResult], guidance: Sequence[str]) -> None:
    print("# Code Quality Report")
    print()
    print("| Check | Status | Command |")
    print("| --- | --- | --- |")
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"| {result.name} | {status} | `{format_command(result.command)}` |")
    print()
    for result in results:
        if result.stdout:
            print(f"## {result.name}")
            print()
            print("```text")
            print(result.stdout)
            print("```")
            print()
        if result.stderr:
            print(f"## {result.name} stderr")
            print()
            print("```text")
            print(result.stderr)
            print("```")
            print()
    if guidance:
        print("## Agent Guidance")
        print()
        for item in guidance:
            print(f"- {item}")


def emit_json(results: Sequence[CheckResult], guidance: Sequence[str]) -> None:
    payload = {
        "ok": all(result.ok for result in results),
        "results": [
            {
                "name": result.name,
                "ok": result.ok,
                "command": result.command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            for result in results
        ],
        "agent_guidance": list(guidance),
    }
    print(json.dumps(payload, indent=2))


def main() -> int:
    args = parse_args()
    paths = existing_paths(args.paths)

    checks: list[tuple[str, list[str]]] = [("ruff-check", ruff_command(paths, fix=args.ruff_fix))]
    if args.ruff_fix:
        checks.append(("ruff-format", ruff_format_command(paths, check_only=False)))
    if not args.ruff_only:
        checks.extend(
            [
                (
                    "pylint-duplicate-code",
                    pylint_command(paths, min_similarity_lines=args.min_similarity_lines),
                ),
                (
                    "vulture-dead-code",
                    vulture_command(paths, min_confidence=args.min_vulture_confidence),
                ),
                (
                    "radon-complexity",
                    radon_cc_command(paths, max_complexity=args.max_complexity),
                ),
                (
                    "radon-maintainability",
                    radon_mi_command(paths, min_maintainability=args.min_maintainability),
                ),
            ]
        )
    missing = missing_tools([command for _, command in checks])
    if missing:
        print(
            "Missing required tools: " + ", ".join(missing) + ". Install the dev dependencies first.",
            file=sys.stderr,
        )
        return 2

    results = [run_check(name, command) for name, command in checks]
    guidance = summarize_agent_guidance(results, max_complexity=args.max_complexity)

    if args.format == "json":
        emit_json(results, guidance)
    elif args.format == "markdown":
        emit_markdown(results, guidance)
    else:
        emit_text(results, guidance)

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
