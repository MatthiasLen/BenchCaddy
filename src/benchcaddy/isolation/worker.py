"""Dedicated worker bootstrap for isolated benchmark execution.

This file is executed by exact path under Python's isolated mode so the
worker entrypoint comes from the same BenchCaddy codebase as the parent.
That avoids ambient module discovery for the worker itself while still
allowing the child to replay the parent's import roots for the target.
"""

from __future__ import annotations

import sys
from contextlib import suppress
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Insert this package root ahead of import resolution and delegate to the worker entrypoint."""
    package_root = str(Path(__file__).resolve().parents[2])
    sys.path = [package_root, *[entry for entry in sys.path if entry != package_root]]

    # Defensively reconfigure stdout/stderr to UTF-8 with replacement error handling
    # so that benchmark targets with Unicode output don't fail on Windows charmap streams.
    for stream in (sys.stdout, sys.stderr):
        with suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")

    from benchcaddy.isolation.process import _main

    return _main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
