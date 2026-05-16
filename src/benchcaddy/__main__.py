"""Module entrypoint for ``python -m benchcaddy``.

This module should encapsulate only the thin handoff from Python's module
execution mechanism to the CLI application. Argument parsing and command
behavior should remain in the CLI module rather than being implemented
here.
"""

from .cli import main

if __name__ == "__main__":
    main()
