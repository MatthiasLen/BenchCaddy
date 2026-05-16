# BenchCaddy Project Guidelines

## Code Style

- Prefer lean Python code with clear local flow over extra abstraction.
- Do not introduce private helper functions that have one call site and add little semantic value.
- Do not add one-line wrappers around stdlib or NumPy calls unless they enforce reusable policy used in multiple places.
- Avoid temporary internal dataclasses, tuples, or helper objects whose only job is to shuttle a few computed values to one consumer.
- Keep changes minimal and targeted. 

## Architecture

- The code should be well organized into modules that separate concerns.
- Maintainability is key. Avoid over-engineering or premature abstraction.
- Extract a helper only when at least one of these is true:
  - the logic is reused
  - the logic carries real policy that benefits from a name
  - the logic materially improves testability
  - the logic hides non-trivial complexity
