# BenchCaddy Project Guidelines

## Code Style

- Prefer lean Python code with clear local flow over extra abstraction.
- Avoid deep call stacks and multiple layers of indirection.
- Do not introduce helper functions that have one call site and add little semantic value. Do not add "shallow" few-line wrappers.
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
