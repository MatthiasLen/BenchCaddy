# BenchCaddy Project Guidelines

## Code Style

- Prefer lean Python code with clear local flow over extra abstraction.
- Avoid deep call stacks and multiple layers of indirection.
- Do not introduce helper functions that have one call site and add little semantic value. Do not add "shallow" few-line wrappers.
- Avoid temporary internal dataclasses, tuples, or helper objects whose only job is to shuttle a few computed values to one consumer.
- Keep changes minimal and targeted. 

## Comments and docstrings
- Add compact comments to clarify non-obvious code, especially if it contains non-trivial logic or policy.
- IMPORTANT: Don't remove existing inline comments unless they are incorrect! Check them for accuracy and update them if needed!

## Architecture

- The code should be well organized into modules that separate concerns.
- Maintainability is key. Avoid over-engineering or premature abstraction.
- Extract a helper only when at least one of these is true:
  - the logic is reused
  - the logic carries real policy that benefits from a name
  - the logic materially improves testability
  - the logic hides non-trivial complexity

## After completing implementation

- Check the implementation for any violations of the above guidelines and refactor all found violations as needed.
- Iterate checking and refactoring until no meaningful violations remain.
