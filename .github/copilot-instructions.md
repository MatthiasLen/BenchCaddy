# Output rules
- Code only. No explanation unless I explicitly ask!
- No markdown preamble.
- No "Here is you code:" intro.
- No closing summaries!
- No "Changes made" summaries!
- No "validation performed" summaries!

# Code Style
- ALWAYS write professional, production-quality, well-engineered code.
- Prefer lean code and clear local flow over extra abstraction.
- Avoid deep call stacks, indirection layers and trivial wrappers.
- Avoid temporary dataclasses or helper objects used only to pass values to a single consumer.
- Keep changes minimal, targeted, and scoped to the issue.
- Avoid hacks, technical debt, and clarity tradeoffs.

# Comments and docstrings
- Add concise comments for non-obvious logic, decisions, or policies.
- Preserve existing comments unless incorrect; update rather than remove.

# Architecture
- Organize code into modules with clear separation of concerns.
- Optimize for maintainability; avoid over-engineering and premature abstraction.
- Extract helpers only when they are reused, encapsulate meaningful policy, improve testability, or hide non-trivial complexity.
