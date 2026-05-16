"""Importable benchmark targets used by subprocess isolation tests."""

from __future__ import annotations

from benchcaddy.isolation.observability import observe

_warmup_counter = 0


def nested_observed_target(value: int) -> int:
    @observe("time")
    def inner_time(number: int) -> int:
        return number + 1

    @observe("return")
    def inner_return(number: int) -> int:
        return number * 2

    @observe("time", "return")
    def inner_both(number: int) -> int:
        return inner_return(inner_time(number))

    return inner_both(value)


def reset_warmup_sensitive_state() -> None:
    global _warmup_counter
    _warmup_counter = 0


def warmup_sensitive_observed_target() -> int:
    global _warmup_counter
    _warmup_counter += 1

    @observe("return")
    def inner_return() -> int:
        return _warmup_counter

    return inner_return()


def unsupported_nested_return_target() -> str:
    @observe("time", "return")
    def inner_both() -> object:
        return object()

    inner_both()
    return "done"