"""Importable benchmark targets used by subprocess isolation tests."""

from __future__ import annotations

from benchcaddy.isolation.observability import observe

_warmup_counter = 0


@observe("time")
def module_time_helper(value: int) -> int:
    return value + 1


@observe("return")
def module_return_helper(value: int) -> int:
    return value * 2


class ObservableService:
    @staticmethod
    @observe("time")
    def static_time_helper(value: int) -> int:
        return value + 3

    @classmethod
    @observe("return")
    def class_return_helper(cls, value: int) -> int:
        del cls
        return value + 5

    @observe("time", "return")
    def instance_both_helper(self, value: int) -> int:
        del self
        return value + 7


class CallableTarget:
    def __call__(self) -> int:
        return 11


def top_level_module_target(value: int) -> int:
    return module_return_helper(module_time_helper(value))


def realistic_observed_workflow(value: int) -> int:
    service = ObservableService()
    step_one = module_time_helper(value)
    step_two = ObservableService.static_time_helper(step_one)
    step_three = ObservableService.class_return_helper(step_two)
    return service.instance_both_helper(step_three)


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
