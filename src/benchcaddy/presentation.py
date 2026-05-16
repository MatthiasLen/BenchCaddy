"""Presentation helpers for BenchCaddy terminal output.

This module should encapsulate formatting and rendering primitives used
by the CLI and reporters, including textual summaries, JSON rendering,
and Rich table or panel construction. Business logic and data retrieval
should stay out of this layer so presentation remains reusable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from numbers import Real
from pathlib import Path

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

Column = str | tuple[str, str]
Row = Sequence[object]


def dump_json(value: object, *, indent: int | None = 0) -> str:
    n_ind = 0 if indent is None else indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, nested_value in value.items():
            prefix = n_ind * " " + f"{key}: "
            if isinstance(nested_value, dict):
                lines.append(prefix + "\n" + dump_json(nested_value, indent=n_ind + 2))
            else:
                lines.append(prefix + str(nested_value))
        return "\n".join(lines)
    return str(value)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def serialize_json(value: object, *, indent: int | None = 2) -> str:
    return json.dumps(value, indent=indent, default=_json_default, sort_keys=True)


def format_scientific_number(value: Real) -> str:
    return f"{float(value):.6e}"


def truncate_table_value(value: str, *, max_length: int = 22) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 3]}..."


def format_time_summary(mean_seconds: float | None, std_seconds: float | None) -> str:
    mean_value = 0.0 if mean_seconds is None else mean_seconds
    std_value = 0.0 if std_seconds is None else std_seconds
    return f"{mean_value:.6f} +- {std_value:.6f}"


def format_interval(lower_seconds: float | None, upper_seconds: float | None) -> str:
    if lower_seconds is None or upper_seconds is None:
        return "-"
    return f"[{lower_seconds:.6f}, {upper_seconds:.6f}]"


def format_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value * 100.0:.2f}%"


def format_probability(value: float | None) -> str:
    return "-" if value is None else f"{value * 100.0:.1f}%"


def format_warning_list(value: list[str] | tuple[str, ...] | None) -> str:
    if not value:
        return "-"
    return ", ".join(str(item).replace("_", " ") for item in value)


def format_return_value(value: object, *, compact: bool = False) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return truncate_table_value(value) if compact else value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Real):
        return format_scientific_number(value)
    if isinstance(value, list):
        formatted_items = [format_return_value(item) for item in value]
        if not compact:
            return f"[{', '.join(formatted_items)}]"
        if not formatted_items:
            return "[]"
        if len(formatted_items) == 1:
            return f"[{formatted_items[0]}]"
        return f"[{formatted_items[0]}, ...]"
    return str(value)


def format_return_error(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "equal" if value else "different"
    if isinstance(value, Real):
        return f"{float(value) * 100.0:.6f}%"
    return str(value)


def render_table(title: str, columns: Sequence[Column], rows: Iterable[Row]) -> Table:
    table = Table(title=title)
    for column in columns:
        if isinstance(column, tuple):
            table.add_column(column[0], justify=column[1])
        else:
            table.add_column(column)
    for row in rows:
        table.add_row(*(value if isinstance(value, Text) else str(value) for value in row))
    return table


def json_panel(title: str, value: object, *, indent: int | None = None, fit: bool = False) -> Panel:
    return (Panel.fit if fit else Panel)(dump_json(value, indent=indent), title=title)


def summary_panel(title: str, rows: Sequence[tuple[str, object]]) -> Panel:
    summary = Table.grid(padding=(0, 2))
    for label, value in rows:
        summary.add_row(label, value if isinstance(value, Text) else str(value))
    return Panel.fit(summary, title=title)
