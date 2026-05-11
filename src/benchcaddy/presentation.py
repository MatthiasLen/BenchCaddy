from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Real

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

Column = str | tuple[str, str]
Row = Sequence[object]


def dump_json(value: object, *, indent: int | None = 0) -> str:
    n_ind = 0 if indent is None else indent
    if isinstance(value, dict):
        return "\n".join(n_ind * " " + f"{k}: {('\\n' + dump_json(v, indent=n_ind + 2)) if isinstance(v, dict) else str(v)}" for k, v in value.items())
    return str(value)


def format_scientific_number(value: Real) -> str:
    return f"{float(value):.6e}"


def truncate_table_value(value: str, *, max_length: int = 22) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 3]}..."


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
