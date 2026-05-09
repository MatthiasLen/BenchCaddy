from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

Column = str | tuple[str, str]
Row = Sequence[object]


def dump_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(value, indent=indent, sort_keys=True)


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


def summary_panel(title: str, rows: Sequence[tuple[str, str]]) -> Panel:
    summary = Table.grid(padding=(0, 2))
    for label, value in rows:
        summary.add_row(label, value)
    return Panel.fit(summary, title=title)