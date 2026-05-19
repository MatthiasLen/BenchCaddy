from __future__ import annotations

from rich.text import Text


def _styled(value: object, style: str | None = None) -> Text:
    return Text(str(value), style=style)


def _style_row(values: tuple[object, ...], style: str | None = None) -> tuple[object, ...]:
    return tuple(_styled(value, style) if style else value for value in values)


def _format_optional_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def _best_run(runs: list[dict[str, object]]) -> dict[str, object]:
    return min(runs, key=lambda candidate: (candidate["median_seconds"], candidate["id"]))


def _row_style(
    runs: list[dict[str, object]],
    run: dict[str, object],
    *,
    basis_run: dict[str, object] | None,
    highlight_basis: bool,
) -> str | None:
    if basis_run is None:
        return None

    if run["id"] == _best_run(runs)["id"]:
        return "green"

    if highlight_basis and run["id"] == basis_run["id"]:
        return "yellow"

    return None
