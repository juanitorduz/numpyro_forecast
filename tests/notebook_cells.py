"""Read the committed example notebooks for the notebook checks under ``tests/``.

The docs build renders the notebooks with their stored outputs and never executes them, so
the notebook tests read the JSON only. This module holds the small readers that they share,
imported the way the tests import ``example_models``.
"""

import json
import re
from pathlib import Path
from typing import Any

EXAMPLES = Path(__file__).resolve().parent.parent / "docs" / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))

type Cell = dict[str, Any]


def read_cells(notebook: str | Path) -> list[Cell]:
    """Return the cells of a committed notebook, given by name (``"tsb"``) or by path."""
    path = EXAMPLES / f"{notebook}.ipynb" if isinstance(notebook, str) else notebook
    return json.loads(path.read_text())["cells"]


def source(cell: Cell) -> str:
    """Return the source of one cell as a single string."""
    text = cell.get("source", "")
    return "".join(text) if isinstance(text, list) else text


def markdown(cells: list[Cell]) -> str:
    """Return the sources of all markdown cells, separated by blank lines."""
    return "\n\n".join(source(cell) for cell in cells if cell["cell_type"] == "markdown")


def code(cells: list[Cell]) -> str:
    """Return the sources of all code cells, separated by blank lines."""
    return "\n\n".join(source(cell) for cell in cells if cell["cell_type"] == "code")


def outputs(cells: list[Cell]) -> str:
    """Return the stored stream text and ``text/plain`` outputs of all code cells."""
    return "\n".join(
        "".join(output.get("text", output.get("data", {}).get("text/plain", "")))
        for cell in cells
        if cell["cell_type"] == "code"
        for output in cell.get("outputs", [])
    )


def stderr(cells: list[Cell]) -> list[tuple[int, str]]:
    """Return ``(cell index, text)`` for every stored stderr stream output."""
    return [
        (index, "".join(output["text"]))
        for index, cell in enumerate(cells)
        if cell["cell_type"] == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream" and output.get("name") == "stderr"
    ]


def search(pattern: str, text: str, flags: int = 0) -> re.Match[str]:
    """Return the first match of ``pattern`` in ``text``; fail the test if there is none."""
    match = re.search(pattern, text, flags)
    assert match is not None, f"no match for {pattern!r}; update the notebook test"
    return match
