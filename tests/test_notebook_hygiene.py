"""Check the stored outputs and the cross links of the example notebooks.

The docs build renders the committed notebooks with their stored outputs (no re-execution),
so a warning from the author's environment, with its local ``site-packages`` path, shows up
on the published page. The site is also versioned (the stable root and ``/v/dev/``), so a
link to another example or to the API reference must be relative: an absolute link to the
site always lands on the stable root, whose reference is pruned to the last release.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "docs" / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))
SITE = "https://juanitorduz.github.io/numpyro_forecast/"
# Warnings that describe the author's environment, not the example.
ENVIRONMENT_WARNINGS = (
    "PyMC not installed",
    "When multiple credible intervals are plotted",
)
# A relative link from an example page to another example or to a reference page.
RELATIVE_LINK = re.compile(r"\]\((?!https?://)(?P<path>[\w./]+\.html)(?:#[\w-]+)?\)")


def _cells(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())["cells"]


def _source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _stderr(cells: list[dict[str, Any]]) -> list[tuple[int, str]]:
    return [
        (index, "".join(output["text"]))
        for index, cell in enumerate(cells)
        if cell["cell_type"] == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream" and output.get("name") == "stderr"
    ]


def _reference_entries() -> set[str]:
    """The page stems of the API reference: every entry under ``reference:``."""
    config = yaml.safe_load((REPO / "great-docs.yml").read_text())
    return {
        entry
        for section in config["reference"]
        for entry in section["contents"]
        if isinstance(entry, str)
    }


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[path.stem for path in NOTEBOOKS])
def test_no_stored_environment_warnings(path: Path) -> None:
    """No stored stderr output repeats an environment warning or a ``site-packages`` path."""
    found = [
        f"cell {index}: {text.splitlines()[0][:100]}"
        for index, text in _stderr(_cells(path))
        if "site-packages/" in text or any(warning in text for warning in ENVIRONMENT_WARNINGS)
    ]
    assert not found, (
        f"{path.name} stores environment warnings; filter them in the setup cell:\n"
        + "\n".join(found)
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[path.stem for path in NOTEBOOKS])
def test_links_to_the_docs_site_are_relative(path: Path) -> None:
    """Markdown cells link to other examples and to the reference with relative paths."""
    found = [
        f"cell {index}: {link}"
        for index, cell in enumerate(_cells(path))
        if cell["cell_type"] == "markdown"
        for link in re.findall(re.escape(SITE) + r"\S*?(?=[)\s])", _source(cell))
    ]
    assert not found, (
        f"{path.name} has absolute links to the docs site; use `<example>.html` or "
        "`../../reference/<entry>.html`:\n" + "\n".join(found)
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[path.stem for path in NOTEBOOKS])
def test_relative_links_point_at_existing_pages(path: Path) -> None:
    """Every relative link names an example notebook or an entry of the API reference."""
    examples = {notebook.stem for notebook in NOTEBOOKS}
    reference = _reference_entries()
    broken = []
    for index, cell in enumerate(_cells(path)):
        if cell["cell_type"] != "markdown":
            continue
        for match in RELATIVE_LINK.finditer(_source(cell)):
            link = match.group("path")
            if link.startswith("../../reference/"):
                ok = link.removeprefix("../../reference/").removesuffix(".html") in reference
            else:
                ok = "/" not in link and link.removesuffix(".html") in examples
            if not ok:
                broken.append(f"cell {index}: {link}")
    assert not broken, f"{path.name} links to pages that do not exist:\n" + "\n".join(broken)
