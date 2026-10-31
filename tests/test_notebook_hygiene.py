"""Check the rules that hold for every example notebook: outputs, links and prose style.

The docs build renders the committed notebooks with their stored outputs (no re-execution),
so a warning from the author's environment, with its local ``site-packages`` path, would show
up on the published site. The site is also versioned (the stable root and ``/v/dev/``), so a
link to another example or to the API reference must be relative: an absolute link to the
site always lands on the stable root, whose reference is pruned to the last release. The
prose rules (percentages in LaTeX, no claim that ``SVI`` needs ``optax_to_numpyro``) were
broken in more than one notebook, so they are checked here for all of them.
"""

import re
from pathlib import Path

import pytest
import yaml
from notebook_cells import NOTEBOOKS, read_cells, source, stderr

REPO = Path(__file__).resolve().parent.parent
SITE = "https://juanitorduz.github.io/numpyro_forecast/"
# Warnings that describe the author's environment, not the example.
ENVIRONMENT_WARNINGS = (
    "PyMC not installed",
    "When multiple credible intervals are plotted",
)
# A relative link from an example page to another example or to a reference page. Reference
# anchors contain dots (``#numpyro_forecast.models.ssoe``).
RELATIVE_LINK = re.compile(r"\]\((?!https?://)(?P<path>[\w./]+\.html)(?:#[^)\s]*)?\)")
NOTEBOOK_IDS = [path.stem for path in NOTEBOOKS]


def _reference_entries() -> set[str]:
    """The page stems of the API reference: every entry under ``reference:``."""
    config = yaml.safe_load((REPO / "great-docs.yml").read_text())
    return {
        entry if isinstance(entry, str) else entry["name"]
        for section in config["reference"]
        for entry in section["contents"]
    }


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_no_stored_environment_warnings(path: Path) -> None:
    """No stored stderr output repeats an environment warning or a ``site-packages`` path."""
    found = [
        f"cell {index}: {text.splitlines()[0][:100]}"
        for index, text in stderr(read_cells(path))
        if "site-packages/" in text or any(warning in text for warning in ENVIRONMENT_WARNINGS)
    ]
    assert not found, (
        f"{path.name} stores environment warnings; filter them in the setup cell:\n"
        + "\n".join(found)
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_links_to_the_docs_site_are_relative(path: Path) -> None:
    """Markdown cells link to other examples and to the reference with relative paths."""
    found = [
        f"cell {index}: {link}"
        for index, cell in enumerate(read_cells(path))
        if cell["cell_type"] == "markdown"
        for link in re.findall(re.escape(SITE) + r"\S*?(?=[)\s])", source(cell))
    ]
    assert not found, (
        f"{path.name} has absolute links to the docs site; use `<example>.html` or "
        "`../../reference/<entry>.html`:\n" + "\n".join(found)
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_relative_links_point_at_existing_pages(path: Path) -> None:
    """Every relative link names an example notebook or an entry of the API reference."""
    examples = {notebook.stem for notebook in NOTEBOOKS}
    reference = _reference_entries()
    broken = []
    for index, cell in enumerate(read_cells(path)):
        if cell["cell_type"] != "markdown":
            continue
        for match in RELATIVE_LINK.finditer(source(cell)):
            link = match.group("path")
            if link.startswith("../../reference/"):
                ok = link.removeprefix("../../reference/").removesuffix(".html") in reference
            else:
                ok = "/" not in link and link.removesuffix(".html") in examples
            if not ok:
                broken.append(f"cell {index}: {link}")
    assert not broken, f"{path.name} links to pages that do not exist:\n" + "\n".join(broken)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_percentages_in_prose_use_latex(path: Path) -> None:
    r"""Markdown cells write percentages as LaTeX (``$30\%$``), not as code spans or plain text."""
    found = []
    for index, cell in enumerate(read_cells(path)):
        if cell["cell_type"] != "markdown":
            continue
        text = re.sub(r"\$\$.*?\$\$", "", source(cell), flags=re.DOTALL)
        text = re.sub(r"\$[^$\n]*\$", "", text)
        found += [f"cell {index}: {match}" for match in re.findall(r"\S*\d%\S*", text)]
    assert not found, f"{path.name} writes percentages outside LaTeX:\n" + "\n".join(found)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_no_claim_that_svi_needs_optax_to_numpyro(path: Path) -> None:
    """``numpyro.infer.SVI`` wraps an optax ``GradientTransformation`` itself."""
    found = [
        f"cell {index}"
        for index, cell in enumerate(read_cells(path))
        if cell["cell_type"] == "markdown" and "expects a NumPyro optimizer" in source(cell)
    ]
    assert not found, f"{path.name} says that SVI needs optax_to_numpyro: {found}"
