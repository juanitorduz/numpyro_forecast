"""Check the prose of the ``inference_methods_comparison`` notebook against the facts.

The docs build renders the committed notebook as is, so a sentence that describes package
or NumPyro behavior that does not exist, or that disagrees with a stored output, reaches the
published page. These checks read the notebook JSON only; nothing is executed.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

NOTEBOOK = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "examples"
    / "inference_methods_comparison.ipynb"
)
SCALARS = ("bias", "drift_scale", "nu", "sigma")
# Vehtari et al. (2021): with four chains, trust a summary when R-hat < 1.01.
R_HAT_THRESHOLD = 1.01


@pytest.fixture(scope="module")
def cells() -> list[dict[str, Any]]:
    """The cells of the committed notebook."""
    return json.loads(NOTEBOOK.read_text())["cells"]


def _source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _markdown(cells: list[dict[str, Any]]) -> str:
    return "\n\n".join(_source(cell) for cell in cells if cell["cell_type"] == "markdown")


def _summary_table(cells: list[dict[str, Any]]) -> str:
    """The stored ``text/plain`` output of the ``az.summary`` cell."""
    for cell in cells:
        if cell["cell_type"] == "code" and "az.summary(" in _source(cell):
            return "".join(
                "".join(output["data"]["text/plain"])
                for output in cell["outputs"]
                if "text/plain" in output.get("data", {})
            )
    raise AssertionError("no az.summary cell in the notebook")


def test_no_claim_that_the_mclmc_adapter_warns(cells: list[dict[str, Any]]) -> None:
    """The BlackJAX adapters document ``num_warmup=0``; they emit no warning for it."""
    assert "adapter warns" not in _markdown(cells)


def test_no_claim_that_svi_needs_optax_to_numpyro(cells: list[dict[str, Any]]) -> None:
    """``numpyro.infer.SVI`` wraps an optax ``GradientTransformation`` itself."""
    markdown = _markdown(cells)
    assert "expects a NumPyro optimizer" not in markdown
    assert "needs one line of glue" not in markdown


def test_mixing_claim_matches_the_stored_summary(cells: list[dict[str, Any]]) -> None:
    """The prose says the four scalars "mix well" only if their stored R-hat allows it."""
    # pandas wraps the wide table: the first block, up to the first blank line, has r_hat.
    header, *lines = _summary_table(cells).split("\n\n")[0].splitlines()
    column = header.split().index("r_hat") + 1  # each row starts with the variable name
    rows = {line.split()[0]: line.split() for line in lines}
    r_hat = {name: float(rows[name][column]) for name in SCALARS}
    if max(r_hat.values()) >= R_HAT_THRESHOLD:
        assert "mix well" not in _markdown(cells), (
            f"the stored R-hat values {r_hat} do not support the claim that they mix well"
        )


def test_prose_names_the_interval_of_the_summary(cells: list[dict[str, Any]]) -> None:
    """The stored summary prints ArviZ's default interval; the prose must say which one."""
    header = _summary_table(cells).splitlines()[0]
    match = re.search(r"\b(eti|hdi)(\d+)_lb\b", header)
    assert match is not None, header
    kind, level = match.groups()
    markdown = _markdown(cells)
    name = "equal-tailed" if kind == "eti" else "HDI"
    assert f"${level}\\%$ {name}" in markdown, (
        f"the summary prints a {level}% {kind.upper()}; say so in the prose, "
        f"for example ${level}\\%$ {name}"
    )


def test_percentages_in_prose_use_latex(cells: list[dict[str, Any]]) -> None:
    r"""Prose writes percentages as LaTeX (``$30\%$``), not as code spans or plain text."""
    markdown = re.sub(r"\$\$.*?\$\$", "", _markdown(cells), flags=re.DOTALL)
    markdown = re.sub(r"\$[^$\n]*\$", "", markdown)
    assert not re.findall(r"\d%", markdown)
