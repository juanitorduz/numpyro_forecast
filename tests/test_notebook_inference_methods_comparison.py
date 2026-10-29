"""Check the NUTS diagnostics prose of ``inference_methods_comparison`` against its summary.

The docs build renders the committed notebook as is, so the sentences that read the stored
``az.summary`` table (the mixing claim, the low effective sample sizes and the interval
columns) must agree with it. These checks read the notebook JSON only; nothing is executed.
The rules for every notebook (percentages in LaTeX, the optax wording) are in
``tests/test_notebook_hygiene.py``.
"""

import re

import pytest
from notebook_cells import Cell, markdown, read_cells, source

SCALARS = ("bias", "drift_scale", "nu", "sigma")
# Vehtari et al. (2021): with four chains, trust a summary when R-hat < 1.01 and the bulk
# effective sample size is above 400.
R_HAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400


@pytest.fixture(scope="module")
def cells() -> list[Cell]:
    """The cells of the committed notebook."""
    return read_cells("inference_methods_comparison")


def _summary_table(cells: list[Cell]) -> str:
    """The stored ``text/plain`` output of the ``az.summary`` cell."""
    for cell in cells:
        if cell["cell_type"] == "code" and "az.summary(" in source(cell):
            return "".join(
                "".join(output["data"]["text/plain"])
                for output in cell["outputs"]
                if "text/plain" in output.get("data", {})
            )
    raise AssertionError("no az.summary cell in the notebook")


def _summary_column(cells: list[Cell], name: str) -> dict[str, str]:
    """One column of the first block of the stored summary, keyed by variable name."""
    # pandas wraps the wide table: the first block, up to the first blank line, has the
    # r_hat and ess_bulk columns.
    header, *lines = _summary_table(cells).split("\n\n")[0].splitlines()
    column = header.split().index(name) + 1  # each row starts with the variable name
    return {line.split()[0]: line.split()[column] for line in lines}


def test_mixing_claim_matches_the_stored_summary(cells: list[Cell]) -> None:
    """The prose quotes the stored R-hat range and says "mix well" only if it allows it."""
    column = _summary_column(cells, "r_hat")
    r_hat = {name: float(column[name]) for name in SCALARS}
    text = markdown(cells)
    quoted = {f"`{value:.2f}`" for value in (min(r_hat.values()), max(r_hat.values()))}
    assert all(value in text for value in quoted), (
        f"the prose does not quote the stored R-hat range {sorted(quoted)}"
    )
    if max(r_hat.values()) >= R_HAT_THRESHOLD:
        assert "mix well" not in text, (
            f"the stored R-hat values {r_hat} do not support the claim that they mix well"
        )


def test_low_bulk_ess_is_named_in_the_prose(cells: list[Cell]) -> None:
    """Every forecast scalar whose stored bulk ESS is below the threshold is quoted."""
    column = _summary_column(cells, "ess_bulk")
    low = {name: column[name] for name in SCALARS if int(column[name]) < ESS_THRESHOLD}
    text = markdown(cells)
    missing = {name: ess for name, ess in low.items() if f"`{ess}`" not in text}
    assert not missing, f"the prose does not quote the low bulk ESS of {missing}"


def test_prose_names_the_interval_of_the_summary(cells: list[Cell]) -> None:
    """The prose names the interval columns of the stored summary and their level."""
    header = _summary_table(cells).splitlines()[0]
    match = re.search(r"\b(eti|hdi)(\d+)_lb\b", header)
    assert match is not None, header
    kind, level = match.groups()
    text = markdown(cells)
    name = "equal-tailed" if kind == "eti" else "HDI"
    column = match.group(0)
    assert f"`{column}`" in text and f"${level}\\%$ {name}" in text, (
        f"the summary prints {column}; name that column and the ${level}\\%$ {name} in the prose"
    )
