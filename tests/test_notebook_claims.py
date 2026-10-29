"""Check small numeric and wording claims of the example notebooks against their outputs.

The docs build renders the committed notebooks as they are, so a number in the prose that
disagrees with a stored output or with the stated math reaches the published page. These
checks read the notebook JSON only (nothing is executed) and recompute each quantity from the
stored outputs or from the parameters in the code cells.
"""

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "docs" / "examples"
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _cells(name: str) -> list[dict[str, Any]]:
    return json.loads((EXAMPLES / f"{name}.ipynb").read_text())["cells"]


def _source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _markdown(cells: list[dict[str, Any]]) -> str:
    return "\n\n".join(_source(cell) for cell in cells if cell["cell_type"] == "markdown")


def _code(cells: list[dict[str, Any]]) -> str:
    return "\n\n".join(_source(cell) for cell in cells if cell["cell_type"] == "code")


def _outputs(cells: list[dict[str, Any]]) -> str:
    return "\n".join(
        "".join(output.get("text", output.get("data", {}).get("text/plain", "")))
        for cell in cells
        if cell["cell_type"] == "code"
        for output in cell.get("outputs", [])
    )


def _search(pattern: str, text: str, flags: int = 0) -> re.Match[str]:
    match = re.search(pattern, text, flags)
    assert match is not None, f"no match for {pattern!r}; update this test"
    return match


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def test_censored_demand_mae_gap_matches_the_table() -> None:
    """The MAE gap quoted in the prose is the gap in the stored metrics table."""
    cells = _cells("censored_demand")
    maes = [float(x) for x in re.findall(r"latent demand\s+([\d.]+)", _outputs(cells))[:2]]
    stored = round(abs(maes[1] - maes[0]) * 1_000)
    match = re.search(r"the two MAEs differ by (?:about )?(\w+) thousandths?", _markdown(cells))
    assert match is not None, "the prose no longer quotes the MAE gap; update this test"
    assert NUMBER_WORDS[match.group(1)] == stored, (
        f"the prose says {match.group(1)} thousandths, the stored MAEs {maes} differ by {stored}"
    )


def test_censored_demand_expected_sales_gap_matches_the_process() -> None:
    r"""The expected gap between demand and sales is not the friction $\delta$ alone.

    With $s_t = \max(0, \min(d_t + \varepsilon^s_t - \delta, d_t))$ and demand well above
    zero, $d_t - s_t = \max(\delta - \varepsilon^s_t, 0)$, whose mean is
    $\delta \, \Phi(\delta / \sigma_s) + \sigma_s \, \varphi(\delta / \sigma_s)$.
    """
    cells = _cells("censored_demand")
    code = _code(cells)
    delta = float(_search(r"demand_sales_delta: float = ([\d.]+)", code).group(1))
    sigma = float(_search(r"sales_noise: float = ([\d.]+)", code).group(1))
    expected_gap = delta * _normal_cdf(delta / sigma) + sigma * _normal_pdf(delta / sigma)
    markdown = _markdown(cells)
    assert "sits below demand by the friction" not in markdown
    match = re.search(r"sits below demand by about \$([\d.]+)\$ in expectation", markdown)
    assert match is not None, "the prose must quote the expected gap between demand and sales"
    assert float(match.group(1)) == pytest.approx(expected_gap, abs=0.01)


@pytest.mark.parametrize("name", ["croston", "tsb"])
def test_beta_prior_is_not_described_as_the_classical_range(name: str) -> None:
    r"""$\text{Beta}(2, 20)$ puts most of its mass below $0.1$, not in $[0.1, 0.3]$."""
    markdown = _markdown(_cells(name))
    assert "Beta}(2, 20)" in markdown
    assert "consistent with the standard practice" not in markdown
    assert "concentrates on the same low-smoothing region" not in markdown


def test_tsb_first_fold_forecast_is_not_called_low() -> None:
    """At the first fold the TSB forecast has not decayed yet."""
    assert "now-low forecast" not in _markdown(_cells("tsb"))


def test_exponential_smoothing_period_matches_the_grid() -> None:
    """The cosine has ``(n + 1) * n - 1`` steps over ``n + 1`` cycles, not exactly ``n``."""
    cells = _cells("exponential_smoothing_state_space")
    code = _code(cells)
    n_seasons = int(_search(r"^n_seasons = (\d+)$", code, re.MULTILINE).group(1))
    assert "t = jnp.linspace(0, n_seasons + 1, (n_seasons + 1) * n_seasons)" in code
    samples_per_cycle = ((n_seasons + 1) * n_seasons - 1) / (n_seasons + 1)
    markdown = _markdown(cells)
    assert f"with period $m = {n_seasons}$" not in markdown
    assert f"{samples_per_cycle:.2f}" in markdown


def test_dynestyx_over_coverage_is_mentioned() -> None:
    r"""A stored $50\%$ coverage far above nominal needs a sentence in the prose."""
    cells = _cells("dynestyx_integration")
    coverage_50 = [
        float(x)
        for line in _outputs(cells).splitlines()
        if line.startswith("coverage (50%)")
        for x in line.split()[2:]
    ]
    assert coverage_50
    if max(coverage_50) > 0.75:
        assert "wider than nominal" in _markdown(cells), (
            f"the stored 50% coverage is {coverage_50}; say that the intervals over-cover"
        )
