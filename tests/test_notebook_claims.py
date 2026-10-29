"""Check small numeric claims of the example notebooks against their outputs and code.

The docs build renders the committed notebooks as they are, so a number in the prose that
disagrees with a stored output or with the stated math reaches the published page. These
checks read the notebook JSON only (nothing is executed) and recompute each quantity from the
stored outputs or from the parameters in the code cells.
"""

import math
import re

import pytest
from notebook_cells import code, markdown, outputs, read_cells, search

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _beta_cdf(x: float, a: int, b: int) -> float:
    """CDF of a Beta distribution with integer shapes, written as a binomial tail."""
    n = a + b - 1
    return sum(math.comb(n, j) * x**j * (1.0 - x) ** (n - j) for j in range(a, n + 1))


def test_censored_demand_mae_gap_matches_the_table() -> None:
    """The MAE gap quoted in the prose is the gap in the stored metrics table."""
    cells = read_cells("censored_demand")
    maes = [float(x) for x in re.findall(r"latent demand\s+([\d.]+)", outputs(cells))[:2]]
    stored = round(abs(maes[1] - maes[0]) * 1_000)
    match = search(r"the two MAEs differ by (?:about )?(\w+) thousandths?", markdown(cells))
    assert NUMBER_WORDS[match.group(1)] == stored, (
        f"the prose says {match.group(1)} thousandths, the stored MAEs {maes} differ by {stored}"
    )


def test_censored_demand_expected_sales_gap_matches_the_process() -> None:
    r"""The expected gap between demand and sales is not the friction $\delta$ alone.

    With $s_t = \max(0, \min(d_t + \varepsilon^s_t - \delta, d_t))$ and demand well above
    zero, $d_t - s_t = \max(\delta - \varepsilon^s_t, 0)$, whose mean is
    $\delta \, \Phi(\delta / \sigma_s) + \sigma_s \, \varphi(\delta / \sigma_s)$.
    """
    cells = read_cells("censored_demand")
    code_text = code(cells)
    delta = float(search(r"demand_sales_delta: float = ([\d.]+)", code_text).group(1))
    sigma = float(search(r"sales_noise: float = ([\d.]+)", code_text).group(1))
    expected_gap = delta * _normal_cdf(delta / sigma) + sigma * _normal_pdf(delta / sigma)
    match = search(r"sits below demand by about \$([\d.]+)\$ in expectation", markdown(cells))
    assert float(match.group(1)) == pytest.approx(expected_gap, abs=0.01)


@pytest.mark.parametrize("name", ["croston", "tsb"])
def test_beta_prior_masses_match_the_prior(name: str) -> None:
    r"""The quoted masses of the smoothing prior below $0.1$ and $0.3$ match the prior."""
    cells = read_cells(name)
    pattern = r"dist\.Beta\(concentration1=(\d+), concentration0=(\d+)\)"
    a, b = (int(x) for x in search(pattern, code(cells)).groups())
    text = markdown(cells)
    for upper in (0.1, 0.3):
        mass = 100 * _beta_cdf(upper, a, b)
        assert f"${mass:.1f}\\%$" in text, (
            f"the prose does not quote the {mass:.1f}% of the prior below {upper}"
        )


def test_exponential_smoothing_period_matches_the_grid() -> None:
    """The cosine has ``(n + 1) * n - 1`` steps over ``n + 1`` cycles, not exactly ``n``."""
    cells = read_cells("exponential_smoothing_state_space")
    code_text = code(cells)
    n_seasons = int(search(r"^n_seasons = (\d+)$", code_text, re.MULTILINE).group(1))
    assert "t = jnp.linspace(0, n_seasons + 1, (n_seasons + 1) * n_seasons)" in code_text
    samples_per_cycle = ((n_seasons + 1) * n_seasons - 1) / (n_seasons + 1)
    assert f"{samples_per_cycle:.2f}" in markdown(cells)


def test_dynestyx_coverage_matches_the_table() -> None:
    r"""The prose quotes the stored $50\%$ coverage, which is the same for the three strategies."""
    cells = read_cells("dynestyx_integration")
    coverage_50 = {
        x
        for line in outputs(cells).splitlines()
        if line.startswith("coverage (50%)")
        for x in line.split()[2:]
    }
    assert len(coverage_50) == 1, f"the strategies store different 50% coverages: {coverage_50}"
    assert f"(${coverage_50.pop()}$)" in markdown(cells)
