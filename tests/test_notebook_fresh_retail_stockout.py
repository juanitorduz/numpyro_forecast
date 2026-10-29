"""Check the ``fresh_retail_stockout`` notebook prose and its reusable backtest closure.

The notebook needs the FreshRetailNet-50K download, so the suite never runs it. These checks
read the committed JSON: the prose must say which window the prior-informing statistics use,
and the closure that the notebook offers for rolling-origin backtesting must run on its own
(no notebook globals) and survive a series with no sales in its training window.
"""

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random
from jaxtyping import Float
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from numpyro_forecast import Horizon, draw_posterior, forecast, predict
from numpyro_forecast.typing import Array, ForecastModel

NOTEBOOK = (
    Path(__file__).resolve().parent.parent / "docs" / "examples" / "fresh_retail_stockout.ipynb"
)


@pytest.fixture(scope="module")
def cells() -> list[dict[str, Any]]:
    """The cells of the committed notebook."""
    return json.loads(NOTEBOOK.read_text())["cells"]


def _source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _markdown(cells: list[dict[str, Any]]) -> str:
    return "\n\n".join(_source(cell) for cell in cells if cell["cell_type"] == "markdown")


def test_floor_prose_names_its_window(cells: list[dict[str, Any]]) -> None:
    """The empirical floor is computed on all 90 days, so the prose that uses it says so."""
    paragraphs = [p for p in _markdown(cells).splitlines() if "empirical floor" in p]
    assert paragraphs, "no markdown paragraph mentions the empirical floor"
    missing = [p[:80] for p in paragraphs if "test window" not in p]
    assert not missing, f"paragraphs that use the floor without naming its window: {missing}"


def test_no_claim_that_svi_needs_optax_to_numpyro(cells: list[dict[str, Any]]) -> None:
    """``numpyro.infer.SVI`` wraps an optax ``GradientTransformation`` itself."""
    assert "expects a NumPyro optimizer" not in _markdown(cells)


def _toy_model(covariates: Array, data: Array | None = None) -> None:
    h = Horizon.from_data(covariates, data)
    level = jnp.asarray(numpyro.sample("level", dist.Normal(1.0, 1.0).expand([3]).to_event(1)))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    predict(h, dist.Normal(0.0, sigma), jnp.broadcast_to(level, (h.duration, 3)))


def test_backtest_closure_is_self_contained_and_guards_a_zero_scale(
    cells: list[dict[str, Any]],
) -> None:
    """The closure factory runs without notebook globals and keeps a zero-sales series finite."""
    optax = pytest.importorskip("optax")
    from numpyro.optim import optax_to_numpyro

    (source,) = [
        _source(cell)
        for cell in cells
        if cell["cell_type"] == "code" and "def scaled_forecast_fn(" in _source(cell)
    ]
    # Only the names that the notebook's setup cell imports; no fit settings.
    namespace: dict[str, Any] = {
        "Array": Array,
        "AutoNormal": AutoNormal,
        "Float": Float,
        "ForecastModel": ForecastModel,
        "SVI": SVI,
        "Trace_ELBO": Trace_ELBO,
        "draw_posterior": draw_posterior,
        "forecast": forecast,
        "jnp": jnp,
        "np": np,
        "optax": optax,
        "optax_to_numpyro": optax_to_numpyro,
        "random": random,
    }
    exec(source, namespace)  # noqa: S102 - a code cell of our own committed notebook
    # The factory takes the SVI settings; a bare closure would have to read them as globals
    # and fails here with a NameError.
    factory = namespace.get("make_scaled_forecast_fn")
    scaled_forecast_fn = (
        namespace["scaled_forecast_fn"] if factory is None else factory(optax.adam(0.01), 200)
    )

    sales = jnp.abs(random.normal(random.PRNGKey(0), (24, 3))) + 1.0
    sales = sales.at[:20, 2].set(0.0)  # series 2: no sales in the 20-day training window
    covariates = jnp.zeros((1, 24, 3))
    pred = scaled_forecast_fn(
        random.PRNGKey(1), _toy_model, sales[:20], covariates[:, :20], covariates, 50
    )
    assert pred.shape == (50, 4, 3)
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(pred))))
