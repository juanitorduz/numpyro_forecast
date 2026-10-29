"""Check the ``fresh_retail_stockout`` notebook prose and its reusable backtest closure.

The notebook needs the FreshRetailNet-50K download, so the suite never runs it. These checks
read the committed JSON: the prose must say which window the prior-informing statistics use,
and the closure that the notebook offers for rolling-origin backtesting must run on its own
(no notebook globals) and survive a series with no sales in its training window. The optax
wording rule for every notebook is in ``tests/test_notebook_hygiene.py``.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random
from jaxtyping import Float
from notebook_cells import Cell, markdown, read_cells, source
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from numpyro_forecast import Horizon, draw_posterior, forecast, predict
from numpyro_forecast.typing import Array, ForecastFn, ForecastModel


@pytest.fixture(scope="module")
def cells() -> list[Cell]:
    """The cells of the committed notebook."""
    return read_cells("fresh_retail_stockout")


def test_floor_prose_names_its_window(cells: list[Cell]) -> None:
    """The prose that uses the empirical floor names the window it is computed on."""
    paragraphs = [p for p in markdown(cells).splitlines() if "empirical floor" in p]
    assert paragraphs, "no markdown paragraph mentions the empirical floor"
    missing = [p[:80] for p in paragraphs if not ("test window" in p or "training window" in p)]
    assert not missing, f"paragraphs that use the floor without naming its window: {missing}"


def _toy_model(covariates: Array, data: Array | None = None) -> None:
    h = Horizon.from_data(covariates, data)
    level = jnp.asarray(numpyro.sample("level", dist.Normal(1.0, 1.0).expand([3]).to_event(1)))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    predict(h, dist.Normal(0.0, sigma), jnp.broadcast_to(level, (h.duration, 3)))


def test_backtest_closure_is_self_contained_and_guards_a_zero_scale(cells: list[Cell]) -> None:
    """The closure factory runs without notebook globals and keeps a zero-sales series finite."""
    optax = pytest.importorskip("optax")
    from numpyro.optim import optax_to_numpyro

    (closure_cell,) = [
        source(cell)
        for cell in cells
        if cell["cell_type"] == "code" and "def scaled_forecast_fn(" in source(cell)
    ]
    # Only the names that the notebook's setup cell imports; no fit settings.
    namespace: dict[str, Any] = {
        "Array": Array,
        "AutoNormal": AutoNormal,
        "Float": Float,
        "ForecastFn": ForecastFn,
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
    exec(closure_cell, namespace)  # noqa: S102 - a code cell of our own committed notebook
    scaled_forecast_fn = namespace["make_scaled_forecast_fn"](optax.adam(0.01), 200)

    sales = jnp.abs(random.normal(random.PRNGKey(0), (24, 3))) + 1.0
    sales = sales.at[:20, 2].set(0.0)  # series 2: no sales in the 20-day training window
    covariates = jnp.zeros((1, 24, 3))
    pred = scaled_forecast_fn(
        random.PRNGKey(1), _toy_model, sales[:20], covariates[:, :20], covariates, 50
    )
    assert pred.shape == (50, 4, 3)
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(pred))))
