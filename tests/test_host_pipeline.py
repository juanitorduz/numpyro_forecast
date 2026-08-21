"""End-to-end ``device="host"`` pipeline test (spec roadmap, host-offload contract).

Every host-offload-aware function in the package (``draw_posterior``,
``predict_in_sample``, ``forecast``, and internally ``to_datatree``) is typed
to accept ``Array | np.ndarray`` wherever a posterior or draw tensor flows in,
so a NumPy draw produced by one stage can be fed straight into the next
without a device round-trip. That widened annotation is enforced at *runtime*
by the project's jaxtyping import hook, not merely checked statically: the
moment a signature in the chain regresses back to an ``Array``-only
annotation, the hook rejects a NumPy argument the instant it is passed in,
regardless of what ``ty`` says. A unit test scoped to a single function (e.g.
only ``draw_posterior(..., device="host")`` asserting its own output is
``np.ndarray``) never exercises that hook for the *next* function, because it
never feeds the NumPy output onward. Only a full walk through
``draw_posterior -> predict_in_sample -> forecast -> to_datatree``, all pinned
to ``device="host"`` and each stage consuming the previous stage's NumPy
output, catches a regression anywhere in that chain, and is exactly the
scenario that matters on GPU, where there may be no CPU backend to fall back
to at all.
"""

import jax.numpy as jnp
import numpy as np
from conftest import empty_covariates, rw_model, svi_guide_params
from jax import random
from jax import tree_util as jtu

from numpyro_forecast.convert import to_datatree
from numpyro_forecast.functional import draw_posterior, forecast, predict_in_sample


def test_host_pipeline_draw_predict_forecast_datatree(fast_svi: dict[str, int]) -> None:
    """Walk ``draw_posterior -> predict_in_sample -> forecast -> to_datatree`` on the host.

    Fits the shared random-walk model with plain SVI (``fast_svi``-sized), then
    drives every stage with ``device="host"`` and a ``batch_size`` strictly
    below the draw count so chunking actually runs. Each stage's NumPy output
    is passed directly as the next stage's posterior input: that a
    host-offloaded (NumPy) posterior is *accepted* downstream is itself the
    contract under test, not just the output type of any single call.
    """
    t = 20
    horizon = 6
    n = 16
    batch_size = 8
    assert n > batch_size  # otherwise draw_posterior would take the unchunked path

    guide, params = svi_guide_params(t, num_steps=fast_svi["num_steps"])
    key_draw, key_in_sample, key_forecast, key_tree = random.split(random.PRNGKey(0), 4)

    # Stage 1: draw the posterior, entirely on the host.
    posterior = draw_posterior(key_draw, guide, params, n, batch_size=batch_size, device="host")
    assert all(isinstance(leaf, np.ndarray) for leaf in jtu.tree_leaves(posterior))

    # Stage 2: in-sample posterior predictive, fed the NumPy posterior from stage 1.
    covariates = empty_covariates(t)
    in_sample = predict_in_sample(
        key_in_sample, rw_model, posterior, covariates, batch_size=batch_size, device="host"
    )
    assert isinstance(in_sample, np.ndarray)
    assert in_sample.shape == (n, t, 1)

    # Same deterministic recipe svi_guide_params uses internally, so `data` matches
    # what the guide was actually fit on.
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (t, 1)), axis=-2)
    covariates_full = empty_covariates(t + horizon)

    # Stage 3: forecast, again fed the NumPy posterior.
    forecasts = forecast(
        key_forecast,
        rw_model,
        posterior,
        data,
        covariates_full,
        batch_size=batch_size,
        device="host",
    )
    assert isinstance(forecasts, np.ndarray)
    assert forecasts.shape == (n, horizon, 1)

    # Stage 4: to_datatree, once more fed the NumPy posterior; covariates_full extends
    # beyond data so it exercises to_datatree's internal forecast() call too (default
    # predictive_device="host"), producing the predictions groups alongside the
    # in-sample ones.
    tree = to_datatree(key_tree, rw_model, posterior, data, covariates_full)
    assert set(tree.children) == {
        "posterior",
        "posterior_predictive",
        "observed_data",
        "constant_data",
        "predictions",
        "predictions_constant_data",
    }
    post = tree["posterior"]
    assert post.sizes["chain"] == 1
    assert post.sizes["draw"] == n
    assert tree["posterior_predictive"].sizes["time"] == t
    assert tree["predictions"].sizes["time"] == horizon
