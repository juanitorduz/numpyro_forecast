"""End-to-end ``device="host"`` pipeline test (spec roadmap, host-offload contract).

``device="host"`` keeps every result a :class:`jax.Array` while committing it to
host memory (a ``"pinned_host"``/``"unpinned_host"`` memory kind on the array's
own device), so nothing of a draw occupies accelerator memory. Every
host-offload-aware function in the package (``draw_posterior``,
``predict_in_sample``, ``forecast``, and internally ``to_datatree``) must both
*produce* such arrays and *accept* them as input, so the output of one stage can
be fed straight into the next without a device round-trip. Those signatures are
enforced at *runtime* by the project's jaxtyping import hook, not merely checked
statically, and the memory kind is enforced by ``assert_host_resident``.

A unit test scoped to a single function (e.g. only ``draw_posterior(...,
device="host")`` asserting its own output) never exercises the next function in
the chain, because it never feeds its output onward. Only a full walk through
``draw_posterior -> predict_in_sample -> forecast -> to_datatree``, all pinned to
``device="host"`` and each stage consuming the previous stage's host-committed
output, catches a regression anywhere in that chain, and is exactly the scenario
that matters on GPU, where there may be no CPU backend to fall back to at all.
"""

import jax.numpy as jnp
from conftest import assert_host_resident, empty_covariates, rw_model, svi_guide_params
from jax import random

from numpyro_forecast.convert import to_datatree
from numpyro_forecast.functional import draw_posterior, forecast, predict_in_sample


def test_host_pipeline_draw_predict_forecast_datatree(fast_svi: dict[str, int]) -> None:
    """Walk ``draw_posterior -> predict_in_sample -> forecast -> to_datatree`` on the host.

    Fits the shared random-walk model with plain SVI (``fast_svi``-sized), then
    drives every stage with ``device="host"`` and a ``batch_size`` strictly
    below the draw count so chunking actually runs. Each stage's host-committed
    output is passed directly as the next stage's posterior input: that a
    host-offloaded posterior is *accepted* downstream is itself the contract
    under test, not just the placement of any single call's result.
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
    assert_host_resident(posterior)

    # Stage 2: in-sample posterior predictive, fed the host posterior from stage 1.
    covariates = empty_covariates(t)
    in_sample = predict_in_sample(
        key_in_sample, rw_model, posterior, covariates, batch_size=batch_size, device="host"
    )
    assert_host_resident(in_sample)
    assert in_sample.shape == (n, t, 1)

    # Same deterministic recipe svi_guide_params uses internally, so `data` matches
    # what the guide was actually fit on.
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (t, 1)), axis=-2)
    covariates_full = empty_covariates(t + horizon)

    # Stage 3: forecast, again fed the host posterior.
    forecasts = forecast(
        key_forecast,
        rw_model,
        posterior,
        data,
        covariates_full,
        batch_size=batch_size,
        device="host",
    )
    assert_host_resident(forecasts)
    assert forecasts.shape == (n, horizon, 1)

    # Stage 4: to_datatree, once more fed the host posterior; covariates_full extends
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
