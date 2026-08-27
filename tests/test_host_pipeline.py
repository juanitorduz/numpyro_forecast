"""End-to-end ``device="host"`` pipeline test (spec roadmap, host-offload contract).

``device="host"`` keeps every result in pageable host memory: a `jax.Array`
committed to the CPU backend device when that backend is initialized, and a
NumPy array (one ``jax.device_get`` per chunk, no backend needed) when it is
not, so nothing of a draw occupies accelerator memory and nothing is pinned.
Every host-offload-aware function in the package (``draw_posterior``,
``predict_in_sample``, ``forecast``, and internally ``to_datatree``) must both
*produce* such arrays and *accept* them as input, so the output of one stage can
be fed straight into the next without a device round-trip. Those signatures are
enforced at *runtime* by the project's jaxtyping import hook, not merely checked
statically; placement is enforced by ``assert_host_resident`` /
``assert_numpy_host`` and, because committedness propagates through gathers and
jitted calls on a CPU-only machine, by a per-chunk transfer spy for the stages
fed a committed posterior.

A unit test scoped to a single function (e.g. only ``draw_posterior(...,
device="host")`` asserting its own output) never exercises the next function in
the chain, because it never feeds its output onward. Only a full walk through
``draw_posterior -> predict_in_sample -> forecast -> to_datatree``, all pinned to
``device="host"`` and each stage consuming the previous stage's host-committed
output, catches a regression anywhere in that chain, and is exactly the scenario
that matters on GPU; the backend-free walk covers the setup where there is no
CPU backend at all (``jax_platforms`` restricted to the accelerator).
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    assert_host_resident,
    assert_numpy_host,
    empty_covariates,
    fail_devices_for,
    rw_model,
    svi_guide_params,
)
from jax import random

from numpyro_forecast.convert import to_datatree
from numpyro_forecast.predictive import draw_posterior, forecast, predict_in_sample


def _is_cpu_target(device: object) -> bool:
    """Whether a ``jax.device_put`` target is the CPU device or a sharding on it."""
    if isinstance(device, jax.Device):
        return device.platform == "cpu"
    if isinstance(device, jax.sharding.SingleDeviceSharding):
        return all(d.platform == "cpu" for d in device.device_set)
    return False


def test_host_pipeline_draw_predict_forecast_datatree(
    fast_svi: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Walk ``draw_posterior -> predict_in_sample -> forecast -> to_datatree`` on the host.

    Fits the shared random-walk model with plain SVI (``fast_svi``-sized), then
    drives every stage with ``device="host"`` and a ``batch_size`` strictly
    below the draw count so chunking actually runs. Each stage's host-committed
    output is passed directly as the next stage's posterior input: that a
    host-offloaded posterior is *accepted* downstream is itself the contract
    under test, not just the placement of any single call's result. A spy on
    ``jax.device_put`` records every CPU-device transfer, so stages 2 and 3
    are pinned by their per-chunk transfers (one per chunk plus the stitched
    result) rather than by placement alone, which the committed posterior
    would satisfy even without ``device="host"``.
    """
    t = 20
    horizon = 6
    n = 16
    batch_size = 8
    assert n > batch_size  # otherwise draw_posterior would take the unchunked path

    transfers: list[tuple[int, ...]] = []
    real_device_put = jax.device_put

    def spy_device_put(x: jax.Array, device: object = None, **kwargs: object) -> jax.Array:
        if _is_cpu_target(device):
            transfers.append(tuple(x.shape))
        return real_device_put(x, device, **kwargs)

    monkeypatch.setattr(jax, "device_put", spy_device_put)

    guide, params = svi_guide_params(t, num_steps=fast_svi["num_steps"])
    key_draw, key_in_sample, key_forecast, key_tree = random.split(random.PRNGKey(0), 4)

    # Stage 1: draw the posterior, entirely on the host.
    posterior = draw_posterior(key_draw, guide, params, n, batch_size=batch_size, device="host")
    assert_host_resident(posterior)
    transfers.clear()

    # Stage 2: in-sample posterior predictive, fed the host posterior from stage 1.
    covariates = empty_covariates(t)
    in_sample = predict_in_sample(
        key_in_sample, rw_model, posterior, covariates, batch_size=batch_size, device="host"
    )
    assert_host_resident(in_sample)
    assert in_sample.shape == (n, t, 1)
    assert transfers == [(batch_size, t, 1), (batch_size, t, 1), (n, t, 1)]
    transfers.clear()

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
    assert transfers == [(batch_size, horizon, 1), (batch_size, horizon, 1), (n, horizon, 1)]

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


def test_host_pipeline_without_cpu_backend_is_numpy_end_to_end(
    fast_svi: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same four stages with no CPU backend run backend-free on NumPy, silently.

    ``numpyro.set_platform("cuda")`` leaves only the accelerator backend, so
    ``jax.devices("cpu")`` raises; every stage must still complete off the
    accelerator with NumPy results (one ``jax.device_get`` per chunk, nothing
    pinned) and without a single warning: this is the canonical GPU setup, not
    a degraded one.
    """
    t = 20
    horizon = 6
    n = 16
    batch_size = 8
    gets: list[tuple[int, ...]] = []
    real_device_get = jax.device_get

    def spy_device_get(x: jax.Array) -> np.ndarray:
        gets.append(tuple(np.shape(x)))
        return real_device_get(x)

    guide, params = svi_guide_params(t, num_steps=fast_svi["num_steps"])
    key_draw, key_in_sample, key_forecast, key_tree = random.split(random.PRNGKey(0), 4)
    monkeypatch.setattr(jax, "devices", fail_devices_for("cpu"))
    monkeypatch.setattr(jax, "device_get", spy_device_get)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        posterior = draw_posterior(
            key_draw, guide, params, n, batch_size=batch_size, device="host"
        )
        assert_numpy_host(posterior)
        gets.clear()
        in_sample = predict_in_sample(
            key_in_sample,
            rw_model,
            posterior,
            empty_covariates(t),
            batch_size=batch_size,
            device="host",
        )
        assert_numpy_host(in_sample)
        assert gets == [(batch_size, t, 1), (batch_size, t, 1)]  # np.concatenate is no transfer
        gets.clear()
        data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (t, 1)), axis=-2)
        forecasts = forecast(
            key_forecast,
            rw_model,
            posterior,
            data,
            empty_covariates(t + horizon),
            batch_size=batch_size,
            device="host",
        )
        assert_numpy_host(forecasts)
        assert gets == [(batch_size, horizon, 1), (batch_size, horizon, 1)]
        tree = to_datatree(key_tree, rw_model, posterior, data, empty_covariates(t + horizon))
    assert tree["predictions"].sizes["time"] == horizon
