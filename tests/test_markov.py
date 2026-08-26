"""Tests for :func:`markov_series` (roadmap §7.5)."""

from collections.abc import Callable

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast.models import Horizon, markov_series, predict
from numpyro_forecast.predictive import draw_posterior, forecast
from tests.conftest import empty_covariates

PHI = 0.85
DRIFT = 0.08


def _ar1_transition(
    carry: Array, _: Array | None
) -> tuple[dist.Distribution, Callable[[Array], Array]]:
    return dist.Normal(PHI * carry, DRIFT), lambda z: z


def _ar1_body(h: Horizon, covariates: Array) -> None:
    init = jnp.zeros((1,))
    z = markov_series(h, "z", init, _ar1_transition)
    predict(h, dist.Normal(0.0, 0.05), z)


def _ar1_model(covariates: Array, data: Array | None = None) -> None:
    """Plain-function AR(1) model on :func:`_ar1_body`."""
    _ar1_body(Horizon.from_data(covariates, data), covariates)


@pytest.mark.parametrize("fitter", ["svi", "mcmc"])
def test_ar1_forecast_matches_closed_form(fitter: str, rng_key: Array) -> None:
    """Posterior-mean forecast decays as ``phi^k * z_last`` under SVI and NUTS."""
    t_obs, future = 20, 8
    duration = t_obs + future
    cov = empty_covariates(duration)
    data = jnp.zeros((t_obs, 1))
    if fitter == "svi":
        key_fit, key_draw = random.split(rng_key)
        guide = AutoNormal(_ar1_model)
        svi = SVI(_ar1_model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
        state = svi.run(key_fit, 400, cov[:t_obs], data, progress_bar=False)
        post = draw_posterior(key_draw, guide, state.params, 200)
    else:
        # MCMC posterior samples (mcmc.get_samples()) go straight to forecast(),
        # with no draw_posterior step (that's guide-based only); fit for exactly
        # the wanted draw count instead of thinning/resampling after the fact.
        mcmc = MCMC(
            NUTS(_ar1_model),
            num_warmup=100,
            num_samples=200,
            progress_bar=False,
        )
        mcmc.run(rng_key, cov[:t_obs], data)
        post = mcmc.get_samples()
    z_last = float(post["z"][:, -1, 0].mean())
    preds = forecast(random.PRNGKey(2), _ar1_model, post, data, cov, batch_size=50)
    mean_fc = preds.mean(axis=0)[:, 0]
    closed = jnp.array([PHI ** (k + 1) * z_last for k in range(future)])
    assert jnp.allclose(mean_fc, closed, rtol=0.15, atol=0.15)


def test_carry_threading_extreme_state() -> None:
    """Horizon-1 forecast tracks ``phi * z_last``, not the unconditional mean."""
    t_obs, future = 12, 4
    duration = t_obs + future
    cov = empty_covariates(duration)
    data = jnp.zeros((t_obs, 1))
    z_last = 5.0
    posterior = {"z": jnp.full((3, t_obs, 1), z_last)}
    pred = Predictive(_ar1_model, posterior_samples=posterior, return_sites=["z_future"])
    zf = pred(random.PRNGKey(0), cov, data)["z_future"]
    mean_step0 = float(zf[:, 0, 0].mean())
    assert mean_step0 > 3.0
    assert abs(mean_step0 - PHI * z_last) < 0.5


def test_batched_plates_argument() -> None:
    """A ``(B,)`` plate yields scan storage ``(t, B, obs)``."""
    h = Horizon.from_data(empty_covariates(10), jnp.zeros((6, 1)))

    def trans(carry: Array, _: Array | None) -> tuple[dist.Distribution, Callable[[Array], Array]]:
        return dist.Normal(PHI * carry, DRIFT), lambda z: z

    def body(h_: Horizon) -> None:
        markov_series(
            h_,
            "z",
            jnp.zeros((3, 1)),
            trans,
            plates=[("series", 3)],
        )

    tr = trace(seed(lambda: body(h), random.PRNGKey(0))).get_trace()
    assert tr["z"]["value"].shape == (6, 3, 1)


def test_missing_obs_dim_raises_with_guidance() -> None:
    """C7: a scalar per-step distribution raises with actionable guidance."""

    def bad_trans(
        carry: Array, _: Array | None
    ) -> tuple[dist.Distribution, Callable[[Array], Array]]:
        return dist.Normal(PHI * carry, DRIFT), lambda z: z

    h = Horizon.from_data(empty_covariates(5), jnp.zeros((5, 1)))
    with pytest.raises(ValueError, match="observation dimension"):
        markov_series(h, "z", jnp.zeros(()), bad_trans)


def test_enclosing_plate_rejected_with_guidance() -> None:
    """An enclosing user plate is rejected with guidance."""
    h = Horizon.from_data(empty_covariates(5), jnp.zeros((5, 1)))

    def body() -> None:
        with numpyro.plate("outer", 2):
            markov_series(h, "z", jnp.zeros((1,)), _ar1_transition)

    with pytest.raises(ValueError, match="plates= argument"):
        trace(seed(body, random.PRNGKey(0))).get_trace()


def test_reparam_config_end_to_end() -> None:
    """Decentered reparam inside the scan body exposes ``z_decentered``."""
    h = Horizon.from_data(empty_covariates(10), jnp.zeros((10, 1)))

    def traced() -> None:
        markov_series(
            h,
            "z",
            jnp.zeros((1,)),
            _ar1_transition,
            reparam_config={"z": LocScaleReparam(0.0)},
        )

    tr = trace(seed(traced, random.PRNGKey(0))).get_trace()
    assert "z_decentered" in tr
    assert tr["z"]["type"] == "deterministic"


def test_xs_threading() -> None:
    """Exogenous ``xs`` are sliced and moved into scan layout correctly."""
    h = Horizon.from_data(empty_covariates(8), jnp.zeros((8, 1)))
    xs = jnp.arange(8.0)[:, None]

    def driven(
        carry: Array, x_t: Array | None
    ) -> tuple[dist.Distribution, Callable[[Array], Array]]:
        exog = jnp.zeros((1,)) if x_t is None else x_t
        return dist.Normal(PHI * carry + exog, DRIFT), lambda z: z

    tr = trace(
        seed(
            lambda: markov_series(h, "z", jnp.zeros((1,)), driven, xs=xs),
            random.PRNGKey(0),
        )
    ).get_trace()
    assert tr["z"]["value"].shape[0] == 8


def test_ar1_model_end_to_end_shape() -> None:
    """The plain-function AR(1) model fits and forecasts with the expected shape."""
    data = jnp.zeros((10, 1))
    cov = empty_covariates(14)
    key_fit, key_draw = random.split(random.PRNGKey(0))
    guide = AutoNormal(_ar1_model)
    svi = SVI(_ar1_model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    state = svi.run(key_fit, 50, cov[:10], data, progress_bar=False)
    post = draw_posterior(key_draw, guide, state.params, 20)
    preds = forecast(random.PRNGKey(2), _ar1_model, post, data, cov)
    assert preds.shape == (20, 4, 1)
