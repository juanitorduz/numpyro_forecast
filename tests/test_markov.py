"""Tests for :func:`markov_time_series` (roadmap §7.5)."""

from collections.abc import Callable

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast.forecaster import ForecastingModel
from numpyro_forecast.functional import (
    Horizon,
    draw_posterior,
    fit_mcmc,
    fit_svi,
    forecast,
    forecasting_model,
    markov_time_series,
    predict,
)
from tests.conftest import as_autoguide, empty_covariates

PHI = 0.85
DRIFT = 0.08


def _ar1_transition(
    carry: Array, _: Array | None
) -> tuple[dist.Distribution, Callable[[Array], Array]]:
    return dist.Normal(PHI * carry, DRIFT), lambda z: z


def _ar1_body(h: Horizon, covariates: Array) -> None:
    init = jnp.zeros((1,))
    z = markov_time_series(h, "z", init, _ar1_transition)
    predict(h, dist.Normal(0.0, 0.05), z)


def _ar1_model() -> ForecastingModel:
    class AR1(ForecastingModel):
        def model(self, zero_data: Array | None, covariates: Array) -> None:
            init = jnp.zeros((1,))
            z = self.markov_time_series("z", init, _ar1_transition)
            self.predict(dist.Normal(0.0, 0.05), z)

    return AR1()


@pytest.mark.parametrize("fitter", ["svi", "mcmc"])
def test_ar1_forecast_matches_closed_form(fitter: str, rng_key: Array) -> None:
    """Posterior-mean forecast decays as ``phi^k * z_last`` under SVI and NUTS."""
    t_obs, future = 20, 8
    duration = t_obs + future
    cov = empty_covariates(duration)
    data = jnp.zeros((t_obs, 1))
    model = _ar1_model()
    if fitter == "svi":
        fit = fit_svi(rng_key, model, data, cov[:t_obs], num_steps=400)
        post = draw_posterior(random.PRNGKey(1), as_autoguide(fit.guide), fit.params, 200)
    else:
        # MCMC posterior samples (mcmc.get_samples()) go straight to forecast(),
        # with no draw_posterior step (that's guide-based only); fit for exactly
        # the wanted draw count instead of thinning/resampling after the fact.
        fit = fit_mcmc(
            rng_key,
            model,
            data,
            cov[:t_obs],
            num_warmup=100,
            num_samples=200,
        )
        post = fit.samples
    z_last = float(post["z"][:, -1, 0].mean())
    preds = forecast(random.PRNGKey(2), model, post, data, cov, batch_size=50)
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
    model = _ar1_model()
    pred = Predictive(model, posterior_samples=posterior, return_sites=["z_future"])
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
        markov_time_series(
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
        markov_time_series(h, "z", jnp.zeros(()), bad_trans)


def test_enclosing_plate_rejected_with_guidance() -> None:
    """An enclosing user plate is rejected with guidance."""
    h = Horizon.from_data(empty_covariates(5), jnp.zeros((5, 1)))

    def body() -> None:
        with numpyro.plate("outer", 2):
            markov_time_series(h, "z", jnp.zeros((1,)), _ar1_transition)

    with pytest.raises(ValueError, match="plates= argument"):
        trace(seed(body, random.PRNGKey(0))).get_trace()


def test_reparam_config_end_to_end() -> None:
    """Decentered reparam inside the scan body exposes ``z_decentered``."""
    h = Horizon.from_data(empty_covariates(10), jnp.zeros((10, 1)))

    def traced() -> None:
        markov_time_series(
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
            lambda: markov_time_series(h, "z", jnp.zeros((1,)), driven, xs=xs),
            random.PRNGKey(0),
        )
    ).get_trace()
    assert tr["z"]["value"].shape[0] == 8


def test_functional_ar1_via_forecasting_model() -> None:
    """The functional wrapper composes with :func:`forecasting_model`."""
    model = forecasting_model(_ar1_body)
    data = jnp.zeros((10, 1))
    cov = empty_covariates(14)
    fit = fit_svi(random.PRNGKey(0), model, data, cov[:10], num_steps=50)
    post = draw_posterior(random.PRNGKey(1), as_autoguide(fit.guide), fit.params, 20)
    preds = forecast(random.PRNGKey(2), model, post, data, cov)
    assert preds.shape == (20, 4, 1)
