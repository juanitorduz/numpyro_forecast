"""Tests for the BlackJAX kernel adapters (extras leg).

These are skip-marked when ``blackjax`` is not installed, so the base CI leg
(which must not import optional dependencies, invariant I8) never runs them.
"""

import pickle
from collections import namedtuple

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast.forecaster import ForecastingModel, PathfinderForecaster
from numpyro_forecast.functional import draw_posterior, fit_mcmc, forecast
from numpyro_forecast.metrics import crps_empirical
from numpyro_forecast.util import _api_canary

pytest.importorskip("blackjax")

from numpyro_forecast.contrib.blackjax import (
    BlackjaxCustomKernel,
    BlackjaxMCLMCKernel,
    BlackjaxNUTSKernel,
    PathfinderFit,
    fit_pathfinder,
)


class MeanModel(ForecastingModel):
    """Constant-level model with a conjugate Normal prior on the level.

    Observations are ``Normal(mu, 1)`` with ``mu ~ Normal(0, 10)``, so the
    posterior mean of ``mu`` is close to the sample mean of the data: a cheap
    closed-form recovery target.
    """

    def model(self, zero_data: Array | None, covariates: Array) -> None:
        mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
        level = jnp.broadcast_to(mu, (covariates.shape[-2], 1))
        self.predict(dist.Normal(0.0, 1.0), level)


class ReparamModel(ForecastingModel):
    """Random-walk model with a ``LocScaleReparam`` drift (adds a ``_decentered`` site)."""

    def model(self, zero_data: Array | None, covariates: Array) -> None:
        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
        sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 1.0))
        drift = self.time_series(
            "drift", lambda: dist.Normal(0.0, drift_scale), reparam=LocScaleReparam()
        )
        level = jnp.cumsum(drift, axis=-2)
        self.predict(dist.Normal(0.0, sigma), level)


def _empty_covariates(duration: int) -> Array:
    return jnp.zeros((duration, 0))


def test_blackjax_nuts_recovers_normal_normal_posterior() -> None:
    truth = 3.0
    data = truth + random.normal(random.PRNGKey(0), (200, 1))
    covariates = _empty_covariates(200)
    fit = fit_mcmc(
        random.PRNGKey(1),
        MeanModel(),
        data,
        covariates,
        kernel=BlackjaxNUTSKernel,
        kernel_kwargs={"num_adaptation_steps": 300},
        num_warmup=0,
        num_samples=500,
    )
    posterior_mean = float(fit.samples["mu"].mean())
    # Posterior mean ~ sample mean with the near-flat prior; loose MC tolerance.
    assert abs(posterior_mean - float(data.mean())) < 0.1


def test_blackjax_keyset_equals_nuts() -> None:
    """Invariant I2: a blackjax kernel yields the same posterior sites as NUTS."""
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (30, 1)), axis=-2)
    covariates = _empty_covariates(30)

    nuts_fit = fit_mcmc(
        random.PRNGKey(1), ReparamModel(), data, covariates, num_warmup=100, num_samples=100
    )
    bj_fit = fit_mcmc(
        random.PRNGKey(1),
        ReparamModel(),
        data,
        covariates,
        kernel=BlackjaxNUTSKernel,
        kernel_kwargs={"num_adaptation_steps": 100},
        num_warmup=0,
        num_samples=100,
    )
    assert "drift_decentered" in nuts_fit.samples
    assert set(nuts_fit.samples) == set(bj_fit.samples)


def test_blackjax_mclmc_forecast_finite_and_competitive() -> None:
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (30, 1)), axis=-2)
    train_covariates = _empty_covariates(30)
    forecast_covariates = _empty_covariates(36)
    model = ReparamModel()

    nuts_fit = fit_mcmc(
        random.PRNGKey(1), model, data, train_covariates, num_warmup=200, num_samples=200
    )
    mclmc_fit = fit_mcmc(
        random.PRNGKey(1),
        model,
        data,
        train_covariates,
        kernel=BlackjaxMCLMCKernel,
        kernel_kwargs={"num_tuning_steps": 200},
        num_warmup=0,
        num_samples=200,
    )

    nuts_post = draw_posterior(random.PRNGKey(2), nuts_fit, 200)
    mclmc_post = draw_posterior(random.PRNGKey(2), mclmc_fit, 200)
    nuts_fc = forecast(random.PRNGKey(3), model, nuts_post, data, forecast_covariates)
    mclmc_fc = forecast(random.PRNGKey(3), model, mclmc_post, data, forecast_covariates)

    assert bool(jnp.all(jnp.isfinite(mclmc_fc)))
    truth = data[-6:]
    nuts_crps = float(crps_empirical(nuts_fc, truth).mean())
    mclmc_crps = float(crps_empirical(mclmc_fc, truth).mean())
    assert mclmc_crps <= 2.0 * nuts_crps


def _nuts_build_fn(rng_key, logdensity_fn, position, num_warmup):  # type: ignore[no-untyped-def]
    """A minimal custom build_fn wrapping BlackJAX window-adapted NUTS."""
    import blackjax

    adapt = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
    result, _info = adapt.run(rng_key, position, num_steps=200)  # ty: ignore[unknown-argument]
    kernel = blackjax.nuts(logdensity_fn, **result.parameters)
    return result.state, kernel.step


def test_blackjax_custom_kernel_happy_path() -> None:
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (25, 1)), axis=-2)
    covariates = _empty_covariates(25)
    fit = fit_mcmc(
        random.PRNGKey(1),
        RandomWalkForCustom(),
        data,
        covariates,
        kernel=BlackjaxCustomKernel,
        kernel_kwargs={"build_fn": _nuts_build_fn},
        num_warmup=0,
        num_samples=100,
    )
    assert fit.samples["sigma"].shape == (100,)
    assert bool(jnp.all(jnp.isfinite(fit.samples["sigma"])))


class RandomWalkForCustom(ForecastingModel):
    """Plain random-walk model used by the custom-kernel happy path."""

    def model(self, zero_data: Array | None, covariates: Array) -> None:
        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
        sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 1.0))
        drift = self.time_series("drift", lambda: dist.Normal(0.0, drift_scale))
        level = jnp.cumsum(drift, axis=-2)
        self.predict(dist.Normal(0.0, sigma), level)


_BadState = namedtuple("_BadState", ["position"])


def _malformed_build_fn(rng_key, logdensity_fn, position, num_warmup):  # type: ignore[no-untyped-def]
    """Return a state whose position keys do not match the model's sites."""
    return _BadState(position={"not_a_real_site": jnp.zeros(())}), (lambda k, s: (s, None))


def test_blackjax_custom_kernel_malformed_state_raises() -> None:
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (25, 1)), axis=-2)
    covariates = _empty_covariates(25)
    with pytest.raises(TypeError, match="differ from the model"):
        fit_mcmc(
            random.PRNGKey(1),
            RandomWalkForCustom(),
            data,
            covariates,
            kernel=BlackjaxCustomKernel,
            kernel_kwargs={"build_fn": _malformed_build_fn},
            num_warmup=0,
            num_samples=10,
        )


def test_blackjax_kernel_rejects_non_sequential_chain_method() -> None:
    """A Blackjax* kernel with chain_method='vectorized' raises before running."""
    from numpyro_forecast.exceptions import KernelConfigError

    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (12, 1)), axis=-2)
    cov = _empty_covariates(12)
    with pytest.raises(KernelConfigError, match="sequential"):
        fit_mcmc(
            random.PRNGKey(1),
            MeanModel(),
            data,
            cov,
            kernel=BlackjaxNUTSKernel,
            chain_method="vectorized",
            num_warmup=0,
            num_samples=5,
        )


def test_blackjax_kernel_warns_on_num_warmup() -> None:
    """num_warmup>0 warns, attributed to the caller of fit_mcmc (stacklevel=3)."""
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (12, 1)), axis=-2)
    cov = _empty_covariates(12)
    with pytest.warns(UserWarning, match="warmup") as record:
        fit_mcmc(
            random.PRNGKey(1),
            MeanModel(),
            data,
            cov,
            kernel=BlackjaxNUTSKernel,
            kernel_kwargs={"num_adaptation_steps": 20},
            num_warmup=2,
            num_samples=5,
        )
    # stacklevel=3 points the warning at this test file, not fit_mcmc's frame.
    assert any(w.filename.endswith("test_blackjax.py") for w in record)


def test_blackjax_api_canaries() -> None:
    """Pin the exact BlackJAX symbols the adapters rely on (risk K1)."""
    _api_canary("blackjax", ["nuts", "window_adaptation", "mclmc", "mclmc_find_L_and_step_size"])
    _api_canary("blackjax.mcmc.mclmc", ["init", "build_kernel"])


# --- Pathfinder (P11) --------------------------------------------------------


def test_pathfinder_approximate_return_structure_canary() -> None:
    """Pin the ``approximate`` ELBO path and ``sample`` arity (risk K1)."""
    import blackjax

    _api_canary("blackjax.vi.pathfinder", ["approximate", "sample", "PathfinderState"])
    assert "elbo" in blackjax.vi.pathfinder.PathfinderState._fields

    def logdensity(p: dict[str, Array]) -> Array:
        return -0.5 * (p["x"] ** 2).sum()

    state, _info = blackjax.vi.pathfinder.approximate(
        random.PRNGKey(0), logdensity, {"x": jnp.zeros(2)}, num_samples=50, ftol=1e-4
    )
    out = blackjax.vi.pathfinder.sample(random.PRNGKey(1), state, 5)
    assert isinstance(out, tuple)
    assert len(out) == 2  # (samples, log_weights)


def test_pathfinder_constrained_support() -> None:
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (24, 1)), axis=-2)
    covariates = _empty_covariates(24)
    fit = fit_pathfinder(
        random.PRNGKey(1),
        RandomWalkForCustom(),
        data,
        covariates,
        num_elbo_samples=100,
        ftol=1e-4,
    )
    assert isinstance(fit, PathfinderFit)
    post = draw_posterior(random.PRNGKey(2), fit, 200)
    assert bool(jnp.all(post["sigma"] > 0.0))
    assert bool(jnp.all(post["drift_scale"] > 0.0))


def test_pathfinder_draw_posterior_splits_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pathfinder draw splits rng_key: model init and sampling get distinct subkeys."""
    import blackjax

    import numpyro_forecast.contrib.blackjax as bj

    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (24, 1)), axis=-2)
    covariates = _empty_covariates(24)
    fit = fit_pathfinder(
        random.PRNGKey(1), RandomWalkForCustom(), data, covariates, num_elbo_samples=100
    )

    captured: dict[str, Array] = {}
    real_init = bj.initialize_model
    real_sample = blackjax.vi.pathfinder.sample

    def spy_init(rng_key: Array, *args: object, **kwargs: object) -> object:
        captured["init"] = rng_key
        return real_init(rng_key, *args, **kwargs)

    def spy_sample(rng_key: Array, *args: object, **kwargs: object) -> object:
        captured["sample"] = rng_key
        return real_sample(rng_key, *args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(bj, "initialize_model", spy_init)
    monkeypatch.setattr(blackjax.vi.pathfinder, "sample", spy_sample)

    parent = random.PRNGKey(2)
    draw_posterior(parent, fit, 50)

    assert not jnp.array_equal(captured["init"], captured["sample"])
    assert not jnp.array_equal(captured["init"], parent)
    assert not jnp.array_equal(captured["sample"], parent)


def test_pathfinder_forecast_composes() -> None:
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (24, 1)), axis=-2)
    forecaster = PathfinderForecaster(
        random.PRNGKey(1),
        RandomWalkForCustom(),
        data,
        _empty_covariates(24),
        num_elbo_samples=100,
        ftol=1e-4,
    )
    forecast_samples = forecaster(random.PRNGKey(2), data, _empty_covariates(30), 100)
    assert forecast_samples.shape == (100, 6, 1)
    assert bool(jnp.all(jnp.isfinite(forecast_samples)))
    assert isinstance(forecaster.elbo, float)


def test_pathfinder_fit_pickle_round_trip() -> None:
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (24, 1)), axis=-2)
    covariates = _empty_covariates(24)
    fit = fit_pathfinder(
        random.PRNGKey(1), RandomWalkForCustom(), data, covariates, num_elbo_samples=100
    )
    restored = pickle.loads(pickle.dumps(fit))  # noqa: S301 - round-trip of our own data
    assert isinstance(restored, PathfinderFit)
    assert restored.elbo == fit.elbo
    # The restored fit still draws a valid constrained posterior.
    post = draw_posterior(random.PRNGKey(2), restored, 50)
    assert bool(jnp.all(post["sigma"] > 0.0))


def test_pathfinder_as_backtest_forecaster_fn() -> None:
    from numpyro_forecast.evaluate import backtest

    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (24, 1)), axis=-2)
    covariates = _empty_covariates(24)

    def make(rng_key, model, train_data, train_covariates, **options):  # type: ignore[no-untyped-def]
        return PathfinderForecaster(
            rng_key, model, train_data, train_covariates, num_elbo_samples=80, ftol=1e-4
        )

    results = backtest(
        random.PRNGKey(1),
        data,
        covariates,
        RandomWalkForCustom,
        forecaster_fn=make,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=50,
    )
    assert results
    for r in results:
        assert set(r.metrics)
        assert all(isinstance(v, float) for v in r.metrics.values())
