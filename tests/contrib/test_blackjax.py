"""Tests for the BlackJAX kernel adapters (extras leg).

These are skip-marked when ``blackjax`` is not installed, so the base CI leg
(which must not import optional dependencies, invariant I8) never runs them.
"""

import pickle
from collections import namedtuple

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast.forecaster import ForecastingModel, PathfinderForecaster
from numpyro_forecast.functional import fit_mcmc, forecast
from numpyro_forecast.metrics import crps_empirical
from numpyro_forecast.optional import _api_canary

pytest.importorskip("blackjax")

from numpyro_forecast.contrib.blackjax import (
    BlackjaxCustomKernel,
    BlackjaxMCLMCKernel,
    BlackjaxNUTSKernel,
    PathfinderFit,
    fit_pathfinder,
    pathfinder_samples,
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

    # MCMC posterior samples (mcmc.get_samples()) go straight to forecast(), with
    # no draw_posterior step (that's guide-based only); both fits already hold
    # exactly the 200 draws requested below.
    nuts_post = nuts_fit.samples
    mclmc_post = mclmc_fit.samples
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
    # The stable-sampler patch replaces this exact symbol in both namespaces.
    _api_canary("blackjax.optimizers.lbfgs", ["bfgs_sample", "minimize_lbfgs"])
    _api_canary("blackjax.vi.pathfinder", ["bfgs_sample"])


def test_blackjax_mclmc_tuning_signature_canary() -> None:
    """Pin the blackjax>=1.6 MCLMC tuning contract the adapter targets (risk K1).

    In 1.6 ``build_kernel`` stopped closing over ``logdensity_fn``/
    ``inverse_mass_matrix`` (both moved to the per-step kernel call) and
    ``mclmc_find_L_and_step_size`` grew a required ``logdensity_fn`` parameter
    taking the kernel directly instead of a mass-matrix factory. If upstream
    flips either back, fail here with a precise message rather than deep inside
    ``BlackjaxMCLMCKernel._build``.
    """
    import inspect

    import blackjax

    build_params = inspect.signature(blackjax.mcmc.mclmc.build_kernel).parameters
    assert "integrator" in build_params
    assert "logdensity_fn" not in build_params
    assert "inverse_mass_matrix" not in build_params

    tune_params = inspect.signature(blackjax.mclmc_find_L_and_step_size).parameters
    assert "logdensity_fn" in tune_params
    assert "mclmc_kernel" in tune_params


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
    post = pathfinder_samples(random.PRNGKey(2), fit, 200)
    assert bool(jnp.all(post["sigma"] > 0.0))
    assert bool(jnp.all(post["drift_scale"] > 0.0))


def test_pathfinder_samples_chunked_matches_unchunked_shape() -> None:
    """Chunked and unchunked pathfinder draws agree on shape (values differ: distinct subkeys)."""
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (24, 1)), axis=-2)
    covariates = _empty_covariates(24)
    fit = fit_pathfinder(
        random.PRNGKey(1), RandomWalkForCustom(), data, covariates, num_elbo_samples=100
    )
    unchunked = pathfinder_samples(random.PRNGKey(2), fit, 10)
    chunked = pathfinder_samples(random.PRNGKey(2), fit, 10, batch_size=4)
    assert set(chunked) == set(unchunked)
    for name in unchunked:
        assert chunked[name].shape == unchunked[name].shape
        assert bool(jnp.all(jnp.isfinite(chunked[name])))


def test_pathfinder_samples_device_host_returns_numpy() -> None:
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (24, 1)), axis=-2)
    covariates = _empty_covariates(24)
    fit = fit_pathfinder(
        random.PRNGKey(1), RandomWalkForCustom(), data, covariates, num_elbo_samples=100
    )
    hosted = pathfinder_samples(random.PRNGKey(2), fit, 20, device="host")
    assert all(isinstance(leaf, np.ndarray) for leaf in hosted.values())
    assert hosted["sigma"].shape == (20,)


def test_pathfinder_samples_splits_key(monkeypatch: pytest.MonkeyPatch) -> None:
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
    pathfinder_samples(parent, fit, 50)

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
    post = pathfinder_samples(random.PRNGKey(2), restored, 50)
    assert bool(jnp.all(post["sigma"] > 0.0))


def test_stable_bfgs_sample_matches_dense_gaussian_low_dim() -> None:
    """The stable sampler's log density equals the dense MVN logpdf it factorizes.

    The factors ``(alpha, beta, gamma)`` define the approximation's covariance
    ``Sigma = diag(alpha) + beta @ gamma @ beta.T`` (formula II.1 of Zhang et al.),
    so the returned log density must match the explicit multivariate normal.
    """
    from jax.scipy.stats import multivariate_normal

    from numpyro_forecast.contrib.blackjax import _stable_bfgs_sample

    key_factors, key_sample = random.split(random.PRNGKey(0))
    k1, k2, k3, k4 = random.split(key_factors, 4)
    dim, rank = 5, 4
    alpha = random.uniform(k1, (dim,), minval=0.5, maxval=1.5)
    beta = 0.3 * random.normal(k2, (dim, rank))
    core = 0.3 * random.normal(k3, (rank, rank))
    gamma = core @ core.T  # PSD keeps both Sigma and I + R @ gamma @ R.T valid
    position = random.normal(k4, (dim,))
    grad_position = jnp.zeros(dim)

    phi, logq = _stable_bfgs_sample(key_sample, 100, position, grad_position, alpha, beta, gamma)
    sigma_dense = jnp.diag(alpha) + beta @ gamma @ beta.T
    expected = multivariate_normal.logpdf(phi, position, sigma_dense)
    assert bool(jnp.allclose(logq, expected, rtol=1e-3, atol=1e-3))


def test_stable_bfgs_sample_finite_beyond_float_underflow() -> None:
    """Regression: upstream's ``log(prod(alpha))`` underflows to ``-inf`` in high dim.

    With 600 curvature scales of ``1e-3`` the product underflows in float32 and
    float64 alike, so upstream floors the log density at ``+inf`` and every path
    ELBO at ``-inf``. The sum-of-logs form must stay finite.
    """
    from numpyro_forecast.contrib.blackjax import _stable_bfgs_sample

    dim, rank = 600, 4
    alpha = jnp.full((dim,), 1e-3)
    beta = 1e-3 * random.normal(random.PRNGKey(0), (dim, rank))
    gamma = 0.1 * jnp.eye(rank)
    assert bool(jnp.isneginf(jnp.log(jnp.prod(alpha))))  # the upstream failure mode

    _phi, logq = _stable_bfgs_sample(
        random.PRNGKey(1), 20, jnp.zeros(dim), jnp.zeros(dim), alpha, beta, gamma
    )
    assert bool(jnp.all(jnp.isfinite(logq)))


def test_stable_bfgs_sample_patch_applied() -> None:
    import blackjax.optimizers.lbfgs as lbfgs_module
    import blackjax.vi.pathfinder as pathfinder_module

    from numpyro_forecast.contrib.blackjax import (
        _ensure_stable_bfgs_sample,
        _stable_bfgs_sample,
    )

    _ensure_stable_bfgs_sample()
    assert lbfgs_module.bfgs_sample is _stable_bfgs_sample
    assert pathfinder_module.bfgs_sample is _stable_bfgs_sample


def test_fit_pathfinder_high_dim_finite_elbo() -> None:
    """Regression: a 300-step random walk (302 parameters) gets a finite ELBO.

    Without the stable sampler every ELBO on the L-BFGS path is ``-inf`` for a
    model of this size and the returned state is the degenerate first iterate.
    """
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (300, 1)), axis=-2)
    fit = fit_pathfinder(
        random.PRNGKey(1),
        ReparamModel(),
        data,
        _empty_covariates(300),
        num_elbo_samples=100,
        maxiter=50,
    )
    assert bool(jnp.isfinite(jnp.asarray(fit.elbo)))


def test_fit_pathfinder_maxiter_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    import blackjax

    captured: dict[str, object] = {}
    real_approximate = blackjax.vi.pathfinder.approximate

    def spy_approximate(*args: object, **kwargs: object) -> object:
        captured["maxiter"] = kwargs["maxiter"]
        return real_approximate(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(blackjax.vi.pathfinder, "approximate", spy_approximate)

    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (24, 1)), axis=-2)
    fit_pathfinder(
        random.PRNGKey(1),
        RandomWalkForCustom(),
        data,
        _empty_covariates(24),
        num_elbo_samples=50,
        maxiter=17,
    )
    assert captured["maxiter"] == 17


def test_backtest_accepts_a_forecast_fn_closure() -> None:
    """``backtest`` runs against a plain closure (canned draws; no real fit needed here).

    Pathfinder-as-a-backtest-fitter is retested properly once ``backtest`` grows a
    real closure-based Pathfinder helper (Task 5); this only exercises the generic
    closure contract from this (BlackJAX-optional) test module.
    """
    from numpyro_forecast.evaluate import backtest

    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (24, 1)), axis=-2)
    covariates = _empty_covariates(24)

    def forecast_fn(  # type: ignore[no-untyped-def]
        rng_key,
        model,
        train_data,
        train_covariates,
        test_covariates,
        num_samples,
        *,
        batch_size=None,
    ):
        horizon = test_covariates.shape[-2] - train_data.shape[-2]
        return train_data.mean() + random.normal(
            rng_key, (num_samples, horizon, train_data.shape[-1])
        )

    results = backtest(
        random.PRNGKey(1),
        data,
        covariates,
        RandomWalkForCustom,
        forecast_fn=forecast_fn,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=50,
    )
    assert results
    for r in results:
        assert set(r.metrics)
        assert all(isinstance(v, float) for v in r.metrics.values())
