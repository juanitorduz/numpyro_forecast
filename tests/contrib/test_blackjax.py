"""Tests for the BlackJAX kernel adapters (extras leg).

These are skip-marked when ``blackjax`` is not installed, so the base CI leg
(which must not import optional dependencies, invariant I8) never runs them.
"""

import dataclasses
import pickle
from collections import namedtuple
from typing import cast

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast.exceptions import KernelConfigError
from numpyro_forecast.functional import Horizon, forecast, predict, time_series
from numpyro_forecast.metrics import crps_empirical
from numpyro_forecast.optional import _api_canary
from tests.conftest import rw_model

pytest.importorskip("blackjax")

from numpyro_forecast.contrib.blackjax import (
    BlackjaxCustomKernel,
    BlackjaxMCLMCKernel,
    BlackjaxNUTSKernel,
    MultiPathfinderFit,
    PathfinderFit,
    fit_multipathfinder,
    fit_pathfinder,
    multipathfinder_samples,
    pathfinder_samples,
)


def mean_model(covariates: Array, data: Array | None = None) -> None:
    """Constant-level model with a conjugate Normal prior on the level.

    Observations are ``Normal(mu, 1)`` with ``mu ~ Normal(0, 10)``, so the
    posterior mean of ``mu`` is close to the sample mean of the data: a cheap
    closed-form recovery target.
    """
    h = Horizon.from_data(covariates, data)
    mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
    level = jnp.broadcast_to(mu, (h.duration, 1))
    predict(h, dist.Normal(0.0, 1.0), level)


def reparam_model(covariates: Array, data: Array | None = None) -> None:
    """Random-walk model with a ``LocScaleReparam`` drift (adds a ``_decentered`` site)."""
    h = Horizon.from_data(covariates, data)
    drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 1.0))
    drift = time_series(
        h, "drift", lambda: dist.Normal(0.0, drift_scale), reparam=LocScaleReparam()
    )
    level = jnp.cumsum(drift, axis=-2)
    predict(h, dist.Normal(0.0, sigma), level)


def _empty_covariates(duration: int) -> Array:
    return jnp.zeros((duration, 0))


def test_blackjax_nuts_recovers_normal_normal_posterior() -> None:
    truth = 3.0
    data = truth + random.normal(random.PRNGKey(0), (200, 1))
    covariates = _empty_covariates(200)
    mcmc = MCMC(
        BlackjaxNUTSKernel(mean_model, num_adaptation_steps=300),
        num_warmup=0,
        num_samples=500,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(1), covariates, data)
    posterior_mean = float(mcmc.get_samples()["mu"].mean())
    # Posterior mean ~ sample mean with the near-flat prior; loose MC tolerance.
    assert abs(posterior_mean - float(data.mean())) < 0.1


def test_blackjax_keyset_equals_nuts() -> None:
    """Invariant I2: a blackjax kernel yields the same posterior sites as NUTS."""
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (30, 1)), axis=-2)
    covariates = _empty_covariates(30)

    nuts_mcmc = MCMC(NUTS(reparam_model), num_warmup=100, num_samples=100, progress_bar=False)
    nuts_mcmc.run(random.PRNGKey(1), covariates, data)
    nuts_samples = nuts_mcmc.get_samples()

    bj_mcmc = MCMC(
        BlackjaxNUTSKernel(reparam_model, num_adaptation_steps=100),
        num_warmup=0,
        num_samples=100,
        chain_method="sequential",
        progress_bar=False,
    )
    bj_mcmc.run(random.PRNGKey(1), covariates, data)
    bj_samples = bj_mcmc.get_samples()

    assert "drift_decentered" in nuts_samples
    assert set(nuts_samples) == set(bj_samples)


def test_blackjax_mclmc_forecast_finite_and_competitive() -> None:
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (30, 1)), axis=-2)
    train_covariates = _empty_covariates(30)
    forecast_covariates = _empty_covariates(36)
    model = reparam_model

    nuts_mcmc = MCMC(NUTS(model), num_warmup=200, num_samples=200, progress_bar=False)
    nuts_mcmc.run(random.PRNGKey(1), train_covariates, data)
    nuts_post = nuts_mcmc.get_samples()

    mclmc_mcmc = MCMC(
        BlackjaxMCLMCKernel(model, num_tuning_steps=200),
        num_warmup=0,
        num_samples=200,
        chain_method="sequential",
        progress_bar=False,
    )
    mclmc_mcmc.run(random.PRNGKey(1), train_covariates, data)
    mclmc_post = mclmc_mcmc.get_samples()

    # MCMC posterior samples (mcmc.get_samples()) go straight to forecast(), with
    # no draw_posterior step (that's guide-based only); both fits already hold
    # exactly the 200 draws requested above.
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
    mcmc = MCMC(
        BlackjaxCustomKernel(rw_model, build_fn=_nuts_build_fn),
        num_warmup=0,
        num_samples=100,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(1), covariates, data)
    samples = mcmc.get_samples()
    assert samples["sigma"].shape == (100,)
    assert bool(jnp.all(jnp.isfinite(samples["sigma"])))


_BadState = namedtuple("_BadState", ["position"])


def _malformed_build_fn(rng_key, logdensity_fn, position, num_warmup):  # type: ignore[no-untyped-def]
    """Return a state whose position keys do not match the model's sites."""
    return _BadState(position={"not_a_real_site": jnp.zeros(())}), (lambda k, s: (s, None))


def test_blackjax_custom_kernel_malformed_state_raises() -> None:
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (25, 1)), axis=-2)
    covariates = _empty_covariates(25)
    mcmc = MCMC(
        BlackjaxCustomKernel(rw_model, build_fn=_malformed_build_fn),
        num_warmup=0,
        num_samples=10,
        chain_method="sequential",
        progress_bar=False,
    )
    with pytest.raises(TypeError, match="differ from the model"):
        mcmc.run(random.PRNGKey(1), covariates, data)


# --- Spec-required failure-mode pins (replacing the old fit_mcmc-level checks) ----


def test_blackjax_kernel_vectorized_chain_method_raises_under_batched_rng() -> None:
    """A BlackJAX kernel run with chain_method='vectorized' fails on a batched rng_key.

    Determined empirically: with more than one chain, NumPyro's "vectorized" chain
    method hands ``_BlackjaxKernel.init`` a stacked ``rng_key`` of shape
    ``(num_chains, 2)`` instead of tracing a single per-chain call through
    ``jax.vmap``. The base kernel calls ``jax.random.split(rng_key, 3)``
    unconditionally, and JAX's own ``_check_prng_key`` rejects a key array whose
    leading axis is not scalar-shaped, so the failure surfaces as a plain
    ``ValueError`` from ``jax.random.split``, not from this package's own
    validation (there is none left: run-config validation lived in the deleted
    ``fit_mcmc``, not in the kernel itself). With a single chain there is nothing
    to batch, so this needs ``num_chains=2`` to actually manifest.
    """
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (12, 1)), axis=-2)
    cov = _empty_covariates(12)
    mcmc = MCMC(
        BlackjaxNUTSKernel(mean_model, num_adaptation_steps=20),
        num_warmup=0,
        num_samples=5,
        num_chains=2,
        chain_method="vectorized",
        progress_bar=False,
    )
    with pytest.raises(ValueError, match="split accepts a single key"):
        mcmc.run(random.PRNGKey(1), cov, data)


def test_blackjax_kernel_num_warmup_positive_runs_and_wastes_steps() -> None:
    """num_warmup>0 is documented waste, not an error: it still yields num_samples draws.

    Adaptation for BlackJAX kernels happens once inside ``_BlackjaxKernel.init``
    (see the "Run configuration" docstring sections on the kernel classes), so
    NumPyro's own warmup phase runs the model that many extra times for nothing;
    it is neither rejected nor does it change the returned sample count.
    """
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (12, 1)), axis=-2)
    cov = _empty_covariates(12)
    mcmc = MCMC(
        BlackjaxNUTSKernel(mean_model, num_adaptation_steps=20),
        num_warmup=3,
        num_samples=5,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(1), cov, data)
    samples = mcmc.get_samples()
    assert samples["mu"].shape == (5,)
    assert bool(jnp.all(jnp.isfinite(samples["mu"])))


def test_blackjax_kernel_unbound_raises_kernel_config_error() -> None:
    """A kernel constructed without a model raises KernelConfigError from init."""
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (12, 1)), axis=-2)
    cov = _empty_covariates(12)
    mcmc = MCMC(
        BlackjaxNUTSKernel(),
        num_warmup=0,
        num_samples=5,
        chain_method="sequential",
        progress_bar=False,
    )
    with pytest.raises(KernelConfigError, match="no bound model"):
        mcmc.run(random.PRNGKey(1), cov, data)


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
        rw_model,
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
    fit = fit_pathfinder(random.PRNGKey(1), rw_model, data, covariates, num_elbo_samples=100)
    unchunked = pathfinder_samples(random.PRNGKey(2), fit, 10)
    chunked = pathfinder_samples(random.PRNGKey(2), fit, 10, batch_size=4)
    assert set(chunked) == set(unchunked)
    for name in unchunked:
        assert chunked[name].shape == unchunked[name].shape
        assert bool(jnp.all(jnp.isfinite(chunked[name])))


def test_pathfinder_samples_device_host_returns_numpy() -> None:
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (24, 1)), axis=-2)
    covariates = _empty_covariates(24)
    fit = fit_pathfinder(random.PRNGKey(1), rw_model, data, covariates, num_elbo_samples=100)
    hosted = pathfinder_samples(random.PRNGKey(2), fit, 20, device="host")
    assert all(isinstance(leaf, np.ndarray) for leaf in hosted.values())
    assert hosted["sigma"].shape == (20,)


def test_pathfinder_samples_splits_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pathfinder draw splits rng_key: model init and sampling get distinct subkeys."""
    import blackjax

    import numpyro_forecast.contrib.blackjax as bj

    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (24, 1)), axis=-2)
    covariates = _empty_covariates(24)
    fit = fit_pathfinder(random.PRNGKey(1), rw_model, data, covariates, num_elbo_samples=100)

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
    covariates = _empty_covariates(24)
    fit = fit_pathfinder(
        random.PRNGKey(1),
        rw_model,
        data,
        covariates,
        num_elbo_samples=100,
        ftol=1e-4,
    )
    posterior = pathfinder_samples(random.PRNGKey(2), fit, 100)
    forecast_samples = forecast(
        random.PRNGKey(3), rw_model, posterior, data, _empty_covariates(30)
    )
    assert forecast_samples.shape == (100, 6, 1)
    assert bool(jnp.all(jnp.isfinite(forecast_samples)))
    assert isinstance(fit.elbo, float)


def test_pathfinder_fit_pickle_round_trip() -> None:
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (24, 1)), axis=-2)
    covariates = _empty_covariates(24)
    fit = fit_pathfinder(random.PRNGKey(1), rw_model, data, covariates, num_elbo_samples=100)
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
        reparam_model,
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
        rw_model,
        data,
        _empty_covariates(24),
        num_elbo_samples=50,
        maxiter=17,
    )
    assert captured["maxiter"] == 17


# --- Multi-path Pathfinder ------------------------------------------------

# Shared cheap settings for the multipath tests below: a short random walk and
# the smallest num_paths/num_elbo_samples/maxiter that still exercise the API.
_MULTIPATH_T = 24
_MULTIPATH_NUM_PATHS = 2
_MULTIPATH_NUM_ELBO_SAMPLES = 50
_MULTIPATH_MAXITER = 50


@pytest.fixture(scope="module")
def multipathfinder_fit() -> MultiPathfinderFit:
    """A cheap multipath fit shared by tests that only inspect its outputs.

    Only 2 paths and 50 ELBO samples reliably push ``pareto_k`` above the 0.7
    warning threshold on this toy posterior; that ``UserWarning`` is expected
    here (a byproduct of the cheap settings, not something this fixture tests)
    and is captured so it does not leak into the test-output summary.
    """
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (_MULTIPATH_T, 1)), axis=-2)
    covariates = _empty_covariates(_MULTIPATH_T)
    with pytest.warns(UserWarning, match="pareto"):
        fit = fit_multipathfinder(
            random.PRNGKey(1),
            rw_model,
            data,
            covariates,
            num_paths=_MULTIPATH_NUM_PATHS,
            num_elbo_samples=_MULTIPATH_NUM_ELBO_SAMPLES,
            maxiter=_MULTIPATH_MAXITER,
        )
    return fit


def test_multipathfinder_api_canary() -> None:
    """Pin the exact ``blackjax.vi.multipathfinder`` symbols this module relies on."""
    import inspect

    import blackjax.vi.multipathfinder as multipathfinder_module

    _api_canary(
        "blackjax.vi.multipathfinder",
        ["multi_approximate", "psis_weights", "MultipathfinderState", "as_top_level_api"],
    )
    assert multipathfinder_module.MultipathfinderState._fields == (
        "path_states",
        "samples",
        "logp",
        "logq",
    )
    params = inspect.signature(multipathfinder_module.multi_approximate).parameters
    for name in ("maxcor", "maxls", "gtol", "ftol", "maxiter"):
        assert name in params


def test_stable_bfgs_patch_covers_multipath_route() -> None:
    """``_ensure_stable_bfgs_sample`` also covers the multipath route.

    ``multi_approximate`` calls ``approximate``/``sample`` imported from
    ``blackjax.vi.pathfinder``, so those two functions' ``__globals__`` are the
    same dict that :func:`fit_pathfinder`'s patch already targets.
    """
    import blackjax.vi.multipathfinder as multipathfinder_module

    from numpyro_forecast.contrib.blackjax import (
        _ensure_stable_bfgs_sample,
        _stable_bfgs_sample,
    )

    _ensure_stable_bfgs_sample()
    assert multipathfinder_module.approximate.__globals__["bfgs_sample"] is _stable_bfgs_sample
    assert multipathfinder_module.sample.__globals__["bfgs_sample"] is _stable_bfgs_sample


def test_fit_multipathfinder_basic(multipathfinder_fit: MultiPathfinderFit) -> None:
    from jax.scipy.special import logsumexp

    fit = multipathfinder_fit
    assert len(fit.elbos) == _MULTIPATH_NUM_PATHS
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(fit.elbos))))
    assert isinstance(fit.pareto_k, float)
    assert bool(jnp.isfinite(jnp.asarray(fit.pareto_k)))
    pool_size = _MULTIPATH_NUM_PATHS * _MULTIPATH_NUM_ELBO_SAMPLES
    assert fit.log_weights.shape == (pool_size,)
    assert float(logsumexp(fit.log_weights)) == pytest.approx(0.0, abs=1e-4)


def test_multipathfinder_samples_constrained_support(
    multipathfinder_fit: MultiPathfinderFit,
) -> None:
    post = multipathfinder_samples(random.PRNGKey(2), multipathfinder_fit, 200)
    assert bool(jnp.all(post["sigma"] > 0.0))
    assert bool(jnp.all(post["drift_scale"] > 0.0))
    assert post["sigma"].shape == (200,)


def test_multipathfinder_samples_chunked_matches_unchunked_shape(
    multipathfinder_fit: MultiPathfinderFit,
) -> None:
    """Chunked and unchunked multipath draws agree on shape (values differ: distinct subkeys)."""
    unchunked = multipathfinder_samples(random.PRNGKey(2), multipathfinder_fit, 10)
    chunked = multipathfinder_samples(random.PRNGKey(2), multipathfinder_fit, 10, batch_size=4)
    assert set(chunked) == set(unchunked)
    for name in unchunked:
        assert chunked[name].shape == unchunked[name].shape
        assert bool(jnp.all(jnp.isfinite(chunked[name])))


def test_multipathfinder_samples_device_host_contract(
    multipathfinder_fit: MultiPathfinderFit,
) -> None:
    hosted = multipathfinder_samples(random.PRNGKey(2), multipathfinder_fit, 20, device="host")
    assert all(isinstance(leaf, np.ndarray) for leaf in hosted.values())
    assert hosted["sigma"].shape == (20,)


def test_multipathfinder_samples_reproducible_per_key(
    multipathfinder_fit: MultiPathfinderFit,
) -> None:
    """Resampling is deterministic in the key: same key repeats, different key differs."""
    key = random.PRNGKey(7)
    first = multipathfinder_samples(key, multipathfinder_fit, 50)
    second = multipathfinder_samples(key, multipathfinder_fit, 50)
    for name in first:
        assert bool(jnp.array_equal(first[name], second[name]))

    different = multipathfinder_samples(random.PRNGKey(8), multipathfinder_fit, 50)
    assert any(not bool(jnp.array_equal(first[name], different[name])) for name in first)


def test_multipathfinder_samples_elbo_exceeds_pool_size(
    multipathfinder_fit: MultiPathfinderFit,
) -> None:
    """``resample="elbo"`` draws fresh per-path samples, so the fit-time pool is no cap.

    The fixture's stored pool holds only ``num_paths * num_elbo_samples = 100``
    draws; asking for 500 used to duplicate them, and now draws 500 fresh ones
    per path instead.
    """
    pool_size = _MULTIPATH_NUM_PATHS * _MULTIPATH_NUM_ELBO_SAMPLES
    num_samples = 5 * pool_size
    post = multipathfinder_samples(
        random.PRNGKey(2), multipathfinder_fit, num_samples, resample="elbo"
    )
    assert post["sigma"].shape == (num_samples,)
    assert bool(jnp.all(post["sigma"] > 0.0))
    assert bool(jnp.all(post["drift_scale"] > 0.0))
    for leaf in post.values():
        assert bool(jnp.all(jnp.isfinite(leaf)))
    # Fresh draws: every one of the 500 is distinct, unlike pool resampling.
    assert len(np.unique(np.asarray(post["drift"]), axis=0)) == num_samples


def test_multipathfinder_samples_psis_reproducible_per_key(
    multipathfinder_fit: MultiPathfinderFit,
) -> None:
    """Explicit PSIS resampling works and is deterministic in the key.

    The cheap fixture's importance weights are degenerate, so the sampling-time
    ``pareto_k`` warning is expected here and captured rather than left to leak.
    """
    key = random.PRNGKey(7)
    with pytest.warns(UserWarning, match="pareto"):
        first = multipathfinder_samples(key, multipathfinder_fit, 50, resample="psis")
    with pytest.warns(UserWarning, match="pareto"):
        second = multipathfinder_samples(key, multipathfinder_fit, 50, resample="psis")
    assert first["sigma"].shape == (50,)
    assert bool(jnp.all(first["sigma"] > 0.0))
    for name in first:
        assert bool(jnp.array_equal(first[name], second[name]))


def test_multipathfinder_samples_auto_follows_pareto_k(
    multipathfinder_fit: MultiPathfinderFit,
) -> None:
    """``resample="auto"`` resolves to ``"elbo"`` above the 0.7 gate and ``"psis"`` below it."""
    assert multipathfinder_fit.pareto_k > 0.7
    key = random.PRNGKey(11)

    auto_high = multipathfinder_samples(key, multipathfinder_fit, 40)
    elbo = multipathfinder_samples(key, multipathfinder_fit, 40, resample="elbo")
    for name in elbo:
        assert bool(jnp.array_equal(auto_high[name], elbo[name]))

    trusted = dataclasses.replace(multipathfinder_fit, pareto_k=0.3)
    with pytest.warns(UserWarning, match="pareto"):
        auto_low = multipathfinder_samples(key, trusted, 40)
    with pytest.warns(UserWarning, match="pareto"):
        psis = multipathfinder_samples(key, trusted, 40, resample="psis")
    for name in psis:
        assert bool(jnp.array_equal(auto_low[name], psis[name]))


def test_multipathfinder_samples_invalid_resample_raises(
    multipathfinder_fit: MultiPathfinderFit,
) -> None:
    with pytest.raises(ValueError, match="resample must be one of"):
        multipathfinder_samples(
            random.PRNGKey(2),
            multipathfinder_fit,
            10,
            resample="nope",  # ty: ignore[invalid-argument-type]
        )


def test_multipathfinder_forecast_composes(multipathfinder_fit: MultiPathfinderFit) -> None:
    posterior = multipathfinder_samples(random.PRNGKey(3), multipathfinder_fit, 100)
    forecast_samples = forecast(
        random.PRNGKey(4),
        rw_model,
        posterior,
        multipathfinder_fit.data,
        _empty_covariates(30),
    )
    assert forecast_samples.shape == (100, 6, 1)
    assert bool(jnp.all(jnp.isfinite(forecast_samples)))


def test_multipathfinder_fit_pickle_round_trip(multipathfinder_fit: MultiPathfinderFit) -> None:
    restored = pickle.loads(pickle.dumps(multipathfinder_fit))  # noqa: S301 - our own data
    assert isinstance(restored, MultiPathfinderFit)
    # The restored fit still resamples a valid constrained posterior.
    post = multipathfinder_samples(random.PRNGKey(5), restored, 50)
    assert bool(jnp.all(post["sigma"] > 0.0))
    assert bool(jnp.all(post["drift_scale"] > 0.0))


def test_fit_multipathfinder_knob_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """maxcor/maxls/gtol/ftol/maxiter reach ``multi_approximate`` (pareto_k warning expected).

    The real ``multi_approximate``/``psis_weights`` still run behind the spy, so
    the same cheap-settings ``pareto_k > 0.7`` warning as the module fixture
    fires here too; it is captured rather than left to leak into test output.
    """
    import blackjax.vi.multipathfinder as multipathfinder_module

    captured: dict[str, object] = {}
    real_multi_approximate = multipathfinder_module.multi_approximate

    def spy_multi_approximate(*args: object, **kwargs: object) -> object:
        captured["maxcor"] = kwargs["maxcor"]
        captured["maxls"] = kwargs["maxls"]
        captured["gtol"] = kwargs["gtol"]
        captured["ftol"] = kwargs["ftol"]
        captured["maxiter"] = kwargs["maxiter"]
        return real_multi_approximate(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(multipathfinder_module, "multi_approximate", spy_multi_approximate)

    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (_MULTIPATH_T, 1)), axis=-2)
    with pytest.warns(UserWarning, match="pareto"):
        fit_multipathfinder(
            random.PRNGKey(1),
            rw_model,
            data,
            _empty_covariates(_MULTIPATH_T),
            num_paths=_MULTIPATH_NUM_PATHS,
            num_elbo_samples=_MULTIPATH_NUM_ELBO_SAMPLES,
            maxiter=41,
            maxcor=6,
            maxls=222,
            gtol=1e-6,
            ftol=1e-4,
        )
    assert captured == {
        "maxcor": 6,
        "maxls": 222,
        "gtol": 1e-6,
        "ftol": 1e-4,
        "maxiter": 41,
    }


def test_fit_pathfinder_maxcor_gtol_maxls_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """maxcor/maxls/gtol reach ``approximate``, and ``init_params`` is its positional position."""
    import blackjax
    from numpyro.infer.util import initialize_model

    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (24, 1)), axis=-2)
    covariates = _empty_covariates(24)
    param_info, _potential_fn_gen, _postprocess_fn, _ = initialize_model(
        random.PRNGKey(9),
        rw_model,
        dynamic_args=True,
        model_args=(covariates, data),
    )
    init_params = param_info.z

    captured: dict[str, object] = {}
    real_approximate = blackjax.vi.pathfinder.approximate

    def spy_approximate(*args: object, **kwargs: object) -> object:
        captured["position"] = args[2]
        captured["maxcor"] = kwargs["maxcor"]
        captured["maxls"] = kwargs["maxls"]
        captured["gtol"] = kwargs["gtol"]
        return real_approximate(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(blackjax.vi.pathfinder, "approximate", spy_approximate)

    fit_pathfinder(
        random.PRNGKey(1),
        rw_model,
        data,
        covariates,
        num_elbo_samples=50,
        maxiter=50,
        maxcor=6,
        maxls=222,
        gtol=1e-6,
        init_params=init_params,
    )

    assert captured["maxcor"] == 6
    assert captured["maxls"] == 222
    assert captured["gtol"] == 1e-6
    position = captured["position"]
    assert isinstance(position, dict)
    assert set(position) == set(init_params)
    typed_position = cast("dict[str, Array]", position)
    for name, value in init_params.items():
        assert bool(jnp.array_equal(typed_position[name], value))


def test_fit_multipathfinder_num_paths_one() -> None:
    """A single-path multipath fit runs and yields a pool-sized weight vector.

    No equality with the single-path :func:`fit_pathfinder` API is asserted: the
    PRNG streams differ by construction (per-path init keys vs. a single init
    key), so their draws are not expected to match. A single path with only 50
    ELBO samples reliably trips the ``pareto_k > 0.7`` warning, so it is
    captured here rather than left to leak into test output.
    """
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (_MULTIPATH_T, 1)), axis=-2)
    with pytest.warns(UserWarning, match="pareto"):
        fit = fit_multipathfinder(
            random.PRNGKey(1),
            rw_model,
            data,
            _empty_covariates(_MULTIPATH_T),
            num_paths=1,
            num_elbo_samples=_MULTIPATH_NUM_ELBO_SAMPLES,
            maxiter=_MULTIPATH_MAXITER,
        )
    assert fit.log_weights.shape == (_MULTIPATH_NUM_ELBO_SAMPLES,)
    post = multipathfinder_samples(random.PRNGKey(2), fit, 50)
    assert bool(jnp.all(post["sigma"] > 0.0))
    assert bool(jnp.all(post["drift_scale"] > 0.0))


def test_fit_multipathfinder_invalid_num_paths_raises() -> None:
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (_MULTIPATH_T, 1)), axis=-2)
    with pytest.raises(ValueError, match="num_paths must be positive"):
        fit_multipathfinder(
            random.PRNGKey(1),
            rw_model,
            data,
            _empty_covariates(_MULTIPATH_T),
            num_paths=0,
        )


def test_fit_multipathfinder_high_dim_finite_elbos() -> None:
    """Regression: a 300-step random walk gets finite ELBOs on every multipath path.

    Multipath analogue of ``test_fit_pathfinder_high_dim_finite_elbo``: end-to-end
    proof the stable-bfgs patch holds on the multipath route too (upstream floors
    every ELBO at ``-inf`` beyond a few hundred parameters). The high-dimensional
    posterior reliably pushes ``pareto_k`` above the 0.7 warning threshold at
    these cheap settings, so that incidental warning is captured here.
    """
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (300, 1)), axis=-2)
    with pytest.warns(UserWarning, match="pareto"):
        fit = fit_multipathfinder(
            random.PRNGKey(1),
            reparam_model,
            data,
            _empty_covariates(300),
            num_paths=_MULTIPATH_NUM_PATHS,
            num_elbo_samples=_MULTIPATH_NUM_ELBO_SAMPLES,
            maxiter=_MULTIPATH_MAXITER,
        )
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(fit.elbos))))


def test_fit_multipathfinder_warns_on_high_pareto_k(monkeypatch: pytest.MonkeyPatch) -> None:
    """A high ``pareto_k`` from PSIS resampling warns (call-time attribute lookup)."""
    import blackjax.vi.multipathfinder as multipathfinder_module

    pool_size = _MULTIPATH_NUM_PATHS * _MULTIPATH_NUM_ELBO_SAMPLES
    uniform_log_weights = -jnp.log(float(pool_size)) * jnp.ones(pool_size)

    def fake_psis_weights(state: object) -> tuple[Array, Array]:
        return uniform_log_weights, jnp.asarray(0.9)

    monkeypatch.setattr(multipathfinder_module, "psis_weights", fake_psis_weights)

    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (_MULTIPATH_T, 1)), axis=-2)
    with pytest.warns(UserWarning, match="pareto"):
        fit_multipathfinder(
            random.PRNGKey(1),
            rw_model,
            data,
            _empty_covariates(_MULTIPATH_T),
            num_paths=_MULTIPATH_NUM_PATHS,
            num_elbo_samples=_MULTIPATH_NUM_ELBO_SAMPLES,
            maxiter=_MULTIPATH_MAXITER,
        )


def test_backtest_accepts_a_forecast_fn_closure() -> None:
    """``backtest`` runs against a plain closure (canned draws; no real fit needed here).

    Pathfinder-as-a-backtest-fitter is retested properly once ``backtest`` grows a
    real closure-based Pathfinder helper (Task 5); this only exercises the generic
    closure contract from this (BlackJAX-optional) test module.
    """
    from numpyro_forecast.evaluate import backtest
    from tests.conftest import rw_model_factory

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
        rw_model_factory,
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
