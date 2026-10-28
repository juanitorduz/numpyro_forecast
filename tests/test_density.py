"""Tests for forecast log densities: readers, the driver, the reductions, the backtest hook.

Every test is an identity or a contract, never a snapshot of fitted numbers: closed-form
densities of models with known parameters, the two reductions against their NumPy
definitions, the chain-rule route against the in-sample filter, and the backtest fields
against the bare-draws run. Tests that reach a not-yet-implemented kernel are marked
``scaffold`` (a strict xfail on ``NotImplementedError``): they turn into failures the
moment the kernel lands, which is the reminder to drop the mark.
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from conftest import (
    assert_host_resident,
    assert_numpy_host,
    empty_covariates,
    svi_forecast_fn,
)
from jax import Array, random
from jax.scipy.special import logsumexp
from jaxtyping import TypeCheckError
from numpyro.handlers import condition, seed, substitute, trace
from numpyro.infer.reparam import LocScaleReparam
from numpyro.infer.util import log_likelihood

from numpyro_forecast import (
    Horizon,
    backtest,
    backtest_vectorized,
    conditional_log_density,
    forecast_log_density,
    innovations,
    log_density_ess,
    predict,
    predict_density,
    reduce_log_density,
    ssoe,
    ssoe_density,
    time_reparam,
)
from numpyro_forecast.exceptions import DensitySiteError
from numpyro_forecast.typing import ForecastFn, ForecastModel, ForecastOutput

scaffold = pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="stacking scaffold: kernel not implemented"
)

T_OBS, FUTURE, SAMPLES = 12, 4, 64


# --- Models with known parameters ------------------------------------------------------------


def mean_model(covariates: Array, data: Array | None = None) -> None:
    """Constant mean, Normal noise: no future latents, so every reduction is closed form."""
    h = Horizon.from_data(covariates, data)
    bias = numpyro.sample("bias", dist.Normal(0.0, 10.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(0.0, 1.0))
    prediction = jnp.broadcast_to(bias, (h.duration, 1))
    predict(h, dist.Normal(0.0, sigma), prediction)


def rw_reparam_model(covariates: Array, data: Array | None = None) -> None:
    """Random-walk level with a non-centered innovation site (``drift_future_decentered``)."""
    h = Horizon.from_data(covariates, data)
    drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 1.0))
    drift = innovations(h, "drift", dist.Normal(0.0, drift_scale), reparam=LocScaleReparam(0))
    predict(h, dist.Normal(0.0, sigma), jnp.cumsum(drift, axis=-2))


def poisson_model(covariates: Array, data: Array | None = None) -> None:
    """Constant log rate with a Poisson link (discrete support)."""
    h = Horizon.from_data(covariates, data)
    log_rate = numpyro.sample("log_rate", dist.Normal(0.0, 1.0))
    prediction = jnp.broadcast_to(log_rate, (h.duration, 1))
    predict(h, lambda eta: dist.Poisson(jnp.exp(eta)), prediction)


def mvn_model(covariates: Array, data: Array | None = None) -> None:
    """Zero mean with a time-correlated ``MultivariateNormal`` noise (time-leading layout)."""
    h = Horizon.from_data(covariates, data)
    rho = numpyro.sample("rho", dist.Uniform(0.0, 0.9))
    idx = jnp.arange(h.duration)
    cov = rho ** jnp.abs(idx[:, None] - idx[None, :])
    predict(h, dist.MultivariateNormal(jnp.zeros(h.duration), cov), jnp.zeros((h.duration, 1)))


def series_model(covariates: Array, data: Array | None = None) -> None:
    """Three independent constant-mean series under a series plate (batch dim ``-3``)."""
    h = Horizon.from_data(covariates, data)
    with numpyro.plate("series", 3, dim=-3):
        bias = numpyro.sample("bias", dist.Normal(0.0, 10.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(0.0, 1.0))
    prediction = jnp.broadcast_to(bias, (3, h.duration, 1))
    predict(h, dist.Normal(0.0, sigma), prediction)


def ar1_ssoe_model(covariates: Array, data: Array | None = None) -> None:
    """AR(1) in error-feedback form on `ssoe()`, registering ``forecast_loc`` for the reader."""
    h = Horizon.from_data(covariates, data)
    phi = numpyro.sample("phi", dist.Uniform(-1.0, 1.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 1.0))
    y = covariates[..., : h.t_obs, :1]
    r = ssoe(
        h,
        "ar",
        y,
        jnp.zeros((1,)),
        lambda carry, _: phi * carry,
        lambda carry, y_t, eps_t, _: y_t,
        dist.Normal(0.0, sigma),
    )
    numpyro.sample("obs", dist.Normal(r.mu, sigma), obs=h.data)
    if h.future > 0:
        numpyro.deterministic("forecast_loc", r.mu_future)
        numpyro.deterministic("forecast", r.y_future)


def tiled(num_samples: int, **values: float | Array) -> dict[str, Array]:
    """A posterior dict of fixed values tiled to a leading sample axis."""
    return {
        name: jnp.broadcast_to(jnp.asarray(value), (num_samples, *jnp.shape(value)))
        for name, value in values.items()
    }


def horizon_arrays(t_obs: int = T_OBS, future: int = FUTURE) -> tuple[Array, Array, Array]:
    """``(covariates, data, truth)`` for a univariate window with an empty covariate axis."""
    covariates = empty_covariates(t_obs + future)
    data = jnp.zeros((t_obs, 1))
    truth = jnp.linspace(-1.0, 1.0, future)[:, None]
    return covariates, data, truth


# --- Readers ---------------------------------------------------------------------------------


@scaffold
def test_predict_density_known_normal_matches_closed_form() -> None:
    covariates, data, truth = horizon_arrays()
    posterior = tiled(SAMPLES, bias=0.3, sigma=1.7)
    got = forecast_log_density(random.PRNGKey(0), mean_model, posterior, data, covariates, truth)
    expected = dist.Normal(0.3, 1.7).log_prob(truth)
    assert got.shape == (SAMPLES, FUTURE, 1)
    np.testing.assert_allclose(got, jnp.broadcast_to(expected, got.shape), rtol=1e-5)


def test_predict_density_missing_site_raises() -> None:
    # A trace from a model that registers ``forecast`` itself has no ``obs_future``;
    # the error has to name the fix (ssoe_density, or a custom observed site).
    covariates, data, truth = horizon_arrays()
    full = jnp.concatenate([data, truth], axis=-2)
    model_trace = trace(
        seed(substitute(ar1_ssoe_model, {"phi": 0.5, "sigma": 1.0}), random.PRNGKey(0))
    ).get_trace(jnp.concatenate([full, covariates], axis=-1), data)
    with pytest.raises(DensitySiteError, match="ssoe_density"):
        predict_density(model_trace, truth)


@scaffold
def test_predict_density_reads_a_custom_observed_site() -> None:
    # A model that registers its own predictive over the horizon (here under the name
    # ``demand_future``) is read with ``partial(predict_density, site=...)``.
    def demand_model(covariates: Array, data: Array | None = None) -> None:
        h = Horizon.from_data(covariates, data)
        rate = numpyro.sample("rate", dist.LogNormal(0.0, 1.0))
        numpyro.sample("obs", dist.Exponential(rate).expand([h.t_obs, 1]), obs=h.data)
        if h.future > 0:
            draw = numpyro.sample("demand_future", dist.Exponential(rate).expand([h.future, 1]))
            numpyro.deterministic("forecast", draw)

    covariates, data, _ = horizon_arrays()
    truth = jnp.full((FUTURE, 1), 0.5)
    posterior = tiled(SAMPLES, rate=2.0)
    got = forecast_log_density(
        random.PRNGKey(0),
        demand_model,
        posterior,
        jnp.abs(data) + 1.0,
        covariates,
        truth,
        reader=partial(predict_density, site="demand_future"),
    )
    expected = dist.Exponential(2.0).log_prob(truth)
    np.testing.assert_allclose(got, jnp.broadcast_to(expected, got.shape), rtol=1e-5)


@scaffold
def test_mvn_obs_returns_the_joint_block_density() -> None:
    # A correlated family has the time axis in its event shape: one value per draw,
    # returned as (sample, 1, 1), equal to the conditional MVN of the block given the
    # observed prefix. The event-shape branch runs *before* log_prob: a (future, 1)
    # truth broadcast against a (future,) event would otherwise return finite garbage.
    covariates, data, truth = horizon_arrays()
    rho = 0.6
    posterior = tiled(SAMPLES, rho=rho)
    got = forecast_log_density(random.PRNGKey(0), mvn_model, posterior, data, covariates, truth)
    assert got.shape == (SAMPLES, 1, 1)

    idx = np.arange(T_OBS + FUTURE)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])
    s11, s12, s22 = cov[:T_OBS, :T_OBS], cov[:T_OBS, T_OBS:], cov[T_OBS:, T_OBS:]
    gain = np.linalg.solve(s11, s12).T
    cond_mean = gain @ np.asarray(data[:, 0])
    cond_cov = s22 - gain @ s12
    expected = dist.MultivariateNormal(jnp.asarray(cond_mean), jnp.asarray(cond_cov)).log_prob(
        truth[:, 0]
    )
    np.testing.assert_allclose(got[:, 0, 0], jnp.full((SAMPLES,), expected), rtol=1e-4)


@scaffold
def test_unsupported_event_shape_raises() -> None:
    # Neither elementwise nor the time-leading layout: refuse rather than guess.
    class Fake:
        event_shape = (FUTURE, 2)
        batch_shape = ()

        def log_prob(self, value: Array) -> Array:  # pragma: no cover - never reached
            return jnp.zeros(())

    model_trace: dict[str, Any] = {"obs_future": {"fn": Fake(), "type": "sample"}}
    with pytest.raises(ValueError, match="event shape"):
        predict_density(model_trace, jnp.zeros((FUTURE, 2)))


@scaffold
def test_ssoe_density_matches_ar1_closed_form() -> None:
    # Along simulated predecessors the h-step predictive of an AR(1) is
    # Normal(phi^h y_t, sigma^2 (1 - phi^(2h)) / (1 - phi^2)); the reader recovers it
    # from the sampled error path (reviewer probe, ssoe_density.py).
    phi, sigma, num = 0.7, 0.8, 20_000
    y_obs = jnp.full((T_OBS, 1), 2.0)
    covariates = jnp.concatenate([y_obs, jnp.zeros((FUTURE, 1))], axis=-2)
    truth = jnp.array([[1.9], [1.2], [0.4], [-0.3]])
    posterior = tiled(num, phi=phi, sigma=sigma)
    per_draw = forecast_log_density(
        random.PRNGKey(0),
        ar1_ssoe_model,
        posterior,
        y_obs,
        covariates,
        truth,
        reader=ssoe_density("ar"),
    )
    marginal = reduce_log_density(per_draw, "marginal")[:, 0]
    steps = jnp.arange(1, FUTURE + 1)
    var = sigma**2 * (1 - phi ** (2 * steps)) / (1 - phi**2)
    expected = dist.Normal(phi**steps * 2.0, jnp.sqrt(var)).log_prob(truth[:, 0])
    np.testing.assert_allclose(marginal, expected, atol=0.03)


@scaffold
def test_ssoe_density_requires_forecast_loc_deterministic() -> None:
    def bare_ar1(covariates: Array, data: Array | None = None) -> None:
        h = Horizon.from_data(covariates, data)
        sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 1.0))
        y = covariates[..., : h.t_obs, :1]
        r = ssoe(
            h,
            "ar",
            y,
            jnp.zeros((1,)),
            lambda carry, _: 0.5 * carry,
            lambda carry, y_t, eps_t, _: y_t,
            dist.Normal(0.0, sigma),
        )
        numpyro.sample("obs", dist.Normal(r.mu, sigma), obs=h.data)
        if h.future > 0:
            numpyro.deterministic("forecast", r.y_future)

    y_obs = jnp.ones((T_OBS, 1))
    covariates = jnp.concatenate([y_obs, jnp.zeros((FUTURE, 1))], axis=-2)
    with pytest.raises(DensitySiteError, match="forecast_loc"):
        forecast_log_density(
            random.PRNGKey(0),
            bare_ar1,
            tiled(8, sigma=1.0),
            y_obs,
            covariates,
            jnp.zeros((FUTURE, 1)),
            reader=ssoe_density("ar"),
        )


# --- Driver ----------------------------------------------------------------------------------


@scaffold
def test_random_walk_marginal_matches_analytic_h_step_density() -> None:
    # With the in-sample drift substituted, the level at t is known and the h-step
    # marginal is Normal(level_t, sqrt(h drift_scale^2 + sigma^2)): the Monte Carlo
    # average over prior-drawn future paths must recover it (reviewer probe, mc_error.py).
    covariates, data, truth = horizon_arrays()
    num, drift_scale, sigma = 8_000, 0.5, 0.3
    drift = jnp.full((T_OBS, 1), 0.1)
    posterior = tiled(
        num, drift_scale=drift_scale, sigma=sigma, drift_decentered=drift / drift_scale
    )
    per_draw = forecast_log_density(
        random.PRNGKey(0), rw_reparam_model, posterior, data, covariates, truth
    )
    marginal = reduce_log_density(per_draw, "marginal")[:, 0]
    level_t = float(drift.sum())
    steps = jnp.arange(1, FUTURE + 1)
    expected = dist.Normal(level_t, jnp.sqrt(steps * drift_scale**2 + sigma**2)).log_prob(
        truth[:, 0]
    )
    np.testing.assert_allclose(marginal, expected, atol=0.03)


@scaffold
def test_future_latents_are_drawn_from_the_prior_and_in_sample_sites_are_pinned() -> None:
    # Two keys give different densities when the future path is random (drift_scale 1)
    # and the same densities when it is not (drift_scale ~ 0): the seed draws the
    # ``drift_future_decentered`` auxiliary, substitute pins the in-sample sites.
    covariates, data, truth = horizon_arrays()
    for drift_scale, differs in [(1.0, True), (1e-6, False)]:
        posterior = tiled(
            16, drift_scale=drift_scale, sigma=0.5, drift_decentered=jnp.zeros((T_OBS, 1))
        )
        a = forecast_log_density(
            random.PRNGKey(1), rw_reparam_model, posterior, data, covariates, truth
        )
        b = forecast_log_density(
            random.PRNGKey(2), rw_reparam_model, posterior, data, covariates, truth
        )
        assert bool(jnp.abs(a - b).max() > 1e-3) is differs


@scaffold
def test_time_reparam_model_matches_the_plain_model_marginal() -> None:
    # time_reparam substitutes the in-sample ``drift_decentered_haar`` site and leaves
    # the future sites to the seed; the h-step marginal is unchanged.
    covariates, data, truth = horizon_arrays()
    wrapped = time_reparam(rw_reparam_model, "haar")
    plain_trace = trace(seed(rw_reparam_model, random.PRNGKey(0))).get_trace(
        covariates[:T_OBS], data
    )
    haar_trace = trace(seed(wrapped, random.PRNGKey(0))).get_trace(covariates[:T_OBS], data)
    plain_post = {
        k: v["value"]
        for k, v in plain_trace.items()
        if v["type"] == "sample" and not v["is_observed"]
    }
    haar_post = {
        k: v["value"]
        for k, v in haar_trace.items()
        if v["type"] == "sample" and not v["is_observed"]
    }
    num = 4_000
    got_plain = reduce_log_density(
        forecast_log_density(
            random.PRNGKey(0),
            rw_reparam_model,
            tiled(num, **plain_post),
            data,
            covariates,
            truth,
        )
    )
    got_haar = reduce_log_density(
        forecast_log_density(
            random.PRNGKey(0), wrapped, tiled(num, **haar_post), data, covariates, truth
        )
    )
    # Same parameters, independent paths: the two estimates agree to Monte Carlo error
    # only if the in-sample level under each parameterization is the same, which the
    # traces guarantee here (both posteriors come from the same seed).
    np.testing.assert_allclose(got_plain, got_haar, atol=0.05)


@scaffold
def test_rejects_posterior_carrying_future_sites() -> None:
    # Such a dict pins the future path by name through ``substitute`` and would return
    # the same densities for every key with no error (reviewer probe, misc_checks.py).
    covariates, data, truth = horizon_arrays()
    posterior = tiled(
        8,
        drift_scale=1.0,
        sigma=0.5,
        drift_decentered=jnp.zeros((T_OBS, 1)),
        drift_future_decentered=jnp.zeros((FUTURE, 1)),
    )
    with pytest.raises(ValueError, match="drift_future_decentered"):
        forecast_log_density(
            random.PRNGKey(0), rw_reparam_model, posterior, data, covariates, truth
        )


@scaffold
def test_rejects_truth_shape_mismatch() -> None:
    covariates, data, _ = horizon_arrays()
    with pytest.raises(ValueError, match="truth"):
        forecast_log_density(
            random.PRNGKey(0),
            mean_model,
            tiled(8, bias=0.0, sigma=1.0),
            data,
            covariates,
            jnp.zeros((FUTURE + 1, 1)),
        )


@scaffold
def test_discrete_family_requires_integer_truth() -> None:
    covariates, _, _ = horizon_arrays()
    counts = jnp.ones((T_OBS, 1), dtype=jnp.int32)
    with pytest.raises(ValueError, match="integer-dtyped"):
        forecast_log_density(
            random.PRNGKey(0),
            poisson_model,
            tiled(8, log_rate=0.0),
            counts,
            covariates,
            jnp.ones((FUTURE, 1)),
        )


@scaffold
def test_poisson_density_matches_closed_form() -> None:
    covariates, _, _ = horizon_arrays()
    counts = jnp.ones((T_OBS, 1), dtype=jnp.int32)
    truth = jnp.array([[0], [1], [2], [5]], dtype=jnp.int32)
    got = forecast_log_density(
        random.PRNGKey(0),
        poisson_model,
        tiled(8, log_rate=jnp.log(1.5)),
        counts,
        covariates,
        truth,
    )
    expected = dist.Poisson(1.5).log_prob(truth)
    np.testing.assert_allclose(got, jnp.broadcast_to(expected, got.shape), rtol=1e-5)


@scaffold
def test_batch_dims_follow_the_series_plate() -> None:
    covariates = jnp.zeros((3, T_OBS + FUTURE, 0))
    data = jnp.zeros((3, T_OBS, 1))
    truth = jnp.zeros((3, FUTURE, 1))
    posterior = tiled(SAMPLES, bias=jnp.array([-1.0, 0.0, 1.0])[:, None, None], sigma=1.0)
    got = forecast_log_density(random.PRNGKey(0), series_model, posterior, data, covariates, truth)
    assert got.shape == (SAMPLES, 3, FUTURE, 1)
    expected = dist.Normal(jnp.array([-1.0, 0.0, 1.0])[:, None, None], 1.0).log_prob(truth)
    np.testing.assert_allclose(got, jnp.broadcast_to(expected, got.shape), rtol=1e-5)


@scaffold
def test_chunked_matches_unchunked_without_future_randomness(count_compilations) -> None:
    # The mean model has no future latents, so chunking cannot change the values; and
    # every chunk shares one shape, so the jitted kernel compiles once for the loop.
    covariates, data, truth = horizon_arrays()
    posterior = tiled(SAMPLES, bias=0.3, sigma=1.7)
    whole = forecast_log_density(random.PRNGKey(0), mean_model, posterior, data, covariates, truth)
    jax.block_until_ready(whole)
    with count_compilations() as tally:
        chunked = forecast_log_density(
            random.PRNGKey(0), mean_model, posterior, data, covariates, truth, batch_size=20
        )
        jax.block_until_ready(chunked)
    np.testing.assert_allclose(chunked, whole, rtol=1e-6)
    assert tally.count <= 1


@scaffold
def test_parallel_false_matches_parallel_true_without_future_randomness() -> None:
    covariates, data, truth = horizon_arrays()
    posterior = tiled(SAMPLES, bias=0.3, sigma=1.7)
    vmapped = forecast_log_density(
        random.PRNGKey(0), mean_model, posterior, data, covariates, truth
    )
    mapped = forecast_log_density(
        random.PRNGKey(0), mean_model, posterior, data, covariates, truth, parallel=False
    )
    np.testing.assert_allclose(mapped, vmapped, rtol=1e-6)


@scaffold
@pytest.mark.parametrize("device", ["host", "numpy"])
def test_host_placement_and_numpy_posterior(device: str) -> None:
    covariates, data, truth = horizon_arrays()
    posterior = {k: np.asarray(v) for k, v in tiled(SAMPLES, bias=0.3, sigma=1.7).items()}
    got = forecast_log_density(
        random.PRNGKey(0), mean_model, posterior, data, covariates, truth, device=device
    )
    if device == "host":
        assert_host_resident(got)
    else:
        assert_numpy_host(got)
    expected = dist.Normal(0.3, 1.7).log_prob(truth)
    np.testing.assert_allclose(np.asarray(got), np.broadcast_to(expected, got.shape), rtol=1e-5)


# --- Reductions ------------------------------------------------------------------------------


@scaffold
def test_reduce_marginal_is_log_mean_exp_per_cell() -> None:
    per_draw = random.normal(random.PRNGKey(0), (50, 2, FUTURE, 1)) - 2.0
    got = reduce_log_density(per_draw, "marginal")
    expected = np.log(np.mean(np.exp(np.asarray(per_draw)), axis=0))
    assert got.shape == (2, FUTURE, 1)
    np.testing.assert_allclose(got, expected, rtol=1e-5)


@scaffold
def test_reduce_joint_sums_over_time_before_averaging() -> None:
    per_draw = random.normal(random.PRNGKey(0), (50, 2, FUTURE, 1)) - 2.0
    got = reduce_log_density(per_draw, "joint")
    expected = np.log(np.mean(np.exp(np.asarray(per_draw).sum(axis=-2, keepdims=True)), axis=0))
    assert got.shape == (2, 1, 1)
    np.testing.assert_allclose(got, expected, rtol=1e-5)


@scaffold
def test_reductions_coincide_at_horizon_one() -> None:
    per_draw = random.normal(random.PRNGKey(0), (50, 1, 1)) - 2.0
    np.testing.assert_allclose(
        reduce_log_density(per_draw, "marginal"), reduce_log_density(per_draw, "joint"), rtol=1e-6
    )


@scaffold
def test_reduce_is_stable_for_tiny_densities() -> None:
    per_draw = random.normal(random.PRNGKey(0), (50, FUTURE, 1))
    shifted = reduce_log_density(per_draw - 800.0, "marginal")
    assert bool(jnp.all(jnp.isfinite(shifted)))
    np.testing.assert_allclose(
        shifted + 800.0, reduce_log_density(per_draw, "marginal"), atol=1e-4
    )


def test_reduce_rejects_unknown_reduction_at_the_type_boundary() -> None:
    # ``Reduction`` is a Literal and the beartype import hook enforces it at the call
    # boundary, so a misspelled reduction never reaches the kernel.
    with pytest.raises(TypeCheckError):
        reduce_log_density(jnp.zeros((5, FUTURE, 1)), "mean")  # type: ignore[arg-type]


@scaffold
def test_log_density_ess_bounds_and_extremes() -> None:
    equal = jnp.zeros((50, FUTURE, 1))
    np.testing.assert_allclose(log_density_ess(equal), jnp.full((FUTURE, 1), 50.0), rtol=1e-6)
    one_dominant = jnp.zeros((50, FUTURE, 1)).at[0].set(100.0)
    np.testing.assert_allclose(log_density_ess(one_dominant), jnp.ones((FUTURE, 1)), atol=1e-6)
    noisy = log_density_ess(random.normal(random.PRNGKey(0), (50, FUTURE, 1)))
    assert bool(jnp.all((noisy >= 1.0) & (noisy <= 50.0)))


@scaffold
def test_reduce_accepts_numpy_input() -> None:
    per_draw = np.random.default_rng(0).normal(size=(50, FUTURE, 1)) - 2.0
    got = reduce_log_density(per_draw)
    np.testing.assert_allclose(got, np.log(np.mean(np.exp(per_draw), axis=0)), rtol=1e-5)


# --- Conditional (chain-rule) route ----------------------------------------------------------


@scaffold
def test_conditional_log_density_equals_full_window_obs_slice() -> None:
    # For an ssoe() model the block conditioned on the realized values is the in-sample
    # filter extended by the block: the result equals the ``obs`` log density of a run on
    # the concatenated series, sliced to the last ``future`` rows.
    y_obs = jnp.cumsum(0.3 * random.normal(random.PRNGKey(0), (T_OBS, 1)), axis=0)
    truth = y_obs[-1] + jnp.array([[0.2], [-0.1], [0.4], [0.0]])
    full = jnp.concatenate([y_obs, truth], axis=-2)
    covariates = full
    posterior = {"phi": jnp.array([0.4, 0.8]), "sigma": jnp.array([0.5, 1.0])}
    got = conditional_log_density(ar1_ssoe_model, posterior, y_obs, covariates, truth)
    reference = log_likelihood(ar1_ssoe_model, posterior, covariates, full)["obs"][:, T_OBS:, :]
    assert got.shape == (2, FUTURE, 1)
    np.testing.assert_allclose(got, reference, rtol=1e-6)


@scaffold
def test_conditional_marginal_at_horizon_one_matches_forecast_density() -> None:
    # At one step the chain-rule term and the simulated-predecessor marginal are the same
    # quantity; the two drivers must agree to Monte Carlo error.
    y_obs = jnp.cumsum(0.3 * random.normal(random.PRNGKey(0), (T_OBS, 1)), axis=0)
    truth = y_obs[-1:] + 0.2
    covariates = jnp.concatenate([y_obs, truth], axis=-2)
    posterior = tiled(4_000, phi=0.6, sigma=0.7)
    exact = reduce_log_density(
        conditional_log_density(ar1_ssoe_model, posterior, y_obs, covariates, truth)
    )
    sampled = reduce_log_density(
        forecast_log_density(
            random.PRNGKey(1),
            ar1_ssoe_model,
            posterior,
            y_obs,
            covariates,
            truth,
            reader=ssoe_density("ar"),
        )
    )
    np.testing.assert_allclose(sampled, exact, atol=0.02)


@scaffold
def test_conditional_log_density_rejects_time_indexed_latents() -> None:
    covariates, data, truth = horizon_arrays()
    posterior = tiled(8, drift_scale=1.0, sigma=0.5, drift_decentered=jnp.zeros((T_OBS, 1)))
    with pytest.raises(ValueError, match="drift_decentered"):
        conditional_log_density(rw_reparam_model, posterior, data, covariates, truth)


# --- Backtest hook ---------------------------------------------------------------------------


def _forecast_output_fn(num_steps: int = 30) -> ForecastFn:
    """A `ForecastFn` returning `ForecastOutput` by wrapping the shared SVI closure."""
    bare = svi_forecast_fn(num_steps)

    def forecast_fn(
        rng_key: Array,
        model: ForecastModel,
        train_data: Array,
        train_covariates: Array,
        full_covariates: Array,
        num_samples: int,
        *,
        batch_size: int | None = None,
    ) -> ForecastOutput:
        # Refit through the shared closure for the draws, and expose a posterior of the
        # right shape for the density pass; a real closure fits once and returns both.
        draws = bare(
            rng_key,
            model,
            train_data,
            train_covariates,
            full_covariates,
            num_samples,
            batch_size=batch_size,
        )
        posterior = {
            "drift_scale": jnp.full((num_samples,), 0.3),
            "sigma": jnp.full((num_samples,), 0.5),
            "drift": jnp.zeros((num_samples, train_data.shape[-2], 1)),
        }
        return ForecastOutput(draws=draws, posterior=posterior)

    return forecast_fn


def test_backtest_log_density_requires_forecast_output(rng_key: Array) -> None:
    from conftest import rw_model

    data = jnp.cumsum(0.1 * random.normal(rng_key, (30, 1)), axis=-2)
    with pytest.raises(ValueError, match="ForecastOutput"):
        backtest(
            rng_key,
            lambda: rw_model,
            data,
            empty_covariates(30),
            forecast_fn=svi_forecast_fn(5),
            min_train_window=20,
            test_window=4,
            stride=4,
            num_samples=8,
            log_density="marginal",
        )


def test_backtest_forecast_output_without_log_density_matches_bare_draws(rng_key: Array) -> None:
    # Returning ForecastOutput changes nothing unless a density is requested: the draws
    # are read and the posterior ignored, so the metrics equal the bare-draws run.
    from conftest import rw_model

    data = jnp.cumsum(0.1 * random.normal(rng_key, (30, 1)), axis=-2)
    kwargs: dict[str, Any] = {
        "min_train_window": 20,
        "test_window": 4,
        "stride": 4,
        "num_samples": 8,
    }
    bare = backtest(
        rng_key,
        lambda: rw_model,
        data,
        empty_covariates(30),
        forecast_fn=svi_forecast_fn(5),
        **kwargs,
    )
    wrapped = backtest(
        rng_key,
        lambda: rw_model,
        data,
        empty_covariates(30),
        forecast_fn=_forecast_output_fn(5),
        **kwargs,
    )
    assert [r.metrics for r in bare] == [r.metrics for r in wrapped]
    assert all(r.log_density is None and r.log_density_reduction is None for r in wrapped)


@scaffold
@pytest.mark.parametrize("reduction", ["marginal", "joint"])
def test_backtest_stores_reduced_log_density_with_its_label(
    rng_key: Array, reduction: str
) -> None:
    from conftest import rw_model

    data = jnp.cumsum(0.1 * random.normal(rng_key, (30, 1)), axis=-2)
    results = backtest(
        rng_key,
        lambda: rw_model,
        data,
        empty_covariates(30),
        forecast_fn=_forecast_output_fn(5),
        min_train_window=20,
        test_window=4,
        stride=4,
        num_samples=8,
        log_density=reduction,
    )
    expected_shape = (4, 1) if reduction == "marginal" else (1, 1)
    for result in results:
        assert result.log_density is not None
        assert result.log_density.shape == expected_shape
        assert result.log_density_reduction == reduction
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(result.log_density))))


@scaffold
def test_backtest_log_density_changes_no_other_field(rng_key: Array) -> None:
    # The density pass uses its own key substream and the untransformed truth, so metrics
    # and predictions are identical with and without it, and a transform that rescales
    # the metrics leaves the density alone.
    from conftest import rw_model

    data = jnp.cumsum(0.1 * random.normal(rng_key, (30, 1)), axis=-2)
    kwargs: dict[str, Any] = {
        "min_train_window": 20,
        "test_window": 4,
        "stride": 4,
        "num_samples": 8,
        "keep_predictions": True,
        "forecast_fn": _forecast_output_fn(5),
    }
    without = backtest(rng_key, lambda: rw_model, data, empty_covariates(30), **kwargs)
    with_density = backtest(
        rng_key, lambda: rw_model, data, empty_covariates(30), log_density="marginal", **kwargs
    )
    scaled = backtest(
        rng_key,
        lambda: rw_model,
        data,
        empty_covariates(30),
        log_density="marginal",
        transform=lambda pred, truth: (10.0 * pred, 10.0 * truth),
        **kwargs,
    )
    for a, b, c in zip(without, with_density, scaled, strict=True):
        assert a.metrics == b.metrics
        np.testing.assert_array_equal(np.asarray(a.prediction), np.asarray(b.prediction))
        assert c.metrics["mae"] == pytest.approx(10.0 * a.metrics["mae"], rel=1e-5)
        np.testing.assert_allclose(np.asarray(b.log_density), np.asarray(c.log_density))


@scaffold
def test_backtest_vectorized_log_density_shape_and_label() -> None:
    from conftest import rw_model
    from numpyro.infer.autoguide import AutoNormal

    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (40, 1)), axis=-2)
    result = backtest_vectorized(
        random.PRNGKey(1),
        lambda: rw_model,
        data,
        empty_covariates(40),
        train_window=20,
        test_window=5,
        stride=5,
        guide=AutoNormal(rw_model),
        num_steps=20,
        num_samples=8,
        log_density="marginal",
    )
    assert result.log_density is not None
    assert result.log_density.shape == (result.t0.shape[0], 5, 1)
    assert result.log_density_reduction == "marginal"
    assert bool(jnp.all(jnp.isfinite(result.log_density)))


def test_backtest_vectorized_log_density_default_is_absent() -> None:
    from conftest import rw_model
    from numpyro.infer.autoguide import AutoNormal

    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (40, 1)), axis=-2)
    result = backtest_vectorized(
        random.PRNGKey(1),
        lambda: rw_model,
        data,
        empty_covariates(40),
        train_window=20,
        test_window=5,
        guide=AutoNormal(rw_model),
        num_steps=5,
        num_samples=4,
    )
    assert result.log_density is None and result.log_density_reduction is None


# --- Sanity of the per-cell quantity itself --------------------------------------------------


@scaffold
def test_marginal_equals_logsumexp_identity_on_the_driver_output() -> None:
    # The reduction is exactly logsumexp over the draw axis minus log(S): pin the formula
    # on real driver output (not on random numbers) so the two never drift apart.
    covariates, data, truth = horizon_arrays()
    posterior = tiled(SAMPLES, drift_scale=0.5, sigma=0.3, drift_decentered=jnp.zeros((T_OBS, 1)))
    per_draw = forecast_log_density(
        random.PRNGKey(0), rw_reparam_model, posterior, data, covariates, truth
    )
    expected = logsumexp(per_draw, axis=0) - jnp.log(SAMPLES)
    np.testing.assert_allclose(reduce_log_density(per_draw), expected, rtol=1e-6)


def test_condition_substitute_seed_pattern_pins_only_in_sample_sites() -> None:
    # The mechanism the driver is built on, checked at trace level without the driver:
    # the in-sample sites take the substituted values, the future auxiliary is drawn.
    covariates, data, truth = horizon_arrays()
    draw = {"drift_scale": 0.5, "sigma": 0.3, "drift_decentered": jnp.ones((T_OBS, 1))}
    model = condition(substitute(rw_reparam_model, draw), {"obs_future": truth})
    tr = trace(seed(model, random.PRNGKey(0))).get_trace(covariates, data)
    np.testing.assert_array_equal(tr["drift_decentered"]["value"], jnp.ones((T_OBS, 1)))
    assert tr["drift_future_decentered"]["value"].shape == (FUTURE, 1)
    assert tr["obs_future"]["is_observed"]
    np.testing.assert_array_equal(tr["obs_future"]["value"], truth)
    assert tr["obs_future"]["fn"].log_prob(truth).shape == (FUTURE, 1)
