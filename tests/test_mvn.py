"""MultivariateNormal distribution-surgery tests (roadmap §9)."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array, random

from numpyro_forecast.functional import (
    Horizon,
    draw_posterior,
    fit_svi,
    forecasting_model,
    predict,
)
from numpyro_forecast.util import prefix_condition, shift_loc, slice_time
from tests.conftest import empty_covariates


def _ar_covariance(time: int, phi: float) -> Array:
    idx = jnp.arange(time)
    return phi ** jnp.abs(idx[:, None] - idx[None, :])


def test_mvn_prefix_matches_numpy_closed_form() -> None:
    """Cholesky conditional matches a direct numpy solve on an AR covariance."""
    time, t = 5, 3
    phi = 0.6
    loc = jnp.zeros(time)
    cov = _ar_covariance(time, phi)
    mvn = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)
    data = jnp.array([[0.2], [0.1], [-0.3]])
    cond = prefix_condition(mvn, data)
    assert isinstance(cond, dist.MultivariateNormal)
    mu_f = loc[t:]
    sigma_pp = cov[:t, :t]
    sigma_fp = cov[t:, :t]
    expected_mean = mu_f + sigma_fp @ jnp.linalg.solve(sigma_pp, data[:, 0] - loc[:t])
    assert jnp.allclose(cond.loc, expected_mean, atol=1e-4)


def test_mvn_diagonal_matches_independent_normal_slice() -> None:
    """``MVN(mu, sigma^2 I)`` conditional equals ``Independent(Normal)`` slice."""
    time, t = 6, 4
    loc = jnp.arange(float(time))
    cov = jnp.eye(time)
    mvn = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)
    indep = dist.Normal(loc=loc[:, None], scale=1.0)
    data = jnp.zeros((t, 1))
    mvn_future = prefix_condition(mvn, data)
    norm_future = prefix_condition(indep, data)
    grid = jnp.linspace(-2.0, 2.0, 7)
    for x in grid:
        xv = jnp.full((time - t, 1), x)
        assert jnp.allclose(
            mvn_future.log_prob(xv[:, 0]),
            norm_future.log_prob(xv).sum(),
            atol=1e-5,
        )


def test_mvn_near_singular_prefix_still_valid() -> None:
    """A near-singular ``Sigma_pp`` yields a valid distribution via jitter."""
    time, t = 4, 2
    cov = jnp.ones((time, time)) * 0.99 + jnp.eye(time) * 0.01
    mvn = dist.MultivariateNormal(loc=jnp.zeros(time), covariance_matrix=cov)
    data = jnp.zeros((t, 1))
    cond = prefix_condition(mvn, data)
    assert jnp.all(jnp.isfinite(cond.covariance_matrix))
    assert jnp.all(jnp.linalg.eigvalsh(cond.covariance_matrix) > 0)


def test_mvn_shift_loc_adds_to_mean() -> None:
    loc = jnp.zeros(4)
    cov = jnp.eye(4)
    mvn = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)
    shift = jnp.ones((4, 1))
    shifted = shift_loc(mvn, shift)
    assert jnp.allclose(shifted.loc, jnp.ones(4))


def test_mvn_slice_time_marginal_block() -> None:
    loc = jnp.arange(6.0)
    cov = _ar_covariance(6, 0.5)
    mvn = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)
    sliced = slice_time(mvn, slice(1, 4))
    assert sliced.loc.shape == (3,)
    assert sliced.covariance_matrix.shape == (3, 3)


def test_gp_noise_end_to_end() -> None:
    """GP-style correlated noise fits through ``fit_svi``."""

    def gp_body(h: Horizon, covariates: Array) -> None:
        time = h.duration
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        rho = numpyro.sample("rho", dist.Beta(2.0, 2.0))
        cov = _ar_covariance(time, rho) * sigma**2
        noise = dist.MultivariateNormal(loc=jnp.zeros(time), covariance_matrix=cov)
        level = numpyro.sample("level", dist.Normal(0.0, 1.0))
        predict(h, noise, level + jnp.zeros((time, 1)))

    model = forecasting_model(gp_body)
    data = random.normal(random.PRNGKey(0), (12, 1))
    cov = empty_covariates(16)
    fit = fit_svi(random.PRNGKey(1), model, data, cov[:12], num_steps=150)
    post = draw_posterior(random.PRNGKey(2), fit, 30)
    assert set(post) >= {"sigma", "rho", "level"}
    assert jnp.all(jnp.isfinite(post["sigma"]))
