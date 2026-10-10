"""Tests for the dynestyx state space building block (``numpyro_forecast.contrib.dynestyx``)."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random
from numpyro.infer.util import log_density

from numpyro_forecast import Horizon
from numpyro_forecast.contrib.dynestyx import state_space
from numpyro_forecast.typing import Array

pytest.importorskip("dynestyx")
from dynestyx import DynamicalModel, Smoother
from dynestyx.inference.configs.smoother import KFSmootherConfig
from dynestyx.models import LinearGaussianObservation, LinearGaussianStateEvolution

T_OBS, FUTURE = 24, 6
Q_TRUE, R_TRUE, P0 = 0.3, 0.2, 4.0


def local_level_dynamics(q: Array, r: Array) -> DynamicalModel:
    """Random walk level observed with noise: ``x_t = x_{t-1} + q e_t``, ``y_t = x_t + r v_t``."""
    return DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), P0 * jnp.eye(1)),
        state_evolution=LinearGaussianStateEvolution(A=jnp.eye(1), cov=q**2 * jnp.eye(1)),
        observation_model=LinearGaussianObservation(H=jnp.eye(1), R=r**2 * jnp.eye(1)),
    )


def local_level(covariates: Array, data: Array | None = None) -> None:
    """The series travels in the covariates (column 0), as with ``ssoe``."""
    h = Horizon.from_data(covariates, data)
    q = jnp.asarray(numpyro.sample("q", dist.HalfNormal(1.0)))
    r = jnp.asarray(numpyro.sample("r", dist.HalfNormal(1.0)))
    state_space(h, "f", covariates[..., : h.t_obs, :], local_level_dynamics(q, r))


def simulate(rng_key: Array, n: int) -> Array:
    """A random walk observed with noise; dynestyx observes the initial state at the first time."""
    key_x0, key_q, key_r = random.split(rng_key, 3)
    x0 = jnp.sqrt(P0) * random.normal(key_x0)
    steps = jnp.concatenate([jnp.zeros(1), jnp.cumsum(Q_TRUE * random.normal(key_q, (n - 1,)))])
    return (x0 + steps + R_TRUE * random.normal(key_r, (n,)))[:, None]


SERIES = simulate(random.PRNGKey(0), T_OBS + FUTURE)
TRAIN, COVARIATES = SERIES[:T_OBS], SERIES
PARAMS = {"q": jnp.asarray(Q_TRUE), "r": jnp.asarray(R_TRUE)}
smoother = Smoother(smoother_config=KFSmootherConfig(filter_source="cuthbert"))


def closed_form_loglik(y: Array, q: float, r: float) -> Array:
    """Marginal Gaussian log likelihood of a random walk plus noise observed at ``0..T-1``."""
    t = jnp.arange(y.shape[0])
    cov = P0 + q**2 * jnp.minimum(t[:, None], t[None, :]) + r**2 * jnp.eye(y.shape[0])
    return dist.MultivariateNormal(jnp.zeros(y.shape[0]), cov).log_prob(y[:, 0])


def test_training_registers_the_marginal_likelihood() -> None:
    """Under a Smoother the model's log density is the priors plus the exact Kalman marginal."""
    with smoother:
        joint, tr = log_density(local_level, (TRAIN, TRAIN), {}, PARAMS)
    priors = dist.HalfNormal(1.0).log_prob(PARAMS["q"]) + dist.HalfNormal(1.0).log_prob(
        PARAMS["r"]
    )
    assert jnp.allclose(joint, priors + closed_form_loglik(TRAIN, Q_TRUE, R_TRUE), rtol=1e-4)
    assert "obs" not in tr
    assert "forecast" not in tr
