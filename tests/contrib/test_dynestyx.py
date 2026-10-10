"""Tests for the dynestyx state space building block (``numpyro_forecast.contrib.dynestyx``)."""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random
from numpyro.infer.util import log_density

from numpyro_forecast import Horizon, backtest, forecast, predict_in_sample, to_datatree
from numpyro_forecast.contrib.dynestyx import state_space
from numpyro_forecast.optional import _api_canary
from numpyro_forecast.typing import Array, ForecastModel

pytest.importorskip("dynestyx")
from dynestyx import DiscreteTimeSimulator, DynamicalModel, Filter, LatentPathBuilder, Smoother
from dynestyx.inference.configs.smoother import KFSmootherConfig
from dynestyx.inference.filters import KFConfig
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


def _repeat(params: dict[str, Array], n: int) -> dict[str, Array]:
    """A posterior of ``n`` identical draws, so predictive draws are iid at fixed parameters."""
    return {k: jnp.broadcast_to(v, (n, *v.shape)) for k, v in params.items()}


def _smoothed_moments() -> tuple[Array, Array]:
    """Smoothed state mean and variance at ``PARAMS``, ``(t_obs, 1)`` each, from dynestyx's sites."""
    with smoother:
        tr = numpyro.handlers.trace(
            numpyro.handlers.substitute(
                numpyro.handlers.seed(local_level, random.PRNGKey(0)), data=PARAMS
            )
        ).get_trace(TRAIN, TRAIN)
    return tr["f_smoothed_states_mean"]["value"], tr["f_smoothed_states_cov_diag"]["value"]


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


def test_in_sample_predictive_matches_the_smoothing_moments() -> None:
    """Under a Smoother the ``obs`` draws have mean ``m_t`` and variance ``P_t + r^2`` per step."""
    n = 4_000
    with smoother:
        draws = predict_in_sample(random.PRNGKey(1), local_level, _repeat(PARAMS, n), TRAIN)
    mean, var = _smoothed_moments()
    predictive_var = var + R_TRUE**2
    assert draws.shape == (n, T_OBS, 1)
    assert jnp.allclose(draws.mean(axis=0), mean, atol=4 * jnp.sqrt(predictive_var / n).max())
    assert jnp.allclose(draws.var(axis=0), predictive_var, rtol=0.15)


def test_in_sample_predictive_replays_the_latent_path() -> None:
    """Under a LatentPathBuilder the ``obs`` draws are the posterior path plus observation noise.

    The builder must be the instance that fitted the model: it caches the observation
    missingness layout from the concrete observations of the fit, which the jitted
    predictive cannot infer from traced arrays. The eager trace below stands in for the fit.
    """
    n = 2_000
    path = jnp.linspace(-1.0, 1.0, T_OBS)[:, None]
    posterior = _repeat({**PARAMS, "f_state_path_params": path}, n)
    builder = LatentPathBuilder()
    with builder:
        numpyro.handlers.trace(numpyro.handlers.seed(local_level, random.PRNGKey(0))).get_trace(
            TRAIN, TRAIN
        )
        draws = predict_in_sample(random.PRNGKey(1), local_level, posterior, TRAIN)
    residual = draws - path
    assert jnp.allclose(residual.mean(axis=0), 0.0, atol=4 * R_TRUE / jnp.sqrt(n))
    assert jnp.allclose(residual.std(axis=0), R_TRUE, rtol=0.1)


def test_forecast_rolls_the_smoothed_state_forward() -> None:
    """``forecast`` reads the Simulator rollout: ``(sample, future, obs)`` with the random walk law."""
    n = 2_000
    with DiscreteTimeSimulator(n_simulations=1), smoother:
        draws = forecast(random.PRNGKey(2), local_level, _repeat(PARAMS, n), TRAIN, COVARIATES)
    mean, var = _smoothed_moments()
    assert draws.shape == (n, FUTURE, 1)
    # a random walk's forecast mean is the last smoothed state, the variance grows by q^2 per step
    expected_var = var[-1] + Q_TRUE**2 * jnp.arange(1, FUTURE + 1)[:, None] + R_TRUE**2
    assert jnp.allclose(draws.mean(axis=0), mean[-1], atol=4 * jnp.sqrt(expected_var / n).max())
    assert jnp.allclose(draws.var(axis=0), expected_var, rtol=0.15)


def test_to_datatree_exports_both_predictive_groups() -> None:
    """One handler stack serves the in-sample predictive and the forecast of ``to_datatree``."""
    with DiscreteTimeSimulator(n_simulations=1), smoother:
        tree = to_datatree(
            random.PRNGKey(3), local_level, _repeat(PARAMS, 8), TRAIN, COVARIATES, num_chains=2
        )
    assert tree["posterior_predictive"]["obs"].shape == (2, 4, T_OBS, 1)
    assert tree["predictions"]["obs"].shape == (2, 4, FUTURE, 1)
    assert np.isfinite(tree["posterior_predictive"]["obs"].to_numpy()).all()


def test_missing_observations_widen_the_in_sample_band() -> None:
    """NaN rows are skipped by the cuthbert smoother and the predictive is widest inside the gap."""
    gap = slice(8, 14)
    train = TRAIN.at[gap].set(jnp.nan)
    with smoother:
        draws = predict_in_sample(random.PRNGKey(4), local_level, _repeat(PARAMS, 2_000), train)
    sd = draws.std(axis=0)[:, 0]
    assert jnp.isfinite(draws).all()
    assert sd[gap].min() > sd[: gap.start].max()


def test_in_sample_predictive_rejects_a_filter() -> None:
    """A Filter has only the filtering marginals: the block refuses and points at the Smoother."""
    with (
        Filter(filter_config=KFConfig(filter_source="cuthbert")),
        pytest.raises(ValueError, match="Smoother"),
    ):
        predict_in_sample(random.PRNGKey(5), local_level, _repeat(PARAMS, 2), TRAIN)


def test_forecast_requires_a_single_simulation() -> None:
    """``n_simulations != 1`` is rejected: the package's forecast site has no simulation axis."""
    with (
        DiscreteTimeSimulator(n_simulations=2),
        smoother,
        pytest.raises(ValueError, match="n_simulations=1"),
    ):
        forecast(random.PRNGKey(6), local_level, _repeat(PARAMS, 2), TRAIN, COVARIATES)


def test_window_shapes_are_validated() -> None:
    """The time axes must match the horizon; a batch dim fails the jaxtyping rank check."""
    h = Horizon.from_data(COVARIATES, TRAIN)
    dynamics = local_level_dynamics(jnp.asarray(Q_TRUE), jnp.asarray(R_TRUE))
    with pytest.raises(ValueError, match="t_obs"):
        state_space(h, "f", TRAIN[:-1], dynamics)
    with pytest.raises(ValueError, match="duration"):
        state_space(h, "f", TRAIN, dynamics, controls=jnp.zeros((T_OBS, 2)))
    with pytest.raises(TypeError):  # jaxtyping.TypeCheckError: " time obs" is exactly 2-D
        state_space(h, "f", TRAIN[None], dynamics)


def test_backtest_scores_in_sample_under_one_handler_stack() -> None:
    """``backtest(eval_train=True)`` runs ``predict_in_sample`` and ``forecast`` through the block."""

    def forecast_fn(
        rng_key: Array,
        model: ForecastModel,
        train: Array,
        train_covariates: Array,
        full_covariates: Array,
        num_samples: int,
        /,
        *,
        batch_size: int | None = None,
    ) -> Array | np.ndarray:
        return forecast(rng_key, model, _repeat(PARAMS, num_samples), train, full_covariates)

    def in_sample_fn(
        rng_key: Array,
        model: ForecastModel,
        train: Array,
        train_covariates: Array,
        num_samples: int,
        /,
        *,
        batch_size: int | None = None,
    ) -> Array | np.ndarray:
        return predict_in_sample(rng_key, model, _repeat(PARAMS, num_samples), train_covariates)

    with DiscreteTimeSimulator(n_simulations=1), smoother:
        results = backtest(
            random.PRNGKey(7),
            lambda: local_level,
            SERIES,
            COVARIATES,
            forecast_fn=forecast_fn,
            in_sample_fn=in_sample_fn,
            train_window=12,
            test_window=6,
            stride=6,
            num_samples=20,
            eval_train=True,
        )
    assert len(results) == 3
    assert all(np.isfinite(r.metrics["crps"]) for r in results)
    assert all(np.isfinite(r.train_metrics["crps"]) for r in results)


def test_dynestyx_surface_canary() -> None:
    """The attribute and site-name surface the block depends on, pinned against drift."""
    _api_canary(
        "dynestyx",
        ["sample", "types.ConditionedResult", "types.LatentStateResult", "DiscreteTimeSimulator"],
    )
    with DiscreteTimeSimulator(n_simulations=1), smoother:
        tr = numpyro.handlers.trace(
            numpyro.handlers.substitute(
                numpyro.handlers.seed(local_level, random.PRNGKey(0)), data=PARAMS
            )
        ).get_trace(COVARIATES, TRAIN)
    assert tr["f_smoothed_states_mean"]["value"].shape == (T_OBS, 1)
    assert tr["f_predicted_observations"]["value"].shape == (1, FUTURE + 1, 1)
    assert tr["forecast"]["value"].shape == (FUTURE, 1)
