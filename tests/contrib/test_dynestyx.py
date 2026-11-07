"""Tests for the dynestyx state space building block (``numpyro_forecast.contrib.dynestyx``)."""

import warnings
from typing import Literal

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.inspect import get_model_relations
from numpyro.infer.util import log_density
from numpyro.optim import Adam

from numpyro_forecast import (
    Horizon,
    backtest,
    draw_posterior,
    forecast,
    predict_in_sample,
    to_datatree,
)
from numpyro_forecast.contrib.dynestyx import StateSpaceResult, state_space
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
# u_t = 100 t enters through D = 1, so any misalignment of the controls is a 100-unit error
CONTROL = 100.0 * jnp.arange(T_OBS + FUTURE, dtype=jnp.float32)[:, None]
CONTROLLED = jnp.concatenate([SERIES + CONTROL, CONTROL], axis=-1)  # column 0 series, 1 control


def level_with_controls(covariates: Array, data: Array | None = None) -> None:
    """The local level at the true scales, observed through ``D = 1`` of covariate column 1."""
    h = Horizon.from_data(covariates, data)
    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), P0 * jnp.eye(1)),
        state_evolution=LinearGaussianStateEvolution(A=jnp.eye(1), cov=Q_TRUE**2 * jnp.eye(1)),
        observation_model=LinearGaussianObservation(
            H=jnp.eye(1), R=R_TRUE**2 * jnp.eye(1), D=jnp.eye(1)
        ),
        control_dim=1,
    )
    state_space(h, "f", covariates[..., : h.t_obs, :1], dynamics, controls=covariates[..., 1:])


def closed_form_loglik(y: Array, q: float, r: float) -> Array:
    """Marginal Gaussian log likelihood of a random walk plus noise observed at ``0..T-1``."""
    t = jnp.arange(y.shape[0])
    cov = P0 + q**2 * jnp.minimum(t[:, None], t[None, :]) + r**2 * jnp.eye(y.shape[0])
    return dist.MultivariateNormal(jnp.zeros(y.shape[0]), cov).log_prob(y[:, 0])


def _repeat(params: dict[str, Array], n: int) -> dict[str, Array]:
    """A posterior of ``n`` identical draws, so predictive draws are iid at fixed parameters."""
    return {k: jnp.broadcast_to(v, (n, *v.shape)) for k, v in params.items()}


def _smoothed_moments() -> tuple[Array, Array]:
    """Smoothed state mean and variance at ``PARAMS``, ``(t_obs, 1)`` each, from the sites."""
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
    """``forecast`` reads the Simulator rollout: ``(sample, future, obs)``, random walk law."""
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
    """NaN rows are skipped by the cuthbert smoother; the predictive is widest inside the gap."""
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


def test_forecast_under_a_filter_matches_the_smoother() -> None:
    """A Filter forecasts inside the jitted driver, with the law of the Smoother's forecast.

    The rollout starts from the conditioned state at the last observation, where the
    filtering and smoothing distributions coincide. The observation grid is concrete under
    the driver's jit, which the Filter's host-side rollout needs.
    """
    n = 4_000
    draws = {}
    handlers = {
        "filter": Filter(filter_config=KFConfig(filter_source="cuthbert")),
        "smoother": smoother,
    }
    for label, handler in handlers.items():
        with DiscreteTimeSimulator(n_simulations=1), handler:
            draws[label] = forecast(
                random.PRNGKey(2), local_level, _repeat(PARAMS, n), TRAIN, COVARIATES
            )
    sd = draws["smoother"].std(axis=0)
    assert jnp.allclose(
        draws["filter"].mean(axis=0),
        draws["smoother"].mean(axis=0),
        atol=4 * sd.max() / jnp.sqrt(n),
    )
    assert jnp.allclose(draws["filter"].std(axis=0), sd, rtol=0.1)


def test_forecast_requires_a_single_simulation() -> None:
    """``n_simulations != 1`` is rejected: the package's forecast site has no simulation axis."""
    with (
        DiscreteTimeSimulator(n_simulations=2),
        smoother,
        pytest.raises(ValueError, match="n_simulations=1"),
    ):
        forecast(random.PRNGKey(6), local_level, _repeat(PARAMS, 2), TRAIN, COVARIATES)


def test_forecast_rolls_the_latent_path_forward() -> None:
    """Under a LatentPathBuilder the forecast starts from the final path state and random walks."""
    n = 2_000
    path = jnp.linspace(-1.0, 1.0, T_OBS)[:, None]
    posterior = _repeat({**PARAMS, "f_state_path_params": path}, n)
    builder = LatentPathBuilder()
    with builder:
        numpyro.handlers.trace(numpyro.handlers.seed(local_level, random.PRNGKey(0))).get_trace(
            TRAIN, TRAIN
        )
    with DiscreteTimeSimulator(n_simulations=1), builder:
        draws = forecast(random.PRNGKey(2), local_level, posterior, TRAIN, COVARIATES)
    expected_var = Q_TRUE**2 * jnp.arange(1, FUTURE + 1)[:, None] + R_TRUE**2
    assert draws.shape == (n, FUTURE, 1)
    assert jnp.allclose(draws.mean(axis=0), path[-1], atol=4 * jnp.sqrt(expected_var / n).max())
    assert jnp.allclose(draws.var(axis=0), expected_var, rtol=0.15)


def test_controls_are_aligned_with_their_time_steps() -> None:
    """``u_t = 100 t`` enters through ``D = 1``: any misalignment is a 100-unit error."""
    posterior = {"unused": jnp.zeros(400)}
    with DiscreteTimeSimulator(n_simulations=1), smoother:
        in_sample = predict_in_sample(
            random.PRNGKey(8), level_with_controls, posterior, CONTROLLED[:T_OBS]
        )
        ahead = forecast(
            random.PRNGKey(9), level_with_controls, posterior, CONTROLLED[:T_OBS, :1], CONTROLLED
        )
    assert jnp.abs(in_sample.mean(axis=0) - CONTROLLED[:T_OBS, :1]).max() < 1.0
    assert jnp.abs(ahead.mean(axis=0) - CONTROLLED[T_OBS:, :1]).max() < 3.0


def test_window_shapes_are_validated() -> None:
    """The time axes must match the horizon; a batch dim fails the jaxtyping rank check."""
    h = Horizon.from_data(COVARIATES, TRAIN)
    dynamics = local_level_dynamics(jnp.asarray(Q_TRUE), jnp.asarray(R_TRUE))
    with smoother, pytest.raises(ValueError, match="t_obs"):
        state_space(h, "f", TRAIN[:-1], dynamics)
    with smoother, pytest.raises(ValueError, match="duration"):
        state_space(h, "f", TRAIN, dynamics, controls=jnp.zeros((T_OBS, 2)))
    with pytest.raises(TypeError):  # jaxtyping.TypeCheckError: " time obs" is exactly 2-D
        state_space(h, "f", TRAIN[None], dynamics)


def test_backtest_scores_in_sample_under_one_handler_stack() -> None:
    """``backtest(eval_train=True)`` runs the in-sample and the forecast legs under one stack.

    The in-sample legs run under an idle Simulator, whose warning the block silences, and
    they score the smoothing predictive, which has seen the observations it scores: every
    window's in-sample CRPS is below its forecast CRPS.
    """

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

    with (
        warnings.catch_warnings(record=True) as caught,
        DiscreteTimeSimulator(n_simulations=1),
        smoother,
    ):
        warnings.simplefilter("always")
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
            num_samples=200,
            eval_train=True,
        )
    assert not [w for w in caught if "no predict_times" in str(w.message)]
    assert all(r.train_metrics["crps"] < r.metrics["crps"] for r in results)


def test_dynestyx_surface_canary() -> None:
    """The attribute and site-name surface the block depends on, pinned against drift."""
    _api_canary(
        "dynestyx",
        [
            "sample",
            "LatentStateResult",
            "DiscreteTimeSimulator",
            "handlers._dynestyx_stack_kind",
            "handlers._DynestyxStackKind",
        ],
    )
    from dynestyx import handlers

    kinds = {kind.name for kind in handlers._DynestyxStackKind}
    assert {"FILTER", "SMOOTHER", "LATENT_PATH_BUILDER"} <= kinds
    with smoother:
        active = [kind.name for kind in handlers._dynestyx_stack_kind()]
    assert active == ["SMOOTHER"]
    with DiscreteTimeSimulator(n_simulations=1), smoother:
        tr = numpyro.handlers.trace(
            numpyro.handlers.substitute(
                numpyro.handlers.seed(local_level, random.PRNGKey(0)), data=PARAMS
            )
        ).get_trace(COVARIATES, TRAIN)
    assert tr["f_smoothed_states_mean"]["value"].shape == (T_OBS, 1)
    assert tr["f_predicted_observations"]["value"].shape == (1, FUTURE + 1, 1)
    assert tr["f_predicted_states"]["value"].shape == (1, FUTURE + 1, 1)
    assert "f_marginal_loglik" in tr
    assert tr["forecast"]["value"].shape == (FUTURE, 1)


def test_result_returns_the_states_of_the_current_mode() -> None:
    """``x_in_sample`` is the draw behind ``"obs"``, ``x_future`` the rollout of ``"forecast"``.

    A model registers its latent level from the result; under a Smoother the level
    draws have the smoothed mean.
    """

    def with_level(covariates: Array, data: Array | None = None) -> None:
        h = Horizon.from_data(covariates, data)
        q = jnp.asarray(numpyro.sample("q", dist.HalfNormal(1.0)))
        r = jnp.asarray(numpyro.sample("r", dist.HalfNormal(1.0)))
        result = state_space(h, "f", covariates[..., : h.t_obs, :], local_level_dynamics(q, r))
        if result.x_in_sample is not None:
            numpyro.deterministic("level", result.x_in_sample)
        if result.x_future is not None:
            numpyro.deterministic("level_future", result.x_future)

    n = 2_000
    posterior = _repeat(PARAMS, n)
    with DiscreteTimeSimulator(n_simulations=1), smoother:
        sites = Predictive(with_level, posterior, return_sites=["obs", "level"])(
            random.PRNGKey(10), TRAIN
        )
        ahead = Predictive(with_level, posterior, return_sites=["forecast", "level_future"])(
            random.PRNGKey(11), COVARIATES, TRAIN
        )
        training = numpyro.handlers.trace(
            numpyro.handlers.substitute(
                numpyro.handlers.seed(local_level, random.PRNGKey(0)), data=PARAMS
            )
        ).get_trace(TRAIN, TRAIN)
    mean, var = _smoothed_moments()
    assert sites["level"].shape == sites["obs"].shape == (n, T_OBS, 1)
    assert jnp.allclose(sites["level"].mean(axis=0), mean, atol=4 * jnp.sqrt(var / n).max())
    assert jnp.allclose((sites["obs"] - sites["level"]).std(axis=0), R_TRUE, rtol=0.1)
    assert ahead["level_future"].shape == ahead["forecast"].shape == (n, FUTURE, 1)
    assert jnp.allclose((ahead["forecast"] - ahead["level_future"]).std(axis=0), R_TRUE, rtol=0.1)
    assert "level" not in training
    h = Horizon.from_data(TRAIN, TRAIN)
    with smoother:
        fitted = numpyro.handlers.seed(state_space, random.PRNGKey(0))(
            h, "f", TRAIN, local_level_dynamics(PARAMS["q"], PARAMS["r"])
        )
    assert fitted == StateSpaceResult(x_in_sample=None, x_future=None)


def test_latent_path_builder_reads_the_controls() -> None:
    """The explicit path also aligns ``u_t = 100 t`` (through ``D = 1``) with its time step."""
    n = 400
    path = SERIES[:T_OBS]  # the level path that generated the series, up to observation noise
    posterior = {"f_state_path_params": jnp.broadcast_to(path, (n, *path.shape))}
    builder = LatentPathBuilder()
    with builder:  # the eager run that a fit would make, so the builder sees concrete data
        numpyro.handlers.trace(
            numpyro.handlers.seed(level_with_controls, random.PRNGKey(0))
        ).get_trace(CONTROLLED[:T_OBS], CONTROLLED[:T_OBS, :1])
    with DiscreteTimeSimulator(n_simulations=1), builder:
        in_sample = predict_in_sample(
            random.PRNGKey(12), level_with_controls, posterior, CONTROLLED[:T_OBS]
        )
        ahead = forecast(
            random.PRNGKey(13), level_with_controls, posterior, CONTROLLED[:T_OBS, :1], CONTROLLED
        )
    assert jnp.abs(in_sample.mean(axis=0) - CONTROLLED[:T_OBS, :1]).max() < 1.0
    assert jnp.abs(ahead.mean(axis=0) - CONTROLLED[T_OBS:, :1]).max() < 3.0


def test_model_relations_trace_a_latent_path_with_controls() -> None:
    """``numpyro.render_model`` traces the model under ``jax.eval_shape``, where the builder
    samples the path from its prior and dynestyx indexes the control grid with traced
    indices; a NumPy control grid fails there with ``TracerArrayConversionError``.
    """
    builder = LatentPathBuilder()
    with builder:  # the eager run that a fit would make, so the builder sees concrete data
        numpyro.handlers.trace(
            numpyro.handlers.seed(level_with_controls, random.PRNGKey(0))
        ).get_trace(CONTROLLED[:T_OBS], CONTROLLED[:T_OBS, :1])
        relations = get_model_relations(
            level_with_controls, model_args=(CONTROLLED[:T_OBS], CONTROLLED[:T_OBS, :1])
        )
    assert relations["sample_sample"]["f_joint_log_prob_factor"] == ["f_state_path_params"]
    assert "f_joint_log_prob_factor" in relations["observed"]


@pytest.mark.parametrize("call", ["training", "forecast", "predict_in_sample"])
def test_the_block_refuses_to_run_without_a_conditioning_handler(call: str) -> None:
    """Without a Filter, Smoother or LatentPathBuilder the block raises, in every mode.

    dynestyx 0.8 would pick a default handler itself and only log the choice, so a
    forgotten handler would fit or forecast under an inference method nobody chose. A
    Simulator alone is not a conditioning handler.
    """
    with pytest.raises(ValueError, match="needs a dynestyx Filter, Smoother or"):
        if call == "training":
            log_density(local_level, (TRAIN, TRAIN), {}, PARAMS)
        elif call == "forecast":
            with DiscreteTimeSimulator(n_simulations=1):
                forecast(random.PRNGKey(14), local_level, _repeat(PARAMS, 2), TRAIN, COVARIATES)
        else:
            predict_in_sample(random.PRNGKey(15), local_level, _repeat(PARAMS, 2), TRAIN)


def test_forecast_chunks_keep_the_law() -> None:
    """``batch_size`` chunking of the driver gives the same forecast law as one pass."""
    n = 2_000
    with DiscreteTimeSimulator(n_simulations=1), smoother:
        whole = forecast(random.PRNGKey(16), local_level, _repeat(PARAMS, n), TRAIN, COVARIATES)
        chunked = forecast(
            random.PRNGKey(16), local_level, _repeat(PARAMS, n), TRAIN, COVARIATES, batch_size=300
        )
    sd = whole.std(axis=0)
    assert chunked.shape == whole.shape == (n, FUTURE, 1)
    assert jnp.allclose(chunked.mean(axis=0), whole.mean(axis=0), atol=6 * sd.max() / jnp.sqrt(n))
    assert jnp.allclose(chunked.std(axis=0), sd, rtol=0.15)


@pytest.mark.parametrize("handler_name", ["smoother", "filter"])
def test_state_controls_drive_the_first_forecast_step(handler_name: str) -> None:
    """``u_{t_obs - 1}`` enters the first transition: the anchor row gives one transition per step.

    With ``x_{t+1} = x_t + B u_t`` and a single control of ``100`` at the last observed step,
    every forecast step sits ``100`` above the last smoothed state; without the anchor row the
    rollout would start at ``t_obs`` and skip that control.
    """
    u = jnp.zeros((T_OBS + FUTURE, 1)).at[T_OBS - 1].set(100.0)
    covariates = jnp.concatenate([SERIES, u], axis=-1)

    def with_state_controls(covariates: Array, data: Array | None = None) -> None:
        h = Horizon.from_data(covariates, data)
        dynamics = DynamicalModel(
            initial_condition=dist.MultivariateNormal(jnp.zeros(1), P0 * jnp.eye(1)),
            state_evolution=LinearGaussianStateEvolution(
                A=jnp.eye(1), B=jnp.eye(1), cov=Q_TRUE**2 * jnp.eye(1)
            ),
            observation_model=LinearGaussianObservation(H=jnp.eye(1), R=R_TRUE**2 * jnp.eye(1)),
            control_dim=1,
        )
        state_space(h, "f", covariates[..., : h.t_obs, :1], dynamics, controls=covariates[..., 1:])

    handler = (
        smoother
        if handler_name == "smoother"
        else Filter(filter_config=KFConfig(filter_source="cuthbert"))
    )
    posterior = {"unused": jnp.zeros(2_000)}
    with DiscreteTimeSimulator(n_simulations=1), handler:
        ahead = forecast(
            random.PRNGKey(17), with_state_controls, posterior, covariates[:T_OBS, :1], covariates
        )
    mean, _ = _smoothed_moments()
    assert jnp.allclose(ahead.mean(axis=0), mean[-1] + 100.0, atol=0.5)


def test_scalar_event_observation_model_keeps_the_obs_axis() -> None:
    """An observation model with a scalar event gets its ``obs`` axis back: ``(t_obs, 1)``."""

    def scalar_obs(covariates: Array, data: Array | None = None) -> None:
        h = Horizon.from_data(covariates, data)
        dynamics = DynamicalModel(
            initial_condition=dist.MultivariateNormal(jnp.zeros(1), P0 * jnp.eye(1)),
            state_evolution=LinearGaussianStateEvolution(A=jnp.eye(1), cov=Q_TRUE**2 * jnp.eye(1)),
            observation_model=lambda x, u, t: dist.Normal(x[0], R_TRUE),
        )
        state_space(h, "f", covariates[..., : h.t_obs, :], dynamics)

    path = TRAIN
    posterior = {"f_state_path_params": jnp.broadcast_to(path, (50, *path.shape))}
    builder = LatentPathBuilder()
    with builder:
        numpyro.handlers.trace(numpyro.handlers.seed(scalar_obs, random.PRNGKey(0))).get_trace(
            TRAIN, TRAIN
        )
        draws = predict_in_sample(random.PRNGKey(18), scalar_obs, posterior, TRAIN)
    assert draws.shape == (50, T_OBS, 1)
    assert jnp.isfinite(draws).all()


def test_missing_observations_under_the_latent_path_builder() -> None:
    """The builder skips NaN rows of ``y`` and draws the in-sample predictive through the gap."""
    gap = slice(8, 14)
    train = TRAIN.at[gap].set(jnp.nan)
    path = jnp.linspace(-1.0, 1.0, T_OBS)[:, None]
    posterior = _repeat({**PARAMS, "f_state_path_params": path}, 1_000)
    builder = LatentPathBuilder()
    with builder:
        numpyro.handlers.trace(numpyro.handlers.seed(local_level, random.PRNGKey(0))).get_trace(
            train, train
        )
        draws = predict_in_sample(random.PRNGKey(19), local_level, posterior, train)
    assert jnp.isfinite(draws).all()
    assert jnp.allclose(draws.mean(axis=0), path, atol=4 * R_TRUE / jnp.sqrt(1_000))


def test_in_sample_steps_draw_with_independent_keys() -> None:
    """Under a Smoother the per-step state draws are independent: adjacent steps do not correlate.

    One key reused for every step would give every step the same standard normal draw, so
    the standardized draws of neighboring steps would be perfectly correlated.
    """

    def with_level(covariates: Array, data: Array | None = None) -> None:
        h = Horizon.from_data(covariates, data)
        dynamics = local_level_dynamics(jnp.asarray(Q_TRUE), jnp.asarray(R_TRUE))
        result = state_space(h, "f", covariates[..., : h.t_obs, :], dynamics)
        if result.x_in_sample is not None:
            numpyro.deterministic("level", result.x_in_sample)

    with smoother:
        level = Predictive(with_level, {"unused": jnp.zeros(4_000)}, return_sites=["level"])(
            random.PRNGKey(20), TRAIN
        )["level"][..., 0]
    z = (level - level.mean(axis=0)) / level.std(axis=0)
    adjacent = (z[:, 1:] * z[:, :-1]).mean(axis=0)
    assert jnp.abs(adjacent).max() < 0.1


def test_in_sample_predictive_needs_a_seed_handler() -> None:
    """The in-sample draws consume the seed handler's key; without one the block says so."""
    h = Horizon.from_data(TRAIN, None)
    dynamics = local_level_dynamics(jnp.asarray(Q_TRUE), jnp.asarray(R_TRUE))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # numpyro's "Cannot generate JAX PRNG key" stays silent
        with smoother, pytest.raises(RuntimeError, match="seed handler"):
            state_space(h, "f", TRAIN, dynamics)


def test_an_untested_dynestyx_version_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """The extra is capped at the tested minor version; another installed version warns."""
    import dynestyx

    monkeypatch.setattr(dynestyx, "__version__", "0.9.0")
    with smoother, pytest.warns(UserWarning, match="tested with dynestyx 0.8"):
        log_density(local_level, (TRAIN, TRAIN), {}, PARAMS)


@pytest.mark.parametrize("call", ["training", "forecast", "predict_in_sample"])
@pytest.mark.parametrize(("prefix", "divider"), [("a", "/"), ("f", "_")])
def test_scope_gives_a_clear_error(call: str, prefix: str, divider: str) -> None:
    """Under ``handlers.scope`` the dynestyx sites are renamed, which the block reports in
    every mode (training would otherwise pass and the drivers fail with a bare ``KeyError``),
    also when the prefix is the block's name and the divider is ``_``.
    """
    scoped = numpyro.handlers.scope(local_level, prefix=prefix, divider=divider)
    with (
        DiscreteTimeSimulator(n_simulations=1),
        smoother,
        pytest.raises(ValueError, match=r"handlers\.scope"),
    ):
        if call == "training":
            params = {f"{prefix}{divider}{k}": v for k, v in PARAMS.items()}
            log_density(scoped, (TRAIN, TRAIN), {}, params)
        elif call == "forecast":
            forecast(random.PRNGKey(21), scoped, _repeat(PARAMS, 2), TRAIN, COVARIATES)
        else:
            predict_in_sample(random.PRNGKey(21), scoped, _repeat(PARAMS, 2), TRAIN)


def test_numpy_controls_are_rejected_by_the_annotation() -> None:
    """``controls`` is annotated as a jax array, so NumPy controls fail at the call, clearly.

    Without the check they would reach dynestyx's eager simulator scan and fail there with
    a ``TracerArrayConversionError``.
    """
    h = Horizon.from_data(COVARIATES, TRAIN)
    dynamics = local_level_dynamics(jnp.asarray(Q_TRUE), jnp.asarray(R_TRUE))
    with smoother, pytest.raises(TypeError):  # jaxtyping.TypeCheckError
        state_space(
            h,
            "f",
            TRAIN,
            dynamics,
            controls=np.zeros((T_OBS + FUTURE, 1)),  # ty: ignore[invalid-argument-type]
        )


def test_a_scalar_state_path_is_rejected_with_a_clear_message() -> None:
    """A scalar state gives a ``(time,)`` path, which breaks the ``(t_obs, state)`` contract.

    dynestyx deprecates scalar initial conditions, so this path may close upstream; the
    block's check also covers other state paths that are not ``(t_obs, state)``.
    """
    h = Horizon.from_data(TRAIN, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # dynestyx warns about the scalar initial condition
        dynamics = DynamicalModel(
            initial_condition=dist.Normal(0.0, jnp.sqrt(P0)),
            state_evolution=lambda x, u, t_now, t_next: dist.Normal(x, Q_TRUE),
            observation_model=lambda x, u, t: dist.Normal(x, R_TRUE),
        )
        with (
            LatentPathBuilder(),
            pytest.raises(ValueError, match="one state vector per observed step"),
        ):
            numpyro.handlers.seed(state_space, random.PRNGKey(23))(h, "f", TRAIN, dynamics)


@pytest.mark.parametrize("filter_source", ["cuthbert", "cd_dynamax"])
def test_filter_and_smoother_add_the_same_log_density(
    filter_source: Literal["cuthbert", "cd_dynamax"],
) -> None:
    """The Kalman filter and smoother of one backend add the same marginal likelihood: a
    Filter fit's draws can be exported under the Smoother without a refit.
    """
    with Filter(filter_config=KFConfig(filter_source=filter_source)):
        filtered, _ = log_density(local_level, (TRAIN, TRAIN), {}, PARAMS)
    with Smoother(smoother_config=KFSmootherConfig(filter_source=filter_source)):
        smoothed, _ = log_density(local_level, (TRAIN, TRAIN), {}, PARAMS)
    assert jnp.allclose(filtered, smoothed, rtol=1e-5)


def test_a_filter_posterior_exports_under_the_smoother() -> None:
    """The ``f_filtered_states_*`` keys of a Filter fit name no site of the model under the
    Smoother, so ``to_datatree`` keeps them in the ``posterior`` group (issue #277).
    """
    with Filter(filter_config=KFConfig(filter_source="cuthbert")):
        fitted = Predictive(local_level, posterior_samples=_repeat(PARAMS, 4))(
            random.PRNGKey(4), TRAIN, TRAIN
        )
    posterior = {**_repeat(PARAMS, 4), **fitted}
    filtered = {name for name in posterior if name.startswith("f_filtered_states")}
    assert filtered
    with smoother:
        tree = to_datatree(random.PRNGKey(3), local_level, posterior, TRAIN, TRAIN)
    assert filtered <= set(tree["posterior"].data_vars)
    assert tree["posterior_predictive"]["obs"].shape == (1, 4, T_OBS, 1)


def test_svi_fit_and_draw_posterior_run_through_the_block() -> None:
    """SVI with an AutoGuide fits through the block, and ``draw_posterior`` needs the handler.

    ``AutoNormal`` re-runs the model in its postprocess function to compute deterministic
    sites, so ``draw_posterior`` without the handler hits the block's guard.
    """
    guide = AutoNormal(local_level)
    svi = SVI(local_level, guide, Adam(0.05), Trace_ELBO())
    with smoother:
        result = svi.run(random.PRNGKey(24), 300, TRAIN, TRAIN, progress_bar=False)
    with pytest.raises(ValueError, match="needs a dynestyx Filter, Smoother or"):
        draw_posterior(random.PRNGKey(25), guide, result.params, 200)
    with DiscreteTimeSimulator(n_simulations=1), smoother:
        posterior = draw_posterior(random.PRNGKey(25), guide, result.params, 200)
        ahead = forecast(random.PRNGKey(26), local_level, posterior, TRAIN, COVARIATES)
    assert jnp.isfinite(result.losses[-1])
    assert result.losses[-1] < result.losses[0]
    assert ahead.shape == (200, FUTURE, 1)
    assert jnp.isfinite(ahead).all()


def test_to_datatree_under_the_latent_path_builder() -> None:
    """The builder that fitted the model serves both predictive groups of ``to_datatree``.

    The in-sample draws scatter around the posterior path, and the forecast starts from its
    final state, so neither group comes from a prior path.
    """
    n = 400
    path = jnp.linspace(-1.0, 1.0, T_OBS)[:, None]
    posterior = _repeat({**PARAMS, "f_state_path_params": path}, n)
    builder = LatentPathBuilder()
    with builder:
        numpyro.handlers.trace(numpyro.handlers.seed(local_level, random.PRNGKey(0))).get_trace(
            TRAIN, TRAIN
        )
    with DiscreteTimeSimulator(n_simulations=1), builder:
        tree = to_datatree(
            random.PRNGKey(27), local_level, posterior, TRAIN, COVARIATES, num_chains=2
        )
    in_sample = tree["posterior_predictive"]["obs"]
    ahead = tree["predictions"]["obs"]
    assert in_sample.shape == (2, n // 2, T_OBS, 1)
    assert ahead.shape == (2, n // 2, FUTURE, 1)
    expected_var = Q_TRUE**2 * np.arange(1, FUTURE + 1)[:, None] + R_TRUE**2
    assert np.allclose(
        in_sample.mean(("chain", "draw")).to_numpy(), path, atol=4 * R_TRUE / np.sqrt(n)
    )
    assert np.allclose(
        ahead.mean(("chain", "draw")).to_numpy(),
        path[-1],
        atol=4 * np.sqrt(expected_var / n).max(),
    )


@pytest.mark.parametrize("handler_name", ["smoother", "builder"])
def test_two_observed_series_keep_their_columns(handler_name: str) -> None:
    """Two series observed through ``H = I`` keep their ``obs`` column in both sites.

    The second series sits ``50`` above the first, so a transposed or broadcast axis shows
    as a ``50``-unit error in the in-sample and in the forecast draws.
    """
    offset = jnp.array([0.0, 50.0])
    series = offset + jnp.concatenate(
        [SERIES, simulate(random.PRNGKey(29), T_OBS + FUTURE)], axis=-1
    )
    train = series[:T_OBS]

    def two_series(covariates: Array, data: Array | None = None) -> None:
        h = Horizon.from_data(covariates, data)
        dynamics = DynamicalModel(
            initial_condition=dist.MultivariateNormal(offset, P0 * jnp.eye(2)),
            state_evolution=LinearGaussianStateEvolution(A=jnp.eye(2), cov=Q_TRUE**2 * jnp.eye(2)),
            observation_model=LinearGaussianObservation(H=jnp.eye(2), R=R_TRUE**2 * jnp.eye(2)),
        )
        state_space(h, "f", covariates[..., : h.t_obs, :], dynamics)

    n = 400
    if handler_name == "smoother":
        handler, posterior = smoother, {"unused": jnp.zeros(n)}
    else:
        handler = LatentPathBuilder()
        with handler:  # the eager run that a fit would make, so the builder sees concrete data
            numpyro.handlers.trace(numpyro.handlers.seed(two_series, random.PRNGKey(0))).get_trace(
                train, train
            )
        posterior = {"f_state_path_params": jnp.broadcast_to(train, (n, *train.shape))}
    with DiscreteTimeSimulator(n_simulations=1), handler:
        in_sample = predict_in_sample(random.PRNGKey(30), two_series, posterior, train)
        ahead = forecast(random.PRNGKey(31), two_series, posterior, train, series)
    assert in_sample.shape == (n, T_OBS, 2)
    assert ahead.shape == (n, FUTURE, 2)
    assert jnp.abs(in_sample.mean(axis=0) - train).max() < 1.0
    assert jnp.abs(ahead.mean(axis=0) - train[-1]).max() < 1.0
