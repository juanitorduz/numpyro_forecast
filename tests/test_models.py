"""Tests for the model building blocks (``numpyro_forecast.models``)."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from conftest import as_model, empty_covariates, rw_body, rw_model
from jax import random
from numpyro.handlers import seed, substitute, trace
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast.models import Horizon, innovations, predict
from numpyro_forecast.predictive import forecast
from numpyro_forecast.surgery import shift_loc
from numpyro_forecast.typing import Array, ForecastModel


def test_horizon_training() -> None:
    data = jnp.zeros((20, 1))
    covariates = jnp.zeros((20, 0))
    h = Horizon.from_data(covariates, data)
    assert h.duration == 20
    assert h.t_obs == 20
    assert h.future == 0
    assert h.data is data


def test_horizon_forecast() -> None:
    data = jnp.zeros((20, 1))
    covariates = jnp.zeros((25, 0))
    h = Horizon.from_data(covariates, data)
    assert h.duration == 25
    assert h.t_obs == 20
    assert h.future == 5


def test_horizon_prior() -> None:
    covariates = jnp.zeros((20, 0))
    h = Horizon.from_data(covariates, None)
    assert h.duration == 20
    assert h.t_obs == 20
    assert h.future == 0
    assert h.data is None


def test_horizon_rejects_data_longer_than_covariates() -> None:
    data = jnp.zeros((20, 1))
    covariates = jnp.zeros((15, 0))
    with pytest.raises(ValueError, match="data must not be longer than covariates"):
        Horizon.from_data(covariates, data)


@pytest.mark.parametrize(
    ("covariates", "hint"),
    [(jnp.arange(8.0), r"x\[\.\.\., None\]"), (jnp.array(0.0), "A 0-d value has no time axis")],
)
def test_horizon_rejects_covariates_without_time_axis(covariates: Array, hint: str) -> None:
    """1-D (or 0-d) covariates name the ``(time, k)`` layout instead of an ``IndexError``."""
    with pytest.raises(
        ValueError, match=rf"covariates must have time at axis -2 and the covariate dim.*{hint}"
    ):
        Horizon.from_data(covariates, None)


@pytest.mark.parametrize(
    ("data", "hint"),
    [(jnp.zeros(6), r"got shape \(6,\)\. Add the last axis"), (jnp.array(0.0), "A 0-d value")],
)
def test_horizon_rejects_data_without_time_axis(data: Array, hint: str) -> None:
    with pytest.raises(
        ValueError, match=rf"data must have time at axis -2 and the observation dim.*{hint}"
    ):
        Horizon.from_data(jnp.zeros((8, 0)), data)


@pytest.mark.parametrize("duration", [8, 0])
def test_horizon_rejects_data_without_observations(duration: int) -> None:
    """A cold-start ``data`` of shape ``(0, obs)`` fails before ``plate("time", 0)``."""
    with pytest.raises(ValueError, match=r"data has no observed time steps \(shape \(0, 1\)\)"):
        Horizon.from_data(jnp.zeros((duration, 0)), jnp.zeros((0, 1)))


def test_horizon_rejects_covariates_without_time_steps() -> None:
    with pytest.raises(ValueError, match=r"covariates has no time steps \(shape \(0, 0\)\)"):
        Horizon.from_data(jnp.zeros((0, 0)), None)


def test_horizon_zero_data_shape() -> None:
    data = jnp.zeros((20, 1))
    covariates = jnp.zeros((25, 0))
    h = Horizon.from_data(covariates, data)
    assert h.zero_data is not None
    assert h.zero_data.shape == (25, 1)


def test_horizon_zero_data_none_for_prior() -> None:
    covariates = jnp.zeros((20, 0))
    h = Horizon.from_data(covariates, None)
    assert h.zero_data is None


def test_horizon_rejects_inconsistent_duration() -> None:
    with pytest.raises(ValueError, match="duration must equal t_obs \\+ future"):
        Horizon(data=None, t_obs=5, future=10, duration=20)


def test_horizon_rejects_negative_future() -> None:
    with pytest.raises(ValueError, match="t_obs and future must be non-negative"):
        Horizon(data=None, t_obs=5, future=-1, duration=4)


def test_innovations_predict_training_sites() -> None:
    data = jnp.zeros((20, 1))
    covariates = jnp.zeros((20, 0))
    h = Horizon.from_data(covariates, data)
    tr = trace(seed(lambda: rw_body(h, covariates), random.PRNGKey(0))).get_trace()
    assert tr["drift"]["value"].shape == (20, 1)
    assert "obs" in tr
    assert "drift_future" not in tr
    assert "obs_future" not in tr
    assert "forecast" not in tr


def test_innovations_predict_forecast_sites() -> None:
    data = jnp.zeros((20, 1))
    covariates = jnp.zeros((25, 0))
    h = Horizon.from_data(covariates, data)
    tr = trace(seed(lambda: rw_body(h, covariates), random.PRNGKey(0))).get_trace()
    # In-sample site keeps its training shape; the horizon uses a separate site.
    assert tr["drift"]["value"].shape == (20, 1)
    assert tr["drift_future"]["value"].shape == (5, 1)
    assert tr["forecast"]["value"].shape == (5, 1)


def test_innovations_reparam_applies() -> None:
    data = jnp.zeros((10, 1))
    covariates = jnp.zeros((10, 0))
    h = Horizon.from_data(covariates, data)

    def body() -> None:
        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
        drift = innovations(h, "drift", dist.Normal(0.0, drift_scale), reparam=LocScaleReparam(0))
        predict(h, dist.Normal(0.0, 1.0), jnp.cumsum(drift, axis=-2))

    tr = trace(seed(body, random.PRNGKey(0))).get_trace()
    assert tr["drift"]["value"].shape == (10, 1)
    # LocScaleReparam introduces a decentered companion site.
    assert any("decentered" in name for name in tr)


def test_innovations_shares_one_distribution_across_both_time_plates() -> None:
    covariates = jnp.zeros((25, 0))
    h = Horizon.from_data(covariates, jnp.zeros((20, 1)))
    prior = dist.Normal(0.0, 0.3)

    def body() -> None:
        innovations(h, "drift", prior)

    tr = trace(seed(body, random.PRNGKey(0))).get_trace()
    assert tr["drift"]["value"].shape == (20, 1)
    assert tr["drift_future"]["value"].shape == (5, 1)
    assert prior.batch_shape == ()  # the plates expand a copy, never the instance


def test_predict_forecast_requires_data() -> None:
    # future > 0 but data is None: forecasting needs observed data to condition on.
    h = Horizon(data=None, t_obs=10, future=5, duration=15)
    with pytest.raises(RuntimeError, match="forecasting requires observed data"):
        trace(
            seed(lambda: predict(h, dist.Normal(0.0, 1.0), jnp.zeros((15, 1))), random.PRNGKey(0))
        ).get_trace()


@pytest.mark.parametrize(
    ("body", "data_shape", "duration"),
    [
        pytest.param(
            lambda h: predict(h, dist.Normal(0.0, 1.0), jnp.zeros(h.duration)),
            (6, 1),
            6,
            id="1d-predictor",
        ),
        pytest.param(
            lambda h: predict(h, dist.Normal(0.0, 1.0), jnp.zeros(h.duration)),
            (6, 1),
            8,
            id="1d-predictor-forecast",
        ),
        pytest.param(
            lambda h: predict(h, lambda mu: dist.Normal(mu, 1.0), jnp.zeros(h.duration)),
            (6, 1),
            8,
            id="1d-predictor-link",
        ),
        pytest.param(
            lambda h: predict(h, dist.Normal(0.0, 1.0), jnp.zeros(h.duration)),
            (4, 4),
            4,
            id="1d-predictor-square-panel",
        ),
        pytest.param(
            lambda h: predict(h, dist.Normal(0.0, 1.0), jnp.zeros(h.duration)),
            (4, 3),
            4,
            id="1d-predictor-panel",
        ),
        pytest.param(
            lambda h: predict(
                h, dist.Normal(0.0, jnp.ones((h.duration, 1))), jnp.zeros(h.duration)
            ),
            None,
            6,
            id="1d-predictor-time-varying-scale-prior",
        ),
        pytest.param(
            lambda h: predict(
                h, dist.Normal(0.0, 1.0), jnp.zeros((h.duration, 1)) + jnp.arange(h.duration)
            ),
            (6, 1),
            6,
            id="mixed-term",
        ),
        pytest.param(
            lambda h: predict(
                h,
                lambda mu: dist.Normal(mu, 1.0),
                jnp.zeros((h.duration, 1)) + jnp.arange(h.duration),
            ),
            (6, 1),
            8,
            id="mixed-term-link",
        ),
        pytest.param(
            lambda h: predict(
                h, dist.Normal(0.0, jnp.linspace(1.0, 2.0, h.duration)), jnp.zeros((h.duration, 1))
            ),
            (6, 1),
            6,
            id="1d-time-varying-scale",
        ),
        pytest.param(
            lambda h: predict(h, dist.Normal(0.0, 1.0), jnp.zeros((h.duration, 1, 1))),
            (6, 1),
            6,
            id="second-obs-axis",
        ),
        pytest.param(
            lambda h: predict(
                h, lambda mu: dist.Normal(mu[..., 0], 1.0), jnp.zeros((h.duration, 1))
            ),
            (6, 1),
            8,
            id="link-drops-obs-axis",
        ),
        pytest.param(
            lambda h: predict(
                h, lambda mu: dist.Normal(mu[..., 0], 1.0), jnp.zeros((h.duration, 1))
            ),
            None,
            6,
            id="link-drops-obs-axis-prior",
        ),
        pytest.param(
            lambda h: predict(h, dist.Normal(0.0, 1.0), jnp.zeros((3, h.duration))),
            (6, 3),
            6,
            id="series-first",
        ),
        pytest.param(
            lambda h: predict(h, dist.Normal(0.0, 1.0), jnp.zeros((3, h.duration))),
            (3, 6, 1),
            8,
            id="series-first-batched-data",
        ),
        pytest.param(
            lambda h: predict(h, dist.Normal(0.0, 1.0), jnp.zeros((h.duration, 2))),
            (6, 3),
            6,
            id="wrong-width",
        ),
        pytest.param(
            lambda h: predict(h, dist.Normal(0.0, 1.0), jnp.zeros(())),
            (6, 1),
            6,
            id="constant-without-time-axis",
        ),
        pytest.param(
            lambda h: predict(h, dist.Normal(0.0, 1.0), jnp.zeros(3)),
            (6, 3),
            8,
            id="per-series-without-time-axis",
        ),
        pytest.param(
            lambda h: predict(h, dist.Normal(0.0, 1.0), jnp.zeros((5, 1))),
            (1, 1),
            1,
            id="repeats-single-observed-step",
        ),
        pytest.param(
            lambda h: predict(
                h,
                dist.MultivariateNormal(jnp.zeros(h.duration), jnp.eye(h.duration)),
                jnp.zeros(()),
            ),
            (6, 1),
            8,
            id="mvn-noise-without-obs-axis",
        ),
        pytest.param(
            lambda h: predict(
                h,
                dist.MultivariateNormal(jnp.zeros(h.duration), jnp.eye(h.duration)),
                jnp.zeros(()),
            ),
            None,
            6,
            id="mvn-noise-without-obs-axis-prior",
        ),
        pytest.param(
            lambda h: predict(
                h,
                lambda mu: dist.MultivariateNormal(mu[..., 0], jnp.eye(h.duration)),
                jnp.zeros((h.duration, 1)),
            ),
            (6, 1),
            6,
            id="link-returns-bare-mvn",
        ),
        pytest.param(
            lambda h: predict(
                h,
                dist.MultivariateNormal(jnp.zeros((1, h.duration)), jnp.eye(h.duration)).to_event(
                    1
                ),
                jnp.zeros(()),
            ),
            (6, 1),
            8,
            id="mvn-to-event-without-obs-axis",
        ),
    ],
)
def test_predict_rejects_an_observation_distribution_that_does_not_fit_the_data(
    body: Callable[[Horizon], None], data_shape: tuple[int, ...] | None, duration: int
) -> None:
    """Issue #166: the obs distribution needs ``(time, obs)`` axes and must not grow the data.

    A ``(duration,)`` predictor (``covariates @ beta``) or a ``(duration,)``
    term inside the distribution broadcasts its time steps onto the
    observation axis, so ``(time, 1)`` data was scored on a ``(time, time)``
    grid without an error; a second ``[..., None]`` or a predictor longer than
    a single observed step counted each observation more than once; a bare
    ``MultivariateNormal`` over time (no observation axis) scored ``(time, 1)``
    data as ``time`` broadcast values; and the other rows failed later without
    naming the cause. ``data=None`` rows cover prior sampling, where there is
    no data to compare against.
    """
    data = None if data_shape is None else jnp.zeros(data_shape)
    model = as_model(lambda h, covariates: body(h))
    with pytest.raises(ValueError, match=r"^predict expects"):
        trace(seed(model, random.PRNGKey(0))).get_trace(empty_covariates(duration), data)


@pytest.mark.parametrize(
    ("prediction_shape", "data_shape"),
    [pytest.param((), (6, 1), id="constant-mean"), pytest.param((3,), (6, 3), id="per-series")],
)
def test_predict_accepts_a_noise_scale_that_carries_the_time_axis(
    prediction_shape: tuple[int, ...], data_shape: tuple[int, ...]
) -> None:
    """A 0-d or per-series predictor scores and forecasts with a time-varying scale."""

    def body(h: Horizon, covariates: Array) -> None:
        scale = jnp.linspace(1.0, 2.0, h.duration)[:, None] * jnp.ones(data_shape[-1])
        predict(h, dist.Normal(0.0, scale), jnp.zeros(prediction_shape))

    tr = trace(seed(as_model(body), random.PRNGKey(0))).get_trace(
        empty_covariates(8), jnp.zeros(data_shape)
    )
    assert tr["obs"]["fn"].batch_shape == data_shape
    assert tr["forecast"]["value"].shape == (2, data_shape[-1])


def _one_region_body(h: Horizon) -> None:
    """A level sampled in a size-1 plate at ``dim=-3``: a ``(1, duration, 1)`` predictor."""
    with numpyro.plate("region", 1, dim=-3):
        level = jnp.asarray(numpyro.sample("level", dist.Normal(0.0, 1.0)))
    predict(h, dist.Normal(0.0, 1.0), level + jnp.zeros((h.duration, 1)))


@pytest.mark.parametrize("future", [0, 2])
@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            lambda h: predict(h, dist.Normal(0.0, 1.0), jnp.zeros((1, h.duration, 1))),
            id="predictor",
        ),
        pytest.param(_one_region_body, id="size-one-plate"),
        pytest.param(
            lambda h: predict(
                h,
                dist.MultivariateNormal(jnp.zeros((1, h.duration)), jnp.eye(h.duration)).to_event(
                    1
                ),
                jnp.zeros((h.duration, 1)),
            ),
            id="mvn-size-one-batch",
        ),
    ],
)
def test_predict_accepts_a_size_one_axis_that_the_data_does_not_have(
    body: Callable[[Horizon], None], future: int
) -> None:
    """A size-1 leading axis repeats no observation, so ``obs`` scores the data once.

    Every row has mean 0 and unit noise (an identity covariance for the MVN
    row), so the in-sample density is the standard normal density of the data.
    """
    data = jnp.linspace(-1.0, 1.0, 6)[:, None]
    model = substitute(
        as_model(lambda h, covariates: body(h)), data={"level": jnp.zeros((1, 1, 1))}
    )
    tr = trace(seed(model, random.PRNGKey(0))).get_trace(empty_covariates(6 + future), data)
    obs_log_density = tr["obs"]["fn"].log_prob(tr["obs"]["value"]).sum()
    assert jnp.allclose(obs_log_density, dist.Normal(0.0, 1.0).log_prob(data).sum())


@pytest.mark.parametrize("future", [0, 2])
@pytest.mark.parametrize(
    "obs_dist",
    [
        pytest.param(dist.Normal(0.0, 1.0), id="noise"),
        pytest.param(lambda mu: dist.Normal(mu, 1.0), id="link"),
    ],
)
def test_predict_scores_a_shared_predictor_against_every_panel_series(
    obs_dist: dist.Distribution | Callable[[Array], dist.Distribution], future: int
) -> None:
    """A ``(duration, 1)`` predictor in a series plate scores and forecasts every series.

    The checks see the distribution before the plate expands it, so its size-1
    observation axis must pass them.
    """
    t_obs = 4
    offsets = jnp.array([-1.0, 0.0, 2.0])

    def body(h: Horizon, covariates: Array) -> None:
        with numpyro.plate("series", 3, dim=-1):
            predict(h, obs_dist, 10.0 * jnp.arange(h.duration, dtype=jnp.float32)[:, None])

    panel = 10.0 * jnp.arange(t_obs, dtype=jnp.float32)[:, None] + offsets
    tr = trace(seed(as_model(body), random.PRNGKey(0))).get_trace(
        empty_covariates(t_obs + future), panel
    )
    obs_log_prob = tr["obs"]["fn"].log_prob(tr["obs"]["value"])
    assert obs_log_prob.shape == (t_obs, 3)
    assert jnp.allclose(obs_log_prob.sum(), t_obs * dist.Normal(0.0, 1.0).log_prob(offsets).sum())
    if future:
        assert tr["forecast"]["value"].shape == (future, 3)


def test_rw_model_training_trace() -> None:
    """The plain-function ``rw_model`` produces the expected training-time trace."""
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(1), (20, 1)), axis=-2)
    tr = trace(seed(rw_model, random.PRNGKey(0))).get_trace(empty_covariates(20), data)
    assert tr["drift"]["value"].shape == (20, 1)
    assert "obs" in tr
    assert "drift_future" not in tr
    assert "forecast" not in tr


def test_rw_model_forecast_trace() -> None:
    """The plain-function ``rw_model`` produces the expected forecast-horizon trace."""
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(1), (20, 1)), axis=-2)
    tr = trace(seed(rw_model, random.PRNGKey(0))).get_trace(empty_covariates(25), data)
    # In-sample site keeps its training shape; the horizon uses a separate site.
    assert tr["drift"]["value"].shape == (20, 1)
    assert tr["drift_future"]["value"].shape == (5, 1)
    assert tr["forecast"]["value"].shape == (5, 1)


def test_rw_model_prior_sampling() -> None:
    # data=None: pure prior sampling. The whole horizon is in-sample, so "obs" is
    # sampled (not observed) and there are no forecast-horizon sites.
    tr = trace(seed(rw_model, random.PRNGKey(0))).get_trace(empty_covariates(15))
    assert tr["drift"]["value"].shape == (15, 1)
    assert tr["obs"]["is_observed"] is False
    assert "drift_future" not in tr
    assert "forecast" not in tr


# --- predict: distribution form vs link form ----------------------------------


def _link_form_body(h: Horizon, covariates: Array) -> None:
    """Random-walk body written with the link form of predict and a shift_loc link."""
    drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 1.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-1.0, 1.0))
    drift = innovations(h, "drift", dist.Normal(0.0, drift_scale))
    predict(h, lambda mu: shift_loc(dist.Normal(0.0, sigma), mu), jnp.cumsum(drift, axis=-2))


def _traces_equal(model_a: ForecastModel, model_b: ForecastModel, *args: Array) -> None:
    key = random.PRNGKey(0)
    trace_a = trace(seed(model_a, key)).get_trace(*args)
    trace_b = trace(seed(model_b, key)).get_trace(*args)
    assert set(trace_a) == set(trace_b)
    for name, site_a in trace_a.items():
        site_b = trace_b[name]
        assert site_a["type"] == site_b["type"]
        if "value" in site_a and site_a["value"] is not None:
            assert jnp.allclose(site_a["value"], site_b["value"])
        if site_a["type"] == "sample":
            la = site_a["fn"].log_prob(site_a["value"])
            lb = site_b["fn"].log_prob(site_b["value"])
            assert jnp.allclose(la, lb)


@pytest.mark.parametrize("future", [0, 6])
def test_predict_distribution_and_link_forms_are_equivalent(future: int) -> None:
    """The distribution form of predict equals the link form through shift_loc (same traces)."""
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(1), (24, 1)), axis=-2)
    covariates = empty_covariates(24 + future)
    model_glm = as_model(_link_form_body)
    _traces_equal(rw_model, model_glm, covariates, data)


def test_predict_rejects_link_that_returns_no_distribution() -> None:
    """A link returning the predictor itself fails with a message naming both forms."""

    def bad_body(h: Horizon, covariates: Array) -> None:
        level = innovations(h, "level", dist.Normal(0.0, 1.0))
        # ty rejects this link statically; the runtime guard covers untyped callers.
        predict(h, lambda mu: mu, jnp.cumsum(level, axis=-2))  # ty: ignore[invalid-argument-type]

    model = as_model(bad_body)
    with pytest.raises(TypeError, match="returns a Distribution, got ArrayImpl"):
        trace(seed(model, random.PRNGKey(1))).get_trace(empty_covariates(12), jnp.zeros((12, 1)))


def test_predict_rejects_float_data_for_discrete_obs() -> None:
    def poisson_body(h: Horizon, covariates: Array) -> None:
        rate = innovations(h, "log_rate", dist.Normal(0.0, 1.0))
        predict(h, lambda eta: dist.Poisson(jnp.exp(eta)), jnp.cumsum(rate, axis=-2))

    model = as_model(poisson_body)
    float_data = jnp.abs(random.normal(random.PRNGKey(0), (12, 1)))
    with pytest.raises(ValueError, match="discrete support"):
        trace(seed(model, random.PRNGKey(1))).get_trace(empty_covariates(12), float_data)


class _UndeterminedSupportNormal(dist.Normal):
    """A Normal whose ``support`` is numpyro's bare ``constraints.dependent`` default.

    ``constraints.dependent.is_discrete`` raises ``NotImplementedError`` (the
    support cannot be determined statically), which is what a custom
    ``Distribution`` subclass that never sets ``support`` inherits.
    """

    support = dist.constraints.dependent


def test_predict_treats_undetermined_support_as_continuous() -> None:
    """An undetermined support must not raise from the discrete-data check."""

    def body(h: Horizon, covariates: Array) -> None:
        level = innovations(h, "level", dist.Normal(0.0, 1.0))
        predict(h, lambda mu: _UndeterminedSupportNormal(mu, 0.5), jnp.cumsum(level, axis=-2))

    with pytest.raises(NotImplementedError, match="cannot be determined statically"):
        _ = dist.constraints.dependent.is_discrete
    model = as_model(body)
    float_data = random.normal(random.PRNGKey(0), (12, 1))
    tr = trace(seed(model, random.PRNGKey(1))).get_trace(empty_covariates(12), float_data)
    assert "obs" in tr


def test_predict_accepts_integer_data_for_discrete_obs() -> None:
    def poisson_body(h: Horizon, covariates: Array) -> None:
        rate = innovations(h, "log_rate", dist.Normal(0.0, 1.0))
        predict(h, lambda eta: dist.Poisson(jnp.exp(eta)), jnp.cumsum(rate, axis=-2))

    model = as_model(poisson_body)
    int_data = jnp.asarray(random.poisson(random.PRNGKey(0), 3.0, (12, 1)), dtype=jnp.int32)
    tr = trace(seed(model, random.PRNGKey(1))).get_trace(empty_covariates(12), int_data)
    assert "obs" in tr


def test_poisson_local_level_end_to_end() -> None:
    """A Poisson GLM local level: forecasts are integer, non-negative, and track the rate."""

    def poisson_body(h: Horizon, covariates: Array) -> None:
        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-1.0, 0.5))
        log_rate = innovations(h, "log_rate", dist.Normal(0.0, drift_scale))
        predict(h, lambda eta: dist.Poisson(jnp.exp(eta)), jnp.cumsum(log_rate, axis=-2))

    model = as_model(poisson_body)
    true_rate = 5.0
    data = jnp.asarray(random.poisson(random.PRNGKey(0), true_rate, (40, 1)), dtype=jnp.int32)
    mcmc = MCMC(
        NUTS(model),
        num_warmup=200,
        num_samples=200,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(1), empty_covariates(40), data)
    # MCMC posterior samples (mcmc.get_samples()) go straight to forecast(), with
    # no draw_posterior step (that's guide-based only).
    post = mcmc.get_samples()
    fc = forecast(random.PRNGKey(3), model, post, data, empty_covariates(46))
    assert fc.shape == (200, 6, 1)
    assert bool(jnp.all(fc >= 0.0))
    assert bool(jnp.all(fc == jnp.floor(fc)))  # integer-valued counts
    # The forecast median should be in the right ballpark of the true rate.
    assert 2.0 < float(jnp.median(fc)) < 10.0


def test_horizon_is_a_pytree_with_static_shapes() -> None:
    h = Horizon.from_data(jnp.zeros((25, 0)), jnp.ones((20, 1)))
    # data is the only leaf; the three ints are static metadata.
    (leaf,) = jax.tree.leaves(h)
    assert leaf is h.data

    def forecast_zeros(h: Horizon) -> Array:
        # future is static metadata, so it is a legal shape inside jit and vmap.
        assert h.data is not None
        return jnp.zeros((h.future, 1)) + h.data.sum()

    assert jax.jit(forecast_zeros)(h).shape == (5, 1)
    batched = Horizon.from_data(jnp.zeros((3, 25, 0)), jnp.ones((3, 20, 1)))
    assert jax.vmap(forecast_zeros)(batched).shape == (3, 5, 1)


def test_horizon_unflattens_with_host_leaves() -> None:
    # The beartype hook checks construction, not tree_unflatten: NumPy leaves are fine.
    h = Horizon.from_data(jnp.zeros((25, 0)), jnp.ones((20, 1)))
    host = jax.tree.map(np.asarray, h)
    assert isinstance(host.data, np.ndarray)
    assert (host.t_obs, host.future, host.duration) == (20, 5, 25)
    with pytest.raises(TypeError):
        Horizon(data=np.ones((20, 1)), t_obs=20, future=5, duration=25)  # ty: ignore[invalid-argument-type]
