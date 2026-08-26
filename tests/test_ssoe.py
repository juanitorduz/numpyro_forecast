"""Tests for :func:`ssoe`, the single-source-of-error building block."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from conftest import as_model, empty_covariates, svi_forecast_fn, svi_in_sample_fn
from example_models import croston_model
from jax import Array, random
from numpyro.handlers import seed, substitute, trace
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from numpyro_forecast.arrays import pad_future
from numpyro_forecast.convert import to_datatree
from numpyro_forecast.evaluate import backtest
from numpyro_forecast.models import Horizon, SSOEResult, ssoe
from numpyro_forecast.predictive import draw_posterior, forecast, predict_in_sample
from numpyro_forecast.typing import ForecastModel

CarryFn = Callable[[Array, Array], Any]
Body = Callable[[Horizon, Array], SSOEResult]

# --- helpers -----------------------------------------------------------------


def _series(t: int, key: int = 1) -> Array:
    """A ``(t, 1)`` random-walk series."""
    return jnp.cumsum(0.3 * random.normal(random.PRNGKey(key), (t, 1)), axis=-2)


def _capture(body: Body) -> tuple[ForecastModel, list[SSOEResult]]:
    """Wrap a body returning an :class:`SSOEResult` as a model that records the result."""
    box: list[SSOEResult] = []

    def wrapped(h: Horizon, covariates: Array) -> None:
        box.append(body(h, covariates))

    return as_model(wrapped), box


def _get_trace(
    model: ForecastModel,
    covariates: Array,
    data: Array | None,
    *,
    substitutions: dict[str, Array] | None = None,
) -> dict[str, Any]:
    fn = seed(model, random.PRNGKey(0))
    if substitutions:
        fn = substitute(fn, data=substitutions)
    return trace(fn).get_trace(covariates, data)


def _plate_frames(site: dict[str, Any]) -> list[tuple[str, int, int]]:
    return [(f.name, f.dim, f.size) for f in site["cond_indep_stack"]]


def _sites(tr: dict[str, Any]) -> set[str]:
    """Sample and deterministic site names (NumPyro also records plates in the trace)."""
    return {name for name, site in tr.items() if site["type"] != "plate"}


# --- ARMA(1,1): the hand-rolled notebook reference and the ssoe form -------------


def _arma_params() -> tuple[Array, Array, Array, Array]:
    mu = jnp.asarray(numpyro.sample("mu", dist.Normal(0.0, 1.0)))
    phi = jnp.asarray(numpyro.sample("phi", dist.Uniform(-1.0, 1.0)))
    theta = jnp.asarray(numpyro.sample("theta", dist.Uniform(-1.0, 1.0)))
    sigma = jnp.asarray(numpyro.sample("sigma", dist.HalfNormal(1.0)))
    return mu, phi, theta, sigma


def _arma_reference_body(h: Horizon, covariates: Array) -> None:
    """The hand-rolled ARMA(1,1) filter of the arma notebook, kept as the executable spec."""
    y = covariates[..., : h.t_obs, 0]
    mu, phi, theta, sigma = _arma_params()

    def transition_fn(carry: tuple[Array, Array], y_t: Array) -> tuple[tuple[Array, Array], Array]:
        y_prev, error_prev = carry
        pred = mu + phi * y_prev + theta * error_prev
        return (y_t, y_t - pred), pred

    init_carry = (mu, jnp.zeros(()))
    (_, error_last), preds = jax.lax.scan(transition_fn, init_carry, y)
    numpyro.deterministic("mu_t", preds[:, None])
    numpyro.sample("obs", dist.Normal(preds[:, None], sigma), obs=h.data)

    if h.future > 0:
        eps_future = jnp.asarray(
            numpyro.sample("eps_future", dist.Normal(0.0, sigma).expand([h.future]).to_event(1))
        )

        def forecast_fn(
            carry: tuple[Array, Array], eps: Array
        ) -> tuple[tuple[Array, Array], Array]:
            y_prev, error_prev = carry
            pred = mu + phi * y_prev + theta * error_prev
            y_next = pred + eps
            return (y_next, eps), y_next

        _, y_future = jax.lax.scan(forecast_fn, (y[-1], error_last), eps_future)
        numpyro.deterministic("forecast", y_future[:, None])


def _arma_ssoe_body(h: Horizon, covariates: Array) -> SSOEResult:
    y = covariates[..., : h.t_obs, :]
    mu, phi, theta, sigma = _arma_params()

    def step(carry: tuple[Array, Array], _: object) -> tuple[Array, CarryFn]:
        y_prev, eps_prev = carry
        return mu + phi * y_prev + theta * eps_prev, lambda y_t, eps_t: (y_t, eps_t)

    r = ssoe(h, "eps", y, (mu[None], jnp.zeros((1,))), step, dist.Normal(0.0, sigma))
    numpyro.deterministic("mu_t", r.mu)
    numpyro.sample("obs", dist.Normal(r.mu, sigma), obs=h.data)
    if h.future > 0:
        numpyro.deterministic("forecast", r.y_future)
    return r


ARMA_REFERENCE = as_model(_arma_reference_body)
ARMA_SSOE, _ = _capture(_arma_ssoe_body)

# --- ETS-shaped body: tuple carry with scalar leaves, a Python-float init leaf ---

PHI_ETS = 0.9
EtsCarry = tuple[Any, Any, Array]


def _ets_body(h: Horizon, covariates: Array) -> SSOEResult:
    y = covariates[..., : h.t_obs, :]
    sigma = jnp.asarray(numpyro.sample("sigma", dist.HalfNormal(1.0)))

    def step(carry: EtsCarry, _: object) -> tuple[Array, CarryFn]:
        level, trend, season = carry
        mu_t = jnp.asarray(level + PHI_ETS * trend + season[0])

        def carry_fn(y_t: Array, eps_t: Array) -> EtsCarry:
            e = eps_t[0]
            new_season = season[0] + 0.2 * e
            return (
                level + PHI_ETS * trend + 0.5 * e,
                PHI_ETS * trend + 0.1 * e,
                jnp.concatenate([season[1:], new_season[None]]),
            )

        return mu_t[None], carry_fn

    init: EtsCarry = (0.0, 0.0, jnp.zeros((4,)))
    r = ssoe(h, "eps", y, init, step, dist.Normal(0.0, sigma))
    numpyro.sample("obs", dist.Normal(r.mu, sigma), obs=h.data)
    if h.future > 0:
        numpyro.deterministic("forecast", r.y_future)
    return r


# --- 1. sites while training vs forecasting ----------------------------------------


@pytest.mark.parametrize("future", [0, 5])
@pytest.mark.parametrize("body", [_arma_ssoe_body, _ets_body], ids=["arma", "ets"])
def test_sites_training_vs_forecast(body: Body, future: int) -> None:
    t_obs = 12
    covariates = _series(t_obs + future)
    data = covariates[:t_obs]
    model, box = _capture(body)
    tr = _get_trace(model, covariates, data)
    r = box[-1]

    model_train, box_train = _capture(body)
    _get_trace(model_train, covariates[:t_obs], data)
    assert jnp.array_equal(box_train[-1].mu, r.mu)
    assert r.mu.shape == (t_obs, 1)

    if future == 0:
        assert "eps_future" not in tr
        assert "forecast" not in tr
        assert r.mu_future.shape == (0, 1)
        assert r.y_future.shape == (0, 1)
        assert r.mu_future.dtype == r.mu.dtype
        assert r.y_future.dtype == r.mu.dtype
    else:
        site = tr["eps_future"]
        assert site["value"].shape == (future, 1)
        assert _plate_frames(site) == [("time_future", -2, future)]
        assert r.mu_future.shape == (future, 1)
        assert r.y_future.shape == (future, 1)
        assert jnp.array_equal(tr["forecast"]["value"], r.y_future)


# --- 2. the ssoe form reproduces the hand-rolled notebook filter -------------------


@pytest.mark.parametrize("future", [0, 6])
def test_arma_matches_hand_rolled_reference(future: int) -> None:
    t_obs = 24
    covariates = _series(t_obs + future)
    data = covariates[:t_obs]
    params = {
        "mu": jnp.asarray(0.1),
        "phi": jnp.asarray(0.6),
        "theta": jnp.asarray(-0.3),
        "sigma": jnp.asarray(0.4),
    }
    eps = 0.4 * random.normal(random.PRNGKey(3), (future,))
    ref = _get_trace(ARMA_REFERENCE, covariates, data, substitutions={**params, "eps_future": eps})
    new = _get_trace(
        ARMA_SSOE, covariates, data, substitutions={**params, "eps_future": eps[:, None]}
    )
    assert _sites(ref) == _sites(new)
    assert jnp.allclose(ref["mu_t"]["value"], new["mu_t"]["value"], rtol=1e-5, atol=1e-6)
    if future > 0:
        assert jnp.allclose(
            ref["forecast"]["value"], new["forecast"]["value"], rtol=1e-5, atol=1e-6
        )


# --- 3. a frozen gate gives a flat forecast (panel and batched layouts) -------------


@pytest.mark.parametrize("layout", ["panel", "batched"])
def test_frozen_gate_forecast_is_flat(layout: str) -> None:
    t_obs, future = 10, 4
    if layout == "panel":
        y = jnp.abs(random.normal(random.PRNGKey(4), (t_obs, 3)))
        init = jnp.zeros((3,))
        noise = dist.Normal(0.0, jnp.full((3,), 0.5))
        expected = (future, 3)
    else:
        y = jnp.abs(random.normal(random.PRNGKey(4), (2, t_obs, 1)))
        init = jnp.zeros((2, 1))
        noise = dist.Normal(0.0, jnp.full((2, 1, 1), 0.5))
        expected = (2, future, 1)
    gate = y > 0.5

    def body(h: Horizon, covariates: Array) -> SSOEResult:
        def step(level: Array, gate_t: Array | None) -> tuple[Array, CarryFn]:
            assert gate_t is not None
            return level, lambda y_t, _: jnp.where(gate_t, 0.3 * y_t + 0.7 * level, level)

        return ssoe(h, "eps", y, init, step, noise, xs=pad_future(gate, h.future))

    model, box = _capture(body)
    tr = _get_trace(model, empty_covariates(t_obs + future), y)
    r = box[-1]
    eps = tr["eps_future"]["value"]
    assert eps.shape == expected
    assert r.mu.shape == (*expected[:-2], t_obs, expected[-1])
    assert r.mu_future.shape == expected
    assert jnp.array_equal(r.mu_future, jnp.broadcast_to(r.mu_future[..., :1, :], expected))
    assert jnp.array_equal(r.y_future, r.mu_future + eps)
    assert bool(jnp.any(r.y_future != r.mu_future))


# --- 4. two channels compose through the drivers -----------------------------------


def _intermittent(t: int) -> Array:
    key_size, key_event = random.split(random.PRNGKey(5))
    size = 1.0 + jnp.abs(random.normal(key_size, (t, 1)))
    event = random.bernoulli(key_event, 0.4, (t, 1))
    return jnp.where(event, size, 0.0)


def test_two_channels_compose(rng_key: Array, fast_mcmc: dict[str, int]) -> None:
    t_obs, future = 16, 3
    series = _intermittent(t_obs + future)
    data = series[:t_obs]
    params = {
        "z_smoothing",
        "z_init",
        "z_noise",
        "p_inv_smoothing",
        "p_inv_init",
        "p_inv_noise",
    }
    tr_train = _get_trace(croston_model, series[:t_obs], data)
    assert _sites(tr_train) == params | {"rate", "obs", "obs_intervals"}
    tr_fc = _get_trace(croston_model, series, data)
    assert _sites(tr_fc) == params | {
        "rate",
        "obs",
        "obs_intervals",
        "z_eps_future",
        "p_inv_eps_future",
        "rate_future",
        "forecast",
    }
    assert _plate_frames(tr_fc["z_eps_future"]) == [("z_time_future", -2, future)]
    assert _plate_frames(tr_fc["p_inv_eps_future"]) == [("p_inv_time_future", -2, future)]
    assert tr_fc["forecast"]["value"].shape == (future, 1)

    num_samples = 5
    posterior = {
        name: jnp.broadcast_to(tr_train[name]["value"], (num_samples,)) for name in params
    }
    in_sample = predict_in_sample(rng_key, croston_model, posterior, series[:t_obs])
    assert in_sample.shape == (num_samples, t_obs, 1)
    fc = forecast(rng_key, croston_model, posterior, data, series)
    assert fc.shape == (num_samples, future, 1)
    tree = to_datatree(rng_key, croston_model, posterior, data, series)
    assert {"posterior", "posterior_predictive", "predictions"} <= set(tree.children)
    assert tree["predictions"]["obs"].shape == (1, num_samples, future, 1)

    mcmc = MCMC(
        NUTS(croston_model),
        num_warmup=fast_mcmc["num_warmup"],
        num_samples=fast_mcmc["num_samples"],
        progress_bar=False,
    )
    mcmc.run(rng_key, series[:t_obs], data)
    samples = mcmc.get_samples()
    assert "rate" in samples
    assert "forecast" not in samples
    assert "rate_future" not in samples
    assert "z_eps_future" not in samples


# --- 5. the forecast scan feeds the drawn errors; clips act on the carry only ---------


def test_forecast_scan_feeds_drawn_errors_and_clips_carry() -> None:
    t_obs, future = 8, 5
    y = random.normal(random.PRNGKey(6), (t_obs, 1))
    eps = -jnp.abs(random.normal(random.PRNGKey(7), (future, 1))) - 0.5
    covariates = empty_covariates(t_obs + future)

    def identity_body(h: Horizon, covariates: Array) -> SSOEResult:
        def step(carry: Array, _: object) -> tuple[Array, CarryFn]:
            return carry, lambda y_t, eps_t: eps_t

        return ssoe(h, "eps", y, jnp.zeros((1,)), step, dist.Normal(0.0, 1.0))

    model, box = _capture(identity_body)
    _get_trace(model, covariates, y, substitutions={"eps_future": eps})
    r = box[-1]
    assert jnp.array_equal(r.mu[1:], y[:-1] - r.mu[:-1])
    assert jnp.array_equal(r.mu_future[0], y[-1] - r.mu[-1])
    assert jnp.array_equal(r.mu_future[1:], eps[:-1])

    def clipping_body(h: Horizon, covariates: Array) -> SSOEResult:
        def step(carry: Array, _: object) -> tuple[Array, CarryFn]:
            return carry, lambda y_t, eps_t: jnp.clip(y_t, 0.0)

        return ssoe(h, "eps", y, jnp.zeros((1,)), step, dist.Normal(0.0, 1.0))

    model, box = _capture(clipping_body)
    _get_trace(model, covariates, y, substitutions={"eps_future": eps})
    r = box[-1]
    assert bool(jnp.any(r.y_future < 0.0))
    assert jnp.array_equal(r.mu_future[1:], jnp.clip(r.y_future[:-1], 0.0))


def test_xs_pytree_reaches_step_and_censored_obs_samples_under_data_none() -> None:
    t_obs, future = 10, 3
    duration = t_obs + future
    key_y, key_a, key_c = random.split(random.PRNGKey(8), 3)
    sales = jnp.abs(random.normal(key_y, (duration, 1)))
    available = random.bernoulli(key_a, 0.8, (duration, 1)).astype(sales.dtype)
    censored = random.bernoulli(key_c, 0.2, (duration, 1)).astype(sales.dtype)
    seasonal = jnp.sin(jnp.arange(duration, dtype=sales.dtype) / 3.0)[:, None]
    covariates = jnp.concatenate([sales, available, censored, seasonal], axis=-1)
    data = sales[:t_obs]

    def censored_body(h: Horizon, covariates: Array) -> SSOEResult:
        y = covariates[..., : h.t_obs, 0:1]
        available = covariates[..., : h.t_obs, 1:2]
        censored = covariates[..., : h.t_obs, 2:3]
        seasonal = covariates[..., 3:4]
        sigma = jnp.asarray(numpyro.sample("sigma", dist.HalfNormal(1.0)))

        def step(
            carry: tuple[Array, Array], x_t: tuple[Array, Array, Array] | None
        ) -> tuple[Array, CarryFn]:
            assert x_t is not None
            seasonal_t, available_t, censored_t = x_t
            lag_1, lag_2 = carry
            pred = 0.5 * lag_1 + 0.2 * lag_2 + seasonal_t

            def carry_fn(y_t: Array, _: Array) -> tuple[Array, Array]:
                on_shelf = jnp.where(censored_t == 1, jnp.maximum(y_t, pred), y_t)
                return jnp.clip(jnp.where(available_t == 1, on_shelf, pred), 0.0), lag_1

            return pred, carry_fn

        xs = (seasonal, pad_future(available, h.future, value=1.0), pad_future(censored, h.future))
        r = ssoe(h, "eps", y, (y[0], y[0]), step, dist.Normal(0.0, sigma), xs=xs)
        valid = (jnp.arange(h.t_obs)[:, None] >= 2) & (available == 1)
        numpyro.sample(
            "obs",
            dist.RightCensoredDistribution(dist.Normal(r.mu, sigma), censored=censored).mask(
                valid
            ),
            obs=h.data,
        )
        if h.future > 0:
            numpyro.deterministic("forecast", jnp.clip(r.y_future, 0.0))
        return r

    model, box = _capture(censored_body)
    tr = _get_trace(model, covariates, data)
    r = box[-1]
    assert jnp.allclose(r.mu[0], 0.7 * sales[0] + seasonal[0])
    assert tr["forecast"]["value"].shape == (future, 1)
    assert bool(jnp.all(tr["forecast"]["value"] >= 0.0))

    posterior = {"sigma": jnp.full((3,), 0.5)}
    in_sample = predict_in_sample(random.PRNGKey(0), model, posterior, covariates[:t_obs])
    assert in_sample.shape == (3, t_obs, 1)
    assert bool(jnp.all(jnp.isfinite(in_sample)))


# --- 6. validation messages ------------------------------------------------------------


def _run_body(body: Callable[[Horizon], Any], t_obs: int = 6, future: int = 2) -> None:
    def model(covariates: Array, data: Array | None = None) -> None:
        body(Horizon.from_data(covariates, data))

    _get_trace(model, empty_covariates(t_obs + future), jnp.zeros((t_obs, 1)))


def _identity_step(carry: Array, _: object) -> tuple[Array, CarryFn]:
    return carry, lambda y_t, eps_t: carry


T_OBS, FUTURE = 6, 2
_ONE = jnp.zeros((1,))


def _y_none(h: Horizon) -> None:
    ssoe(h, "eps", None, _ONE, _identity_step, dist.Normal(0.0, 1.0))


def _y_1d(h: Horizon) -> None:
    ssoe(h, "eps", jnp.zeros((h.t_obs,)), _ONE, _identity_step, dist.Normal(0.0, 1.0))


def _y_long(h: Horizon) -> None:
    ssoe(h, "eps", jnp.zeros((h.t_obs + 1, 1)), _ONE, _identity_step, dist.Normal(0.0, 1.0))


def _xs_short(h: Horizon) -> None:
    y = jnp.zeros((h.t_obs, 1))
    ssoe(h, "eps", y, _ONE, _identity_step, dist.Normal(0.0, 1.0), xs={"gate": y})


def _xs_1d(h: Horizon) -> None:
    y = jnp.zeros((h.t_obs, 1))
    xs = jnp.zeros((h.duration,))
    ssoe(h, "eps", y, _ONE, _identity_step, dist.Normal(0.0, 1.0), xs=xs)


def _mu_scalar(h: Horizon) -> None:
    def step(carry: Array, _: object) -> tuple[Array, CarryFn]:
        return carry[0], lambda y_t, eps_t: carry

    ssoe(h, "eps", jnp.zeros((h.t_obs, 1)), _ONE, step, dist.Normal(0.0, 1.0))


def _carry_tree(h: Horizon) -> None:
    def step(carry: tuple[Array], _: object) -> tuple[Array, CarryFn]:
        return carry[0], lambda y_t, eps_t: (y_t, eps_t)

    ssoe(h, "eps", jnp.zeros((h.t_obs, 1)), (_ONE,), step, dist.Normal(0.0, 1.0))


def _carry_shape(h: Horizon) -> None:
    def step(carry: Array, _: object) -> tuple[Array, CarryFn]:
        return carry, lambda y_t, eps_t: jnp.zeros((2,))

    ssoe(h, "eps", jnp.zeros((h.t_obs, 1)), _ONE, step, dist.Normal(0.0, 1.0))


def _carry_dtype(h: Horizon) -> None:
    def step(carry: Array, _: object) -> tuple[Array, CarryFn]:
        return carry.astype(jnp.float32), lambda y_t, eps_t: y_t

    init = jnp.zeros((1,), dtype=jnp.int32)
    ssoe(h, "eps", jnp.zeros((h.t_obs, 1)), init, step, dist.Normal(0.0, 1.0))


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (_y_none, "data=None"),
        (_y_1d, "time at axis -2"),
        (_y_long, "exactly the observed window"),
        (_xs_short, r"span the full horizon.*\['gate'\]"),
        (_xs_1d, r"Add the axis"),
        (_mu_scalar, "per-step mean"),
        (_carry_tree, "same tree structure"),
        (_carry_shape, r"changed carry leaf"),
        (_carry_dtype, r"astype"),
    ],
    ids=[
        "y-none",
        "y-1d",
        "y-long",
        "xs-short",
        "xs-1d",
        "mu-scalar",
        "carry-tree",
        "carry-shape",
        "carry-dtype",
    ],
)
def test_validation_messages(body: Callable[[Horizon], None], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _run_body(body, T_OBS, FUTURE)


# --- 7. the future-error shape is enforced; step must not sample -----------------------


@pytest.mark.parametrize(
    ("noise_fn", "y_shape", "init_shape"),
    [
        (lambda future: dist.Normal(0.0, 1.0), (T_OBS, 3), (3,)),
        (lambda future: dist.Normal(0.0, jnp.ones((future, 1))), (T_OBS, 3), (3,)),
        (lambda future: dist.Normal(0.0, jnp.ones((1,))), (2, T_OBS, 1), (2, 1)),
        (lambda future: dist.Normal(0.0, 1.0).expand([3]).to_event(1), (T_OBS, 3), (3,)),
    ],
    ids=["scalar-vs-obs3", "future-as-batch", "obs-vs-panel", "event-shaped"],
)
def test_noise_shape_is_enforced(
    noise_fn: Callable[[int], dist.Distribution],
    y_shape: tuple[int, ...],
    init_shape: tuple[int, ...],
) -> None:
    y = jnp.zeros(y_shape)

    def body(h: Horizon) -> None:
        ssoe(h, "eps", y, jnp.zeros(init_shape), _identity_step, noise_fn(h.future))

    _run_body(body, T_OBS, 0)
    with pytest.raises(ValueError, match="time_future plate"):
        _run_body(body, T_OBS, FUTURE)


def test_step_must_not_sample() -> None:
    def body(h: Horizon) -> None:
        def step(carry: Array, _: object) -> tuple[Array, CarryFn]:
            shock = jnp.asarray(numpyro.sample("shock", dist.Normal(0.0, 1.0)))
            return carry + shock, lambda y_t, eps_t: y_t

        ssoe(h, "eps", jnp.zeros((h.t_obs, 1)), _ONE, step, dist.Normal(0.0, 1.0))

    with pytest.raises(ValueError, match=r"shock.*markov_series"):
        _run_body(body, T_OBS, 0)


# --- 8. backtest with the series as covariate; the frozen gate survives real rows --------


def test_backtest_with_series_as_covariate(rng_key: Array) -> None:
    series = _series(40, key=9)
    results = backtest(
        rng_key,
        series,
        series,
        lambda: ARMA_SSOE,
        forecast_fn=svi_forecast_fn(20),
        in_sample_fn=svi_in_sample_fn(20),
        min_train_window=30,
        test_window=5,
        stride=5,
        num_samples=10,
        eval_train=True,
        keep_predictions=True,
    )
    assert [r.t1 for r in results] == [30, 35]
    for r in results:
        assert r.prediction is not None
        assert r.prediction.shape == (10, 5, 1)
        assert all(jnp.isfinite(v) for v in r.metrics.values())
        assert all(jnp.isfinite(v) for v in r.train_metrics.values())

    t_obs, future = 16, 4
    full = _intermittent(t_obs + future)
    assert bool(jnp.any(full[t_obs:] > 0))  # real future rows carry demand
    data = full[:t_obs]
    tr = _get_trace(croston_model, full[:t_obs], data)
    posterior = {
        name: jnp.broadcast_to(tr[name]["value"], (4,))
        for name in ("z_smoothing", "z_init", "z_noise", "p_inv_smoothing", "p_inv_init")
    } | {"p_inv_noise": jnp.broadcast_to(tr["p_inv_noise"]["value"], (4,))}
    predictive = Predictive(
        croston_model, posterior_samples=posterior, return_sites=["rate_future"]
    )
    rate_future = predictive(rng_key, full, data)["rate_future"]
    assert rate_future.shape == (4, future, 1)
    assert jnp.array_equal(rate_future, jnp.broadcast_to(rate_future[:, :1], rate_future.shape))


# --- 9. AR(1) round trip against the closed form -------------------------------------


PHI_AR = 0.8
SIGMA_AR = 0.1


def _ar1_ssoe_body(h: Horizon, covariates: Array) -> SSOEResult:
    y = covariates[..., : h.t_obs, :]
    phi = jnp.asarray(numpyro.sample("phi", dist.Uniform(-1.0, 1.0)))
    sigma = jnp.asarray(numpyro.sample("sigma", dist.HalfNormal(1.0)))

    def step(carry: Array, _: object) -> tuple[Array, CarryFn]:
        return phi * carry, lambda y_t, eps_t: y_t

    r = ssoe(h, "eps", y, jnp.zeros((1,)), step, dist.Normal(0.0, sigma))
    numpyro.sample("obs", dist.Normal(r.mu, sigma), obs=h.data)
    if h.future > 0:
        numpyro.deterministic("forecast", r.y_future)
    return r


AR1_SSOE, _ = _capture(_ar1_ssoe_body)


def _ar1_series(t: int) -> Array:
    noise = SIGMA_AR * random.normal(random.PRNGKey(10), (t,))

    def step(y_prev: Array, eps: Array) -> tuple[Array, Array]:
        y_t = PHI_AR * y_prev + eps
        return y_t, y_t

    _, y = jax.lax.scan(step, jnp.asarray(0.5), noise)
    return y[:, None]


@pytest.mark.parametrize("fitter", ["svi", "mcmc"])
def test_ar1_round_trip_closed_form(fitter: str, rng_key: Array) -> None:
    t_obs, future = 60, 6
    series = _ar1_series(t_obs + future)
    data = series[:t_obs]
    if fitter == "svi":
        key_fit, key_draw = random.split(rng_key)
        guide = AutoNormal(AR1_SSOE)
        svi = SVI(AR1_SSOE, guide, numpyro.optim.Adam(0.02), Trace_ELBO())
        state = svi.run(key_fit, 600, series[:t_obs], data, progress_bar=False)
        post = draw_posterior(key_draw, guide, state.params, 400)
        tree = to_datatree(random.PRNGKey(1), AR1_SSOE, post, data, series)
        assert tree["predictions"]["obs"].shape == (1, 400, future, 1)
    else:
        mcmc = MCMC(NUTS(AR1_SSOE), num_warmup=100, num_samples=400, progress_bar=False)
        mcmc.run(rng_key, series[:t_obs], data)
        post = mcmc.get_samples()
    phi = jnp.asarray(post["phi"])
    assert abs(float(phi.mean()) - PHI_AR) < 0.15
    preds = forecast(random.PRNGKey(2), AR1_SSOE, post, data, series, batch_size=100)
    mean_fc = preds.mean(axis=0)[:, 0]
    powers = jnp.arange(1, future + 1)
    closed = (phi[:, None] ** powers).mean(axis=0) * data[-1, 0]
    assert jnp.allclose(mean_fc, closed, rtol=0.05, atol=0.03)
