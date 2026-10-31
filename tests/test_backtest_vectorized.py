"""Tests for `backtest_vectorized()` (roadmap §4.3).

Covers estimator equivalence with the loop `backtest()` (window indices +
statistical metric closeness), every validator, the window-count formula, the
compile-discipline gate (I3: the vmapped SVI fit does not recompile per window),
and the I6 acceptance check that a subsequent eager plain-NumPyro SVI fit with a
fresh guide runs without an ``UnexpectedTracerError``.
"""

import threading
import types
from collections.abc import Callable
from contextlib import AbstractContextManager
from functools import partial

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from conftest import rw_model, svi_forecast_fn
from jax import Array, random
from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import (
    AutoGuide,
    AutoGuideList,
    AutoIAFNormal,
    AutoLaplaceApproximation,
    AutoMultivariateNormal,
    AutoNormal,
)

from numpyro_forecast import Horizon, predict
from numpyro_forecast.evaluate import (
    DEFAULT_METRICS,
    VectorizedBacktestResult,
    _window_key_streams,
    backtest,
    backtest_vectorized,
    eval_coverage,
    eval_crps,
)
from numpyro_forecast.exceptions import BacktestWindowError, VectorizedMetricError
from numpyro_forecast.predictive import draw_posterior, forecast
from numpyro_forecast.typing import Guide

TRAIN, TEST = 30, 5


def _series(duration: int) -> tuple[Array, Array]:
    """A smooth univariate series ``(duration, 1)`` and empty covariates."""
    t = jnp.linspace(0.0, 6.0, duration)
    y = (jnp.sin(t) + 0.05 * random.normal(random.PRNGKey(0), (duration,)))[:, None]
    return y, jnp.zeros((duration, 0))


def test_window_indices_match_loop_backtest() -> None:
    """Vectorized window indices equal the rolling-window loop ``backtest``."""
    duration = 55
    data, cov = _series(duration)
    stride = 3
    loop = backtest(
        random.PRNGKey(0),
        lambda: rw_model,
        data,
        cov,
        forecast_fn=svi_forecast_fn(num_steps=50),
        train_window=TRAIN,
        test_window=TEST,
        stride=stride,
        num_samples=20,
    )
    model = rw_model
    vec = backtest_vectorized(
        random.PRNGKey(0),
        lambda: model,
        data,
        cov,
        train_window=TRAIN,
        test_window=TEST,
        guide=AutoNormal(model),
        stride=stride,
        num_steps=50,
        num_samples=20,
    )
    assert [int(x) for x in vec.t0] == [r.t0 for r in loop]
    assert [int(x) for x in vec.t1] == [r.t1 for r in loop]
    assert [int(x) for x in vec.t2] == [r.t2 for r in loop]


def _trend_model(covariates: Array, data: Array | None = None) -> None:
    """A linear trend in the time covariate, with only global latents.

    The data determine the three parameters well, so a converged SVI fit gives
    almost the same forecast for every seed. The forecast of ``rw_model``
    depends on the fitted level of every training step, so its MAE over a
    5-step window changes by a factor of up to 10 with the seed.
    """
    h = Horizon.from_data(covariates, data)
    level = jnp.asarray(numpyro.sample("level", dist.Normal(0.0, 1.0)))
    slope = jnp.asarray(numpyro.sample("slope", dist.Normal(0.0, 1.0)))
    sigma = jnp.asarray(numpyro.sample("sigma", dist.LogNormal(-2.0, 1.0)))
    predict(h, dist.Normal(0.0, sigma), level + slope * covariates)


def test_metrics_statistically_close_to_loop() -> None:
    """Per-window MAE and CRPS of the two backtests agree, averaged over seeds.

    The two estimators use different PRNG layouts, so one seed compares two
    random draws. Both start each window from its own ``init_to_uniform``
    point. From a far start the fit can need about 4,000 steps of
    ``Adam(0.01)`` to converge (2,000 steps left some windows of both
    estimators at a 3 to 8 times larger MAE), so the fits run 5,000 steps. The
    mean over a few seeds of a well-determined model then removes most of the
    noise: over 6 seeds and all 20 triples of them, the largest per-window
    relative difference of the 3-seed mean MAE was about 0.35, so a tolerance
    of 0.5 checks agreement and does not depend on the seeds.
    """
    duration = 50
    time = jnp.linspace(0.0, 1.0, duration)[:, None]
    data = 0.5 + 2.0 * time + 0.1 * random.normal(random.PRNGKey(0), (duration, 1))
    model = _trend_model
    loop_metrics: dict[str, list[Array]] = {"mae": [], "crps": []}
    vec_metrics: dict[str, list[Array]] = {"mae": [], "crps": []}
    for seed in range(3):
        loop = backtest(
            random.PRNGKey(seed),
            lambda: model,
            data,
            time,
            forecast_fn=svi_forecast_fn(num_steps=5_000),
            train_window=TRAIN,
            test_window=TEST,
            stride=5,
            num_samples=200,
        )
        vec = backtest_vectorized(
            random.PRNGKey(seed),
            lambda: model,
            data,
            time,
            train_window=TRAIN,
            test_window=TEST,
            guide=AutoNormal(model),
            stride=5,
            num_steps=5_000,
            num_samples=200,
        )
        for name in loop_metrics:
            loop_metrics[name].append(jnp.array([r.metrics[name] for r in loop]))
            vec_metrics[name].append(jnp.asarray(vec.metrics[name]))
    for name in loop_metrics:
        loop_mean = jnp.stack(loop_metrics[name]).mean(axis=0)
        vec_mean = jnp.stack(vec_metrics[name]).mean(axis=0)
        assert bool(jnp.all(jnp.isfinite(loop_mean)))
        assert bool(jnp.all(jnp.isfinite(vec_mean)))
        rel_err = jnp.abs(vec_mean - loop_mean) / loop_mean
        assert float(rel_err.max()) < 0.5, (name, loop_mean, vec_mean)


@pytest.mark.parametrize(
    ("train_window", "test_window", "stride", "message"),
    [
        (0, TEST, 1, "train_window must be >= 1"),
        (TRAIN, 0, 1, "test_window must be >= 1"),
        (TRAIN, TEST, 0, "stride must be >= 1"),
    ],
)
def test_window_size_validators(
    train_window: int, test_window: int, stride: int, message: str
) -> None:
    """Each window-size/stride constraint raises its own ``BacktestWindowError``."""
    data, cov = _series(50)
    model = rw_model
    with pytest.raises(BacktestWindowError, match=message):
        backtest_vectorized(
            random.PRNGKey(0),
            lambda: model,
            data,
            cov,
            train_window=train_window,
            test_window=test_window,
            guide=AutoNormal(model),
            stride=stride,
            num_steps=10,
        )


def _rw_mean_field_guide(covariates: Array, data: Array | None = None) -> None:
    """A hand-written mean-field guide over ``rw_model``'s sites for a ``TRAIN``-long window.

    Mirrors the model's ``time`` plate for the ``drift`` innovations so the
    guide's site shapes match the model's ``(TRAIN, 1)`` layout.
    """
    positive = dist.constraints.positive
    scale_loc = numpyro.param("drift_scale_loc", -1.0)
    scale_sd = numpyro.param("drift_scale_sd", 1.0, constraint=positive)
    numpyro.sample("drift_scale", dist.LogNormal(scale_loc, scale_sd))
    sigma_loc = numpyro.param("sigma_loc", -1.0)
    sigma_sd = numpyro.param("sigma_sd", 1.0, constraint=positive)
    numpyro.sample("sigma", dist.LogNormal(sigma_loc, sigma_sd))
    drift_loc = numpyro.param("drift_loc", jnp.zeros((TRAIN, 1)))
    drift_sd = numpyro.param("drift_sd", 0.1 * jnp.ones((TRAIN, 1)), constraint=positive)
    with numpyro.plate("time", TRAIN, dim=-2):
        numpyro.sample("drift", dist.Normal(drift_loc, drift_sd))


def test_handwritten_guide_runs_and_matches_autoguide_shapes() -> None:
    """A hand-written guide is fitted and sampled in the same vmapped run as an autoguide.

    Any guide with the model's signature is accepted: an autoguide is sampled
    through ``sample_posterior``, a plain guide function through
    ``Predictive(guide, params=...)`` on each window's training data. Both
    runs must produce the same window indices and result shapes.
    """
    data, cov = _series(50)
    hand = backtest_vectorized(
        random.PRNGKey(0),
        lambda: rw_model,
        data,
        cov,
        train_window=TRAIN,
        test_window=TEST,
        guide=_rw_mean_field_guide,
        stride=5,
        num_steps=50,
        num_samples=20,
        keep_predictions=True,
    )
    auto = backtest_vectorized(
        random.PRNGKey(0),
        lambda: rw_model,
        data,
        cov,
        train_window=TRAIN,
        test_window=TEST,
        guide=AutoNormal(rw_model),
        stride=5,
        num_steps=50,
        num_samples=20,
        keep_predictions=True,
    )
    for name in ("t0", "t1", "t2"):
        assert jnp.array_equal(getattr(hand, name), getattr(auto, name))
    assert hand.losses.shape == auto.losses.shape
    assert bool(jnp.all(jnp.isfinite(hand.losses)))
    assert hand.metrics.keys() == auto.metrics.keys()
    for name, values in hand.metrics.items():
        assert values.shape == auto.metrics[name].shape
        assert bool(jnp.all(jnp.isfinite(values)))
    assert hand.predictions is not None and auto.predictions is not None
    assert hand.predictions.shape == auto.predictions.shape
    assert bool(jnp.all(jnp.isfinite(hand.predictions)))


def test_duration_too_short_rejected() -> None:
    """A series with no room for a single window raises ``BacktestWindowError``."""
    data, cov = _series(TRAIN + TEST - 1)
    model = rw_model
    with pytest.raises(BacktestWindowError, match="no window fits"):
        backtest_vectorized(
            random.PRNGKey(0),
            lambda: model,
            data,
            cov,
            train_window=TRAIN,
            test_window=TEST,
            guide=AutoNormal(model),
            num_steps=10,
        )


def test_covariates_without_time_axis_rejected() -> None:
    """1-D covariates name the ``(time, k)`` layout instead of an ``IndexError``."""
    data, _ = _series(50)
    with pytest.raises(ValueError, match=r"covariates must have time at axis -2"):
        backtest_vectorized(
            random.PRNGKey(0),
            lambda: rw_model,
            data,
            jnp.zeros(50),
            train_window=TRAIN,
            test_window=TEST,
            guide=AutoNormal(rw_model),
        )


def test_covariate_length_mismatch_rejected() -> None:
    """Mismatched data/covariate durations raise ``ValueError``."""
    data, _ = _series(50)
    cov = jnp.zeros((49, 0))
    model = rw_model
    with pytest.raises(ValueError, match="share the time axis"):
        backtest_vectorized(
            random.PRNGKey(0),
            lambda: model,
            data,
            cov,
            train_window=TRAIN,
            test_window=TEST,
            guide=AutoNormal(model),
            num_steps=10,
        )


def _never_called_guide(*_args: object, **_kwargs: object) -> None:
    """A guide that fails the test if ``backtest_vectorized`` ever fits or samples it."""
    pytest.fail("the guide should never have been called")


@pytest.mark.parametrize(
    ("num_samples", "metrics", "match"),
    [
        (0, None, "num_samples must be positive"),
        (0, {}, "num_samples must be positive"),
        (1, None, r"num_samples=1 is too small for the metrics \['crps'\]"),
        (1, {"crps_half": partial(eval_crps, batch_size=2)}, "num_samples=1 is too small"),
    ],
)
def test_too_few_samples_rejected_before_any_fit(
    num_samples: int, metrics: dict[str, Callable[..., Array]] | None, match: str
) -> None:
    """A ``num_samples`` the metrics cannot score raises before the SVI fit."""
    data, cov = _series(50)
    with pytest.raises(ValueError, match=match):
        backtest_vectorized(
            random.PRNGKey(0),
            lambda: rw_model,
            data,
            cov,
            train_window=TRAIN,
            test_window=TEST,
            guide=_never_called_guide,
            num_steps=10,
            num_samples=num_samples,
            metrics=metrics,
        )


@pytest.mark.parametrize(
    ("duration", "train", "test", "stride"),
    [
        (50, 30, 5, 1),
        (50, 30, 5, 3),
        (60, 20, 10, 2),
        (40, 25, 5, 4),
    ],
)
def test_window_count_matches_formula(duration: int, train: int, test: int, stride: int) -> None:
    """The number of windows follows ``floor((D - train - test) / stride) + 1``."""
    data, cov = _series(duration)
    model = rw_model
    result = backtest_vectorized(
        random.PRNGKey(0),
        lambda: model,
        data,
        cov,
        train_window=train,
        test_window=test,
        guide=AutoNormal(model),
        stride=stride,
        num_steps=10,
        num_samples=5,
    )
    expected = (duration - train - test) // stride + 1
    assert result.t0.shape[0] == expected


def test_single_svi_compilation(
    count_compilations: Callable[[], AbstractContextManager[types.SimpleNamespace]],
) -> None:
    """I3: the fused fit is O(1) in window count, not O(num_windows).

    The vmapped SVI fit treats windows as a batch dimension, so compilation
    count stays bounded (fit + posterior + forecast stages) rather than scaling
    linearly with the number of windows. Replaying the same shapes triggers no
    further backend compilations.

    A cold process pays, once, for every small NumPyro/JAX primitive this call
    touches (distribution log_probs, the optimizer update, scan bodies, ...)
    the first time it ever sees that shape signature -- a process-global JIT
    dispatch cache keyed by abstract shape, shared across every test in the
    session, not by which test triggers it first. That one-time cost is
    unrelated to the invariant under test (whether *this* fused call's own
    compilation scales with window count or repeated calls) and made the test
    order-dependent: ~3 backend compiles when warmed by earlier tests in the
    same process/xdist worker, 75-132+ standalone (measured directly; see the
    warm-up below). So an untracked warm-up call with identical shapes runs
    first to prime those caches; only the *counted* calls below measure this
    call's own (repeat) compilation behavior, which is what I3 actually claims.
    """
    import jax

    duration = TRAIN + TEST + 9  # exactly 10 windows at stride 1
    data, cov = _series(duration)
    model = rw_model

    def run() -> None:
        result = backtest_vectorized(
            random.PRNGKey(0),
            lambda: model,
            data,
            cov,
            train_window=TRAIN,
            test_window=TEST,
            guide=AutoNormal(model),
            stride=1,
            num_steps=30,
            num_samples=10,
        )
        jax.block_until_ready(result.losses)

    # Untracked warm-up: absorbs the cold-process, order-dependent compilation
    # cost described above so the counted calls below isolate the invariant.
    run()

    with count_compilations() as tally:
        run()
    first = int(tally.count)
    # Warmed, a repeat call with identical shapes measured a stable 3 backend
    # compiles across repeated fresh-process runs (the fresh AutoNormal instance
    # of each call is traced and compiled again, plus a couple of residual
    # traces); this is a generous margin above that, and nowhere near
    # the 75-132+ a cold process pays.
    assert first < 10

    with count_compilations() as tally:
        run()
    # A further identical call should not compile more than the previous one.
    assert tally.count <= max(first, 1)


def test_guide_instance_stays_fresh_for_later_calls_and_eager_fits() -> None:
    """I6 acceptance: the guide passed in is left unset and usable afterwards.

    Each window sets up its own copy of the guide under ``vmap``, so no tracer
    reaches the caller's instance: it can be passed to a second vectorized run
    (which rejects an already set-up guide) and then fitted eagerly without an
    ``UnexpectedTracerError``.
    """
    duration = TRAIN + TEST + 7
    data, cov = _series(duration)
    model = rw_model
    guide = AutoNormal(model)
    for seed in range(2):
        backtest_vectorized(
            random.PRNGKey(seed),
            lambda: model,
            data,
            cov,
            train_window=TRAIN,
            test_window=TEST,
            guide=guide,
            num_steps=30,
            num_samples=10,
        )
    train_data = data[:TRAIN]
    train_cov = cov[:TRAIN]
    full_cov = cov[: TRAIN + TEST]
    key_fit, key_draw = random.split(random.PRNGKey(2))
    svi = SVI(rw_model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    state = svi.run(key_fit, 30, train_cov, train_data, progress_bar=False)
    posterior = draw_posterior(key_draw, guide, state.params, 10)
    preds = forecast(random.PRNGKey(4), rw_model, posterior, train_data, full_cov)
    assert preds.shape[0] == 10
    assert jnp.all(jnp.isfinite(preds))


def test_keep_predictions_shape() -> None:
    """``keep_predictions=True`` retains ``(windows, samples, test, obs)`` forecasts."""
    duration = TRAIN + TEST + 6
    data, cov = _series(duration)
    model = rw_model
    result = backtest_vectorized(
        random.PRNGKey(0),
        lambda: model,
        data,
        cov,
        train_window=TRAIN,
        test_window=TEST,
        guide=AutoNormal(model),
        num_steps=20,
        num_samples=15,
        keep_predictions=True,
    )
    assert isinstance(result, VectorizedBacktestResult)
    assert result.predictions is not None
    num_windows = result.t0.shape[0]
    assert result.predictions.shape == (num_windows, 15, TEST, 1)


def test_window_streams_are_disjoint() -> None:
    """Per-window init/posterior/forecast keys are pairwise distinct, never a parent.

    Guards the PRNG-hygiene rule from round-1 item 1: ``fold_in(rng_key, i)`` is
    a split parent only, so no stream may reuse it directly.
    """
    import jax

    rng_key = random.PRNGKey(0)
    num_windows = 2
    init_keys, post_keys, fc_keys = _window_key_streams(rng_key, num_windows)
    parents = jax.vmap(lambda i: random.fold_in(rng_key, i))(jnp.arange(num_windows))

    keys = []
    for i in range(num_windows):
        keys += [init_keys[i], post_keys[i], fc_keys[i], parents[i]]
    raw = [tuple(int(x) for x in jnp.ravel(random.key_data(k))) for k in keys]
    assert len(set(raw)) == len(raw)


def _level_model(covariates: Array, data: Array | None = None) -> None:
    """A constant-level model over 2 series whose ``forecast`` is the sampled level ``mu``.

    Two series give a 2-dimensional latent, the minimum for ``AutoIAFNormal``.
    """
    h = Horizon.from_data(covariates, data)
    with numpyro.plate("series", 2, dim=-1):
        mu = jnp.asarray(numpyro.sample("mu", dist.Normal(0.0, 10.0)))
        with numpyro.plate("time", h.t_obs, dim=-2):
            numpyro.sample("obs", dist.Normal(mu, 1.0), obs=data)
    if h.future:
        numpyro.deterministic("forecast", jnp.broadcast_to(mu, (h.future, 2)))


def _keyed_level_guide(covariates: Array, data: Array | None = None) -> None:
    """A hand-written guide whose initial ``loc`` is drawn from the SVI-init key."""
    loc = numpyro.param("loc", lambda key: random.normal(key, (2,)))
    with numpyro.plate("series", 2, dim=-1):
        numpyro.sample("mu", dist.Normal(loc, 1e-6))


def _starting_points(monkeypatch: pytest.MonkeyPatch, rng_key: Array, guide: Guide) -> Array:
    """Each window's forecast of the first ``mu`` after ``num_steps=0``.

    Every window has the same (all-zero) data, and the posterior and forecast
    keys of window 0 are reused for all windows, so the forecasts differ
    across windows only if the SVI starting points differ.
    """

    def shared_draw_keys(key: Array, num_windows: int) -> tuple[Array, Array, Array]:
        init_keys, post_keys, fc_keys = _window_key_streams(key, num_windows)
        post_keys = jnp.broadcast_to(post_keys[0], post_keys.shape)
        fc_keys = jnp.broadcast_to(fc_keys[0], fc_keys.shape)
        return init_keys, post_keys, fc_keys

    monkeypatch.setattr("numpyro_forecast.evaluate._window_key_streams", shared_draw_keys)
    data, cov = jnp.zeros((30, 2)), jnp.zeros((30, 0))
    result = backtest_vectorized(
        rng_key,
        lambda: _level_model,
        data,
        cov,
        train_window=10,
        test_window=2,
        stride=5,
        guide=guide,
        num_steps=0,
        num_samples=2,
        metrics={},
        keep_predictions=True,
    )
    assert result.predictions is not None
    return result.predictions[:, 0, 0, 0]


@pytest.mark.parametrize(
    "make_guide",
    [
        pytest.param(lambda: AutoNormal(_level_model), id="AutoNormal"),
        pytest.param(lambda: AutoMultivariateNormal(_level_model), id="AutoMultivariateNormal"),
        pytest.param(lambda: AutoIAFNormal(_level_model), id="AutoIAFNormal"),
    ],
)
def test_autoguide_windows_start_where_a_fresh_guide_starts(
    monkeypatch: pytest.MonkeyPatch, make_guide: Callable[[], AutoGuide]
) -> None:
    """Each window starts where a fresh guide initialized with its SVI-init key starts.

    The guide copy of each window runs its setup (``init_loc_fn``, here the
    default ``init_to_uniform``, and the network weights of a flow) with that
    window's SVI-init key, as a fresh guide per window in `backtest()` does, so
    the windows start from different points. With ``num_steps=0`` and the shared
    posterior key, each window's first draw equals the first draw of a fresh
    guide initialized with that window's key.
    """
    rng_key = random.PRNGKey(0)
    starts = _starting_points(monkeypatch, rng_key, make_guide())
    init_keys, post_keys, _ = _window_key_streams(rng_key, starts.shape[0])
    expected = []
    for key in init_keys:
        guide = make_guide()
        svi = SVI(_level_model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
        state = svi.init(key, jnp.zeros((10, 0)), jnp.zeros((10, 2)))
        params = svi.get_params(state)
        expected.append(
            guide.sample_posterior(post_keys[0], params, sample_shape=(2,))["mu"][0, 0]
        )
    assert jnp.allclose(starts, jnp.stack(expected), atol=1e-4)
    assert len({round(float(x), 4) for x in starts}) == starts.shape[0]


def test_keyed_param_initializer_uses_the_per_window_init_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callable ``numpyro.param`` initializer starts each window differently."""
    starts = _starting_points(monkeypatch, random.PRNGKey(0), _keyed_level_guide)
    assert len({round(float(x), 4) for x in starts}) == starts.shape[0]


def test_result_schema_and_dataframe_row_shape() -> None:
    """Result fields are per-window and results_to_dataframe is one row per window."""
    from numpyro_forecast.evaluate import results_to_dataframe

    duration = TRAIN + TEST + 8
    data, cov = _series(duration)
    model = rw_model
    result = backtest_vectorized(
        random.PRNGKey(0),
        lambda: model,
        data,
        cov,
        train_window=TRAIN,
        test_window=TEST,
        guide=AutoNormal(model),
        stride=1,
        num_steps=20,
        num_samples=15,
    )
    num_windows = result.t0.shape[0]
    # Every per-window field has a leading window axis.
    assert result.t0.shape == (num_windows,)
    assert result.t1.shape == (num_windows,)
    assert result.t2.shape == (num_windows,)
    for values in result.metrics.values():
        assert values.shape == (num_windows,)

    df = results_to_dataframe(result)
    assert len(df) == num_windows
    metric_cols = [c for c in df.columns if c.startswith("metric_")]
    assert metric_cols  # at least one metric column
    assert {"t0", "t1", "t2", "num_samples"}.issubset(df.columns)
    # A vectorized run has no per-window walltimes or train_metrics.
    assert not any(c.startswith("train_metric_") or c.startswith("param_") for c in df.columns)
    assert "walltime" not in df.columns


def test_custom_coverage_alpha_stays_vectorized() -> None:
    """A partial-bound coverage level runs through the same fused vmapped scoring.

    Regression for the PR #47 review finding: a custom ``metrics`` mapping used
    to silently fall back to a per-window host loop, so a non-default coverage
    alpha lost the single fused computation. Under the array-metric contract the
    custom mapping is vmapped exactly like the default one.
    """
    duration = 50
    data, cov = _series(duration)
    metrics_50 = {**DEFAULT_METRICS, "coverage": partial(eval_coverage, alpha=0.5)}
    model_default = rw_model
    vec_default = backtest_vectorized(
        random.PRNGKey(5),
        lambda: model_default,
        data,
        cov,
        train_window=TRAIN,
        test_window=TEST,
        guide=AutoNormal(model_default),
        stride=5,
        num_steps=50,
        num_samples=100,
        keep_predictions=True,
    )
    model_50 = rw_model
    vec_50 = backtest_vectorized(
        random.PRNGKey(5),
        lambda: model_50,
        data,
        cov,
        train_window=TRAIN,
        test_window=TEST,
        guide=AutoNormal(model_50),
        stride=5,
        num_steps=50,
        num_samples=100,
        metrics=metrics_50,
        keep_predictions=True,
    )
    # Same key, same fits: everything but the coverage level is identical.
    for name in ("mae", "rmse", "crps"):
        assert jnp.array_equal(vec_50.metrics[name], vec_default.metrics[name])
    # Central quantile intervals are nested, so coverage is monotone in alpha.
    assert bool(jnp.all(vec_50.metrics["coverage"] <= vec_default.metrics["coverage"]))
    # The vectorized values match scoring each window's kept predictions directly.
    assert vec_50.predictions is not None
    for i in range(int(vec_50.t0.shape[0])):
        t1, t2 = int(vec_50.t1[i]), int(vec_50.t2[i])
        expected = eval_coverage(vec_50.predictions[i], data[t1:t2], alpha=0.5)
        assert float(vec_50.metrics["coverage"][i]) == pytest.approx(float(expected))


def test_chunked_metric_raises_actionable_error() -> None:
    """A chunked metric stages values on the host, so it too must be rejected clearly.

    ``eval_crps(..., batch_size=n)`` runs the host-staging chunk loop
    (``_chunked_cell_metric``), which cannot work on a traced window axis. The
    failure must surface as ``VectorizedMetricError`` and not as a raw
    ``AttributeError`` from ``leaf_view`` reaching for ``sharding`` on a
    tracer: that regression is exactly what the tracer passthrough in
    ``leaf_view`` guards.
    """
    duration = TRAIN + TEST + 2
    data, cov = _series(duration)

    model = rw_model
    with pytest.raises(VectorizedMetricError):
        backtest_vectorized(
            random.PRNGKey(0),
            lambda: model,
            data,
            cov,
            train_window=TRAIN,
            test_window=TEST,
            guide=AutoNormal(model),
            num_steps=1,
            num_samples=10,
            metrics={"crps": partial(eval_crps, batch_size=2)},
        )


def _host_float_mae(pred: Array, truth: Array) -> Array:
    """Convert a traced value with ``float(...)`` (``ConcretizationTypeError``)."""
    return jnp.asarray(float(jnp.abs(pred - truth).mean()))


def _host_range_count(pred: Array, truth: Array) -> Array:
    """Convert a traced value with ``range(...)`` (``TracerIntegerConversionError``)."""
    return jnp.asarray(len(range(jnp.sum(truth > 0))), jnp.float32) + 0.0 * pred.mean()


def _host_list_index(pred: Array, truth: Array) -> Array:
    """Index a Python list with a traced value (``TracerIntegerConversionError``)."""
    return jnp.asarray([0.0, 1.0][jnp.argmax(truth.ravel()) % 2]) + 0.0 * pred.mean()


def _host_boolean_mask(pred: Array, truth: Array) -> Array:
    """Index with a traced boolean mask (``NonConcreteBooleanIndexError``)."""
    return pred[pred > truth].mean()


@pytest.mark.parametrize(
    "host_mae",
    [_host_float_mae, _host_range_count, _host_list_index, _host_boolean_mask],
    ids=["float", "range", "list_index", "boolean_mask"],
)
def test_host_metric_raises_actionable_error(host_mae: Callable[[Array, Array], Array]) -> None:
    """A host conversion or a value-dependent shape in a metric raises ``VectorizedMetricError``.

    The ``range`` and ``list_index`` cases raise
    `jax.errors.TracerIntegerConversionError` and the ``boolean_mask`` case raises
    `jax.errors.NonConcreteBooleanIndexError` (a ``JAXIndexError``); neither is a
    ``ConcretizationTypeError``, so each needs its own entry in the ``except`` tuple.
    """
    duration = TRAIN + TEST + 2
    data, cov = _series(duration)

    model = rw_model
    with pytest.raises(VectorizedMetricError):
        backtest_vectorized(
            random.PRNGKey(0),
            lambda: model,
            data,
            cov,
            train_window=TRAIN,
            test_window=TEST,
            guide=AutoNormal(model),
            num_steps=1,
            num_samples=10,
            metrics={"host_mae": host_mae},
        )


def test_vectorized_result_to_dict_does_not_copy_predictions() -> None:
    """``to_dict()`` is a view: the kept predictions are the result's own object (#199)."""
    predictions = jnp.ones((2, 10, TEST, 1))
    result = VectorizedBacktestResult(
        t0=jnp.zeros(2, dtype=jnp.int32),
        t1=jnp.full(2, TRAIN),
        t2=jnp.full(2, TRAIN + TEST),
        num_samples=10,
        losses=jnp.zeros((2, 5)),
        metrics={"mae": jnp.ones(2)},
        predictions=predictions,
    )
    flat = result.to_dict()
    assert set(flat) == {"t0", "t1", "t2", "num_samples", "losses", "metrics", "predictions"}
    assert flat["predictions"] is predictions
    assert flat["losses"] is result.losses
    flat["metrics"]["rmse"] = jnp.zeros(2)
    assert set(result.metrics) == {"mae"}


def test_explicit_optim_is_forwarded_to_svi(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``optim`` is the exact instance every ``SVI`` is constructed with.

    ``backtest_vectorized`` resolves ``resolved_optim = Adam(0.01) if optim is None
    else optim`` and builds each ``SVI(model, guide, resolved_optim, Trace_ELBO())``
    from it. Monkeypatching `numpyro_forecast.evaluate.SVI` with a thin recording
    subclass pins both branches: an explicit ``optim`` is forwarded unchanged, and
    ``optim=None`` falls back to a fresh ``Adam`` instance rather than the one we
    passed in a previous call.
    """
    recorded_optims: list[object] = []

    class _RecordingSVI(SVI):
        """``SVI`` subclass that records its ``optim`` argument, otherwise delegates."""

        def __init__(
            self,
            model: object,
            guide: object,
            optim: object,
            loss: object,
            **static_kwargs: object,
        ) -> None:
            recorded_optims.append(optim)
            super().__init__(model, guide, optim, loss, **static_kwargs)

    monkeypatch.setattr("numpyro_forecast.evaluate.SVI", _RecordingSVI)

    duration = TRAIN + TEST
    data, cov = _series(duration)
    model = rw_model
    expected_windows = (duration - TRAIN - TEST) // 1 + 1
    explicit_optim = numpyro.optim.Adam(0.1)

    result = backtest_vectorized(
        random.PRNGKey(0),
        lambda: model,
        data,
        cov,
        train_window=TRAIN,
        test_window=TEST,
        guide=AutoNormal(model),
        optim=explicit_optim,
        stride=1,
        num_steps=10,
        num_samples=5,
    )
    assert recorded_optims
    assert all(optim is explicit_optim for optim in recorded_optims)
    assert result.t0.shape[0] == expected_windows
    assert bool(jnp.all(jnp.isfinite(result.metrics["crps"])))

    recorded_optims.clear()
    backtest_vectorized(
        random.PRNGKey(0),
        lambda: model,
        data,
        cov,
        train_window=TRAIN,
        test_window=TEST,
        guide=AutoNormal(model),
        optim=None,
        stride=1,
        num_steps=10,
        num_samples=5,
    )
    assert recorded_optims
    assert all(isinstance(optim, numpyro.optim.Adam) for optim in recorded_optims)
    assert all(optim is not explicit_optim for optim in recorded_optims)


def _squeezed_forecast_model(covariates: Array, data: Array | None = None) -> None:
    """A model whose ``"forecast"`` site drops the obs axis: ``(future,)``, not ``(future, 1)``."""
    h = Horizon.from_data(covariates, data)
    loc = numpyro.sample("loc", dist.Normal(0.0, 1.0))
    if h.data is not None:
        numpyro.sample("obs", dist.Normal(loc, 1.0).expand(h.data.shape).to_event(2), obs=h.data)
    numpyro.deterministic("forecast", jnp.full((h.future,), loc))


def test_forecast_shape_mismatch_raises_before_scoring() -> None:
    """A forecast site that does not match the test window raises ``ValueError`` (#196).

    A custom metric such as ``(pred.mean(0) - truth).mean()`` broadcasts a
    ``(test,)`` forecast against a ``(test, 1)`` truth into a ``(test, test)``
    grid without error, so the check runs once on the stacked windows before
    the metrics are vmapped.
    """
    duration = TRAIN + TEST + 2
    data, cov = _series(duration)

    def bias(pred: Array, truth: Array) -> Array:
        return (pred.mean(axis=0) - truth).mean()

    model = _squeezed_forecast_model
    with pytest.raises(ValueError, match=r"'forecast' site has shape \(5,\) per draw"):
        backtest_vectorized(
            random.PRNGKey(0),
            lambda: model,
            data,
            cov,
            train_window=TRAIN,
            test_window=TEST,
            guide=AutoNormal(model),
            num_steps=1,
            num_samples=10,
            metrics={"bias": bias},
        )


def test_forecast_shape_check_is_skipped_without_metrics() -> None:
    """With ``metrics={}`` nothing is scored, so a mismatched forecast site still returns.

    A predictions-only run does not compare the shapes: the shape guard
    exists to protect the metrics, not to validate the model.
    """
    duration = TRAIN + TEST + 2
    data, cov = _series(duration)
    model = _squeezed_forecast_model
    result = backtest_vectorized(
        random.PRNGKey(0),
        lambda: model,
        data,
        cov,
        train_window=TRAIN,
        test_window=TEST,
        guide=AutoNormal(model),
        num_steps=1,
        num_samples=10,
        metrics={},
        keep_predictions=True,
    )
    assert result.predictions is not None
    assert result.predictions.shape == (3, 10, TEST)
    assert result.metrics == {}


def _noise_model(covariates: Array, data: Array | None = None) -> None:
    """A constant level ``mu`` with an unknown noise scale; the forecast is ``mu`` itself."""
    h = Horizon.from_data(covariates, data)
    mu = jnp.asarray(numpyro.sample("mu", dist.Normal(0.0, 10.0)))
    log_sigma = jnp.asarray(numpyro.sample("log_sigma", dist.Normal(0.0, 3.0)))
    with numpyro.plate("time", h.t_obs, dim=-2):
        numpyro.sample("obs", dist.Normal(mu, jnp.exp(log_sigma)), obs=data)
    if h.future:
        numpyro.deterministic("forecast", jnp.full((h.future, 1), mu))


def test_laplace_covariance_comes_from_each_window() -> None:
    """Each window's Laplace covariance is the inverse Hessian of its own loss (#185).

    Window 1 has 4 times the noise of window 0. At the MAP point the Laplace
    posterior sd of ``mu`` is ``sigma_hat / sqrt(n)``, with ``sigma_hat`` the
    (population) sd of the window's training data. A guide set up on window 0
    only evaluates the window-0 Hessian at the window-1 MAP point, which gives a
    wrong (here zero) sd for window 1.
    """
    train, test = 16, 2
    stride = train + test
    z = random.normal(random.PRNGKey(0), (2 * stride, 1))
    z = (z - z.mean()) / z.std()
    data = jnp.where(jnp.arange(2 * stride)[:, None] < stride, z, 4.0 * z)
    cov = jnp.zeros((2 * stride, 0))
    result = backtest_vectorized(
        random.PRNGKey(1),
        lambda: _noise_model,
        data,
        cov,
        train_window=train,
        test_window=test,
        stride=stride,
        guide=AutoLaplaceApproximation(_noise_model),
        optim=numpyro.optim.Adam(0.05),
        num_steps=2_000,
        num_samples=2_000,
        keep_predictions=True,
    )
    assert result.predictions is not None
    for w in range(2):
        t0 = int(result.t0[w])
        expected = data[t0 : t0 + train].std() / jnp.sqrt(train)
        actual = result.predictions[w, :, 0, 0].std()
        assert float(jnp.abs(actual - expected) / expected) < 0.15, (w, actual, expected)


def _bounded_level_model(covariates: Array, data: Array | None = None) -> None:
    """A level whose support is the observed range of the training data."""
    assert data is not None, "the support of the level needs the training data"
    h = Horizon.from_data(covariates, data)
    level = jnp.asarray(numpyro.sample("level", dist.Uniform(data.min(), data.max())))
    sigma = jnp.asarray(numpyro.sample("sigma", dist.LogNormal(-1.0, 0.5)))
    with numpyro.plate("time", h.t_obs, dim=-2):
        numpyro.sample("obs", dist.Normal(level, sigma), obs=data)
    if h.future:
        numpyro.deterministic("forecast", jnp.full((h.future, 1), level))


def _bounded_level_guide_list() -> AutoGuide:
    """An ``AutoGuideList`` with one part per latent of `_bounded_level_model`."""
    guide = AutoGuideList(_bounded_level_model)
    for site in ("level", "sigma"):
        guide.append(AutoNormal(handlers.block(_bounded_level_model, expose=[site])))
    return guide


@pytest.mark.parametrize(
    "make_guide",
    [
        pytest.param(lambda: AutoNormal(_bounded_level_model), id="AutoNormal"),
        pytest.param(_bounded_level_guide_list, id="AutoGuideList"),
    ],
)
def test_data_dependent_support_comes_from_each_window(
    make_guide: Callable[[], AutoGuide],
) -> None:
    """A latent's support transform comes from each window's own data (#185).

    The level is ``Uniform(min(data), max(data))`` and window 1 lies 10 units
    above window 0, so every forecast draw of a window must stay inside that
    window's training range. Each part of an ``AutoGuideList`` caches its own
    setup, so the parts are set up per window too, and a second call with the
    same instance gives the same check.
    """
    train, test = 12, 2
    stride = train + test
    u = random.uniform(random.PRNGKey(0), (2 * stride, 1))
    data = jnp.where(jnp.arange(2 * stride)[:, None] < stride, u, u + 10.0)
    cov = jnp.zeros((2 * stride, 0))
    guide = make_guide()
    for seed in range(2):
        result = backtest_vectorized(
            random.PRNGKey(seed),
            lambda: _bounded_level_model,
            data,
            cov,
            train_window=train,
            test_window=test,
            stride=stride,
            guide=guide,
            num_steps=500,
            num_samples=100,
            keep_predictions=True,
        )
        assert result.predictions is not None
        for w in range(2):
            t0 = int(result.t0[w])
            window = data[t0 : t0 + train]
            draws = result.predictions[w, :, 0, 0]
            assert bool(jnp.all((draws >= window.min()) & (draws <= window.max()))), (seed, w)


def _fitted_autonormal(data: Array, cov: Array) -> AutoGuide:
    guide = AutoNormal(rw_model)
    SVI(rw_model, guide, numpyro.optim.Adam(0.01), Trace_ELBO()).run(
        random.PRNGKey(0), 5, cov, data, progress_bar=False
    )
    return guide


def _list_with_fitted_part(data: Array, cov: Array) -> AutoGuide:
    guide = AutoGuideList(rw_model)
    guide.append(_fitted_autonormal(data, cov))
    return guide


@pytest.mark.parametrize(
    "make_guide",
    [
        pytest.param(_fitted_autonormal, id="fitted-AutoNormal"),
        pytest.param(_list_with_fitted_part, id="AutoGuideList-with-fitted-part"),
    ],
)
def test_guide_already_set_up_is_rejected(
    make_guide: Callable[[Array, Array], AutoGuide],
) -> None:
    """An autoguide set up on other data raises an actionable ``ValueError`` (#185).

    Its cached setup (prototype trace, supports, shapes) belongs to the other
    data, here the full series, so the driver asks for a fresh instance instead
    of failing later with a shape mismatch deep inside NumPyro. A part of an
    ``AutoGuideList`` caches its own setup, so a fitted part is rejected too.
    """
    data, cov = _series(TRAIN + TEST + 4)
    with pytest.raises(ValueError, match="fresh"):
        backtest_vectorized(
            random.PRNGKey(1),
            lambda: rw_model,
            data,
            cov,
            train_window=TRAIN,
            test_window=TEST,
            guide=make_guide(data, cov),
            num_steps=5,
            num_samples=4,
        )


class _LockedRandomWalk:
    """Owns a lock, so it cannot be copied; its bound ``model`` method is the model."""

    def __init__(self) -> None:
        self.lock = threading.Lock()

    def model(self, covariates: Array, data: Array | None = None) -> None:
        """Delegate to ``rw_model``."""
        rw_model(covariates, data)


def test_model_bound_to_an_uncopyable_object_runs() -> None:
    """The per-window guide copies share the model and never copy what it is bound to.

    ``owner.model`` builds a new bound-method object on each access, so the
    model of the guide is not the object ``model_fn`` returns. A deep copy of
    the guide would copy the owner and fail on its lock.
    """
    data, cov = _series(TRAIN + TEST + 2)
    owner = _LockedRandomWalk()
    result = backtest_vectorized(
        random.PRNGKey(0),
        lambda: owner.model,
        data,
        cov,
        train_window=TRAIN,
        test_window=TEST,
        guide=AutoNormal(owner.model),
        num_steps=5,
        num_samples=4,
    )
    assert bool(jnp.all(jnp.isfinite(result.losses)))


def _log_normal_model(covariates: Array, data: Array | None = None) -> None:
    """A log-normal likelihood, which has no finite density at an observed zero."""
    h = Horizon.from_data(covariates, data)
    mu = jnp.asarray(numpyro.sample("mu", dist.Normal(0.0, 1.0)))
    with numpyro.plate("time", h.t_obs, dim=-2):
        numpyro.sample("obs", dist.LogNormal(mu, 1.0), obs=data)
    if h.future:
        numpyro.deterministic("forecast", jnp.full((h.future, 1), jnp.exp(mu)))


def test_invalid_start_on_window_zero_raises() -> None:
    """A start with no finite density raises NumPyro's error before the vmapped fit.

    NumPyro checks the initial point only on concrete values; under ``vmap`` a
    window without a valid start would give NaN losses and metrics silently.
    """
    data, cov = jnp.ones((TRAIN + TEST, 1)).at[0, 0].set(0.0), jnp.zeros((TRAIN + TEST, 0))
    with (
        pytest.warns(UserWarning, match="Out-of-support"),
        pytest.raises(RuntimeError, match="valid initial parameters"),
    ):
        backtest_vectorized(
            random.PRNGKey(0),
            lambda: _log_normal_model,
            data,
            cov,
            train_window=TRAIN,
            test_window=TEST,
            guide=AutoNormal(_log_normal_model),
            num_steps=5,
            num_samples=4,
        )
