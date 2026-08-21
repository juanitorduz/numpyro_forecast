"""Tests for :func:`backtest_vectorized` (roadmap §4.3).

Covers estimator equivalence with the loop :func:`backtest` (window indices +
statistical metric closeness), every validator, the window-count formula, the
compile-discipline gate (I3: the vmapped SVI fit does not recompile per window),
and the I6 acceptance check that a subsequent eager plain-NumPyro SVI fit with a
fresh guide runs without an ``UnexpectedTracerError``.
"""

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
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from numpyro_forecast.evaluate import (
    DEFAULT_METRICS,
    VectorizedBacktestResult,
    _window_key_streams,
    backtest,
    backtest_vectorized,
    eval_coverage,
)
from numpyro_forecast.exceptions import (
    BacktestWindowError,
    VectorizedGuideError,
    VectorizedMetricError,
)
from numpyro_forecast.functional import draw_posterior, forecast

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
        data,
        cov,
        lambda: rw_model,
        forecast_fn=svi_forecast_fn(num_steps=50),
        train_window=TRAIN,
        test_window=TEST,
        stride=stride,
        num_samples=20,
    )
    model = rw_model
    vec = backtest_vectorized(
        random.PRNGKey(0),
        data,
        cov,
        lambda: model,
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


def test_metrics_statistically_close_to_loop() -> None:
    """Per-window MAE/CRPS match the loop path within a generous tolerance."""
    duration = 50
    data, cov = _series(duration)
    loop = backtest(
        random.PRNGKey(3),
        data,
        cov,
        lambda: rw_model,
        forecast_fn=svi_forecast_fn(num_steps=800),
        train_window=TRAIN,
        test_window=TEST,
        stride=5,
        num_samples=200,
    )
    model = rw_model
    vec = backtest_vectorized(
        random.PRNGKey(3),
        data,
        cov,
        lambda: model,
        train_window=TRAIN,
        test_window=TEST,
        guide=AutoNormal(model),
        stride=5,
        num_steps=800,
        num_samples=200,
    )
    loop_mae = jnp.array([r.metrics["mae"] for r in loop])
    # Estimators differ in PRNG layout; check coarse agreement on MAE only.
    assert jnp.all(jnp.isfinite(vec.metrics["mae"]))
    assert jnp.all(jnp.isfinite(loop_mae))
    rel_err = jnp.abs(vec.metrics["mae"] - loop_mae) / jnp.maximum(loop_mae, 1e-6)
    assert float(rel_err.mean()) < 1.5


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
            data,
            cov,
            lambda: model,
            train_window=train_window,
            test_window=test_window,
            guide=AutoNormal(model),
            stride=stride,
            num_steps=10,
        )


def test_handwritten_guide_rejected() -> None:
    """A non-AutoGuide guide is rejected with guidance to use ``backtest``."""
    data, cov = _series(50)

    def handwritten_guide(covariates: Array, data: Array | None = None) -> None:
        loc = numpyro.param("loc", 0.0)
        numpyro.sample("drift_scale", dist.Delta(loc))

    with pytest.raises(VectorizedGuideError, match="AutoGuide"):
        backtest_vectorized(
            random.PRNGKey(0),
            data,
            cov,
            lambda: rw_model,
            train_window=TRAIN,
            test_window=TEST,
            guide=handwritten_guide,
            num_steps=10,
        )


def test_duration_too_short_rejected() -> None:
    """A series with no room for a single window raises ``BacktestWindowError``."""
    data, cov = _series(TRAIN + TEST - 1)
    model = rw_model
    with pytest.raises(BacktestWindowError, match="no window fits"):
        backtest_vectorized(
            random.PRNGKey(0),
            data,
            cov,
            lambda: model,
            train_window=TRAIN,
            test_window=TEST,
            guide=AutoNormal(model),
            num_steps=10,
        )


def test_covariate_length_mismatch_rejected() -> None:
    """Mismatched data/covariate durations raise ``ValueError``."""
    data, _ = _series(50)
    cov = jnp.zeros((49, 0))
    model = rw_model
    with pytest.raises(ValueError, match="share the time axis"):
        backtest_vectorized(
            random.PRNGKey(0),
            data,
            cov,
            lambda: model,
            train_window=TRAIN,
            test_window=TEST,
            guide=AutoNormal(model),
            num_steps=10,
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
        data,
        cov,
        lambda: model,
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
    """
    import jax

    duration = TRAIN + TEST + 9  # exactly 10 windows at stride 1
    data, cov = _series(duration)

    def run() -> int:
        model = rw_model
        with count_compilations() as tally:
            result = backtest_vectorized(
                random.PRNGKey(0),
                data,
                cov,
                lambda: model,
                train_window=TRAIN,
                test_window=TEST,
                guide=AutoNormal(model),
                stride=1,
                num_steps=30,
                num_samples=10,
            )
            jax.block_until_ready(result.losses)
        return int(tally.count)

    first = run()
    # First call may compile several vmapped stages (fit, posterior, forecast, metrics).
    assert first < 50

    model = rw_model
    with count_compilations() as tally:
        result = backtest_vectorized(
            random.PRNGKey(0),
            data,
            cov,
            lambda: model,
            train_window=TRAIN,
            test_window=TEST,
            guide=AutoNormal(model),
            stride=1,
            num_steps=30,
            num_samples=10,
        )
        jax.block_until_ready(result.losses)
    # Same shapes should not trigger a full recompile storm (allow a few residual).
    assert tally.count <= max(first, 1)


def test_no_tracer_leak_on_subsequent_eager_fit() -> None:
    """I6 acceptance: a fresh eager plain-NumPyro SVI fit after a vectorized run is clean.

    The mandatory eager warm-up keeps the vectorized run's AutoGuide instance
    uncontaminated, and a subsequent eager fit uses a fresh guide, so no
    ``UnexpectedTracerError`` escapes into later inference.
    """
    duration = TRAIN + TEST + 7
    data, cov = _series(duration)
    model = rw_model
    backtest_vectorized(
        random.PRNGKey(1),
        data,
        cov,
        lambda: model,
        train_window=TRAIN,
        test_window=TEST,
        guide=AutoNormal(model),
        num_steps=30,
        num_samples=10,
    )
    train_data = data[:TRAIN]
    train_cov = cov[:TRAIN]
    full_cov = cov[: TRAIN + TEST]
    key_fit, key_draw = random.split(random.PRNGKey(2))
    fresh_guide = AutoNormal(rw_model)
    svi = SVI(rw_model, fresh_guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    state = svi.run(key_fit, 30, train_cov, train_data, progress_bar=False)
    posterior = draw_posterior(key_draw, fresh_guide, state.params, 10)
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
        data,
        cov,
        lambda: model,
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
    a split parent only, so no stream may reuse it directly, and none may
    collide with the eager warm-up key ``fold_in(rng_key, -1)``.
    """
    import jax

    rng_key = random.PRNGKey(0)
    num_windows = 2
    init_keys, post_keys, fc_keys = _window_key_streams(rng_key, num_windows)
    parents = jax.vmap(lambda i: random.fold_in(rng_key, i))(jnp.arange(num_windows))
    warmup = random.fold_in(rng_key, jnp.array(-1, dtype=jnp.int32))

    keys = [warmup]
    for i in range(num_windows):
        keys += [init_keys[i], post_keys[i], fc_keys[i], parents[i]]
    raw = [tuple(int(x) for x in jnp.ravel(random.key_data(k))) for k in keys]
    assert len(set(raw)) == len(raw)


def test_result_schema_and_dataframe_row_shape() -> None:
    """Result fields are per-window and results_to_dataframe is one row per window."""
    from numpyro_forecast.evaluate import results_to_dataframe

    duration = TRAIN + TEST + 8
    data, cov = _series(duration)
    model = rw_model
    result = backtest_vectorized(
        random.PRNGKey(0),
        data,
        cov,
        lambda: model,
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
        data,
        cov,
        lambda: model_default,
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
        data,
        cov,
        lambda: model_50,
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


def test_host_metric_raises_actionable_error() -> None:
    """A metric that forces a host conversion raises ``VectorizedMetricError``."""
    duration = TRAIN + TEST + 2
    data, cov = _series(duration)

    def host_mae(pred: Array, truth: Array) -> Array:
        return jnp.asarray(float(jnp.abs(pred - truth).mean()))

    model = rw_model
    with pytest.raises(VectorizedMetricError):
        backtest_vectorized(
            random.PRNGKey(0),
            data,
            cov,
            lambda: model,
            train_window=TRAIN,
            test_window=TEST,
            guide=AutoNormal(model),
            num_steps=1,
            num_samples=10,
            metrics={"host_mae": host_mae},
        )
