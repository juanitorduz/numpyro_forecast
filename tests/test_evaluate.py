"""Tests for backtesting and evaluation metrics."""

from functools import partial
from typing import cast

import jax.numpy as jnp
import pytest
from conftest import RandomWalkModel
from jax import Array, random

from numpyro_forecast.evaluate import (
    DEFAULT_METRICS,
    BacktestResult,
    _iter_windows,
    _resolve_options,
    _run_window,
    _scalar_params,
    _slice_window,
    _timed,
    backtest,
    eval_coverage,
    eval_crps,
    eval_mae,
    eval_rmse,
    evaluate_forecast,
)
from numpyro_forecast.forecaster import HMCForecaster
from numpyro_forecast.typing import ForecasterFactory


def test_eval_mae_uses_median() -> None:
    pred = jnp.array([1.0, 2.0, 9.0]).reshape(3, 1)  # median 2
    truth = jnp.array([0.0])
    assert eval_mae(pred, truth) == 2.0


def test_eval_rmse_uses_mean() -> None:
    pred = jnp.array([0.0, 4.0]).reshape(2, 1)  # mean 2
    truth = jnp.array([0.0])
    assert eval_rmse(pred, truth) == 2.0


def test_eval_crps_returns_float() -> None:
    pred = random.normal(random.PRNGKey(0), (50, 4))
    truth = random.normal(random.PRNGKey(1), (4,))
    value = eval_crps(pred, truth)
    assert isinstance(value, float)
    assert value >= 0.0


def test_eval_coverage_perfect_and_zero() -> None:
    # Samples spread symmetrically around 0; truth at 0 is inside any central band.
    pred = jnp.linspace(-1.0, 1.0, 101).reshape(101, 1)
    assert eval_coverage(pred, jnp.array([0.0])) == 1.0
    # Truth far outside the sample support falls outside the band.
    assert eval_coverage(pred, jnp.array([100.0])) == 0.0


def test_eval_coverage_returns_float() -> None:
    pred = random.normal(random.PRNGKey(0), (200, 4))
    truth = random.normal(random.PRNGKey(1), (4,))
    value = eval_coverage(pred, truth, alpha=0.8)
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_default_metrics_keys() -> None:
    assert set(DEFAULT_METRICS) == {"mae", "rmse", "crps", "coverage"}


def test_evaluate_forecast_matches_individual_metrics() -> None:
    pred = random.normal(random.PRNGKey(0), (200, 4))
    truth = random.normal(random.PRNGKey(1), (4,))
    report = evaluate_forecast(pred, truth)
    assert set(report) == set(DEFAULT_METRICS)
    assert report["mae"] == eval_mae(pred, truth)
    assert report["rmse"] == eval_rmse(pred, truth)
    assert report["crps"] == eval_crps(pred, truth)
    assert report["coverage"] == eval_coverage(pred, truth)


def test_evaluate_forecast_honors_custom_metrics() -> None:
    pred = random.normal(random.PRNGKey(0), (50, 3))
    truth = random.normal(random.PRNGKey(1), (3,))
    report = evaluate_forecast(pred, truth, metrics={"mae": eval_mae})
    assert set(report) == {"mae"}
    assert report["mae"] == eval_mae(pred, truth)


def test_evaluate_forecast_honors_partial_coverage_metric() -> None:
    # Samples symmetric on [-1, 1]; a truth at 0.85 sits inside the wide 0.9
    # central band but outside the narrower 0.8 band, so coverage must differ.
    # The 0.8 level is supplied via a partial-bound metric in the mapping.
    pred = jnp.linspace(-1.0, 1.0, 101).reshape(101, 1)
    truth = jnp.array([0.85])
    metrics_80 = {**DEFAULT_METRICS, "coverage": partial(eval_coverage, alpha=0.8)}
    report_80 = evaluate_forecast(pred, truth, metrics=metrics_80)
    report_90 = evaluate_forecast(pred, truth)  # default coverage at 0.9
    assert report_80["coverage"] == eval_coverage(pred, truth, alpha=0.8)
    assert report_90["coverage"] == eval_coverage(pred, truth, alpha=0.9)
    assert report_80["coverage"] != report_90["coverage"]


def test_evaluate_forecast_multidim_batch() -> None:
    # Exercises the ``*batch`` part of the ``(sample, *batch)`` annotation.
    pred = random.normal(random.PRNGKey(0), (200, 5, 2))  # (sample, time, obs)
    truth = random.normal(random.PRNGKey(1), (5, 2))
    report = evaluate_forecast(pred, truth)
    assert set(report) == set(DEFAULT_METRICS)
    assert all(isinstance(value, float) for value in report.values())


def test_backtest_expanding_window(rng_key: Array) -> None:
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    results = backtest(
        rng_key,
        data,
        covariates,
        RandomWalkModel,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=20,
        forecaster_options={"num_steps": 30},
    )
    # Windows at t1 in {12, 16, 20} (stop = 24 - 4 + 1 = 21).
    assert [r.t1 for r in results] == [12, 16, 20]
    for r in results:
        assert isinstance(r, BacktestResult)
        assert r.t0 == 0  # expanding window
        assert set(r.metrics) == {"mae", "rmse", "crps", "coverage"}
        assert r.train_walltime >= 0.0
        # AutoNormal exposes per-site variational params (e.g. *_auto_loc).
        assert any("drift_scale" in name for name in r.params)


def test_backtest_honors_partial_coverage_metric(rng_key: Array) -> None:
    # A metric-specific parameter (coverage's alpha) is supplied through the
    # ``metrics`` mapping via ``partial``, so ``backtest`` needs no dedicated
    # parameter. With the same rng key both runs draw identical forecast
    # samples, so a wider central band can only cover more: per-window coverage
    # is monotonically non-decreasing in alpha.
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    run = partial(
        backtest,
        rng_key,
        data,
        covariates,
        RandomWalkModel,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=50,
        forecaster_options={"num_steps": 30},
    )
    narrow = run(metrics={**DEFAULT_METRICS, "coverage": partial(eval_coverage, alpha=0.5)})
    wide = run()  # default coverage at 0.9
    assert narrow and len(narrow) == len(wide)
    for r_narrow, r_wide in zip(narrow, wide, strict=True):
        assert set(r_wide.metrics) == {"mae", "rmse", "crps", "coverage"}
        assert r_wide.metrics["coverage"] >= r_narrow.metrics["coverage"]


def test_backtest_result_to_dict() -> None:
    result = BacktestResult(
        t0=0,
        t1=10,
        t2=14,
        num_samples=20,
        train_walltime=0.5,
        test_walltime=0.25,
        metrics={"mae": 1.0},
        params={"sigma": 0.3},
    )
    flat = result.to_dict()
    assert flat["t0"] == 0
    assert flat["t1"] == 10
    assert flat["metrics"] == {"mae": 1.0}
    assert flat["params"] == {"sigma": 0.3}
    assert flat["train_metrics"] == {}
    assert flat["prediction"] is None
    assert set(flat) == {
        "t0",
        "t1",
        "t2",
        "num_samples",
        "train_walltime",
        "test_walltime",
        "metrics",
        "params",
        "train_metrics",
        "prediction",
    }


def test_backtest_hmc_has_no_scalar_params(rng_key: Array) -> None:
    # HMCForecaster has no ``params`` mapping, so ``_scalar_params`` returns {}.
    data = jnp.cumsum(0.1 * random.normal(rng_key, (20, 1)), axis=-2)
    covariates = jnp.zeros((20, 0))
    results = backtest(
        rng_key,
        data,
        covariates,
        RandomWalkModel,
        forecaster_fn=HMCForecaster,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=10,
        forecaster_options={"num_warmup": 10, "num_samples": 10},
    )
    assert results
    for r in results:
        assert r.params == {}


def test_backtest_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="share the time axis length"):
        backtest(random.PRNGKey(0), jnp.zeros((20, 1)), jnp.zeros((18, 0)), RandomWalkModel)


def test_backtest_callable_options_and_transform(rng_key: Array) -> None:
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    seen_windows: list[tuple[int, int, int]] = []

    def options_for(t0: int, t1: int, t2: int) -> dict[str, int]:
        seen_windows.append((t0, t1, t2))
        return {"num_steps": 30}

    transform_calls = {"count": 0}

    def transform(pred: Array, truth: Array) -> tuple[Array, Array]:
        transform_calls["count"] += 1
        return jnp.exp(pred), jnp.exp(truth)

    results = backtest(
        rng_key,
        data,
        covariates,
        RandomWalkModel,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=20,
        forecaster_options=options_for,
        transform=transform,
    )
    assert len(results) == 3
    assert seen_windows == [(0, 12, 16), (0, 16, 20), (0, 20, 24)]
    assert transform_calls["count"] == 3


def test_backtest_defaults_leave_train_metrics_and_prediction_empty(rng_key: Array) -> None:
    # Pyro-faithful defaults: no in-sample scoring, no retained predictions.
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    results = backtest(
        rng_key,
        data,
        covariates,
        RandomWalkModel,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=20,
        forecaster_options={"num_steps": 30},
    )
    assert results
    for r in results:
        assert r.train_metrics == {}
        assert r.prediction is None


def test_backtest_eval_train_populates_train_metrics(rng_key: Array) -> None:
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    results = backtest(
        rng_key,
        data,
        covariates,
        RandomWalkModel,
        metrics={"crps": eval_crps},
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=20,
        forecaster_options={"num_steps": 30},
        eval_train=True,
    )
    assert results
    for r in results:
        assert set(r.train_metrics) == set(r.metrics) == {"crps"}
        assert isinstance(r.train_metrics["crps"], float)


def test_backtest_keep_predictions_stores_oos_samples(rng_key: Array) -> None:
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    results = backtest(
        rng_key,
        data,
        covariates,
        RandomWalkModel,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=20,
        forecaster_options={"num_steps": 30},
        keep_predictions=True,
    )
    assert results
    for r in results:
        assert r.prediction is not None
        assert r.prediction.shape == (20, 4, 1)


def test_backtest_eval_train_applies_transform_twice_per_window(rng_key: Array) -> None:
    # With eval_train the same transform is applied to the OOS pair and the
    # in-sample pair, so it runs twice per window.
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    transform_calls = {"count": 0}

    def transform(pred: Array, truth: Array) -> tuple[Array, Array]:
        transform_calls["count"] += 1
        return jnp.exp(pred), jnp.exp(truth)

    results = backtest(
        rng_key,
        data,
        covariates,
        RandomWalkModel,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=20,
        forecaster_options={"num_steps": 30},
        transform=transform,
        eval_train=True,
    )
    assert len(results) == 3
    assert transform_calls["count"] == 6


def test_backtest_eval_train_requires_predict_in_sample(rng_key: Array) -> None:
    # A custom forecaster without predict_in_sample cannot be scored in-sample.
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    with pytest.raises(TypeError, match="predict_in_sample"):
        backtest(
            rng_key,
            data,
            covariates,
            RandomWalkModel,
            forecaster_fn=cast("ForecasterFactory", lambda *args, **kwargs: _FakeForecaster()),
            test_window=4,
            min_train_window=12,
            stride=4,
            num_samples=10,
            eval_train=True,
        )


# --- unit tests for the private backtest sub-components ------------------------


def test_iter_windows_expanding() -> None:
    # train_window=None -> t0 stays 0 and the window expands from the start.
    windows = list(
        _iter_windows(
            24,
            train_window=None,
            min_train_window=12,
            test_window=4,
            min_test_window=1,
            stride=4,
        )
    )
    assert windows == [(0, 12, 16), (0, 16, 20), (0, 20, 24)]


def test_iter_windows_fixed_train_window_rolls() -> None:
    # A fixed train_window makes t0 track t1 (rolling, not expanding).
    windows = list(
        _iter_windows(
            24,
            train_window=6,
            min_train_window=1,
            test_window=4,
            min_test_window=1,
            stride=4,
        )
    )
    assert windows == [(0, 6, 10), (4, 10, 14), (8, 14, 18), (12, 18, 22)]
    assert all(t1 - t0 == 6 for t0, t1, _ in windows)


def test_iter_windows_test_window_none_forecasts_to_end() -> None:
    # test_window=None -> every window forecasts to the end of the series.
    windows = list(
        _iter_windows(
            10,
            train_window=None,
            min_train_window=8,
            test_window=None,
            min_test_window=1,
            stride=1,
        )
    )
    assert windows == [(0, 8, 10), (0, 9, 10)]
    assert all(t2 == 10 for _, _, t2 in windows)


def test_iter_windows_default_stride_steps_by_one() -> None:
    windows = list(
        _iter_windows(
            8,
            train_window=None,
            min_train_window=4,
            test_window=2,
            min_test_window=1,
            stride=1,
        )
    )
    assert [t1 for _, t1, _ in windows] == [4, 5, 6]


def test_resolve_options_none_is_empty() -> None:
    assert _resolve_options(None, 0, 1, 2) == {}


def test_resolve_options_passes_mapping_through() -> None:
    opts = {"num_steps": 30}
    assert _resolve_options(opts, 0, 1, 2) is opts


def test_resolve_options_invokes_callable_with_window() -> None:
    seen: list[tuple[int, int, int]] = []

    def options_for(t0: int, t1: int, t2: int) -> dict[str, int]:
        seen.append((t0, t1, t2))
        return {"num_steps": 7}

    assert _resolve_options(options_for, 3, 5, 9) == {"num_steps": 7}
    assert seen == [(3, 5, 9)]


def test_slice_window_returns_train_test_truth() -> None:
    data = jnp.arange(10, dtype=jnp.float32).reshape(10, 1)
    covariates = jnp.arange(20, dtype=jnp.float32).reshape(10, 2)
    train_data, train_covariates, test_covariates, truth = _slice_window(data, covariates, 2, 6, 8)
    assert jnp.array_equal(train_data, data[2:6])
    assert jnp.array_equal(train_covariates, covariates[2:6])
    assert jnp.array_equal(test_covariates, covariates[2:8])
    assert jnp.array_equal(truth, data[6:8])


def test_timed_returns_result_and_nonnegative_seconds() -> None:
    result, seconds = _timed(lambda: 42)
    assert result == 42
    assert isinstance(seconds, float)
    assert seconds >= 0.0


def test_scalar_params_keeps_only_scalars() -> None:
    class _Fitted:
        params = {"sigma": jnp.array(0.5), "loc_vec": jnp.arange(3)}

    assert _scalar_params(_Fitted()) == {"sigma": 0.5}


def test_scalar_params_without_params_is_empty() -> None:
    assert _scalar_params(object()) == {}


def test_scalar_params_non_mapping_is_empty() -> None:
    class _Fitted:
        params = "not a mapping"

    assert _scalar_params(_Fitted()) == {}


class _FakeForecaster:
    """Deterministic stand-in forecaster for ``_run_window`` unit tests."""

    def __init__(self) -> None:
        self.params = {"sigma": jnp.array(0.3), "level_vec": jnp.arange(4)}

    def __call__(
        self,
        rng_key: Array,
        data: Array,
        covariates: Array,
        num_samples: int,
        *,
        batch_size: int | None = None,
    ) -> Array:
        # Forecast horizon is the suffix of ``covariates`` beyond ``data``.
        horizon = covariates.shape[-2] - data.shape[-2]
        return jnp.ones((num_samples, horizon, data.shape[-1]))


def test_run_window_builds_result_and_applies_transform() -> None:
    data = jnp.arange(8, dtype=jnp.float32).reshape(8, 1)
    covariates = jnp.zeros((8, 0))
    transform_calls = {"count": 0}

    def transform(pred: Array, truth: Array) -> tuple[Array, Array]:
        transform_calls["count"] += 1
        return 2.0 * pred, 2.0 * truth

    result = _run_window(
        random.PRNGKey(0),
        2,
        6,
        8,
        data=data,
        covariates=covariates,
        model_fn=RandomWalkModel,
        forecaster_fn=cast("ForecasterFactory", lambda *args, **kwargs: _FakeForecaster()),
        options={},
        num_samples=16,
        batch_size=None,
        metrics=DEFAULT_METRICS,
        transform=transform,
        eval_train=False,
        keep_predictions=False,
    )
    assert isinstance(result, BacktestResult)
    assert (result.t0, result.t1, result.t2) == (2, 6, 8)
    assert result.num_samples == 16
    assert set(result.metrics) == {"mae", "rmse", "crps", "coverage"}
    assert result.params == {"sigma": pytest.approx(0.3)}  # the vector param is dropped
    assert result.train_walltime >= 0.0
    assert result.test_walltime >= 0.0
    assert result.train_metrics == {}
    assert result.prediction is None
    assert transform_calls["count"] == 1
