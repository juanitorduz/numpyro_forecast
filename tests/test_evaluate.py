"""Tests for backtesting and evaluation metrics."""

from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import commit_host, rw_model_factory, svi_forecast_fn, svi_in_sample_fn
from jax import Array, random

from numpyro_forecast.evaluate import (
    DEFAULT_METRICS,
    BacktestResult,
    _expanding_windows,
    _iter_windows,
    _resolve_window_type,
    _rolling_windows,
    _run_window,
    _slice_window,
    _timed,
    backtest,
    eval_coverage,
    eval_crps,
    eval_mae,
    eval_rmse,
    evaluate_forecast,
)
from numpyro_forecast.exceptions import DeviceMemoryError
from numpyro_forecast.typing import ForecastFn


def _canned_forecast_fn(
    rng_key: Array,
    model: object,
    train_data: Array,
    train_covariates: Array,
    test_covariates: Array,
    num_samples: int,
    *,
    batch_size: int | None = None,
) -> Array:
    """An rng-sensitive, shape-correct stand-in forecast with no real inference.

    Deterministic given ``rng_key`` (so identical keys give identical draws, the
    property the ``eval_train``/coverage tests below rely on) but does no actual
    model fitting; used wherever a test only needs a *working* closure, not a
    faithfully fitted one.
    """
    del model  # unused: the canned closure ignores the model entirely
    horizon = test_covariates.shape[-2] - train_data.shape[-2]
    return train_data.mean() + random.normal(rng_key, (num_samples, horizon, train_data.shape[-1]))


def _canned_in_sample_fn(
    rng_key: Array,
    model: object,
    train_data: Array,
    train_covariates: Array,
    num_samples: int,
    *,
    batch_size: int | None = None,
) -> Array:
    """The in-sample counterpart of :func:`_canned_forecast_fn`."""
    del model
    return train_data.mean() + random.normal(rng_key, (num_samples, *train_data.shape))


def _spy_closure(calls: list[dict[str, Any]]) -> ForecastFn:
    """A closure recording every call's ``batch_size`` kwarg, returning canned draws."""

    def forecast_fn(
        rng_key: Array,
        model: object,
        train_data: Array,
        train_covariates: Array,
        test_covariates: Array,
        num_samples: int,
        *,
        batch_size: int | None = None,
    ) -> Array:
        calls.append({"batch_size": batch_size})
        return _canned_forecast_fn(
            rng_key,
            model,
            train_data,
            train_covariates,
            test_covariates,
            num_samples,
            batch_size=batch_size,
        )

    return forecast_fn


def _never_called(*_args: object, **_kwargs: object) -> Array:
    """A closure that fails the test if ``backtest`` ever invokes it."""
    pytest.fail("this closure should never have been called")


def test_timed_returns_result_and_nonnegative_time() -> None:
    result, seconds = _timed(lambda: jnp.ones(3) * 2.0)
    assert bool(jnp.all(result == 2.0))
    assert seconds >= 0.0


@pytest.mark.slow
def test_walltime_includes_compute() -> None:
    # A blocked forecast timing must exceed a trivially small threshold: it
    # includes real compile+compute, not just async dispatch.
    t = jnp.linspace(0, 4 * jnp.pi, 60)
    data = (jnp.sin(t) + 0.1 * random.normal(random.PRNGKey(0), (60,)))[:, None]
    covariates = jnp.zeros((60, 0))

    results = backtest(
        random.PRNGKey(1),
        data,
        covariates,
        rw_model_factory,
        forecast_fn=svi_forecast_fn(num_steps=100),
        train_window=40,
        test_window=5,
        num_samples=50,
    )
    assert results
    assert all(r.walltime > 0.0 for r in results)


def test_eval_mae_uses_median() -> None:
    pred = jnp.array([1.0, 2.0, 9.0]).reshape(3, 1)  # median 2
    truth = jnp.array([0.0])
    assert eval_mae(pred, truth) == 2.0


def test_eval_rmse_uses_mean() -> None:
    pred = jnp.array([0.0, 4.0]).reshape(2, 1)  # mean 2
    truth = jnp.array([0.0])
    assert eval_rmse(pred, truth) == 2.0


def test_eval_crps_returns_scalar_array() -> None:
    pred = random.normal(random.PRNGKey(0), (50, 4))
    truth = random.normal(random.PRNGKey(1), (4,))
    value = eval_crps(pred, truth)
    assert value.shape == ()
    assert float(value) >= 0.0


def test_eval_coverage_perfect_and_zero() -> None:
    # Samples spread symmetrically around 0; truth at 0 is inside any central band.
    pred = jnp.linspace(-1.0, 1.0, 101).reshape(101, 1)
    assert eval_coverage(pred, jnp.array([0.0])) == 1.0
    # Truth far outside the sample support falls outside the band.
    assert eval_coverage(pred, jnp.array([100.0])) == 0.0


def test_eval_coverage_returns_scalar_array() -> None:
    pred = random.normal(random.PRNGKey(0), (200, 4))
    truth = random.normal(random.PRNGKey(1), (4,))
    value = eval_coverage(pred, truth, alpha=0.8)
    assert value.shape == ()
    assert 0.0 <= float(value) <= 1.0


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_eval_coverage_rejects_out_of_range_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match=r"alpha must be in"):
        eval_coverage(jnp.zeros((5, 2)), jnp.zeros((2,)), alpha=alpha)


def test_default_metrics_keys() -> None:
    assert set(DEFAULT_METRICS) == {"mae", "rmse", "crps", "coverage"}


def _metric_inputs(n_cells: int = 30) -> tuple[Array, Array]:
    pred = random.normal(random.PRNGKey(0), (40, 3, n_cells // 3))
    truth = random.normal(random.PRNGKey(1), (3, n_cells // 3))
    return pred, truth


# --- Host-committed inputs (device="host" draws mixed with a device-resident truth,
# or vice versa) -------------------------------------------------------------------


def _host(x: Array) -> Array:
    """Commit ``x`` to pinned host memory, the ``device="host"`` fallback target."""
    return commit_host(x, "pinned")


@pytest.mark.parametrize(
    ("commit_pred", "commit_truth"),
    [(True, False), (False, True), (True, True)],
    ids=["host_pred_device_truth", "device_pred_host_truth", "both_host"],
)
def test_unchunked_metrics_accept_host_committed_inputs(
    commit_pred: bool, commit_truth: bool
) -> None:
    """Every unchunked metric must accept any mix of host-committed/device-resident inputs.

    Mixing a pinned ``jax.Array`` (the ``device="host"`` fallback) with a
    device-resident one inside a fused ``jnp``/jitted kernel used to raise
    (``memory_space of all inputs ... must be the same``); each metric now
    moves a host-resident operand to device memory first, so any mix produces
    the same value as the fully device-resident call.
    """
    pred, truth = _metric_inputs()
    pred_in = _host(pred) if commit_pred else pred
    truth_in = _host(truth) if commit_truth else truth

    np.testing.assert_allclose(eval_mae(pred_in, truth_in), eval_mae(pred, truth))
    np.testing.assert_allclose(eval_rmse(pred_in, truth_in), eval_rmse(pred, truth))
    np.testing.assert_allclose(eval_crps(pred_in, truth_in), eval_crps(pred, truth))
    np.testing.assert_allclose(eval_coverage(pred_in, truth_in), eval_coverage(pred, truth))

    expected = evaluate_forecast(pred, truth)
    got = evaluate_forecast(pred_in, truth_in)
    assert got.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(got[name], expected[name])


@pytest.mark.parametrize(
    ("commit_pred", "commit_truth"),
    [(True, False), (False, True), (True, True)],
    ids=["host_pred_device_truth", "device_pred_host_truth", "both_host"],
)
def test_chunked_metrics_accept_host_committed_inputs(
    commit_pred: bool, commit_truth: bool
) -> None:
    """The chunked (``batch_size``-bounded) path must accept the same input mixes.

    :func:`~numpyro_forecast.evaluate._chunked_cell_metric` slices ``pred``
    and ``truth`` into cell blocks through
    :func:`~numpyro_forecast._offload._leaf_view` rather than the
    fused kernel used by the unchunked path, so it needs its own coverage: any
    mix of host-committed/device-resident inputs must reproduce the
    all-device, unchunked result.
    """
    pred, truth = _metric_inputs()
    pred_in = _host(pred) if commit_pred else pred
    truth_in = _host(truth) if commit_truth else truth

    # rtol matches test_eval_crps_chunked_matches_single_pass: chunking only
    # changes the summation order of the final mean (f32 rounding), regardless
    # of which operand is host-committed.
    np.testing.assert_allclose(
        eval_crps(pred_in, truth_in, batch_size=7), eval_crps(pred, truth), rtol=1e-6
    )
    np.testing.assert_allclose(
        eval_coverage(pred_in, truth_in, batch_size=7),
        eval_coverage(pred, truth),
        rtol=1e-6,
    )


def test_metrics_accept_cpu_committed_inputs_under_accelerator_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU-committed ``pred`` (the primary ``device="host"`` result) scores unchanged.

    On a CPU-only machine the move in ``_device_view`` is a same-device no-op,
    so this pins the branch structurally: with the default backend reported as
    an accelerator, the metric kernels must still accept the operand and
    reproduce the plain value (both fused and chunked paths).
    """
    pred, truth = _metric_inputs()
    pred_in = commit_host(pred, "cpu")
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    np.testing.assert_allclose(eval_mae(pred_in, truth), eval_mae(pred, truth))
    np.testing.assert_allclose(eval_crps(pred_in, truth), eval_crps(pred, truth))
    np.testing.assert_allclose(
        eval_crps(pred_in, truth, batch_size=7), eval_crps(pred, truth), rtol=1e-6
    )


# --- Chunked cell evaluation (memory-bounded scoring on wide panels) --------------


def test_eval_crps_chunked_matches_single_pass() -> None:
    # A non-divisor batch size exercises the wrapped final block; chunking only
    # changes the summation order of the final mean.
    pred, truth = _metric_inputs()
    single = float(eval_crps(pred, truth))
    chunked = float(eval_crps(pred, truth, batch_size=7))
    assert np.isclose(single, chunked, rtol=1e-6)


def test_eval_coverage_chunked_preserves_the_count() -> None:
    # Coverage counts exact 0/1 indicators, so chunking recovers the identical
    # count; only the precision of the final division differs (f32 vs f64 mean).
    pred, truth = _metric_inputs()
    single = float(eval_coverage(pred, truth, alpha=0.8))
    chunked = float(eval_coverage(pred, truth, alpha=0.8, batch_size=7))
    n_cells = truth.size
    assert round(single * n_cells) == round(chunked * n_cells)
    assert np.isclose(single, chunked, rtol=1e-6)


def test_eval_metrics_batch_ge_cells_is_passthrough() -> None:
    # At or above the cell count the single-pass kernel runs: bitwise equal.
    pred, truth = _metric_inputs()
    assert float(eval_crps(pred, truth, batch_size=64)) == float(eval_crps(pred, truth))
    assert float(eval_coverage(pred, truth, batch_size=64)) == float(eval_coverage(pred, truth))


def test_eval_metrics_accept_numpy_inputs() -> None:
    pred, truth = _metric_inputs()
    pred_np, truth_np = np.asarray(pred), np.asarray(truth)
    assert np.isclose(
        float(eval_crps(pred_np, truth_np, batch_size=7)),
        float(eval_crps(pred, truth, batch_size=7)),
        rtol=1e-6,
    )
    assert float(eval_coverage(pred_np, truth_np, batch_size=7)) == float(
        eval_coverage(pred, truth, batch_size=7)
    )


def test_eval_crps_chunked_single_compile(count_compilations) -> None:
    """Wrapped fixed-size cell blocks keep the chunked metric at one compiled kernel."""
    pred, truth = _metric_inputs()
    eval_crps(pred, truth, batch_size=7)  # warm-up

    with count_compilations() as tally:
        eval_crps(pred, truth, batch_size=7)
    assert tally.count == 0


def test_eval_crps_rejects_non_positive_batch_size() -> None:
    pred, truth = _metric_inputs()
    with pytest.raises(ValueError, match="batch_size must be positive"):
        eval_crps(pred, truth, batch_size=0)


def test_chunked_metric_oom_reports_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpyro_forecast.evaluate as evaluate_mod

    def raise_oom(pred: Array, truth: Array) -> Array:
        msg = "RESOURCE_EXHAUSTED: Out of memory while trying to allocate 3800000000 bytes."
        raise RuntimeError(msg)

    monkeypatch.setattr(evaluate_mod, "_crps_cells", raise_oom)
    pred, truth = _metric_inputs()
    with pytest.raises(DeviceMemoryError, match=r"metric evaluation.*lower batch_size"):
        eval_crps(pred, truth, batch_size=7)


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
        rw_model_factory,
        forecast_fn=_canned_forecast_fn,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=20,
    )
    # Windows at t1 in {12, 16, 20} (stop = 24 - 4 + 1 = 21).
    assert [r.t1 for r in results] == [12, 16, 20]
    for r in results:
        assert isinstance(r, BacktestResult)
        assert r.t0 == 0  # expanding window
        assert set(r.metrics) == {"mae", "rmse", "crps", "coverage"}
        assert r.walltime >= 0.0


def test_backtest_rolling_window(rng_key: Array) -> None:
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    results = backtest(
        rng_key,
        data,
        covariates,
        rw_model_factory,
        forecast_fn=_canned_forecast_fn,
        window_type="rolling",
        train_window=12,
        test_window=4,
        stride=4,
        num_samples=20,
    )
    # Windows at t1 in {12, 16, 20} (stop = 24 - 4 + 1 = 21), each rolling.
    assert [r.t1 for r in results] == [12, 16, 20]
    for r in results:
        assert isinstance(r, BacktestResult)
        assert r.t1 - r.t0 == 12  # fixed-size rolling window
        assert set(r.metrics) == {"mae", "rmse", "crps", "coverage"}


def test_backtest_infers_rolling_from_train_window(rng_key: Array) -> None:
    # Omitting window_type but setting train_window infers "rolling" (backward compat),
    # producing the same windows as an explicit window_type="rolling".
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    run = partial(
        backtest,
        rng_key,
        data,
        covariates,
        rw_model_factory,
        forecast_fn=_canned_forecast_fn,
        train_window=12,
        test_window=4,
        stride=4,
        num_samples=20,
    )
    inferred = run()
    explicit = run(window_type="rolling")
    assert [(r.t0, r.t1, r.t2) for r in inferred] == [(r.t0, r.t1, r.t2) for r in explicit]
    assert all(r.t1 - r.t0 == 12 for r in inferred)


def test_backtest_rolling_requires_train_window(rng_key: Array) -> None:
    with pytest.raises(ValueError, match="'rolling' requires a fixed train_window"):
        backtest(
            rng_key,
            jnp.zeros((24, 1)),
            jnp.zeros((24, 0)),
            rw_model_factory,
            forecast_fn=_never_called,
            window_type="rolling",
            test_window=4,
        )


def test_backtest_expanding_rejects_train_window(rng_key: Array) -> None:
    with pytest.raises(ValueError, match="'expanding' is incompatible with train_window"):
        backtest(
            rng_key,
            jnp.zeros((24, 1)),
            jnp.zeros((24, 0)),
            rw_model_factory,
            forecast_fn=_never_called,
            window_type="expanding",
            train_window=12,
            test_window=4,
        )


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
        rw_model_factory,
        forecast_fn=_canned_forecast_fn,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=50,
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
        walltime=0.5,
        metrics={"mae": 1.0},
    )
    flat = result.to_dict()
    assert flat["t0"] == 0
    assert flat["t1"] == 10
    assert flat["metrics"] == {"mae": 1.0}
    assert flat["train_metrics"] == {}
    assert flat["prediction"] is None
    assert set(flat) == {
        "t0",
        "t1",
        "t2",
        "num_samples",
        "walltime",
        "metrics",
        "train_metrics",
        "prediction",
    }


def test_backtest_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="share the time axis length"):
        backtest(
            random.PRNGKey(0),
            jnp.zeros((20, 1)),
            jnp.zeros((18, 0)),
            rw_model_factory,
            forecast_fn=_never_called,
        )


def test_backtest_defaults_leave_train_metrics_and_prediction_empty(rng_key: Array) -> None:
    # Pyro-faithful defaults: no in-sample scoring, no retained predictions.
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    results = backtest(
        rng_key,
        data,
        covariates,
        rw_model_factory,
        forecast_fn=_canned_forecast_fn,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=20,
    )
    assert results
    for r in results:
        assert r.train_metrics == {}
        assert r.prediction is None


def test_backtest_per_window_metrics_hook(rng_key: Array) -> None:
    from numpyro_forecast.metrics import make_mase

    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))

    def per_window(t0: int, t1: int, t2: int) -> dict[str, object]:
        # A MASE scaled by this window's own training slice.
        return {"mase": make_mase(data[..., t0:t1, :], seasonality=1)}

    results = backtest(
        rng_key,
        data,
        covariates,
        rw_model_factory,
        forecast_fn=_canned_forecast_fn,
        metrics={"crps": eval_crps},
        per_window_metrics=cast("Any", per_window),
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=20,
    )
    assert results
    for r in results:
        assert set(r.metrics) == {"crps", "mase"}
        assert isinstance(r.metrics["mase"], float)


def test_backtest_eval_train_populates_train_metrics(rng_key: Array) -> None:
    # A real (if cheap) SVI fit, to exercise in_sample_fn end to end at least once.
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    results = backtest(
        rng_key,
        data,
        covariates,
        rw_model_factory,
        forecast_fn=svi_forecast_fn(num_steps=20),
        in_sample_fn=svi_in_sample_fn(num_steps=20),
        metrics={"crps": eval_crps},
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=20,
        eval_train=True,
    )
    assert results
    for r in results:
        assert set(r.train_metrics) == set(r.metrics) == {"crps"}
        assert isinstance(r.train_metrics["crps"], float)


def test_backtest_eval_train_does_not_change_oos_metrics(rng_key: Array) -> None:
    # Enabling the in-sample diagnostic must not perturb the key passed to
    # forecast_fn, so the out-of-sample metrics are identical with eval_train
    # off and on. _canned_forecast_fn is rng-sensitive (its draws depend on the
    # key it receives), so this only passes if that key is undisturbed.
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))

    def run(*, eval_train: bool) -> list[BacktestResult]:
        return backtest(
            rng_key,
            data,
            covariates,
            rw_model_factory,
            forecast_fn=_canned_forecast_fn,
            in_sample_fn=_canned_in_sample_fn,
            metrics={"crps": eval_crps},
            test_window=4,
            min_train_window=12,
            stride=4,
            num_samples=20,
            eval_train=eval_train,
        )

    without_train = run(eval_train=False)
    with_train = run(eval_train=True)
    assert without_train and len(without_train) == len(with_train)
    for a, b in zip(without_train, with_train, strict=True):
        assert a.metrics == b.metrics


def test_backtest_keep_predictions_stores_oos_samples(rng_key: Array) -> None:
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    results = backtest(
        rng_key,
        data,
        covariates,
        rw_model_factory,
        forecast_fn=_canned_forecast_fn,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=20,
        keep_predictions=True,
    )
    assert results
    for r in results:
        assert r.prediction is not None
        assert r.prediction.shape == (20, 4, 1)


def test_backtest_keeps_numpy_predictions_from_host_forecast_fn(rng_key: Array) -> None:
    """A ``forecast_fn`` returning NumPy (the backend-free ``device="host"`` result) is kept as is.

    ``BacktestResult.prediction`` is type-checked at construction by the
    jaxtyping import hook, so it must admit the NumPy container the host path
    returns when no CPU backend is initialized.
    """
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))

    def numpy_forecast_fn(*args: object, **kwargs: object) -> np.ndarray:
        return np.asarray(_canned_forecast_fn(*args, **kwargs))  # ty: ignore[invalid-argument-type]

    results = backtest(
        rng_key,
        data,
        covariates,
        rw_model_factory,
        forecast_fn=numpy_forecast_fn,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=20,
        keep_predictions=True,
    )
    assert results
    for r in results:
        assert isinstance(r.prediction, np.ndarray)
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
        rw_model_factory,
        forecast_fn=_canned_forecast_fn,
        in_sample_fn=_canned_in_sample_fn,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=20,
        transform=transform,
        eval_train=True,
    )
    assert len(results) == 3
    assert transform_calls["count"] == 6


def test_backtest_eval_train_requires_in_sample_fn(rng_key: Array) -> None:
    # eval_train=True with the default in_sample_fn=None raises before any
    # window runs, so the forecast_fn spy is never invoked.
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    with pytest.raises(ValueError, match="eval_train=True requires in_sample_fn"):
        backtest(
            rng_key,
            data,
            covariates,
            rw_model_factory,
            forecast_fn=_never_called,
            test_window=4,
            min_train_window=12,
            stride=4,
            num_samples=10,
            eval_train=True,
        )


def test_backtest_forwards_batch_size_into_both_closures(rng_key: Array) -> None:
    # Spec requirement: batch_size is forwarded unchanged into forecast_fn and
    # in_sample_fn (spy closures record the kwarg they were called with).
    data = jnp.cumsum(0.1 * random.normal(rng_key, (24, 1)), axis=-2)
    covariates = jnp.zeros((24, 0))
    forecast_calls: list[dict[str, Any]] = []
    in_sample_calls: list[dict[str, Any]] = []

    def in_sample_fn(
        rng_key: Array,
        model: object,
        train_data: Array,
        train_covariates: Array,
        num_samples: int,
        *,
        batch_size: int | None = None,
    ) -> Array:
        in_sample_calls.append({"batch_size": batch_size})
        return _canned_in_sample_fn(
            rng_key, model, train_data, train_covariates, num_samples, batch_size=batch_size
        )

    results = backtest(
        rng_key,
        data,
        covariates,
        rw_model_factory,
        forecast_fn=_spy_closure(forecast_calls),
        in_sample_fn=in_sample_fn,
        test_window=4,
        min_train_window=12,
        stride=4,
        num_samples=10,
        batch_size=7,
        eval_train=True,
    )
    assert results
    assert forecast_calls and all(call["batch_size"] == 7 for call in forecast_calls)
    assert in_sample_calls and all(call["batch_size"] == 7 for call in in_sample_calls)


# --- unit tests for the private backtest sub-components ------------------------


def test_iter_windows_expanding() -> None:
    # window_type="expanding" -> t0 stays 0 and the window expands from the start.
    windows = list(
        _iter_windows(
            24,
            window_type="expanding",
            train_window=None,
            min_train_window=12,
            test_window=4,
            min_test_window=1,
            stride=4,
        )
    )
    assert windows == [(0, 12, 16), (0, 16, 20), (0, 20, 24)]


def test_iter_windows_fixed_train_window_rolls() -> None:
    # window_type="rolling" makes t0 track t1 (rolling, not expanding).
    windows = list(
        _iter_windows(
            24,
            window_type="rolling",
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
            window_type="expanding",
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
            window_type="expanding",
            train_window=None,
            min_train_window=4,
            test_window=2,
            min_test_window=1,
            stride=1,
        )
    )
    assert [t1 for _, t1, _ in windows] == [4, 5, 6]


def test_iter_windows_rolling_requires_train_window() -> None:
    # The dispatcher is independently correct: rolling without a train_window fails.
    with pytest.raises(ValueError, match="rolling windows require train_window"):
        list(
            _iter_windows(
                24,
                window_type="rolling",
                train_window=None,
                min_train_window=1,
                test_window=4,
                min_test_window=1,
                stride=4,
            )
        )


def test_resolve_window_type_infers_expanding_without_train_window() -> None:
    assert _resolve_window_type(None, None) == "expanding"


def test_resolve_window_type_infers_rolling_from_train_window() -> None:
    assert _resolve_window_type(None, 6) == "rolling"


def test_resolve_window_type_explicit_expanding_passes_through() -> None:
    assert _resolve_window_type("expanding", None) == "expanding"


def test_resolve_window_type_explicit_rolling_passes_through() -> None:
    assert _resolve_window_type("rolling", 6) == "rolling"


def test_resolve_window_type_rolling_requires_train_window() -> None:
    with pytest.raises(ValueError, match="'rolling' requires a fixed train_window"):
        _resolve_window_type("rolling", None)


def test_resolve_window_type_expanding_rejects_train_window() -> None:
    with pytest.raises(ValueError, match="'expanding' is incompatible with train_window"):
        _resolve_window_type("expanding", 6)


def test_expanding_windows_start_at_zero() -> None:
    windows = list(
        _expanding_windows(24, min_train_window=12, test_window=4, min_test_window=1, stride=4)
    )
    assert windows == [(0, 12, 16), (0, 16, 20), (0, 20, 24)]
    assert all(t0 == 0 for t0, _, _ in windows)


def test_expanding_windows_test_window_none_forecasts_to_end() -> None:
    windows = list(
        _expanding_windows(10, min_train_window=8, test_window=None, min_test_window=1, stride=1)
    )
    assert windows == [(0, 8, 10), (0, 9, 10)]
    assert all(t2 == 10 for _, _, t2 in windows)


def test_rolling_windows_have_constant_train_length() -> None:
    windows = list(
        _rolling_windows(24, train_window=6, test_window=4, min_test_window=1, stride=4)
    )
    assert windows == [(0, 6, 10), (4, 10, 14), (8, 14, 18), (12, 18, 22)]
    assert all(t1 - t0 == 6 for t0, t1, _ in windows)


def test_rolling_windows_test_window_none_forecasts_to_end() -> None:
    windows = list(
        _rolling_windows(10, train_window=4, test_window=None, min_test_window=1, stride=1)
    )
    assert windows == [(0, 4, 10), (1, 5, 10), (2, 6, 10), (3, 7, 10), (4, 8, 10), (5, 9, 10)]
    assert all(t2 == 10 for _, _, t2 in windows)
    assert all(t1 - t0 == 4 for t0, t1, _ in windows)


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


def _fake_forecast_fn(
    rng_key: Array,
    model: object,
    data: Array,
    covariates: Array,
    test_covariates: Array,
    num_samples: int,
    *,
    batch_size: int | None = None,
) -> Array:
    """Deterministic stand-in forecast for ``_run_window`` unit tests."""
    # Forecast horizon is the suffix of ``test_covariates`` beyond ``data``.
    horizon = test_covariates.shape[-2] - data.shape[-2]
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
        model_fn=rw_model_factory,
        shared_model=None,
        forecast_fn=_fake_forecast_fn,
        in_sample_fn=None,
        num_samples=16,
        batch_size=None,
        metrics=DEFAULT_METRICS,
        per_window_metrics=None,
        transform=transform,
        eval_train=False,
        keep_predictions=False,
    )
    assert isinstance(result, BacktestResult)
    assert (result.t0, result.t1, result.t2) == (2, 6, 8)
    assert result.num_samples == 16
    assert set(result.metrics) == {"mae", "rmse", "crps", "coverage"}
    assert result.walltime >= 0.0
    assert result.train_metrics == {}
    assert result.prediction is None
    assert transform_calls["count"] == 1


# --- P8: results_to_dataframe ------------------------------------------------


def _make_result(t0: int, **kwargs: object) -> BacktestResult:
    base: dict[str, Any] = {
        "t0": t0,
        "t1": t0 + 10,
        "t2": t0 + 14,
        "num_samples": 20,
        "walltime": 0.5,
        "metrics": {"crps": 1.0, "mae": 2.0},
        "train_metrics": {"crps": 0.8},
    }
    base.update(kwargs)
    return BacktestResult(**cast("Any", base))


def test_results_to_dataframe_schema_and_row_count() -> None:
    from numpyro_forecast.evaluate import results_to_dataframe

    results = [_make_result(0), _make_result(4)]
    df = results_to_dataframe(results)
    assert len(df) == 2
    assert set(df.columns) == {
        "t0",
        "t1",
        "t2",
        "num_samples",
        "walltime",
        "metric_crps",
        "metric_mae",
        "train_metric_crps",
    }
    assert list(df["t0"]) == [0, 4]
    assert list(df["metric_crps"]) == [1.0, 1.0]
    assert list(df["walltime"]) == [0.5, 0.5]


def test_results_to_dataframe_empty_input() -> None:
    from numpyro_forecast.evaluate import results_to_dataframe

    df = results_to_dataframe([])
    assert len(df) == 0


def test_results_to_dataframe_heterogeneous_metrics() -> None:
    from numpyro_forecast.evaluate import results_to_dataframe

    # One window carries an extra per-window metric; its column is NaN elsewhere.
    a = _make_result(0)
    b = _make_result(4, metrics={"crps": 1.0, "mae": 2.0, "mase": 0.9})
    df = results_to_dataframe([a, b])
    assert "metric_mase" in df.columns
    assert df["metric_mase"].isna().tolist() == [True, False]


def test_results_to_dataframe_no_metrics_or_params() -> None:
    from numpyro_forecast.evaluate import results_to_dataframe

    r = _make_result(0, metrics={}, train_metrics={})
    df = results_to_dataframe([r])
    assert set(df.columns) == {
        "t0",
        "t1",
        "t2",
        "num_samples",
        "walltime",
    }


def test_results_to_dataframe_missing_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    from numpyro_forecast.evaluate import results_to_dataframe

    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> object:
        if name == "pandas":
            raise ImportError("no pandas")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(ImportError, match=r"dataframes"):
        results_to_dataframe([_make_result(0)])


def test_rolling_backtest_reuse_model_predict_cache(
    count_compilations,
    rng_key: Array,
) -> None:
    """I3: ``reuse_model=True`` reuses one model so forecast kernels cache across windows."""
    import jax

    from numpyro_forecast.predictive import _predict

    duration = 80
    train, test, stride = 25, 5, 5
    data = jnp.sin(jnp.linspace(0, 6, duration))[:, None]
    cov = jnp.zeros((duration, 0))
    num_windows = (duration - train - test) // stride + 1

    def run(reuse: bool) -> int:
        _predict.clear_cache()  # ty: ignore[unresolved-attribute]
        with count_compilations() as tally:  # type: ignore[operator]
            backtest(
                rng_key,
                data,
                cov,
                rw_model_factory,
                forecast_fn=svi_forecast_fn(num_steps=30),
                train_window=train,
                test_window=test,
                stride=stride,
                num_samples=10,
                reuse_model=reuse,
            )
            jax.block_until_ready(data)
        return int(tally.count)  # type: ignore[attr-defined]

    without_reuse = run(False)
    with_reuse = run(True)
    assert with_reuse <= without_reuse
    assert _predict._cache_size() <= num_windows  # ty: ignore[unresolved-attribute]
