"""Backtesting and evaluation metrics.

This is the JAX/NumPyro port of ``pyro.contrib.forecast.evaluate``. Unlike Pyro
there is no global parameter store, so each backtest window is a pure call that
fits its own forecaster. :func:`backtest` supports two windowing strategies, an
``"expanding"`` window (always trains from ``t0 = 0``) and a fixed-size
``"rolling"`` window (see :data:`WindowType`), selected via ``window_type``.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random
from jaxtyping import Float
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoGuide

from numpyro_forecast.exceptions import (
    BacktestWindowError,
    VectorizedGuideError,
    VectorizedMetricError,
)
from numpyro_forecast.forecaster import Forecaster
from numpyro_forecast.functional import resolve_guide, resolve_optimizer
from numpyro_forecast.metrics import crps_empirical
from numpyro_forecast.optional import require
from numpyro_forecast.typing import Array, ForecasterFactory, ForecastModel, Metric, ModelFactory

if TYPE_CHECKING:
    import pandas as pd

    from numpyro_forecast.typing import GuideLike, OptimizerLike


_DEFAULT_COVERAGE_ALPHA = 0.9
"""Default nominal level for the central coverage interval."""


@jax.jit
def eval_mae(pred: Float[Array, " sample *batch"], truth: Float[Array, " *batch"]) -> Array:
    """Mean absolute error using the forecast sample median as point estimate.

    A pure JAX scalar kernel (see :data:`~numpyro_forecast.typing.Metric`).

    Parameters
    ----------
    pred
        Forecast samples with the sample axis first.
    truth
        Ground-truth values (matching ``pred`` without the sample axis).

    Returns
    -------
    Array
        The mean absolute error as a scalar array.
    """
    return jnp.abs(jnp.median(pred, axis=0) - truth).mean()


@jax.jit
def eval_rmse(pred: Float[Array, " sample *batch"], truth: Float[Array, " *batch"]) -> Array:
    """Root mean squared error using the forecast sample mean as point estimate.

    A pure JAX scalar kernel (see :data:`~numpyro_forecast.typing.Metric`).

    Parameters
    ----------
    pred
        Forecast samples with the sample axis first.
    truth
        Ground-truth values (matching ``pred`` without the sample axis).

    Returns
    -------
    Array
        The root mean squared error as a scalar array.
    """
    return jnp.sqrt(jnp.square(pred.mean(axis=0) - truth).mean())


@jax.jit
def eval_crps(pred: Float[Array, " sample *batch"], truth: Float[Array, " *batch"]) -> Array:
    """Empirical CRPS averaged over all data elements.

    A pure JAX scalar kernel (see :data:`~numpyro_forecast.typing.Metric`).

    Parameters
    ----------
    pred
        Forecast samples with the sample axis first.
    truth
        Ground-truth values (matching ``pred`` without the sample axis).

    Returns
    -------
    Array
        The mean empirical CRPS as a scalar array.
    """
    return crps_empirical(pred, truth).mean()


@partial(jax.jit, static_argnames=("alpha",))
def eval_coverage(
    pred: Float[Array, " sample *batch"],
    truth: Float[Array, " *batch"],
    *,
    alpha: float = _DEFAULT_COVERAGE_ALPHA,
) -> Array:
    """Empirical coverage of the central ``alpha`` prediction interval.

    The central ``alpha`` interval is bounded by the ``(1 - alpha) / 2`` and
    ``1 - (1 - alpha) / 2`` quantiles of the forecast samples; the metric is the
    fraction of ground-truth values that fall inside it. A well-calibrated
    forecast has coverage close to ``alpha``. A pure JAX scalar kernel (see
    :data:`~numpyro_forecast.typing.Metric`); bind a non-default level with
    ``functools.partial(eval_coverage, alpha=...)``.

    Parameters
    ----------
    pred
        Forecast samples with the sample axis first.
    truth
        Ground-truth values (matching ``pred`` without the sample axis).
    alpha
        Nominal interval level in ``(0, 1)``; defaults to ``0.9``.

    Returns
    -------
    Array
        The fraction of ground truth inside the central ``alpha`` interval, as
        a scalar array.

    Raises
    ------
    ValueError
        If ``alpha`` is not strictly inside ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ValueError(msg)
    tail = (1.0 - alpha) / 2.0
    lo = jnp.quantile(pred, tail, axis=0)
    hi = jnp.quantile(pred, 1.0 - tail, axis=0)
    return ((truth >= lo) & (truth <= hi)).mean()


DEFAULT_METRICS: dict[str, Metric] = {
    "mae": eval_mae,
    "rmse": eval_rmse,
    "crps": eval_crps,
    "coverage": eval_coverage,
}
"""Default metrics used by :func:`backtest` and :func:`backtest_vectorized`."""


def evaluate_forecast(
    pred: Float[Array, " sample *batch"] | Float[np.ndarray, " sample *batch"],
    truth: Float[Array, " *batch"] | Float[np.ndarray, " *batch"],
    *,
    metrics: Mapping[str, Metric] | None = None,
) -> dict[str, float]:
    """Evaluate forecast samples against ground truth for several metrics at once.

    A one-call convenience that applies each metric in ``metrics`` to the same
    forecast samples and ground truth, converting to host floats at the end. It
    is the one-shot counterpart to :func:`backtest` and is also used internally
    by :func:`backtest` to score each rolling window.

    Metric-specific parameters live with the metric in the ``metrics`` mapping,
    not on this function. To tune a metric, bind its keyword with
    :func:`functools.partial`; for example, to score coverage at the 80% level::

        from functools import partial

        metrics = {**DEFAULT_METRICS, "coverage": partial(eval_coverage, alpha=0.8)}
        evaluate_forecast(pred, truth, metrics=metrics)

    Parameters
    ----------
    pred
        Forecast samples with the sample axis first, shape ``(sample, *batch)``.
    truth
        Ground-truth values with shape ``(*batch)``.
    metrics
        Mapping of metric name to function; when ``None`` defaults to
        :data:`DEFAULT_METRICS` (``mae``, ``rmse``, ``crps`` and ``coverage``).
        Each function takes ``(pred, truth)`` and returns a scalar array (see
        :data:`~numpyro_forecast.typing.Metric`); bind any extra parameters
        with :func:`functools.partial` (see above).

    Returns
    -------
    dict[str, float]
        Each metric name mapped to its value.
    """
    metrics = DEFAULT_METRICS if metrics is None else metrics
    if not metrics:
        return {}
    # NumPy inputs (e.g. draws sampled with device="host") enter the jitted
    # metric kernels as device arrays either way; convert once up front.
    pred_arr, truth_arr = jnp.asarray(pred), jnp.asarray(truth)
    # Evaluate every metric kernel, then pull the whole batch across the device
    # boundary in a single host transfer instead of one sync per metric.
    stacked = jnp.stack([fn(pred_arr, truth_arr) for fn in metrics.values()])
    return dict(zip(metrics, stacked.tolist(), strict=True))


@dataclass(frozen=True)
class BacktestResult:
    """Per-window result of a :func:`backtest` run.

    Attributes
    ----------
    t0, t1, t2
        Train-begin, train/test split, and test-end time indices.
    num_samples
        Number of forecast samples drawn.
    train_walltime, test_walltime
        Wall-clock seconds for fitting and forecasting.
    metrics
        Mapping of metric name to value for the window.
    params
        Mapping of scalar parameter name to value (when available).
    train_metrics
        Mapping of metric name to in-sample value for the window. Empty unless
        ``backtest`` was called with ``eval_train=True``.
    prediction
        Out-of-sample forecast samples for the window (sample axis leading), or
        ``None`` unless ``backtest`` was called with ``keep_predictions=True``.
    """

    t0: int
    t1: int
    t2: int
    num_samples: int
    train_walltime: float
    test_walltime: float
    metrics: dict[str, float]
    params: dict[str, float] = field(default_factory=dict)
    train_metrics: dict[str, float] = field(default_factory=dict)
    prediction: Array | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dictionary view (Pyro-style access).

        Returns
        -------
        dict[str, Any]
            All fields as a plain dictionary.
        """
        return asdict(self)


def _scalar_params(forecaster: object) -> dict[str, float]:
    """Extract scalar variational parameters from a fitted forecaster, if any."""
    params = getattr(forecaster, "params", None)
    if not isinstance(params, Mapping):
        return {}
    return {name: float(value) for name, value in params.items() if jnp.size(value) == 1}


def _block_object[T](obj: T) -> T:
    """Block until every JAX array transitively reachable from ``obj`` is ready.

    JAX arrays materialize asynchronously, so timing a call that returns a
    container (e.g. a fitted forecaster) without blocking would report only the
    dispatch time. This passes ``vars(obj)`` (the instance ``__dict__``) to
    :func:`jax.block_until_ready`, which recurses through the dict as a pytree:
    array leaves nested inside lists/tuples/dicts/registered dataclasses are all
    awaited, not just top-level attributes. Objects without a ``__dict__``
    (``__slots__``-only or ``vars``-less) are a safe no-op.

    Parameters
    ----------
    obj
        Any object; its array-valued attributes are awaited.

    Returns
    -------
    T
        ``obj`` unchanged (returned so it can wrap a call inline).
    """
    try:
        attrs = vars(obj)
    except TypeError:
        return obj
    jax.block_until_ready(attrs)
    return obj


def _timed[T](fn: Callable[[], T]) -> tuple[T, float]:
    """Run ``fn``, block until its result is ready, and return ``(result, seconds)``.

    The ``jax.block_until_ready`` call folds asynchronous compute time into the
    measurement so reported walltimes reflect real work rather than dispatch
    time; array results are awaited directly, containers should be pre-wrapped
    with :func:`_block_object`.
    """
    start = perf_counter()
    result = fn()
    jax.block_until_ready(result)
    return result, perf_counter() - start


WindowType = Literal["expanding", "rolling"]
"""Backtest windowing strategy: an expanding (``t0=0``) or fixed-size rolling window."""


def _resolve_window_type(window_type: WindowType | None, train_window: int | None) -> WindowType:
    """Resolve and validate the backtest window strategy.

    When ``window_type`` is ``None`` the strategy is inferred from ``train_window``
    for backward compatibility: a fixed ``train_window`` implies ``"rolling"``,
    otherwise ``"expanding"``. An explicit ``window_type`` is validated against
    ``train_window`` so contradictory combinations fail loudly.

    Parameters
    ----------
    window_type
        Requested strategy, or ``None`` to infer from ``train_window``.
    train_window
        Fixed training-window size, or ``None`` for an expanding window.

    Returns
    -------
    Literal["expanding", "rolling"]
        The resolved strategy.

    Raises
    ------
    ValueError
        If ``window_type="rolling"`` but ``train_window`` is ``None``, or
        ``window_type="expanding"`` but ``train_window`` is set.
    """
    if window_type is None:
        return "rolling" if train_window is not None else "expanding"
    if window_type == "rolling" and train_window is None:
        msg = "window_type='rolling' requires a fixed train_window"
        raise ValueError(msg)
    if window_type == "expanding" and train_window is not None:
        msg = "window_type='expanding' is incompatible with train_window (t0 is always 0)"
        raise ValueError(msg)
    return window_type


def _expanding_windows(
    duration: int,
    *,
    min_train_window: int,
    test_window: int | None,
    min_test_window: int,
    stride: int,
) -> Iterator[tuple[int, int, int]]:
    """Yield ``(t0, t1, t2)`` train-begin/split/test-end indices for expanding windows.

    Every window starts at ``t0 = 0``: the training span ``[0, t1)`` lengthens by
    ``stride`` each step while the test span ``[t1, t2)`` slides forward, so all
    history up to ``t1`` is used to forecast the next window.

    Parameters
    ----------
    duration
        Total number of time steps in the series.
    min_train_window
        Length of the first (smallest) training window.
    test_window
        Test-window size; if ``None`` every window forecasts to ``duration``.
    min_test_window
        Minimum test-window size used to bound the last split when
        ``test_window`` is ``None``.
    stride
        Step between successive train/test splits.

    Yields
    ------
    tuple[int, int, int]
        The ``(t0, t1, t2)`` train-begin/split/test-end indices for each window.
    """
    stop = duration - (min_test_window if test_window is None else test_window) + 1
    for t1 in range(min_train_window, stop, stride):
        yield 0, t1, (duration if test_window is None else t1 + test_window)


def _rolling_windows(
    duration: int,
    *,
    train_window: int,
    test_window: int | None,
    min_test_window: int,
    stride: int,
) -> Iterator[tuple[int, int, int]]:
    """Yield ``(t0, t1, t2)`` train-begin/split/test-end indices for rolling windows.

    The training span ``[t0, t1)`` has constant length ``train_window`` and slides
    forward by ``stride`` each step (``t0 = t1 - train_window``), so old
    observations drop off the left edge as new ones enter the right.

    Parameters
    ----------
    duration
        Total number of time steps in the series.
    train_window
        Fixed training-window size held constant across all windows.
    test_window
        Test-window size; if ``None`` every window forecasts to ``duration``.
    min_test_window
        Minimum test-window size used to bound the last split when
        ``test_window`` is ``None``.
    stride
        Step between successive train/test splits.

    Yields
    ------
    tuple[int, int, int]
        The ``(t0, t1, t2)`` train-begin/split/test-end indices for each window.
    """
    stop = duration - (min_test_window if test_window is None else test_window) + 1
    for t1 in range(train_window, stop, stride):
        yield t1 - train_window, t1, (duration if test_window is None else t1 + test_window)


def _iter_windows(
    duration: int,
    *,
    window_type: WindowType,
    train_window: int | None,
    min_train_window: int,
    test_window: int | None,
    min_test_window: int,
    stride: int,
) -> Iterator[tuple[int, int, int]]:
    """Dispatch to the expanding or rolling window generator for ``window_type``.

    Parameters
    ----------
    duration
        Total number of time steps in the series.
    window_type
        Resolved strategy (see :func:`_resolve_window_type`).
    train_window
        Fixed training-window size; required (not ``None``) when
        ``window_type="rolling"`` and ignored otherwise.
    min_train_window
        Length of the first training window for the expanding strategy.
    test_window
        Test-window size; if ``None`` every window forecasts to ``duration``.
    min_test_window
        Minimum test-window size when ``test_window`` is ``None``.
    stride
        Step between successive train/test splits.

    Returns
    -------
    Iterator[tuple[int, int, int]]
        The ``(t0, t1, t2)`` indices for each window.

    Raises
    ------
    ValueError
        If ``window_type="rolling"`` but ``train_window`` is ``None``.
    """
    if window_type == "rolling":
        if train_window is None:
            msg = "rolling windows require train_window"
            raise ValueError(msg)
        return _rolling_windows(
            duration,
            train_window=train_window,
            test_window=test_window,
            min_test_window=min_test_window,
            stride=stride,
        )
    return _expanding_windows(
        duration,
        min_train_window=min_train_window,
        test_window=test_window,
        min_test_window=min_test_window,
        stride=stride,
    )


def _resolve_options(
    forecaster_options: Mapping[str, Any] | Callable[..., Mapping[str, Any]] | None,
    t0: int,
    t1: int,
    t2: int,
) -> Mapping[str, Any]:
    """Resolve per-window forecaster options from a mapping or a ``(t0, t1, t2)`` callable."""
    if forecaster_options is None:
        return {}
    if callable(forecaster_options) and not isinstance(forecaster_options, Mapping):
        return forecaster_options(t0=t0, t1=t1, t2=t2)
    return cast("Mapping[str, Any]", forecaster_options)


def _slice_window(
    data: Array, covariates: Array, t0: int, t1: int, t2: int
) -> tuple[Array, Array, Array, Array]:
    """Slice ``(train_data, train_covariates, test_covariates, truth)`` for one window."""
    train_data = data[..., t0:t1, :]
    train_covariates = covariates[..., t0:t1, :]
    test_covariates = covariates[..., t0:t2, :]
    truth = data[..., t1:t2, :]
    return train_data, train_covariates, test_covariates, truth


def _eval_train_window(
    rng_key: Array,
    forecaster: object,
    train_data: Array,
    train_covariates: Array,
    *,
    num_samples: int,
    batch_size: int | None,
    metrics: Mapping[str, Metric],
    transform: Callable[[Array, Array], tuple[Array, Array]] | None,
) -> dict[str, float]:
    """Score the in-sample posterior predictive over one training window.

    ``forecaster`` is typed loosely (``object``) so ``ty`` does not reject
    duck-typed forecasters at the ``forecaster`` boundary; the explicit guard
    below raises a clear :class:`TypeError` when ``predict_in_sample`` is missing.

    Raises
    ------
    TypeError
        If ``forecaster`` does not expose ``predict_in_sample``.
    """
    predict_in_sample = getattr(forecaster, "predict_in_sample", None)
    if not callable(predict_in_sample):
        msg = (
            "eval_train=True requires a forecaster exposing "
            "predict_in_sample(rng_key, covariates, num_samples, *, batch_size=None); "
            f"{type(forecaster).__name__} does not."
        )
        raise TypeError(msg)
    train_pred = cast(
        "Array", predict_in_sample(rng_key, train_covariates, num_samples, batch_size=batch_size)
    )
    train_truth = train_data
    if transform is not None:
        train_pred, train_truth = transform(train_pred, train_truth)
    return evaluate_forecast(train_pred, train_truth, metrics=metrics)


def _run_window(
    rng_key: Array,
    t0: int,
    t1: int,
    t2: int,
    *,
    data: Array,
    covariates: Array,
    model_fn: ModelFactory,
    shared_model: ForecastModel | None,
    forecaster_fn: ForecasterFactory,
    options: Mapping[str, Any],
    num_samples: int,
    batch_size: int | None,
    metrics: Mapping[str, Metric],
    per_window_metrics: Callable[[int, int, int], Mapping[str, Metric]] | None,
    transform: Callable[[Array, Array], tuple[Array, Array]] | None,
    eval_train: bool,
    keep_predictions: bool,
) -> BacktestResult:
    """Fit, forecast, and score a single backtest window into a :class:`BacktestResult`."""
    train_data, train_covariates, test_covariates, truth = _slice_window(
        data, covariates, t0, t1, t2
    )
    if per_window_metrics is not None:
        metrics = {**metrics, **per_window_metrics(t0, t1, t2)}
    key_fit, key_forecast = random.split(rng_key)

    model = shared_model if shared_model is not None else model_fn()
    forecaster, train_walltime = _timed(
        lambda: _block_object(
            forecaster_fn(key_fit, model, train_data, train_covariates, **options)
        )
    )
    pred, test_walltime = _timed(
        # Without a device argument the forecaster always returns a jax.Array
        # (the NumPy variant only arises for device="host").
        lambda: cast(
            "Array",
            forecaster(
                key_forecast, train_data, test_covariates, num_samples, batch_size=batch_size
            ),
        )
    )

    if transform is not None:
        pred, truth = transform(pred, truth)

    train_metrics: dict[str, float] = {}
    if eval_train:
        # In-sample scoring is an optional diagnostic: derive its key as an
        # independent substream via fold_in so enabling it never shifts the
        # fit/forecast keys above.
        train_metrics = _eval_train_window(
            random.fold_in(rng_key, 2),
            forecaster,
            train_data,
            train_covariates,
            num_samples=num_samples,
            batch_size=batch_size,
            metrics=metrics,
            transform=transform,
        )

    return BacktestResult(
        t0=t0,
        t1=t1,
        t2=t2,
        num_samples=num_samples,
        train_walltime=train_walltime,
        test_walltime=test_walltime,
        metrics=evaluate_forecast(pred, truth, metrics=metrics),
        params=_scalar_params(forecaster),
        train_metrics=train_metrics,
        prediction=pred if keep_predictions else None,
    )


def backtest(
    rng_key: Array,
    data: Array,
    covariates: Array,
    model_fn: ModelFactory,
    *,
    forecaster_fn: ForecasterFactory = Forecaster,
    metrics: Mapping[str, Metric] | None = None,
    per_window_metrics: Callable[[int, int, int], Mapping[str, Metric]] | None = None,
    transform: Callable[[Array, Array], tuple[Array, Array]] | None = None,
    window_type: WindowType | None = None,
    train_window: int | None = None,
    min_train_window: int = 1,
    test_window: int | None = None,
    min_test_window: int = 1,
    stride: int = 1,
    num_samples: int = 100,
    batch_size: int | None = None,
    forecaster_options: Mapping[str, Any] | Callable[..., Mapping[str, Any]] | None = None,
    eval_train: bool = False,
    keep_predictions: bool = False,
    reuse_model: bool = True,
) -> list[BacktestResult]:
    """Backtest a forecasting model on a moving window of ``(train, test)`` data.

    Parameters
    ----------
    rng_key
        Base PRNG key (used for every window, matching Pyro).
    data
        Dataset with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2`` (same duration as ``data``).
    model_fn
        Factory returning a fresh :class:`ForecastingModel` per window.
    forecaster_fn
        Factory returning a fitted forecaster (defaults to :class:`Forecaster`).
    metrics
        Mapping of metric name to function; defaults to :data:`DEFAULT_METRICS`.
        Each function takes ``(pred, truth)`` and returns a scalar array (see
        :data:`~numpyro_forecast.typing.Metric`); bind any metric-specific
        parameters with :func:`functools.partial`, e.g.
        ``{**DEFAULT_METRICS, "coverage": partial(eval_coverage, alpha=0.8)}``.
    per_window_metrics
        Optional ``(t0, t1, t2) -> Mapping[str, Metric]`` callable producing
        extra metrics merged over ``metrics`` for each window. Use it for
        window-dependent metrics such as a MASE scaled by that window's training
        data (:func:`numpyro_forecast.metrics.make_mase`).
    transform
        Optional ``(pred, truth) -> (pred, truth)`` applied before metrics.
    window_type
        Windowing strategy. If ``None`` (default) it is inferred from
        ``train_window``: ``"expanding"`` when ``train_window`` is ``None`` and
        ``"rolling"`` when it is set, matching the historical behavior. Pass
        ``"expanding"`` to always train on all history from ``t0 = 0``, or
        ``"rolling"`` to hold the training length fixed at ``train_window`` and
        slide it forward. ``"expanding"`` and ``train_window`` are mutually
        exclusive, and ``"rolling"`` requires ``train_window`` (both validated).
    train_window
        Training window size; if ``None`` the window expands from the start.
        Required for ``window_type="rolling"``.
    min_train_window
        Minimum training window size for the expanding strategy (used when
        ``train_window`` is ``None``).
    test_window
        Test window size; if ``None`` forecasts to the end of the data.
    min_test_window
        Minimum test window size when ``test_window`` is ``None``.
    stride
        Step between successive train/test splits.
    num_samples
        Number of forecast samples per window.
    batch_size
        Optional forecast-sampling chunk size.
    forecaster_options
        Options dict passed to ``forecaster_fn``, or a callable
        ``(t0, t1, t2) -> dict`` returning per-window options.
    eval_train
        If ``True``, also score the in-sample posterior predictive over each training
        window with the same ``metrics`` and store them in
        ``BacktestResult.train_metrics``. Requires a forecaster exposing
        ``predict_in_sample`` (the built-in :class:`Forecaster` and
        :class:`HMCForecaster` do).
    keep_predictions
        If ``True``, store each window's out-of-sample forecast samples (after
        ``transform``) on ``BacktestResult.prediction``. Defaults to ``False`` to
        avoid retaining large Monte Carlo arrays.
    reuse_model
        When ``True`` (default) and the windowing strategy is rolling, the model
        instance returned by the first ``model_fn()`` call is reused for every
        window so forecast/predict kernels can cache across windows. SVI still
        recompiles per window; for a single fused fit over all windows use
        :func:`backtest_vectorized`. Ignored for expanding windows and when
        ``False``.

    Returns
    -------
    list[BacktestResult]
        One result per backtest window.
    """
    if data.shape[-2] != covariates.shape[-2]:
        msg = "data and covariates must share the time axis length"
        raise ValueError(msg)
    window_type = _resolve_window_type(window_type, train_window)
    metrics = DEFAULT_METRICS if metrics is None else metrics
    duration = data.shape[-2]

    shared_model: ForecastModel | None = None
    if reuse_model and window_type == "rolling" and train_window is not None:
        shared_model = model_fn()

    results: list[BacktestResult] = []
    for t0, t1, t2 in _iter_windows(
        duration,
        window_type=window_type,
        train_window=train_window,
        min_train_window=min_train_window,
        test_window=test_window,
        min_test_window=min_test_window,
        stride=stride,
    ):
        results.append(
            _run_window(
                rng_key,
                t0,
                t1,
                t2,
                data=data,
                covariates=covariates,
                model_fn=model_fn,
                shared_model=shared_model,
                forecaster_fn=forecaster_fn,
                options=_resolve_options(forecaster_options, t0, t1, t2),
                num_samples=num_samples,
                batch_size=batch_size,
                metrics=metrics,
                per_window_metrics=per_window_metrics,
                transform=transform,
                eval_train=eval_train,
                keep_predictions=keep_predictions,
            )
        )

    return results


@dataclass(frozen=True)
class VectorizedBacktestResult:
    """Result of a :func:`backtest_vectorized` run (all windows at once).

    Unlike :class:`BacktestResult` this holds every window's values stacked
    along a leading window axis, because the windows are fitted, drawn, and
    scored in single vmapped passes rather than one call each. There are no
    per-window walltimes (a single fused computation covers all windows).

    Attributes
    ----------
    t0, t1, t2
        Train-begin, train/test split, and test-end time indices, each an
        integer array of shape ``(num_windows,)``.
    num_samples
        Number of forecast samples drawn per window.
    losses
        SVI loss history with shape ``(num_windows, num_steps)``.
    metrics
        Mapping of metric name to a ``(num_windows,)`` array of per-window
        values.
    predictions
        Stacked out-of-sample forecast samples with shape
        ``(num_windows, num_samples, *batch, test_window, obs)``, or ``None``
        unless ``keep_predictions=True``.
    """

    t0: Array
    t1: Array
    t2: Array
    num_samples: int
    losses: Array
    metrics: dict[str, Array]
    predictions: Array | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dictionary view.

        Returns
        -------
        dict[str, Any]
            All fields as a plain dictionary.
        """
        return asdict(self)


def _vmapped_metrics(
    metrics: Mapping[str, Metric], predictions: Array, truth: Array
) -> dict[str, Array]:
    """Score every window at once, returning ``name -> (num_windows,)`` arrays.

    Metrics are pure JAX scalar kernels (see
    :data:`~numpyro_forecast.typing.Metric`), so any mapping, default or
    custom, is scored by one ``vmap`` per metric over the leading window axis;
    the single device-to-host transfer is deferred to the caller.
    """
    try:
        return {name: jax.vmap(fn)(predictions, truth) for name, fn in metrics.items()}
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError) as err:
        raise VectorizedMetricError() from err


def _window_key_streams(rng_key: Array, num_windows: int) -> tuple[Array, Array, Array]:
    """Derive disjoint per-window ``(init, posterior, forecast)`` key streams.

    ``fold_in(rng_key, i)`` is the window-``i`` parent and is used as a split
    parent only, never consumed directly: consuming a key that also parents a
    split gives streams with no independence guarantee.

    Parameters
    ----------
    rng_key
        Base PRNG key.
    num_windows
        Number of rolling windows.

    Returns
    -------
    tuple[Array, Array, Array]
        ``(init_keys, post_keys, fc_keys)``, each a stacked key array with a
        leading ``num_windows`` axis.
    """
    window_keys = jax.vmap(lambda i: random.fold_in(rng_key, i))(jnp.arange(num_windows))
    subkeys = jax.vmap(lambda k: random.split(k, 3))(window_keys)
    return subkeys[:, 0], subkeys[:, 1], subkeys[:, 2]


def backtest_vectorized(
    rng_key: Array,
    data: Array,
    covariates: Array,
    model_fn: ModelFactory,
    *,
    train_window: int,
    test_window: int,
    stride: int = 1,
    num_steps: int = 1_001,
    optim: "OptimizerLike" = None,
    guide: "GuideLike" = None,
    num_samples: int = 100,
    metrics: Mapping[str, Metric] | None = None,
    keep_predictions: bool = False,
) -> VectorizedBacktestResult:
    """Rolling-window backtest with all windows fitted in one vmapped SVI run.

    Estimator-equivalent to :func:`backtest` with rolling windows; it differs
    only in PRNG stream layout and float reduction order, so the equivalence is
    statistical, not bitwise. Model, guide, and SVI compile once regardless of
    the number of windows, giving order-of-magnitude wall-clock wins for tens
    of windows on small models.

    PRNG: ``fold_in(rng_key, -1)`` seeds a discarded eager warm-up init;
    ``fold_in(rng_key, i)`` is the window-``i`` parent, split into SVI-init,
    posterior-draw, and forecast subkeys (the parent itself is never consumed;
    see :func:`_window_key_streams`).

    Parameters
    ----------
    rng_key
        Base PRNG key.
    data
        Dataset with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2`` (same duration as ``data``).
    model_fn
        Factory returning a fresh model; called exactly once (per-window model
        variation is unsupported here, use :func:`backtest`).
    train_window
        Fixed training-window length (``>= 1``).
    test_window
        Fixed test-window length (``>= 1``).
    stride
        Step between successive windows (``>= 1``).
    num_steps
        Number of SVI steps per window.
    optim
        Optimizer specification resolved by
        :func:`~numpyro_forecast.functional.svi.resolve_optimizer`.
    guide
        Guide specification; must resolve to an ``AutoGuide`` (hand-written
        guides are not vmappable, use :func:`backtest`).
    num_samples
        Number of forecast samples drawn per window.
    metrics
        Mapping of metric name to function; defaults to :data:`DEFAULT_METRICS`.
        Each metric is vmapped over the window axis, so any pure-JAX mapping,
        including partial-bound variants such as
        ``{**DEFAULT_METRICS, "coverage": partial(eval_coverage, alpha=0.5)}``,
        is scored inside the single fused computation. Host-side metrics are
        unsupported here; use :func:`backtest`, or ``keep_predictions=True``
        and score on the host.
    keep_predictions
        If ``True``, retain the stacked forecast samples on the result.

    Returns
    -------
    VectorizedBacktestResult
        The stacked per-window losses, metrics, and window indices.

    Raises
    ------
    ValueError
        If ``data`` and ``covariates`` durations differ.
    BacktestWindowError
        If ``train_window``, ``test_window``, or ``stride`` is ``< 1``, or
        there is no room for a single window.
    VectorizedGuideError
        If the resolved guide is not an ``AutoGuide``.
    VectorizedMetricError
        If a metric forces a host conversion under ``vmap`` (it is not a pure
        JAX function).
    """
    if data.shape[-2] != covariates.shape[-2]:
        msg = "data and covariates must share the time axis length"
        raise ValueError(msg)
    if train_window < 1:
        raise BacktestWindowError("train_window must be >= 1")
    if test_window < 1:
        raise BacktestWindowError("test_window must be >= 1")
    if stride < 1:
        raise BacktestWindowError("stride must be >= 1")

    duration = data.shape[-2]
    last_start = duration - train_window - test_window
    if last_start < 0:
        msg = "train_window + test_window exceeds the series duration; no window fits"
        raise BacktestWindowError(msg)
    starts = jnp.arange(0, last_start + 1, stride)

    def slice_one(t0: Array) -> tuple[Array, Array, Array, Array]:
        train_d = lax.dynamic_slice_in_dim(data, t0, train_window, axis=-2)
        train_c = lax.dynamic_slice_in_dim(covariates, t0, train_window, axis=-2)
        hor_c = lax.dynamic_slice_in_dim(covariates, t0, train_window + test_window, axis=-2)
        truth = lax.dynamic_slice_in_dim(data, t0 + train_window, test_window, axis=-2)
        return train_d, train_c, hor_c, truth

    train_d, train_c, hor_c, truth = jax.vmap(slice_one)(starts)

    model = model_fn()
    resolved_guide = resolve_guide(guide, model)
    if not isinstance(resolved_guide, AutoGuide):
        raise VectorizedGuideError()
    svi = SVI(model, resolved_guide, resolve_optimizer(optim), Trace_ELBO())

    # This eager warm-up on concrete window-0 arrays is mandatory: AutoGuide
    # caches its prototype trace on the instance at first init, so if that
    # first init happens under vmap tracing the instance retains tracers and
    # any later eager use raises UnexpectedTracerError.
    # (fold_in requires a non-negative uint32 index on this JAX version, so
    # the ``-1`` warm-up sentinel is passed as an int32 array.)
    svi.init(random.fold_in(rng_key, jnp.array(-1, dtype=jnp.int32)), train_c[0], train_d[0])

    def fit_one(key: Array, d: Array, c: Array) -> tuple[dict[str, Array], Array]:
        state = svi.init(key, c, d)

        def step(carry: object, _: None) -> tuple[object, Array]:
            carry, loss = svi.update(carry, c, d)
            return carry, loss

        state, losses = lax.scan(step, state, length=num_steps)
        return svi.get_params(state), losses

    init_keys, post_keys, fc_keys = _window_key_streams(rng_key, int(starts.shape[0]))
    params, losses = jax.jit(jax.vmap(fit_one))(init_keys, train_d, train_c)
    posterior = jax.jit(
        jax.vmap(lambda k, p: resolved_guide.sample_posterior(k, p, sample_shape=(num_samples,)))
    )(post_keys, params)

    def forecast_one(key: Array, post_w: dict[str, Array], d: Array, hc: Array) -> Array:
        pred = Predictive(model, posterior_samples=post_w, return_sites=["forecast"])
        return pred(key, hc, d)["forecast"]

    predictions = jax.jit(jax.vmap(forecast_one))(fc_keys, posterior, train_d, hor_c)

    resolved_metrics = DEFAULT_METRICS if metrics is None else metrics
    metric_values = _vmapped_metrics(resolved_metrics, predictions, truth)

    return VectorizedBacktestResult(
        t0=starts,
        t1=starts + train_window,
        t2=starts + train_window + test_window,
        num_samples=num_samples,
        losses=losses,
        metrics=metric_values,
        predictions=predictions if keep_predictions else None,
    )


def results_to_dataframe(
    results: "Sequence[BacktestResult] | VectorizedBacktestResult",
) -> "pd.DataFrame":
    """Flatten backtest results into a tidy one-row-per-window ``DataFrame``.

    Columns are prefix-namespaced so metric, in-sample-metric, and parameter
    names never collide: window metrics become ``metric_<name>``, in-sample
    metrics ``train_metric_<name>``, and scalar parameters ``param_<name>``,
    alongside the window indices ``t0``/``t1``/``t2``, ``num_samples``, and
    ``train_walltime``/``test_walltime``. Forecast samples are excluded.
    Windows may carry different metric sets (e.g. via
    ``backtest(per_window_metrics=...)``); the union of columns is used and
    missing entries are left as ``NaN``.

    A :class:`VectorizedBacktestResult` from :func:`backtest_vectorized` is also
    accepted; it produces the same ``metric_<name>`` columns for the same metric
    set, but has no ``train_metric_*``, ``param_*``, or walltime columns (a
    vectorized run has no per-window walltimes), which are simply absent.

    Parameters
    ----------
    results
        A sequence of :class:`BacktestResult` from :func:`backtest`, or a single
        :class:`VectorizedBacktestResult` from :func:`backtest_vectorized`.

    Returns
    -------
    pandas.DataFrame
        One row per window with the namespaced columns described above.

    Raises
    ------
    ImportError
        If ``pandas`` is not installed (``pip install numpyro_forecast[dataframes]``).
    """
    pandas = require("pandas", extra="dataframes")
    if isinstance(results, VectorizedBacktestResult):
        return pandas.DataFrame(_vectorized_rows(results))
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "t0": result.t0,
            "t1": result.t1,
            "t2": result.t2,
            "num_samples": result.num_samples,
            "train_walltime": result.train_walltime,
            "test_walltime": result.test_walltime,
        }
        for name, value in result.metrics.items():
            row[f"metric_{name}"] = value
        for name, value in result.train_metrics.items():
            row[f"train_metric_{name}"] = value
        for name, value in result.params.items():
            row[f"param_{name}"] = value
        rows.append(row)
    return pandas.DataFrame(rows)


def _vectorized_rows(result: VectorizedBacktestResult) -> list[dict[str, Any]]:
    """Build one namespaced row dict per window from a vectorized result."""
    t0 = [int(x) for x in result.t0]
    t1 = [int(x) for x in result.t1]
    t2 = [int(x) for x in result.t2]
    metric_columns = {name: [float(x) for x in values] for name, values in result.metrics.items()}
    rows: list[dict[str, Any]] = []
    for i in range(len(t0)):
        row: dict[str, Any] = {
            "t0": t0[i],
            "t1": t1[i],
            "t2": t2[i],
            "num_samples": result.num_samples,
        }
        for name, column in metric_columns.items():
            row[f"metric_{name}"] = column[i]
        rows.append(row)
    return rows
