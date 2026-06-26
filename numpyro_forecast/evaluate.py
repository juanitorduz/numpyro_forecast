"""Backtesting and evaluation metrics.

This is the JAX/NumPyro port of ``pyro.contrib.forecast.evaluate``. Unlike Pyro
there is no global parameter store, so each backtest window is a pure call that
fits its own forecaster.
"""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from functools import partial
from time import perf_counter
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Float

from numpyro_forecast.forecaster import Forecaster
from numpyro_forecast.metrics import crps_empirical
from numpyro_forecast.typing import Array, ForecasterFactory, Metric, ModelFactory


@jax.jit
def _mae(pred: Array, truth: Array) -> Array:
    """Jitted scalar MAE kernel (sample median as point estimate)."""
    return jnp.abs(jnp.median(pred, axis=0) - truth).mean()


@jax.jit
def _rmse(pred: Array, truth: Array) -> Array:
    """Jitted scalar RMSE kernel (sample mean as point estimate)."""
    return jnp.sqrt(jnp.square(pred.mean(axis=0) - truth).mean())


@jax.jit
def _crps(pred: Array, truth: Array) -> Array:
    """Jitted scalar mean-CRPS kernel."""
    return crps_empirical(pred, truth).mean()


_DEFAULT_COVERAGE_ALPHA = 0.9
"""Default nominal level for the central coverage interval."""


@partial(jax.jit, static_argnums=(2,))
def _coverage(pred: Array, truth: Array, alpha: float) -> Array:
    """Jitted scalar coverage kernel for the central ``alpha`` interval."""
    tail = (1.0 - alpha) / 2.0
    lo = jnp.quantile(pred, tail, axis=0)
    hi = jnp.quantile(pred, 1.0 - tail, axis=0)
    return ((truth >= lo) & (truth <= hi)).mean()


def eval_mae(pred: Array, truth: Array) -> float:
    """Mean absolute error using the forecast sample median as point estimate.

    Parameters
    ----------
    pred
        Forecast samples with the sample axis first.
    truth
        Ground-truth values (matching ``pred`` without the sample axis).

    Returns
    -------
    float
        The mean absolute error.
    """
    return float(_mae(pred, truth))


def eval_rmse(pred: Array, truth: Array) -> float:
    """Root mean squared error using the forecast sample mean as point estimate.

    Parameters
    ----------
    pred
        Forecast samples with the sample axis first.
    truth
        Ground-truth values (matching ``pred`` without the sample axis).

    Returns
    -------
    float
        The root mean squared error.
    """
    return float(_rmse(pred, truth))


def eval_crps(pred: Array, truth: Array) -> float:
    """Empirical CRPS averaged over all data elements.

    Parameters
    ----------
    pred
        Forecast samples with the sample axis first.
    truth
        Ground-truth values (matching ``pred`` without the sample axis).

    Returns
    -------
    float
        The mean empirical CRPS.
    """
    return float(_crps(pred, truth))


def eval_coverage(pred: Array, truth: Array, *, alpha: float = _DEFAULT_COVERAGE_ALPHA) -> float:
    """Empirical coverage of the central ``alpha`` prediction interval.

    The central ``alpha`` interval is bounded by the ``(1 - alpha) / 2`` and
    ``1 - (1 - alpha) / 2`` quantiles of the forecast samples; the metric is the
    fraction of ground-truth values that fall inside it. A well-calibrated
    forecast has coverage close to ``alpha``.

    Parameters
    ----------
    pred
        Forecast samples with the sample axis first.
    truth
        Ground-truth values (matching ``pred`` without the sample axis).
    alpha
        Nominal interval level in ``(0, 1)`` (defaults to ``0.9``).

    Returns
    -------
    float
        The fraction of ground-truth values inside the central ``alpha`` interval.
    """
    return float(_coverage(pred, truth, alpha))


DEFAULT_METRICS: dict[str, Metric] = {
    "mae": eval_mae,
    "rmse": eval_rmse,
    "crps": eval_crps,
    "coverage": eval_coverage,
}
"""Default metrics used by :func:`backtest`."""


def evaluate_forecast(
    pred: Float[Array, " sample *batch"],
    truth: Float[Array, " *batch"],
    *,
    metrics: Mapping[str, Metric] | None = None,
) -> dict[str, float]:
    """Evaluate forecast samples against ground truth for several metrics at once.

    A one-call convenience that applies each metric in ``metrics`` to the same
    forecast samples and ground truth. It is the one-shot counterpart to
    :func:`backtest` and is also used internally by :func:`backtest` to score
    each rolling window.

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
        Each function takes ``(pred, truth)`` and returns a float; bind any extra
        parameters with :func:`functools.partial` (see above).

    Returns
    -------
    dict[str, float]
        Each metric name mapped to its value.
    """
    if metrics is None or metrics is DEFAULT_METRICS:
        # Default path: evaluate the four jitted kernels, then pull the whole
        # batch across the device boundary in a single host transfer instead of
        # one ``float(...)`` sync per metric.
        stacked = jnp.stack(
            [
                _mae(pred, truth),
                _rmse(pred, truth),
                _crps(pred, truth),
                _coverage(pred, truth, _DEFAULT_COVERAGE_ALPHA),
            ]
        )
        mae, rmse, crps, coverage = stacked.tolist()
        return {"mae": mae, "rmse": rmse, "crps": crps, "coverage": coverage}
    return {name: fn(pred, truth) for name, fn in metrics.items()}


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
    """

    t0: int
    t1: int
    t2: int
    num_samples: int
    train_walltime: float
    test_walltime: float
    metrics: dict[str, float]
    params: dict[str, float] = field(default_factory=dict)

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


def _timed[T](fn: Callable[[], T]) -> tuple[T, float]:
    """Run ``fn`` and return its result alongside the wall-clock seconds it took."""
    start = perf_counter()
    result = fn()
    return result, perf_counter() - start


def _iter_windows(
    duration: int,
    *,
    train_window: int | None,
    min_train_window: int,
    test_window: int | None,
    min_test_window: int,
    stride: int,
) -> Iterator[tuple[int, int, int]]:
    """Yield ``(t0, t1, t2)`` train-begin/split/test-end indices for each window."""
    stop = duration - (min_test_window if test_window is None else test_window) + 1
    start = min_train_window if train_window is None else train_window
    for t1 in range(start, stop, stride):
        t0 = 0 if train_window is None else t1 - train_window
        t2 = duration if test_window is None else t1 + test_window
        yield t0, t1, t2


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


def _run_window(
    rng_key: Array,
    t0: int,
    t1: int,
    t2: int,
    *,
    data: Array,
    covariates: Array,
    model_fn: ModelFactory,
    forecaster_fn: ForecasterFactory,
    options: Mapping[str, Any],
    num_samples: int,
    batch_size: int | None,
    metrics: Mapping[str, Metric],
    transform: Callable[[Array, Array], tuple[Array, Array]] | None,
) -> BacktestResult:
    """Fit, forecast, and score a single backtest window into a :class:`BacktestResult`."""
    train_data, train_covariates, test_covariates, truth = _slice_window(
        data, covariates, t0, t1, t2
    )
    key_fit, key_forecast = random.split(rng_key)

    forecaster, train_walltime = _timed(
        lambda: forecaster_fn(key_fit, model_fn(), train_data, train_covariates, **options)
    )
    pred, test_walltime = _timed(
        lambda: forecaster(
            key_forecast, train_data, test_covariates, num_samples, batch_size=batch_size
        )
    )

    if transform is not None:
        pred, truth = transform(pred, truth)

    return BacktestResult(
        t0=t0,
        t1=t1,
        t2=t2,
        num_samples=num_samples,
        train_walltime=train_walltime,
        test_walltime=test_walltime,
        metrics=evaluate_forecast(pred, truth, metrics=metrics),
        params=_scalar_params(forecaster),
    )


def backtest(
    rng_key: Array,
    data: Array,
    covariates: Array,
    model_fn: ModelFactory,
    *,
    forecaster_fn: ForecasterFactory = Forecaster,
    metrics: Mapping[str, Metric] | None = None,
    transform: Callable[[Array, Array], tuple[Array, Array]] | None = None,
    train_window: int | None = None,
    min_train_window: int = 1,
    test_window: int | None = None,
    min_test_window: int = 1,
    stride: int = 1,
    num_samples: int = 100,
    batch_size: int | None = None,
    forecaster_options: Mapping[str, Any] | Callable[..., Mapping[str, Any]] | None = None,
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
        Each function takes ``(pred, truth)`` and returns a float; bind any
        metric-specific parameters with :func:`functools.partial`, e.g.
        ``{**DEFAULT_METRICS, "coverage": partial(eval_coverage, alpha=0.8)}``.
    transform
        Optional ``(pred, truth) -> (pred, truth)`` applied before metrics.
    train_window
        Training window size; if ``None`` the window expands from the start.
    min_train_window
        Minimum training window size when ``train_window`` is ``None``.
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

    Returns
    -------
    list[BacktestResult]
        One result per backtest window.
    """
    if data.shape[-2] != covariates.shape[-2]:
        msg = "data and covariates must share the time axis length"
        raise ValueError(msg)
    metrics = DEFAULT_METRICS if metrics is None else metrics
    duration = data.shape[-2]

    results: list[BacktestResult] = []
    for t0, t1, t2 in _iter_windows(
        duration,
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
                forecaster_fn=forecaster_fn,
                options=_resolve_options(forecaster_options, t0, t1, t2),
                num_samples=num_samples,
                batch_size=batch_size,
                metrics=metrics,
                transform=transform,
            )
        )

    return results
