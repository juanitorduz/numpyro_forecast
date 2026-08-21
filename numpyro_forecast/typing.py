"""Shared type aliases used across the package.

Keeping these in a dependency-free module avoids import cycles between
``evaluate`` and the rest of the package.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax

if TYPE_CHECKING:
    import numpy as np

Array = jax.Array
"""A JAX array (alias of :class:`jax.Array`)."""

BlackjaxBuildFn = Callable[..., object]
"""A blackjax sampler build function ``(rng_key, logdensity_fn, position, num_warmup)``.

Returns ``(inner_state, step_fn)``. ``rng_key`` is first, matching the package's
rng-key-first convention. Consumed by
:class:`~numpyro_forecast.contrib.blackjax.BlackjaxCustomKernel`.
"""

Metric = Callable[[Array, Array], Array]
"""A metric maps ``(pred, truth)`` forecast samples and ground truth to a scalar array.

``pred`` has the sample axis first, shape ``(sample, *batch)``; ``truth`` has
shape ``(*batch)``; the result is a 0-d array. Metrics must be pure JAX
functions (jit- and vmap-compatible): host floats appear only at result
boundaries (:func:`~numpyro_forecast.evaluate.evaluate_forecast`,
:class:`~numpyro_forecast.evaluate.BacktestResult`), never inside metrics, so
:func:`~numpyro_forecast.evaluate.backtest_vectorized` can vmap any metric over
the window axis. Parametrize by closure: ``functools.partial`` for keywords
(e.g. ``partial(eval_coverage, alpha=0.5)``) or a factory like
:func:`~numpyro_forecast.metrics.make_mase`.
"""


@runtime_checkable
class ForecastModel(Protocol):
    """A NumPyro forecasting model: a callable ``(covariates, data=None) -> None``.

    Any plain function with this signature satisfies this Protocol structurally
    (for example, one that derives its
    :class:`~numpyro_forecast.functional.models.Horizon` from the shapes via
    ``Horizon.from_data`` and calls the functional primitives), so nothing
    needs to subclass it. The parameters are positional-only so a user model's
    own parameter names (``cov``, ``y``, ...) stay free instead of being forced
    to match ``covariates``/``data``.

    ``ty`` checks call sites against this signature structurally (duck typing),
    which is the main payoff of the Protocol over a bare ``Callable`` alias.
    At runtime, the beartype import hook's ``isinstance`` check on a
    ``runtime_checkable`` Protocol only verifies that the named methods exist
    (Python runtime protocols never inspect signatures), so it reduces to
    ``callable(model)``: a model missing the ``data=None`` default still
    passes this check and only fails loudly at the first driver call that
    invokes it with ``data=None``.
    """

    def __call__(self, covariates: Array, data: Array | None = None, /) -> None:
        """Run the forecasting model against ``covariates`` and optional ``data``."""


ModelFactory = Callable[[], ForecastModel]
"""A zero-argument callable returning a fresh :class:`ForecastModel` instance."""

ForecastFn = Callable[..., "Array | np.ndarray"]
"""A closure that fits a model on a training window and forecasts its test horizon.

Called by :func:`~numpyro_forecast.evaluate.backtest` positionally as
``forecast_fn(rng_key, model, train_data, train_covariates, test_covariates,
num_samples, *, batch_size=None)``, where ``test_covariates`` spans the *full*
window (train followed by test, i.e. ``covariates[..., t0:t2, :]``). Must return
forecast samples with the sample axis first, shape
``(num_samples, *batch, t2 - t1, obs)``. ``batch_size`` is forwarded unchanged
from ``backtest`` so a chunked closure can bound its own device memory; a
closure that offloads work internally must return the draws back on-device
(the metrics scoring them are jitted). Typed loosely (a bare ``Callable``, like
:data:`Metric`) because per-backend fit options differ; the exact shapes are
pinned above rather than in the type itself.
"""

InSampleFn = Callable[..., "Array | np.ndarray"]
"""A closure that fits a model on a training window and scores its in-sample fit.

Called by :func:`~numpyro_forecast.evaluate.backtest` (only when
``eval_train=True``) positionally as ``in_sample_fn(rng_key, model, train_data,
train_covariates, num_samples, *, batch_size=None)``. Must return in-sample
posterior-predictive samples with the sample axis first, shape
``(num_samples, *batch, t1 - t0, obs)``. The same ``batch_size``/on-device
requirements as :data:`ForecastFn` apply.
"""
