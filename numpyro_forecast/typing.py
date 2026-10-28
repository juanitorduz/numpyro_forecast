"""Shared type aliases used across the package.

Keeping these in a dependency-free module avoids import cycles between
``evaluate`` and the rest of the package.
"""

from collections.abc import Callable, Mapping
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

import jax
import numpy as np

Array = jax.Array
"""A JAX array (alias of `jax.Array`)."""

BlackjaxBuildFn = Callable[..., object]
"""A blackjax sampler build function ``(rng_key, logdensity_fn, position, num_warmup)``.

Returns ``(inner_state, step_fn)``. ``rng_key`` is first, matching the package's
rng-key-first convention. Consumed by
`~~numpyro_forecast.contrib.blackjax.BlackjaxCustomKernel`.
"""

Metric = Callable[[Array, Array], Array]
"""A metric maps ``(pred, truth)`` forecast samples and ground truth to a scalar array.

``pred`` has the sample axis first, shape ``(sample, *batch)``; ``truth`` has
shape ``(*batch)``; the result is a 0-d array. Metrics must be pure JAX
functions (jit- and vmap-compatible): host floats appear only at result
boundaries (`~~numpyro_forecast.evaluate.evaluate_forecast()`,
`~~numpyro_forecast.evaluate.BacktestResult`), never inside metrics, so
`~~numpyro_forecast.evaluate.backtest_vectorized()` can vmap any metric over
the window axis. Parametrize by closure: ``functools.partial`` for keywords
(e.g. ``partial(eval_coverage, alpha=0.5)``) or a factory like
`~~numpyro_forecast.metrics.make_mase()`.
"""


@runtime_checkable
class ForecastModel(Protocol):
    """A NumPyro forecasting model: a callable ``(covariates, data=None) -> None``.

    Any plain function with this signature satisfies this Protocol structurally
    (for example, one that derives its
    `~~numpyro_forecast.models.Horizon` from the shapes via
    ``Horizon.from_data`` and calls the model building blocks), so nothing
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
"""A zero-argument callable returning a fresh `ForecastModel` instance."""


@runtime_checkable
class Guide(Protocol):
    """A NumPyro guide for a `ForecastModel`: a callable with the model's signature.

    A guide is called with the same ``(covariates, data=None)`` arguments as the
    model it approximates, so this Protocol has the same shape as
    `ForecastModel`. Both an autoguide instance
    (`numpyro.infer.autoguide.AutoGuide` subclasses are callables) and a
    hand-written guide function satisfy it structurally; nothing needs to
    subclass it. `~~numpyro_forecast.evaluate.backtest_vectorized()` samples an
    autoguide through its ``sample_posterior`` and any other guide through
    ``numpyro.infer.Predictive(guide, params=...)``.

    The same runtime caveat as `ForecastModel` applies: the beartype hook's
    ``isinstance`` check on a ``runtime_checkable`` Protocol reduces to
    ``callable(guide)`` (Python never inspects signatures at runtime), while
    ``ty`` checks the signature structurally at the call site.
    """

    def __call__(self, covariates: Array, data: Array | None = None, /) -> None:
        """Run the guide against ``covariates`` and optional ``data``."""


class ForecastOutput(NamedTuple):
    """Forecast draws plus the posterior they were drawn from, returned by a `ForecastFn`.

    A `ForecastFn` may return this instead of bare draws so that
    `~~numpyro_forecast.evaluate.backtest()` can score the realized test block with
    `~~numpyro_forecast.predictive.forecast_log_density()` on the *same* posterior,
    without a second fit and without ever handing the closure the test data (the
    closure keeps seeing only the training window, which is what makes a backtest
    leak-free by construction). ``backtest`` requires it when ``log_density`` is
    requested and otherwise reads ``draws`` and ignores ``posterior``.

    Attributes
    ----------
    draws
        Forecast samples with the sample axis first, shape
        ``(num_samples, *batch, t2 - t1, obs)``; the host-offload rules of
        `ForecastFn` apply.
    posterior
        Posterior samples of the in-sample latent sites, sample axis leading, as
        accepted by `~~numpyro_forecast.predictive.forecast()`. Must not carry
        ``_future`` sites (see `~~numpyro_forecast.predictive.forecast_log_density()`).
    """

    draws: Array | np.ndarray
    posterior: Mapping[str, Array | np.ndarray]


@runtime_checkable
class ForecastFn(Protocol):
    """A closure that fits a model on a training window and forecasts its test horizon.

    Called by `~~numpyro_forecast.evaluate.backtest()` positionally, with
    ``full_covariates`` spanning the *full* window (train followed by test, i.e.
    ``covariates[..., t0:t2, :]``) and ``batch_size`` forwarded unchanged from
    ``backtest`` so a chunked closure can bound its own device memory. Returns
    forecast samples with the sample axis first, shape
    ``(num_samples, *batch, t2 - t1, obs)``. The draws may stay in host memory
    (e.g. via ``device="host"``): a jax Array committed to the CPU backend
    device or, without a CPU backend, a NumPy array. Every metric in
    `~~numpyro_forecast.evaluate.DEFAULT_METRICS` accepts such a ``pred``
    or ``truth`` (or both), in any mix and regardless of ``batch_size``, moving
    a host-resident operand to device memory first where needed; draws already
    on-device avoid that hop for the metrics scored every window. A closure may
    instead return a `ForecastOutput` (the draws plus the posterior they came
    from), which is what lets ``backtest`` score forecast log densities on the
    same fit when asked. The parameters are positional-only so a closure keeps
    its own parameter names.
    At runtime the beartype hook only checks that the value is callable
    (Python protocols never inspect signatures); ``ty`` checks the signature
    structurally at the ``backtest`` call site.
    """

    def __call__(
        self,
        rng_key: Array,
        model: ForecastModel,
        train_data: Array,
        train_covariates: Array,
        full_covariates: Array,
        num_samples: int,
        /,
        *,
        batch_size: int | None = None,
    ) -> Array | np.ndarray | ForecastOutput:
        """Fit on the training window and return forecast draws for the test window."""


@runtime_checkable
class InSampleFn(Protocol):
    """A closure that fits a model on a training window and scores its in-sample fit.

    Called by `~~numpyro_forecast.evaluate.backtest()` (only when
    ``eval_train=True``) positionally. Returns in-sample posterior-predictive
    samples with the sample axis first, shape ``(num_samples, *batch, t1 - t0,
    obs)``. The same ``batch_size``/host-offload notes as `ForecastFn`
    apply.
    """

    def __call__(
        self,
        rng_key: Array,
        model: ForecastModel,
        train_data: Array,
        train_covariates: Array,
        num_samples: int,
        /,
        *,
        batch_size: int | None = None,
    ) -> Array | np.ndarray:
        """Fit on the training window and return in-sample predictive draws."""


Reduction = Literal["marginal", "joint"]
"""How per-draw forecast log densities are reduced over the sample axis.

``"marginal"`` averages the density cell by cell (``logsumexp(axis=0) - log(S)``):
the log density of the $j$-step-ahead forecast distribution at the realized value,
with the intermediate steps not revealed, the distribution `~~numpyro_forecast.predictive.forecast()`
samples and the sample metrics score. ``"joint"`` sums over the horizon block first and
averages the block density: the chain-rule block score of `arviz_stats.loo.lfo_cv`.
The two coincide at horizon one only (for any longer block the chain rule conditions
on realized intermediate values that update the posterior) and must never be compared
with each other (see `~~numpyro_forecast.predictive.reduce_log_density()`).
"""

GroupPrior = Literal["exchangeable", "random_walk"]
"""Prior over the per-group offsets of `~~numpyro_forecast.stacking.hierarchical_weights()`.

``"exchangeable"`` is eq. 9 of Yao et al. (2022), non-centered; ``"random_walk"`` is a
centered Gaussian random walk over the group index, for ordinal groups such as the
horizon step.
"""

DensityReader = Callable[[Mapping[str, Any], Array], Array]
"""``(trace, truth) -> log_density``: read a forecast density from one traced model pass.

Called by `~~numpyro_forecast.predictive.forecast_log_density()` once per posterior
draw, under ``vmap``, with the trace of a seeded, conditioned pass of the model over the
full horizon and the realized ``truth`` of shape ``(*batch, future, obs)``. Returns the
log density of ``truth`` given that pass's sampled future path, with the same shape
as ``truth`` (a correlated family returns a size-one time axis; see
`~~numpyro_forecast.predictive.predict_density()`). Built-in readers:
`~~numpyro_forecast.predictive.predict_density()` for models built on
`~~numpyro_forecast.models.predict()` and `~~numpyro_forecast.predictive.ssoe_density()`
for models built on `~~numpyro_forecast.models.ssoe()`.
"""
