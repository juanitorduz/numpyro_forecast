"""Shared type aliases used across the package.

Keeping these in a dependency-free module avoids import cycles between
``evaluate`` and the rest of the package.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import jax
import numpy as np

Array = jax.Array
"""A JAX array (alias of `jax.Array`)."""

BlackjaxBuildFn = Callable[..., object]
"""A blackjax sampler build function ``(rng_key, logdensity_fn, position, num_warmup)``.

Runs the adaptation and returns ``(inner_state, params)``, with ``params`` a pytree of
numeric arrays holding the tuned parameters. Static settings that blackjax needs as
concrete Python values (for example an integer ``max_num_doublings``) do not go in
``params``: bind them in the `~~numpyro_forecast.typing.BlackjaxStepFn`. ``rng_key``
is first, matching the package's rng-key-first convention. Consumed by
`~~numpyro_forecast.contrib.blackjax.BlackjaxCustomKernel`.
"""

BlackjaxStepFn = Callable[..., object]
"""A blackjax step function ``(rng_key, inner_state, logdensity_fn, params)``.

Advances ``inner_state`` by one step and returns ``(inner_state, info)``. It receives
the log density of the current run and the ``params`` of a
`~~numpyro_forecast.typing.BlackjaxBuildFn` as arguments and must use those: a
closure over the log density or the tuned values of a build call keeps the values of
the first call. A closure over static configuration (for example
``max_num_doublings``) is fine. Consumed by
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

Every driver checks each result on its own, so the outcome for one metric
never depends on the other metrics in the mapping: a 0-d bool or integer
result is converted to float, and any other shape or dtype (for example the
per-cell values of `~~numpyro_forecast.metrics.crps_empirical()`, whose
metric counterpart is `~~numpyro_forecast.evaluate.eval_crps()`) raises
``ValueError`` with the name, shape and dtype of the metric.
`~~numpyro_forecast.evaluate.evaluate_forecast()` and
`~~numpyro_forecast.evaluate.backtest()` check the value they score;
`~~numpyro_forecast.evaluate.backtest_vectorized()` checks the value of one
window inside its ``vmap``, and traces the scoring on the window shapes
before the fit, so the error comes before any window is fitted.

The built-in metrics and `~~numpyro_forecast.evaluate.evaluate_forecast()`
check that ``truth.shape`` is ``pred.shape[1:]`` and raise ``ValueError``
otherwise. With the runtime type hook on (the default), the jaxtyping
annotations reject a mismatched array first and raise
``jaxtyping.TypeCheckError``. A Python or NumPy scalar ``truth`` passes the
annotations, so the explicit check raises ``ValueError`` for it. Either way a
mismatched ``truth`` never broadcasts into a wrong value.

``pred`` and ``truth`` may have any real dtype. A count model (a discrete
observation family in `~~numpyro_forecast.models.predict()`) needs integer
data, so its forecast draws are integers. The built-in metrics cast an integer
operand to the default float dtype (``float32``, or ``float64`` with x64
enabled) before they compute, and keep a floating operand in its own dtype.
``float32`` holds every integer up to $2^{24}$ exactly; larger counts round
unless x64 is enabled.
`~~numpyro_forecast.evaluate.evaluate_forecast()` (and with it
`~~numpyro_forecast.evaluate.backtest()`) and
`~~numpyro_forecast.evaluate.backtest_vectorized()` cast once before they
call the metrics, so every metric receives floating arrays and can compute
``jnp.abs(pred - truth)`` without integer wraparound; a custom metric that
needs the integer draws must read them from the kept predictions
(``keep_predictions=True``) or cast back inside the metric. The built-in
metrics reject boolean and complex arrays.
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
    on-device avoid that hop for the metrics scored every window. The
    parameters are positional-only so a closure keeps its own parameter names.
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
    ) -> Array | np.ndarray:
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
