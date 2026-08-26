"""Package exception hierarchy."""

from typing import ClassVar


class NumpyroForecastError(Exception):
    """Base class for all deliberate ``numpyro_forecast`` errors.

    Subclasses may set :attr:`default_message`; instantiating without arguments
    then carries that message, and a positional message overrides it.

    Parameters
    ----------
    message
        Optional explicit message; defaults to the class
        :attr:`default_message`.
    """

    default_message: ClassVar[str] = ""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(self.default_message if message is None else message)


class BacktestWindowError(NumpyroForecastError, ValueError):
    """A backtest window configuration is invalid.

    Raised by :func:`~numpyro_forecast.evaluate.backtest_vectorized` when
    ``train_window``, ``test_window``, or ``stride`` is below 1, or when the
    series has no room for a single window.
    """


class VectorizedGuideError(NumpyroForecastError, ValueError):
    """The vectorized backtest requires an ``AutoGuide`` instance.

    Raised by :func:`~numpyro_forecast.evaluate.backtest_vectorized` when
    ``guide`` is hand-written: those are not vmappable, use
    :func:`~numpyro_forecast.evaluate.backtest` instead.
    """

    default_message = (
        "backtest_vectorized requires an AutoGuide instance; "
        "hand-written guides are not vmappable here, use backtest() instead."
    )


class VectorizedMetricError(NumpyroForecastError, TypeError):
    """A metric is not vmappable in the vectorized backtest.

    Raised by :func:`~numpyro_forecast.evaluate.backtest_vectorized` when a
    metric forces a host conversion (e.g. ``float(...)`` or ``numpy`` calls)
    under ``vmap``. Metrics must be pure JAX functions returning a scalar
    array; see :data:`~numpyro_forecast.typing.Metric`.
    """

    default_message = (
        "backtest_vectorized metrics must be pure JAX functions returning a scalar "
        "array; one forced a host conversion (e.g. float(...) or numpy) under vmap. "
        "Use backtest(), or keep_predictions=True and score on the host."
    )


class CovariateDimsError(NumpyroForecastError, ValueError):
    """Covariate dimension names are inconsistent or malformed.

    Raised by :func:`~numpyro_forecast.convert.to_datatree` and
    :func:`~numpyro_forecast.convert.add_forecast_groups` when
    ``covariate_dims`` does not name every covariates axis, or when the names
    passed to (or inherited by) :func:`~numpyro_forecast.convert.add_forecast_groups`
    disagree with the dimension names already stored on the tree's
    ``constant_data`` covariates.
    """


class KernelConfigError(NumpyroForecastError, ValueError):
    """A ``contrib.blackjax`` kernel is run unbound or misconfigured.

    Raised by :meth:`~numpyro_forecast.contrib.blackjax._BlackjaxKernel.init`
    when the kernel was constructed with no model bound (e.g.
    ``BlackjaxNUTSKernel()`` instead of ``BlackjaxNUTSKernel(model)``); the fix
    is to pass the model as the kernel's first argument at construction time,
    before handing the kernel to :class:`~numpyro.infer.MCMC`. BlackJAX kernels
    also require ``chain_method="sequential"`` and
    ``num_warmup=0``; see the "Run configuration" section of
    :class:`~numpyro_forecast.contrib.blackjax.BlackjaxNUTSKernel` and its
    sibling kernels for why, and how misconfiguring either surfaces.
    """


class MVNLayoutError(NumpyroForecastError, NotImplementedError):
    """A ``MultivariateNormal`` layout is unsupported for time-axis surgery.

    Raised by :func:`~numpyro_forecast.surgery.shift_loc`,
    :func:`~numpyro_forecast.surgery.slice_time`, and
    :func:`~numpyro_forecast.surgery.prefix_condition` on MVN noise whose
    ``loc``/``covariance_matrix`` shapes do not match the supported
    time-leading layout.
    """

    default_message = (
        "MultivariateNormal time-axis surgery requires obs == 1 with time as the "
        "leading correlation axis (loc shape ``(*batch, time)`` or "
        "``(*batch, time, 1)`` with matching ``(*batch, time, time)`` covariance)."
    )


class DeviceMemoryError(NumpyroForecastError, RuntimeError):
    """A memory pool ran out during posterior or predictive sampling.

    Raised by :func:`~numpyro_forecast.predictive.draw_posterior` and
    the predictive drivers (:func:`~numpyro_forecast.predictive.forecast`,
    :func:`~numpyro_forecast.predictive.predict_in_sample`, and
    everything built on them, e.g. :func:`~numpyro_forecast.convert.to_datatree`)
    when XLA reports ``RESOURCE_EXHAUSTED``. For an accelerator OOM the message
    embeds the device's memory budget and the lever: the per-chunk footprint
    scales linearly with ``batch_size`` times the panel width, so lower (or
    set) ``batch_size``, free large device arrays still referenced elsewhere,
    and keep results off the accelerator with ``device="host"``. For a pinned
    host pool OOM (``Out of host memory``, reached only through an explicit
    ``device="pinned_host"`` or the caller's own pinned arrays) it names the
    pool's cap (``XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB``) and points at
    ``device="host"``, which lands results in pageable host memory, instead.
    The original XLA error is chained as ``__cause__``.
    """
