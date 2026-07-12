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
    """The vectorized backtest requires an ``AutoGuide``.

    Raised by :func:`~numpyro_forecast.evaluate.backtest_vectorized` when the
    resolved guide is hand-written: those are not vmappable, use
    :func:`~numpyro_forecast.evaluate.backtest` instead.
    """

    default_message = (
        "backtest_vectorized requires an AutoGuide (guide resolves to one); "
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


class OptimizerResolutionError(NumpyroForecastError, TypeError):
    """An optimizer specification could not be resolved.

    Raised by :func:`~numpyro_forecast.functional.svi.resolve_optimizer` for
    boolean inputs of any form (``bool`` is an ``int`` subclass, so a bool
    would silently mean ``Adam(1.0)``; the default message) and for any other
    unrecognized type.
    """

    default_message = (
        "resolve_optimizer() does not accept bool; pass a positive float learning "
        "rate, an optax.GradientTransformation, a numpyro optimizer, or None."
    )


class GuideResolutionError(NumpyroForecastError, TypeError):
    """A guide specification could not be resolved.

    Raised by :func:`~numpyro_forecast.functional.svi.resolve_guide` for a
    callable shaped like a guide *factory* (the default message) or for an
    unsupported type.
    """

    default_message = (
        "resolve_guide() received a callable taking a single required positional "
        "argument and no defaults. This is ambiguous: it looks like a guide *factory* "
        "(e.g. `lambda model: AutoNormal(model)`), which must be passed as the class "
        "or a functools.partial of it, not a lambda; or it is a hand-written guide, "
        "which must use the model signature `(covariates, data=None)`."
    )


class GuideSampleArgsError(NumpyroForecastError, ValueError):
    """Drawing from a hand-written guide needs the in-sample arguments.

    Raised by :func:`~numpyro_forecast.functional.posterior.draw_posterior` when an
    :class:`~numpyro_forecast.functional.svi.SVIFit` holding a hand-written guide
    was constructed without its in-sample covariates/data.
    """

    default_message = (
        "drawing from a hand-written guide requires the in-sample covariates/data, "
        "which this SVIFit was constructed without. Fit via fit_svi (which records "
        "them) or provide them explicitly."
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


class KernelResolutionError(NumpyroForecastError, TypeError):
    """A kernel specification could not be resolved.

    Raised by :func:`~numpyro_forecast.functional.mcmc.resolve_kernel` for a type
    that is neither ``None``, an ``MCMCKernel`` subclass, nor an ``MCMCKernel``
    instance.
    """


class KernelConfigError(NumpyroForecastError, ValueError):
    """A kernel is combined with an invalid configuration.

    Raised by :func:`~numpyro_forecast.functional.mcmc.resolve_kernel` when
    ``kernel_kwargs`` accompany an already-constructed instance, and by
    :func:`~numpyro_forecast.functional.mcmc.fit_mcmc` for run-config constraints
    (ensemble samplers need multiple vectorized chains; BlackJAX kernels need
    sequential chains).
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
