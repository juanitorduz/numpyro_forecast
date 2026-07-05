"""Shared type aliases used across the package.

Keeping these in a dependency-free module avoids import cycles between
``forecaster`` and ``evaluate``.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax

if TYPE_CHECKING:
    import optax
    from numpyro.infer.autoguide import AutoGuide
    from numpyro.infer.mcmc import MCMCKernel
    from numpyro.optim import _NumPyroOptim

    from numpyro_forecast.forecaster import _BaseForecaster

Array = jax.Array
"""A JAX array (alias of :class:`jax.Array`)."""

if TYPE_CHECKING:
    # Precise unions for static checkers. At runtime (the ``else`` branch) each is
    # ``object`` so the beartype import hook accepts any resolvable form without
    # forcing an ``optax``/``numpyro`` import at package import time.
    OptimizerLike = float | int | _NumPyroOptim | optax.GradientTransformation | None
    """An optimizer specification accepted by :func:`~numpyro_forecast.functional.fit_svi`.

    Resolved by :func:`~numpyro_forecast.functional.resolve_optimizer`: ``None``
    (default ``Adam``), a positive scalar learning rate, an
    ``optax.GradientTransformation``, or a NumPyro ``_NumPyroOptim``.
    """

    GuideLike = AutoGuide | type[AutoGuide] | Callable[..., object] | None
    """A guide specification accepted by :func:`~numpyro_forecast.functional.fit_svi`.

    Resolved by :func:`~numpyro_forecast.functional.resolve_guide`: ``None``
    (``AutoNormal``), an ``AutoGuide`` instance, an ``AutoGuide`` subclass or a
    ``functools.partial`` factory of one, or a hand-written guide function.
    """

    KernelLike = MCMCKernel | type[MCMCKernel] | None
    """A kernel specification accepted by :func:`~numpyro_forecast.functional.fit_mcmc`.

    Resolved by :func:`~numpyro_forecast.functional.resolve_kernel`: ``None``
    (``NUTS``), an ``MCMCKernel`` instance, or an ``MCMCKernel`` subclass.
    """
else:
    OptimizerLike = GuideLike = KernelLike = object

BlackjaxBuildFn = Callable[..., object]
"""A blackjax sampler build function ``(rng_key, logdensity_fn, position, num_warmup)``.

Returns ``(inner_state, step_fn)``. ``rng_key`` is first, matching the package's
rng-key-first convention. Consumed by
:class:`~numpyro_forecast.contrib.blackjax.BlackjaxCustomKernel`.
"""

Metric = Callable[[Array, Array], float]
"""A metric maps ``(pred, truth)`` forecast/ground-truth arrays to a scalar."""

ForecastModel = Callable[..., None]
"""A NumPyro forecasting model callable ``(covariates, data=None) -> None``.

Both an OOP :class:`~numpyro_forecast.forecaster.ForecastingModel` instance and a
plain function built by :func:`numpyro_forecast.functional.forecasting_model`
satisfy this. Typed loosely (a bare ``Callable``) on purpose: the package's
beartype import hook performs an ``isinstance``-style check on annotated
parameters, so a nominal ``ForecastingModel`` hint would reject functional
models at runtime, whereas ``Callable`` accepts either.
"""

ModelFactory = Callable[[], ForecastModel]
"""A zero-argument callable returning a fresh forecasting model (OOP or functional)."""

ForecasterFactory = Callable[..., "_BaseForecaster"]
"""Callable ``(rng_key, model, data, covariates, **options)`` returning a forecaster.

Typed loosely (like Pyro's ``forecaster_fn``) because per-backend options differ;
the concrete classes are :class:`Forecaster` and :class:`HMCForecaster`.
"""
