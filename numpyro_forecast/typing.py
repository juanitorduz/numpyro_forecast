"""Shared type aliases used across the package.

Keeping these in a dependency-free module avoids import cycles between
``forecaster`` and ``evaluate``.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax

if TYPE_CHECKING:
    from numpyro_forecast.forecaster import _BaseForecaster

Array = jax.Array
"""A JAX array (alias of :class:`jax.Array`)."""

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
"""Callable ``(model, data, covariates, *, rng_key, **options)`` returning a forecaster.

Typed loosely (like Pyro's ``forecaster_fn``) because per-backend options differ;
the concrete classes are :class:`Forecaster` and :class:`HMCForecaster`.
"""
