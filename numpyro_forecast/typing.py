"""Shared type aliases used across the package.

Keeping these in a dependency-free module avoids import cycles between
``forecaster`` and ``evaluate``.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax

if TYPE_CHECKING:
    from numpyro_forecast.forecaster import ForecastingModel, _BaseForecaster

Array = jax.Array
"""A JAX array (alias of :class:`jax.Array`)."""

Metric = Callable[[Array, Array], float]
"""A metric maps ``(pred, truth)`` forecast/ground-truth arrays to a scalar."""

ModelFactory = Callable[[], "ForecastingModel"]
"""A zero-argument callable returning a fresh :class:`ForecastingModel`."""

ForecasterFactory = Callable[..., "_BaseForecaster"]
"""Callable ``(model, data, covariates, *, rng_key, **options)`` returning a forecaster.

Typed loosely (like Pyro's ``forecaster_fn``) because per-backend options differ;
the concrete classes are :class:`Forecaster` and :class:`HMCForecaster`.
"""
