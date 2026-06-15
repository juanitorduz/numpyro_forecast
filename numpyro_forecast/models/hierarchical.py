"""Hierarchical origin-destination forecasting model.

Reproduces NumPyro's hierarchical forecasting tutorial, re-expressed in the
package convention ``(origin, time, destin)`` (time at axis ``-2``): per-station
seasonality, a per-destination random-walk level, a pairwise affinity term, and
additive origin/destination noise scales.
"""

from typing import cast

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast.forecaster import ForecastingModel
from numpyro_forecast.typing import Array
from numpyro_forecast.util import periodic_repeat


class HierarchicalForecaster(ForecastingModel):
    """Hierarchical OD model with per-station seasonality and drift.

    Data and covariates use the ``(origin, time, destin)`` layout. ``covariates``
    is only used for its shape (a dummy zero panel, as in the tutorial).

    Parameters
    ----------
    period
        Seasonal period in time steps (default ``24 * 7`` for hourly data).
    """

    def __init__(self, period: int = 24 * 7) -> None:
        super().__init__()
        self.period = period

    def model(self, zero_data: Array | None, covariates: Array) -> None:
        """Define the hierarchical forecasting model.

        Parameters
        ----------
        zero_data
            Unused; shapes are derived from ``covariates``.
        covariates
            Dummy covariates shaped ``(origin, duration, destin)``.
        """
        n_origin = covariates.shape[-3]
        n_destin = covariates.shape[-1]
        duration = covariates.shape[-2]

        origin_plate = numpyro.plate("origin", n_origin, dim=-3)
        destin_plate = numpyro.plate("destin", n_destin, dim=-1)
        hour_plate = numpyro.plate("hour_of_week", self.period, dim=-2)

        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-20.0, 5.0))
        destin_centered = numpyro.sample("destin_centered", dist.Uniform(0.0, 1.0))

        with origin_plate, hour_plate:
            origin_seasonal = numpyro.sample("origin_seasonal", dist.Normal(0.0, 5.0))
        with hour_plate, destin_plate:
            destin_seasonal = numpyro.sample("destin_seasonal", dist.Normal(0.0, 5.0))

        with destin_plate:
            drift = self.time_series(
                "drift",
                lambda: dist.Normal(0.0, drift_scale),
                reparam=LocScaleReparam(centered=destin_centered),
            )
        level = jnp.cumsum(drift, axis=-2)

        with origin_plate, destin_plate:
            pairwise = numpyro.sample("pairwise", dist.Normal(0.0, 1.0))

        with origin_plate:
            origin_scale = numpyro.sample("origin_scale", dist.LogNormal(-5.0, 5.0))
        with destin_plate:
            destin_scale = numpyro.sample("destin_scale", dist.LogNormal(-5.0, 5.0))
        scale = origin_scale + destin_scale

        seasonal = cast("Array", origin_seasonal + destin_seasonal)
        seasonal_repeat = periodic_repeat(seasonal, duration, axis=-2)
        prediction = level + seasonal_repeat + pairwise

        self.predict(dist.Normal(0.0, scale), prediction)
