"""Univariate local-level forecasting model.

Reproduces the model from Pyro's univariate forecasting tutorial (and its
NumPyro port): a random-walk local level plus a linear regression on (Fourier)
covariates, with a Student-T likelihood.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast.forecaster import ForecastingModel
from numpyro_forecast.typing import Array


class UnivariateForecaster(ForecastingModel):
    """Local level + regression model with Student-T observations.

    The mean is ``bias + level_t + weight @ covariates_t`` where ``level`` is a
    Gaussian random walk (``LocScaleReparam`` improves the SVI geometry). The
    regression design ``covariates`` is supplied by the caller (e.g. Fourier
    features from :func:`numpyro_forecast.util.fourier_features`).
    """

    def model(self, zero_data: Array | None, covariates: Array) -> None:
        """Define the univariate forecasting model.

        Parameters
        ----------
        zero_data
            Unused; shapes are derived from ``covariates``.
        covariates
            Regression design with time at axis ``-2``, shape
            ``(duration, num_features)``.
        """
        num_features = covariates.shape[-1]

        bias = numpyro.sample("bias", dist.Normal(0.0, 10.0))
        weight = numpyro.sample("weight", dist.Normal(0.0, 0.1).expand([num_features]).to_event(1))
        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-20.0, 5.0))
        nu = numpyro.sample("nu", dist.Gamma(10.0, 2.0))
        sigma = numpyro.sample("sigma", dist.LogNormal(-5.0, 5.0))
        centered = numpyro.sample("centered", dist.Uniform(0.0, 1.0))

        drift = self.time_series(
            "drift",
            lambda: dist.Normal(0.0, drift_scale),
            reparam=LocScaleReparam(centered=centered),
        )
        # Cumulative sum over time is the random-walk level (= the tutorials' scan).
        level = jnp.cumsum(drift, axis=-2)
        regression = (weight * covariates).sum(axis=-1, keepdims=True)
        prediction = level + bias + regression

        self.predict(dist.StudentT(df=nu, loc=0.0, scale=sigma), prediction)
