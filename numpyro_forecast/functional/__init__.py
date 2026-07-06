"""Functional forecasting API: the pure core shared with the OOP classes.

This package is the functional counterpart of :mod:`numpyro_forecast.forecaster`.
Where the OOP API carries the train/forecast split as mutable state on a
:class:`~numpyro_forecast.forecaster.ForecastingModel` instance, here it is an
explicit, immutable :class:`Horizon` value threaded into the primitives. The
class-based API is a thin shim over the functions defined here, so the two
styles are fully interchangeable: both produce a NumPyro model callable
``(covariates, data=None)`` and consume a posterior dict of latent draws.

The package is organized by scope: :mod:`~numpyro_forecast.functional.models`
(the :class:`Horizon` value and the model-building primitives),
:mod:`~numpyro_forecast.functional.svi` (optimizer/guide resolution and SVI
fitting), :mod:`~numpyro_forecast.functional.mcmc` (kernel resolution and MCMC
fitting), :mod:`~numpyro_forecast.functional.posterior` (drawing posterior
samples from a fit), and :mod:`~numpyro_forecast.functional.prediction`
(forecasting and in-sample prediction). Every public name is re-exported here.
"""

from numpyro_forecast.functional.mcmc import MCMCFit, fit_mcmc, resolve_kernel
from numpyro_forecast.functional.models import (
    Horizon,
    Transition,
    forecasting_model,
    markov_time_series,
    predict,
    predict_glm,
    time_series,
)
from numpyro_forecast.functional.posterior import draw_posterior
from numpyro_forecast.functional.prediction import forecast, predict_in_sample
from numpyro_forecast.functional.svi import SVIFit, fit_svi, resolve_guide, resolve_optimizer

__all__ = [
    "Horizon",
    "MCMCFit",
    "SVIFit",
    "Transition",
    "draw_posterior",
    "fit_mcmc",
    "fit_svi",
    "forecast",
    "forecasting_model",
    "markov_time_series",
    "predict",
    "predict_glm",
    "predict_in_sample",
    "resolve_guide",
    "resolve_kernel",
    "resolve_optimizer",
    "time_series",
]
