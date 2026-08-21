"""Functional forecasting API: the pure core of the package.

The train/forecast split is carried as an explicit, immutable :class:`Horizon`
value threaded into the primitives (never mutable state on an object), and
every model is a plain NumPyro model callable ``(covariates, data=None)``
consuming a posterior dict of latent draws. Fitting is left to the caller: use
NumPyro's ``SVI``/``MCMC`` directly (or an optional backend such as
:mod:`numpyro_forecast.contrib.blackjax`) and pass the resulting guide/params
or posterior-sample dict to :func:`draw_posterior` and
:func:`~numpyro_forecast.functional.prediction.forecast`.

The package is organized by scope: :mod:`~numpyro_forecast.functional.models`
(the :class:`Horizon` value and the model-building primitives),
:mod:`~numpyro_forecast.functional.posterior` (drawing posterior samples from a
fitted variational guide), and :mod:`~numpyro_forecast.functional.prediction`
(forecasting and in-sample prediction). Every public name is re-exported here.
"""

from numpyro_forecast.functional.models import (
    Horizon,
    Transition,
    markov_time_series,
    predict,
    predict_glm,
    time_series,
)
from numpyro_forecast.functional.posterior import draw_posterior
from numpyro_forecast.functional.prediction import forecast, predict_in_sample

__all__ = [
    "Horizon",
    "Transition",
    "draw_posterior",
    "forecast",
    "markov_time_series",
    "predict",
    "predict_glm",
    "predict_in_sample",
    "time_series",
]
