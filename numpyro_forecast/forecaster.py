"""Forecasting model base class and SVI/MCMC forecasters.

This is the JAX/NumPyro port of Pyro's ``pyro.contrib.forecast.forecaster``.
The classes here are thin object-oriented shims over the functional core in
:mod:`numpyro_forecast.functional`: :class:`ForecastingModel` threads the
train/forecast :class:`~numpyro_forecast.functional.Horizon` for you, and the
forecaster classes wrap :func:`~numpyro_forecast.functional.fit_svi` /
:func:`~numpyro_forecast.functional.fit_mcmc` plus
:func:`~numpyro_forecast.functional.forecast`. The two styles are fully
interchangeable: both consume a model callable ``(covariates, data=None)`` and a
posterior dict of latent draws.
"""

import abc
from collections.abc import Callable

import numpyro.distributions as dist
from jax import random
from jaxtyping import Float
from numpyro.infer.autoguide import AutoGuide
from numpyro.infer.reparam import Reparam
from numpyro.optim import _NumPyroOptim

from numpyro_forecast.functional import (
    Horizon,
    draw_posterior,
    fit_mcmc,
    fit_svi,
)
from numpyro_forecast.functional import forecast as _forecast
from numpyro_forecast.functional import predict as _predict
from numpyro_forecast.functional import time_series as _time_series
from numpyro_forecast.typing import Array, ForecastModel


class ForecastingModel(abc.ABC):
    """Abstract base class for forecasting models.

    Subclasses implement :meth:`model`, which must call :meth:`predict` exactly
    once. The instance itself is the (pure) NumPyro model function with signature
    ``model_instance(covariates, data=None)``: the forecast horizon is inferred
    from the shapes (``future = covariates.shape[-2] - data.shape[-2]``).

    This is the object-oriented façade over the functional API: :meth:`time_series`
    and :meth:`predict` delegate to the free functions in
    :mod:`numpyro_forecast.functional`, passing the current
    :class:`~numpyro_forecast.functional.Horizon`.
    """

    def __init__(self) -> None:
        self._horizon: Horizon | None = None

    @abc.abstractmethod
    def model(self, zero_data: Array | None, covariates: Array) -> None:
        """Define the generative model and call :meth:`predict` exactly once.

        Parameters
        ----------
        zero_data
            Zeros shaped like the data extended to the covariate duration
            (shape/dtype only; ``None`` during pure prior sampling).
        covariates
            Covariates with time at axis ``-2`` and shape
            ``(*batch, duration, cov)``.
        """
        raise NotImplementedError

    def _require_horizon(self) -> Horizon:
        if self._horizon is None:
            msg = "model state is only available during a model call"
            raise RuntimeError(msg)
        return self._horizon

    @property
    def duration(self) -> int:
        """Total horizon length ``t + future`` (in time steps)."""
        return self._require_horizon().duration

    @property
    def t_obs(self) -> int:
        """Number of observed (in-sample) time steps ``t``."""
        return self._require_horizon().t_obs

    @property
    def future(self) -> int:
        """Number of forecast time steps ``f`` (``0`` while training)."""
        return self._require_horizon().future

    def time_series(
        self,
        name: str,
        dist_fn: Callable[[], dist.Distribution],
        *,
        reparam: Reparam | None = None,
    ) -> Array:
        """Sample a time-varying latent over the full horizon.

        Thin wrapper over :func:`numpyro_forecast.functional.time_series`.

        Parameters
        ----------
        name
            Base sample-site name for the in-sample latent.
        dist_fn
            Zero-argument callable returning the per-step prior distribution.
        reparam
            Optional reparameterization (e.g. ``LocScaleReparam``) applied to
            both the in-sample and forecast sites.

        Returns
        -------
        Array
            The latent over the full horizon with time at axis ``-2``.
        """
        return _time_series(self._require_horizon(), name, dist_fn, reparam=reparam)

    def predict(self, noise_dist: dist.Distribution, prediction: Array) -> None:
        """Register the observation/forecast sites for the model.

        Thin wrapper over :func:`numpyro_forecast.functional.predict`.

        Parameters
        ----------
        noise_dist
            Zero-centered observation noise (e.g. ``Normal(0, sigma)``).
        prediction
            Deterministic mean with time at axis ``-2``, shape
            ``(*batch, duration, obs)``.
        """
        _predict(self._require_horizon(), noise_dist, prediction)

    def __call__(self, covariates: Array, data: Array | None = None) -> None:
        """Run the model as a NumPyro model function.

        Parameters
        ----------
        covariates
            Covariates with time at axis ``-2`` spanning the full horizon.
        data
            Observed data with time at axis ``-2`` (``None`` for prior sampling).
        """
        horizon = Horizon.from_data(covariates, data)
        self._horizon = horizon
        try:
            self.model(horizon.zero_data, covariates)
        finally:
            self._horizon = None


class _BaseForecaster(abc.ABC):
    """Shared forecasting logic over a fitted posterior."""

    def __init__(self, model: ForecastModel) -> None:
        self.model = model

    @abc.abstractmethod
    def _draw_posterior(self, rng_key: Array, num_samples: int) -> dict[str, Array]:
        """Return ``num_samples`` posterior draws of the latent sites."""
        raise NotImplementedError

    def __call__(
        self,
        rng_key: Array,
        data: Array,
        covariates: Array,
        num_samples: int,
        *,
        batch_size: int | None = None,
    ) -> Float[Array, " sample *batch future obs"]:
        """Sample forecasts for the steps in ``[t, duration)``.

        Parameters
        ----------
        rng_key
            PRNG key.
        data
            Observed data with time at axis ``-2`` and length ``t``.
        covariates
            Covariates with time at axis ``-2`` and length ``duration > t``.
        num_samples
            Number of forecast samples to draw.
        batch_size
            Optional chunk size for sampling (caps peak memory).

        Returns
        -------
        Float[Array, " sample *batch future obs"]
            Forecast samples over the ``future = duration - t`` horizon.
        """
        if data.shape[-2] >= covariates.shape[-2]:
            msg = "covariates must extend beyond data along the time axis"
            raise ValueError(msg)
        if num_samples <= 0:
            msg = "num_samples must be positive"
            raise ValueError(msg)
        key_post, key_pred = random.split(rng_key)
        posterior = self._draw_posterior(key_post, num_samples)
        return _forecast(key_pred, self.model, posterior, data, covariates, batch_size=batch_size)


class Forecaster(_BaseForecaster):
    """Fit a forecasting model with stochastic variational inference.

    Parameters
    ----------
    rng_key
        PRNG key for inference.
    model
        The forecasting model to fit (OOP instance or functional model).
    data
        In-sample data with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2`` and the same duration as ``data``.
    guide
        Variational guide; defaults to ``AutoNormal(model)``.
    optim
        NumPyro optimizer; defaults to ``Adam(0.01)``.
    num_steps
        Number of SVI steps.
    num_particles
        Number of ELBO particles.
    progress_bar
        Whether to display the SVI progress bar.
    """

    def __init__(
        self,
        rng_key: Array,
        model: ForecastModel,
        data: Array,
        covariates: Array,
        *,
        guide: AutoGuide | None = None,
        optim: _NumPyroOptim | None = None,
        num_steps: int = 1_001,
        num_particles: int = 1,
        progress_bar: bool = False,
    ) -> None:
        super().__init__(model)
        self._fit = fit_svi(
            rng_key,
            model,
            data,
            covariates,
            guide=guide,
            optim=optim,
            num_steps=num_steps,
            num_particles=num_particles,
            progress_bar=progress_bar,
        )
        self.guide: AutoGuide = self._fit.guide
        self.params: dict[str, Array] = self._fit.params
        self.losses: Array = self._fit.losses

    def _draw_posterior(self, rng_key: Array, num_samples: int) -> dict[str, Array]:
        return draw_posterior(rng_key, self._fit, num_samples)


class HMCForecaster(_BaseForecaster):
    """Fit a forecasting model with NUTS (Hamiltonian Monte Carlo).

    Parameters
    ----------
    rng_key
        PRNG key for inference.
    model
        The forecasting model to fit (OOP instance or functional model).
    data
        In-sample data with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2`` and the same duration as ``data``.
    num_warmup
        Number of warmup steps.
    num_samples
        Number of posterior samples.
    num_chains
        Number of MCMC chains.
    progress_bar
        Whether to display the MCMC progress bar.
    """

    def __init__(
        self,
        rng_key: Array,
        model: ForecastModel,
        data: Array,
        covariates: Array,
        *,
        num_warmup: int = 1_000,
        num_samples: int = 1_000,
        num_chains: int = 1,
        progress_bar: bool = False,
    ) -> None:
        super().__init__(model)
        self._fit = fit_mcmc(
            rng_key,
            model,
            data,
            covariates,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
        )
        self.posterior_samples: dict[str, Array] = self._fit.samples

    def _draw_posterior(self, rng_key: Array, num_samples: int) -> dict[str, Array]:
        return draw_posterior(rng_key, self._fit, num_samples)
