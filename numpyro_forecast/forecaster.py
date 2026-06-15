"""Forecasting model base class and SVI/MCMC forecasters.

This is the JAX/NumPyro port of Pyro's ``pyro.contrib.forecast.forecaster``.
The forecast horizon is handled with separate in-sample / ``_future`` latent
sites (see :meth:`ForecastingModel.time_series`) so the variational guide is
never resized and ``Predictive`` draws the forecast suffix from the prior.
"""

import abc
from collections.abc import Callable
from contextlib import ExitStack
from typing import cast

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from jax.tree_util import tree_map
from jaxtyping import Float
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoGuide, AutoNormal
from numpyro.infer.reparam import Reparam
from numpyro.optim import Adam, _NumPyroOptim

from numpyro_forecast.typing import Array
from numpyro_forecast.util import (
    concat_future,
    prefix_condition,
    shift_loc,
    slice_time,
    zero_data_like,
)


class ForecastingModel(abc.ABC):
    """Abstract base class for forecasting models.

    Subclasses implement :meth:`model`, which must call :meth:`predict` exactly
    once. The instance itself is the (pure) NumPyro model function with signature
    ``model_instance(covariates, data=None)``: the forecast horizon is inferred
    from the shapes (``future = covariates.shape[-2] - data.shape[-2]``).
    """

    def __init__(self) -> None:
        self._data: Array | None = None
        self._duration: int = -1
        self._t_obs: int = -1
        self._future: int = -1

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

    def _assert_running(self) -> None:
        if self._duration < 0:
            msg = "model state is only available during a model call"
            raise RuntimeError(msg)

    @property
    def duration(self) -> int:
        """Total horizon length ``t + future`` (in time steps)."""
        self._assert_running()
        return self._duration

    @property
    def t_obs(self) -> int:
        """Number of observed (in-sample) time steps ``t``."""
        self._assert_running()
        return self._t_obs

    @property
    def future(self) -> int:
        """Number of forecast time steps ``f`` (``0`` while training)."""
        self._assert_running()
        return self._future

    def time_series(
        self,
        name: str,
        dist_fn: Callable[[], dist.Distribution],
        *,
        reparam: Reparam | None = None,
    ) -> Array:
        """Sample a time-varying latent over the full horizon.

        The in-sample portion is sampled under ``plate("time", t)`` with the
        fixed site ``name``; when forecasting, the horizon portion is sampled
        under a separate site ``f"{name}_future"`` and concatenated. The
        separate site keeps the guide shape fixed and lets ``Predictive`` draw
        the forecast suffix from the prior.

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
        prefix = self._sample_time_block(name, self.t_obs, "time", dist_fn, reparam)
        if self.future <= 0:
            return prefix
        suffix = self._sample_time_block(
            f"{name}_future", self.future, "time_future", dist_fn, reparam
        )
        return concat_future(prefix, suffix, axis=-2)

    def _sample_time_block(
        self,
        site: str,
        size: int,
        plate_name: str,
        dist_fn: Callable[[], dist.Distribution],
        reparam: Reparam | None,
    ) -> Array:
        with ExitStack() as stack:
            if reparam is not None:
                stack.enter_context(numpyro.handlers.reparam(config={site: reparam}))
            stack.enter_context(numpyro.plate(plate_name, size, dim=-2))
            return cast(Array, numpyro.sample(site, dist_fn()))

    def predict(self, noise_dist: dist.Distribution, prediction: Array) -> None:
        """Register the observation/forecast sites for the model.

        ``noise_dist`` is a zero-centered observation noise distribution and
        ``prediction`` the deterministic mean over the full horizon. While
        training the residual is observed; while forecasting the in-sample
        prefix is observed and the forecast suffix is sampled and exposed as the
        ``"forecast"`` deterministic site.

        Parameters
        ----------
        noise_dist
            Zero-centered observation noise (e.g. ``Normal(0, sigma)``).
        prediction
            Deterministic mean with time at axis ``-2``, shape
            ``(*batch, duration, obs)``.
        """
        self._assert_running()
        obs_dist = shift_loc(noise_dist, prediction)
        if self._future == 0:
            numpyro.sample("obs", obs_dist, obs=self._data)
            return
        data = self._data
        if data is None:  # pragma: no cover - guarded by shape logic
            msg = "forecasting requires observed data"
            raise RuntimeError(msg)
        prefix = slice_time(obs_dist, slice(None, self._t_obs))
        numpyro.sample("obs", prefix, obs=data)
        forecast = numpyro.sample("obs_future", prefix_condition(obs_dist, data))
        numpyro.deterministic("forecast", forecast)

    def __call__(self, covariates: Array, data: Array | None = None) -> None:
        """Run the model as a NumPyro model function.

        Parameters
        ----------
        covariates
            Covariates with time at axis ``-2`` spanning the full horizon.
        data
            Observed data with time at axis ``-2`` (``None`` for prior sampling).
        """
        duration = covariates.shape[-2]
        t_obs = duration if data is None else data.shape[-2]
        if t_obs > duration:
            msg = "data must not be longer than covariates along the time axis"
            raise ValueError(msg)
        self._duration = duration
        self._t_obs = t_obs
        self._future = duration - t_obs
        self._data = data
        try:
            zero_data = None if data is None else zero_data_like(data, covariates)
            self.model(zero_data, covariates)
        finally:
            self._data = None
            self._duration = self._t_obs = self._future = -1


def _index_tree(tree: dict[str, Array], index: Array | slice) -> dict[str, Array]:
    """Index every leaf of a posterior-sample pytree along its sample axis."""
    return tree_map(lambda leaf: leaf[index], tree)


class _BaseForecaster(abc.ABC):
    """Shared forecasting logic over a fitted posterior."""

    def __init__(self, model: ForecastingModel) -> None:
        self.model = model

    @abc.abstractmethod
    def _draw_posterior(self, num_samples: int, rng_key: Array) -> dict[str, Array]:
        """Return ``num_samples`` posterior draws of the latent sites."""
        raise NotImplementedError

    def __call__(
        self,
        data: Array,
        covariates: Array,
        num_samples: int,
        *,
        rng_key: Array,
        batch_size: int | None = None,
    ) -> Float[Array, " sample *batch future obs"]:
        """Sample forecasts for the steps in ``[t, duration)``.

        Parameters
        ----------
        data
            Observed data with time at axis ``-2`` and length ``t``.
        covariates
            Covariates with time at axis ``-2`` and length ``duration > t``.
        num_samples
            Number of forecast samples to draw.
        rng_key
            PRNG key.
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
        posterior = self._draw_posterior(num_samples, key_post)
        return self._forecast(posterior, data, covariates, num_samples, key_pred, batch_size)

    def _forecast(
        self,
        posterior: dict[str, Array],
        data: Array,
        covariates: Array,
        num_samples: int,
        rng_key: Array,
        batch_size: int | None,
    ) -> Array:
        if batch_size is None or batch_size >= num_samples:
            return self._predict(posterior, data, covariates, rng_key)
        outputs = []
        for start in range(0, num_samples, batch_size):
            stop = min(start + batch_size, num_samples)
            rng_key, sub_key = random.split(rng_key)
            chunk = _index_tree(posterior, slice(start, stop))
            outputs.append(self._predict(chunk, data, covariates, sub_key))
        return jnp.concatenate(outputs, axis=0)

    def _predict(
        self,
        posterior: dict[str, Array],
        data: Array,
        covariates: Array,
        rng_key: Array,
    ) -> Array:
        predictive = Predictive(self.model, posterior_samples=posterior, return_sites=["forecast"])
        return predictive(rng_key, covariates, data)["forecast"]


class Forecaster(_BaseForecaster):
    """Fit a forecasting model with stochastic variational inference.

    Parameters
    ----------
    model
        The forecasting model to fit.
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
    rng_key
        PRNG key for inference.
    progress_bar
        Whether to display the SVI progress bar.
    """

    def __init__(
        self,
        model: ForecastingModel,
        data: Array,
        covariates: Array,
        *,
        guide: AutoGuide | None = None,
        optim: _NumPyroOptim | None = None,
        num_steps: int = 1001,
        num_particles: int = 1,
        rng_key: Array,
        progress_bar: bool = False,
    ) -> None:
        if data.shape[-2] != covariates.shape[-2]:
            msg = "fit expects data and covariates of equal duration"
            raise ValueError(msg)
        super().__init__(model)
        self.guide = AutoNormal(model) if guide is None else guide
        optimizer = Adam(0.01) if optim is None else optim
        svi = SVI(model, self.guide, optimizer, Trace_ELBO(num_particles=num_particles))
        result = svi.run(rng_key, num_steps, covariates, data, progress_bar=progress_bar)
        self.params: dict[str, Array] = result.params
        self.losses: Array = result.losses

    def _draw_posterior(self, num_samples: int, rng_key: Array) -> dict[str, Array]:
        return self.guide.sample_posterior(rng_key, self.params, sample_shape=(num_samples,))


class HMCForecaster(_BaseForecaster):
    """Fit a forecasting model with NUTS (Hamiltonian Monte Carlo).

    Parameters
    ----------
    model
        The forecasting model to fit.
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
    rng_key
        PRNG key for inference.
    progress_bar
        Whether to display the MCMC progress bar.
    """

    def __init__(
        self,
        model: ForecastingModel,
        data: Array,
        covariates: Array,
        *,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 1,
        rng_key: Array,
        progress_bar: bool = False,
    ) -> None:
        if data.shape[-2] != covariates.shape[-2]:
            msg = "fit expects data and covariates of equal duration"
            raise ValueError(msg)
        super().__init__(model)
        mcmc = MCMC(
            NUTS(model),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
        )
        mcmc.run(rng_key, covariates, data)
        self.posterior_samples: dict[str, Array] = mcmc.get_samples()

    def _draw_posterior(self, num_samples: int, rng_key: Array) -> dict[str, Array]:
        leaves = list(self.posterior_samples.values())
        available = leaves[0].shape[0]
        indices = random.choice(rng_key, available, shape=(num_samples,), replace=True)
        return _index_tree(self.posterior_samples, indices)
