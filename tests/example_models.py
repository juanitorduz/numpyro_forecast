"""Validated example forecasting models, kept for end-to-end tests.

Reference models for `tests.test_examples`: plain functional models (a pure
``(covariates, data=None)`` callable opening with ``Horizon.from_data``) that
mirror the example notebooks under ``docs/examples/`` and act as a regression
target for the full fit-draw-forecast path.
"""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import scope
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast.arrays import pad_future
from numpyro_forecast.features import periodic_repeat
from numpyro_forecast.models import Horizon, SSOEResult, innovations, predict, ssoe
from numpyro_forecast.typing import Array, ForecastModel


def univariate_model(covariates: Array, data: Array | None = None) -> None:
    """Local level + regression model with Student-T observations.

    The mean is ``bias + level_t + weight @ covariates_t`` where ``level`` is a
    Gaussian random walk (``LocScaleReparam`` improves the SVI geometry). The
    regression design ``covariates`` is supplied by the caller (e.g. Fourier
    features from `numpyro_forecast.features.fourier_features()`).

    Parameters
    ----------
    covariates
        Regression design with time at axis ``-2``, shape
        ``(duration, num_features)``.
    data
        Observed data with time at axis ``-2`` (``None`` for prior sampling).
    """
    h = Horizon.from_data(covariates, data)
    num_features = covariates.shape[-1]

    bias = numpyro.sample("bias", dist.Normal(0.0, 10.0))
    weight = numpyro.sample("weight", dist.Normal(0.0, 0.1).expand([num_features]).to_event(1))
    drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-20.0, 5.0))
    nu = numpyro.sample("nu", dist.Gamma(10.0, 2.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-5.0, 5.0))
    centered = numpyro.sample("centered", dist.Uniform(0.0, 1.0))

    drift = innovations(
        h,
        "drift",
        lambda: dist.Normal(0.0, drift_scale),
        reparam=LocScaleReparam(centered=centered),
    )
    # Cumulative sum over time is the random-walk level (= the tutorials' scan).
    level = jnp.cumsum(drift, axis=-2)
    regression = (weight * covariates).sum(axis=-1, keepdims=True)
    prediction = level + bias + regression

    predict(h, dist.StudentT(df=nu, loc=0.0, scale=sigma), prediction)


def make_hierarchical_model(period: int = 24 * 7) -> ForecastModel:
    """Build a hierarchical OD model with per-station seasonality and drift.

    Data and covariates use the ``(origin, time, destin)`` layout. ``covariates``
    is only used for its shape (a dummy zero panel, as in the tutorial).

    Parameters
    ----------
    period
        Seasonal period in time steps (default ``24 * 7`` for hourly data).

    Returns
    -------
    ForecastModel
        A callable ``(covariates, data=None) -> None`` closed over ``period``.
    """

    def hierarchical_model(covariates: Array, data: Array | None = None) -> None:
        """Define the hierarchical forecasting model.

        Parameters
        ----------
        covariates
            Dummy covariates shaped ``(origin, duration, destin)``.
        data
            Observed data with time at axis ``-2`` (``None`` for prior sampling).
        """
        h = Horizon.from_data(covariates, data)
        n_origin = covariates.shape[-3]
        n_destin = covariates.shape[-1]
        duration = covariates.shape[-2]

        origin_plate = numpyro.plate("origin", n_origin, dim=-3)
        destin_plate = numpyro.plate("destin", n_destin, dim=-1)
        hour_plate = numpyro.plate("hour_of_week", period, dim=-2)

        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-20.0, 5.0))
        destin_centered = numpyro.sample("destin_centered", dist.Uniform(0.0, 1.0))

        with origin_plate, hour_plate:
            origin_seasonal = numpyro.sample("origin_seasonal", dist.Normal(0.0, 5.0))
        with hour_plate, destin_plate:
            destin_seasonal = numpyro.sample("destin_seasonal", dist.Normal(0.0, 5.0))

        with destin_plate:
            drift = innovations(
                h,
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

        predict(h, dist.Normal(0.0, scale), prediction)

    return hierarchical_model


def _level_channel(h: Horizon, name: str, values: Array, gate: Array) -> tuple[SSOEResult, Array]:
    """Gated simple exponential smoothing level channel on `ssoe()`.

    Samples the channel priors (``smoothing``, ``init``, ``noise``) and runs the
    where-gated level recursion; the level updates only where ``gate`` is true
    and is frozen over the forecast horizon (``pad_future`` zeroes the gate
    there), so the forecast is the last level plus iid errors. Meant to be
    called under `numpyro.handlers.scope()`, which prefixes the parameter
    sites and the block's ``f"{name}_future"`` site per channel.

    Parameters
    ----------
    h
        The train/forecast horizon for the current model call.
    name
        Error-site name handed to `ssoe()` (``"eps"`` under a scope).
    values
        Driving series ``(t_obs, 1)``; read only where ``gate`` is true.
    gate
        Boolean update gate ``(t_obs, 1)``.

    Returns
    -------
    tuple[SSOEResult, Array]
        The block result (in-sample means, forecast means, sampled future
        values) and the observation noise scale.
    """
    smoothing = jnp.asarray(numpyro.sample("smoothing", dist.Beta(2.0, 20.0)))
    init = jnp.asarray(numpyro.sample("init", dist.Normal(0.0, 1.0)))
    noise = jnp.asarray(numpyro.sample("noise", dist.HalfNormal(1.0)))

    def step(level: Array, gate_t: Array | None) -> tuple[Array, Callable[[Array, Array], Array]]:
        assert gate_t is not None  # xs is always passed here

        def carry_fn(y_t: Array, _: Array) -> Array:
            return jnp.where(gate_t, smoothing * y_t + (1.0 - smoothing) * level, level)

        return level, carry_fn

    result = ssoe(
        h, name, values, init[None], step, dist.Normal(0.0, noise), xs=pad_future(gate, h.future)
    )
    return result, noise


def croston_model(covariates: Array, data: Array | None = None) -> None:
    """Croston's method as two scoped level channels composed on `ssoe()`.

    The observed demand series doubles as the covariate: ``covariates`` is the
    series over the full horizon with shape ``(duration, 1)`` and only its first
    ``h.t_obs`` rows are read (the block's ``t_obs`` check is the leak guard).
    The demand-size channel ``z`` smooths the demand at demand events; the
    inverse-interval channel ``p_inv`` smooths ``1 / interval`` at the same
    events. The likelihoods and the ``"forecast"`` product are registered
    outside the scopes, from the channels' results.

    Parameters
    ----------
    covariates
        The demand series over the full horizon, shape ``(duration, 1)``.
    data
        Observed demand with time at axis ``-2`` (``None`` for prior sampling
        and under ``predict_in_sample``).
    """
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :]
    is_demand = y > 0
    idx = jnp.arange(h.t_obs)[:, None]
    last_at_or_before = jax.lax.cummax(jnp.where(is_demand, idx, -1), axis=0)
    last_before = jnp.concatenate([jnp.full((1, 1), -1), last_at_or_before[:-1]])
    p_inv_obs = 1.0 / (idx - last_before).astype(y.dtype)

    z, z_noise = scope(_level_channel, "z", divider="_")(h, "eps", y, is_demand)
    p_inv, p_inv_noise = scope(_level_channel, "p_inv", divider="_")(
        h, "eps", p_inv_obs, is_demand
    )

    numpyro.deterministic("rate", z.mu * p_inv.mu)
    numpyro.sample("obs", dist.Normal(z.mu, z_noise).mask(is_demand), obs=h.data)
    numpyro.sample(
        "obs_intervals", dist.Normal(p_inv.mu, p_inv_noise).mask(is_demand), obs=p_inv_obs
    )
    if h.future > 0:
        numpyro.deterministic("rate_future", z.mu_future * p_inv.mu_future)
        numpyro.deterministic("forecast", z.y_future * p_inv.y_future)
