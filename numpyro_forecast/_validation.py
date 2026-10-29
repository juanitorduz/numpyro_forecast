"""Shared argument-validation helpers for the functional API submodules."""

from collections.abc import Mapping

import numpy as np
from jax.typing import ArrayLike

from numpyro_forecast.typing import Array


def _require_positive_num_samples(num_samples: int) -> None:
    """Raise ``ValueError`` if ``num_samples`` is not positive."""
    if num_samples <= 0:
        msg = "num_samples must be positive"
        raise ValueError(msg)


def _require_covariates_extend_data(data: Array, covariates: Array) -> None:
    """Raise ``ValueError`` unless ``covariates`` is longer than ``data`` in time."""
    _require_time_axis("data", data)
    _require_time_axis("covariates", covariates)
    if data.shape[-2] >= covariates.shape[-2]:
        msg = "covariates must extend beyond data along the time axis"
        raise ValueError(msg)


def _require_covariates_cover_data(data: Array, covariates: Array) -> None:
    """Raise ``ValueError`` unless ``covariates`` is at least as long as ``data`` in time."""
    _require_time_axis("data", data)
    _require_time_axis("covariates", covariates)
    if covariates.shape[-2] < data.shape[-2]:
        msg = (
            f"covariates cover {covariates.shape[-2]} time steps but data has "
            f"{data.shape[-2]}; covariates must cover at least the in-sample window"
        )
        raise ValueError(msg)


def _require_time_axis(name: str, x: Array | np.ndarray) -> None:
    """Raise ``ValueError`` unless ``x`` has a time axis at ``-2`` and a last axis at ``-1``."""
    if x.ndim >= 2:
        return
    is_covariates = name == "covariates"
    last_axis = "the covariate dim" if is_covariates else "the observation dim"
    layout = "(..., time, k), k may be 0" if is_covariates else "(..., time, obs)"
    if x.ndim == 1:
        hint = "Add the last axis with x[..., None] if the array is one series over time " + (
            "(a model without covariates takes jnp.zeros((duration, 0)))"
            if is_covariates
            else "(a univariate series has shape (time, 1))"
        )
    else:
        hint = "A 0-d value has no time axis: pass " + (
            "a (time, k) array, for example jnp.zeros((duration, 0)) for a model "
            "without covariates"
            if is_covariates
            else "the series as a (time, obs) array (a univariate series has shape (time, 1))"
        )
    msg = (
        f"{name} must have time at axis -2 and {last_axis} at axis -1 "
        f"(shape {layout}); got shape {tuple(x.shape)}. {hint}"
    )
    raise ValueError(msg)


def _require_sample_axis(posterior: Mapping[str, ArrayLike]) -> None:
    """Raise ``ValueError`` naming the ``posterior`` leaves that have no leading sample axis."""
    missing = [name for name, leaf in posterior.items() if np.ndim(leaf) == 0]
    if missing:
        msg = (
            f"posterior sites {missing} have no sample axis (0-d leaves); every leaf needs a "
            "leading sample axis, as returned by draw_posterior() or mcmc.get_samples(). "
            "Repeat a point estimate x over the draws with "
            "jnp.broadcast_to(x, (num_samples, *jnp.shape(x)))"
        )
        raise ValueError(msg)
