"""Shared argument-validation helpers for the functional API submodules."""

from collections.abc import Mapping

import numpy as np
from jax.typing import ArrayLike

from numpyro_forecast.typing import Array


def require_positive_num_samples(num_samples: int) -> None:
    """Raise ``ValueError`` if ``num_samples`` is not positive."""
    if num_samples <= 0:
        msg = "num_samples must be positive"
        raise ValueError(msg)


def require_covariates_extend_data(data: Array, covariates: Array) -> None:
    """Raise ``ValueError`` unless ``covariates`` is longer than ``data`` in time."""
    require_time_axis("data", data)
    require_time_axis("covariates", covariates)
    if data.shape[-2] >= covariates.shape[-2]:
        msg = "covariates must extend beyond data along the time axis"
        raise ValueError(msg)


def require_covariates_cover_data(data: Array, covariates: Array) -> None:
    """Raise ``ValueError`` unless ``data`` is non-empty and ``covariates`` cover it in time."""
    require_time_axis("data", data)
    require_time_axis("covariates", covariates)
    if data.shape[-2] == 0:
        msg = (
            f"data has no observed time steps (shape {tuple(data.shape)}); the in-sample "
            "window needs at least one step"
        )
        raise ValueError(msg)
    if covariates.shape[-2] < data.shape[-2]:
        msg = (
            f"covariates cover {covariates.shape[-2]} time steps but data has "
            f"{data.shape[-2]}; covariates must cover at least the in-sample window"
        )
        raise ValueError(msg)


def require_time_axis(name: str, x: Array | np.ndarray) -> None:
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


def require_sample_axis(posterior: Mapping[str, ArrayLike]) -> None:
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


def _is_horizon_site(name: str) -> bool:
    """Whether ``name`` follows the naming of a forecast-horizon site.

    The model building blocks name their horizon sites ``f"{name}_future"``; a
    reparameterization adds a suffix (``drift_future_decentered``), a scope a
    prefix (``z/eps_future``), and `~~numpyro_forecast.models.predict()` adds
    ``obs_future`` and the ``forecast`` deterministic. An in-sample name that
    only contains the letters (``beta_futures``) does not match.
    """
    return name == "forecast" or name.endswith("_future") or "_future_" in name


def require_no_horizon_sites(posterior: Mapping[str, ArrayLike]) -> None:
    """Raise ``ValueError`` naming the ``posterior`` keys that hold forecast-horizon values.

    The predictive drivers hand ``posterior`` to ``numpyro.infer.Predictive``,
    which substitutes every sample site of the posterior by name. A
    forecast-horizon key with values (a key that ends in ``"_future"`` or
    contains ``"_future_"``, or the key ``"forecast"``) comes from a
    ``Predictive`` call over a horizon, and a posterior mixed with that output
    pins the draws to its stale values instead of drawing them from the prior,
    with no error (issue #161). ``Predictive`` does not substitute a
    deterministic site such as ``"forecast"``, but only such a call returns it
    with values, so it marks the same mistake. A horizon key with an empty
    leaf passes (see `drop_empty_horizon_sites()`).

    Parameters
    ----------
    posterior
        Posterior samples, sample axis leading.

    Raises
    ------
    ValueError
        If a forecast-horizon key has a non-empty leaf; the message lists every
        such key.
    """
    horizon = sorted(
        name for name, leaf in posterior.items() if _is_horizon_site(name) and np.size(leaf) > 0
    )
    if horizon:
        msg = (
            f"posterior holds values for the forecast-horizon sites {horizon}, which a "
            "numpyro.infer.Predictive call over a horizon returns. The drivers substitute every "
            "posterior sample site by name, so a posterior mixed with such an output pins the "
            "draws to its stale values instead of drawing them from the prior. Pass the "
            "posterior of the training run: mcmc.get_samples() or draw_posterior() of a fit "
            "whose covariates are no longer than data (h.future == 0). Keys that end in "
            "'_future' or contain '_future_', and the key 'forecast', count as forecast-horizon "
            "sites."
        )
        raise ValueError(msg)


def drop_empty_horizon_sites(posterior: Mapping[str, ArrayLike]) -> dict[str, ArrayLike]:
    """Return ``posterior`` without its forecast-horizon keys, which must be empty.

    `require_no_horizon_sites()` rejects a horizon key with values, so every
    horizon key that remains has an empty leaf: a site that the model
    registers on every call (for example ``numpyro.deterministic("forecast",
    r.y_future)`` without the ``h.future > 0`` guard), which a training run
    records with a size-0 time axis. Substituted, an empty leaf of a sample
    site puts no steps where the horizon needs ``future`` of them; dropped,
    ``Predictive`` draws (or, for a deterministic site, computes) the site, as
    for a model that registers it only when ``h.future > 0``. An empty leaf
    never comes from a call over a horizon, so dropping it hides no mistake.

    Parameters
    ----------
    posterior
        Posterior samples, sample axis leading.

    Returns
    -------
    dict[str, ArrayLike]
        ``posterior`` without its forecast-horizon keys.

    Raises
    ------
    ValueError
        If a forecast-horizon key has a non-empty leaf (see
        `require_no_horizon_sites()`).
    """
    require_no_horizon_sites(posterior)
    return {name: leaf for name, leaf in posterior.items() if not _is_horizon_site(name)}


def drop_obs_site(posterior: Mapping[str, ArrayLike]) -> dict[str, ArrayLike]:
    """Return ``posterior`` without its ``"obs"`` key.

    ``numpyro.infer.Predictive`` substitutes every posterior sample site by
    name, so an ``"obs"`` key (the output of a ``Predictive`` call with
    ``data``, merged into the posterior) would make every in-sample draw equal
    to it (issue #260). A prior predictive dict carries ``"obs"`` too: without
    the key, ``obs`` is drawn again from the same latents.

    Parameters
    ----------
    posterior
        Posterior samples, sample axis leading.

    Returns
    -------
    dict[str, ArrayLike]
        ``posterior`` without the ``"obs"`` key.

    Raises
    ------
    ValueError
        If ``"obs"`` is the only key, so no latent site is left to draw from.
    """
    kept = {name: leaf for name, leaf in posterior.items() if name != "obs"}
    if not kept and "obs" in posterior:
        msg = (
            "posterior holds only the 'obs' site, which the drivers drop because it would pin "
            "every in-sample draw to its values. Pass the posterior of the training run: "
            "mcmc.get_samples() or draw_posterior(), which hold the latent sites."
        )
        raise ValueError(msg)
    return kept


def require_pred_truth_shapes(pred: ArrayLike, truth: ArrayLike) -> None:
    """Raise when ``truth.shape`` is not ``pred.shape[1:]``.

    Every metric scores ``pred`` of shape ``(sample, *batch)`` against
    ``truth`` of shape ``(*batch)``. The jaxtyping annotations check this only
    while the runtime type hook is on; with ``JAXTYPING_DISABLE=1`` a
    mismatched ``truth`` (for example ``(f,)`` against ``(sample, f, 1)``)
    broadcasts into a wrong grid and returns a silently wrong value. The check
    reads static shapes only, so it costs nothing and also works on tracers
    under ``jit`` and ``vmap``.

    Parameters
    ----------
    pred
        Forecast samples with the sample axis first.
    truth
        Ground-truth values.

    Raises
    ------
    ValueError
        If ``pred`` has no sample axis or ``truth.shape != pred.shape[1:]``.
    """
    pred_shape, truth_shape = np.shape(pred), np.shape(truth)
    if not pred_shape:
        msg = (
            "pred must have the sample axis first, shape (sample, *batch), got a scalar; "
            "pass the forecast draws, not a point forecast"
        )
        raise ValueError(msg)
    if pred_shape[1:] != truth_shape:
        msg = (
            f"truth must have shape pred.shape[1:] = {pred_shape[1:]}, got {truth_shape}. "
            "pred has shape (sample, *batch) and truth has shape (*batch): remove only "
            "the sample axis, and restore any axis dropped from truth (for example "
            "truth[..., None] for a missing obs axis)"
        )
        raise ValueError(msg)
