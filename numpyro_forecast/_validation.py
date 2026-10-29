"""Shared argument-validation helpers for the functional API submodules."""

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
    if data.shape[-2] >= covariates.shape[-2]:
        msg = "covariates must extend beyond data along the time axis"
        raise ValueError(msg)


def _require_covariates_cover_data(data: Array, covariates: Array) -> None:
    """Raise ``ValueError`` unless ``covariates`` is at least as long as ``data`` in time."""
    if covariates.shape[-2] < data.shape[-2]:
        msg = (
            f"covariates cover {covariates.shape[-2]} time steps but data has "
            f"{data.shape[-2]}; covariates must cover at least the in-sample window"
        )
        raise ValueError(msg)


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
