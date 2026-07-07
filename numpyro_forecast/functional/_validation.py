"""Shared argument-validation helpers for the functional API submodules."""

from numpyro_forecast.typing import Array


def _require_positive_num_samples(num_samples: int) -> None:
    """Raise ``ValueError`` if ``num_samples`` is not positive."""
    if num_samples <= 0:
        msg = "num_samples must be positive"
        raise ValueError(msg)


def _require_equal_duration(data: Array, covariates: Array) -> None:
    """Raise ``ValueError`` if ``data`` and ``covariates`` differ in duration."""
    if data.shape[-2] != covariates.shape[-2]:
        msg = "fit expects data and covariates of equal duration"
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
