"""Posterior sampling from fit results for the functional API.

:func:`draw_posterior` draws posterior samples of the latent sites from a fit
result. It dispatches on the fit type via the private singledispatch generic
``_draw_posterior_impl``, on which both in-package fit types
(:class:`~numpyro_forecast.functional.svi.SVIFit`,
:class:`~numpyro_forecast.functional.mcmc.MCMCFit`) and extensions (e.g. the
blackjax Pathfinder backend) register implementations.
"""

from functools import singledispatch

import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoDelta, AutoGuide

from numpyro_forecast.exceptions import GuideSampleArgsError
from numpyro_forecast.functional._validation import _require_positive_num_samples
from numpyro_forecast.functional.mcmc import MCMCFit
from numpyro_forecast.functional.svi import SVIFit
from numpyro_forecast.typing import Array


def _ensure_sample_axis_for_delta(samples: dict[str, Array], num_samples: int) -> dict[str, Array]:
    """Tile ``AutoDelta`` point estimates to a leading sample axis.

    Called only when the fit's guide is an ``AutoDelta`` (guide-type dispatch,
    never shape inspection): every leaf is broadcast to
    ``(num_samples, *leaf.shape)`` unconditionally. For all other Auto guides
    ``sample_posterior`` already returns the axis and this is never invoked.

    Parameters
    ----------
    samples
        The MAP point estimate, one leaf per latent site.
    num_samples
        The leading sample-axis size to tile to.

    Returns
    -------
    dict[str, Array]
        Each leaf broadcast to ``(num_samples, *leaf.shape)``.
    """
    return {
        name: jnp.broadcast_to(leaf, (num_samples, *jnp.shape(leaf)))
        for name, leaf in samples.items()
    }


def _index_tree(tree: dict[str, Array], index: Array | slice) -> dict[str, Array]:
    """Index every leaf of a posterior-sample pytree along its sample axis."""
    return tree_map(lambda leaf: leaf[index], tree)


@singledispatch
def _draw_posterior_impl(fit: object, num_samples: int, rng_key: Array) -> dict[str, Array]:
    """Dispatch on the fit type to draw posterior samples (see :func:`draw_posterior`)."""
    msg = f"draw_posterior() does not support {type(fit).__name__}"
    raise NotImplementedError(msg)


@_draw_posterior_impl.register
def _(fit: SVIFit, num_samples: int, rng_key: Array) -> dict[str, Array]:
    _require_positive_num_samples(num_samples)
    if isinstance(fit.guide, AutoDelta):
        point = fit.guide.sample_posterior(rng_key, fit.params)
        return _ensure_sample_axis_for_delta(point, num_samples)
    if isinstance(fit.guide, AutoGuide):
        return fit.guide.sample_posterior(rng_key, fit.params, sample_shape=(num_samples,))
    if fit.covariates is None:
        raise GuideSampleArgsError()
    predictive = Predictive(fit.guide, params=fit.params, num_samples=num_samples)
    return predictive(rng_key, fit.covariates, fit.data)


@_draw_posterior_impl.register
def _(fit: MCMCFit, num_samples: int, rng_key: Array) -> dict[str, Array]:
    _require_positive_num_samples(num_samples)
    leaves = list(fit.samples.values())
    available = leaves[0].shape[0]
    if num_samples <= available:
        # Thin the genuine posterior draws on an evenly spaced grid: this is
        # deterministic, order-preserving, and free of the duplicate draws that
        # sampling with replacement would otherwise inject.
        indices = jnp.linspace(0, available - 1, num_samples).round().astype(int)
    else:
        # More draws requested than the chain holds: fall back to resampling
        # with replacement (the only way to grow the sample count).
        indices = random.choice(rng_key, available, shape=(num_samples,), replace=True)
    return _index_tree(fit.samples, indices)


def draw_posterior(rng_key: Array, fit: object, num_samples: int) -> dict[str, Array]:
    """Draw ``num_samples`` posterior samples of the latent sites from a fit.

    Dispatches on the fit type (e.g. :class:`~numpyro_forecast.functional.svi.SVIFit`,
    :class:`~numpyro_forecast.functional.mcmc.MCMCFit`). The returned dict has the
    sample axis leading and is ready to pass to
    :func:`~numpyro_forecast.functional.prediction.forecast` or NumPyro's ``Predictive``.

    Parameters
    ----------
    rng_key
        PRNG key.
    fit
        A fit result produced by :func:`~numpyro_forecast.functional.svi.fit_svi`
        or :func:`~numpyro_forecast.functional.mcmc.fit_mcmc`.
    num_samples
        Number of posterior draws.

    Returns
    -------
    dict[str, Array]
        Posterior samples of the latent sites, sample axis leading.

    Raises
    ------
    NotImplementedError
        If ``fit`` is of an unsupported type.
    GuideSampleArgsError
        If the fit holds a hand-written guide but was constructed without its
        in-sample covariates/data.

    Notes
    -----
    For an :class:`~numpyro_forecast.functional.mcmc.MCMCFit`, when
    ``num_samples`` does not exceed the number of draws in the chain the draws
    are thinned on an evenly spaced grid (no duplicates); only when more samples
    are requested than the chain holds are they resampled with replacement. For
    an :class:`~numpyro_forecast.functional.svi.SVIFit` the draws are sampled
    afresh from the fitted guide.
    """
    return _draw_posterior_impl(fit, num_samples, rng_key)
