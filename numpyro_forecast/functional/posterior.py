"""Posterior sampling from fit results for the functional API.

:func:`draw_posterior` draws posterior samples of the latent sites from a fit
result. It dispatches on the fit type via the private singledispatch generic
``_draw_posterior_impl``, on which both in-package fit types
(:class:`~numpyro_forecast.functional.svi.SVIFit`,
:class:`~numpyro_forecast.functional.mcmc.MCMCFit`) and extensions (e.g. the
blackjax Pathfinder backend) register implementations.
"""

from collections.abc import Mapping
from functools import singledispatch

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.tree_util import tree_map
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoDelta, AutoGuide

from numpyro_forecast.exceptions import GuideSampleArgsError
from numpyro_forecast.functional._offload import _resolve_device, _stitch_chunks, _transfer
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


def _index_tree[Leaf: Array | np.ndarray](
    tree: Mapping[str, Leaf], index: Array | slice
) -> dict[str, Leaf]:
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


def draw_posterior(
    rng_key: Array,
    fit: object,
    num_samples: int,
    *,
    batch_size: int | None = None,
    device: jax.Device | str | None = None,
) -> dict[str, Array | np.ndarray]:
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
    batch_size
        Optional chunk size for the drawing itself. Sampling a variational
        posterior materializes every latent and deterministic site for all
        draws at once, which on a wide panel is the largest allocation of the
        whole workflow. With ``batch_size`` set (strictly below
        ``num_samples``), the draws are sampled in chunks of exactly this many
        samples, each chunk is moved per ``device`` before the next is drawn,
        and the final chunk's overdraw is discarded, so accelerator memory is
        bounded by one chunk. Chunking changes the PRNG stream layout: draws
        are reproducible per ``(rng_key, batch_size)``. Ignored for
        :class:`~numpyro_forecast.functional.mcmc.MCMCFit` (see Notes).
    device
        Where each chunk of draws is moved as soon as it is drawn. ``"host"``
        copies to host memory with :func:`jax.device_get` and returns NumPy
        leaves; it needs no CPU backend, so it works even when
        ``numpyro.set_platform("cuda")`` (or ``jax_platforms``) leaves only an
        accelerator backend initialized. A :class:`jax.Device` or platform
        name like ``"cpu"`` commits the draws to that device instead
        (``"cpu"`` falls back to ``"host"`` with a :class:`UserWarning` when
        the CPU backend is not initialized). ``None`` keeps everything on the
        default device. ``device`` never changes the draw values.

    Returns
    -------
    dict[str, Array | np.ndarray]
        Posterior samples of the latent sites, sample axis leading (NumPy
        leaves when ``device`` resolves to ``"host"``).

    Raises
    ------
    NotImplementedError
        If ``fit`` is of an unsupported type.
    GuideSampleArgsError
        If the fit holds a hand-written guide but was constructed without its
        in-sample covariates/data.
    ValueError
        If ``batch_size`` is not positive.

    Notes
    -----
    For an :class:`~numpyro_forecast.functional.mcmc.MCMCFit`, when
    ``num_samples`` does not exceed the number of draws in the chain the draws
    are thinned on an evenly spaced grid (no duplicates); only when more samples
    are requested than the chain holds are they resampled with replacement.
    Because that selection is deterministic and the chain draws are already
    materialized, ``batch_size`` is ignored for MCMC fits (there is no drawing
    peak to bound and chunked selection would duplicate draws); ``device``
    still applies. For an :class:`~numpyro_forecast.functional.svi.SVIFit` (or
    a Pathfinder fit) the draws are sampled afresh from the fitted guide, and
    chunks drawn from independent subkeys remain valid i.i.d. posterior
    samples.
    """
    resolved = _resolve_device(device)
    if batch_size is not None and batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    if isinstance(fit, MCMCFit) or batch_size is None or batch_size >= num_samples:
        samples = _draw_posterior_impl(fit, num_samples, rng_key)
        return {name: _transfer(leaf, resolved) for name, leaf in samples.items()}
    _require_positive_num_samples(num_samples)
    num_chunks = -(-num_samples // batch_size)  # ceil; the last chunk overdraws
    keys = random.split(rng_key, num_chunks)
    chunks = [
        {name: _transfer(leaf, resolved) for name, leaf in draws.items()}
        for draws in (_draw_posterior_impl(fit, batch_size, key) for key in keys)
    ]
    return {
        name: _stitch_chunks([chunk[name] for chunk in chunks], num_samples, resolved)
        for name in chunks[0]
    }
