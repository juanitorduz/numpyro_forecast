"""Posterior sampling from a fitted guide for the functional API.

:func:`draw_posterior` draws posterior samples of the latent sites from a fitted
variational guide (an :class:`~numpyro.infer.autoguide.AutoGuide`) and its learned
parameters, as returned by ``AutoGuide``/``SVI.run`` (``guide``/``state.params``).
It is guide-only on purpose: MCMC users already hold their posterior samples via
``mcmc.get_samples()``, and hand-written-guide users draw with a single
``numpyro.infer.Predictive(guide, params=params, num_samples=n)(rng_key,
covariates, data)`` call; neither needs this function. The blackjax Pathfinder
backend has its own analogous entry point,
:func:`~numpyro_forecast.contrib.blackjax.pathfinder_samples`, built on the same
shared chunk-and-transfer loop, :func:`~numpyro_forecast.functional._offload._draw_chunked`.
"""

from collections.abc import Callable, Mapping
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from numpyro.infer.autoguide import AutoDelta, AutoGuide

from numpyro_forecast.functional._offload import _draw_chunked
from numpyro_forecast.functional._validation import _require_positive_num_samples
from numpyro_forecast.typing import Array


def _ensure_sample_axis_for_delta(samples: dict[str, Array], num_samples: int) -> dict[str, Array]:
    """Tile ``AutoDelta`` point estimates to a leading sample axis.

    Called only when the guide is an ``AutoDelta`` (guide-type dispatch, never
    shape inspection): every leaf is broadcast to ``(num_samples, *leaf.shape)``
    unconditionally. For all other Auto guides ``sample_posterior`` already
    returns the axis and this is never invoked.

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


@lru_cache(maxsize=8)
def _jitted_sample_posterior(guide: AutoGuide) -> Callable[..., dict[str, Array]]:
    """Return a jitted, per-guide-cached ``sample_posterior`` with static ``sample_shape``.

    Eagerly, ``sample_posterior`` materializes every latent and deterministic
    site (and their intermediates) at full sample size with no buffer planning,
    which is what blows accelerator memory on wide panels. Under ``jax.jit``
    XLA schedules and reuses those buffers, and caching per guide instance
    means every chunk of a :func:`draw_posterior` call (and repeated calls
    with the same fit and sample count) shares one compiled executable, the
    same single-compile discipline as the predictive drivers.

    Parameters
    ----------
    guide
        The fitted autoguide whose bound ``sample_posterior`` is wrapped.

    Returns
    -------
    Callable[..., dict[str, Array]]
        The jitted ``sample_posterior``; call it exactly like the eager bound
        method (``sample_shape`` must be passed by keyword).
    """
    return jax.jit(guide.sample_posterior, static_argnames=("sample_shape",))


def draw_posterior(
    rng_key: Array,
    guide: AutoGuide,
    params: dict[str, Array],
    num_samples: int,
    *,
    batch_size: int | None = None,
    device: jax.Device | str | None = None,
) -> dict[str, Array | np.ndarray]:
    """Draw ``num_samples`` posterior samples of the latent sites from a fitted guide.

    The returned dict has the sample axis leading and is ready to pass to
    :func:`~numpyro_forecast.functional.prediction.forecast` or NumPyro's ``Predictive``.
    An ``AutoDelta`` guide is a MAP point estimate: it is drawn once and tiled to
    ``num_samples`` (:func:`_ensure_sample_axis_for_delta`), since it carries no
    posterior spread of its own. Every other ``AutoGuide`` is sampled through a
    jitted, per-guide-cached ``sample_posterior`` (:func:`_jitted_sample_posterior`).

    Parameters
    ----------
    rng_key
        PRNG key.
    guide
        The fitted variational guide, e.g. the ``AutoGuide`` instance passed to
        ``SVI``.
    params
        The learned variational parameters, e.g. the trained parameters from
        ``svi.run``'s result.
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
        are reproducible per ``(rng_key, batch_size)``.
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
    ValueError
        If ``num_samples`` or ``batch_size`` is not positive.

    Notes
    -----
    For an MCMC fit, use its samples directly (``mcmc.get_samples()``); this
    function draws afresh from a variational guide, and chunks drawn from
    independent subkeys remain valid i.i.d. posterior samples.
    """
    _require_positive_num_samples(num_samples)

    def draw_fn(key: Array, n: int) -> dict[str, Array]:
        if isinstance(guide, AutoDelta):
            point = guide.sample_posterior(key, params)
            return _ensure_sample_axis_for_delta(point, n)
        sample = _jitted_sample_posterior(guide)
        return sample(key, params, sample_shape=(n,))

    return _draw_chunked(
        rng_key,
        draw_fn,
        num_samples,
        batch_size=batch_size,
        device=device,
        stage="posterior drawing",
    )
