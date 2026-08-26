"""Producing draws: posterior sampling from a fitted guide and predictive sampling.

:func:`draw_posterior` draws posterior samples of the latent sites from a fitted
variational guide (an :class:`~numpyro.infer.autoguide.AutoGuide`) and its learned
parameters, as returned by ``AutoGuide``/``SVI.run`` (``guide``/``state.params``).
It is guide-only on purpose: MCMC users already hold their posterior samples via
``mcmc.get_samples()``, and hand-written-guide users draw with a single
``numpyro.infer.Predictive(guide, params=params, num_samples=n)(rng_key,
covariates, data)`` call; neither needs this function. The blackjax Pathfinder
backend has its own analogous entry point,
:func:`~numpyro_forecast.contrib.blackjax.pathfinder_samples`, built on the same
shared chunk-and-transfer loop, :func:`~numpyro_forecast._offload._draw_chunked`.

:func:`forecast` samples forecasts over the horizon from posterior draws, and
:func:`predict_in_sample` samples the in-sample posterior predictive of the
``obs`` site. Both drive a jitted ``Predictive`` wrapper through the shared
chunking driver ``_chunked_draws``, which caps peak memory while compiling the
predictive exactly once. With ``device`` set (e.g. ``"host"``: pageable host
memory, as jax Arrays on the CPU backend device or as NumPy arrays when that
backend is not initialized), every chunk is moved off the accelerator before
the next one is drawn, so accelerator memory is bounded by a single chunk
instead of the full draw array. The ``device`` contract is documented once, on
:func:`draw_posterior`; the predictive drivers share it.
"""

from collections.abc import Callable, Mapping
from functools import lru_cache, partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.tree_util import tree_map
from jax.typing import ArrayLike
from jaxtyping import Num
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoDelta, AutoGuide

from numpyro_forecast._offload import (
    _draw_chunked,
    _leaf_view,
    _oom_advice,
    _resolve_device,
    _ResolvedDevice,
    _stitch_chunks,
    _transfer,
)
from numpyro_forecast._validation import (
    _require_covariates_extend_data,
    _require_positive_num_samples,
)
from numpyro_forecast.typing import Array, ForecastModel


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
    :func:`forecast` or NumPyro's ``Predictive``.
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
        keeps every leaf in pageable host memory, so nothing of the result
        occupies accelerator memory: with the JAX CPU backend initialized it
        commits each leaf to ``jax.devices("cpu")[0]`` and returns committed
        :class:`jax.Array` leaves (``np.asarray`` on one is a zero-copy view);
        without it (for example after ``numpyro.set_platform("cuda")``, or a
        ``JAX_PLATFORMS`` preset) it copies each chunk with
        :func:`jax.device_get` and returns NumPy arrays, since a CUDA client
        offers no pageable ``jax.Array`` container. It therefore needs no CPU
        backend and never pins memory. ``"numpy"`` forces the NumPy path;
        ``"pinned_host"`` commits to the accelerator's pinned host memory kind
        instead, a pool capped by ``XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB`` (64 GB
        by default on CUDA), so prefer ``"host"`` for large panels. A
        :class:`jax.Device` or platform name like ``"cpu"`` commits the draws
        to that device (``"cpu"`` warns and takes the NumPy path when the CPU
        backend is missing). ``None`` keeps everything on the default device.
        ``device`` never changes the draw values. A host-committed jax result
        is not a drop-in replacement for a device array in your own ``jnp``
        code: mixed with an uncommitted array an op runs on the CPU and
        returns a CPU-committed array, mixed with an accelerator-committed
        array it raises, and a pinned array raises on any mix. Feed a host
        posterior (in either container) straight into
        :func:`forecast`,
        :func:`predict_in_sample`, or
        :func:`~numpyro_forecast.convert.to_datatree` (all of which already
        accept it), or convert explicitly first with ``np.asarray(x)`` (stays
        on host) or ``jax.device_put(x, device)`` (moves it to an accelerator).

    Returns
    -------
    dict[str, Array | np.ndarray]
        Posterior samples of the latent sites, sample axis leading. With
        ``device="host"`` the leaves are committed to the CPU device, or NumPy
        arrays when no CPU backend is initialized.

    Raises
    ------
    ValueError
        If ``num_samples`` or ``batch_size`` is not positive.
    RuntimeError
        If ``device="pinned_host"`` is requested on a device that exposes no
        host memory kind (see
        :func:`~numpyro_forecast._offload._host_memory_kind`).

    Warns
    -----
    UserWarning
        If ``device="cpu"`` is requested and the JAX CPU backend is not
        initialized, so the draws take the NumPy path of ``"host"`` instead.

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


def _sample_axis_size(posterior: Mapping[str, Array | np.ndarray]) -> int:
    """Validate and return the shared sample-axis length of a posterior.

    Parameters
    ----------
    posterior
        Posterior samples, sample axis leading (all leaves must agree on that
        axis).

    Returns
    -------
    int
        The sample-axis length shared by every leaf.

    Raises
    ------
    ValueError
        If ``posterior`` is empty or the leaves disagree on the sample-axis
        length (the message names the offending sites).
    """
    if not posterior:
        msg = "posterior must be non-empty"
        raise ValueError(msg)
    sizes = {name: leaf.shape[0] for name, leaf in posterior.items()}
    distinct = set(sizes.values())
    if len(distinct) != 1:
        msg = f"posterior leaves disagree on the sample axis: {sizes}"
        raise ValueError(msg)
    return distinct.pop()


def _chunk_indices(num_samples: int, batch_size: int) -> list[Array]:
    """Build wrapped sample-axis index blocks of exactly ``batch_size`` rows each.

    Block ``i`` holds ``arange(i * batch_size, (i + 1) * batch_size) % num_samples``:
    every block shares one shape, so the jitted predictive compiles exactly
    once regardless of ``num_samples``, and the final block wraps around to
    re-use leading draws (when ``num_samples`` is not an exact multiple of
    ``batch_size``), which :func:`_chunked_draws` discards with a final
    ``[:num_samples]`` slice. The blocks reproduce the draws of the former
    posterior-padding scheme bit for bit without materializing a padded copy of
    the posterior.

    Parameters
    ----------
    num_samples
        Total number of posterior draws (callers with
        ``num_samples <= batch_size`` take the unchunked passthrough instead).
    batch_size
        Positive chunk size.

    Returns
    -------
    list[Array]
        One integer index block of length ``batch_size`` per chunk.

    Raises
    ------
    ValueError
        If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    starts = range(0, num_samples, batch_size)
    return [jnp.arange(start, start + batch_size) % num_samples for start in starts]


def _chunked_draws(
    rng_key: Array,
    predict_fn: Callable[[Array, Mapping[str, Array | np.ndarray]], Array],
    posterior: Mapping[str, ArrayLike],
    batch_size: int | None,
    device: _ResolvedDevice = None,
) -> Array | np.ndarray:
    """Run ``predict_fn`` over fixed-size posterior chunks and stitch the draws.

    Shared chunk driver for :func:`forecast` and :func:`predict_in_sample`.
    With ``batch_size`` ``None`` (or at least the sample count) ``predict_fn``
    is called once on the full posterior with ``rng_key`` itself. Otherwise the
    posterior is gathered into wrapped index blocks of exactly ``batch_size``
    rows (:func:`_chunk_indices`) so every chunk shares one shape and the
    jitted ``predict_fn`` compiles exactly once; one subkey is split per chunk,
    and the wrapped draws are discarded by a final slice (skipped when the
    sample count is an exact multiple of ``batch_size``, since JAX slices copy
    and nothing wrapped). Every leaf of ``posterior`` is first normalized with
    :func:`~numpyro_forecast._offload._leaf_view`, so a
    host-committed posterior (the output of a previous ``device="host"`` stage)
    is gathered on the host and only the chunk reaches the accelerator. When
    ``device`` is given, every chunk is moved there (:func:`_transfer`) before
    the next one is drawn and the stitched result lives on ``device``, bounding
    accelerator memory by a single chunk; every host target (the CPU device,
    the ``"numpy"`` sentinel, and the ``"pinned_host"`` sentinel) stitches
    through NumPy so the result never touches an accelerator. ``device`` is the
    *resolved* value: the public drivers run ``"host"`` through
    :func:`~numpyro_forecast._offload._resolve_device` first.

    Parameters
    ----------
    rng_key
        PRNG key; consumed directly when unchunked, split per chunk otherwise.
    predict_fn
        ``(rng_key, posterior) -> draws`` with the sample axis leading.
    posterior
        Posterior samples of the latent sites, sample axis leading.
    batch_size
        Optional chunk size (caps peak memory).
    device
        Optional resolved device (or the ``"numpy"``/``"pinned_host"``
        sentinel) each chunk and the stitched result are moved to.

    Returns
    -------
    Array | np.ndarray
        The stitched draws for the original sample count (committed to
        ``device`` when one is given; NumPy for ``"numpy"``).
    """
    staged = {
        name: _leaf_view(cast("Array | np.ndarray", leaf)) for name, leaf in posterior.items()
    }
    num = _sample_axis_size(staged)
    if batch_size is None or batch_size >= num:
        with _oom_advice("predictive sampling", batch_size):
            return _transfer(predict_fn(rng_key, staged), device)
    indices = _chunk_indices(num, batch_size)
    keys = random.split(rng_key, len(indices))
    with _oom_advice("predictive sampling", batch_size):
        chunks = [
            _transfer(predict_fn(keys[i], _index_tree(staged, idx)), device)
            for i, idx in enumerate(indices)
        ]
        return _stitch_chunks(chunks, num, device)


@partial(jax.jit, static_argnums=(1,), static_argnames=("parallel",))
def _predict(
    rng_key: Array,
    model: ForecastModel,
    posterior: Mapping[str, Array | np.ndarray],
    data: Array,
    covariates: Array,
    *,
    parallel: bool = True,
) -> Array:
    """Run ``Predictive`` over the full horizon and return the ``forecast`` site.

    Jitted with ``model`` (and ``parallel``) static: each
    ``(model, parallel, shape)`` combination compiles once and is reused, which
    is what makes the chunked :func:`forecast` loop cheap (the per-call
    ``Predictive`` tracing cost is paid a single time). ``parallel`` selects
    ``Predictive``'s sample-axis mapping (``vmap`` when ``True``, serial
    ``lax.map`` when ``False``).
    """
    predictive = Predictive(
        model, posterior_samples=dict(posterior), return_sites=["forecast"], parallel=parallel
    )
    return predictive(rng_key, covariates, data)["forecast"]


def forecast(
    rng_key: Array,
    model: ForecastModel,
    posterior: Mapping[str, ArrayLike],
    data: Array,
    covariates: Array,
    *,
    batch_size: int | None = None,
    parallel: bool = True,
    device: jax.Device | str | None = None,
) -> Num[Array | np.ndarray, " sample *batch future obs"]:
    """Sample forecasts for the steps in ``[t, duration)`` from a posterior.

    Runs ``Predictive`` with full-horizon ``covariates`` and the in-sample
    ``data``: the in-sample latent sites are drawn from ``posterior`` while the
    ``_future`` suffix is drawn from the prior, and the ``"forecast"`` site is
    returned. The number of forecast samples equals the leading (sample) axis of
    ``posterior`` (see :func:`~numpyro_forecast.predictive.draw_posterior`).

    Parameters
    ----------
    rng_key
        PRNG key.
    model
        The forecasting model callable (the same one that produced ``posterior``).
    posterior
        Posterior samples of the latent sites, sample axis leading. The output
        of a ``device="host"`` stage (CPU-committed jax leaves or NumPy leaves)
        is accepted directly.
    data
        Observed data with time at axis ``-2`` and length ``t``.
    covariates
        Covariates with time at axis ``-2`` and length ``duration > t``.
    batch_size
        Optional chunk size for sampling (caps peak memory).
    parallel
        Whether ``Predictive`` vectorizes over the sample axis with ``vmap``
        (``True``, faster, higher peak memory) or maps it serially with
        ``lax.map`` (``False``). With ``parallel=True`` the samples in each
        ``batch_size`` chunk are vectorized while the chunks are looped over, so
        ``batch_size`` remains the peak-memory governor. The two settings produce
        the same draws up to floating-point reduction order.
    device
        Where each chunk of draws is placed as soon as it is drawn and where
        the stitched result lives; the same placement contract as the
        ``device`` argument of :func:`draw_posterior` (``"host"`` for pageable
        host memory as a CPU-committed :class:`jax.Array`, or a NumPy array
        when no CPU backend is initialized; ``"numpy"``; ``"pinned_host"``; a
        :class:`jax.Device` or platform name; ``None`` for the default device),
        including its mixing rules for host-committed results. With
        ``batch_size`` set on an accelerator, any host target bounds
        accelerator memory by a single chunk instead of the full
        ``(sample, future, obs)`` array; the draw values are unchanged, only where
        the result lives. The bound requires ``batch_size`` strictly below the
        sample count: at or above it, the single-shot path runs and the full
        array is materialized on the default device before the one transfer.
        The result feeds straight into :func:`~numpyro_forecast.convert.to_datatree`
        and the ``batch_size``-chunked evaluation metrics in
        :mod:`~numpyro_forecast.evaluate`, which accept host-resident draws.

    Returns
    -------
    Num[Array, " sample *batch future obs"]
        Forecast samples over the ``future = duration - t`` horizon (floating
        point for continuous observations, integer for discrete/count models
        built with :func:`~numpyro_forecast.models.predict`;
        with ``device="host"`` committed to the CPU device, or a NumPy array
        when no CPU backend is initialized).

    Raises
    ------
    ValueError
        If ``covariates`` does not extend beyond ``data`` along the time axis.
    RuntimeError
        If ``device="pinned_host"`` is requested on a device that exposes no
        host memory kind (see
        :func:`~numpyro_forecast._offload._host_memory_kind`).

    Warns
    -----
    UserWarning
        If ``device="cpu"`` is requested and the JAX CPU backend is not
        initialized, so the draws take the NumPy path of ``"host"`` instead.

    Notes
    -----
    Chunking is a memory knob, not a reproducibility knob: reproducibility is
    per ``(rng_key, batch_size)``. Every chunk shares the exact ``batch_size``
    shape (the final chunk wraps around to re-used draws that are discarded),
    so the underlying ``_predict`` compiles exactly once for a fixed shape, but
    changing ``batch_size`` changes the PRNG stream layout and therefore the
    exact draws. ``device`` never changes the draws, only where they live.
    """
    _require_covariates_extend_data(data, covariates)

    def predict_fn(key: Array, post: Mapping[str, Array | np.ndarray]) -> Array:
        return _predict(key, model, post, data, covariates, parallel=parallel)

    return _chunked_draws(rng_key, predict_fn, posterior, batch_size, _resolve_device(device))


@partial(jax.jit, static_argnums=(1,), static_argnames=("parallel",))
def _predict_obs(
    rng_key: Array,
    model: ForecastModel,
    posterior: Mapping[str, Array | np.ndarray],
    covariates: Array,
    *,
    parallel: bool = True,
) -> Array:
    """Run ``Predictive`` over the observed window and return the ``obs`` site.

    Jitted with ``model`` (and ``parallel``) static (see :func:`_predict`): each
    ``(model, parallel, shape)`` combination compiles once and is reused, keeping
    the chunked :func:`predict_in_sample` loop cheap.
    """
    predictive = Predictive(
        model, posterior_samples=dict(posterior), return_sites=["obs"], parallel=parallel
    )
    return predictive(rng_key, covariates)["obs"]


def predict_in_sample(
    rng_key: Array,
    model: ForecastModel,
    posterior: Mapping[str, ArrayLike],
    covariates: Array,
    *,
    batch_size: int | None = None,
    parallel: bool = True,
    device: jax.Device | str | None = None,
) -> Num[Array | np.ndarray, " sample *batch time obs"]:
    """Sample the in-sample posterior predictive of the ``obs`` site.

    Runs ``Predictive`` with the in-sample ``covariates`` and the supplied posterior
    latent draws. Unlike :func:`forecast` there is no forecast horizon: ``covariates``
    span only the observed window, so the model's ``obs`` site is sampled at every
    step. The number of predictive samples equals the leading (sample) axis of
    ``posterior`` (see :func:`~numpyro_forecast.predictive.draw_posterior`).

    Parameters
    ----------
    rng_key
        PRNG key.
    model
        The forecasting model callable (the same one that produced ``posterior``).
    posterior
        Posterior samples of the latent sites, sample axis leading. The output
        of a ``device="host"`` stage (CPU-committed jax leaves or NumPy leaves)
        is accepted directly.
    covariates
        Covariates with time at axis ``-2`` spanning the observed window. Its time
        length must match the data the ``posterior`` was fit on, since the in-sample
        latent sites are sized to that window.
    batch_size
        Optional chunk size for sampling (caps peak memory).
    parallel
        Whether ``Predictive`` vectorizes over the sample axis with ``vmap``
        (``True``, faster, higher peak memory) or maps it serially with
        ``lax.map`` (``False``). See :func:`forecast` for how this interacts with
        ``batch_size``.
    device
        Where each chunk of draws is placed as soon as it is drawn and where
        the stitched result lives; the same placement contract as the
        ``device`` argument of :func:`draw_posterior` (``"host"`` for pageable
        host memory as a CPU-committed :class:`jax.Array`, or a NumPy array
        when no CPU backend is initialized; ``"numpy"``; ``"pinned_host"``; a
        :class:`jax.Device` or platform name; ``None`` for the default device),
        including its mixing rules for host-committed results. With
        ``batch_size`` set on an accelerator, any host target bounds
        accelerator memory by a single chunk instead of the full
        ``(sample, time, obs)`` array; the draw values are unchanged, only where
        the result lives. The bound requires ``batch_size`` strictly below the
        sample count: at or above it, the single-shot path runs and the full
        array is materialized on the default device before the one transfer.
        The result feeds straight into :func:`~numpyro_forecast.convert.to_datatree`
        and the ``batch_size``-chunked evaluation metrics in
        :mod:`~numpyro_forecast.evaluate`, which accept host-resident draws.

    Returns
    -------
    Num[Array, " sample *batch time obs"]
        In-sample posterior-predictive draws of the ``obs`` site (with
        ``device="host"`` committed to the CPU device, or a NumPy array when no
        CPU backend is initialized).

    Raises
    ------
    RuntimeError
        If ``device="pinned_host"`` is requested on a device that exposes no
        host memory kind (see
        :func:`~numpyro_forecast._offload._host_memory_kind`).

    Warns
    -----
    UserWarning
        If ``device="cpu"`` is requested and the JAX CPU backend is not
        initialized, so the draws take the NumPy path of ``"host"`` instead.
    """

    def predict_fn(key: Array, post: Mapping[str, Array | np.ndarray]) -> Array:
        return _predict_obs(key, model, post, covariates, parallel=parallel)

    return _chunked_draws(rng_key, predict_fn, posterior, batch_size, _resolve_device(device))
