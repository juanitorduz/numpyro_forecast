"""Producing draws: posterior sampling from a fitted guide and predictive sampling.

`draw_posterior()` draws posterior samples of the latent sites from a fitted
variational guide (an `numpyro.infer.autoguide.AutoGuide`) and its learned
parameters, as returned by ``AutoGuide``/``SVI.run`` (``guide``/``state.params``).
It is guide-only on purpose: MCMC users already hold their posterior samples via
``mcmc.get_samples()``, and hand-written-guide users draw with a single
``numpyro.infer.Predictive(guide, params=params, num_samples=n)(rng_key,
covariates, data)`` call; neither needs this function. The blackjax Pathfinder
backend has its own analogous entry point,
`~~numpyro_forecast.contrib.blackjax.pathfinder_samples()`, built on the same
shared chunk-and-transfer loop, `~~numpyro_forecast._offload._draw_chunked()`.

`forecast()` samples forecasts over the horizon from posterior draws, and
`predict_in_sample()` samples the in-sample posterior predictive of the
``obs`` site. Both drive a jitted ``Predictive`` wrapper through the shared
chunking driver ``_chunked_draws``, which caps peak memory while compiling the
predictive exactly once. With ``device`` set (e.g. ``"host"``: pageable host
memory, as jax Arrays on the CPU backend device or as NumPy arrays when that
backend is not initialized), every chunk is moved off the accelerator before
the next one is drawn, so accelerator memory is bounded by a single chunk
instead of the full draw array. The ``device`` contract is documented once, on
`draw_posterior()`; the predictive drivers share it.

`forecast_log_density()` is the density counterpart of `forecast()`: one seeded,
conditioned, traced pass of the model per posterior draw, with a
`~~numpyro_forecast.typing.DensityReader` turning each trace into the log density of
the realized future values given that draw's sampled future latent path. The per-draw
array it returns is reduced over draws by `reduce_log_density()` into one of the two
h-step scores (`~~numpyro_forecast.typing.Reduction`), and
`conditional_log_density()` is the exact chain-rule route for error-feedback models
(`~~numpyro_forecast.models.ssoe()`), where the block density conditioned on the
realized values is the in-sample filter extended by the block. The design is in
``docs/dev/stacking_design.md``.
"""

from collections.abc import Callable, Mapping
from functools import lru_cache, partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax import random
from jax.tree_util import tree_map
from jax.typing import ArrayLike
from jaxtyping import Float, Num
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
from numpyro_forecast.exceptions import DensitySiteError
from numpyro_forecast.typing import Array, DensityReader, ForecastModel, Reduction


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
    means every chunk of a `draw_posterior()` call (and repeated calls
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
    `forecast()` or NumPyro's ``Predictive``.
    An ``AutoDelta`` guide is a MAP point estimate: it is drawn once and tiled to
    ``num_samples`` (`_ensure_sample_axis_for_delta()`), since it carries no
    posterior spread of its own. Every other ``AutoGuide`` is sampled through a
    jitted, per-guide-cached ``sample_posterior`` (`_jitted_sample_posterior()`).

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
        `jax.Array` leaves (``np.asarray`` on one is a zero-copy view);
        without it (for example after ``numpyro.set_platform("cuda")``, or a
        ``JAX_PLATFORMS`` preset) it copies each chunk with
        `jax.device_get()` and returns NumPy arrays, since a CUDA client
        offers no pageable ``jax.Array`` container. It therefore needs no CPU
        backend and never pins memory. ``"numpy"`` forces the NumPy path;
        ``"pinned_host"`` commits to the accelerator's pinned host memory kind
        instead, a pool capped by ``XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB`` (64 GB
        by default on CUDA), so prefer ``"host"`` for large panels. A
        `jax.Device` or platform name like ``"cpu"`` commits the draws
        to that device (``"cpu"`` warns and takes the NumPy path when the CPU
        backend is missing). ``None`` keeps everything on the default device.
        ``device`` never changes the draw values. A host-committed jax result
        is not a drop-in replacement for a device array in your own ``jnp``
        code: mixed with an uncommitted array an op runs on the CPU and
        returns a CPU-committed array, mixed with an accelerator-committed
        array it raises, and a pinned array raises on any mix. Feed a host
        posterior (in either container) straight into
        `forecast()`,
        `predict_in_sample()`, or
        `~~numpyro_forecast.convert.to_datatree()` (all of which already
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
    HostMemoryKindError
        If ``device="pinned_host"`` is requested on a device that exposes no
        host memory kind (see
        `~~numpyro_forecast._offload._host_memory_kind()`).
    DevicePlatformError
        If ``device`` names a platform whose backend is not initialized (see
        `~~numpyro_forecast._offload._resolve_device()`).

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
    ``batch_size``), which `_chunked_draws()` discards with a final
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

    Shared chunk driver for `forecast()` and `predict_in_sample()`.
    With ``batch_size`` ``None`` (or at least the sample count) ``predict_fn``
    is called once on the full posterior with ``rng_key`` itself. Otherwise the
    posterior is gathered into wrapped index blocks of exactly ``batch_size``
    rows (`_chunk_indices()`) so every chunk shares one shape and the
    jitted ``predict_fn`` compiles exactly once; one subkey is split per chunk,
    and the wrapped draws are discarded by a final slice (skipped when the
    sample count is an exact multiple of ``batch_size``, since JAX slices copy
    and nothing wrapped). Every leaf of ``posterior`` is first normalized with
    `~~numpyro_forecast._offload._leaf_view()`, so a
    host-committed posterior (the output of a previous ``device="host"`` stage)
    is gathered on the host and only the chunk reaches the accelerator. When
    ``device`` is given, every chunk is moved there (`_transfer()`) before
    the next one is drawn and the stitched result lives on ``device``, bounding
    accelerator memory by a single chunk; every host target (the CPU device,
    the ``"numpy"`` sentinel, and the ``"pinned_host"`` sentinel) stitches
    through NumPy so the result never touches an accelerator. ``device`` is the
    *resolved* value: the public drivers run ``"host"`` through
    `~~numpyro_forecast._offload._resolve_device()` first.

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
    is what makes the chunked `forecast()` loop cheap (the per-call
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
    ``posterior`` (see `~~numpyro_forecast.predictive.draw_posterior()`).

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
        ``device`` argument of `draw_posterior()` (``"host"`` for pageable
        host memory as a CPU-committed `jax.Array`, or a NumPy array
        when no CPU backend is initialized; ``"numpy"``; ``"pinned_host"``; a
        `jax.Device` or platform name; ``None`` for the default device),
        including its mixing rules for host-committed results. With
        ``batch_size`` set on an accelerator, any host target bounds
        accelerator memory by a single chunk instead of the full
        ``(sample, future, obs)`` array; the draw values are unchanged, only where
        the result lives. The bound requires ``batch_size`` strictly below the
        sample count: at or above it, the single-shot path runs and the full
        array is materialized on the default device before the one transfer.
        The result feeds straight into `~~numpyro_forecast.convert.to_datatree()`
        and the ``batch_size``-chunked evaluation metrics in
        `~~numpyro_forecast.evaluate`, which accept host-resident draws.

    Returns
    -------
    Num[Array, " sample *batch future obs"]
        Forecast samples over the ``future = duration - t`` horizon (floating
        point for continuous observations, integer for discrete/count models
        built with `~~numpyro_forecast.models.predict()`;
        with ``device="host"`` committed to the CPU device, or a NumPy array
        when no CPU backend is initialized).

    Raises
    ------
    ValueError
        If ``covariates`` does not extend beyond ``data`` along the time axis.
    HostMemoryKindError
        If ``device="pinned_host"`` is requested on a device that exposes no
        host memory kind (see
        `~~numpyro_forecast._offload._host_memory_kind()`).
    DevicePlatformError
        If ``device`` names a platform whose backend is not initialized (see
        `~~numpyro_forecast._offload._resolve_device()`).

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

    Jitted with ``model`` (and ``parallel``) static (see `_predict()`): each
    ``(model, parallel, shape)`` combination compiles once and is reused, keeping
    the chunked `predict_in_sample()` loop cheap.
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
    latent draws. Unlike `forecast()` there is no forecast horizon: ``covariates``
    span only the observed window, so the model's ``obs`` site is sampled at every
    step. The number of predictive samples equals the leading (sample) axis of
    ``posterior`` (see `~~numpyro_forecast.predictive.draw_posterior()`).

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
        ``lax.map`` (``False``). See `forecast()` for how this interacts with
        ``batch_size``.
    device
        Where each chunk of draws is placed as soon as it is drawn and where
        the stitched result lives; the same placement contract as the
        ``device`` argument of `draw_posterior()` (``"host"`` for pageable
        host memory as a CPU-committed `jax.Array`, or a NumPy array
        when no CPU backend is initialized; ``"numpy"``; ``"pinned_host"``; a
        `jax.Device` or platform name; ``None`` for the default device),
        including its mixing rules for host-committed results. With
        ``batch_size`` set on an accelerator, any host target bounds
        accelerator memory by a single chunk instead of the full
        ``(sample, time, obs)`` array; the draw values are unchanged, only where
        the result lives. The bound requires ``batch_size`` strictly below the
        sample count: at or above it, the single-shot path runs and the full
        array is materialized on the default device before the one transfer.
        The result feeds straight into `~~numpyro_forecast.convert.to_datatree()`
        and the ``batch_size``-chunked evaluation metrics in
        `~~numpyro_forecast.evaluate`, which accept host-resident draws.

    Returns
    -------
    Num[Array, " sample *batch time obs"]
        In-sample posterior-predictive draws of the ``obs`` site (with
        ``device="host"`` committed to the CPU device, or a NumPy array when no
        CPU backend is initialized).

    Raises
    ------
    HostMemoryKindError
        If ``device="pinned_host"`` is requested on a device that exposes no
        host memory kind (see
        `~~numpyro_forecast._offload._host_memory_kind()`).
    DevicePlatformError
        If ``device`` names a platform whose backend is not initialized (see
        `~~numpyro_forecast._offload._resolve_device()`).

    Warns
    -----
    UserWarning
        If ``device="cpu"`` is requested and the JAX CPU backend is not
        initialized, so the draws take the NumPy path of ``"host"`` instead.
    """

    def predict_fn(key: Array, post: Mapping[str, Array | np.ndarray]) -> Array:
        return _predict_obs(key, model, post, covariates, parallel=parallel)

    return _chunked_draws(rng_key, predict_fn, posterior, batch_size, _resolve_device(device))


# --- Forecast log densities (stacking design, docs/dev/stacking_design.md) ----------------


_FUTURE_OBS_SITE = "obs_future"
"""The observation site `~~numpyro_forecast.models.predict()` samples over the horizon."""


def _site_log_prob(fn: dist.Distribution, truth: Array) -> Array:
    """``fn.log_prob`` of a ``(*batch, future, obs)`` block, laid out per cell or per block.

    Branches on ``fn.event_shape`` **before** calling ``log_prob``, because a
    correlated family silently broadcasts a wrongly shaped ``truth``: a
    ``MultivariateNormal`` with event ``(future,)`` evaluated at ``truth`` of shape
    ``(future, 1)`` returns ``(future,)`` finite garbage, one scalar broadcast against
    the whole event, with no error. Elementwise families (empty event shape) return the
    per-cell value ``(*batch, future, obs)``. An event shape of ``(future,)`` (time-leading
    MVN, the layout `~~numpyro_forecast.surgery` supports) evaluates ``truth[..., 0]``
    and returns the joint block value reshaped to ``(*batch, 1, 1)``; an event shape of
    ``(obs,)`` (an `ssoe()` noise distribution over the observation axis) evaluates the
    block and returns ``(*batch, future, 1)``. Any other event shape raises
    ``ValueError`` naming it, rather than guessing.
    """
    raise NotImplementedError


def predict_density(
    model_trace: Mapping[str, Any],
    truth: Num[Array, " *batch future obs"],
    *,
    site: str = _FUTURE_OBS_SITE,
) -> Float[Array, " *batch _future _obs"]:
    """Read the forecast log density from the ``obs_future`` site of a traced pass.

    The built-in `~~numpyro_forecast.typing.DensityReader` for models built on
    `~~numpyro_forecast.models.predict()`: returns
    ``model_trace[site]["fn"].log_prob(truth)`` with ``site="obs_future"``, the log density of the
    realized block under the forecast distribution conditioned on the pass's sampled
    future latent path. For an elementwise observation family the result has one
    value per cell, shape ``(*batch, future, obs)``; for a correlated family whose
    event shape covers the time axis the single joint value per draw is returned
    as ``(*batch, 1, 1)``, so only the ``"joint"`` reduction is meaningful
    downstream (the event shape is checked before ``log_prob`` is called, since a
    mis-shaped ``truth`` would otherwise broadcast into finite garbage; see
    `_site_log_prob()`).

    Parameters
    ----------
    model_trace
        The trace of one seeded, conditioned pass of the model over the full
        horizon (what `forecast_log_density()` builds).
    truth
        Realized values over the horizon, shape ``(*batch, future, obs)``, with
        the integer dtype `~~numpyro_forecast.models.predict()` requires for a
        discrete family.
    site
        The observed site to read; ``"obs_future"`` by default. A model that
        registers its own predictive over the horizon under another name (a
        zero-inflated demand predictive next to an `ssoe()` recursion, say) is
        read with ``functools.partial(predict_density, site=...)``.

    Returns
    -------
    Float[Array, " *batch _future _obs"]
        The per-cell log density, or the joint block log density with size-one
        trailing axes.

    Raises
    ------
    DensitySiteError
        If the trace has no ``obs_future`` site: the model registers ``forecast``
        itself (an `~~numpyro_forecast.models.ssoe()` model) and needs
        `ssoe_density()` instead.
    ValueError
        If the site's event shape is neither empty nor the time-leading
        ``(future,)`` layout.
    """
    traced = model_trace.get(site)
    if traced is None:
        msg = (
            f"the traced model pass has no {site!r} site, so predict_density cannot read a "
            "forecast density: the model registers 'forecast' itself. For an additive "
            "model built on ssoe(), pass reader=ssoe_density(name) and register "
            "numpyro.deterministic(f'{name}_mu_future', r.mu_future) when forecasting; "
            "otherwise register the predictive to score as an observed site and point "
            "predict_density at it with site=."
        )
        raise DensitySiteError(msg)
    raise NotImplementedError


def ssoe_density(name: str) -> DensityReader:
    """Build the `~~numpyro_forecast.typing.DensityReader` for an `ssoe()` model.

    `~~numpyro_forecast.models.ssoe()` registers only the future error site
    ``f"{name}_future"`` and returns the path means; the caller writes the likelihood
    and registers ``forecast``. The forecast distribution at step ``j`` along a sampled
    error path is therefore ``noise_dist`` shifted by ``mu_future[j]``, and the reader
    computes ``noise_fn.log_prob(truth - mu_future)`` with ``noise_fn`` read from the
    error site (the draw's own parameters, expanded under the ``time_future`` plate) and
    ``mu_future`` read from the deterministic site ``f"{name}_mu_future"``, which the
    model has to register when forecasting:

    ```python
    r = ssoe(h, "arma", y, init, mean, update, dist.Normal(0.0, sigma))
    numpyro.sample("obs", dist.Normal(r.mu, sigma), obs=h.data)
    if h.future > 0:
        numpyro.deterministic("arma_mu_future", r.mu_future)
        numpyro.deterministic("forecast", r.y_future)
    ```

    This is the horizon-marginal density along simulated predecessors, the same
    quantity `predict_density()` reads for `~~numpyro_forecast.models.predict()`
    models, and it is correct for **additive single-channel** models (ARMA,
    exponential smoothing, a local level), where the observation is the channel's
    mean plus the channel's noise. It is *not* the demand density of the
    intermittent-demand models (Croston, TSB), whose ``forecast`` is a product of two
    channels and whose predictive, a zero-inflated mixture, exists nowhere in the
    trace: such a model has to register an ``obs_future``-style site holding the
    predictive the user wants scored, and then `predict_density()` applies (with
    ``site`` pointing at it). The chain-rule block density of an `ssoe()` model is
    exact and cheaper through `conditional_log_density()`.

    Parameters
    ----------
    name
        The site prefix passed to `~~numpyro_forecast.models.ssoe()`.

    Returns
    -------
    DensityReader
        ``(trace, truth) -> log_density`` of shape ``(*batch, future, obs_or_1)``.

    Raises
    ------
    DensitySiteError
        From the returned reader, when ``f"{name}_future"`` or
        ``f"{name}_mu_future"`` is missing from the trace.
    """
    raise NotImplementedError


@partial(jax.jit, static_argnums=(1, 5), static_argnames=("parallel",))
def _log_density(
    rng_key: Array,
    model: ForecastModel,
    posterior: Mapping[str, Array | np.ndarray],
    data: Array,
    covariates: Array,
    reader: DensityReader,
    truth: Array,
    *,
    parallel: bool = True,
) -> Array:
    """Per-draw forecast log densities: one seeded, conditioned, traced pass per draw.

    For each draw: ``substitute`` its in-sample sites, ``seed`` the pass so every
    ``_future`` site and reparameterization auxiliary is drawn from the prior
    (exactly as ``Predictive`` does in `_predict()`), ``condition`` the
    ``obs_future`` site on ``truth`` so its density is recorded rather than sampled,
    ``trace`` the model over the full horizon, and hand the trace to ``reader``. The
    draw axis is mapped with ``vmap`` (``parallel=True``) or ``lax.map``. Jitted with
    ``model`` and ``reader`` static, so each ``(model, reader, shape)`` compiles once
    and the chunked `forecast_log_density()` loop reuses it.
    """
    raise NotImplementedError


def forecast_log_density(
    rng_key: Array,
    model: ForecastModel,
    posterior: Mapping[str, ArrayLike],
    data: Array,
    covariates: Array,
    truth: Num[Array, " *batch future obs"],
    *,
    reader: DensityReader = predict_density,
    batch_size: int | None = None,
    parallel: bool = True,
    device: jax.Device | str | None = None,
) -> Float[Array | np.ndarray, " sample *batch _future _obs"]:
    r"""Per-draw log densities of the realized future values under the forecast distribution.

    The density counterpart of `forecast()`, with the same arguments plus ``truth``:
    for posterior draw $s$ with parameters $\theta^{(s)}$ and a future latent path
    $z^{(s)}$ drawn from the prior exactly as `forecast()` draws it, the result is

    $$
    \ell^{(s)}_{j} = \log p\left(y_{t+j} \mid z^{(s)}, \theta^{(s)}, y_{\le t}\right),
    $$

    one value per draw and cell, shape ``(sample, *batch, future, obs)``. It is the
    raw material of the two h-step scores: `reduce_log_density()` averages it over
    draws cell by cell (``"marginal"``, the density of the $j$-step-ahead forecast
    distribution, the one `forecast()` samples) or over the block (``"joint"``, the
    chain-rule block density). ``reader`` turns each traced pass into its log density:
    `predict_density()` (default) for models built on
    `~~numpyro_forecast.models.predict()`, `ssoe_density()` for models built on
    `~~numpyro_forecast.models.ssoe()`. The function never returns forecast draws;
    `forecast()` does, with its own key.

    ``posterior`` must hold in-sample sites only. A dict that also carries
    ``_future`` sites (the output of a ``Predictive`` call over a horizon, or of a
    previous `forecast()` trace) would pin the future path by name through
    ``substitute`` and return the same densities for every key with no error, so the
    driver rejects such keys with ``ValueError`` listing them (any key containing
    ``"_future"``, plus ``"obs_future"`` and ``"forecast"``). Chunking, ``parallel``,
    ``device`` and the host-offload rules are those of `forecast()`.

    Parameters
    ----------
    rng_key
        PRNG key for the future latent paths.
    model
        The forecasting model callable (the same one that produced ``posterior``).
    posterior
        Posterior samples of the in-sample latent sites, sample axis leading; the
        output of a ``device="host"`` stage is accepted directly.
    data
        Observed data with time at axis ``-2`` and length ``t``.
    covariates
        Covariates with time at axis ``-2`` and length ``duration > t``.
    truth
        Realized values over the horizon, shape ``(*batch, duration - t, obs)``,
        integer-dtyped for a discrete observation family.
    reader
        The `~~numpyro_forecast.typing.DensityReader` for the model's forecast site.
    batch_size
        Optional chunk size for the draw axis (caps peak memory).
    parallel
        ``vmap`` (``True``) or ``lax.map`` (``False``) over the draw axis.
    device
        Placement of each chunk and the stitched result; the contract of
        `draw_posterior()`.

    Returns
    -------
    Float[Array | np.ndarray, " sample *batch _future _obs"]
        Per-draw log densities; a size-one time axis for a correlated family
        (see `predict_density()`).

    Raises
    ------
    ValueError
        If ``covariates`` does not extend beyond ``data``, if ``truth`` does not
        have shape ``(*batch, duration - t, obs)``, if ``posterior`` carries a
        future site, or if the observation family is discrete and ``truth`` is not
        integer-dtyped (the `~~numpyro_forecast.models.predict()` message).
    DensitySiteError
        If ``reader`` cannot find its site in the trace.

    Notes
    -----
    Reproducibility is per ``(rng_key, batch_size)`` as for `forecast()`. A density
    and a forecast drawn with the same key are *not* on the same future paths
    (``condition`` changes the key consumption of the pass); nothing downstream
    needs them to be. The number of draws must grow with the horizon: the
    ``"marginal"`` average at a cell where the realized value is far in the tail of
    most sampled paths is carried by few draws (`log_density_ess()`), and the
    log-mean-exp estimator is biased downward by about half its squared standard
    error, a bias that is larger for the model with the wider latent path.
    """
    raise NotImplementedError


def reduce_log_density(
    log_density: Float[ArrayLike, " sample *batch future obs"],
    reduction: Reduction = "marginal",
) -> Float[Array, " *batch _future _obs"]:
    r"""Reduce per-draw forecast log densities over the draw axis into an h-step score.

    ``"marginal"`` (default) averages the density cell by cell,

    $$
    \hat\ell_{j} = \log \frac{1}{S} \sum_s \exp \ell^{(s)}_{j},
    $$

    the log density of the $j$-step-ahead forecast distribution at the realized
    value with the intermediate steps not revealed: the distribution `forecast()`
    samples and every sample metric scores, and the quantity that stacks per
    horizon step. The result keeps the cell layout ``(*batch, future, obs)``.
    ``"joint"`` sums over the time axis first,

    $$
    \hat L = \log \frac{1}{S} \sum_s \exp \sum_j \ell^{(s)}_{j},
    $$

    the block density, equal by the chain rule to the sum of one-step-ahead
    densities each conditional on the realized predecessors (what
    `arviz_stats.loo.lfo_cv` computes). It is an importance-sampling-like average
    whose effective sample size collapses with the horizon length
    (`log_density_ess()` measures it; on the prototype it kept 11% of the draws at
    14 steps against 80% for the marginal); the result has a size-one time axis,
    ``(*batch, 1, obs)``. The two coincide at horizon one only (for a longer block
    the chain rule conditions on realized intermediate values that also update the
    posterior over the parameters, so they differ even for an iid model) and must
    never be compared with each other. Both reductions are ``logsumexp`` based and
    stable for densities of any magnitude; both accept host-committed or NumPy
    input.

    Parameters
    ----------
    log_density
        Per-draw log densities from `forecast_log_density()`, sample axis first.
    reduction
        ``"marginal"`` or ``"joint"`` (see `~~numpyro_forecast.typing.Reduction`).

    Returns
    -------
    Float[Array, " *batch _future _obs"]
        The per-cell score (``"marginal"``) or the block score with a size-one
        time axis (``"joint"``). A name outside the
        `~~numpyro_forecast.typing.Reduction` literal is rejected at the call
        boundary by the runtime type check.
    """
    raise NotImplementedError


def log_density_ess(
    log_density: Float[ArrayLike, " sample *batch future obs"],
    reduction: Reduction = "marginal",
) -> Float[Array, " *batch _future _obs"]:
    r"""Effective sample size of a `reduce_log_density()` estimate, as a count of draws.

    With normalized weights $v_s \propto \exp \ell^{(s)}$ (per cell for
    ``"marginal"``, per block for ``"joint"``), returns $1 / \sum_s v_s^2$, between
    one and the number of draws. A value far below the draw count means the
    average is carried by a few draws whose sampled path happens to be close to the
    realized values; the ``"joint"`` reduction over a long block is the usual case.

    Parameters
    ----------
    log_density
        Per-draw log densities from `forecast_log_density()`, sample axis first.
    reduction
        Which reduction's weights to diagnose.

    Returns
    -------
    Float[Array, " *batch _future _obs"]
        The effective sample size, laid out like the reduction's result.
    """
    raise NotImplementedError


def conditional_log_density(
    model: ForecastModel,
    posterior: Mapping[str, ArrayLike],
    data: Array,
    covariates: Array,
    truth: Num[Array, " *batch future obs"],
    *,
    batch_size: int | None = None,
    device: jax.Device | str | None = None,
) -> Float[Array | np.ndarray, " sample *batch future obs"]:
    r"""Per-draw one-step-ahead log densities of a realized block, by the chain rule.

    Runs the model on the concatenated series ``concat(data, truth)`` with a zero
    horizon (``covariates`` sliced to the in-sample length), so the ``obs`` site
    records $\log p(y_{t+j} \mid y_{\le t+j-1}, \theta^{(s)})$ for every draw, and
    returns the last ``future`` rows. For an error-feedback model built on
    `~~numpyro_forecast.models.ssoe()` this is exact and cheap: the recursion
    conditioned on the realized block is the in-sample filter extended by the block,
    and the ``"marginal"`` reduction of the result (`reduce_log_density()`) is the
    posterior-averaged chain-rule term at each step, whose sum over the block is the
    block score of `arviz_stats.loo.lfo_cv`. Takes no ``rng_key`` because the pass
    draws nothing: every sample site is substituted from ``posterior``.

    The route exists only for models whose in-sample sample sites keep their shape
    when the window grows: a model with a time-indexed latent
    (`~~numpyro_forecast.models.innovations()`, `~~numpyro_forecast.models.markov_series()`)
    has a ``drift``-like site of length ``t`` in ``posterior`` that cannot be
    substituted into a window of length ``t + future``, and the function raises
    before tracing. Use `forecast_log_density()` with the ``"joint"`` reduction for
    those.

    Parameters
    ----------
    model
        The forecasting model callable.
    posterior
        Posterior samples of every sample site of the in-sample model, sample
        axis leading.
    data
        Observed data with time at axis ``-2`` and length ``t``.
    covariates
        Covariates over at least ``t + future`` steps.
    truth
        Realized values over the ``future`` steps after ``data``.
    batch_size
        Optional chunk size for the draw axis.
    device
        Placement of the result; the contract of `draw_posterior()`.

    Returns
    -------
    Float[Array | np.ndarray, " sample *batch future obs"]
        Per-draw chain-rule log densities over the block.

    Raises
    ------
    ValueError
        If ``covariates`` does not cover ``data`` plus ``truth``, or if a site in
        ``posterior`` has a time axis of length ``t`` (a time-indexed latent).
    """
    raise NotImplementedError
