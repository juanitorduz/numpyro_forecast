"""Predictive sampling drivers for the functional API.

:func:`forecast` samples forecasts over the horizon from posterior draws, and
:func:`predict_in_sample` samples the in-sample posterior predictive of the
``obs`` site. Both drive a jitted ``Predictive`` wrapper through the shared
chunking driver ``_chunked_draws``, which caps peak memory while compiling the
predictive exactly once. With ``device`` set (e.g. ``"cpu"``), every chunk is
committed to that device before the next one is drawn, so accelerator memory
is bounded by a single chunk instead of the full draw array.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Num
from numpyro.infer import Predictive

from numpyro_forecast.functional._validation import _require_covariates_extend_data
from numpyro_forecast.functional.posterior import _index_tree
from numpyro_forecast.typing import Array, ForecastModel


def _resolve_device(device: jax.Device | str | None) -> jax.Device | None:
    """Resolve a device spec to a concrete :class:`jax.Device`.

    Parameters
    ----------
    device
        A device, a platform name accepted by :func:`jax.devices` (e.g.
        ``"cpu"``, resolved to the platform's first device), or ``None``.

    Returns
    -------
    jax.Device | None
        The resolved device, or ``None`` when ``device`` is ``None``.
    """
    if device is None or isinstance(device, jax.Device):
        return device
    return jax.devices(device)[0]


def _transfer(draws: Array, device: jax.Device | None) -> Array:
    """Commit ``draws`` to ``device`` and wait for the transfer to finish.

    Blocking makes the memory profile deterministic: the source buffer is
    released before the next chunk is drawn, so the accelerator holds at most
    one chunk of draws at a time. ``device`` ``None`` is the identity.

    Parameters
    ----------
    draws
        The draws to move.
    device
        Target device, or ``None`` to leave ``draws`` where they are.

    Returns
    -------
    Array
        ``draws`` committed to ``device`` (or ``draws`` itself when ``None``).
    """
    if device is None:
        return draws
    return jax.device_put(draws, device).block_until_ready()


def _sample_axis_size(posterior: dict[str, Array]) -> int:
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
    predict_fn: Callable[[Array, dict[str, Array]], Array],
    posterior: dict[str, Array],
    batch_size: int | None,
    device: jax.Device | None = None,
) -> Array:
    """Run ``predict_fn`` over fixed-size posterior chunks and stitch the draws.

    Shared chunk driver for :func:`forecast` and :func:`predict_in_sample`.
    With ``batch_size`` ``None`` (or at least the sample count) ``predict_fn``
    is called once on the full posterior with ``rng_key`` itself. Otherwise the
    posterior is gathered into wrapped index blocks of exactly ``batch_size``
    rows (:func:`_chunk_indices`) so every chunk shares one shape and the
    jitted ``predict_fn`` compiles exactly once; one subkey is split per chunk,
    and the wrapped draws are discarded by a final slice (skipped when the
    sample count is an exact multiple of ``batch_size``, since JAX slices copy
    and nothing wrapped). When ``device`` is
    given, every chunk is committed there (:func:`_transfer`) before the next
    one is drawn and the stitched result lives on ``device``, bounding
    accelerator memory by a single chunk.

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
        Optional device each chunk (and the stitched result) is committed to.

    Returns
    -------
    Array
        The stitched draws for the original sample count.
    """
    num = _sample_axis_size(posterior)
    if batch_size is None or batch_size >= num:
        return _transfer(predict_fn(rng_key, posterior), device)
    indices = _chunk_indices(num, batch_size)
    keys = random.split(rng_key, len(indices))
    chunks = [
        _transfer(predict_fn(keys[i], _index_tree(posterior, idx)), device)
        for i, idx in enumerate(indices)
    ]
    stitched = jnp.concatenate(chunks, axis=0)
    return stitched if stitched.shape[0] == num else stitched[:num]


@partial(jax.jit, static_argnums=(1,), static_argnames=("parallel",))
def _predict(
    rng_key: Array,
    model: ForecastModel,
    posterior: dict[str, Array],
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
        model, posterior_samples=posterior, return_sites=["forecast"], parallel=parallel
    )
    return predictive(rng_key, covariates, data)["forecast"]


def forecast(
    rng_key: Array,
    model: ForecastModel,
    posterior: dict[str, Array],
    data: Array,
    covariates: Array,
    *,
    batch_size: int | None = None,
    parallel: bool = True,
    device: jax.Device | str | None = None,
) -> Num[Array, " sample *batch future obs"]:
    """Sample forecasts for the steps in ``[t, duration)`` from a posterior.

    Runs ``Predictive`` with full-horizon ``covariates`` and the in-sample
    ``data``: the in-sample latent sites are drawn from ``posterior`` while the
    ``_future`` suffix is drawn from the prior, and the ``"forecast"`` site is
    returned. The number of forecast samples equals the leading (sample) axis of
    ``posterior`` (see :func:`~numpyro_forecast.functional.posterior.draw_posterior`).

    Parameters
    ----------
    rng_key
        PRNG key.
    model
        The forecasting model callable (the same one that produced ``posterior``).
    posterior
        Posterior samples of the latent sites, sample axis leading.
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
        Optional device (or platform name like ``"cpu"``) where each chunk of
        draws is placed as soon as it is drawn and where the stitched result
        lives. With ``batch_size`` set on an accelerator, ``"cpu"`` bounds
        accelerator memory by a single chunk instead of the full
        ``(sample, future, obs)`` array; the draw values are unchanged, only
        the placement of the result. The bound requires ``batch_size`` strictly
        below the sample count: at or above it, the single-shot path runs and
        the full array is materialized on the default device before the one
        transfer. ``None`` keeps everything on the default device.

    Returns
    -------
    Num[Array, " sample *batch future obs"]
        Forecast samples over the ``future = duration - t`` horizon (floating
        point for continuous observations, integer for discrete/count models
        built with :func:`~numpyro_forecast.functional.models.predict_glm`).

    Raises
    ------
    ValueError
        If ``covariates`` does not extend beyond ``data`` along the time axis.

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

    def predict_fn(key: Array, post: dict[str, Array]) -> Array:
        return _predict(key, model, post, data, covariates, parallel=parallel)

    return _chunked_draws(rng_key, predict_fn, posterior, batch_size, _resolve_device(device))


@partial(jax.jit, static_argnums=(1,), static_argnames=("parallel",))
def _predict_obs(
    rng_key: Array,
    model: ForecastModel,
    posterior: dict[str, Array],
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
        model, posterior_samples=posterior, return_sites=["obs"], parallel=parallel
    )
    return predictive(rng_key, covariates)["obs"]


def predict_in_sample(
    rng_key: Array,
    model: ForecastModel,
    posterior: dict[str, Array],
    covariates: Array,
    *,
    batch_size: int | None = None,
    parallel: bool = True,
    device: jax.Device | str | None = None,
) -> Num[Array, " sample *batch time obs"]:
    """Sample the in-sample posterior predictive of the ``obs`` site.

    Runs ``Predictive`` with the in-sample ``covariates`` and the supplied posterior
    latent draws. Unlike :func:`forecast` there is no forecast horizon: ``covariates``
    span only the observed window, so the model's ``obs`` site is sampled at every
    step. The number of predictive samples equals the leading (sample) axis of
    ``posterior`` (see :func:`~numpyro_forecast.functional.posterior.draw_posterior`).

    Parameters
    ----------
    rng_key
        PRNG key.
    model
        The forecasting model callable (the same one that produced ``posterior``).
    posterior
        Posterior samples of the latent sites, sample axis leading.
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
        Optional device (or platform name like ``"cpu"``) where each chunk of
        draws is placed as soon as it is drawn and where the stitched result
        lives. With ``batch_size`` set on an accelerator, ``"cpu"`` bounds
        accelerator memory by a single chunk instead of the full
        ``(sample, time, obs)`` array; the draw values are unchanged, only the
        placement of the result. The bound requires ``batch_size`` strictly
        below the sample count: at or above it, the single-shot path runs and
        the full array is materialized on the default device before the one
        transfer. ``None`` keeps everything on the default device.

    Returns
    -------
    Num[Array, " sample *batch time obs"]
        In-sample posterior-predictive draws of the ``obs`` site.
    """

    def predict_fn(key: Array, post: dict[str, Array]) -> Array:
        return _predict_obs(key, model, post, covariates, parallel=parallel)

    return _chunked_draws(rng_key, predict_fn, posterior, batch_size, _resolve_device(device))
