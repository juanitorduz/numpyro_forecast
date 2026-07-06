"""Predictive sampling drivers for the functional API.

:func:`forecast` samples forecasts over the horizon from posterior draws, and
:func:`predict_in_sample` samples the in-sample posterior predictive of the
``obs`` site. Both drive a jitted ``Predictive`` wrapper through the shared
chunking driver ``_chunked_draws``, which caps peak memory while compiling the
predictive exactly once.
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


def _pad_posterior(posterior: dict[str, Array], batch_size: int) -> tuple[dict[str, Array], int]:
    """Pad a posterior's sample axis up to a whole multiple of ``batch_size``.

    Padding lets the chunk loop slice fixed-size ``batch_size`` blocks so
    ``_predict`` compiles exactly once regardless of ``num_samples``. The pad
    rows are wrapped-around copies of existing draws and are discarded by the
    caller's final ``[:num]`` slice, so they never affect the result.

    Parameters
    ----------
    posterior
        Posterior samples, sample axis leading (all leaves agree on that axis).
    batch_size
        Positive chunk size.

    Returns
    -------
    tuple[dict[str, Array], int]
        The padded posterior and the original (pre-pad) sample count.

    Raises
    ------
    ValueError
        If ``posterior`` is empty, ``batch_size`` is not positive, or the leaves
        disagree on the sample-axis length (the message names the offending sites).
    """
    if not posterior:
        msg = "_pad_posterior() requires a non-empty posterior"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    sizes = {name: leaf.shape[0] for name, leaf in posterior.items()}
    distinct = set(sizes.values())
    if len(distinct) != 1:
        msg = f"posterior leaves disagree on the sample axis: {sizes}"
        raise ValueError(msg)
    num = distinct.pop()
    remainder = num % batch_size
    if remainder == 0:
        return posterior, num
    pad = batch_size - remainder
    pad_indices = jnp.arange(pad) % num
    padded = {
        name: jnp.concatenate([leaf, leaf[pad_indices]], axis=0)
        for name, leaf in posterior.items()
    }
    return padded, num


def _chunked_draws(
    rng_key: Array,
    predict_fn: Callable[[Array, dict[str, Array]], Array],
    posterior: dict[str, Array],
    batch_size: int | None,
) -> Array:
    """Run ``predict_fn`` over fixed-size posterior chunks and stitch the draws.

    Shared chunk driver for :func:`forecast` and :func:`predict_in_sample`.
    With ``batch_size`` ``None`` (or at least the sample count) ``predict_fn``
    is called once on the full posterior with ``rng_key`` itself. Otherwise the
    posterior is padded to a whole multiple of ``batch_size``
    (:func:`_pad_posterior`) so every chunk shares one shape and the jitted
    ``predict_fn`` compiles exactly once; one subkey is split per chunk, and
    the pad draws are discarded by the final slice.

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

    Returns
    -------
    Array
        The stitched draws for the original (pre-pad) sample count.
    """
    num_samples = next(iter(posterior.values())).shape[0]
    if batch_size is None or batch_size >= num_samples:
        return predict_fn(rng_key, posterior)
    padded, num = _pad_posterior(posterior, batch_size)
    num_padded = next(iter(padded.values())).shape[0]
    keys = random.split(rng_key, num_padded // batch_size)
    chunks = [
        predict_fn(keys[i], _index_tree(padded, slice(start, start + batch_size)))
        for i, start in enumerate(range(0, num_padded, batch_size))
    ]
    return jnp.concatenate(chunks, axis=0)[:num]


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
    per ``(rng_key, batch_size)``. Each chunk is padded to a whole multiple of
    ``batch_size`` so the underlying ``_predict`` compiles exactly once for a
    fixed shape (the pad draws are discarded), but changing ``batch_size``
    changes the PRNG stream layout and therefore the exact draws.
    """
    _require_covariates_extend_data(data, covariates)

    def predict_fn(key: Array, post: dict[str, Array]) -> Array:
        return _predict(key, model, post, data, covariates, parallel=parallel)

    return _chunked_draws(rng_key, predict_fn, posterior, batch_size)


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

    Returns
    -------
    Num[Array, " sample *batch time obs"]
        In-sample posterior-predictive draws of the ``obs`` site.
    """

    def predict_fn(key: Array, post: dict[str, Array]) -> Array:
        return _predict_obs(key, model, post, covariates, parallel=parallel)

    return _chunked_draws(rng_key, predict_fn, posterior, batch_size)
