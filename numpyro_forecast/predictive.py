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
shared chunk-and-transfer loop, `~~numpyro_forecast.offload.draw_chunked()`.

`forecast()` samples forecasts over the horizon from posterior draws, and
`predict_in_sample()` samples the in-sample posterior predictive of the
``obs`` site. Each call jit-compiles a ``Predictive`` wrapper of the model once
(`_compiled_predictive()`) and drives it through the shared chunking driver
``_chunked_draws``, which caps peak memory while reusing that one program for
every chunk. Nothing is cached across calls: like ``numpyro.infer.Predictive``,
a driver call sees the model as it is at call time (including the globals,
closure cells and attributes it reads), and the model need not be hashable.
With ``device`` set (e.g. ``"host"``: pageable host memory, as jax Arrays on
the CPU backend device or as NumPy arrays when that backend is not
initialized), every chunk is moved off the accelerator before the next one is
drawn, so accelerator memory is bounded by a single chunk instead of the full
draw array. The ``device`` contract is documented once, on
`draw_posterior()`; the predictive drivers share it.
"""

from collections.abc import Callable, Mapping
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

from numpyro_forecast.offload import (
    ResolvedDevice,
    chunk_indices,
    draw_chunked,
    leaf_view,
    oom_advice,
    resolve_device,
    stitch_chunks,
    transfer,
)
from numpyro_forecast.typing import Array, ForecastModel
from numpyro_forecast.validation import (
    drop_empty_horizon_sites,
    drop_obs_site,
    require_covariates_extend_data,
    require_positive_num_samples,
    require_sample_axis,
    require_time_axis,
)


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


def _jitted_sample_posterior(guide: AutoGuide) -> Callable[..., dict[str, Array]]:
    """Return a freshly jitted ``sample_posterior`` of ``guide`` with static ``sample_shape``.

    Eagerly, ``sample_posterior`` materializes every latent and deterministic
    site (and their intermediates) at full sample size with no buffer planning,
    which is what blows accelerator memory on wide panels. Under ``jax.jit``
    XLA schedules and reuses those buffers. `draw_posterior()` builds it once
    per call and reuses it for every chunk, so a call compiles a single
    program whatever the number of chunks.

    Like the predictive drivers (`_compiled_predictive()`), nothing is cached
    across calls. An ``AutoContinuous`` guide such as ``AutoNormal`` re-runs the
    model through ``Predictive`` inside ``sample_posterior`` to recompute its
    deterministic sites (for example the ``<name>`` site of a
    `~~numpyro_forecast.reparam.time_reparam()` model), so a program cached on
    the guide would replay the first trace of the model and return stale
    deterministic draws after a global, closure value or attribute the model
    reads changes (issue #167). The guide also need not be hashable.

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
    jitted ``sample_posterior`` (`_jitted_sample_posterior()`) that the call
    compiles once and shares across its chunks; nothing is cached across calls,
    so the draws (deterministic sites included) reflect the model as it is at
    call time.

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
        `~~numpyro_forecast.offload._host_memory_kind()`).
    DevicePlatformError
        If ``device`` names a platform whose backend is not initialized (see
        `~~numpyro_forecast.offload.resolve_device()`).

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
    require_positive_num_samples(num_samples)
    is_delta = isinstance(guide, AutoDelta)
    sample = None if is_delta else _jitted_sample_posterior(guide)

    def draw_fn(key: Array, n: int) -> dict[str, Array]:
        if sample is None:
            point = guide.sample_posterior(key, params)
            return _ensure_sample_axis_for_delta(point, n)
        return sample(key, params, sample_shape=(n,))

    return draw_chunked(
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
        If ``posterior`` is empty, a leaf has no sample axis (a 0-d leaf), or
        the leaves disagree on the sample-axis length (the message names the
        offending sites).
    """
    if not posterior:
        msg = "posterior must be non-empty"
        raise ValueError(msg)
    require_sample_axis(posterior)
    sizes = {name: leaf.shape[0] for name, leaf in posterior.items()}
    distinct = set(sizes.values())
    if len(distinct) != 1:
        msg = f"posterior leaves disagree on the sample axis: {sizes}"
        raise ValueError(msg)
    return distinct.pop()


def _chunked_draws(
    rng_key: Array,
    predict_fn: Callable[[Array, Mapping[str, Array | np.ndarray]], Array],
    posterior: Mapping[str, ArrayLike],
    batch_size: int | None,
    device: ResolvedDevice = None,
) -> Array | np.ndarray:
    """Run ``predict_fn`` over fixed-size posterior chunks and stitch the draws.

    Shared chunk driver for `forecast()` and `predict_in_sample()`.
    With ``batch_size`` ``None`` (or at least the sample count) ``predict_fn``
    is called once on the full posterior with ``rng_key`` itself. Otherwise the
    posterior is gathered into wrapped index blocks of exactly ``batch_size``
    rows (`chunk_indices()`) so every chunk shares one shape and the
    jitted ``predict_fn`` compiles exactly once; one subkey is split per chunk,
    and the wrapped draws are discarded by a final slice (skipped when the
    sample count is an exact multiple of ``batch_size``, since JAX slices copy
    and nothing wrapped). Every leaf of ``posterior`` is first normalized with
    `~~numpyro_forecast.offload.leaf_view()`, so a
    host-committed posterior (the output of a previous ``device="host"`` stage)
    is gathered on the host and only the chunk reaches the accelerator. When
    ``device`` is given, every chunk is moved there (`transfer()`) before
    the next one is drawn and the stitched result lives on ``device``, bounding
    accelerator memory by a single chunk; every host target (the CPU device,
    the ``"numpy"`` sentinel, and the ``"pinned_host"`` sentinel) stitches
    through NumPy so the result never touches an accelerator. ``device`` is the
    *resolved* value: the public drivers run ``"host"`` through
    `~~numpyro_forecast.offload.resolve_device()` first.

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
    # Before `leaf_view()`: a Python scalar leaf would fail its type check unnamed.
    require_sample_axis(posterior)
    staged = {
        name: leaf_view(cast("Array | np.ndarray", leaf)) for name, leaf in posterior.items()
    }
    num = _sample_axis_size(staged)
    if batch_size is None or batch_size >= num:
        with oom_advice("predictive sampling", batch_size):
            return transfer(predict_fn(rng_key, staged), device)
    indices = chunk_indices(num, batch_size)
    keys = random.split(rng_key, len(indices))
    with oom_advice("predictive sampling", batch_size):
        chunks = [
            transfer(predict_fn(keys[i], _index_tree(staged, idx)), device)
            for i, idx in enumerate(indices)
        ]
        return stitch_chunks(chunks, num, device)


def _compiled_predictive(
    model: ForecastModel, site: str, *, parallel: bool
) -> Callable[..., Array]:
    """Return a freshly jitted ``Predictive`` of ``model`` that returns the ``site`` draws.

    The returned callable takes ``(rng_key, posterior, *model_args)`` and returns
    ``Predictive(model, ...)(rng_key, *model_args)[site]``. `forecast()` and
    `predict_in_sample()` build it once per call and reuse it for every
    ``batch_size`` chunk, so a call compiles a single program whatever the
    number of chunks.

    ``model`` is closed over, not a static ``jax.jit`` argument: it is never
    hashed (any callable model works, for example a plain ``@dataclass`` with
    ``__call__``), and no compiled program outlives the driver call. A
    module-level jit with a static model would key its cache on the model
    object and replay the first trace, so a model that reads a global, a
    closure cell or a mutable attribute would silently return stale draws after
    that value changes (issue #167). The model arguments (``data``,
    ``covariates``) stay traced jit arguments rather than closed-over constants.

    Parameters
    ----------
    model
        The forecasting model callable.
    site
        Name of the site to return (``"forecast"`` or ``"obs"``).
    parallel
        ``Predictive``'s sample-axis mapping (``vmap`` when ``True``, serial
        ``lax.map`` when ``False``).

    Returns
    -------
    Callable[..., Array]
        The jitted ``(rng_key, posterior, *model_args) -> draws`` function.
    """

    def run(
        rng_key: Array, posterior: Mapping[str, Array | np.ndarray], *model_args: Array
    ) -> Array:
        predictive = Predictive(
            model, posterior_samples=dict(posterior), return_sites=[site], parallel=parallel
        )
        return predictive(rng_key, *model_args)[site]

    return jax.jit(run)


def _require_time_steps(
    shape: tuple[int, ...], *, driver: str, site: str, steps: int, rule: str, train_rows: int
) -> None:
    """Raise unless every draw of ``site`` has ``steps`` rows at axis ``-2``.

    The drivers pass the shape from ``jax.eval_shape`` of the jitted
    predictive, which shares its trace with the call that follows, so a
    mismatch raises before anything is compiled or drawn. A model that writes
    its own sites, or keeps time latents outside
    `~~numpyro_forecast.models.innovations()` (which checks its own sites),
    can otherwise return draws of the wrong length without an error (issue
    #168).

    Parameters
    ----------
    shape
        The shape of the draws of ``site``, sample axis leading.
    driver
        Name of the calling driver, for the message.
    site
        Name of the site the draws come from (``"forecast"`` or ``"obs"``).
    steps
        The required number of time steps at axis ``-2``.
    rule
        How ``steps`` follows from the driver's arguments, for the message.
    train_rows
        The training length the posterior must come from, for the message.

    Raises
    ------
    ValueError
        If a draw has fewer than two axes or not ``steps`` rows at axis ``-2``.
    """
    per_draw = tuple(shape[1:])
    if len(per_draw) < 2:
        msg = (
            f"{driver} expects each draw of the model's {site!r} site to have shape (..., time, "
            "obs), with time at axis -2 and the observation axis at -1, but it has shape "
            f"{per_draw} per draw. A univariate series keeps a size-1 observation axis: "
            "(time, 1), not (time,)."
        )
        raise ValueError(msg)
    if per_draw[-2] != steps:
        msg = (
            f"{driver} expects the model's {site!r} site to have {steps} time steps at axis -2 "
            f"({rule}), but it has shape {per_draw} per draw. Check that time is at axis -2, "
            "that the model slices the correct rows, and that each exogenous input spans the "
            "covariates. A posterior drawn for a different training length also causes this "
            f"when its time latents are not innovations() sites: they must have {train_rows} "
            "rows."
        )
        raise ValueError(msg)


def _require_data_axes(shape: tuple[int, ...], data_shape: tuple[int, ...], steps: int) -> None:
    """Raise unless every forecast draw ends in the ``(*batch, steps, obs)`` axes of ``data``.

    Runs after `_require_time_steps()`, so only the axes other than time can
    differ. The ``"obs"`` log density broadcasts the observation distribution
    against ``data``, so a model whose predictor lacks the series axis (a
    ``(duration, 1)`` predictor over ``(time, n)`` data, without a series
    plate) or a batch axis of ``data`` scores every series in training but
    draws one forecast for all of them (issue #259). The draw includes the axes
    of the plates around the site, so a ``(duration, 1)`` predictor inside a
    size-``n`` plate at ``dim=-1`` passes. Extra leading axes pass too, for
    example a scenario axis that batched covariates add over unbatched data.

    Parameters
    ----------
    shape
        The shape of the ``"forecast"`` draws, sample axis leading.
    data_shape
        The shape of the observed data.
    steps
        The number of forecast steps.

    Raises
    ------
    ValueError
        If a draw does not end in the axes
        ``(*data_shape[:-2], steps, data_shape[-1])``.
    """
    per_draw = tuple(shape[1:])
    expected = (*data_shape[:-2], steps, data_shape[-1])
    if per_draw[-len(expected) :] == expected:
        return
    msg = (
        f"forecast expects the model's 'forecast' site to end in the axes {expected} per draw, "
        f"the axes of data (shape {tuple(data_shape)}) over {steps} steps, but it has shape "
        f"{per_draw}. The usual cause is a predictor shared by the n series of data without "
        "the series axis, which draws one forecast for all of them: call predict inside a "
        "size-n plate at dim=-1, numpyro.plate('series', n, dim=-1), or give the predictor "
        "shape (duration, n), for example jnp.broadcast_to(prediction, (h.duration, n)). A "
        "batch axis of data needs a plate at its dim or that axis on the predictor, and a "
        "size-1 batch axis that the model does not use can be removed from data."
    )
    raise ValueError(msg)


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
        Posterior samples of the latent sites of the training run, sample axis
        leading (``mcmc.get_samples()`` of the fit, or `draw_posterior()`). The
        output of a ``device="host"`` stage (CPU-committed jax leaves or NumPy
        leaves) is accepted directly. A forecast-horizon key with values (one
        that ends in ``"_future"`` or contains ``"_future_"``, or
        ``"forecast"``) raises ``ValueError``: it comes from a
        ``numpyro.infer.Predictive`` call over a horizon, whose sample sites
        would pin the horizon instead of drawing it from the prior, since the
        driver substitutes every posterior sample site by name. Such a key with
        an empty leaf (a site that the model registers on every call, recorded
        by the training run with a size-0 time axis) is dropped, so the model
        draws or computes the site as if the key were absent.
    data
        Observed data with time at axis ``-2`` and length ``t``. When
        ``posterior`` holds in-sample time latents, ``t`` must be the training
        length that ``posterior`` was drawn for; a
        `~~numpyro_forecast.models.innovations()` site of another length
        raises ``ValueError``.
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
        If ``data`` or ``covariates`` has fewer than two axes (time must be at
        axis ``-2``), if ``data`` has no time steps, if ``covariates`` does not
        extend beyond ``data`` along the time axis, if ``posterior`` is
        empty, a ``posterior`` leaf has no sample axis, or the leaves disagree
        on the sample-axis length, if a forecast-horizon key of ``posterior``
        (one that ends in ``"_future"`` or contains ``"_future_"``, or
        ``"forecast"``) has a non-empty leaf, if a ``posterior`` value of an
        `~~numpyro_forecast.models.innovations()` site does not have ``t``
        steps (a posterior drawn for a different training length), if the
        ``"forecast"`` draws do not have time at axis ``-2`` with
        ``duration - t`` steps, or if they do not have the other axes of
        ``data`` (a predictor shared by the ``n`` series of ``data`` needs a
        size-``n`` plate at ``dim=-1`` or shape ``(duration, n)``; without
        either, the model draws one forecast for all series).
    HostMemoryKindError
        If ``device="pinned_host"`` is requested on a device that exposes no
        host memory kind (see
        `~~numpyro_forecast.offload._host_memory_kind()`).
    DevicePlatformError
        If ``device`` names a platform whose backend is not initialized (see
        `~~numpyro_forecast.offload.resolve_device()`).

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
    so each call compiles the predictive exactly once, but changing
    ``batch_size`` changes the PRNG stream layout and therefore the exact
    draws. ``device`` never changes the draws, only where they live.

    Each call traces and compiles the model once and keeps no compiled program
    afterwards. In exchange, like a plain ``numpyro.infer.Predictive`` call
    (which re-runs the model every time), the draws always reflect the model
    as it is at call time (a changed global, closure value or attribute it
    reads takes effect on the next call), and ``model`` can be any callable,
    hashable or not. A repeated call with the same model and shapes, for
    example in a scenario loop, therefore compiles again; so does
    `~~numpyro_forecast.convert.to_datatree()`, which runs the drivers itself.
    """
    require_covariates_extend_data(data, covariates)
    posterior = drop_empty_horizon_sites(posterior)
    steps = covariates.shape[-2] - data.shape[-2]
    predictive = _compiled_predictive(model, "forecast", parallel=parallel)

    def predict_fn(key: Array, post: Mapping[str, Array | np.ndarray]) -> Array:
        shape = jax.eval_shape(predictive, key, post, covariates, data).shape
        _require_time_steps(
            shape,
            driver="forecast",
            site="forecast",
            steps=steps,
            rule="covariates.shape[-2] - data.shape[-2]",
            train_rows=data.shape[-2],
        )
        _require_data_axes(shape, data.shape, steps)
        return predictive(key, post, covariates, data)

    return _chunked_draws(rng_key, predict_fn, posterior, batch_size, resolve_device(device))


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
    The model gets ``data=None``, so the draws have the axes that the model
    gives ``obs``, and the driver has no data to check them against: a
    predictor shared by ``n`` series without a size-``n`` plate at ``dim=-1``
    draws one column. `forecast()` and `~~numpyro_forecast.convert.to_datatree()`
    get ``data`` and raise ``ValueError`` in that case.

    Parameters
    ----------
    rng_key
        PRNG key.
    model
        The forecasting model callable (the same one that produced ``posterior``).
    posterior
        Posterior samples of the latent sites of the training run, sample axis
        leading (``mcmc.get_samples()`` of the fit, or `draw_posterior()`). The
        output of a ``device="host"`` stage (CPU-committed jax leaves or NumPy
        leaves) is accepted directly. Forecast-horizon keys follow the rule of
        `forecast()`: one with values raises ``ValueError``, because it marks
        the output of a ``numpyro.infer.Predictive`` call over a horizon, and
        one with an empty leaf is dropped. An ``"obs"`` key is dropped too, so
        the draws come from the latent sites: ``Predictive`` substitutes every
        posterior sample site by name, and an ``"obs"`` key (the output of a
        ``Predictive`` call with ``data``, or of a prior predictive call) would
        make every draw equal to it. Only ``"obs"`` is dropped: keep the other
        sites that the model observes when it gets ``data`` (a second
        likelihood) out of ``posterior``, or they are substituted.
    covariates
        Covariates with time at axis ``-2`` spanning the observed window. Its time
        length must match the data the ``posterior`` was fit on, since the in-sample
        latent sites are sized to that window. An
        `~~numpyro_forecast.models.innovations()` site of another length raises
        ``ValueError`` however the model slices its ``"obs"`` site. For other
        latents the driver checks only that the ``"obs"`` draws have one step
        per row, and a `~~numpyro_forecast.models.markov_series()` latent
        replays a shorter posterior as a prefix and draws the remaining steps
        from the prior (a longer one raises NumPyro's ``RuntimeError``).
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
    ValueError
        If ``posterior`` is empty, a ``posterior`` leaf has no sample axis, or
        the leaves disagree on the sample-axis length, if a forecast-horizon
        key of ``posterior`` (one that ends in ``"_future"`` or contains
        ``"_future_"``, or ``"forecast"``) has a non-empty leaf, if ``"obs"``
        is the only key of ``posterior``, if
        ``covariates`` has fewer than two axes (time must be at axis ``-2``)
        or no time steps, if a ``posterior`` value of an
        `~~numpyro_forecast.models.innovations()` site does not have one step
        per ``covariates`` row (a posterior drawn for a different training
        length), or if the ``"obs"`` draws do not have time at axis ``-2`` with
        one step per ``covariates`` row.
    HostMemoryKindError
        If ``device="pinned_host"`` is requested on a device that exposes no
        host memory kind (see
        `~~numpyro_forecast.offload._host_memory_kind()`).
    DevicePlatformError
        If ``device`` names a platform whose backend is not initialized (see
        `~~numpyro_forecast.offload.resolve_device()`).

    Warns
    -----
    UserWarning
        If ``device="cpu"`` is requested and the JAX CPU backend is not
        initialized, so the draws take the NumPy path of ``"host"`` instead.

    Notes
    -----
    Like `forecast()`, each call compiles the predictive once (shared by every
    ``batch_size`` chunk) and keeps no compiled program afterwards, so the draws
    reflect the model as it is at call time and ``model`` need not be hashable.
    """
    require_time_axis("covariates", covariates)
    posterior = drop_obs_site(drop_empty_horizon_sites(posterior))
    steps = covariates.shape[-2]
    # No data argument: the model must see ``data=None`` to sample ``obs``.
    predictive = _compiled_predictive(model, "obs", parallel=parallel)

    def predict_fn(key: Array, post: Mapping[str, Array | np.ndarray]) -> Array:
        _require_time_steps(
            jax.eval_shape(predictive, key, post, covariates).shape,
            driver="predict_in_sample",
            site="obs",
            steps=steps,
            rule="one per row of the in-sample covariates",
            train_rows=steps,
        )
        return predictive(key, post, covariates)

    return _chunked_draws(rng_key, predict_fn, posterior, batch_size, resolve_device(device))
