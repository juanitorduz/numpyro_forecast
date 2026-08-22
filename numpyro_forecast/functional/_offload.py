"""Device-offload helpers shared by the chunked posterior and predictive drivers.

Both the posterior drivers (:func:`~numpyro_forecast.functional.posterior.draw_posterior`,
:func:`~numpyro_forecast.contrib.blackjax.pathfinder_samples`) and the predictive
drivers in :mod:`~numpyro_forecast.functional.prediction` draw large sample arrays
in fixed-size chunks and move each chunk off the accelerator before the next one
is produced. The helpers here own the device semantics of that loop:
:func:`_resolve_device` turns the public ``device`` argument into a concrete
target (a :class:`jax.Device`, the backend-free ``"host"`` sentinel, or
``None``), :func:`_transfer` moves one chunk there and blocks, and
:func:`_stitch_chunks` concatenates the transferred chunks without pulling
them back onto the accelerator. :func:`_draw_chunked` is the shared chunk-and-transfer
loop itself: it owns batch-size validation, the single-shot passthrough, key
splitting, and the per-chunk draw/transfer/stitch sequence, so
:func:`~numpyro_forecast.functional.posterior.draw_posterior` and
:func:`~numpyro_forecast.contrib.blackjax.pathfinder_samples` differ only in the
``draw_fn`` they supply.

The ``"host"`` target keeps results in host memory while keeping them
:class:`jax.Array` values: a chunk is committed to a
:class:`jax.sharding.SingleDeviceSharding` carrying a host memory kind
(:func:`_host_memory_kind`, ``"pinned_host"`` when the backend offers it), so
nothing of the stitched result occupies accelerator memory and the whole
pipeline stays jax-native. :func:`_leaf_view` is the counterpart used by the
chunk drivers when they *consume* such arrays: it exposes a host-committed leaf
as a NumPy view so per-chunk gathers run on the host instead of dragging the
full leaf back onto the accelerator. Anyone who needs plain pageable host memory
can still call ``np.asarray(draws)`` on the result.
"""

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.sharding import SingleDeviceSharding

from numpyro_forecast.exceptions import DeviceMemoryError
from numpyro_forecast.typing import Array

_HOST_MEMORY_KINDS: tuple[str, ...] = ("pinned_host", "unpinned_host")
"""Host memory kinds, in preference order (``pinned_host`` is jax's offload target)."""


def _memory_budget_line() -> str:
    """Summarize the default device's memory budget for OOM error messages.

    Returns
    -------
    str
        One line with ``bytes_limit`` / ``bytes_in_use`` / ``peak_bytes_in_use``
        in GiB, or a placeholder when the backend exposes no memory statistics
        (e.g. the CPU backend).
    """
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:  # diagnostics must never mask the OOM itself
        stats = None
    if not stats:
        return "device memory statistics are unavailable on this backend"
    parts = [
        f"{key}={stats[key] / 2**30:.2f} GiB"
        for key in ("bytes_limit", "bytes_in_use", "peak_bytes_in_use")
        if key in stats
    ]
    return "device memory budget: " + ", ".join(parts)


@contextmanager
def _oom_advice(stage: str, batch_size: int | None) -> Iterator[None]:
    """Re-raise device OOM errors from ``stage`` with the budget and the lever.

    Anything whose message does not contain ``RESOURCE_EXHAUSTED`` propagates
    untouched. A device OOM is re-raised as
    :class:`~numpyro_forecast.exceptions.DeviceMemoryError` (original error
    chained) reporting the stage, the active ``batch_size``, the device memory
    budget, and the actionable knobs.

    Parameters
    ----------
    stage
        Human-readable name of the sampling stage (e.g. ``"posterior
        drawing"``), used in the error message.
    batch_size
        The chunk size active in the failing stage, or ``None`` when the
        single-shot path ran.

    Yields
    ------
    None
        Context in which the sampling stage runs.

    Raises
    ------
    DeviceMemoryError
        When the wrapped stage fails with an XLA ``RESOURCE_EXHAUSTED`` error.
    """
    try:
        yield
    except DeviceMemoryError:
        raise
    except Exception as err:
        if "RESOURCE_EXHAUSTED" not in str(err):
            raise
        knob = (
            f"lower batch_size (currently {batch_size})"
            if batch_size is not None
            else "set batch_size to sample in chunks"
        )
        msg = (
            f"the accelerator ran out of memory during the {stage}; "
            f"{_memory_budget_line()}. The per-chunk footprint scales linearly with "
            f"batch_size times the panel width, so {knob}, free large device arrays "
            "still referenced by earlier code, and keep results in host memory "
            'with device="host".'
        )
        raise DeviceMemoryError(msg) from err


@lru_cache(maxsize=32)
def _host_memory_kind(device: jax.Device) -> str:
    """Return the host memory kind ``device`` exposes, preferring pinned memory.

    ``"pinned_host"`` is picked whenever the backend offers it: it is the only
    host kind CUDA exposes, it is jax's standard offload target, and it takes
    the fast DMA path, so the CPU backend and CUDA agree on the memory kind of
    an offloaded result. Cached per device, since the answer is a fixed
    property of the backend.

    Parameters
    ----------
    device
        The device whose addressable memories are inspected.

    Returns
    -------
    str
        ``"pinned_host"`` when available, otherwise ``"unpinned_host"``.

    Raises
    ------
    RuntimeError
        If ``device`` addresses no host memory kind at all.
    """
    kinds = {memory.kind for memory in device.addressable_memories()}
    for kind in _HOST_MEMORY_KINDS:
        if kind in kinds:
            return kind
    msg = (
        f"device {device} exposes no host memory kind (addressable kinds: {sorted(kinds)}), "
        'so device="host" cannot commit results to host memory. Pass a jax.Device or a '
        "platform name (e.g. device='cpu') instead, or call np.asarray on the result."
    )
    raise RuntimeError(msg)


def _host_sharding(x: Array) -> SingleDeviceSharding:
    """Build the host-memory sharding that ``x`` is offloaded to.

    Derived from ``x`` itself (``x.devices()``) rather than :func:`jax.devices`,
    so the offload never needs a backend other than the one ``x`` already lives
    on: ``numpyro.set_platform("cuda")`` leaves the CPU backend uninitialized,
    and the host path must still work there.

    Parameters
    ----------
    x
        The array whose device supplies the offload target.

    Returns
    -------
    SingleDeviceSharding
        ``x``'s device with its host memory kind (see
        :func:`_host_memory_kind`).
    """
    device = next(iter(x.devices()))
    return SingleDeviceSharding(device, memory_kind=_host_memory_kind(device))


def _leaf_view(leaf: Array | np.ndarray) -> Array | np.ndarray:
    """Expose ``leaf`` in the form that keeps downstream gathers off the accelerator.

    A device-resident :class:`jax.Array` is returned unchanged. A
    host-committed one (a memory kind in :data:`_HOST_MEMORY_KINDS`, as produced
    by ``device="host"``) is wrapped as a NumPy array, which is the piece that
    keeps chunked prediction memory-bounded on an accelerator: indexing a
    host-committed ``jax.Array`` would copy the whole leaf back into device
    memory before the gather, whereas indexing the NumPy view gathers on the
    host and hands only the chunk to the jitted predictive. Anything else
    (NumPy arrays, lists, scalars) is normalized with :func:`numpy.asarray`.

    Parameters
    ----------
    leaf
        One posterior leaf, in whatever container the caller supplied.

    Returns
    -------
    Array | np.ndarray
        ``leaf`` itself when it is device-resident, a NumPy view otherwise.
    """
    if isinstance(leaf, jax.Array):
        if leaf.sharding.memory_kind in _HOST_MEMORY_KINDS:
            return np.asarray(leaf)
        return leaf
    return np.asarray(leaf)


def _resolve_device(device: jax.Device | str | None) -> jax.Device | Literal["host"] | None:
    """Resolve a device spec to a :class:`jax.Device` or the ``"host"`` sentinel.

    Parameters
    ----------
    device
        A device, ``"host"`` (host memory on the array's own device, needs no
        extra backend), a platform name accepted by :func:`jax.devices`
        (e.g. ``"cpu"``, resolved to the platform's first device), or ``None``.

    Returns
    -------
    jax.Device | Literal["host"] | None
        The resolved device, ``"host"``, or ``None`` when ``device`` is
        ``None``. ``"cpu"`` also resolves to ``"host"`` (with a
        :class:`UserWarning`) when the CPU backend is not initialized, e.g.
        after ``numpyro.set_platform("cuda")`` restricted ``jax_platforms``.

    Raises
    ------
    ValueError
        If ``device`` names any other platform whose backend is not
        initialized.
    """
    if device is None or isinstance(device, jax.Device):
        return device
    if device == "host":
        return "host"
    try:
        return jax.devices(device)[0]
    except RuntimeError as err:
        available = [d.platform for d in jax.devices()]
        if device == "cpu":
            warnings.warn(
                f"the JAX CPU backend is not initialized (available platforms: {available}), "
                "so draws fall back to device='host' (returned as jax Arrays committed to "
                "host memory). Pass device='host' explicitly, or initialize the CPU backend, "
                "e.g. numpyro.set_platform('cuda,cpu'), to silence this warning.",
                UserWarning,
                stacklevel=3,
            )
            return "host"
        msg = (
            f"JAX platform {device!r} is not initialized (available platforms: {available}). "
            "Pass one of those platforms, a jax.Device, 'host' (jax Arrays committed to host "
            "memory), or initialize the platform via jax_platforms, "
            f"e.g. numpyro.set_platform('{available[0]},{device}')."
        )
        raise ValueError(msg) from err


def _transfer(draws: Array, device: jax.Device | Literal["host"] | None) -> Array:
    """Move ``draws`` to ``device`` and wait for the transfer to finish.

    Blocking makes the memory profile deterministic: the source buffer is
    released before the next chunk is drawn, so the accelerator holds at most
    one chunk of draws at a time. ``device`` ``None`` is the identity;
    ``"host"`` commits the chunk to its own device's host memory
    (:func:`_host_sharding`), one device-to-host DMA per chunk, which needs no
    backend beyond the one ``draws`` already live on.

    Parameters
    ----------
    draws
        The draws to move.
    device
        Target device, ``"host"``, or ``None`` to leave ``draws`` where they
        are.

    Returns
    -------
    Array
        ``draws`` committed to ``device`` (``draws`` itself when ``None``, a
        host-memory-kind :class:`jax.Array` when ``"host"``).
    """
    if device is None:
        return draws
    if device == "host":
        return jax.device_put(draws, _host_sharding(draws)).block_until_ready()
    return jax.device_put(draws, device).block_until_ready()


def _stitch_chunks(
    chunks: list[Array],
    num_samples: int,
    device: jax.Device | Literal["host"] | None,
) -> Array:
    """Concatenate transferred chunks along the sample axis and drop the overdraw.

    Fixed-size chunking overdraws in the final chunk when ``num_samples`` is
    not an exact multiple of the chunk size; the trailing rows are discarded by
    a final slice (skipped when nothing was overdrawn, since slices copy).

    With ``device`` ``"host"`` the concatenation and the slice deliberately run
    through NumPy: a ``jnp`` op on host-committed arrays would copy them back
    into accelerator memory to run, which is exactly what the host path exists
    to avoid. The chunks are viewed as NumPy (host to host), stitched and
    sliced there, and the single result is committed back to the chunks' host
    sharding. Otherwise :func:`jax.numpy.concatenate` runs on the chunks'
    (committed) device.

    Parameters
    ----------
    chunks
        Same-shaped chunks with the sample axis leading, already transferred
        per ``device``.
    num_samples
        The requested sample count the stitched result is cut back to.
    device
        The target the chunks were transferred to (selects the concatenation
        backend).

    Returns
    -------
    Array
        The stitched draws with exactly ``num_samples`` leading rows, living
        wherever the chunks lived.
    """
    if device == "host":
        staged = np.concatenate([np.asarray(chunk) for chunk in chunks], axis=0)
        if staged.shape[0] != num_samples:
            staged = staged[:num_samples]
        return jax.device_put(staged, chunks[0].sharding)
    stitched = jnp.concatenate(chunks, axis=0)
    return stitched if stitched.shape[0] == num_samples else stitched[:num_samples]


def _draw_chunked(
    rng_key: Array,
    draw_fn: Callable[[Array, int], dict[str, Array]],
    num_samples: int,
    *,
    batch_size: int | None,
    device: jax.Device | str | None,
    stage: str,
) -> dict[str, Array]:
    """Draw ``num_samples`` samples from ``draw_fn`` in fixed-size, offloaded chunks.

    Shared chunk-and-transfer loop for :func:`~numpyro_forecast.functional.posterior.draw_posterior`
    and :func:`~numpyro_forecast.contrib.blackjax.pathfinder_samples`. With
    ``batch_size`` ``None`` (or at least ``num_samples``) ``draw_fn`` is called once
    for the full count with ``rng_key`` itself (the single-shot passthrough).
    Otherwise ``rng_key`` is split into one subkey per chunk, ``draw_fn`` is called
    with each subkey and exactly ``batch_size`` as the sample count (so every chunk
    shares one shape and whatever ``draw_fn`` compiles internally compiles exactly
    once), every leaf of every chunk is moved to ``device`` as soon as it is drawn
    (:func:`_transfer`), and the final chunk's overdraw is discarded by
    :func:`_stitch_chunks`. Chunking is a memory knob, not a reproducibility knob:
    draws are reproducible per ``(rng_key, batch_size)``, but changing ``batch_size``
    changes the PRNG stream layout and therefore the exact draws.

    Parameters
    ----------
    rng_key
        PRNG key; consumed directly when unchunked, split per chunk otherwise.
    draw_fn
        ``(rng_key, n) -> samples`` drawing exactly ``n`` samples of every latent
        site, sample axis leading.
    num_samples
        Total number of samples to draw.
    batch_size
        Optional chunk size (caps peak memory); ``None`` or at least
        ``num_samples`` takes the single-shot passthrough.
    device
        Where each chunk (and the single-shot result) is moved as soon as it is
        drawn; forwarded to :func:`_resolve_device`.
    stage
        Human-readable stage name forwarded to :func:`_oom_advice` for the OOM
        error message (e.g. ``"posterior drawing"``).

    Returns
    -------
    dict[str, Array]
        The drawn samples with exactly ``num_samples`` leading rows (leaves
        committed to host memory when ``device`` resolves to ``"host"``).

    Raises
    ------
    ValueError
        If ``batch_size`` is not positive.
    DeviceMemoryError
        If ``draw_fn`` fails with a device OOM (see :func:`_oom_advice`).
    """
    resolved = _resolve_device(device)
    if batch_size is not None and batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    if batch_size is None or batch_size >= num_samples:
        with _oom_advice(stage, batch_size):
            samples = draw_fn(rng_key, num_samples)
            return {name: _transfer(leaf, resolved) for name, leaf in samples.items()}
    num_chunks = -(-num_samples // batch_size)  # ceil; the last chunk overdraws
    keys = random.split(rng_key, num_chunks)
    chunks: list[dict[str, Array]] = []
    with _oom_advice(stage, batch_size):
        for key in keys:
            # An explicit loop with pop-per-leaf keeps the accelerator peak at
            # one chunk: every device buffer of chunk k is released (each leaf
            # as soon as its copy lands on ``device``) before chunk k+1 is
            # dispatched.
            draws = dict(draw_fn(key, batch_size))
            chunks.append({name: _transfer(draws.pop(name), resolved) for name in list(draws)})
            del draws
        return {
            name: _stitch_chunks([chunk[name] for chunk in chunks], num_samples, resolved)
            for name in chunks[0]
        }
