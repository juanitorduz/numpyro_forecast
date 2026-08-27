"""Device-offload helpers shared by the chunked posterior and predictive drivers.

Both the posterior drivers (`~~numpyro_forecast.predictive.draw_posterior()`,
`~~numpyro_forecast.contrib.blackjax.pathfinder_samples()`) and the predictive
drivers in `~~numpyro_forecast.predictive` draw large sample arrays
in fixed-size chunks and move each chunk off the accelerator before the next one
is produced. The helpers here own the device semantics of that loop:
`_resolve_device()` turns the public ``device`` argument into a concrete
target (a `jax.Device`, the ``"pinned_host"`` fallback sentinel, or
``None``), `_transfer()` moves one chunk there and blocks, and
`_stitch_chunks()` concatenates the transferred chunks without pulling
them back onto the accelerator. `_draw_chunked()` is the shared chunk-and-transfer
loop itself: it owns batch-size validation, the single-shot passthrough, key
splitting, and the per-chunk draw/transfer/stitch sequence, so
`~~numpyro_forecast.predictive.draw_posterior()` and
`~~numpyro_forecast.contrib.blackjax.pathfinder_samples()` differ only in the
``draw_fn`` they supply.

The ``"host"`` target keeps results in pageable host memory. With the JAX CPU
backend initialized it commits every chunk to the CPU backend device
(``jax.devices("cpu")[0]``): a committed `jax.Array` whose
``np.asarray`` is a zero-copy view. Without a CPU backend
(``numpyro.set_platform("cuda")`` or a ``JAX_PLATFORMS`` preset restricts
``jax_platforms``) there is no pageable ``jax.Array`` container at all, since
a CUDA client offers only device and pinned host memory, so
`_resolve_device()` takes the backend-free path instead: each chunk is
copied with `jax.device_get()` and the results are NumPy arrays (the
``"numpy"`` sentinel, also requestable explicitly). Pinned host memory (a
`jax.sharding.SingleDeviceSharding` carrying a host memory kind on the
accelerator's own device, `_host_memory_kind()`) is used only on an
explicit ``device="pinned_host"``: on CUDA that pool is capped at 64 GB by
default (``XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB``) and its allocator fragments,
which is what ran a 50,000-series posterior (46.8 GiB) out of host memory when
it was the default. Either way nothing of the stitched result occupies
accelerator memory. `_is_host_resident()` is the one predicate behind the
consumers: `_leaf_view()` exposes a host-resident jax leaf as a NumPy view
(NumPy leaves pass straight through) so per-chunk gathers run on the host (a
pinned leaf gathered with a device-resident index raises a memory-space
mismatch; a CPU-committed leaf would instead drag the gather, and the jitted
predictive it feeds, onto the CPU), and `_device_view()` is its
counterpart for the unchunked metric entry points, handing the fused kernel an
accelerator-resident copy.

A host-committed jax result is not a drop-in replacement for a device array in
your own ``jnp`` code. Mixed with an *uncommitted* array (every fresh ``jnp``
array), an op runs on the CPU and returns a CPU-committed array; mixed with an
array committed to an accelerator it raises (``Received incompatible
devices``); a pinned array raises on any mix (``memory_space of all inputs
... must be the same``); a NumPy result simply behaves as NumPy. The
documented contract is: feed a host result straight into this package's own drivers
(`~~numpyro_forecast.predictive.forecast()`,
`~~numpyro_forecast.predictive.predict_in_sample()`,
`~~numpyro_forecast.convert.to_datatree()`, and every evaluation metric in
`~~numpyro_forecast.evaluate` and `~~numpyro_forecast.metrics`), which
all accept host-committed leaves in any mix with device-resident ones, or
convert it explicitly first: ``np.asarray(x)`` stays on host (no device
traffic), and ``jax.device_put(x, device)`` moves it onto an accelerator.
"""

import os
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Literal

import beartype
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from jax import random
from jax.core import Tracer
from jax.extend import backend as jex_backend
from jax.sharding import SingleDeviceSharding
from jax.typing import ArrayLike

from numpyro_forecast.exceptions import DeviceMemoryError, DevicePlatformError, HostMemoryKindError
from numpyro_forecast.typing import Array

_HOST_MEMORY_KINDS: tuple[str, ...] = ("pinned_host", "unpinned_host")
"""Host memory kinds, in preference order (``pinned_host`` is jax's offload target)."""

_ResolvedDevice = jax.Device | Literal["pinned_host", "numpy"] | None
"""A resolved placement: a device, the pinned-memory or NumPy sentinel, or ``None``.

``"numpy"`` is the backend-free path (one `jax.device_get()` per chunk,
NumPy results); ``"pinned_host"`` commits to the host memory kind of the
array's own device. Both are idempotent under `_resolve_device()`.
"""

_WARNING_SKIP_PREFIXES: tuple[str, ...] = tuple(
    os.path.dirname(os.path.abspath(path))
    for path in (
        __file__,  # numpyro_forecast/ (this file lives in the package root)
        jaxtyping.__file__,  # <site-packages>/jaxtyping/
        beartype.__file__,  # <site-packages>/beartype/
    )
)
"""Directories skipped when attributing warnings, so they point at the caller's frame.

The package installs the jaxtyping import hook (beartype-checked), which wraps
every function: a plain ``stacklevel`` lands in ``jaxtyping/_decorator.py`` and
Python's per-location deduplication of warnings never fires.
"""


def _available_platforms() -> list[str]:
    """Return every initialized JAX backend, in initialization order.

    `jax.devices()` lists only the *default* backend's devices, so it
    cannot tell whether the CPU backend exists next to an accelerator;
    `jax.extend.backend.backends()` lists all initialized backends.

    Returns
    -------
    list[str]
        Backend names such as ``["cuda", "cpu"]`` (``jax_platforms`` also
        accepts the ``gpu`` alias, so ``set_platform('gpu,cpu')`` is valid).
    """
    return list(jex_backend.backends().keys())


def _accelerator_platform(platforms: list[str]) -> str:
    """Pick the non-CPU platform named in remedies such as ``set_platform('cuda,cpu')``.

    Parameters
    ----------
    platforms
        Initialized platforms (see `_available_platforms()`).

    Returns
    -------
    str
        The first non-CPU platform, or ``"cuda"`` when none is listed (the
        message must still name a plausible accelerator).
    """
    return next((p for p in platforms if p != "cpu"), "cuda")


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
    `~~numpyro_forecast.exceptions.DeviceMemoryError` (original error
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
        An accelerator OOM reports the device budget and the ``batch_size``
        lever; a pinned host pool OOM (``Out of host memory``, reached only
        through an explicit ``device="pinned_host"``) names the pool's cap and
        points at ``device="host"`` instead, since the device budget is not
        what ran out.
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
        if "Out of host memory" in str(err):
            msg = (
                f"the pinned host memory pool ran out during the {stage} (capped at 64 GB "
                "by default on CUDA, XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB). Only "
                "device='pinned_host' (or pinned arrays of your own) uses that pool: switch "
                "to device='host', which lands results in pageable host memory (the CPU "
                f"backend device when it is initialized, NumPy arrays otherwise), {knob}, or "
                "raise the cap."
            )
            raise DeviceMemoryError(msg) from err
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

    Used by an explicit ``device="pinned_host"`` only (``device="host"`` never
    pins). ``"pinned_host"`` is picked whenever
    the backend offers it: it is the only host kind CUDA exposes, it is jax's
    standard offload target, and it takes the fast DMA path, so the CPU backend
    and CUDA agree on the memory kind of an offloaded result. Cached per
    device, since the answer is a fixed property of the backend.

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
    HostMemoryKindError
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
    raise HostMemoryKindError(msg)


def _host_sharding(x: Array) -> SingleDeviceSharding:
    """Build the host-memory sharding ``device="pinned_host"`` offloads ``x`` to.

    Derived from ``x`` itself (``x.devices()``) rather than `jax.devices()`,
    so pinning never needs a backend other than the one ``x`` already lives on
    (``numpyro.set_platform("cuda")`` leaves the CPU backend uninitialized).

    Parameters
    ----------
    x
        The array whose device supplies the offload target.

    Returns
    -------
    SingleDeviceSharding
        ``x``'s device with its host memory kind (see
        `_host_memory_kind()`).
    """
    device = next(iter(x.devices()))
    return SingleDeviceSharding(device, memory_kind=_host_memory_kind(device))


def _is_cpu_committed(leaf: Array) -> bool:
    """Whether ``leaf`` is committed to CPU backend device(s).

    This is the placement ``device="host"`` produces when the CPU backend is
    initialized: pageable host memory, a committed `jax.Array`.

    Parameters
    ----------
    leaf
        A concrete (non-tracer) array.

    Returns
    -------
    bool
        ``True`` when ``leaf`` is committed and every device it lives on is a
        CPU device.
    """
    return leaf.committed and all(d.platform == "cpu" for d in leaf.devices())


def _is_host_resident(leaf: Array) -> bool:
    """Whether ``leaf`` must be staged before it meets accelerator-side code.

    The one switch behind `_leaf_view()` and `_device_view()`. A leaf
    in a host memory kind (`_HOST_MEMORY_KINDS`, an explicit
    ``device="pinned_host"`` result) is host-resident on every backend. A CPU-committed leaf
    (the primary ``device="host"`` result) counts only when the default backend
    is an accelerator: staging exists to keep gathers and metric kernels off
    the accelerator, and on a CPU-only machine every array already lives on a
    CPU device, so staging would be pure overhead. An uncommitted leaf is never
    host-resident. ``jax.default_backend`` is looked up through the ``jax``
    module so tests can simulate an accelerator default on a CPU-only box.

    Parameters
    ----------
    leaf
        A concrete (non-tracer) array.

    Returns
    -------
    bool
        ``True`` for pinned/unpinned host leaves, and for CPU-committed leaves
        under an accelerator default backend.
    """
    if leaf.sharding.memory_kind in _HOST_MEMORY_KINDS:
        return True
    return jax.default_backend() != "cpu" and _is_cpu_committed(leaf)


def _leaf_view(leaf: Array | np.ndarray) -> Array | np.ndarray:
    """Expose ``leaf`` in the form that keeps downstream gathers off the accelerator.

    A device-resident `jax.Array` is returned unchanged. A host-resident
    one (`_is_host_resident()`: a CPU-committed leaf under an accelerator
    default backend, or a pinned leaf) is wrapped as a zero-copy NumPy view,
    which is the piece that makes chunked prediction work at all on an
    accelerator: indexing a pinned ``jax.Array`` with a device-resident index
    array raises (``memory_space of all inputs ... must be the same``), and
    indexing a CPU-committed one with an uncommitted index runs the gather on
    the CPU and returns a CPU-committed chunk that would then drag the jitted
    predictive onto the CPU, whereas indexing the NumPy view gathers on the host
    and hands only the chunk to the jitted predictive, which places it on the
    default device. Anything else (NumPy arrays, lists, scalars) is normalized
    with `numpy.asarray()`.

    A tracer is passed through untouched. Tracers are ``jax.Array`` instances
    but have no ``sharding``, and more importantly nothing about a traced value
    is host-resident, so the host staging is meaningless under ``vmap``/``jit``.
    Passing it through lets the traced computation fail where it always did
    (e.g. at the first host conversion), which is what
    `~~numpyro_forecast.evaluate._vmapped_metrics()` translates into
    `~~numpyro_forecast.exceptions.VectorizedMetricError`.

    Parameters
    ----------
    leaf
        One posterior leaf, in whatever container the caller supplied.

    Returns
    -------
    Array | np.ndarray
        ``leaf`` itself when it is a tracer or device-resident, a NumPy view
        otherwise.
    """
    if isinstance(leaf, Tracer):
        return leaf
    if isinstance(leaf, jax.Array):
        if _is_host_resident(leaf):
            return np.asarray(leaf)
        return leaf
    return np.asarray(leaf)


def _device_view(leaf: ArrayLike) -> Array:
    """Move a host-resident leaf onto the accelerator before it feeds a metric kernel.

    The counterpart to `_leaf_view()` for the *unchunked* metric entry
    points in `~~numpyro_forecast.evaluate` and
    `~~numpyro_forecast.metrics.crps_empirical()`: those run one fused
    ``jnp``/jitted kernel directly on ``pred`` and ``truth``, and a
    host-resident operand (`_is_host_resident()`) would either make that
    kernel raise (a pinned leaf: ``memory_space of all inputs ... must be the
    same``) or silently run it on the CPU (a CPU-committed leaf mixed with an
    uncommitted one). Calling this on both operands first means a metric
    accepts any mix of host-resident and device-resident inputs. A pinned leaf
    is copied back onto the device it was offloaded from (its own
    ``.devices()``, not `jax.devices()`, so this also works when the CPU
    backend was never initialized) with the ``"device"`` memory kind. A
    CPU-committed leaf becomes an *uncommitted* copy on the default device,
    built from its zero-copy NumPy view: uncommitted on purpose, so it follows
    ``jax.default_device`` and whichever other operand is committed, whereas a
    copy committed to ``jax.devices()[0]`` would raise against a ``truth``
    committed to another accelerator.

    A tracer is passed through untouched, for the same reason as
    `_leaf_view()`: it is host-resident by neither fact nor definition
    under ``vmap``/``jit``, and it has no ``.sharding`` to inspect.

    Parameters
    ----------
    leaf
        One metric operand (``pred`` or ``truth``), in whatever form the
        caller supplied it.

    Returns
    -------
    Array
        ``leaf`` itself when it is a tracer; a device-resident
        `jax.Array` otherwise (moved there first if it was
        host-committed).
    """
    if isinstance(leaf, Tracer):
        return leaf
    if isinstance(leaf, jax.Array) and _is_host_resident(leaf):
        if leaf.sharding.memory_kind in _HOST_MEMORY_KINDS:
            device = next(iter(leaf.devices()))
            try:
                return jax.device_put(leaf, SingleDeviceSharding(device, memory_kind="device"))
            except (ValueError, NotImplementedError):
                return jnp.asarray(np.asarray(leaf))
        # CPU-committed: an *uncommitted* copy on the default device, built from
        # the zero-copy NumPy view. Uncommitted on purpose: it follows
        # jax.default_device and whichever other operand is committed, whereas
        # committing to jax.devices()[0] would raise against a truth committed
        # to another accelerator.
        return jnp.asarray(np.asarray(leaf))
    return jnp.asarray(leaf)


def _resolve_device(device: jax.Device | str | None) -> _ResolvedDevice:
    """Resolve a device spec to a `jax.Device` or a resolved sentinel.

    ``"host"`` resolves to the CPU backend device (pageable host memory,
    committed jax Arrays). When that backend is not initialized (e.g. after
    ``numpyro.set_platform("cuda")`` or a ``JAX_PLATFORMS`` preset restricted
    ``jax_platforms``) it resolves to the ``"numpy"`` sentinel instead, which
    `_transfer()` implements as one `jax.device_get()` per chunk and
    `_stitch_chunks()` as `numpy.concatenate()`: pageable, backend-free,
    NumPy results, and no warning since nothing is lost but the container
    type. ``"cpu"`` is the same request made explicitly, so an unmet one warns
    (a `UserWarning` that skips the package's own frames and the
    jaxtyping/beartype wrappers, `_WARNING_SKIP_PREFIXES`, so it is
    attributed to, and deduplicated per, the user's call site) before taking
    the NumPy path. Both sentinels pass through unchanged, so a pre-resolved
    value can be forwarded (``to_datatree`` resolves once for both drivers).

    Parameters
    ----------
    device
        A device; ``"host"`` (the CPU backend device, NumPy without it);
        ``"cpu"`` (the same, warning when the backend is missing);
        ``"numpy"`` (force the backend-free NumPy path); ``"pinned_host"``
        (commit to the host memory kind of the array's own device, a pool
        capped at 64 GB by default on CUDA); another platform name accepted by
        `jax.devices()` (resolved to the platform's first device); or
        ``None``.

    Returns
    -------
    jax.Device | Literal["pinned_host", "numpy"] | None
        The resolved device, one of the two sentinels, or ``None`` when
        ``device`` is ``None``.

    Raises
    ------
    DevicePlatformError
        If ``device`` names any other platform whose backend is not
        initialized.
    """
    if device is None or isinstance(device, jax.Device):
        return device
    if device == "pinned_host":
        return "pinned_host"
    if device == "numpy":
        return "numpy"
    if device == "host":
        try:
            return jax.devices("cpu")[0]
        except RuntimeError:
            return "numpy"  # backend-free and pageable: fully functional, nothing to warn about
    if device == "cpu":
        try:
            return jax.devices("cpu")[0]
        except RuntimeError:
            platforms = _available_platforms()
            accelerator = _accelerator_platform(platforms)
            warnings.warn(
                f"the JAX CPU backend is not initialized (available platforms: {platforms}), "
                "so device='cpu' falls back to device='host': each chunk is copied to pageable "
                "host memory with jax.device_get and the draws come back as NumPy arrays. Pass "
                "device='host' explicitly, or initialize the CPU backend, e.g. "
                f"numpyro.set_platform('{accelerator},cpu'), to silence this warning.",
                UserWarning,
                skip_file_prefixes=_WARNING_SKIP_PREFIXES,
            )
            return "numpy"
    try:
        return jax.devices(device)[0]
    except RuntimeError as err:
        platforms = _available_platforms()
        msg = (
            f"JAX platform {device!r} is not initialized (available platforms: {platforms}). "
            "Pass one of those platforms, a jax.Device, 'host' (pageable host memory: the CPU "
            "device, or NumPy arrays without a CPU backend), 'numpy', 'pinned_host', or "
            f"initialize the platform via jax_platforms, e.g. "
            f"numpyro.set_platform('{platforms[0]},{device}')."
        )
        raise DevicePlatformError(msg) from err


def _transfer(draws: Array, device: _ResolvedDevice) -> Array | np.ndarray:
    """Move ``draws`` to ``device`` and wait for the transfer to finish.

    Blocking makes the memory profile deterministic: the source buffer is
    released before the next chunk is drawn, so the accelerator holds at most
    one chunk of draws at a time. ``device`` ``None`` is the identity; a
    `jax.Device` (the CPU device for ``"host"`` with a CPU backend) is
    one device-to-device copy; ``"numpy"`` copies the chunk to pageable host
    memory with `jax.device_get()` (a NumPy array, no backend needed);
    ``"pinned_host"`` commits the chunk to its own device's host memory kind
    (`_host_sharding()`), one device-to-host DMA per chunk.

    Parameters
    ----------
    draws
        The draws to move.
    device
        Target device, ``"numpy"``, ``"pinned_host"``, or ``None`` to leave
        ``draws`` where they are.

    Returns
    -------
    Array | np.ndarray
        ``draws`` on ``device`` (``draws`` itself when ``None``, a NumPy array
        when ``"numpy"``, a host-memory-kind `jax.Array` when
        ``"pinned_host"``).
    """
    if device is None:
        return draws
    if device == "numpy":
        return jax.device_get(draws)
    if device == "pinned_host":
        return jax.device_put(draws, _host_sharding(draws)).block_until_ready()
    return jax.device_put(draws, device).block_until_ready()


_ZERO_COPY_ALIGNMENT = 64
"""Byte alignment jax's CPU client requires to alias a NumPy buffer instead of copying it."""


def _aligned_empty(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Allocate an uninitialized NumPy array whose data pointer is 64-byte aligned.

    NumPy only guarantees its allocator's alignment (16 bytes on glibc's mmap'd
    large blocks and on small malloc blocks), while `jax.device_put()`
    aliases a NumPy buffer onto the CPU device only when it is aligned to
    `_ZERO_COPY_ALIGNMENT`; otherwise it copies silently. Over-allocating
    a byte buffer and viewing it at the first aligned offset makes the aliasing
    deterministic rather than a property of the platform allocator.

    Parameters
    ----------
    shape
        Shape of the array to allocate.
    dtype
        Element dtype.

    Returns
    -------
    np.ndarray
        A C-contiguous array of ``shape`` and ``dtype`` with an aligned pointer.
    """
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    raw = np.empty(nbytes + _ZERO_COPY_ALIGNMENT, dtype=np.uint8)
    offset = (-raw.ctypes.data) % _ZERO_COPY_ALIGNMENT
    return raw[offset : offset + nbytes].view(dtype).reshape(shape)


def _stitch_chunks(
    chunks: list[Array | np.ndarray],
    num_samples: int,
    device: _ResolvedDevice,
) -> Array | np.ndarray:
    """Concatenate transferred chunks along the sample axis and drop the overdraw.

    Fixed-size chunking overdraws in the final chunk when ``num_samples`` is
    not an exact multiple of the chunk size; the trailing rows are discarded by
    a final slice (skipped when nothing was overdrawn, since slices copy).

    NumPy chunks (the ``"numpy"`` path) are stitched with
    `numpy.concatenate()` and stay NumPy. On the two jax host paths
    (``"pinned_host"`` chunks, or chunks committed to CPU devices) the
    concatenation and the slice also run through NumPy: a ``jnp`` op on pinned
    arrays would copy them back into accelerator memory to run, and on
    CPU-committed chunks it would compile one executable per site shape for no
    gain. The chunks are viewed as NumPy (zero-copy, host to host), stitched
    into a 64-byte-aligned staging buffer (`_aligned_empty()`) and sliced
    there, and the single result is committed back to the chunks' sharding,
    which for the CPU device is again zero-copy: jax aliases the aligned buffer
    instead of copying it. Otherwise `jax.numpy.concatenate()` runs on the
    chunks' (committed) accelerator.

    ``chunks`` is cleared right after the concatenation, so the per-chunk
    buffers can be collected as soon as their one caller reference (this
    function's own ``chunks`` parameter, the caller is expected to hand over
    the sole reference) is dropped. The host peak is therefore chunks plus one
    stitched copy of a single site (pageable on the NumPy and CPU-device paths;
    pinned, and so counted against the pinned pool's cap, for
    ``"pinned_host"``).

    Parameters
    ----------
    chunks
        Same-shaped chunks with the sample axis leading, already transferred
        per ``device``. The caller should not keep its own reference to this
        list (or to the chunks it contains) after the call, since the list is
        cleared in place to release them as early as possible.
    num_samples
        The requested sample count the stitched result is cut back to.
    device
        The target the chunks were transferred to (selects the concatenation
        backend).

    Returns
    -------
    Array | np.ndarray
        The stitched draws with exactly ``num_samples`` leading rows, living
        wherever the chunks lived (NumPy for NumPy chunks).
    """
    first = chunks[0]
    if isinstance(first, np.ndarray):
        stitched_np = np.concatenate(chunks, axis=0)
        chunks.clear()  # drop the per-chunk copies before the caller stitches the next site
        return stitched_np if stitched_np.shape[0] == num_samples else stitched_np[:num_samples]
    if device == "pinned_host" or _is_cpu_committed(first):
        sharding = first.sharding
        views = [np.asarray(chunk) for chunk in chunks]
        total_rows = sum(view.shape[0] for view in views)
        staged = _aligned_empty((total_rows, *views[0].shape[1:]), views[0].dtype)
        np.concatenate(views, axis=0, out=staged)
        del views
        chunks.clear()  # drop the per-chunk buffers before the final device_put
        if staged.shape[0] != num_samples:
            staged = staged[:num_samples]
        return jax.device_put(staged, sharding).block_until_ready()
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
) -> dict[str, Array | np.ndarray]:
    """Draw ``num_samples`` samples from ``draw_fn`` in fixed-size, offloaded chunks.

    Shared chunk-and-transfer loop for `~~numpyro_forecast.predictive.draw_posterior()`
    and `~~numpyro_forecast.contrib.blackjax.pathfinder_samples()`. With
    ``batch_size`` ``None`` (or at least ``num_samples``) ``draw_fn`` is called once
    for the full count with ``rng_key`` itself (the single-shot passthrough).
    Otherwise ``rng_key`` is split into one subkey per chunk, ``draw_fn`` is called
    with each subkey and exactly ``batch_size`` as the sample count (so every chunk
    shares one shape and whatever ``draw_fn`` compiles internally compiles exactly
    once), every leaf of every chunk is moved to ``device`` as soon as it is drawn
    (`_transfer()`), and the final chunk's overdraw is discarded by
    `_stitch_chunks()`. Chunking is a memory knob, not a reproducibility knob:
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
        drawn; forwarded to `_resolve_device()`.
    stage
        Human-readable stage name forwarded to `_oom_advice()` for the OOM
        error message (e.g. ``"posterior drawing"``).

    Returns
    -------
    dict[str, Array | np.ndarray]
        The drawn samples with exactly ``num_samples`` leading rows (with
        ``device="host"``: leaves committed to the CPU device, or NumPy arrays
        when no CPU backend is initialized).

    Raises
    ------
    ValueError
        If ``batch_size`` is not positive.
    DeviceMemoryError
        If ``draw_fn`` fails with a device OOM (see `_oom_advice()`).
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
    chunks: list[dict[str, Array | np.ndarray]] = []
    with _oom_advice(stage, batch_size):
        for key in keys:
            # An explicit loop with pop-per-leaf keeps the accelerator peak at
            # one chunk: every device buffer of chunk k is released (each leaf
            # as soon as its copy lands on ``device``) before chunk k+1 is
            # dispatched.
            draws = dict(draw_fn(key, batch_size))
            chunks.append({name: _transfer(draws.pop(name), resolved) for name in list(draws)})
            del draws
        # Pop (not index) each leaf out of every per-chunk dict so the list
        # built for _stitch_chunks is the *only* reference to those buffers;
        # _stitch_chunks then frees it as soon as it has staged the result,
        # instead of the per-chunk dicts here keeping a second reference alive
        # for the rest of the loop.
        names = list(chunks[0])
        return {
            name: _stitch_chunks([chunk.pop(name) for chunk in chunks], num_samples, resolved)
            for name in names
        }
