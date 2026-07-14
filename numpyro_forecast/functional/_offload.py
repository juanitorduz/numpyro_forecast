"""Device-offload helpers shared by the chunked posterior and predictive drivers.

Both :mod:`~numpyro_forecast.functional.posterior` and
:mod:`~numpyro_forecast.functional.prediction` draw large sample arrays in
fixed-size chunks and move each chunk off the accelerator before the next one
is produced. The helpers here own the device semantics of that loop:
:func:`_resolve_device` turns the public ``device`` argument into a concrete
target (a :class:`jax.Device`, the backend-free ``"host"`` sentinel, or
``None``), :func:`_transfer` moves one chunk there and blocks, and
:func:`_stitch_chunks` concatenates the transferred chunks without pulling
them back onto the accelerator.
"""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from numpyro_forecast.exceptions import DeviceMemoryError
from numpyro_forecast.typing import Array


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
            "still referenced by earlier code, and keep results off the accelerator "
            'with device="host".'
        )
        raise DeviceMemoryError(msg) from err


def _resolve_device(device: jax.Device | str | None) -> jax.Device | Literal["host"] | None:
    """Resolve a device spec to a :class:`jax.Device` or the ``"host"`` sentinel.

    Parameters
    ----------
    device
        A device, ``"host"`` (plain host memory via :func:`jax.device_get`,
        needs no CPU backend), a platform name accepted by :func:`jax.devices`
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
                "so draws fall back to device='host' (host memory, returned as a NumPy "
                "array). Pass device='host' explicitly, or initialize the CPU backend, "
                "e.g. numpyro.set_platform('cuda,cpu'), to silence this warning.",
                UserWarning,
                stacklevel=3,
            )
            return "host"
        msg = (
            f"JAX platform {device!r} is not initialized (available platforms: {available}). "
            "Pass one of those platforms, a jax.Device, 'host' (host memory, NumPy result), "
            "or initialize the platform via jax_platforms, "
            f"e.g. numpyro.set_platform('{available[0]},{device}')."
        )
        raise ValueError(msg) from err


def _transfer(draws: Array, device: jax.Device | Literal["host"] | None) -> Array | np.ndarray:
    """Move ``draws`` to ``device`` and wait for the transfer to finish.

    Blocking makes the memory profile deterministic: the source buffer is
    released before the next chunk is drawn, so the accelerator holds at most
    one chunk of draws at a time. ``device`` ``None`` is the identity;
    ``"host"`` copies to host memory with :func:`jax.device_get` (a NumPy
    array, inherently blocking, available regardless of which JAX backends are
    initialized).

    Parameters
    ----------
    draws
        The draws to move.
    device
        Target device, ``"host"``, or ``None`` to leave ``draws`` where they
        are.

    Returns
    -------
    Array | np.ndarray
        ``draws`` committed to ``device`` (``draws`` itself when ``None``, a
        NumPy array when ``"host"``).
    """
    if device is None:
        return draws
    if device == "host":
        return jax.device_get(draws)
    return jax.device_put(draws, device).block_until_ready()


def _stitch_chunks(
    chunks: list[Array | np.ndarray],
    num_samples: int,
    device: jax.Device | Literal["host"] | None,
) -> Array | np.ndarray:
    """Concatenate transferred chunks along the sample axis and drop the overdraw.

    With ``device`` ``"host"`` the chunks are NumPy arrays and are stitched
    with :func:`numpy.concatenate` so the result never touches an accelerator;
    otherwise :func:`jax.numpy.concatenate` runs on the chunks' (committed)
    device. Fixed-size chunking overdraws in the final chunk when
    ``num_samples`` is not an exact multiple of the chunk size; the trailing
    rows are discarded by a final slice (skipped when nothing was overdrawn,
    since JAX slices copy).

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
    Array | np.ndarray
        The stitched draws with exactly ``num_samples`` leading rows (a NumPy
        array when ``device`` is ``"host"``).
    """
    if device == "host":
        stitched: Array | np.ndarray = np.concatenate(chunks, axis=0)
    else:
        stitched = jnp.concatenate(chunks, axis=0)
    return stitched if stitched.shape[0] == num_samples else stitched[:num_samples]
