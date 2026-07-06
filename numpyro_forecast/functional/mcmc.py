"""Kernel resolution and MCMC fitting for the functional API.

:func:`resolve_kernel` normalizes the user-facing kernel specification,
:func:`fit_mcmc` runs MCMC, and :class:`MCMCFit` is the frozen fit result it
returns.
"""

import os.path
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import jaxtyping
from numpyro.infer import AIES, ESS, MCMC, NUTS
from numpyro.infer.mcmc import MCMCKernel

from numpyro_forecast.exceptions import KernelConfigError, KernelResolutionError
from numpyro_forecast.functional._validation import _require_equal_duration
from numpyro_forecast.typing import Array, ForecastModel, KernelLike

# Why this exists: user-facing warnings (e.g. the BlackJAX num_warmup warning)
# should point at the *caller's* line, not at a frame inside this package. A
# numeric ``stacklevel`` cannot do that reliably here because the jaxtyping
# import hook wraps every annotated function in a runtime type-check wrapper,
# inserting extra frames whose count is an implementation detail that can
# change between releases. ``warnings.warn(skip_file_prefixes=...)`` (Python
# 3.12+) instead skips every frame from these directories -- this package's
# own modules and jaxtyping's wrapper -- so attribution lands on the first
# user frame regardless of wrapper depth. The first prefix is the
# numpyro_forecast package root (this module lives one level down, in the
# functional/ subpackage, hence the double dirname).
_WARN_SKIP_PREFIXES = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    os.path.dirname(jaxtyping.__file__),
)


@dataclass(frozen=True)
class MCMCFit:
    """The result of fitting a forecasting model with MCMC.

    Attributes
    ----------
    samples
        The posterior samples of the latent sites, sample axis leading and
        stored flattened (``group_by_chain=False``).
    num_chains
        The number of chains the fit was run with (frozen; default ``1``), so
        chain structure survives into :func:`~numpyro_forecast.convert.to_datatree`.
    """

    samples: dict[str, Array]
    num_chains: int = 1


def resolve_kernel(
    kernel: "KernelLike",
    model: ForecastModel,
    kernel_kwargs: Mapping[str, Any] | None,
) -> MCMCKernel:
    """Normalize a kernel specification.

    ``None`` -> ``NUTS(model, **kernel_kwargs)`` (kwargs tune the default kernel
    without naming it); a kernel class -> ``kernel(model, **kernel_kwargs)``; a
    kernel instance -> returned unchanged, and combining an instance with
    non-empty ``kernel_kwargs`` raises ``ValueError`` (ambiguous). Anything else
    -> ``TypeError``.

    Parameters
    ----------
    kernel
        The kernel specification (see :data:`~numpyro_forecast.typing.KernelLike`).
    model
        The model the kernel is built against.
    kernel_kwargs
        Extra keyword arguments forwarded to the kernel constructor (ignored,
        and rejected, for an already-constructed instance).

    Returns
    -------
    MCMCKernel
        The resolved kernel.

    Raises
    ------
    KernelConfigError
        If a kernel instance is combined with non-empty ``kernel_kwargs``.
    KernelResolutionError
        If ``kernel`` is neither ``None``, an ``MCMCKernel`` subclass, nor an
        ``MCMCKernel`` instance.
    """
    kwargs = dict(kernel_kwargs or {})
    if kernel is None:
        return NUTS(model, **kwargs)
    if isinstance(kernel, type) and issubclass(kernel, MCMCKernel):
        factory = cast("Callable[..., MCMCKernel]", kernel)
        return factory(model, **kwargs)
    if isinstance(kernel, MCMCKernel):
        if kwargs:
            msg = (
                "kernel_kwargs cannot be combined with an already-constructed "
                "kernel instance; pass the kernel class instead, or set the "
                "options on the instance."
            )
            raise KernelConfigError(msg)
        return kernel
    msg = (
        f"resolve_kernel() does not support {type(kernel).__name__}; pass None, "
        "an MCMCKernel subclass, or an MCMCKernel instance."
    )
    raise KernelResolutionError(msg)


def _is_blackjax_kernel(kernel: MCMCKernel) -> bool:
    """Return whether ``kernel`` is a ``_BlackjaxKernel`` (by MRO name, import-free)."""
    return any(base.__name__ == "_BlackjaxKernel" for base in type(kernel).__mro__)


def _validate_kernel_run_config(
    kernel: MCMCKernel, num_chains: int, chain_method: str, num_warmup: int
) -> None:
    """Entry-point checks for constraints NumPyro surfaces late or never.

    - ``AIES``/``ESS``: require ``num_chains > 1`` and ``chain_method ==
      "vectorized"`` (the ensemble is the chain batch).
    - ``_BlackjaxKernel`` subclasses: require ``chain_method == "sequential"``
      (instance-held step/postprocess functions capture tracers under vmap/pmap
      tracing); warn when ``num_warmup > 0`` (adaptation lives in
      ``kernel.init``, so warmup steps are discarded work).

    The warmup warning is attributed to the caller of :func:`fit_mcmc` via
    ``skip_file_prefixes`` (this package's frames and jaxtyping's runtime
    type-check wrapper are skipped), which is robust to the wrapper's frame depth.

    Parameters
    ----------
    kernel
        The resolved kernel.
    num_chains
        The requested number of chains.
    chain_method
        The requested chain method (``"sequential"``/``"parallel"``/``"vectorized"``).
    num_warmup
        The requested number of warmup steps.

    Raises
    ------
    KernelConfigError
        For each violated constraint, naming the constraint and the fix.
    """
    if isinstance(kernel, (AIES, ESS)):
        name = type(kernel).__name__
        if num_chains <= 1 or chain_method != "vectorized":
            msg = (
                f"{name} is an ensemble sampler: it requires num_chains > 1 and "
                f'chain_method="vectorized" (got num_chains={num_chains}, '
                f'chain_method="{chain_method}").'
            )
            raise KernelConfigError(msg)
    if _is_blackjax_kernel(kernel):
        if chain_method != "sequential":
            msg = (
                f'{type(kernel).__name__} requires chain_method="sequential"; '
                "its step/postprocess functions capture tracers under "
                f'vmap/pmap (got chain_method="{chain_method}").'
            )
            raise KernelConfigError(msg)
        if num_warmup > 0:
            warnings.warn(
                f"{type(kernel).__name__} performs adaptation in kernel.init; "
                f"num_warmup={num_warmup} warmup steps are discarded work, pass "
                "num_warmup=0.",
                stacklevel=2,
                skip_file_prefixes=_WARN_SKIP_PREFIXES,
            )


def fit_mcmc(
    rng_key: Array,
    model: ForecastModel,
    data: Array,
    covariates: Array,
    *,
    kernel: "KernelLike" = None,
    kernel_kwargs: Mapping[str, Any] | None = None,
    num_warmup: int = 1_000,
    num_samples: int = 1_000,
    num_chains: int = 1,
    chain_method: str = "sequential",
    progress_bar: bool = False,
) -> MCMCFit:
    """Fit a forecasting model with MCMC.

    PRNG: consumed entirely by :class:`~numpyro.infer.MCMC`.

    Parameters
    ----------
    rng_key
        PRNG key for inference.
    model
        The forecasting model callable (OOP instance or functional model).
    data
        In-sample data with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2`` and the same duration as ``data``.
    kernel
        Kernel specification resolved by :func:`resolve_kernel`: ``None``
        (``NUTS``), an ``MCMCKernel`` instance, or an ``MCMCKernel`` subclass.
    kernel_kwargs
        Extra keyword arguments for the kernel constructor (only with ``None``
        or a kernel class; rejected with an instance).
    num_warmup
        Number of warmup steps.
    num_samples
        Number of posterior samples.
    num_chains
        Number of MCMC chains (stored on the returned :class:`MCMCFit`).
    chain_method
        NumPyro chain method (``"sequential"``/``"parallel"``/``"vectorized"``).
    progress_bar
        Whether to display the MCMC progress bar.

    Returns
    -------
    MCMCFit
        The posterior samples (flattened) and ``num_chains``.

    Raises
    ------
    ValueError
        If ``data`` and ``covariates`` have different durations.
    KernelConfigError
        If a run-config constraint is violated (see
        :func:`_validate_kernel_run_config`).
    """
    _require_equal_duration(data, covariates)
    resolved_kernel = resolve_kernel(kernel, model, kernel_kwargs)
    _validate_kernel_run_config(resolved_kernel, num_chains, chain_method, num_warmup)
    mcmc = MCMC(
        resolved_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    mcmc.run(rng_key, covariates, data)
    return MCMCFit(samples=mcmc.get_samples(), num_chains=num_chains)
