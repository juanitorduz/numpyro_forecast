"""Optimizer/guide resolution and SVI fitting for the functional API.

:func:`resolve_optimizer` and :func:`resolve_guide` normalize the user-facing
optimizer and guide specifications, :func:`fit_svi` runs stochastic variational
inference, and :class:`SVIFit` is the frozen fit result it returns.
"""

import functools
import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import jax.numpy as jnp
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoGuide, AutoNormal
from numpyro.optim import Adam, _NumPyroOptim

from numpyro_forecast.exceptions import GuideResolutionError, OptimizerResolutionError
from numpyro_forecast.functional._validation import _require_equal_duration
from numpyro_forecast.typing import Array, ForecastModel, GuideLike, OptimizerLike

_DEFAULT_LEARNING_RATE: float = 0.01
"""Default Adam learning rate used when ``optim`` is ``None``."""


def resolve_optimizer(optim: "OptimizerLike") -> _NumPyroOptim:
    """Normalize an optimizer specification into a NumPyro optimizer.

    Accepted forms: ``None`` (``Adam(0.01)``); a finite positive scalar learning
    rate (``float``/``int``/NumPy scalar/0-d array) giving ``Adam(lr)``; an
    ``optax.GradientTransformation`` (wrapped via
    ``numpyro.optim.optax_to_numpyro``, imported lazily so optax stays a soft
    dependency); a ``_NumPyroOptim`` (returned unchanged).

    Parameters
    ----------
    optim
        The optimizer specification (see :data:`~numpyro_forecast.typing.OptimizerLike`).

    Returns
    -------
    _NumPyroOptim
        The resolved NumPyro optimizer.

    Raises
    ------
    OptimizerResolutionError
        For boolean inputs of any form, including 0-d boolean arrays (``bool`` is
        an ``int`` subclass, so a bool would silently mean ``Adam(1.0)``), and for
        any other unrecognized type; the message lists the accepted forms.
    ValueError
        For a non-finite or non-positive learning rate.
    """
    if optim is None:
        return Adam(_DEFAULT_LEARNING_RATE)
    if isinstance(optim, _NumPyroOptim):
        return optim
    if isinstance(optim, bool):
        raise OptimizerResolutionError()
    is_array_scalar = getattr(optim, "ndim", None) == 0
    if is_array_scalar and jnp.issubdtype(jnp.asarray(optim).dtype, jnp.bool_):
        raise OptimizerResolutionError()  # 0-d bool array: same silent Adam(1.0) trap
    if isinstance(optim, (int, float)) or is_array_scalar:
        lr = float(cast("float", optim))
        if not math.isfinite(lr) or lr <= 0.0:
            msg = f"learning rate must be finite and positive, got {lr}"
            raise ValueError(msg)
        return Adam(lr)
    if hasattr(optim, "init") and hasattr(optim, "update"):
        from numpyro.optim import optax_to_numpyro

        return optax_to_numpyro(optim)
    msg = (
        f"resolve_optimizer() does not support {type(optim).__name__}; pass None, "
        "a positive float learning rate, an optax.GradientTransformation, or a "
        "numpyro optimizer (_NumPyroOptim)."
    )
    raise OptimizerResolutionError(msg)


def _probe_handwritten_guide(guide: Callable[..., object]) -> None:
    """Reject callables whose signature matches a guide *factory*.

    Uses :func:`inspect.signature`: exactly one required positional parameter,
    no defaults, and no var-args raises :class:`GuideResolutionError` with the
    dual-interpretation message. Signatures :mod:`inspect` cannot resolve
    (builtins, some partials) pass the probe: it is a tripwire for the common
    mistyped-factory mistake, not a gatekeeper.

    Parameters
    ----------
    guide
        The candidate hand-written guide callable.

    Raises
    ------
    GuideResolutionError
        If ``guide`` has the single-required-positional-argument factory shape.
    """
    try:
        sig = inspect.signature(guide)
    except (ValueError, TypeError):
        return
    params = list(sig.parameters.values())
    required_positional = [
        p
        for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
    ]
    has_default = any(p.default is not p.empty for p in params)
    has_var = any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params)
    if len(required_positional) == 1 and not has_default and not has_var:
        raise GuideResolutionError()


def resolve_guide(
    guide: "GuideLike",
    model: ForecastModel,
) -> "AutoGuide | Callable[..., None]":
    """Normalize a guide specification against ``model``.

    Resolution: ``None`` -> ``AutoNormal(model)``; an ``AutoGuide`` instance ->
    returned unchanged; an ``AutoGuide`` subclass or a ``functools.partial`` of
    one -> called with ``model``; any other callable -> a hand-written guide,
    after :func:`_probe_handwritten_guide`. Anything else ->
    :class:`GuideResolutionError`.

    Parameters
    ----------
    guide
        The guide specification (see :data:`~numpyro_forecast.typing.GuideLike`).
    model
        The model the guide is built against.

    Returns
    -------
    AutoGuide | Callable[..., None]
        The resolved guide.

    Raises
    ------
    GuideResolutionError
        If ``guide`` is neither an ``AutoGuide`` (instance/subclass/partial) nor
        a callable, or if it has the mistyped-factory shape.
    """
    if guide is None:
        return AutoNormal(model)
    if isinstance(guide, AutoGuide):
        return guide
    if isinstance(guide, type) and issubclass(guide, AutoGuide):
        return guide(model)
    if isinstance(guide, functools.partial):
        target = guide.func
        if isinstance(target, type) and issubclass(target, AutoGuide):
            return cast("AutoGuide", guide(model))
    if callable(guide):
        _probe_handwritten_guide(guide)
        return cast("Callable[..., None]", guide)
    msg = (
        f"resolve_guide() does not support {type(guide).__name__}; pass None, an "
        "AutoGuide instance, an AutoGuide subclass (or functools.partial of one), "
        "or a hand-written guide function `(covariates, data=None)`."
    )
    raise GuideResolutionError(msg)


@dataclass(frozen=True)
class SVIFit:
    """The result of fitting a forecasting model with SVI.

    Attributes
    ----------
    guide
        The fitted variational guide.
    params
        The learned variational parameters.
    losses
        The ELBO loss per SVI step (shape ``(num_steps,)``).
    data
        The in-sample data the model was fit on (needed to draw from
        hand-written guides and by :func:`~numpyro_forecast.convert.to_datatree`).
        ``None`` for fits constructed without it.
    covariates
        The in-sample covariates the model was fit on. ``None`` for fits
        constructed without it.
    """

    guide: "AutoGuide | Callable[..., None]"
    params: dict[str, Array]
    losses: Array
    data: Array | None = None
    covariates: Array | None = None


def fit_svi(
    rng_key: Array,
    model: ForecastModel,
    data: Array,
    covariates: Array,
    *,
    guide: "GuideLike" = None,
    optim: "OptimizerLike" = None,
    num_steps: int = 1_001,
    num_particles: int = 1,
    progress_bar: bool = False,
    stable_update: bool = False,
) -> SVIFit:
    """Fit a forecasting model with stochastic variational inference.

    PRNG: consumes ``rng_key`` once for the SVI run; nothing is retained.

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
    guide
        Guide specification resolved by :func:`resolve_guide`: ``None``
        (``AutoNormal``), an ``AutoGuide`` instance, an ``AutoGuide`` subclass or
        ``functools.partial`` factory of one, or a hand-written guide function.
    optim
        Optimizer specification resolved by :func:`resolve_optimizer`: ``None``
        (``Adam(0.01)``), a positive scalar learning rate, an
        ``optax.GradientTransformation``, or a ``_NumPyroOptim``. For example, a
        cosine-decayed, gradient-clipped Adam::

            import optax

            schedule = optax.cosine_decay_schedule(1e-2, decay_steps=1_000)
            optim = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(schedule))

    num_steps
        Number of SVI steps.
    num_particles
        Number of ELBO particles.
    progress_bar
        Whether to display the SVI progress bar.
    stable_update
        Whether SVI skips parameter updates whose new value is non-finite
        (NumPyro's ``stable_update``).

    Returns
    -------
    SVIFit
        The fitted guide, variational parameters, loss history, and the
        in-sample ``data``/``covariates`` (kept by identity, not copied).

    Raises
    ------
    ValueError
        If ``data`` and ``covariates`` have different durations.
    """
    _require_equal_duration(data, covariates)
    resolved_guide = resolve_guide(guide, model)
    optimizer = resolve_optimizer(optim)
    svi = SVI(model, resolved_guide, optimizer, Trace_ELBO(num_particles=num_particles))
    result = svi.run(
        rng_key,
        num_steps,
        covariates,
        data,
        progress_bar=progress_bar,
        stable_update=stable_update,
    )
    return SVIFit(
        guide=resolved_guide,
        params=result.params,
        losses=result.losses,
        data=data,
        covariates=covariates,
    )
