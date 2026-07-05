"""Utility helpers: array shaping, distribution surgery, and seasonal features.

The distribution helpers (:func:`shift_loc`, :func:`slice_time`,
:func:`prefix_condition`) are implemented with :func:`functools.singledispatch`
so new distribution families can be registered without modifying call sites —
the functional analogue of Pyro's messenger-based dispatch.
"""

import importlib
from collections.abc import Sequence
from functools import lru_cache, singledispatch
from types import ModuleType

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from jaxtyping import Float

from numpyro_forecast.typing import Array


def require(module: str, *, extra: str) -> ModuleType:
    """Import an optional dependency, or raise a targeted ``ImportError``.

    Optional features (dataframes, blackjax, optax) live behind ``pyproject``
    extras and are never imported at package import time. This helper imports
    the backing module lazily at first use and, when it is missing, raises an
    ``ImportError`` naming the exact ``pip install`` invocation that provides it.

    Parameters
    ----------
    module
        The importable module name (e.g. ``"pandas"``).
    extra
        The ``numpyro_forecast`` extra that installs it (e.g. ``"dataframes"``).

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ImportError
        If ``module`` is not importable, with an actionable install hint.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        msg = (
            f"{module!r} is required for this feature; install it with "
            f"'pip install numpyro_forecast[{extra}]'."
        )
        raise ImportError(msg) from exc


def _api_canary(module: str, attrs: Sequence[str]) -> None:
    """Assert that ``module`` exposes every attribute in ``attrs``.

    A tripwire for optional-dependency API drift: extension modules call this at
    import (or in a dedicated canary test) so a renamed or removed upstream
    symbol fails with a precise message instead of a cryptic ``AttributeError``
    deep inside a call.

    Parameters
    ----------
    module
        The importable module name to probe.
    attrs
        Dotted attribute paths expected to resolve on the module (e.g.
        ``"vi.pathfinder.approximate"``).

    Raises
    ------
    AttributeError
        If any attribute path does not resolve; the message names the module,
        the missing path, and the installed version when available.
    """
    mod = importlib.import_module(module)
    missing: list[str] = []
    for attr in attrs:
        obj: object = mod
        for part in attr.split("."):
            if not hasattr(obj, part):
                missing.append(attr)
                break
            obj = getattr(obj, part)
    if missing:
        version = getattr(mod, "__version__", "unknown")
        msg = (
            f"{module} (version {version}) is missing expected attributes "
            f"{missing}; the pinned API surface has drifted."
        )
        raise AttributeError(msg)


def _zeros_like_data(data: Array, duration: int) -> Array:
    """Return zeros shaped like ``data`` with the time axis ``-2`` set to ``duration``.

    Shared core of :func:`zero_data_like` and
    :attr:`numpyro_forecast.functional.Horizon.zero_data`: it exposes the
    shape/dtype of the data over the full forecast horizon without leaking
    observed values into the model.
    """
    shape = (*data.shape[:-2], duration, data.shape[-1])
    return jnp.zeros(shape, dtype=data.dtype)


def zero_data_like(data: Array, covariates: Array) -> Array:
    """Return zeros shaped like ``data`` but extended to the covariate duration.

    Mirrors Pyro's ``zero_data``: it exposes the shape/dtype of the data over the
    full forecast horizon without leaking observed values into the model. The
    functional API exposes the equivalent value as
    :attr:`numpyro_forecast.functional.Horizon.zero_data`.

    Parameters
    ----------
    data
        Observed data with time at axis ``-2``, shape ``(*batch, t, obs)``.
    covariates
        Covariates with time at axis ``-2``, shape ``(*batch, duration, cov)``.

    Returns
    -------
    Array
        Zeros of shape ``(*batch, duration, obs)``.
    """
    return _zeros_like_data(data, covariates.shape[-2])


def concat_future(prefix: Array, suffix: Array, *, axis: int = -2) -> Array:
    """Concatenate in-sample and forecast-horizon arrays along the time axis.

    Parameters
    ----------
    prefix
        In-sample array.
    suffix
        Forecast-horizon array (same shape as ``prefix`` except along ``axis``).
    axis
        Time axis to concatenate along (defaults to ``-2``).

    Returns
    -------
    Array
        The concatenation of ``prefix`` and ``suffix`` along ``axis``.
    """
    return jnp.concatenate([prefix, suffix], axis=axis)


_ELEMENTWISE_FAMILIES: set[type[dist.Distribution]] = set()
"""Distribution families declared elementwise (independent per time/obs cell).

Membership is by exact type (subclasses do not inherit it). Elementwise
families get generic :func:`slice_time` and :func:`prefix_condition` support:
slicing a parameter along the time axis is a valid restriction, and the
forecast-horizon conditional reduces to the horizon marginal.
"""

_ELEMENTWISE_CHECKED: set[type[dist.Distribution]] = set()
"""Cache of families that have passed :func:`_check_elementwise` (keyed by type)."""


def register_elementwise(cls: type[dist.Distribution]) -> type[dist.Distribution]:
    """Declare a distribution family elementwise (usable as a decorator).

    An elementwise family is independent across every batch cell with an empty
    event shape, so :func:`slice_time` may slice its broadcast parameters along
    the time axis and :func:`prefix_condition` may reduce to the horizon
    marginal. Membership is by exact type; register each concrete subclass you
    rely on.

    Parameters
    ----------
    cls
        The distribution class to register.

    Returns
    -------
    type[dist.Distribution]
        ``cls`` unchanged, so this can decorate a class definition.
    """
    _ELEMENTWISE_FAMILIES.add(cls)
    return cls


def _check_elementwise(noise_dist: dist.Distribution) -> None:
    """Validate that ``noise_dist`` is genuinely elementwise (cached per type).

    Requires an empty ``event_shape`` and every constructor parameter to
    broadcast against ``batch_shape``. Passing types are cached so repeated
    slicing pays the check once.

    Parameters
    ----------
    noise_dist
        The instance to validate.

    Raises
    ------
    NotImplementedError
        If the event shape is non-empty or a parameter does not broadcast to the
        batch shape, i.e. the family was registered but is not actually elementwise.
    """
    cls = type(noise_dist)
    if cls in _ELEMENTWISE_CHECKED:
        return
    if noise_dist.event_shape != ():
        msg = (
            f"{cls.__name__} is registered elementwise but has event_shape "
            f"{noise_dist.event_shape}; elementwise families must have an empty "
            "event shape."
        )
        raise NotImplementedError(msg)
    batch = noise_dist.batch_shape
    for name in cls.arg_constraints:
        param = getattr(noise_dist, name)
        try:
            jnp.broadcast_shapes(jnp.shape(param), batch)
        except ValueError as exc:
            msg = (
                f"{cls.__name__} parameter {name!r} with shape {jnp.shape(param)} "
                f"does not broadcast to batch_shape {batch}; it is not elementwise."
            )
            raise NotImplementedError(msg) from exc
    _ELEMENTWISE_CHECKED.add(cls)


def _require_elementwise(noise_dist: dist.Distribution) -> None:
    """Raise unless ``noise_dist``'s exact type is a validated elementwise family."""
    if type(noise_dist) not in _ELEMENTWISE_FAMILIES:
        msg = (
            f"{type(noise_dist).__name__} is not a registered elementwise family; "
            "register it with numpyro_forecast.register_elementwise (if it is "
            "independent over time) or add a dedicated slice_time/prefix_condition "
            "dispatch (for correlated families such as MultivariateNormal)."
        )
        raise NotImplementedError(msg)
    _check_elementwise(noise_dist)


@singledispatch
def shift_loc(noise_dist: dist.Distribution, loc: Array) -> dist.Distribution:
    """Re-center a zero-centered noise distribution at ``loc``.

    This converts Pyro's ``obs = data - prediction`` idiom into an additive
    shift of the observation distribution's location.

    Parameters
    ----------
    noise_dist
        A zero-centered location-family distribution.
    loc
        The deterministic mean to add to the distribution's location.

    Returns
    -------
    dist.Distribution
        A distribution centered at ``loc``.

    Raises
    ------
    NotImplementedError
        If ``noise_dist`` is of an unsupported type.
    """
    msg = f"shift_loc() does not support {type(noise_dist).__name__}"
    raise NotImplementedError(msg)


@shift_loc.register
def _(noise_dist: dist.Normal, loc: Array) -> dist.Distribution:
    return dist.Normal(loc=noise_dist.loc + loc, scale=noise_dist.scale)


@shift_loc.register
def _(noise_dist: dist.StudentT, loc: Array) -> dist.Distribution:
    return dist.StudentT(df=noise_dist.df, loc=noise_dist.loc + loc, scale=noise_dist.scale)


@shift_loc.register
def _(noise_dist: dist.Laplace, loc: Array) -> dist.Distribution:
    return dist.Laplace(loc=noise_dist.loc + loc, scale=noise_dist.scale)


@shift_loc.register
def _(noise_dist: dist.Cauchy, loc: Array) -> dist.Distribution:
    return dist.Cauchy(loc=noise_dist.loc + loc, scale=noise_dist.scale)


@shift_loc.register
def _(noise_dist: dist.Gumbel, loc: Array) -> dist.Distribution:
    return dist.Gumbel(loc=noise_dist.loc + loc, scale=noise_dist.scale)


@shift_loc.register
def _(noise_dist: dist.AsymmetricLaplace, loc: Array) -> dist.Distribution:
    return dist.AsymmetricLaplace(
        loc=noise_dist.loc + loc, scale=noise_dist.scale, asymmetry=noise_dist.asymmetry
    )


@shift_loc.register
def _(noise_dist: dist.Independent, loc: Array) -> dist.Distribution:
    base = shift_loc(noise_dist.base_dist, loc)
    return dist.Independent(base, noise_dist.reinterpreted_batch_ndims)


@singledispatch
def slice_time(noise_dist: dist.Distribution, index: slice) -> dist.Distribution:
    """Slice an elementwise distribution along the time axis ``-2``.

    The default implementation handles registered elementwise families (empty
    ``event_shape``, ``batch_shape`` ending in ``(time, obs)``; e.g. ``Normal``,
    ``StudentT``, ``Poisson``) by slicing each broadcast parameter. Correlated
    families register a dedicated dispatch instead.

    Parameters
    ----------
    noise_dist
        The distribution to slice.
    index
        A ``slice`` applied to the time axis ``-2`` of the batch shape.

    Returns
    -------
    dist.Distribution
        The same distribution family restricted to the selected time steps.

    Raises
    ------
    NotImplementedError
        If ``noise_dist`` is not a registered elementwise family (and has no
        dedicated dispatch).
    """
    _require_elementwise(noise_dist)
    batch = noise_dist.batch_shape
    params = {
        name: jnp.broadcast_to(getattr(noise_dist, name), batch)[..., index, :]
        for name in type(noise_dist).arg_constraints
    }
    return type(noise_dist)(**params)


@slice_time.register
def _(noise_dist: dist.Independent, index: slice) -> dist.Distribution:
    base = slice_time(noise_dist.base_dist, index)
    return dist.Independent(base, noise_dist.reinterpreted_batch_ndims)


@singledispatch
def prefix_condition(noise_dist: dist.Distribution, data: Array) -> dist.Distribution:
    """Condition a ``(t+f)``-length distribution on a ``t``-length data prefix.

    For independent-over-time noise (the elementwise default) the conditional
    reduces to the forecast-horizon marginal, i.e. a time slice ``[t:]``. Only
    registered elementwise families take this path; correlated families (e.g.
    ``MultivariateNormal``) need a dedicated dispatch implementing a genuine
    Gaussian conditional, which is checked here rather than silently reduced to
    a slice (the R1 fix).

    Parameters
    ----------
    noise_dist
        The observation distribution over the full horizon ``(*batch, t+f, obs)``.
    data
        The observed prefix with shape ``(*batch, t, obs)``.

    Returns
    -------
    dist.Distribution
        The forecast-horizon distribution over ``(*batch, f, obs)``.

    Raises
    ------
    NotImplementedError
        If ``noise_dist`` is not a registered elementwise family (and has no
        dedicated dispatch).
    """
    _require_elementwise(noise_dist)
    t = data.shape[-2]
    return slice_time(noise_dist, slice(t, None))


@prefix_condition.register
def _(noise_dist: dist.Independent, data: Array) -> dist.Distribution:
    # Independent-over-time noise: the conditional is the horizon marginal.
    t = data.shape[-2]
    return slice_time(noise_dist, slice(t, None))


_MVN_LAYOUT_MSG = (
    "MultivariateNormal time-axis surgery requires obs == 1 with time as the "
    "leading correlation axis (loc shape ``(*batch, time)`` or "
    "``(*batch, time, 1)`` with matching ``(*batch, time, time)`` covariance)."
)


def _mvn_time_params(
    noise_dist: dist.MultivariateNormal,
) -> tuple[Array, Array]:
    """Return ``(loc, cov)`` with ``loc`` shaped ``(*batch, time)``.

    The number of time steps is read from the covariance matrix
    (``cov.shape[-1]``), which is unambiguous, rather than guessed from ``loc``.
    Supported ``loc`` layouts:

    - ``(*batch, time)`` -- returned unchanged.
    - ``(*batch, time, 1)`` -- the trailing singleton is squeezed.

    When ``time == 1`` the two layouts coincide; the tie is resolved in favor of
    "the trailing axis is time" (i.e. a trailing singleton is squeezed only when
    the preceding axis already matches ``time``).

    Raises
    ------
    NotImplementedError
        If ``loc`` is 0-d or its trailing axes match neither layout
        (:data:`_MVN_LAYOUT_MSG`).
    """
    loc = jnp.asarray(noise_dist.loc)
    cov = jnp.asarray(noise_dist.covariance_matrix)
    time = cov.shape[-1]
    if loc.ndim == 0:
        raise NotImplementedError(_MVN_LAYOUT_MSG)
    if loc.shape[-1] == time:
        return loc, cov
    if loc.shape[-1] == 1 and loc.ndim >= 2 and loc.shape[-2] == time:
        return loc[..., 0], cov  # (*batch, time, 1) -> (*batch, time)
    raise NotImplementedError(_MVN_LAYOUT_MSG)


def _symmetrize(cov: Array) -> Array:
    return 0.5 * (cov + jnp.swapaxes(cov, -1, -2))


def _mvn_jitter(cov: Array, floor: float = 1e-6) -> Array:
    dim = cov.shape[-1]
    eye = jnp.eye(dim, dtype=cov.dtype)
    return _symmetrize(cov) + floor * eye


def _mvn_prefix_condition(loc: Array, cov: Array, data: Array) -> dist.MultivariateNormal:
    """Gaussian conditional of an MVN over time given a prefix observation."""
    t = data.shape[-2]
    x = data[..., 0]
    mu_p = loc[..., :t]
    mu_f = loc[..., t:]
    sigma_pp = cov[..., :t, :t]
    sigma_pf = cov[..., :t, t:]
    sigma_fp = cov[..., t:, :t]
    sigma_ff = cov[..., t:, t:]

    sigma_pp = _mvn_jitter(sigma_pp)
    chol = jnp.linalg.cholesky(sigma_pp)
    diff = (x - mu_p)[..., None]
    # solve sigma_pp @ v = diff
    v = jax.scipy.linalg.cho_solve((chol, True), diff)
    cond_mean = mu_f + (sigma_fp @ v)[..., 0]
    # cond cov: sigma_ff - sigma_fp @ inv(sigma_pp) @ sigma_pf
    w = jax.scipy.linalg.cho_solve((chol, True), sigma_pf)
    cond_cov = _mvn_jitter(sigma_ff - sigma_fp @ w)
    return dist.MultivariateNormal(loc=cond_mean, covariance_matrix=cond_cov)


@shift_loc.register
def _(noise_dist: dist.MultivariateNormal, loc: Array) -> dist.Distribution:
    base_loc, cov = _mvn_time_params(noise_dist)
    shift = loc[..., 0] if loc.ndim >= 1 and loc.shape[-1] == 1 else loc
    # Squeeze only genuine trailing singletons; a non-size-1 extra axis is a
    # layout error, not something to silently drop.
    while shift.ndim > base_loc.ndim:
        if shift.shape[-1] != 1:
            raise NotImplementedError(_MVN_LAYOUT_MSG)
        shift = shift[..., 0]
    return dist.MultivariateNormal(loc=base_loc + shift, covariance_matrix=cov)


@slice_time.register
def _(noise_dist: dist.MultivariateNormal, index: slice) -> dist.Distribution:
    base_loc, cov = _mvn_time_params(noise_dist)
    new_loc = base_loc[..., index]
    new_cov = cov[..., index, :][..., :, index]
    return dist.MultivariateNormal(loc=new_loc, covariance_matrix=new_cov)


@prefix_condition.register
def _(noise_dist: dist.MultivariateNormal, data: Array) -> dist.Distribution:
    base_loc, cov = _mvn_time_params(noise_dist)
    return _mvn_prefix_condition(base_loc, cov, data)


register_elementwise(dist.Normal)
register_elementwise(dist.StudentT)
register_elementwise(dist.Laplace)
register_elementwise(dist.Cauchy)
register_elementwise(dist.AsymmetricLaplace)
register_elementwise(dist.Gumbel)
register_elementwise(dist.Poisson)
register_elementwise(dist.NegativeBinomial2)


@lru_cache(maxsize=128)
def _fourier_features(
    duration: int,
    period: float,
    num_terms: int,
) -> Float[Array, " duration two_num_terms"]:
    """Memoized Fourier-feature core.

    The design matrix is fully determined by ``(duration, period, num_terms)``
    and is built host-side, never inside a trace, so the result is cached per
    argument tuple rather than recomputed (see :func:`fourier_features`).
    """
    time = jnp.arange(duration)[:, None]
    harmonics = jnp.arange(1, num_terms + 1)[None, :]
    angles = 2.0 * jnp.pi * harmonics * time / period
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def fourier_features(
    duration: int,
    period: float,
    num_terms: int,
) -> Float[Array, " duration two_num_terms"]:
    """Build a Fourier seasonality design matrix.

    Parameters
    ----------
    duration
        Number of time steps.
    period
        Seasonal period (in time steps).
    num_terms
        Number of harmonics; the output has ``2 * num_terms`` columns
        (sine then cosine).

    Returns
    -------
    Float[Array, "duration two_num_terms"]
        The design matrix of shape ``(duration, 2 * num_terms)``.
    """
    return _fourier_features(duration, float(period), num_terms)


def periodic_repeat(x: ArrayLike, duration: int, *, axis: int = -1) -> Array:
    """Tile a seasonal pattern to cover ``duration`` time steps.

    Parameters
    ----------
    x
        Seasonal pattern; the repeated axis has length equal to the period.
        Accepts any array-like (e.g. a raw ``numpyro.sample`` draw).
    duration
        Target length along ``axis``.
    axis
        Axis to repeat along (defaults to ``-1``).

    Returns
    -------
    Array
        ``x`` periodically repeated to length ``duration`` along ``axis``.
    """
    array = jnp.asarray(x)
    period = array.shape[axis]
    indices = jnp.arange(duration) % period
    return jnp.take(array, indices, axis=axis)
