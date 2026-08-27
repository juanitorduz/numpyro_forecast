"""Distribution surgery: time-axis operations on observation distributions.

`shift_loc()`, `slice_time()`, and `prefix_condition()` are
implemented with `functools.singledispatch()` so new distribution families
can be registered without modifying call sites, the functional analogue of
Pyro's messenger-based dispatch. Elementwise families (independent per time/obs
cell) are declared with `register_elementwise()`; correlated families such
as ``MultivariateNormal`` register dedicated dispatches for all three
surgeries.
"""

from functools import singledispatch

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from numpyro_forecast.exceptions import MVNLayoutError
from numpyro_forecast.typing import Array

_ELEMENTWISE_FAMILIES: set[type[dist.Distribution]] = set()
"""Distribution families declared elementwise (independent per time/obs cell).

Membership is by exact type (subclasses do not inherit it). Elementwise
families get generic `slice_time()` and `prefix_condition()` support:
slicing a parameter along the time axis is a valid restriction, and the
forecast-horizon conditional reduces to the horizon marginal.
"""

_ELEMENTWISE_CHECKED: set[type[dist.Distribution]] = set()
"""Cache of families that have passed `_check_elementwise()` (keyed by type)."""


def register_elementwise(cls: type[dist.Distribution]) -> type[dist.Distribution]:
    """Declare a distribution family elementwise (usable as a decorator).

    An elementwise family is independent across every batch cell with an empty
    event shape, so `slice_time()` may slice its broadcast parameters along
    the time axis and `prefix_condition()` may reduce to the horizon
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


def _shift_loc_elementwise(noise_dist: dist.Distribution, loc: Array) -> dist.Distribution:
    """Rebuild ``type(noise_dist)`` with ``loc`` shifted, other parameters unchanged.

    Shared implementation for the location-family ``shift_loc`` registrations: the
    family's constructor parameters are read back from ``arg_constraints`` (each a
    distribution attribute), the ``loc`` parameter is shifted by ``loc``, and the
    distribution is reconstructed. It is registered only for the explicit families
    in `_SHIFT_LOC_FAMILIES`, so the generic ``shift_loc`` keeps refusing
    unknown families.
    """
    kwargs = {name: getattr(noise_dist, name) for name in type(noise_dist).arg_constraints}
    kwargs["loc"] = jnp.asarray(kwargs["loc"]) + loc
    return type(noise_dist)(**kwargs)


_SHIFT_LOC_FAMILIES: tuple[type[dist.Distribution], ...] = (
    dist.Normal,
    dist.StudentT,
    dist.Laplace,
    dist.Cauchy,
    dist.Gumbel,
    dist.AsymmetricLaplace,
)
"""Location-scale families that share `_shift_loc_elementwise()`."""

for _family in _SHIFT_LOC_FAMILIES:
    shift_loc.register(_family, _shift_loc_elementwise)


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


def _mvn_time_params(
    noise_dist: dist.MultivariateNormal,
) -> tuple[Array, Array]:
    """Return ``(loc, cov)`` with ``loc`` shaped ``(*batch, time)``.

    The number of time steps is read from the covariance matrix
    (``cov.shape[-1]``), which is unambiguous, rather than guessed from ``loc``.
    Supported ``loc`` layouts:

    - ``(*batch, time, 1)`` -- the trailing singleton is squeezed (checked
      first, so the rule holds even when ``time == 1``).
    - ``(*batch, time)`` -- returned unchanged.

    The remaining ``time == 1`` tie -- a 2-d ``(*batch, 1)`` -- is resolved as
    "the trailing axis is time" (kept unchanged), pinned by
    ``test_mvn_time_params_time_one_tiebreak``. The resolved loc batch must
    broadcast against the cov batch, so a mismatched construction fails here
    rather than deep in ``cho_solve``.

    Raises
    ------
    MVNLayoutError
        If ``loc`` is 0-d, its trailing axes match neither layout, or its batch
        shape does not broadcast against the covariance batch shape.
    """
    loc = jnp.asarray(noise_dist.loc)
    cov = jnp.asarray(noise_dist.covariance_matrix)
    time = cov.shape[-1]
    if loc.ndim == 0:
        raise MVNLayoutError()
    if loc.ndim >= 2 and loc.shape[-1] == 1 and loc.shape[-2] == time:
        resolved = loc[..., 0]  # (*batch, time, 1) -> (*batch, time)
    elif loc.shape[-1] == time:
        resolved = loc
    else:
        raise MVNLayoutError()
    try:
        jnp.broadcast_shapes(resolved.shape[:-1], cov.shape[:-2])
    except ValueError as err:
        raise MVNLayoutError() from err
    return resolved, cov


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
            raise MVNLayoutError()
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
