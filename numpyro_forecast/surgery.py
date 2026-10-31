"""Distribution surgery: time-axis operations on observation distributions.

`shift_loc()`, `slice_time()`, and `prefix_condition()` are
implemented with `functools.singledispatch()` so new distribution families
can be registered without modifying call sites, the functional analogue of
Pyro's messenger-based dispatch. Elementwise families (independent per time/obs
cell) are declared with `register_elementwise()`; correlated families such
as ``MultivariateNormal`` register dedicated dispatches for all three
surgeries. A ``MultivariateNormal`` over time has event shape ``(time,)``, with
no observation axis: `shift_loc()` with a ``(*batch, time, 1)`` predictor
returns it with event shape ``(time, 1)``, the package layout of one series,
and `slice_time()` and `prefix_condition()` keep that layout.
"""

from functools import singledispatch
from typing import cast

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions.transforms import ReshapeTransform

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
    """Shift the location parameter of a noise distribution by ``loc``.

    This converts Pyro's ``obs = data - prediction`` idiom into an additive
    shift of the observation distribution's location: the result is
    ``noise_dist`` with ``loc`` added to its location parameter and every other
    parameter unchanged. For noise with location ``0`` the result has its
    location at ``loc``, and that location is not the mean for every family:

    - ``Normal``, ``Laplace`` and ``MultivariateNormal``: ``loc`` is the mean
      (and also the median and the mode).
    - ``StudentT``: ``loc`` is the mean when ``df > 1`` (and always the median
      and the mode); for ``df <= 1`` there is no mean.
    - ``Cauchy``: ``loc`` is the mode (and the median); there is no mean.
    - ``Gumbel``: ``loc`` is the mode; the mean is
      ``loc + np.euler_gamma * scale`` (about ``loc + 0.5772 * scale``).
    - ``AsymmetricLaplace``: ``loc`` is the mode; the mean is
      ``loc + scale * (1 / asymmetry - asymmetry)``, which is ``loc`` only for
      ``asymmetry = 1`` (the symmetric case).

    An ``Independent`` wrapper follows its base distribution. When the
    predictor must be the mean of a ``Gumbel`` or ``AsymmetricLaplace``
    observation, center the noise yourself, for example
    ``dist.Gumbel(-np.euler_gamma * scale, scale)`` or
    ``dist.AsymmetricLaplace(scale * (asymmetry - 1 / asymmetry), scale, asymmetry)``.

    A ``MultivariateNormal`` over time is noise correlated across the time
    steps of one series (``obs == 1``): a ``(*batch, time)`` location such as
    ``jnp.zeros(h.duration)`` with a ``(*batch, time, time)`` covariance.
    Shifted by a predictor with the observation axis, ``(*batch, time, 1)`` as
    `~~numpyro_forecast.models.predict()` supplies it, the result is the same
    Gaussian with event shape ``(time, 1)``: ``log_prob`` of
    ``(*batch, time, 1)`` data is the joint density over time, one value per
    series, and draws keep the observation axis. The series of a panel go in
    the batch, ``(series, time, 1)``, and a size-1 time axis, ``(*batch, 1, 1)``,
    shifts every time step by the same value. A ``(*batch, time)`` predictor,
    with no more axes than the noise location, keeps the ``(time,)`` event. A link
    callable for `~~numpyro_forecast.models.predict()` gets the
    ``(time, 1)`` event with
    ``lambda mu: shift_loc(dist.MultivariateNormal(jnp.zeros(h.duration), cov), mu)``.

    Parameters
    ----------
    noise_dist
        A location-family noise distribution, usually with location ``0``.
    loc
        The deterministic predictor to add to the distribution's location.

    Returns
    -------
    dist.Distribution
        The same family with its location shifted by ``loc`` (the mean or the
        mode, depending on the family, see above). For a ``MultivariateNormal``
        and a ``(*batch, time, 1)`` predictor, a distribution with event shape
        ``(time, 1)``: its ``mean`` and ``variance`` have the observation axis,
        and its ``base_dist`` is the shifted ``MultivariateNormal`` (``loc``,
        ``covariance_matrix``, ``entropy()``).

    Raises
    ------
    NotImplementedError
        If ``noise_dist`` is of an unsupported type.
    MVNLayoutError
        If ``noise_dist`` is a ``MultivariateNormal`` and its location or
        ``loc`` matches neither layout above, for example a ``(time, obs)``
        predictor with ``obs > 1``.
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
    families register a dedicated dispatch instead: a ``MultivariateNormal``
    over time slices its time axis and keeps its layout, so the event
    ``(time, 1)`` returned by `shift_loc()` stays ``(steps, 1)`` and a bare
    ``(time,)`` event stays ``(steps,)``.

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
    a slice (the R1 fix). An ``Independent`` wrapper (e.g. from ``.to_event(1)``)
    conditions its base distribution and keeps its ``reinterpreted_batch_ndims``,
    so wrapping does not change the conditional. A ``MultivariateNormal`` over
    time keeps its layout: the event ``(time, 1)`` returned by `shift_loc()`
    gives a conditional with event ``(f, 1)``, and a bare ``(time,)`` event gives
    ``(f,)``. Its conditional factors the full covariance once, after adding
    $10^{-6}$ times the variance of each time step to the diagonal. The jitter
    is relative, so the result does not depend on the scale of the covariance,
    overall or per time step: the relative error of a forecast variance is about
    $10^{-6}$ times the ratio of its marginal to its conditional variance.

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
        If ``noise_dist`` (or the base of an ``Independent`` wrapper) is not a
        registered elementwise family (and has no dedicated dispatch).
    """
    _require_elementwise(noise_dist)
    t = data.shape[-2]
    return slice_time(noise_dist, slice(t, None))


@prefix_condition.register
def _(noise_dist: dist.Independent, data: Array) -> dist.Distribution:
    base = prefix_condition(noise_dist.base_dist, data)
    return dist.Independent(base, noise_dist.reinterpreted_batch_ndims)


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


_MVN_RELATIVE_JITTER = 1e-6
"""Jitter added to each MVN covariance diagonal entry, relative to that variance."""


def _mvn_prefix_condition(loc: Array, cov: Array, data: Array) -> dist.MultivariateNormal:
    """Gaussian conditional of an MVN over time given a prefix observation.

    With the lower Cholesky factor ``L`` of the full covariance split into the
    prefix (``p``) and future (``f``) blocks, the conditional mean is
    ``loc_f + L_fp @ solve(L_pp, x - loc_p)`` and the conditional covariance is
    ``L_ff @ L_ff.T``. This needs one factorization and never forms the Schur
    complement by subtraction, so the result stays positive definite. Before the
    factorization each variance grows by the factor ``1 + _MVN_RELATIVE_JITTER``
    (the jitter of the correlation matrix), so the conditional is invariant to
    the scale of each time step and a float32 covariance that is positive
    semidefinite to working precision still factors. A step with zero variance
    gets the jitter of the mean variance, and an all-zero covariance the jitter
    of a unit variance, so a zero noise scale gives a point forecast at the mean.
    """
    t = data.shape[-2]
    variance = jnp.diagonal(cov, axis1=-2, axis2=-1)
    mean_variance = jnp.mean(variance, axis=-1, keepdims=True)
    reference = jnp.where(variance > 0, variance, jnp.where(mean_variance > 0, mean_variance, 1.0))
    eye = jnp.eye(cov.shape[-1], dtype=cov.dtype)
    scale_tril = jnp.linalg.cholesky(cov + _MVN_RELATIVE_JITTER * reference[..., None] * eye)
    diff = (data[..., 0] - loc[..., :t])[..., None]
    chol_pp = scale_tril[..., :t, :t]
    batch = jnp.broadcast_shapes(chol_pp.shape[:-2], diff.shape[:-2])
    white = jax.scipy.linalg.solve_triangular(
        jnp.broadcast_to(chol_pp, (*batch, t, t)),
        jnp.broadcast_to(diff, (*batch, t, 1)),
        lower=True,
    )
    cond_mean = loc[..., t:] + (scale_tril[..., t:, :t] @ white)[..., 0]
    return dist.MultivariateNormal(loc=cond_mean, scale_tril=scale_tril[..., t:, t:])


class _MVNWithObsAxis(dist.TransformedDistribution):
    """A ``MultivariateNormal`` over time with a trailing size-1 observation axis.

    The package layout puts time at axis ``-2`` and the observation axis at
    ``-1``, while a ``MultivariateNormal`` over time has event shape
    ``(time,)``. This wrapper reshapes the event to ``(time, 1)``, so
    ``log_prob`` of ``(*batch, time, 1)`` data is the joint density over time
    (one value per batch cell) and draws keep the observation axis.
    `shift_loc()` returns it for a predictor with the observation axis, and
    `slice_time()` and `prefix_condition()` keep it. The wrapped distribution
    is ``mvn`` (also ``base_dist``); ``mean`` and ``variance`` are its moments
    with the observation axis restored.
    """

    def __init__(self, mvn: dist.MultivariateNormal) -> None:
        time = mvn.event_shape[-1]
        super().__init__(mvn, ReshapeTransform((time, 1), (time,)))

    @property
    def mvn(self) -> dist.MultivariateNormal:
        """The wrapped ``MultivariateNormal``, with event shape ``(time,)``."""
        # The reshape keeps the base batch and event rank, so numpyro stores the
        # MVN itself as ``base_dist`` (no ``expand`` or ``to_event`` wrapper).
        return cast("dist.MultivariateNormal", self.base_dist)

    @property
    def mean(self) -> Array:
        """The mean, shape ``(*batch, time, 1)``."""
        return jnp.asarray(self.mvn.mean)[..., None]

    @property
    def variance(self) -> Array:
        """The marginal variance of each time step, shape ``(*batch, time, 1)``."""
        return jnp.asarray(self.mvn.variance)[..., None]


def _mvn_shift(
    noise_dist: dist.MultivariateNormal, loc: Array
) -> tuple[dist.MultivariateNormal, bool]:
    """Add ``loc`` to the mean of an MVN over time and return if ``loc`` has the obs axis.

    A ``loc`` with a trailing size-1 observation axis, ``(*batch, time, 1)``
    (time may broadcast from size 1), is added along time and its batch axes
    broadcast against the noise batch. A scalar or a ``(*batch, time)`` ``loc``
    with no more axes than the noise location is added as is: the axis limit
    keeps a ``(time, obs)`` predictor with ``obs > 1`` from being read as a
    batch of series that share the time correlation.
    """
    base_loc, cov = _mvn_time_params(noise_dist)
    time = cov.shape[-1]
    if loc.ndim >= 2 and loc.shape[-1] == 1 and loc.shape[-2] in (1, time):
        return dist.MultivariateNormal(loc=base_loc + loc[..., 0], covariance_matrix=cov), True
    if loc.ndim == 0 or (loc.shape[-1] == time and loc.ndim <= base_loc.ndim):
        return dist.MultivariateNormal(loc=base_loc + loc, covariance_matrix=cov), False
    raise MVNLayoutError()


def _mvn_slice_time(noise_dist: dist.MultivariateNormal, index: slice) -> dist.MultivariateNormal:
    """Restrict an MVN over time to the time steps selected by ``index``."""
    base_loc, cov = _mvn_time_params(noise_dist)
    new_cov = cov[..., index, :][..., :, index]
    return dist.MultivariateNormal(loc=base_loc[..., index], covariance_matrix=new_cov)


@shift_loc.register
def _(noise_dist: dist.MultivariateNormal, loc: Array) -> dist.Distribution:
    shifted, obs_axis = _mvn_shift(noise_dist, loc)
    return _MVNWithObsAxis(shifted) if obs_axis else shifted


@shift_loc.register
def _(noise_dist: _MVNWithObsAxis, loc: Array) -> dist.Distribution:
    shifted, _ = _mvn_shift(noise_dist.mvn, loc)
    return _MVNWithObsAxis(shifted)


@slice_time.register
def _(noise_dist: dist.MultivariateNormal, index: slice) -> dist.Distribution:
    return _mvn_slice_time(noise_dist, index)


@slice_time.register
def _(noise_dist: _MVNWithObsAxis, index: slice) -> dist.Distribution:
    return _MVNWithObsAxis(_mvn_slice_time(noise_dist.mvn, index))


@prefix_condition.register
def _(noise_dist: dist.MultivariateNormal, data: Array) -> dist.Distribution:
    base_loc, cov = _mvn_time_params(noise_dist)
    return _mvn_prefix_condition(base_loc, cov, data)


@prefix_condition.register
def _(noise_dist: _MVNWithObsAxis, data: Array) -> dist.Distribution:
    base_loc, cov = _mvn_time_params(noise_dist.mvn)
    return _MVNWithObsAxis(_mvn_prefix_condition(base_loc, cov, data))


register_elementwise(dist.Normal)
register_elementwise(dist.StudentT)
register_elementwise(dist.Laplace)
register_elementwise(dist.Cauchy)
register_elementwise(dist.AsymmetricLaplace)
register_elementwise(dist.Gumbel)
register_elementwise(dist.Poisson)
register_elementwise(dist.NegativeBinomial2)
