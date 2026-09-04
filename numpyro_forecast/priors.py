r"""Shrinkage prior helpers that return distribution moments as arrays.

The helpers here build the location and scale arrays of a prior and leave the
family to the caller (``dist.Normal``, ``dist.StudentT``, a hierarchical scale,
...), so hyperparameters can themselves be sampled. They are independent of the
model-side modules: `minnesota_prior()` shares only the ``(lags, obs, obs)``
coefficient layout with `~~numpyro_forecast.var.var_step()` and imports
nothing from it.
"""

from typing import Literal

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float

from numpyro_forecast.typing import Array


def minnesota_prior(
    n_lags: int,
    n_obs: int,
    tightness: ArrayLike,
    *,
    cross_shrinkage: ArrayLike = 0.5,
    decay: Literal["harmonic", "geometric"] = "harmonic",
    own_lag_mean: ArrayLike = 1.0,
) -> tuple[Float[Array, " lags obs obs"], Float[Array, " lags obs obs"]]:
    r"""Return the Minnesota (Litterman) prior moments for VAR coefficients.

    The prior shrinks the coefficient of variable $j$ at lag $l$ in the
    equation of variable $i$ toward a mean that is nonzero only for the first
    own lag, with a standard deviation that tightens on longer lags and on
    cross-variable lags:

    $$
    m_{l, ij} = \begin{cases} m_{\text{own}} & l = 1,\ i = j \\ 0 & \text{otherwise} \end{cases},
    \qquad
    s_{l, ij} = \lambda \, d(l) \, \begin{cases} 1 & i = j \\ \kappa & i \neq j \end{cases},
    $$

    with $d(l) = 1/l$ (``decay="harmonic"``) or $d(l) = 1/l^2$
    (``decay="geometric"``, the label Impulso uses; in Doan, Litterman and Sims
    it is the harmonic decay with exponent two). $m_{\text{own}} = 1$ is the
    random-walk belief for series in levels; use ``own_lag_mean=0.0`` for
    differenced or otherwise stationary series. The parameterization follows
    Impulso's ``MinnesotaPrior`` (fixed tightness rather than an estimated one,
    no residual-scale ratios), with ``own_lag_mean`` as the one addition. The
    classic Litterman scale ratio for the coefficient of variable $j$ in
    equation $i$ is $\sigma_i / \sigma_j$; apply it, when the series are on
    different scales, as ``scale * (sigma[:, None] / sigma[None, :])``.

    Parameters
    ----------
    n_lags
        Number of lags $p \ge 1$.
    n_obs
        Number of series $k \ge 1$.
    tightness
        Overall shrinkage $\lambda > 0$ (the standard deviation of the first own
        lag). A modeling choice, so it has no default; a jax scalar is accepted,
        which lets a model sample it.
    cross_shrinkage
        Relative shrinkage $\kappa \in [0, 1]$ of cross-variable lags versus own
        lags (``1.0`` treats them alike, ``0.0`` pins them to the mean).
    decay
        Lag decay $d(l)$: ``"harmonic"`` for $1/l$, ``"geometric"`` for $1/l^2$.
    own_lag_mean
        Prior mean $m_{\text{own}}$ of the first own lag.

    Returns
    -------
    loc : Float[Array, "lags obs obs"]
        Prior means in the ``phi`` layout of `~~numpyro_forecast.var.var_step()`.
    scale : Float[Array, "lags obs obs"]
        Prior standard deviations in the same layout.

    Raises
    ------
    ValueError
        If ``n_lags`` or ``n_obs`` is below one, or a Python-number ``tightness``
        is not positive or ``cross_shrinkage`` is outside ``[0, 1]`` (an unknown
        ``decay`` is a type error under the package's runtime type checking).

    Examples
    --------
    ```python
    loc, scale = minnesota_prior(n_lags=2, n_obs=3, tightness=0.5, own_lag_mean=0.0)
    phi = numpyro.sample("phi", dist.Normal(loc, scale).to_event(3))
    ```

    References
    ----------
    Litterman, R. B. (1986). Forecasting with Bayesian vector autoregressions:
    five years of experience. *Journal of Business & Economic Statistics*, 4(1).

    Doan, T., Litterman, R. B. and Sims, C. A. (1984). Forecasting and
    conditional projection using realistic prior distributions. *Econometric
    Reviews*, 3(1).

    Pinder, T. *Impulso*, ``MinnesotaPrior``
    ([documentation](https://thomaspinder.github.io/Impulso/reference/generated/impulso.priors.MinnesotaPrior.html),
    [repository](https://github.com/thomaspinder/impulso)).
    """
    if n_lags < 1:
        msg = f"minnesota_prior requires n_lags >= 1, got {n_lags}"
        raise ValueError(msg)
    if n_obs < 1:
        msg = f"minnesota_prior requires n_obs >= 1, got {n_obs}"
        raise ValueError(msg)
    if isinstance(tightness, int | float) and not tightness > 0:
        msg = f"tightness must be positive, got {tightness}"
        raise ValueError(msg)
    if isinstance(cross_shrinkage, int | float) and not 0 <= cross_shrinkage <= 1:
        msg = f"cross_shrinkage must lie in [0, 1], got {cross_shrinkage}"
        raise ValueError(msg)

    eye = jnp.eye(n_obs)
    lags = jnp.arange(1, n_lags + 1, dtype=float)
    lag_decay = 1.0 / lags if decay == "harmonic" else 1.0 / lags**2
    own_mask = eye + jnp.asarray(cross_shrinkage, dtype=float) * (1.0 - eye)
    scale = jnp.asarray(tightness, dtype=float) * lag_decay[:, None, None] * own_mask[None]
    first_lag = jnp.asarray(own_lag_mean, dtype=float) * eye
    loc = jnp.concatenate([first_lag[None], jnp.zeros((n_lags - 1, n_obs, n_obs))], axis=0)
    return loc, scale
