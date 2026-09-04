r"""Vector autoregression (VAR) components that compose with the model building blocks.

This module is not a VAR package. It supplies the pieces a model function
composes with `~~numpyro_forecast.models.ssoe()` (an observed VAR with
correlated shocks) or `~~numpyro_forecast.models.markov_series()` (a latent
VAR), and the two post-inference quantities every VAR analysis needs, the
companion matrix and the impulse responses. Priors are the caller's
``numpyro.sample`` statements (`~~numpyro_forecast.priors.minnesota_prior()`
returns the moments of one common choice), and the shock distribution is the
caller's ``noise_dist``.

**Layout.** Time sits at axis ``-2`` and the series at axis ``-1``, as
everywhere in the package. Coefficients are ``phi`` of shape
``(*batch, lags, obs, obs)`` with ``phi[..., l, :, :]`` the matrix $\Phi_{l+1}$
and ``phi[..., l, i, j]`` the effect of $y_{t-l-1, j}$ on $y_{t, i}$. A lag
window is ``(*batch, lags, obs)`` in natural time order, most recent row
**last**: ``lags[..., -1, :]`` is $y_{t-1}$, so the first ``lags`` rows of a
series seed the recursion without any reversal. Batch axes broadcast: a shared
``(lags, obs, obs)`` coefficient tensor against a panel of ``(B, lags, obs)``
windows, or posterior draws ``(S, lags, obs, obs)`` against a single window.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float, PyTree

from numpyro_forecast.models import SSOEStep
from numpyro_forecast.typing import Array


def var_mean(
    phi: Float[Array, " *#batch lags obs obs"],
    lags: Float[Array, " *#batch lags obs"],
    intercept: Float[Array, " *#batch obs"] | None = None,
) -> Float[Array, " *batch obs"]:
    r"""Compute the VAR conditional mean of the next row from a lag window.

    $$
    \mu_t = c + \sum_{l=1}^{p} \Phi_l \, y_{t-l},
    $$

    with ``phi[..., l - 1, :, :]`` holding $\Phi_l$ and ``lags[..., -l, :]``
    holding $y_{t-l}$ (most recent row last).

    Parameters
    ----------
    phi
        Coefficient tensor ``(*batch, lags, obs, obs)``; see the module docstring
        for the index convention.
    lags
        Lag window ``(*batch, lags, obs)`` in natural time order.
    intercept
        Optional intercept $c$ of shape ``(*batch, obs)``; ``None`` for a
        zero-mean recursion (the impulse response case).

    Returns
    -------
    Float[Array, "*batch obs"]
        The conditional mean row, broadcast over the batch axes of the inputs.

    Examples
    --------
    A latent VAR under `~~numpyro_forecast.models.markov_series()`: the transition
    returns the next-row distribution and the window update, with ``phi`` and the
    Cholesky factor ``scale_tril`` sampled outside:

    ```python
    def transition(carry, _):
        dist_t = dist.MultivariateNormal(var_mean(phi, carry), scale_tril=scale_tril)
        return dist_t, lambda z: jnp.concatenate([carry[..., 1:, :], z[..., None, :]], axis=-2)


    z = markov_series(h, "z", init_carry=jnp.zeros((n_lags, n_obs)), transition=transition)
    ```
    """
    mu = jnp.einsum("...lij,...lj->...i", phi, lags[..., ::-1, :])
    return mu if intercept is None else mu + intercept


def var_step(
    phi: Float[Array, " *#batch lags obs obs"],
    intercept: Float[Array, " *#batch obs"] | None = None,
) -> SSOEStep[Float[Array, " *batch lags obs"]]:
    r"""Build the `~~numpyro_forecast.models.ssoe()` step of a VAR from its coefficients.

    The carry is the lag window ``(*batch, lags, obs)``. Each step emits
    `var_mean()` of the window as the one-step-ahead mean and, given the row's
    value, drops the oldest row and appends the new one. The step ignores its
    exogenous input ``x_t``; add regressors by wrapping it (a VARX):

    ```python
    base = var_step(phi, intercept)


    def step(carry, x_t):
        mu, carry_fn = base(carry, x_t)
        return mu + beta @ x_t, carry_fn
    ```

    The step knows nothing about priors: ``phi`` and ``intercept`` are whatever
    the model sampled (a weakly informative ``Normal``, the moments of
    `~~numpyro_forecast.priors.minnesota_prior()`, a hierarchical prior, ...), so
    changing the prior never touches the recursion.

    Parameters
    ----------
    phi
        Coefficient tensor ``(*batch, lags, obs, obs)``; see the module docstring.
    intercept
        Optional intercept of shape ``(*batch, obs)``.

    Returns
    -------
    SSOEStep[Float[Array, "*batch lags obs"]]
        A ``(carry, x_t) -> (mu_t, carry_fn)`` callable for `~~numpyro_forecast.models.ssoe()`.

    Raises
    ------
    ValueError
        At step time, if the carry does not hold exactly ``phi.shape[-3]`` rows
        (the usual cause is an ``init_carry`` with the wrong number of lags).

    Examples
    --------
    An observed VAR(p) conditioned on its first ``p`` rows. ``y_init`` is the
    seed window, a constant closed over by the model; the likelihood rows travel
    through ``covariates`` (padded with `~~numpyro_forecast.arrays.pad_future()`
    to fix the forecast horizon) and through ``data``:

    ```python
    def var_model(covariates, data=None):
        h = Horizon.from_data(covariates, data)
        y = covariates[..., : h.t_obs, :]
        intercept = jnp.asarray(
            numpyro.sample("intercept", dist.Normal(0.0, 1.0).expand([k]).to_event(1))
        )
        sigma = jnp.asarray(numpyro.sample("sigma", dist.HalfNormal(1.0).expand([k]).to_event(1)))
        l_omega = jnp.asarray(numpyro.sample("l_omega", dist.LKJCholesky(k, concentration=1.0)))
        phi = jnp.asarray(
            numpyro.sample("phi", dist.Normal(0.0, 1.0).expand([p, k, k]).to_event(3))
        )
        scale_tril = sigma[..., :, None] * l_omega
        noise = dist.MultivariateNormal(jnp.zeros(k), scale_tril=scale_tril)
        r = ssoe(h, "eps", y, y_init, var_step(phi, intercept), noise)
        numpyro.sample("obs", dist.MultivariateNormal(r.mu, scale_tril=scale_tril), obs=h.data)
        if h.future > 0:
            numpyro.deterministic("forecast", r.y_future)
    ```

    Closing over ``y_init`` is right for a single fit but wrong under
    `~~numpyro_forecast.evaluate.backtest()`, which slices ``covariates`` per
    window: for a backtest, ship the seed rows inside ``covariates`` and slice
    the carry from them in the model.
    """
    n_lags, n_obs = phi.shape[-3], phi.shape[-1]

    def step(
        carry: Float[Array, " *batch lags obs"], x_t: PyTree[Array] | None
    ) -> tuple[
        Float[Array, " *batch obs"], Callable[[Array, Array], Float[Array, " *batch lags obs"]]
    ]:
        if carry.shape[-2] != n_lags:
            msg = (
                f"var_step expects a carry of shape (*batch, lags={n_lags}, obs={n_obs}) holding "
                f"the last {n_lags} rows in natural time order; got {carry.shape}. Seed ssoe with "
                f"init_carry=y[..., :{n_lags}, :] and drive it with y[..., {n_lags}:, :]."
            )
            raise ValueError(msg)

        def carry_fn(
            y_t: Float[Array, " *batch obs"], eps_t: Float[Array, " *batch obs"]
        ) -> Float[Array, " *batch lags obs"]:
            return jnp.concatenate([carry[..., 1:, :], y_t[..., None, :]], axis=-2)

        return var_mean(phi, carry, intercept), carry_fn

    return step


def companion_matrix(
    phi: Float[Array, " *#batch lags obs obs"],
) -> Float[Array, " *batch lags_obs lags_obs"]:
    r"""Stack the VAR coefficients into the companion form of a VAR(1).

    $$
    F = \begin{bmatrix}
    \Phi_1 & \Phi_2 & \cdots & \Phi_{p-1} & \Phi_p \\
    I & 0 & \cdots & 0 & 0 \\
    0 & I & \cdots & 0 & 0 \\
    \vdots & & \ddots & & \vdots \\
    0 & 0 & \cdots & I & 0
    \end{bmatrix},
    $$

    a square matrix of size ``lags * obs``. The companion state stacks the lag
    window **most recent first**, the reverse of the ``lags`` layout:
    ``s = lags[..., ::-1, :].reshape(*batch, lags * obs)``, and then
    ``(F @ s)[..., :obs]`` equals `var_mean()` without intercept. The VAR is
    stable exactly when every eigenvalue of $F$ has modulus below one; that is
    the condition for a finite unconditional mean $(I - \sum_l \Phi_l)^{-1} c$
    and for the impulse responses of `impulse_response()` to die out.

    Parameters
    ----------
    phi
        Coefficient tensor ``(*batch, lags, obs, obs)``.

    Returns
    -------
    Float[Array, "*batch lags_obs lags_obs"]
        The companion matrix per batch element.

    Notes
    -----
    JAX implements the eigenvalues of a nonsymmetric matrix on CPU only, so
    compute the spectral radius of posterior draws with NumPy:
    ``np.abs(np.linalg.eigvals(np.asarray(companion_matrix(phi)))).max(-1)``.
    """
    n_lags, n_obs = phi.shape[-3], phi.shape[-1]
    batch = phi.shape[:-3]
    top = jnp.concatenate(list(jnp.moveaxis(phi, -3, 0)), axis=-1)
    shift = jnp.concatenate(
        [
            jnp.eye((n_lags - 1) * n_obs, dtype=phi.dtype),
            jnp.zeros(((n_lags - 1) * n_obs, n_obs), phi.dtype),
        ],
        axis=-1,
    )
    bottom = jnp.broadcast_to(shift, (*batch, *shift.shape))
    return jnp.concatenate([top, bottom], axis=-2)


def impulse_response(
    phi: Float[Array, " *#batch lags obs obs"],
    horizon: int,
    *,
    scale_tril: Float[Array, " *#batch obs obs"] | None = None,
    cumulative: bool = False,
) -> Float[Array, " *batch steps obs obs"]:
    r"""Compute the impulse responses (moving-average coefficients) of a VAR.

    Inverting the lag polynomial of $y_t = c + \sum_{l=1}^{p} \Phi_l y_{t-l} +
    \varepsilon_t$ gives the moving-average representation $y_t = \mu +
    \sum_{h \ge 0} \Psi_h \varepsilon_{t-h}$ whose coefficients follow the
    recursion

    $$
    \Psi_0 = I, \qquad
    \Psi_h = \sum_{j=1}^{\min(h, p)} \Phi_j \Psi_{h-j} \quad (h \ge 1).
    $$

    ``out[..., h, i, j]`` is the response of variable ``i``, ``h`` steps after a
    unit shock to variable ``j`` at step ``0`` (a unit increase of the
    reduced-form residual $\varepsilon_{t, j}$ with the other residuals held at
    zero). With ``scale_tril`` the responses are to *orthogonalized* shocks: for
    a factor $B$ with $\Sigma = B B^\top$ (the Cholesky factor $L$ of the shock
    covariance, so $\varepsilon_t = L u_t$ with $u_t$ standard normal) the
    function returns $\Theta_h = \Psi_h B$, the response to a one-standard-
    deviation shock $u_{t, j}$. With the Cholesky factor this is the recursive
    identification: the first series in the ordering responds to its own shock
    only at $h = 0$, the second to the first two, and so on, so the ordering of
    the series is part of the model. The recursion is defined for any ``phi``;
    the moving-average representation, and $\Psi_h \to 0$, need a stable VAR
    (see `companion_matrix()`). The left recursion above and the right recursion
    $\Psi_h = \sum_j \Psi_{h-j} \Phi_j$ agree, because both are the two-sided
    inverse of the lag polynomial.

    Parameters
    ----------
    phi
        Coefficient tensor ``(*batch, lags, obs, obs)``. Posterior draws pass
        through the leading batch axis unchanged; no ``vmap`` is needed.
    horizon
        Largest step $H \ge 0$ to compute; the result holds ``horizon + 1``
        steps including step ``0`` (like `~~numpyro_forecast.acf.acf()`, which
        includes lag ``0``).
    scale_tril
        Optional square factor ``(*batch, obs, obs)`` of the shock covariance
        (typically the Cholesky factor, ``sigma[..., :, None] * l_omega`` for an
        LKJ correlation factor). Its batch axes broadcast against those of
        ``phi``.
    cumulative
        If ``True``, return the running sum over steps, the response of the
        *level* when the modeled series are differences.

    Returns
    -------
    Float[Array, "*batch steps obs obs"]
        $\Psi_0, \ldots, \Psi_H$ (or $\Theta_h$, or their running sums) with the
        step axis at ``-3``: the package's "time at ``-2``" convention counts
        from the trailing event dims, and the event here is an ``(obs, obs)``
        matrix. Reshape to ``(*batch, steps, obs * obs)`` to plot the grid with
        `~~numpyro_forecast.convert.predictions_to_datatree()`.

    Raises
    ------
    ValueError
        If ``horizon`` is negative.

    Notes
    -----
    The posterior mean of the impulse responses is not the impulse response at
    the posterior mean of ``phi``: the recursion is nonlinear in ``phi``, and
    a single explosive draw dominates the mean at long horizons, so summarize
    draws with quantiles and check stability first.

    References
    ----------
    Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*,
    chapter 2. Springer.

    Orduz, J. *Bayesian VAR in NumPyro*,
    [juanitorduz.github.io/var_numpyro](https://juanitorduz.github.io/var_numpyro/),
    whose ``compute_irf`` this function generalizes.

    Pinder, T. *Impulso*, [github.com/thomaspinder/impulso](https://github.com/thomaspinder/impulso):
    the broadcast-over-draws recursion of its ``compute_ma_phi`` is the shape of
    this implementation, ported to a JAX scan.
    """
    if horizon < 0:
        msg = f"impulse_response requires horizon >= 0, got {horizon}"
        raise ValueError(msg)
    n_lags, n_obs = phi.shape[-3], phi.shape[-1]
    batch = phi.shape[:-3]
    eye = jnp.eye(n_obs, dtype=phi.dtype)
    # The scan carry is the zero-padded history of the last ``lags`` coefficients, most recent
    # last (the same layout as a lag window), so the ``min(h, p)`` bound needs no mask.
    init = jnp.zeros((*batch, n_lags, n_obs, n_obs), phi.dtype).at[..., -1, :, :].set(eye)

    def body(hist: Array, _: None) -> tuple[Array, Array]:
        psi_h = jnp.einsum("...ljk,...lkm->...jm", phi, hist[..., ::-1, :, :])
        return jnp.concatenate([hist[..., 1:, :, :], psi_h[..., None, :, :]], axis=-3), psi_h

    _, rest = jax.lax.scan(body, init, None, length=horizon)
    psi = jnp.concatenate([init[..., -1:, :, :], jnp.moveaxis(rest, 0, -3)], axis=-3)
    if scale_tril is not None:
        # An einsum rather than ``@``: matmul aligns batch axes from the right and cannot
        # broadcast ``(B, steps, obs, obs)`` against ``(B, obs, obs)``.
        psi = jnp.einsum("...hij,...jk->...hik", psi, scale_tril)
    if cumulative:
        psi = jnp.cumsum(psi, axis=-3)
    return psi
