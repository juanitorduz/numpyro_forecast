r"""Model stacking: combine several forecasting models with weights learned from backtests.

The design is in ``docs/dev/stacking_design.md``. This module is the second and third
layer of it: everything here consumes the per-cell forecast log densities that
`~~numpyro_forecast.evaluate.backtest()` stores when asked (``log_density=...``) or the
forecast draws it keeps (``keep_predictions=True``), and nothing here fits a
forecasting model or touches a window.

Three groups of functions:

- **Cells.** `stacking_cells()` gathers the per-model backtest results into one
  `StackingCells` table (model axis last, as ``lpd_point`` in the NumPyro tutorial),
  checking that every model saw the same windows and the same reduction.
  `window_design_effect()` measures how dependent the cells of one window are, which
  is what the weight uncertainty has to be tempered by.
- **Weights from a matrix.** `stacking_weights()` (complete-pooling stacking of
  predictive distributions, Yao et al. 2018), `pseudo_bma_weights()` (the pseudo-BMA+
  baseline, bootstrapping over windows), `crps_stacking_weights()` with
  `mixture_crps_terms()` (CRPS stacking from draws only), and `log_score()` to evaluate
  any weights on held-out cells with a paired standard error across windows. Pure JAX
  kernels, jitted and ``vmap``-able.
- **Hierarchical weights as model building blocks.** `hierarchical_weights()` samples
  the partial-pooling parameters of Yao et al. (2022) and returns input-dependent log
  weights; `stacking_factor()` registers the stacking likelihood. A user's stacking
  model is three lines that call them, fit with their own ``MCMC`` or ``SVI``, exactly
  as a forecasting model calls `~~numpyro_forecast.models.innovations()` and
  `~~numpyro_forecast.models.predict()`. `rectified_features()` builds the paper's
  median-split features for continuous inputs, and `stack_forecasts()` samples the
  weighted mixture of forecast draws.

Stacking weights optimize the predictive score of the mixture; they are not posterior
model probabilities, and a model with weight zero is not thereby wrong.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Float, Int, Num

from numpyro_forecast.evaluate import BacktestResult
from numpyro_forecast.typing import Array, GroupPrior, Reduction

MixturePer = Literal["auto", "path", "cell"]
"""How `stack_forecasts()` assigns models to draws: one model per sample path, or per cell."""


# --- Cells ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class StackingCells:
    """Per-cell forecast log densities of several models over the same backtest windows.

    The host-side table every solver reads, built by `stacking_cells()`. A cell is one
    ``(window, horizon step, series)`` triple; the model axis is last. Like
    `~~numpyro_forecast.evaluate.BacktestResult`, a frozen dataclass of host values,
    not a pytree: the solvers take its arrays, not the object.

    Attributes
    ----------
    log_density
        Per-cell log densities, shape ``(cells, models)``.
    model_names
        The model names, in the column order of ``log_density``.
    horizon
        Zero-based horizon step of each cell, shape ``(cells,)``; ``-1`` for cells of
        a ``"joint"`` reduction (one cell per window and series).
    window
        Zero-based backtest window index of each cell, shape ``(cells,)``; the unit
        of exchangeability for standard errors and bootstraps.
    series
        Zero-based flattened ``(*batch, obs)`` index of each cell, shape ``(cells,)``;
        all zeros for a univariate series.
    truth
        Realized value at each cell, shape ``(cells,)``.
    reduction
        The `~~numpyro_forecast.typing.Reduction` every model's densities hold.
    draws
        Forecast draws per model and cell, shape ``(models, sample, cells)``, when
        every backtest kept predictions; ``None`` otherwise. The input of
        `mixture_crps_terms()`.
    """

    log_density: Float[np.ndarray, " cells models"]
    model_names: tuple[str, ...]
    horizon: Int[np.ndarray, " cells"]
    window: Int[np.ndarray, " cells"]
    series: Int[np.ndarray, " cells"]
    truth: Float[np.ndarray, " cells"]
    reduction: Reduction
    draws: Float[np.ndarray, " models sample cells"] | None = None


def stacking_cells(
    results: Mapping[str, Sequence[BacktestResult]],
    data: Num[ArrayLike, " *batch time obs"],
) -> StackingCells:
    """Gather per-model backtest results into one `StackingCells` table.

    ``results`` maps a model name to the list `~~numpyro_forecast.evaluate.backtest()`
    returned for that model, run with ``log_density`` set. Every list must hold the same
    ``(t0, t1, t2)`` windows in the same order and the same reduction; ``data`` (the
    series the backtests ran on) supplies the realized values. Cells are laid out
    window-major, then horizon step, then flattened series index, so
    ``log_density.reshape(num_windows, future, num_series, models)`` recovers the
    panel for the ``"marginal"`` reduction. ``draws`` is filled when every result
    carries a ``prediction``; the draw axes are aligned by position, so all backtests
    must have used the same ``num_samples``.

    Parameters
    ----------
    results
        ``{model_name: backtest(...) results}`` for at least two models.
    data
        The full series the backtests were run on, time at axis ``-2``.

    Returns
    -------
    StackingCells
        The cell table, model axis last.

    Raises
    ------
    StackingShapeError
        If fewer than two models are given, if any result lacks ``log_density``, if
        the window sequences or reductions differ across models, or if some but
        not all results carry predictions (or with different sample counts).
    """
    raise NotImplementedError


def window_design_effect(
    log_density: Float[ArrayLike, " cells models"], window: Int[ArrayLike, " cells"]
) -> Float[Array, ""]:
    r"""How many cells of one window count as one independent observation.

    The cells of a window share one realized future path and one fitted posterior,
    so the per-cell stacking terms are dependent; summing them as if independent
    overstates the evidence about the weights by this factor. With $d_c$ the
    difference between the best and the second-best model's log density at cell $c$
    (the quantity the weights respond to), the design effect is

    $$
    \text{deff} = \frac{\operatorname{Var}_w\left(\bar d_w\right)}{\operatorname{Var}_c(d_c) / \bar H},
    $$

    the variance of the window means over the variance an independent sample of
    $\bar H$ cells per window would give. One means independent cells; on the
    prototype it was about ten for fourteen-step windows, so a window carried about
    1.4 independent cells. Use ``1 / deff`` as the ``cell_weights`` of
    `stacking_factor()` (a tempered likelihood) and quote standard errors across
    windows, never across cells (`log_score()` with ``unit``).

    Parameters
    ----------
    log_density
        Per-cell log densities, shape ``(cells, models)``.
    window
        Window index of each cell.

    Returns
    -------
    Float[Array, ""]
        The design effect, at least one.
    """
    raise NotImplementedError


# --- Weights from a matrix -------------------------------------------------------------------


def log_score(
    log_density: Float[ArrayLike, " cells models"],
    weights: Float[ArrayLike, " models"] | Float[ArrayLike, " cells models"],
    *,
    unit: Int[ArrayLike, " cells"] | None = None,
) -> Float[Array, " _cells_or_units"]:
    r"""Log score of the weighted mixture at every cell, or averaged per unit.

    Returns $\log \sum_k w_{ck} \exp L_{ck}$ per cell (the stacking objective is its
    sum), or, with ``unit`` (the ``window`` of a `StackingCells`), the mean per unit:
    the vector whose paired differences between two methods, divided by
    $\sqrt{\text{units}}$, give the standard error that a held-out comparison must
    carry. A comparison quoted per cell understates the error by the square root of
    `window_design_effect()`. Equal weights ``jnp.full(K, 1 / K)`` are the baseline
    every stacking result has to beat.

    Parameters
    ----------
    log_density
        Per-cell log densities, shape ``(cells, models)``.
    weights
        One weight vector, or one per cell.
    unit
        Optional unit index per cell; when given, the per-cell scores are averaged
        within each unit and the result has one entry per unit, in sorted unit order.

    Returns
    -------
    Float[Array, " _cells_or_units"]
        Per-cell log scores, or per-unit means.
    """
    raise NotImplementedError


def stacking_weights(
    log_density: Float[ArrayLike, " cells models"],
    *,
    cell_weights: Float[ArrayLike, " cells"] | None = None,
    dirichlet_alpha: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 1_000,
) -> Float[Array, " models"]:
    r"""Complete-pooling stacking of predictive distributions (Yao et al. 2018).

    Solves

    $$
    \hat w = \arg\max_{w \in S_K} \sum_c \pi_c \, \log \sum_k w_k \exp L_{ck}
    + (\alpha - 1) \sum_k \log w_k,
    $$

    over the simplex, with $\pi_c$ the optional ``cell_weights`` (recency weights, or a
    tempering by ``1 / window_design_effect``) and a $\text{Dirichlet}(\alpha)$ log prior
    that is flat at the default $\alpha = 1$ and keeps every weight strictly positive
    for $\alpha > 1$. The objective is concave; the solver parameterizes
    $w = \text{softmax}(\eta)$ with $\eta_K = 0$ and runs BFGS
    (`jax.scipy.optimize.minimize`) from the uniform point, with the per-cell
    ``logsumexp`` shift built in, so densities of order $e^{-700}$ are as stable as
    any other. When one model dominates every cell the optimum is at the boundary and
    is reached only in the limit: the result is the weights at convergence (within
    ``tol``), not an exact zero. Pure JAX; ``vmap`` over a leading axis for
    per-horizon no-pooling weights. Agrees with
    ``arviz_stats.compare(method="stacking")`` on the same matrix (a test).

    Duplicated models are not separately identified: the objective is flat along any
    split of their combined weight, so only the sum is a meaningful quantity (and
    $\alpha > 1$ selects the even split).

    Parameters
    ----------
    log_density
        Per-cell log densities, shape ``(cells, models)``, at least two models.
    cell_weights
        Optional non-negative weight per cell.
    dirichlet_alpha
        Concentration of the Dirichlet log prior on the weights; ``1.0`` is flat.
    tol
        Gradient tolerance of the optimizer.
    max_iter
        Iteration cap of the optimizer.

    Returns
    -------
    Float[Array, " models"]
        Weights on the simplex.

    Raises
    ------
    StackingShapeError
        If ``log_density`` has fewer than two models.
    """
    raise NotImplementedError


def pseudo_bma_weights(
    rng_key: Array | None,
    log_density: Float[ArrayLike, " cells models"],
    *,
    unit: Int[ArrayLike, " cells"] | None = None,
    num_bootstrap: int = 1_000,
) -> Float[Array, " models"]:
    r"""Pseudo-BMA+ weights: softmax of the total log score, bootstrapped over units.

    The baseline of Yao et al. (2018, Section 3.4): $w_k \propto \exp \sum_c L_{ck}$,
    averaged over Bayesian-bootstrap reweightings of the **units** (windows, or series)
    rather than of the cells, since cells within a window are not exchangeable
    (`window_design_effect()`). With ``rng_key=None`` or ``num_bootstrap=0`` the plain
    pseudo-BMA weights are returned. Inherits the instability of the point estimate of
    the total score; `stacking_weights()` is the recommended solver and this is the
    reference it is compared against in the notebook.

    Parameters
    ----------
    rng_key
        PRNG key for the Dirichlet(1) bootstrap weights, or ``None`` for no bootstrap.
    log_density
        Per-cell log densities, shape ``(cells, models)``.
    unit
        Unit index per cell; cells of one unit are resampled together. ``None``
        treats every cell as a unit.
    num_bootstrap
        Number of bootstrap replicates.

    Returns
    -------
    Float[Array, " models"]
        Weights on the simplex.
    """
    raise NotImplementedError


def mixture_crps_terms(
    draws: Float[ArrayLike, " models sample *cells"],
    truth: Float[ArrayLike, " *cells"],
) -> tuple[Float[Array, " *cells models"], Float[Array, " *cells models models"]]:
    r"""Compute the two terms of the mixture CRPS from forecast draws.

    For a mixture $F_w = \sum_k w_k F_k$,

    $$
    \text{CRPS}(F_w, y) = \sum_k w_k a_k - \tfrac{1}{2} \sum_{k,l} w_k w_l B_{kl},
    \qquad a_k = \mathbb{E}|X_k - y|, \quad B_{kl} = \mathbb{E}|X_k - X_l|,
    $$

    which is the energy form of the CRPS applied to the mixture. The terms are
    estimated from the draws once; every weight vector is then evaluated in
    $O(K^2)$ per cell. $a_k - \tfrac12 B_{kk}$ equals
    `~~numpyro_forecast.metrics.crps_empirical()` of model $k$ alone (a test), and
    $B$ is symmetric with a zero diagonal. $B_{kl}$ costs $O(S^2)$ per pair; the
    kernel uses the sorted-sample identity on the pooled pair so the cost is
    $O(S \log S)$.

    Parameters
    ----------
    draws
        Forecast draws per model, shape ``(models, sample, *cells)``; the ``draws``
        of a `StackingCells`.
    truth
        Realized values, shape ``(*cells,)``.

    Returns
    -------
    tuple[Float[Array, " *cells models"], Float[Array, " *cells models models"]]
        The ``a`` and ``B`` terms per cell.
    """
    raise NotImplementedError


def crps_stacking_weights(
    a: Float[ArrayLike, " cells models"],
    b: Float[ArrayLike, " cells models models"],
    *,
    cell_weights: Float[ArrayLike, " cells"] | None = None,
    tol: float = 1e-8,
    max_iter: int = 1_000,
) -> Float[Array, " models"]:
    r"""Weights minimizing the mean mixture CRPS over the simplex.

    Minimizes $\sum_c \pi_c \left( w^\top a_c - \tfrac12 w^\top B_c w \right)$ over
    $S_K$. The CRPS is a convex quadratic in the forecast distribution and the mixture
    is affine in $w$, so this is a convex quadratic program on the simplex (the
    restriction of $-B$ to the simplex tangent space is positive semidefinite; the
    unrestricted $-B$ is indefinite, so a naive definiteness check fails). Solved by
    projected gradient on the simplex with a fixed step from the Lipschitz constant of
    the gradient; the solution may sit on the boundary (exact zeros are returned).
    Needs draws only, so it covers models for which no forecast density is available,
    at the cost of optimizing a different score from `stacking_weights()`: the CRPS
    rewards sharpness less than the log score penalizes tails, and the two weight
    vectors differ in general.

    Parameters
    ----------
    a
        First mixture term from `mixture_crps_terms()`, shape ``(cells, models)``.
    b
        Second mixture term, shape ``(cells, models, models)``.
    cell_weights
        Optional non-negative weight per cell.
    tol
        Stopping tolerance on the change of the weights.
    max_iter
        Iteration cap.

    Returns
    -------
    Float[Array, " models"]
        Weights on the simplex.
    """
    raise NotImplementedError


# --- Hierarchical weights as model building blocks -------------------------------------------


def rectified_features(
    x: Float[ArrayLike, " cells features"],
    *,
    center: Float[ArrayLike, " features"] | None = None,
) -> tuple[Float[Array, " cells 2*features"], Float[Array, " features"]]:
    r"""Median-split features for continuous stacking inputs (Yao et al. 2022, eq. 16).

    Each continuous input $x_d$ becomes the pair $(x_d - m_d)_+$ and $(m_d - x_d)_+$
    with $m_d$ its median over the training cells, so a weight can move differently
    on either side of the typical value; the pair sums to $|x_d - m_d|$ and differs by
    $x_d - m_d$ (both tested). Returns the medians so that held-out cells are
    transformed with the training medians (``center=``), never their own. Standardize
    the inputs first, as the paper recommends.

    Parameters
    ----------
    x
        Continuous inputs, shape ``(cells, features)``.
    center
        Medians to split at; computed from ``x`` when ``None``.

    Returns
    -------
    tuple[Float[Array, " cells 2*features"], Float[Array, " features"]]
        The rectified features (right parts first, then left parts) and the centers
        used.
    """
    raise NotImplementedError


def hierarchical_weights(
    name: str,
    num_models: int,
    groups: Int[Array, " cells"] | Sequence[Int[Array, " cells"]],
    num_groups: int | Sequence[int],
    *,
    features: Float[Array, " cells features"] | None = None,
    group_prior: GroupPrior | Sequence[GroupPrior] = "exchangeable",
    tau_mu: float = 1.0,
    tau_sigma: float = 1.0,
    tau_beta: float = 1.0,
) -> Float[Array, " cells models"]:
    r"""Sample input-dependent stacking log weights with partial pooling (a building block).

    Called inside a user's stacking model, as `~~numpyro_forecast.models.innovations()`
    is called inside a forecasting model; the block owns every sample site and returns
    the log weights. For model $k < K$ and cell $c$ with group index $g(c)$ and
    features $f(c)$,

    $$
    \alpha_k(c) = \mu_k + \sigma_k \, z_{g(c), k} + \sum_m \beta_{mk} f_m(c),
    \qquad \alpha_K(c) = 0,
    $$

    with $\mu_k \sim \text{Normal}(0, \tau_\mu)$, $\sigma_k \sim \text{HalfNormal}(\tau_\sigma)$,
    $z_{jk} \sim \text{Normal}(0, 1)$ (the non-centered form of eq. 9 of Yao et al. 2022)
    and $\beta_{mk} \sim \text{Normal}(0, \tau_\beta)$ (eqs. 13 and 14); the log weights
    are $\log \text{softmax}(\alpha(c))$. Several group indices (horizon step and
    series, say) enter additively, each with its own $\sigma$ and prior. The
    ``"random_walk"`` group prior replaces the exchangeable $z_{jk}$ by a centered
    Gaussian random walk over the group index, for an ordinal group such as the
    horizon step, so neighboring steps share strength. Complete pooling and no pooling
    are the limits $\sigma_k \to 0$ and $\sigma_k \to \infty$; a cell with group index
    ``-1`` gets the population weights $\text{softmax}(\mu)$ (a series with no backtest
    history). Sites: ``f"{name}_mu"`` ``(K - 1,)``, ``f"{name}_sigma"`` ``(K - 1,)`` per
    group index, ``f"{name}_z"`` ``(J, K - 1)`` per group index, ``f"{name}_beta"``
    ``(M, K - 1)``, and the deterministic ``name`` holding the weights ``(cells, K)``.

    Pass the training **and** held-out cells together: the block returns log weights
    for every row, `stacking_factor()` scores the training rows, and the posterior of
    the deterministic ``name`` holds the held-out weights (the ``w_test`` of the NumPyro
    tutorial). The paper reports the posterior mean of the weights, not the MAP, which
    is degenerate on the complete-pooling subspace.

    Parameters
    ----------
    name
        Site prefix.
    num_models
        $K \geq 2$, the number of columns of the log density matrix.
    groups
        One or several integer group indices per cell, each in ``[-1, J)``.
    num_groups
        $J$ for each group index; a Python integer, since it sizes a plate.
    features
        Optional continuous features per cell (standardized, or from
        `rectified_features()`).
    group_prior
        ``"exchangeable"`` or ``"random_walk"``, per group index.
    tau_mu
        Scale of the population-level prior.
    tau_sigma
        Scale of the half-normal prior on the group spread.
    tau_beta
        Scale of the prior on the feature coefficients; shrink toward
        $1 / \sqrt{M}$ as the number of features grows.

    Returns
    -------
    Float[Array, " cells models"]
        Log weights, one row per cell, each row summing to one after ``exp``.

    Raises
    ------
    StackingShapeError
        If ``num_models < 2``, or if ``groups`` and ``num_groups`` disagree in count.
    """
    raise NotImplementedError


def stacking_factor(
    name: str,
    log_density: Float[Array, " cells models"],
    log_weights: Float[Array, " cells models"],
    *,
    cell_weights: Float[Array, " cells"] | None = None,
) -> Float[Array, ""]:
    r"""Register the stacking log likelihood as a ``numpyro.factor`` (a building block).

    Adds $\sum_c \pi_c \, \operatorname{logsumexp}_k \left( L_{ck} + \log w_k(c) \right)$ to
    the joint density, with $\pi_c$ the optional ``cell_weights``: a recency sequence
    (Yao et al. 2022, Section 2.5) or the tempering ``1 / window_design_effect`` that
    keeps the posterior over the weights honest about the dependence of the cells of
    one window. Pass the training rows of the log weights returned by
    `hierarchical_weights()`. A user's stacking model is therefore:

    ```python
    def stacking_model(lpd_train, groups, num_groups, cell_weights):
        log_w = hierarchical_weights("w", lpd_train.shape[-1], groups, num_groups)
        stacking_factor("stacking", lpd_train, log_w[: lpd_train.shape[0]], cell_weights)
    ```

    Parameters
    ----------
    name
        Factor site name.
    log_density
        Per-cell log densities of the training cells, shape ``(cells, models)``.
    log_weights
        Log weights at those cells, shape ``(cells, models)``.
    cell_weights
        Optional non-negative weight per cell.

    Returns
    -------
    Float[Array, ""]
        The value added to the joint log density.
    """
    raise NotImplementedError


# --- Mixture draws ---------------------------------------------------------------------------


def stack_forecasts(
    rng_key: Array,
    draws: Num[ArrayLike, " models sample *batch future obs"],
    weights: Float[ArrayLike, " models"] | Float[ArrayLike, " *batch future obs models"],
    *,
    per: MixturePer = "auto",
) -> Num[Array, " sample *batch future obs"]:
    """Sample from the weighted mixture of several models' forecast draws.

    Draws a model index from ``Categorical(weights)`` and takes that model's draw, so
    the result is a sample from the mixture distribution (a weighted average of
    quantiles would not be). With ``per="path"`` one index is drawn per sample path,
    which keeps every path a coherent forecast from one model (cumulative sums, peak
    timing and, for a panel, cross-series relations within a path stay meaningful);
    with ``per="cell"`` an index is drawn per cell, which is required when the weights
    vary across cells (horizon-dependent or series-dependent weights) and gives the
    correct marginal at every cell at the cost of mixing models along a path.
    ``"auto"`` picks ``"path"`` for a single weight vector and ``"cell"`` otherwise.
    The mixture of calibrated components is wider than any component (Gneiting and
    Ranjan 2013), so the coverage of a stacked forecast is above that of its parts;
    recalibration is out of scope here.

    Parameters
    ----------
    rng_key
        PRNG key for the model assignments.
    draws
        Forecast draws per model, aligned on the sample axis, shape
        ``(models, sample, *batch, future, obs)``.
    weights
        One weight vector ``(models,)``, or weights per cell with the model axis
        last, ``(*batch, future, obs, models)`` (the layout of the ``w`` deterministic
        of `hierarchical_weights()` reshaped to the forecast cells).
    per
        ``"auto"``, ``"path"`` or ``"cell"``.

    Returns
    -------
    Num[Array, " sample *batch future obs"]
        Draws from the mixture.

    Raises
    ------
    StackingShapeError
        If ``weights`` are negative, do not sum to one (within float tolerance), do
        not broadcast against the draws, or if ``per="path"`` is requested with
        cell-varying weights.
    """
    raise NotImplementedError


__all__ = [
    "MixturePer",
    "StackingCells",
    "crps_stacking_weights",
    "hierarchical_weights",
    "log_score",
    "mixture_crps_terms",
    "pseudo_bma_weights",
    "rectified_features",
    "stack_forecasts",
    "stacking_cells",
    "stacking_factor",
    "stacking_weights",
    "window_design_effect",
]
