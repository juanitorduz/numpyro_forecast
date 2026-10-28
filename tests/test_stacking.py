"""Tests for model stacking: cells, weight solvers, hierarchical blocks, mixture draws.

Identities over snapshots: the log-score solver against an independent SciPy reference
and against ArviZ, the mixture CRPS against `crps_empirical` on a proportionally pooled
sample, duplicated models through the invariance of their combined weight (the split is
not identified), the hierarchical block through its limits (complete pooling at zero
spread, population weights for an unseen group), and the mixture sampler through the
one-hot and whole-path properties. Kernels not yet implemented are marked ``scaffold``
(a strict xfail on ``NotImplementedError``).
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from jax import Array, random
from jax.scipy.special import logsumexp
from jaxtyping import TypeCheckError
from numpyro.handlers import condition, seed, trace
from numpyro.infer import MCMC, NUTS

from numpyro_forecast import (
    BacktestResult,
    StackingCells,
    crps_stacking_weights,
    hierarchical_weights,
    log_score,
    mixture_crps_terms,
    pseudo_bma_weights,
    rectified_features,
    stack_forecasts,
    stacking_cells,
    stacking_factor,
    stacking_weights,
    window_design_effect,
)
from numpyro_forecast.exceptions import StackingShapeError
from numpyro_forecast.metrics import crps_empirical

scaffold = pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="stacking scaffold: kernel not implemented"
)


# --- Synthetic cells -------------------------------------------------------------------------


def synthetic_log_density(
    rng: np.random.Generator, windows: int, horizon: int, models: int, *, shared: float = 0.0
) -> np.ndarray:
    """Per-cell log densities ``(windows * horizon, models)`` with an optional window effect.

    Each model's log density is ``-1.5 + noise``; ``shared`` adds a window-level term common
    to every cell of a window, which is what makes the cells of a window dependent.
    """
    cells = rng.normal(-1.5, 0.3, size=(windows, horizon, models))
    cells += shared * rng.normal(size=(windows, 1, models))
    return cells.reshape(windows * horizon, models)


def reference_stacking_weights(log_density: np.ndarray) -> np.ndarray:
    """The ArviZ algorithm (SLSQP on the simplex) as an independent reference solver."""
    from scipy.optimize import Bounds, LinearConstraint, minimize

    exp_lpd = np.exp(log_density - log_density.max(axis=1, keepdims=True))
    k = log_density.shape[1]

    def objective(w: np.ndarray) -> float:
        return -float(np.sum(np.log(exp_lpd @ w)))

    def gradient(w: np.ndarray) -> np.ndarray:
        return -np.sum(exp_lpd / (exp_lpd @ w)[:, None], axis=0)

    result = minimize(
        objective,
        np.full(k, 1.0 / k),
        jac=gradient,
        bounds=Bounds(np.zeros(k), np.ones(k)),
        constraints=LinearConstraint(np.ones(k), lb=1.0, ub=1.0),
    )
    return result.x


def objective_value(log_density: np.ndarray, weights: np.ndarray) -> float:
    return float(
        np.sum(logsumexp(jnp.asarray(log_density) + jnp.log(jnp.asarray(weights)), axis=1))
    )


# --- stacking_weights ------------------------------------------------------------------------


@scaffold
def test_stacking_weights_on_the_simplex() -> None:
    lpd = synthetic_log_density(np.random.default_rng(0), 20, 4, 3)
    w = stacking_weights(lpd)
    assert w.shape == (3,)
    assert bool(jnp.all(w >= 0.0))
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)


@scaffold
def test_stacking_weights_dominant_model_takes_all() -> None:
    lpd = synthetic_log_density(np.random.default_rng(0), 20, 4, 3)
    lpd[:, 1] += 2.0  # strictly better at every cell
    w = stacking_weights(lpd)
    assert float(w[1]) > 0.99


@scaffold
def test_stacking_weights_match_scipy_reference() -> None:
    rng = np.random.default_rng(1)
    lpd = synthetic_log_density(rng, 30, 5, 3)
    lpd[:, 0] += rng.normal(0.3, 0.8, size=lpd.shape[0])  # make the optimum interior
    got = stacking_weights(lpd)
    reference = reference_stacking_weights(lpd)
    np.testing.assert_allclose(got, reference, atol=2e-3)
    assert objective_value(lpd, np.asarray(got)) >= objective_value(lpd, reference) - 1e-4


@scaffold
def test_stacking_weights_match_arviz_compare() -> None:
    # The reviewer's cross-check: arviz_stats.compare(method="stacking") on hand-built
    # ELPDData of the same cells gives the same weights.
    arviz_stats = pytest.importorskip("arviz_stats")
    xr = pytest.importorskip("xarray")
    from arviz_stats.utils import ELPDData

    rng = np.random.default_rng(2)
    lpd = synthetic_log_density(rng, 30, 5, 2)
    lpd[:, 0] += rng.normal(0.2, 0.8, size=lpd.shape[0])
    elpd = {}
    for k in range(2):
        vals = lpd[:, k]
        elpd[f"m{k}"] = ELPDData(
            elpd=float(vals.sum()),
            se=float(vals.std(ddof=1) * np.sqrt(len(vals))),
            p=0.0,
            n_samples=100,
            n_data_points=len(vals),
            scale="log",
            warning=False,
            good_k=0.7,
            elpd_i=xr.DataArray(vals, dims=["cell"]),
        )
    table = arviz_stats.compare(elpd, method="stacking").sort_index()
    got = stacking_weights(lpd)
    np.testing.assert_allclose(got, table["weight"].to_numpy(), atol=2e-3)


@scaffold
def test_duplicated_models_combined_weight_is_invariant() -> None:
    # Duplicating a column leaves the objective flat along the split, so only the sum of
    # the duplicates' weights is identified: it must equal the weight of the merged model,
    # and the two objectives must coincide (reviewer finding 8).
    rng = np.random.default_rng(3)
    lpd = synthetic_log_density(rng, 30, 5, 2)
    lpd[:, 0] += rng.normal(0.2, 0.8, size=lpd.shape[0])
    merged = stacking_weights(lpd)
    duplicated = stacking_weights(np.concatenate([lpd[:, :1], lpd[:, :1], lpd[:, 1:]], axis=1))
    np.testing.assert_allclose(duplicated[0] + duplicated[1], merged[0], atol=2e-3)
    np.testing.assert_allclose(duplicated[2], merged[1], atol=2e-3)


@scaffold
def test_dirichlet_alpha_above_one_splits_duplicates_evenly() -> None:
    rng = np.random.default_rng(3)
    lpd = synthetic_log_density(rng, 30, 5, 2)
    lpd[:, 0] += rng.normal(0.2, 0.8, size=lpd.shape[0])
    w = stacking_weights(
        np.concatenate([lpd[:, :1], lpd[:, :1], lpd[:, 1:]], axis=1), dirichlet_alpha=1.5
    )
    np.testing.assert_allclose(w[0], w[1], atol=1e-3)


@scaffold
def test_log_score_is_concave_in_the_weights() -> None:
    lpd = synthetic_log_density(np.random.default_rng(4), 20, 4, 3)
    rng = np.random.default_rng(5)
    for _ in range(10):
        w1, w2 = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3))
        t = rng.uniform()
        mixed = float(log_score(lpd, t * w1 + (1 - t) * w2).sum())
        separate = t * float(log_score(lpd, w1).sum()) + (1 - t) * float(log_score(lpd, w2).sum())
        assert mixed >= separate - 1e-6


@scaffold
def test_stacking_weights_invariant_to_a_common_shift() -> None:
    # A shift of every log density (densities of order e^-700) must not change the weights.
    lpd = synthetic_log_density(np.random.default_rng(6), 20, 4, 3)
    np.testing.assert_allclose(stacking_weights(lpd - 700.0), stacking_weights(lpd), atol=1e-5)


@scaffold
def test_zero_cell_weights_remove_cells() -> None:
    lpd = synthetic_log_density(np.random.default_rng(7), 20, 4, 3)
    lpd[:40, 2] += 3.0  # model 2 only wins on the first ten windows
    cell_weights = np.ones(lpd.shape[0])
    cell_weights[:40] = 0.0
    kept = stacking_weights(lpd[40:])
    masked = stacking_weights(lpd, cell_weights=cell_weights)
    np.testing.assert_allclose(masked, kept, atol=1e-4)


@scaffold
def test_stacking_weights_vmaps_over_a_leading_horizon_axis() -> None:
    lpd = synthetic_log_density(np.random.default_rng(8), 20, 4, 3).reshape(20, 4, 3)
    per_h = jnp.moveaxis(jnp.asarray(lpd), 1, 0)  # (horizon, windows, models)
    vmapped = jax.vmap(stacking_weights)(per_h)
    looped = jnp.stack([stacking_weights(per_h[j]) for j in range(4)])
    np.testing.assert_allclose(vmapped, looped, atol=1e-5)


@scaffold
def test_stacking_weights_reject_a_single_model() -> None:
    with pytest.raises(StackingShapeError):
        stacking_weights(np.zeros((10, 1)))


def test_stacking_weights_reject_a_vector_at_the_type_boundary() -> None:
    with pytest.raises(TypeCheckError):
        stacking_weights(np.zeros(10))


# --- log_score, pseudo_bma_weights, window_design_effect -------------------------------------


@scaffold
def test_log_score_sum_is_the_stacking_objective() -> None:
    lpd = synthetic_log_density(np.random.default_rng(9), 20, 4, 3)
    w = np.array([0.2, 0.5, 0.3])
    np.testing.assert_allclose(float(log_score(lpd, w).sum()), objective_value(lpd, w), rtol=1e-6)


@scaffold
def test_log_score_per_unit_means_and_equal_weights_baseline() -> None:
    lpd = synthetic_log_density(np.random.default_rng(10), 20, 4, 3)
    window = np.repeat(np.arange(20), 4)
    per_unit = log_score(lpd, jnp.full(3, 1 / 3), unit=window)
    assert per_unit.shape == (20,)
    per_cell = np.asarray(log_score(lpd, jnp.full(3, 1 / 3)))
    np.testing.assert_allclose(per_unit, per_cell.reshape(20, 4).mean(axis=1), rtol=1e-6)


@scaffold
def test_log_score_accepts_per_cell_weights() -> None:
    lpd = synthetic_log_density(np.random.default_rng(11), 20, 4, 3)
    weights = np.random.default_rng(12).dirichlet(np.ones(3), size=lpd.shape[0])
    got = log_score(lpd, weights)
    expected = logsumexp(jnp.asarray(lpd) + jnp.log(jnp.asarray(weights)), axis=1)
    np.testing.assert_allclose(got, expected, rtol=1e-6)


@scaffold
def test_pseudo_bma_without_bootstrap_is_softmax_of_totals() -> None:
    lpd = synthetic_log_density(np.random.default_rng(13), 20, 4, 3)
    got = pseudo_bma_weights(None, lpd)
    np.testing.assert_allclose(got, jax.nn.softmax(jnp.asarray(lpd).sum(axis=0)), rtol=1e-5)


@scaffold
def test_pseudo_bma_bootstrap_is_on_the_simplex_and_less_extreme() -> None:
    lpd = synthetic_log_density(np.random.default_rng(14), 20, 4, 3)
    lpd[:, 0] += 0.1
    window = np.repeat(np.arange(20), 4)
    plain = pseudo_bma_weights(None, lpd)
    boot = pseudo_bma_weights(random.PRNGKey(0), lpd, unit=window, num_bootstrap=500)
    np.testing.assert_allclose(boot.sum(), 1.0, atol=1e-6)
    assert float(boot.max()) <= float(plain.max()) + 1e-6


@scaffold
def test_window_design_effect_is_one_for_independent_cells() -> None:
    lpd = synthetic_log_density(np.random.default_rng(15), 400, 8, 2)
    deff = window_design_effect(lpd, np.repeat(np.arange(400), 8))
    assert 0.8 < float(deff) < 1.3


@scaffold
def test_window_design_effect_grows_with_a_shared_window_term() -> None:
    lpd = synthetic_log_density(np.random.default_rng(16), 400, 8, 2, shared=0.6)
    deff = window_design_effect(lpd, np.repeat(np.arange(400), 8))
    assert float(deff) > 3.0


# --- CRPS stacking ---------------------------------------------------------------------------


def _three_model_draws(num: int = 400) -> tuple[Array, Array]:
    key = random.PRNGKey(20)
    k1, k2, k3, kt = random.split(key, 4)
    truth = random.normal(kt, (6,))
    draws = jnp.stack(
        [
            truth + 0.5 * random.normal(k1, (num, 6)),
            truth + 1.0 + random.normal(k2, (num, 6)),
            2.0 * random.normal(k3, (num, 6)),
        ]
    )
    return draws, truth


@scaffold
def test_mixture_crps_diagonal_is_the_single_model_crps() -> None:
    draws, truth = _three_model_draws()
    a, b = mixture_crps_terms(draws, truth)
    for k in range(3):
        np.testing.assert_allclose(
            a[:, k] - 0.5 * b[:, k, k], crps_empirical(draws[k], truth), rtol=1e-4
        )


@scaffold
def test_mixture_crps_terms_are_symmetric_with_zero_diagonal() -> None:
    draws, truth = _three_model_draws()
    _, b = mixture_crps_terms(draws, truth)
    np.testing.assert_allclose(b, jnp.swapaxes(b, -1, -2), rtol=1e-6)
    np.testing.assert_allclose(jnp.diagonal(b, axis1=-2, axis2=-1), 0.0, atol=1e-6)


@scaffold
@pytest.mark.parametrize("weights", [(0.5, 0.25, 0.25), (0.6, 0.3, 0.1), (0.1, 0.1, 0.8)])
def test_mixture_crps_identity_against_a_proportional_pool(weights: tuple[float, ...]) -> None:
    # CRPS(sum_k w_k F_k) equals crps_empirical on a sample that pools the models' draws in
    # proportion to the weights (a plain concatenation is only the 1/K mixture).
    draws, truth = _three_model_draws(num=1_000)
    a, b = mixture_crps_terms(draws, truth)
    w = jnp.asarray(weights)
    quadratic = a @ w - 0.5 * jnp.einsum("k,ckl,l->c", w, b, w)
    counts = [round(1_000 * x) for x in weights]
    pooled = jnp.concatenate([draws[k, : counts[k]] for k in range(3)], axis=0)
    np.testing.assert_allclose(quadratic, crps_empirical(pooled, truth), atol=2e-3)


@scaffold
def test_crps_stacking_weights_two_models_match_a_grid_search() -> None:
    draws, truth = _three_model_draws()
    a, b = mixture_crps_terms(draws[:2], truth)
    got = crps_stacking_weights(a, b)
    grid = jnp.linspace(0.0, 1.0, 1_001)
    ws = jnp.stack([grid, 1 - grid], axis=1)
    values = (ws @ a.T).mean(axis=1) - 0.5 * jnp.einsum("gk,ckl,gl->g", ws, b, ws) / a.shape[0]
    np.testing.assert_allclose(got[0], grid[values.argmin()], atol=2e-3)


@scaffold
def test_crps_stacking_objective_is_convex_on_the_simplex() -> None:
    draws, truth = _three_model_draws()
    a, b = mixture_crps_terms(draws, truth)

    def value(w: np.ndarray) -> float:
        wj = jnp.asarray(w)
        return float((a @ wj).mean() - 0.5 * jnp.einsum("k,ckl,l->c", wj, b, wj).mean())

    rng = np.random.default_rng(21)
    for _ in range(10):
        w1, w2 = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3))
        assert value(0.5 * (w1 + w2)) <= 0.5 * (value(w1) + value(w2)) + 1e-6


@scaffold
def test_crps_stacking_weights_select_a_perfect_model() -> None:
    draws, truth = _three_model_draws()
    perfect = jnp.broadcast_to(truth, draws[0].shape) + 1e-3 * random.normal(
        random.PRNGKey(22), draws[0].shape
    )
    a, b = mixture_crps_terms(jnp.stack([perfect, draws[2]]), truth)
    got = crps_stacking_weights(a, b)
    assert float(got[0]) > 0.98


# --- rectified_features ----------------------------------------------------------------------


@scaffold
def test_rectified_features_identities() -> None:
    x = np.random.default_rng(23).normal(size=(50, 2))
    feats, center = rectified_features(x)
    assert feats.shape == (50, 4)
    np.testing.assert_allclose(center, np.median(x, axis=0), rtol=1e-6)
    right, left = feats[:, :2], feats[:, 2:]
    assert bool(jnp.all(feats >= 0.0))
    np.testing.assert_allclose(right - left, x - np.median(x, axis=0), atol=1e-6)
    np.testing.assert_allclose(right + left, np.abs(x - np.median(x, axis=0)), atol=1e-6)


@scaffold
def test_rectified_features_reuse_the_training_center() -> None:
    x_train = np.random.default_rng(24).normal(size=(50, 1))
    x_test = x_train + 5.0
    _, center = rectified_features(x_train)
    feats, used = rectified_features(x_test, center=center)
    np.testing.assert_allclose(used, center)
    assert bool(jnp.all(feats[:, 1] == 0.0))  # every test point is right of the median


# --- hierarchical_weights and stacking_factor ------------------------------------------------


def _trace_block(
    groups: Array, num_groups: int, *, conditioned: dict[str, Any] | None = None, **kwargs: Any
) -> tuple[dict[str, Any], Array]:
    def model() -> None:
        log_w = hierarchical_weights("w", 3, groups, num_groups, **kwargs)
        numpyro.deterministic("log_w", log_w)

    fn = condition(model, conditioned) if conditioned else model
    tr = trace(seed(fn, random.PRNGKey(0))).get_trace()
    return tr, tr["log_w"]["value"]


@scaffold
def test_hierarchical_weights_sites_and_shapes() -> None:
    groups = jnp.array([0, 1, 2, 0, 1, 2, -1])
    features = jnp.ones((7, 2))
    tr, log_w = _trace_block(groups, 3, features=features)
    assert tr["w_mu"]["value"].shape == (2,)
    assert tr["w_sigma_0"]["value"].shape == (2,)
    assert tr["w_z_0"]["value"].shape == (3, 2)
    assert tr["w_beta"]["value"].shape == (2, 2)
    assert tr["w"]["type"] == "deterministic"
    assert log_w.shape == (7, 3)
    np.testing.assert_allclose(jnp.exp(log_w).sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(tr["w"]["value"], jnp.exp(log_w), rtol=1e-6)


@scaffold
def test_hierarchical_weights_zero_spread_is_complete_pooling() -> None:
    groups = jnp.array([0, 1, 2, 0, 1, 2])
    _, log_w = _trace_block(groups, 3, conditioned={"w_sigma_0": jnp.zeros(2)})
    np.testing.assert_allclose(log_w, jnp.broadcast_to(log_w[:1], log_w.shape), atol=1e-6)


@scaffold
def test_hierarchical_weights_unseen_group_gets_population_weights() -> None:
    groups = jnp.array([0, 1, -1])
    mu = jnp.array([0.4, -0.2])
    tr, log_w = _trace_block(groups, 2, conditioned={"w_mu": mu})
    expected = jax.nn.log_softmax(jnp.concatenate([mu, jnp.zeros(1)]))
    np.testing.assert_allclose(log_w[2], expected, atol=1e-6)
    assert tr["w_sigma_0"]["value"].shape == (2,)


@scaffold
def test_hierarchical_weights_random_walk_offsets_are_centered() -> None:
    # With mu fixed at zero and K = 2, logit(w) = alpha is the group offset itself; a centered
    # random walk over the groups averages to zero.
    def model() -> None:
        log_w = hierarchical_weights("w", 2, jnp.arange(6), 6, group_prior="random_walk")
        numpyro.deterministic("alpha", log_w[:, 0] - log_w[:, 1])

    tr = trace(seed(condition(model, {"w_mu": jnp.zeros(1)}), random.PRNGKey(0))).get_trace()
    np.testing.assert_allclose(tr["alpha"]["value"].mean(), 0.0, atol=1e-6)
    assert tr["w_z_0"]["value"].shape == (5, 1)  # J - 1 increments


@scaffold
def test_hierarchical_weights_several_group_indices_are_additive() -> None:
    horizon = jnp.array([0, 1, 0, 1])
    series = jnp.array([0, 0, 1, 1])
    tr, log_w = _trace_block(
        (horizon, series), (2, 2), group_prior=("random_walk", "exchangeable")
    )
    assert tr["w_sigma_0"]["value"].shape == (2,) and tr["w_sigma_1"]["value"].shape == (2,)
    assert tr["w_z_1"]["value"].shape == (2, 2)
    assert log_w.shape == (4, 3)


@scaffold
def test_hierarchical_weights_reject_a_single_model() -> None:
    def model() -> None:
        hierarchical_weights("w", 1, jnp.zeros(3, dtype=jnp.int32), 1)

    with pytest.raises(StackingShapeError):
        trace(seed(model, random.PRNGKey(0))).get_trace()


@scaffold
def test_stacking_factor_registers_the_objective() -> None:
    lpd = jnp.asarray(synthetic_log_density(np.random.default_rng(25), 5, 4, 3))
    log_w = jnp.log(jnp.broadcast_to(jnp.array([0.2, 0.5, 0.3]), lpd.shape))
    cell_weights = jnp.linspace(0.5, 1.0, lpd.shape[0])

    def model() -> None:
        stacking_factor("stacking", lpd, log_w, cell_weights=cell_weights)

    tr = trace(seed(model, random.PRNGKey(0))).get_trace()
    site = tr["stacking"]
    expected = float((cell_weights * logsumexp(lpd + log_w, axis=1)).sum())
    np.testing.assert_allclose(site["fn"].log_prob(site["value"]), expected, rtol=1e-6)


@scaffold
@pytest.mark.slow
def test_stacking_model_recovers_group_varying_weights_with_nuts() -> None:
    # Simulated cells whose true mixture weight on model 0 differs by group; the posterior
    # mean of the block's weights must land within 0.1 of each group's truth, and the
    # unseen group (-1) must sit near the population average.
    rng = np.random.default_rng(26)
    truth_w = np.array([0.9, 0.5, 0.1])
    groups = np.repeat(np.arange(3), 400)
    pick = rng.uniform(size=1_200) < truth_w[groups]
    # Model 0 scores higher where it is the better model and vice versa.
    lpd = np.where(pick[:, None], [[-1.0, -3.0]], [[-3.0, -1.0]]) + rng.normal(0, 0.2, (1_200, 2))
    cells = jnp.concatenate([jnp.asarray(groups), jnp.array([-1])])

    def model(lpd_train: Array) -> None:
        log_w = hierarchical_weights("w", 2, cells, 3, tau_sigma=2.0)
        stacking_factor("stacking", lpd_train, log_w[:-1])

    mcmc = MCMC(NUTS(model), num_warmup=300, num_samples=300, progress_bar=False)
    mcmc.run(random.PRNGKey(0), jnp.asarray(lpd))
    w = np.asarray(mcmc.get_samples()["w"]).mean(axis=0)  # (cells + 1, 2)
    for g in range(3):
        assert abs(w[groups == g, 0].mean() - truth_w[g]) < 0.1
    assert 0.2 < w[-1, 0] < 0.8


# --- stack_forecasts -------------------------------------------------------------------------


def _model_draws() -> Array:
    key = random.PRNGKey(27)
    return jnp.stack(
        [
            jnp.cumsum(random.normal(random.fold_in(key, 0), (200, 5, 1)), axis=1),
            10.0 + jnp.cumsum(random.normal(random.fold_in(key, 1), (200, 5, 1)), axis=1),
        ]
    )


@scaffold
def test_stack_forecasts_one_hot_weights_select_that_model() -> None:
    draws = _model_draws()
    got = stack_forecasts(random.PRNGKey(0), draws, jnp.array([0.0, 1.0]))
    np.testing.assert_array_equal(got, draws[1])


@scaffold
def test_stack_forecasts_path_mode_keeps_whole_paths() -> None:
    draws = _model_draws()
    got = stack_forecasts(random.PRNGKey(0), draws, jnp.array([0.5, 0.5]), per="path")
    assert got.shape == (200, 5, 1)
    from_zero = jnp.all(got == draws[0], axis=(1, 2))
    from_one = jnp.all(got == draws[1], axis=(1, 2))
    assert bool(jnp.all(from_zero | from_one))
    assert 40 < int(from_zero.sum()) < 160


@scaffold
def test_stack_forecasts_cell_mode_matches_the_mixture_mean() -> None:
    draws = jnp.stack(
        [
            random.normal(random.PRNGKey(1), (20_000, 3, 1)),
            4.0 + random.normal(random.PRNGKey(2), (20_000, 3, 1)),
        ]
    )
    weights = jnp.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])[:, None, :]  # (future, obs, models)
    got = stack_forecasts(random.PRNGKey(0), draws, weights, per="cell")
    np.testing.assert_allclose(got.mean(axis=0)[:, 0], 4.0 * weights[:, 0, 1], atol=0.1)


@scaffold
def test_stack_forecasts_auto_resolution() -> None:
    draws = _model_draws()
    single = stack_forecasts(random.PRNGKey(0), draws, jnp.array([0.5, 0.5]))
    per_path = stack_forecasts(random.PRNGKey(0), draws, jnp.array([0.5, 0.5]), per="path")
    np.testing.assert_array_equal(single, per_path)
    varying = jnp.broadcast_to(jnp.array([0.5, 0.5]), (5, 1, 2))
    per_cell = stack_forecasts(random.PRNGKey(0), draws, varying)
    assert per_cell.shape == (200, 5, 1)


@scaffold
@pytest.mark.parametrize(
    ("weights", "per"),
    [
        (jnp.array([-0.1, 1.1]), "auto"),
        (jnp.array([0.3, 0.3]), "auto"),
        (jnp.broadcast_to(jnp.array([0.5, 0.5]), (5, 1, 2)), "path"),
    ],
    ids=["negative", "not_normalized", "path_with_cell_weights"],
)
def test_stack_forecasts_reject_bad_weights(weights: Array, per: str) -> None:
    with pytest.raises(StackingShapeError):
        stack_forecasts(random.PRNGKey(0), _model_draws(), weights, per=per)  # type: ignore[arg-type]


# --- stacking_cells --------------------------------------------------------------------------


def _fake_results(
    name_seed: int,
    *,
    windows: int = 3,
    future: int = 4,
    reduction: str = "marginal",
    with_predictions: bool = False,
    num_samples: int = 10,
) -> list[BacktestResult]:
    rng = np.random.default_rng(name_seed)
    results = []
    for w in range(windows):
        t1 = 20 + w * future
        shape = (future, 1) if reduction == "marginal" else (1, 1)
        results.append(
            BacktestResult(
                t0=0,
                t1=t1,
                t2=t1 + future,
                num_samples=num_samples,
                walltime=0.0,
                metrics={},
                prediction=rng.normal(size=(num_samples, future, 1)) if with_predictions else None,
                log_density=rng.normal(-1.5, 0.3, size=shape),
                log_density_reduction=reduction,  # type: ignore[arg-type]
            )
        )
    return results


def _series(length: int = 40) -> Array:
    return jnp.arange(length, dtype=jnp.float32)[:, None]


@scaffold
def test_stacking_cells_layout_is_window_major_then_horizon() -> None:
    results = {"a": _fake_results(0), "b": _fake_results(1)}
    cells = stacking_cells(results, _series())
    assert isinstance(cells, StackingCells)
    assert cells.model_names == ("a", "b")
    assert cells.log_density.shape == (12, 2)
    np.testing.assert_array_equal(cells.window, np.repeat(np.arange(3), 4))
    np.testing.assert_array_equal(cells.horizon, np.tile(np.arange(4), 3))
    np.testing.assert_array_equal(cells.series, np.zeros(12, dtype=int))
    np.testing.assert_array_equal(cells.truth, np.asarray(_series())[20:32, 0])
    np.testing.assert_allclose(
        cells.log_density[:, 0], np.concatenate([r.log_density[:, 0] for r in results["a"]])
    )
    assert cells.reduction == "marginal"
    assert cells.draws is None


@scaffold
def test_stacking_cells_joint_reduction_has_one_cell_per_window() -> None:
    results = {"a": _fake_results(0, reduction="joint"), "b": _fake_results(1, reduction="joint")}
    cells = stacking_cells(results, _series())
    assert cells.log_density.shape == (3, 2)
    np.testing.assert_array_equal(cells.horizon, -np.ones(3, dtype=int))


@scaffold
def test_stacking_cells_collects_draws_when_every_model_kept_them() -> None:
    results = {
        "a": _fake_results(0, with_predictions=True),
        "b": _fake_results(1, with_predictions=True),
    }
    cells = stacking_cells(results, _series())
    assert cells.draws is not None
    assert cells.draws.shape == (2, 10, 12)


@scaffold
def test_stacking_cells_reject_mismatched_windows() -> None:
    mismatched = _fake_results(1)
    mismatched[1] = BacktestResult(**{**mismatched[1].to_dict(), "t1": 99})
    with pytest.raises(StackingShapeError, match="window"):
        stacking_cells({"a": _fake_results(0), "b": mismatched}, _series())


@scaffold
def test_stacking_cells_reject_mixed_reductions() -> None:
    with pytest.raises(StackingShapeError, match="reduction"):
        stacking_cells(
            {"a": _fake_results(0), "b": _fake_results(1, reduction="joint")}, _series()
        )


@scaffold
def test_stacking_cells_require_log_density_on_every_result() -> None:
    missing = [BacktestResult(**{**r.to_dict(), "log_density": None}) for r in _fake_results(1)]
    with pytest.raises(StackingShapeError, match="log_density"):
        stacking_cells({"a": _fake_results(0), "b": missing}, _series())


@scaffold
def test_stacking_cells_require_two_models() -> None:
    with pytest.raises(StackingShapeError, match="two"):
        stacking_cells({"a": _fake_results(0)}, _series())


@scaffold
def test_stacking_cells_reject_partial_predictions() -> None:
    with pytest.raises(StackingShapeError, match="prediction"):
        stacking_cells(
            {"a": _fake_results(0, with_predictions=True), "b": _fake_results(1)}, _series()
        )


# --- End to end on cells: the comparison the notebook has to make ---------------------------


@scaffold
def test_held_out_comparison_carries_a_paired_standard_error_across_windows() -> None:
    # The report format every stacking result must come in: the mean per-unit gain over
    # the equal-weights baseline on held-out windows and its paired standard error across
    # windows, never across cells (reviewer findings 1 and 2).
    rng = np.random.default_rng(30)
    lpd = synthetic_log_density(rng, 60, 7, 3, shared=0.3)
    window = np.repeat(np.arange(60), 7)
    fit, held = window < 40, window >= 40
    w = stacking_weights(lpd[fit])
    gain = np.asarray(
        log_score(lpd[held], w, unit=window[held])
        - log_score(lpd[held], jnp.full(3, 1 / 3), unit=window[held])
    )
    assert gain.shape == (20,)
    se = gain.std(ddof=1) / np.sqrt(gain.size)
    assert np.isfinite(se) and se > 0.0
