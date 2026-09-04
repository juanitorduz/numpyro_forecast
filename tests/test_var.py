"""Tests for the vector autoregression components in `numpyro_forecast.var`."""

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from conftest import get_trace, plate_frames
from jax import Array, random
from jaxtyping import TypeCheckError
from numpyro.handlers import substitute
from numpyro.infer import Predictive

from numpyro_forecast.arrays import pad_future
from numpyro_forecast.convert import to_datatree
from numpyro_forecast.models import Horizon, SSOEResult, ssoe
from numpyro_forecast.predictive import forecast, predict_in_sample
from numpyro_forecast.typing import ForecastModel
from numpyro_forecast.var import companion_matrix, impulse_response, var_mean, var_step

P, K = 2, 2
PosteriorFactory = Callable[[Array, ForecastModel, Array, Array], dict[str, Array]]


def _phi(key: int = 0, scale: float = 0.3, batch: tuple[int, ...] = ()) -> Array:
    """Random coefficient tensor ``(*batch, P, K, K)``; ``scale=0.3`` keeps it stable."""
    return scale * random.normal(random.PRNGKey(key), (*batch, P, K, K))


def _lags(key: int = 1, batch: tuple[int, ...] = ()) -> Array:
    """A random lag window ``(*batch, P, K)``, most recent row last."""
    return random.normal(random.PRNGKey(key), (*batch, P, K))


def _lower_tril(k: int) -> Array:
    return jnp.tril(0.3 * jnp.ones((k, k))) + 0.7 * jnp.eye(k)


def test_var_mean_matches_hand_computed_var2() -> None:
    phi, lags, c = _phi(), _lags(), jnp.array([0.5, -0.5])
    # lags[-1] is y_{t-1} and pairs with Phi_1 = phi[0]; lags[-2] is y_{t-2} and pairs with phi[1].
    expected = c + phi[0] @ lags[-1] + phi[1] @ lags[-2]
    assert jnp.allclose(var_mean(phi, lags, c), expected, atol=1e-6)


def test_var_mean_without_intercept() -> None:
    phi, lags = _phi(), _lags()
    assert jnp.allclose(var_mean(phi, lags), phi[0] @ lags[-1] + phi[1] @ lags[-2], atol=1e-6)


def test_var_mean_broadcasts_shared_phi_over_a_panel() -> None:
    phi, lags = _phi(), _lags(batch=(3,))
    out = var_mean(phi, lags)
    assert out.shape == (3, K)
    for b in range(3):
        assert jnp.allclose(out[b], var_mean(phi, lags[b]), atol=1e-6)


def test_var_mean_broadcasts_batched_phi_over_a_single_window() -> None:
    phi, lags = _phi(batch=(3,)), _lags()
    out = var_mean(phi, lags)
    assert out.shape == (3, K)
    for b in range(3):
        assert jnp.allclose(out[b], var_mean(phi[b], lags), atol=1e-6)


def test_var_mean_rejects_mismatched_lag_count() -> None:
    with pytest.raises(TypeCheckError):
        var_mean(_phi(), jnp.zeros((P + 1, K)))


def test_var_step_mean_and_window_shift() -> None:
    phi, lags, c = _phi(), _lags(), jnp.array([0.1, 0.2])
    mu, carry_fn = var_step(phi, c)(lags, None)
    assert jnp.allclose(mu, var_mean(phi, lags, c))
    y_t = jnp.array([1.0, 2.0])
    new = carry_fn(y_t, y_t - mu)
    assert new.shape == lags.shape
    assert new.dtype == lags.dtype
    assert jnp.array_equal(new[:-1], lags[1:])
    assert jnp.array_equal(new[-1], y_t)


def test_var_step_ignores_exogenous_input() -> None:
    phi, lags = _phi(), _lags()
    step = var_step(phi)
    mu_none, _ = step(lags, None)
    mu_x, _ = step(lags, jnp.ones((4,)))
    assert jnp.array_equal(mu_none, mu_x)


def test_var_step_varx_by_wrapping() -> None:
    phi, lags, beta = _phi(), _lags(), jnp.array([[1.0, 0.0], [0.0, 2.0]])
    base = var_step(phi)

    def step(carry: Array, x_t: Array) -> tuple[Array, Any]:
        mu, carry_fn = base(carry, x_t)
        return mu + beta @ x_t, carry_fn

    x_t = jnp.array([0.5, -1.0])
    mu, _ = step(lags, x_t)
    assert jnp.allclose(mu, var_mean(phi, lags) + beta @ x_t)


def test_var_step_rejects_wrong_lag_count_with_guidance() -> None:
    with pytest.raises(ValueError, match=r"lags=2.*init_carry=y\[\.\.\., :2, :\]"):
        var_step(_phi())(jnp.zeros((P + 1, K)), None)


def _var_series(t: int, phi: Array, c: Array, scale_tril: Array, key: int = 3) -> Array:
    """Simulate ``t`` rows of a VAR(P) from a zero window."""
    shocks = random.normal(random.PRNGKey(key), (t, K)) @ scale_tril.T

    def body(window: Array, eps: Array) -> tuple[Array, Array]:
        y_t = var_mean(phi, window, c) + eps
        return jnp.concatenate([window[1:], y_t[None]], axis=0), y_t

    _, y = jax.lax.scan(body, jnp.zeros((P, K)), shocks)
    return y


def _make_var_model(y_init: Array) -> tuple[ForecastModel, list[SSOEResult]]:
    """The observed-VAR model of the notebook, with a fixed known noise scale."""
    box: list[SSOEResult] = []
    scale_tril = _lower_tril(K)

    def model(covariates: Array, data: Array | None = None) -> None:
        h = Horizon.from_data(covariates, data)
        y = covariates[..., : h.t_obs, :]
        intercept = jnp.asarray(
            numpyro.sample("intercept", dist.Normal(0.0, 1.0).expand([K]).to_event(1))
        )
        phi = jnp.asarray(
            numpyro.sample("phi", dist.Normal(0.0, 0.5).expand([P, K, K]).to_event(3))
        )
        noise = dist.MultivariateNormal(jnp.zeros(K), scale_tril=scale_tril)
        r = ssoe(h, "eps", y, y_init, var_step(phi, intercept), noise)
        box[:] = [r]
        numpyro.deterministic("mu_t", r.mu)
        numpyro.sample("obs", dist.MultivariateNormal(r.mu, scale_tril=scale_tril), obs=h.data)
        if h.future > 0:
            numpyro.deterministic("forecast", r.y_future)

    return model, box


def _split(t_total: int, future: int) -> tuple[Array, Array, Array]:
    """Simulated series split into ``(y_init, data, covariates_full)`` as the notebook does."""
    y_all = _var_series(t_total, _phi(), jnp.array([0.3, -0.2]), _lower_tril(K))
    y_init, data = y_all[:P], y_all[P:]
    return y_init, data, pad_future(data, future)


def test_ssoe_var_sites_and_shapes() -> None:
    future = 4
    y_init, data, covariates = _split(24, future)
    model, box = _make_var_model(y_init)
    tr = get_trace(model, covariates, data)
    site = tr["eps_future"]
    assert site["value"].shape == (future, K)
    assert plate_frames(site) == [("time_future", -1, future)]
    assert tr["forecast"]["value"].shape == (future, K)
    assert box[-1].mu.shape == (data.shape[0], K)


def test_ssoe_var_in_sample_mean_is_the_lag_regression() -> None:
    y_init, data, covariates = _split(24, 0)
    model, box = _make_var_model(y_init)
    tr = get_trace(model, covariates, data)
    phi, c = tr["phi"]["value"], tr["intercept"]["value"]
    full = jnp.concatenate([y_init, data], axis=0)
    expected = jnp.stack([var_mean(phi, full[t : t + P], c) for t in range(data.shape[0])])
    assert jnp.allclose(box[-1].mu, expected, atol=1e-5)


def test_ssoe_var_forecast_with_zero_errors_is_the_deterministic_recursion() -> None:
    future = 5
    y_init, data, covariates = _split(24, future)
    model, box = _make_var_model(y_init)
    tr = get_trace(model, covariates, data, substitutions={"eps_future": jnp.zeros((future, K))})
    phi, c = tr["phi"]["value"], tr["intercept"]["value"]
    window = list(jnp.concatenate([y_init, data], axis=0)[-P:])
    expected = []
    for _ in range(future):
        y_next = var_mean(phi, jnp.stack(window), c)
        expected.append(y_next)
        window = [*window[1:], y_next]
    assert jnp.allclose(box[-1].y_future, jnp.stack(expected), atol=1e-5)
    assert jnp.allclose(tr["forecast"]["value"], jnp.stack(expected), atol=1e-5)


def test_ssoe_var_end_to_end(posterior_factory: PosteriorFactory, rng_key: Array) -> None:
    future = 3
    y_init, data, covariates = _split(30, future)
    model, _ = _make_var_model(y_init)
    key_fit, key_fc, key_pp = random.split(rng_key, 3)
    posterior = posterior_factory(key_fit, model, data, data)
    assert set(posterior) >= {"intercept", "phi"}
    pred = forecast(key_fc, model, posterior, data, covariates)
    assert pred.shape == (posterior["phi"].shape[0], future, K)
    assert bool(jnp.all(jnp.isfinite(pred)))
    in_sample = predict_in_sample(key_pp, model, posterior, data)
    assert in_sample.shape == (posterior["phi"].shape[0], data.shape[0], K)


def test_ssoe_var_to_datatree_dims() -> None:
    future = 3
    y_init, data, covariates = _split(20, future)
    model, _ = _make_var_model(y_init)
    tr = get_trace(model, data, data)
    s = 4
    posterior = {
        name: jnp.broadcast_to(tr[name]["value"], (s, *tr[name]["value"].shape))
        for name in ("intercept", "phi", "mu_t")
    }
    tree = to_datatree(
        random.PRNGKey(1),
        model,
        posterior,
        data,
        covariates,
        posterior_dims={"phi": ["lag", "series", "series_lagged"], "mu_t": ["time", "obs_dim"]},
    )
    assert tree["predictions"]["obs"].shape == (1, s, future, K)
    assert tree["posterior"]["phi"].dims == ("chain", "draw", "lag", "series", "series_lagged")
    assert tree["posterior"]["mu_t"].dims == ("chain", "draw", "time", "obs_dim")


def test_companion_matrix_p1_is_phi() -> None:
    phi = _phi()[:1]
    assert jnp.array_equal(companion_matrix(phi), phi[0])


def test_companion_matrix_block_layout() -> None:
    phi = _phi()
    f = companion_matrix(phi)
    assert f.shape == (P * K, P * K)
    assert jnp.array_equal(f[:K, :K], phi[0])
    assert jnp.array_equal(f[:K, K:], phi[1])
    assert jnp.array_equal(f[K:, :K], jnp.eye(K))
    assert jnp.array_equal(f[K:, K:], jnp.zeros((K, K)))


def test_companion_matrix_state_is_the_reversed_window() -> None:
    phi, lags = _phi(), _lags()
    state = lags[::-1].reshape(-1)  # most recent first
    assert jnp.allclose((companion_matrix(phi) @ state)[:K], var_mean(phi, lags), atol=1e-6)


def test_companion_matrix_batched() -> None:
    phi = _phi(batch=(5,))
    f = companion_matrix(phi)
    assert f.shape == (5, P * K, P * K)
    assert jnp.array_equal(f[2], companion_matrix(phi[2]))


def test_impulse_response_equals_companion_powers() -> None:
    phi, horizon = _phi(), 6
    psi = impulse_response(phi, horizon)
    assert psi.shape == (horizon + 1, K, K)
    f = companion_matrix(phi)
    for h in range(horizon + 1):
        expected = jnp.linalg.matrix_power(f, h)[:K, :K]
        assert jnp.allclose(psi[h], expected, atol=1e-5)


def test_impulse_response_matches_a_unit_shock_simulation() -> None:
    phi, horizon = _phi(), 5
    psi = impulse_response(phi, horizon)
    for j in range(K):
        window = [jnp.zeros(K)] * P
        y_t = jnp.eye(K)[j]  # a unit shock to variable j at step 0, no intercept
        responses = [y_t]
        for _ in range(horizon):
            window = [*window[1:], y_t]
            y_t = var_mean(phi, jnp.stack(window))
            responses.append(y_t)
        assert jnp.allclose(psi[:, :, j], jnp.stack(responses), atol=1e-5)


def test_impulse_response_horizon_zero_is_identity() -> None:
    psi = impulse_response(_phi(), 0)
    assert psi.shape == (1, K, K)
    assert jnp.array_equal(psi[0], jnp.eye(K))


def test_impulse_response_rejects_negative_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        impulse_response(_phi(), -1)


def test_impulse_response_batches_over_draws_like_vmap() -> None:
    phi = _phi(batch=(7,))
    batched = impulse_response(phi, 4)
    mapped = jax.vmap(partial(impulse_response, horizon=4))(phi)
    assert batched.shape == (7, 5, K, K)
    assert jnp.allclose(batched, mapped, atol=1e-6)


@pytest.mark.parametrize("batched_l", [False, True], ids=["shared-L", "per-draw-L"])
def test_impulse_response_orthogonalized_is_psi_times_scale_tril(batched_l: bool) -> None:
    phi = _phi(batch=(3,))
    scale_tril = (
        jnp.stack([_lower_tril(K) * (1.0 + b) for b in range(3)]) if batched_l else _lower_tril(K)
    )
    theta = impulse_response(phi, 4, scale_tril=scale_tril)
    psi = impulse_response(phi, 4)
    for b in range(3):
        l_b = scale_tril[b] if batched_l else scale_tril
        assert jnp.allclose(theta[b], psi[b] @ l_b, atol=1e-6)
    assert jnp.array_equal(theta[:, 0], jnp.broadcast_to(scale_tril, theta[:, 0].shape))


def test_impulse_response_cumulative_is_the_running_sum() -> None:
    phi = _phi()
    assert jnp.allclose(
        impulse_response(phi, 5, cumulative=True), jnp.cumsum(impulse_response(phi, 5), axis=0)
    )


def test_impulse_response_is_jittable() -> None:
    phi = _phi()
    jitted = jax.jit(partial(impulse_response, horizon=5))
    assert jnp.allclose(jitted(phi), impulse_response(phi, 5), atol=1e-6)


def test_companion_matrix_spectral_radius_of_draws() -> None:
    """The stability diagnostic of the notebook runs on the companion matrix of posterior draws."""
    phi = _phi(batch=(6,), scale=0.2)
    radius = np.abs(np.linalg.eigvals(np.asarray(companion_matrix(phi)))).max(-1)
    assert radius.shape == (6,)
    assert bool(np.all(radius < 1.0))


def test_ssoe_var_predictive_with_zero_errors_is_deterministic() -> None:
    """``Predictive`` over posterior draws with zeroed errors reproduces the recursion per draw."""
    y_init, data, covariates = _split(12, 2)
    model, _ = _make_var_model(y_init)
    posterior = {
        "intercept": jnp.zeros((3, K)),
        "phi": jnp.broadcast_to(_phi(), (3, P, K, K)),
    }
    zero_errors = substitute(model, data={"eps_future": jnp.zeros((2, K))})
    pred = Predictive(zero_errors, posterior_samples=posterior, return_sites=["forecast"])
    out = pred(random.PRNGKey(0), covariates, data)["forecast"]
    assert out.shape == (3, 2, K)
    assert jnp.allclose(out[0], out[1])
