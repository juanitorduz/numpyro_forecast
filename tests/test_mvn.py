"""MultivariateNormal distribution-surgery tests (roadmap §9)."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from conftest import as_autoguide
from jax import Array, random

from numpyro_forecast.exceptions import MVNLayoutError
from numpyro_forecast.functional import (
    Horizon,
    draw_posterior,
    fit_svi,
    forecasting_model,
    predict,
)
from numpyro_forecast.surgery import (
    _mvn_time_params,
    prefix_condition,
    shift_loc,
    slice_time,
)
from tests.conftest import empty_covariates


class _RawMVN(dist.MultivariateNormal):
    """An MVN with ``loc``/``covariance_matrix`` set directly, bypassing init.

    numpyro normalizes a real ``MultivariateNormal.loc`` to ``(*batch, time)``, so
    the ``(*batch, time, 1)`` and ``time == 1`` layouts that ``_mvn_time_params``
    documents cannot be built through the real constructor. This subclass keeps
    ``isinstance(..., MultivariateNormal)`` true (so the runtime type check
    accepts it) while exposing the raw arrays those branches expect.
    """

    def __init__(self, loc: Array, cov: Array) -> None:
        self.loc = loc
        self._raw_cov = cov

    @property
    def covariance_matrix(self) -> Array:
        return self._raw_cov


def _fake_mvn(loc: Array, cov: Array) -> dist.MultivariateNormal:
    return _RawMVN(loc, cov)


def _ar_covariance(time: int, phi: Array | float) -> Array:
    idx = jnp.arange(time)
    return phi ** jnp.abs(idx[:, None] - idx[None, :])


def test_mvn_prefix_matches_numpy_closed_form() -> None:
    """Cholesky conditional matches a direct numpy solve on an AR covariance."""
    time, t = 5, 3
    phi = 0.6
    loc = jnp.zeros(time)
    cov = _ar_covariance(time, phi)
    mvn = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)
    data = jnp.array([[0.2], [0.1], [-0.3]])
    cond = prefix_condition(mvn, data)
    assert isinstance(cond, dist.MultivariateNormal)
    mu_f = loc[t:]
    sigma_pp = cov[:t, :t]
    sigma_fp = cov[t:, :t]
    expected_mean = mu_f + sigma_fp @ jnp.linalg.solve(sigma_pp, data[:, 0] - loc[:t])
    assert jnp.allclose(cond.loc, expected_mean, atol=1e-4)


def test_mvn_diagonal_matches_independent_normal_slice() -> None:
    """``MVN(mu, sigma^2 I)`` conditional equals ``Independent(Normal)`` slice."""
    time, t = 6, 4
    loc = jnp.arange(float(time))
    cov = jnp.eye(time)
    mvn = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)
    indep = dist.Normal(loc=loc[:, None], scale=1.0)
    data = jnp.zeros((t, 1))
    mvn_future = prefix_condition(mvn, data)
    norm_future = prefix_condition(indep, data)
    grid = jnp.linspace(-2.0, 2.0, 7)
    for x in grid:
        xv = jnp.full((time - t, 1), x)
        assert jnp.allclose(
            mvn_future.log_prob(xv[:, 0]),
            jnp.asarray(norm_future.log_prob(xv)).sum(),
            atol=1e-5,
        )


def test_mvn_near_singular_prefix_still_valid() -> None:
    """A near-singular ``Sigma_pp`` yields a valid distribution via jitter."""
    time, t = 4, 2
    cov = jnp.ones((time, time)) * 0.99 + jnp.eye(time) * 0.01
    mvn = dist.MultivariateNormal(loc=jnp.zeros(time), covariance_matrix=cov)
    data = jnp.zeros((t, 1))
    cond = prefix_condition(mvn, data)
    assert isinstance(cond, dist.MultivariateNormal)
    cond_cov = jnp.asarray(cond.covariance_matrix)
    assert jnp.all(jnp.isfinite(cond_cov))
    assert jnp.all(jnp.linalg.eigvalsh(cond_cov) > 0)


def test_mvn_shift_loc_adds_to_mean() -> None:
    loc = jnp.zeros(4)
    cov = jnp.eye(4)
    mvn = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)
    shift = jnp.ones((4, 1))
    shifted = shift_loc(mvn, shift)
    assert isinstance(shifted, dist.MultivariateNormal)
    assert jnp.allclose(shifted.loc, jnp.ones(4))


def test_mvn_slice_time_marginal_block() -> None:
    loc = jnp.arange(6.0)
    cov = _ar_covariance(6, 0.5)
    mvn = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)
    sliced = slice_time(mvn, slice(1, 4))
    assert isinstance(sliced, dist.MultivariateNormal)
    assert sliced.loc.shape == (3,)
    assert jnp.asarray(sliced.covariance_matrix).shape == (3, 3)


def test_gp_noise_end_to_end() -> None:
    """GP-style correlated noise fits through ``fit_svi``."""

    def gp_body(h: Horizon, covariates: Array) -> None:
        time = h.duration
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        rho = numpyro.sample("rho", dist.Beta(2.0, 2.0))
        cov = _ar_covariance(time, jnp.asarray(rho)) * sigma**2
        noise = dist.MultivariateNormal(loc=jnp.zeros(time), covariance_matrix=cov)
        level = numpyro.sample("level", dist.Normal(0.0, 1.0))
        predict(h, noise, jnp.asarray(level) + jnp.zeros((time, 1)))

    model = forecasting_model(gp_body)
    data = random.normal(random.PRNGKey(0), (12, 1))
    cov = empty_covariates(16)
    fit = fit_svi(random.PRNGKey(1), model, data, cov[:12], num_steps=150)
    post = draw_posterior(random.PRNGKey(2), as_autoguide(fit.guide), fit.params, 30)
    assert set(post) >= {"sigma", "rho", "level"}
    assert jnp.all(jnp.isfinite(post["sigma"]))


@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
def test_mvn_time_params_accepts_batched_layouts(batch: tuple[int, ...]) -> None:
    """A ``(*batch, time)`` loc resolves unchanged (regression: batched was rejected)."""
    time = 6
    loc = random.normal(random.PRNGKey(0), (*batch, time))
    cov = jnp.broadcast_to(jnp.eye(time), (*batch, time, time))
    mvn = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)
    out_loc, out_cov = _mvn_time_params(mvn)
    assert out_loc.shape == (*batch, time)
    assert out_cov.shape == (*batch, time, time)
    assert jnp.allclose(out_loc, loc)


def test_mvn_time_params_squeezes_trailing_singleton() -> None:
    """A ``(*batch, time, 1)`` loc is squeezed to ``(*batch, time)``."""
    loc = jnp.arange(3.0 * 6.0).reshape(3, 6, 1)
    cov = jnp.broadcast_to(jnp.eye(6), (3, 6, 6))
    out_loc, _ = _mvn_time_params(_fake_mvn(loc, cov))
    assert out_loc.shape == (3, 6)
    assert jnp.allclose(out_loc, loc[..., 0])


def test_mvn_time_params_time_one_tiebreak() -> None:
    """The remaining ``time == 1`` tie: a 2-d ``(*batch, 1)`` loc is kept as-is.

    The 3-axis ``(*batch, time, 1)`` layout is squeezed even at ``time == 1``
    (see ``test_mvn_time_params_time_one_squeezes_three_axis_layout``); the only
    ambiguity left is a 2-d loc whose trailing axis could be time or a squeezable
    singleton, resolved as "the trailing axis is time".
    """
    loc = jnp.zeros((3, 1))
    cov = jnp.broadcast_to(jnp.eye(1), (3, 1, 1))
    out_loc, _ = _mvn_time_params(_fake_mvn(loc, cov))
    # The squeeze branch requires shape[-2] == time (3 != 1 here), so the
    # loc falls through to the (*batch, time) branch and is kept unchanged.
    assert out_loc.shape == (3, 1)


def test_mvn_time_params_time_one_squeezes_three_axis_layout() -> None:
    """A ``(*batch, time, 1)`` loc with ``time == 1`` squeezes to ``(*batch, 1)``.

    Regression: the ``shape[-1] == time`` branch used to match first when
    ``time == 1`` (a trailing singleton also equals time), returning the loc
    unsqueezed with a phantom trailing axis.
    """
    loc = jnp.zeros((3, 1, 1))
    cov = jnp.broadcast_to(jnp.eye(1), (3, 1, 1))
    out_loc, out_cov = _mvn_time_params(_fake_mvn(loc, cov))
    assert out_loc.shape == (3, 1)
    assert out_cov.shape == (3, 1, 1)


def test_mvn_time_params_batch_cov_mismatch_raises() -> None:
    """A loc batch incompatible with the cov batch raises at the boundary.

    Without the guard, a mismatched user construction flows downstream and dies
    deep in ``cho_solve`` with a shape error that never names the layout rule.
    """
    loc = jnp.zeros((3, 6))
    cov = jnp.broadcast_to(jnp.eye(6), (4, 6, 6))
    with pytest.raises(MVNLayoutError, match="MultivariateNormal"):
        _mvn_time_params(_fake_mvn(loc, cov))


def test_mvn_time_params_rejects_mismatched_trailing_axis() -> None:
    """A loc whose trailing axis is neither time nor squeezable-1 raises."""
    loc = jnp.zeros((3, 4))  # trailing 4 != time 6, not 1
    cov = jnp.broadcast_to(jnp.eye(6), (3, 6, 6))
    with pytest.raises(MVNLayoutError, match="MultivariateNormal"):
        _mvn_time_params(_fake_mvn(loc, cov))
    with pytest.raises(MVNLayoutError):
        _mvn_time_params(_fake_mvn(jnp.asarray(0.0), jnp.eye(6)))


def test_shift_loc_mvn_batched() -> None:
    """shift_loc on a batched MVN shifts loc elementwise; a non-singleton extra axis raises."""
    time = 6
    loc = random.normal(random.PRNGKey(0), (3, time))
    cov = jnp.broadcast_to(jnp.eye(time), (3, time, time))
    mvn = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)
    shift = jnp.ones((3, time, 1))  # (*batch, time, obs=1) as predict supplies
    shifted = shift_loc(mvn, shift)
    assert isinstance(shifted, dist.MultivariateNormal)
    assert jnp.allclose(shifted.loc, loc + 1.0)
    # A non-size-1 extra trailing axis is a layout error, not a silent truncation.
    with pytest.raises(MVNLayoutError, match=MVNLayoutError.default_message[:20]):
        shift_loc(mvn, jnp.ones((3, time, 2)))


def test_slice_time_and_prefix_condition_mvn_batched() -> None:
    """Batched slice_time/prefix_condition match the per-batch-element results."""
    batch, time, t = 3, 6, 3
    locs = random.normal(random.PRNGKey(1), (batch, time))
    covs = jnp.stack([_ar_covariance(time, 0.3 + 0.15 * i) for i in range(batch)])
    mvn = dist.MultivariateNormal(loc=locs, covariance_matrix=covs)
    data = jnp.zeros((batch, t, 1))

    sliced = slice_time(mvn, slice(1, 4))
    cond = prefix_condition(mvn, data)
    assert isinstance(sliced, dist.MultivariateNormal)
    assert isinstance(cond, dist.MultivariateNormal)
    assert sliced.loc.shape == (batch, 3)

    for b in range(batch):
        mvn_b = dist.MultivariateNormal(loc=locs[b], covariance_matrix=covs[b])
        sliced_b = slice_time(mvn_b, slice(1, 4))
        cond_b = prefix_condition(mvn_b, data[b])
        assert isinstance(sliced_b, dist.MultivariateNormal)
        assert isinstance(cond_b, dist.MultivariateNormal)
        assert jnp.allclose(sliced.loc[b], sliced_b.loc)
        assert jnp.allclose(cond.loc[b], cond_b.loc, atol=1e-4)
