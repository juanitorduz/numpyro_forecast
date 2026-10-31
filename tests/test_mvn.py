"""MultivariateNormal distribution-surgery tests (roadmap §9)."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro.infer import SVI, Trace_ELBO, log_likelihood
from numpyro.infer.autoguide import AutoNormal

from numpyro_forecast.exceptions import MVNLayoutError
from numpyro_forecast.models import Horizon, predict
from numpyro_forecast.predictive import draw_posterior, forecast, predict_in_sample
from numpyro_forecast.surgery import (
    _mvn_time_params,
    prefix_condition,
    shift_loc,
    slice_time,
)
from tests.conftest import as_model, empty_covariates, get_trace


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
    """A ``(time, 1)`` predictor becomes the mean and keeps its observation axis (#160)."""
    loc = jnp.zeros(4)
    cov = jnp.eye(4)
    mvn = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)
    shift = jnp.arange(4.0)[:, None]
    shifted = shift_loc(mvn, shift)
    assert shifted.event_shape == (4, 1)
    assert jnp.allclose(shifted.mean, shift)


def test_mvn_shift_loc_without_obs_axis_keeps_the_time_event() -> None:
    """A ``(*batch, time)`` predictor keeps the bare ``(time,)`` event of the noise."""
    mvn = dist.MultivariateNormal(loc=jnp.zeros((2, 4)), covariance_matrix=jnp.eye(4))
    shift = jnp.arange(8.0).reshape(2, 4)
    shifted = shift_loc(mvn, shift)
    assert isinstance(shifted, dist.MultivariateNormal)
    assert shifted.batch_shape == (2,)
    assert jnp.allclose(shifted.loc, shift)


@pytest.mark.parametrize(
    ("noise_dim", "predictor_shape"),
    [
        pytest.param(6, (6, 6), id="obs-equals-time"),
        pytest.param(3, (8, 3), id="series-axis-noise"),
    ],
)
def test_mvn_shift_loc_rejects_a_predictor_with_a_wide_obs_axis(
    noise_dim: int, predictor_shape: tuple[int, int]
) -> None:
    """A ``(time, obs)`` predictor with ``obs > 1`` is a layout error, never a batch axis.

    MVN noise over time needs ``obs == 1``; reading the time axis of the
    predictor as a batch axis would correlate the noise across series instead.
    """
    mvn = dist.MultivariateNormal(jnp.zeros(noise_dim), covariance_matrix=jnp.eye(noise_dim))
    with pytest.raises(MVNLayoutError):
        shift_loc(mvn, jnp.zeros(predictor_shape))


def test_mvn_shift_loc_twice_shifts_again_and_keeps_the_obs_axis() -> None:
    """Shifting an already shifted MVN adds the second predictor to the mean."""
    mvn = dist.MultivariateNormal(loc=jnp.zeros(4), covariance_matrix=jnp.eye(4))
    first = shift_loc(mvn, jnp.ones((4, 1)))
    second = shift_loc(first, jnp.arange(8.0).reshape(2, 4, 1))
    assert second.batch_shape == (2,)
    assert second.event_shape == (4, 1)
    assert jnp.allclose(second.mean, 1.0 + jnp.arange(8.0).reshape(2, 4, 1))


def test_mvn_shift_loc_broadcasts_a_constant_predictor_over_time() -> None:
    """A ``(series, 1, 1)`` predictor shifts every time step of its series by one value."""
    cov = _ar_covariance(4, 0.5)
    mvn = dist.MultivariateNormal(jnp.zeros(4), covariance_matrix=cov)
    shifted = shift_loc(mvn, jnp.arange(3.0).reshape(3, 1, 1))
    assert shifted.batch_shape == (3,)
    assert shifted.event_shape == (4, 1)
    assert jnp.allclose(shifted.mean, jnp.broadcast_to(jnp.arange(3.0)[:, None, None], (3, 4, 1)))
    assert jnp.allclose(shifted.variance, jnp.broadcast_to(jnp.diag(cov)[:, None], (3, 4, 1)))


def test_mvn_slice_time_marginal_block() -> None:
    loc = jnp.arange(6.0)
    cov = _ar_covariance(6, 0.5)
    mvn = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)
    sliced = slice_time(mvn, slice(1, 4))
    assert isinstance(sliced, dist.MultivariateNormal)
    assert sliced.loc.shape == (3,)
    assert jnp.asarray(sliced.covariance_matrix).shape == (3, 3)


def test_gp_noise_end_to_end() -> None:
    """GP-style correlated noise fits through plain NumPyro SVI and forecasts.

    Regression for #160: the ``obs`` log density is the joint MVN density of the
    ``(t, 1)`` series, one value per draw (not ``t`` broadcast values), and the
    drivers return draws with the observation axis, ``(sample, time, 1)``.
    """

    def gp_model(covariates: Array, data: Array | None = None) -> None:
        h = Horizon.from_data(covariates, data)
        time = h.duration
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        rho = numpyro.sample("rho", dist.Beta(2.0, 2.0))
        cov = _ar_covariance(time, jnp.asarray(rho)) * sigma**2
        noise = dist.MultivariateNormal(loc=jnp.zeros(time), covariance_matrix=cov)
        level = numpyro.sample("level", dist.Normal(0.0, 1.0))
        predict(h, noise, jnp.asarray(level) + jnp.zeros((time, 1)))

    t, future, num_samples = 12, 4, 30
    data = random.normal(random.PRNGKey(0), (t, 1))
    cov = empty_covariates(t + future)
    key_fit, key_draw, key_forecast, key_in_sample = random.split(random.PRNGKey(1), 4)
    guide = AutoNormal(gp_model)
    svi = SVI(gp_model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    state = svi.run(key_fit, 150, cov[:t], data, progress_bar=False)
    post = draw_posterior(key_draw, guide, state.params, num_samples)
    assert set(post) >= {"sigma", "rho", "level"}
    assert jnp.all(jnp.isfinite(post["sigma"]))

    def joint_log_density(sigma: Array, rho: Array, level: Array) -> Array:
        cov_s = _ar_covariance(t, rho) * sigma**2
        mvn = dist.MultivariateNormal(level * jnp.ones(t), covariance_matrix=cov_s)
        return jnp.asarray(mvn.log_prob(data[:, 0]))

    draws = (jnp.asarray(post[name]) for name in ("sigma", "rho", "level"))
    expected = jax.vmap(joint_log_density)(*draws)
    obs_log_density = log_likelihood(gp_model, post, cov[:t], data)["obs"]
    assert obs_log_density.shape == (num_samples,)
    assert jnp.allclose(obs_log_density, expected, rtol=1e-4, atol=1e-3)
    assert forecast(key_forecast, gp_model, post, data, cov).shape == (num_samples, future, 1)
    in_sample = predict_in_sample(key_in_sample, gp_model, post, cov[:t])
    assert in_sample.shape == (num_samples, t, 1)


def _ar_mvn_body(h: Horizon, covariates: Array) -> None:
    """``predict`` with zero-mean AR(1)-correlated noise over time and a sampled ``rho``."""
    rho = jnp.asarray(numpyro.sample("rho", dist.Uniform(0.0, 0.9)))
    noise = dist.MultivariateNormal(
        jnp.zeros(h.duration), covariance_matrix=_ar_covariance(h.duration, rho)
    )
    predict(h, noise, jnp.zeros((h.duration, 1)))


def test_predict_mvn_forecast_sites_keep_the_obs_axis() -> None:
    """While forecasting, ``obs`` and ``obs_future`` score ``(time, 1)`` values jointly.

    Regression for #160: the MVN surgery dropped the observation axis, so ``obs``
    scored ``(t, 1)`` data as ``t`` broadcast values instead of the joint density
    and ``obs_future`` drew ``(future,)`` instead of ``(future, 1)``.
    """
    t, future, rho = 12, 4, 0.6
    data = 0.1 * random.normal(random.PRNGKey(0), (t, 1))
    truth = 0.1 * random.normal(random.PRNGKey(1), (future, 1))
    tr = get_trace(
        as_model(_ar_mvn_body),
        empty_covariates(t + future),
        data,
        substitutions={"rho": jnp.asarray(rho)},
    )

    c = np.asarray(_ar_covariance(t + future, rho), dtype=np.float64)
    gain = np.linalg.solve(c[:t, :t], c[:t, t:]).T  # Sigma_fp @ inv(Sigma_pp)
    in_sample = dist.MultivariateNormal(jnp.zeros(t), covariance_matrix=jnp.asarray(c[:t, :t]))
    conditional = dist.MultivariateNormal(
        jnp.asarray(gain @ np.asarray(data[:, 0], dtype=np.float64)),
        covariance_matrix=jnp.asarray(c[t:, t:] - gain @ c[:t, t:]),
    )

    obs_fn, future_fn = tr["obs"]["fn"], tr["obs_future"]["fn"]
    assert obs_fn.event_shape == (t, 1)
    assert obs_fn.log_prob(data).shape == ()
    assert jnp.allclose(obs_fn.log_prob(data), in_sample.log_prob(data[:, 0]), atol=1e-4)
    assert future_fn.event_shape == (future, 1)
    assert tr["obs_future"]["value"].shape == (future, 1)
    assert tr["forecast"]["value"].shape == (future, 1)
    assert jnp.allclose(future_fn.log_prob(truth), conditional.log_prob(truth[:, 0]), atol=1e-3)


@pytest.mark.parametrize("wrap", [False, True], ids=["batched", "to_event"])
def test_predict_mvn_panel_scores_each_series_jointly(wrap: bool) -> None:
    """A ``(series, time, 1)`` panel with MVN noise over time is scored per series.

    The noise is either shared across series and batched by the
    ``(series, time, 1)`` predictor, or batched per series and wrapped with
    ``.to_event(1)``. Both raised a broadcasting error before #160.
    """
    series, t, future = 3, 12, 4
    data = 0.1 * random.normal(random.PRNGKey(0), (series, t, 1))
    full_cov = _ar_covariance(t + future, 0.6)

    def body(h: Horizon, covariates: Array) -> None:
        loc = jnp.zeros((series, h.duration)) if wrap else jnp.zeros(h.duration)
        cov = full_cov[: h.duration, : h.duration]
        noise: dist.Distribution = dist.MultivariateNormal(loc, covariance_matrix=cov)
        if wrap:
            noise = noise.to_event(1)
        predict(h, noise, jnp.zeros((series, h.duration, 1)))

    tr = get_trace(as_model(body), empty_covariates(t + future), data)

    in_sample = dist.MultivariateNormal(jnp.zeros(t), covariance_matrix=full_cov[:t, :t])
    per_series = jnp.asarray(in_sample.log_prob(data[..., 0]))
    expected = per_series.sum() if wrap else per_series
    obs_log_density = jnp.asarray(tr["obs"]["fn"].log_prob(data))
    assert obs_log_density.shape == expected.shape
    assert jnp.allclose(obs_log_density, expected, atol=1e-4)
    assert tr["obs_future"]["value"].shape == (series, future, 1)
    c = np.asarray(full_cov, dtype=np.float64)
    gain = np.linalg.solve(c[:t, :t], c[:t, t:]).T  # Sigma_fp @ inv(Sigma_pp)
    expected_mean = np.asarray(data[..., 0], dtype=np.float64) @ gain.T  # (series, future)
    assert jnp.allclose(tr["obs_future"]["fn"].mean, expected_mean[..., None], atol=1e-4)


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
    assert shifted.batch_shape == (3,)
    assert shifted.event_shape == (time, 1)
    assert jnp.allclose(shifted.mean, loc[..., None] + 1.0)
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


@pytest.mark.parametrize("depth", [1, 2], ids=["to_event", "nested_to_event"])
def test_prefix_condition_independent_mvn_is_the_gaussian_conditional(depth: int) -> None:
    """An ``Independent``-wrapped MVN over time conditions on the observed prefix.

    Regression for #171: the ``Independent`` dispatch returned the horizon
    marginal, so a ``.to_event(1)`` panel of time-correlated series ignored the
    prefix and the forecast fell back to the prior mean. ``depth = 2`` wraps the
    MVN twice (numpyro does not merge nested ``Independent`` wrappers).
    """
    time, t = 6, 4
    lead = (1,) * (depth - 1)  # one extra batch axis per extra wrapper
    cov = _ar_covariance(time, 0.9)
    loc = jnp.stack([jnp.zeros(time), jnp.linspace(-1.0, 1.0, time)])  # two series
    data = jnp.stack([jnp.ones((t, 1)), jnp.full((t, 1), -0.5)])  # (2, t, 1)
    noise: dist.Distribution = dist.MultivariateNormal(
        loc=loc.reshape(*lead, 2, time), covariance_matrix=cov
    )
    for _ in range(depth):
        noise = noise.to_event(1)

    cond = prefix_condition(noise, data.reshape(*lead, 2, t, 1))

    c = np.asarray(cov, dtype=np.float64)
    mu = np.asarray(loc, dtype=np.float64)
    x = np.asarray(data[..., 0], dtype=np.float64)
    gain = np.linalg.solve(c[:t, :t], c[:t, t:]).T  # Sigma_fp @ inv(Sigma_pp)
    expected_mean = mu[:, t:] + (x - mu[:, :t]) @ gain.T
    expected_var = np.broadcast_to(np.diag(c[t:, t:] - gain @ c[:t, t:]), (2, time - t))
    assert isinstance(cond, dist.Independent)
    assert cond.event_shape == (*lead, 2, time - t)
    assert jnp.allclose(cond.mean, expected_mean.reshape(*lead, 2, time - t), atol=1e-4)
    assert jnp.allclose(cond.variance, expected_var.reshape(*lead, 2, time - t), atol=1e-4)
