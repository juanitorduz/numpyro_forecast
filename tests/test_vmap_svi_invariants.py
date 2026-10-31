"""Permanent invariant suite ported from ``spikes/spike_vmap_svi.py`` (S-V).

Each test pins a behavior that ``backtest_vectorized`` (roadmap §4.3/§4.4)
relies on; the spike script is kept verbatim for provenance (recorded on jax
0.10.2 / numpyro 0.21.0). The contamination counterfactual is the ⚙ canary
for K11/I6: it proves that a *traced* first ``svi.init`` poisons the AutoGuide
instance it runs on, which is why the vectorized path sets up a copy of
the guide inside the vmapped fit (one per window, on that window's data, #185)
and never the caller's instance. Any future vmapped-inference feature must
cite this test.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, lax, random
from jax.errors import UnexpectedTracerError
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal, AutoNormal

from numpyro_forecast.evaluate import _fresh_copy

T_TRAIN, T_TEST, N_WIN, STEPS, S = 30, 5, 4, 200, 200
DUR = T_TRAIN + (N_WIN - 1) + T_TEST


def _model(covariates: Array, data: Array | None = None) -> None:
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    mu = beta * covariates[:, 0]
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=None if data is None else data[:, 0])


@pytest.fixture
def windows() -> tuple[Array, Array, Array, Array]:
    """Stacked rolling windows (same shapes, different data) per §4.4."""
    rng = random.PRNGKey(0)
    full_cov = jnp.linspace(0, 1, DUR)[:, None]
    full_data = (2.0 * full_cov[:, 0] + 0.1 * random.normal(rng, (DUR,)))[:, None]
    starts = jnp.arange(N_WIN)

    def slice_windows(t0: Array) -> tuple[Array, Array, Array]:
        return (
            lax.dynamic_slice_in_dim(full_data, t0, T_TRAIN, axis=0),
            lax.dynamic_slice_in_dim(full_cov, t0, T_TRAIN, axis=0),
            lax.dynamic_slice_in_dim(full_cov, t0, T_TRAIN + T_TEST, axis=0),
        )

    train_d, train_c, hor_c = jax.vmap(slice_windows)(starts)
    return starts, train_d, train_c, hor_c


def _fit_probe(
    guide_cls: type, starts: Array, train_d: Array, train_c: Array
) -> tuple[AutoNormal, dict[str, Array]]:
    guide = guide_cls(_model)

    # K11: the first init of an AutoGuide caches its setup on the instance, so
    # a traced first init leaves tracers on it. Each window therefore sets up
    # a copy of the guide; the caller's instance is never set up.
    def fit_one(key: Array, d: Array, c: Array) -> tuple[dict[str, Array], Array]:
        window_guide = _fresh_copy(guide)
        svi = SVI(_model, window_guide, numpyro.optim.Adam(0.05), Trace_ELBO())
        state = svi.init(key, c, d)

        def step(s: object, _: None) -> tuple[object, Array]:
            s, loss = svi.update(s, c, d)
            return s, loss

        state, losses = lax.scan(step, state, length=STEPS)
        return svi.get_params(state), losses

    keys = jax.vmap(lambda i: random.fold_in(random.PRNGKey(1), i))(starts)
    params, _ = jax.jit(jax.vmap(fit_one))(keys, train_d, train_c)
    return guide, params


def _set_up_eagerly(guide: AutoNormal, train_c: Array, train_d: Array) -> None:
    """Set ``guide`` up on window 0; ``_model`` has fixed supports, so any window would do."""
    SVI(_model, guide, numpyro.optim.Adam(0.05), Trace_ELBO()).init(
        random.PRNGKey(0xFFFF), train_c[0], train_d[0]
    )


@pytest.mark.parametrize("guide_cls", [AutoNormal, AutoMultivariateNormal])
def test_per_window_guide_copies_fit_under_vmap(
    windows: tuple[Array, Array, Array, Array], guide_cls: type
) -> None:
    """S-V/F1: the traced first init of a per-window guide copy fits and batches params.

    Guide parameters come out with a leading window axis, confirming windows
    are a batch dimension rather than an unroll, and the template guide stays
    unset, so a later eager init on it does not raise ``UnexpectedTracerError``.
    """
    starts, train_d, train_c, _ = windows
    guide, params = _fit_probe(guide_cls, starts, train_d, train_c)
    for value in params.values():
        assert value.shape[0] == N_WIN
    assert guide.prototype_trace is None
    _set_up_eagerly(guide, train_c, train_d)


def test_contamination_counterfactual_raises_on_the_traced_instance(
    windows: tuple[Array, Array, Array, Array],
) -> None:
    """S-V/K11 (I6): a *traced* first ``svi.init`` poisons the AutoGuide.

    Letting the first init of the instance itself happen under vmap leaves
    leaked tracers on it, so a subsequent EAGER init raises
    ``UnexpectedTracerError``. This is the counterfactual that justifies the
    per-window copy of the guide in the vectorized backtest path.
    """
    starts, train_d, train_c, _ = windows
    guide = AutoNormal(_model)
    svi = SVI(_model, guide, numpyro.optim.Adam(0.05), Trace_ELBO())

    def fit_one(key: Array, d: Array, c: Array) -> object:
        return svi.init(key, c, d)

    keys = jax.vmap(lambda i: random.fold_in(random.PRNGKey(1), i))(starts)
    jax.vmap(fit_one)(keys, train_d, train_c)  # first init is TRACED — contaminates

    with pytest.raises(UnexpectedTracerError):
        svi.init(random.PRNGKey(9), train_c[0], train_d[0])


def test_sample_posterior_pure_under_vmap(
    windows: tuple[Array, Array, Array, Array],
) -> None:
    """S-V/F2: ``AutoNormal.sample_posterior`` is pure in its params.

    vmapping over the batched params with a shared key equals the per-window
    Python loop bitwise-close, so the vectorized draw stage is well defined.
    """
    starts, train_d, train_c, _ = windows
    guide, params = _fit_probe(AutoNormal, starts, train_d, train_c)
    _set_up_eagerly(guide, train_c, train_d)
    kpost = random.PRNGKey(2)
    post_vmap = jax.vmap(lambda p: guide.sample_posterior(kpost, p, sample_shape=(S,)))(params)
    post_loop = [
        guide.sample_posterior(
            kpost,
            jax.tree_util.tree_map(lambda x, i=i: x[i], params),
            sample_shape=(S,),
        )
        for i in range(N_WIN)
    ]
    diffs = [
        float(jnp.abs(post_vmap["beta"][i] - post_loop[i]["beta"]).max()) for i in range(N_WIN)
    ]
    assert max(diffs) < 1e-5


def test_nested_vmap_predictive_forecast(
    windows: tuple[Array, Array, Array, Array],
) -> None:
    """S-V/F3: ``Predictive`` nests inside the window vmap.

    A vmap over both the posterior samples and per-window args produces a
    ``(N_WIN, S, horizon)`` forecast stack, the shape the vectorized backtest
    forecast stage returns.
    """
    starts, train_d, train_c, hor_c = windows
    guide, params = _fit_probe(AutoNormal, starts, train_d, train_c)
    _set_up_eagerly(guide, train_c, train_d)
    kpost = random.PRNGKey(2)
    post_vmap = jax.vmap(lambda p: guide.sample_posterior(kpost, p, sample_shape=(S,)))(params)
    keys = jax.vmap(lambda i: random.fold_in(random.PRNGKey(1), i))(starts)

    def forecast_one(key: Array, post_w: dict[str, Array], hc: Array) -> Array:
        pred = Predictive(
            _model,
            posterior_samples={"beta": post_w["beta"], "sigma": post_w["sigma"]},
        )
        return pred(key, hc)["obs"]

    fc = jax.jit(jax.vmap(forecast_one))(keys, post_vmap, hor_c)
    assert fc.shape == (N_WIN, S, T_TRAIN + T_TEST)
