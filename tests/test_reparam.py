"""Behavioral tests for `~~numpyro_forecast.reparam.time_reparam()`.

Grouped by contract: which sites are targeted and what shapes they get, that the
change of coordinates is exact (transform identity, unit Jacobian), that the
drivers consume the auxiliary site, and that the reparameterized model runs end
to end under SVI and NUTS.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random
from numpyro.distributions.transforms import DiscreteCosineTransform, HaarTransform, Transform
from numpyro.handlers import scope
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import log_density
from numpyro.optim import Adam

from numpyro_forecast import draw_posterior, forecast, predict_in_sample, time_reparam
from numpyro_forecast.models import TIME_PLATE, Horizon, innovations, markov_series, predict
from numpyro_forecast.reparam import TimeTransform
from numpyro_forecast.typing import Array, ForecastModel
from tests.conftest import as_model, empty_covariates, get_trace, plate_frames, rw_model
from tests.example_models import make_hierarchical_model

T_OBS = 6
FUTURE = 2
TRANSFORMS: dict[TimeTransform, Transform] = {
    "haar": HaarTransform(dim=-2),
    "dct": DiscreteCosineTransform(dim=-2),
}


def _split(model: ForecastModel) -> tuple[ForecastModel, Array, Array]:
    """Return ``model`` with matching zero data ``(T_OBS, 1)`` and covariates ``(T_OBS + FUTURE, 0)``."""
    return model, jnp.zeros((T_OBS, 1)), empty_covariates(T_OBS + FUTURE)


# --------------------------------------------------------------------------- targeting


@pytest.mark.parametrize("transform", ["haar", "dct"])
def test_time_site_becomes_deterministic_with_event_shaped_auxiliary(
    transform: TimeTransform,
) -> None:
    """The in-sample site is replaced by an aux sample whose event owns the time axis."""
    model, data, covariates = _split(time_reparam(rw_model, transform))
    tr = get_trace(model, covariates, data)
    aux = tr[f"drift_{transform}"]

    assert aux["type"] == "sample"
    assert aux["fn"].batch_shape == ()
    assert aux["fn"].event_shape == (T_OBS, 1)
    assert plate_frames(aux) == []
    assert tr["drift"]["type"] == "deterministic"
    assert tr["drift"]["value"].shape == (T_OBS, 1)


def test_future_and_observed_sites_are_untouched() -> None:
    """The prior-drawn horizon suffix and the likelihood keep their sites and plates."""
    model, data, covariates = _split(time_reparam(rw_model, "haar"))
    tr = get_trace(model, covariates, data)

    assert tr["drift_future"]["type"] == "sample"
    assert plate_frames(tr["drift_future"]) == [("time_future", -2, FUTURE)]
    assert "drift_future_haar" not in tr
    assert tr["obs"]["is_observed"]
    assert "obs_haar" not in tr


@pytest.mark.parametrize(
    ("plate_dim", "batch_shape", "event_shape"),
    [(-1, (), (T_OBS, 3)), (-3, (3,), (T_OBS, 1))],
    ids=["right-of-time", "left-of-time"],
)
def test_plate_axes_from_time_rightward_join_the_auxiliary_event(
    plate_dim: int, batch_shape: tuple[int, ...], event_shape: tuple[int, ...]
) -> None:
    """Time and every plate axis to its right become event axes; axes to the left stay batch.

    Same split as Pyro's ``experimental_allow_batch`` ("the targeted batch dimension
    and all batch dimensions to the right will be converted to event dimensions").
    Either way no plate frame survives on the auxiliary site, so no plate can
    re-expand it.
    """
    n = 3
    shape = batch_shape + event_shape

    def body(h: Horizon, covariates: Array) -> None:
        with numpyro.plate("group", n, dim=plate_dim):
            drift = innovations(h, "drift", lambda: dist.Normal(0.0, 1.0))
        predict(h, dist.Normal(0.0, 1.0), jnp.cumsum(drift, axis=-2))

    model = time_reparam(as_model(body), "dct")
    data = jnp.zeros(shape)
    covariates = empty_covariates(T_OBS + FUTURE)
    tr = get_trace(model, covariates, data)

    assert tr["drift_dct"]["fn"].batch_shape == batch_shape
    assert tr["drift_dct"]["fn"].event_shape == event_shape
    assert plate_frames(tr["drift_dct"]) == []
    assert tr["drift"]["value"].shape == shape


def test_composes_with_per_block_loc_scale_reparam() -> None:
    """After ``innovations(reparam=LocScaleReparam)`` the decentered site is what gets targeted."""
    model = time_reparam(make_hierarchical_model(period=4), "haar")
    n_origin, n_destin = 2, 3
    data = jnp.zeros((n_origin, T_OBS, n_destin))
    covariates = jnp.zeros((n_origin, T_OBS + FUTURE, n_destin))
    tr = get_trace(model, covariates, data)

    assert tr["drift_decentered_haar"]["type"] == "sample"
    assert tr["drift_decentered_haar"]["fn"].event_shape == (T_OBS, n_destin)
    assert tr["drift_decentered"]["type"] == "deterministic"
    assert tr["drift"]["type"] == "deterministic"
    assert tr["drift"]["value"].shape == (T_OBS, n_destin)


def test_discrete_site_under_time_plate_is_skipped() -> None:
    """A discrete latent under ``time`` has no unconstrained image, so it must stay as is."""

    def body(h: Horizon, covariates: Array) -> None:
        with numpyro.plate("time", h.t_obs, dim=-2):
            numpyro.sample("regime", dist.Bernoulli(0.3), infer={"enumerate": "parallel"})
        drift = innovations(h, "drift", lambda: dist.Normal(0.0, 1.0))
        predict(h, dist.Normal(0.0, 1.0), jnp.cumsum(drift, axis=-2))

    model, data, covariates = _split(time_reparam(as_model(body), "haar"))
    tr = get_trace(model, covariates, data)

    assert tr["regime"]["type"] == "sample"
    assert "regime_haar" not in tr
    assert "drift_haar" in tr


def test_markov_series_sites_are_outside_the_time_plate() -> None:
    """Scan sites carry no ``time`` plate, so the reparam is a no-op on them (documented boundary)."""

    def body(h: Horizon, covariates: Array) -> None:
        def transition(
            carry: Array, _: object
        ) -> tuple[dist.Distribution, Callable[[Array], Array]]:
            return dist.Normal(carry, 1.0).to_event(1), lambda z: z

        level = markov_series(h, "level", jnp.zeros((1,)), transition)
        predict(h, dist.Normal(0.0, 1.0), level)

    model, data, covariates = _split(time_reparam(as_model(body), "dct"))
    tr = get_trace(model, covariates, data)

    assert tr["level"]["type"] == "sample"
    assert not any(name.endswith("_dct") for name in tr)


def test_scope_applied_outside_prefixes_the_auxiliary_once() -> None:
    """``scope(time_reparam(model))`` yields ``a/drift_haar`` and no plate re-expansion."""
    model, data, covariates = _split(scope(time_reparam(rw_model, "haar"), prefix="a"))
    tr = get_trace(model, covariates, data)

    assert tr["a/drift_haar"]["fn"].event_shape == (T_OBS, 1)
    assert plate_frames(tr["a/drift_haar"]) == []
    assert tr["a/drift"]["type"] == "deterministic"
    assert "a/a/drift_haar" not in tr


def test_targets_the_plate_innovations_opens() -> None:
    """The shared ``TIME_PLATE`` name is the one `innovations` opens; a rename cannot silently no-op."""
    tr = get_trace(rw_model, empty_covariates(T_OBS), jnp.zeros((T_OBS, 1)))
    assert plate_frames(tr["drift"]) == [(TIME_PLATE, -2, T_OBS)]

    def body(h: Horizon, covariates: Array) -> None:
        with numpyro.plate("steps", h.t_obs, dim=-2):
            drift = numpyro.sample("drift", dist.Normal(0.0, 1.0))
        predict(h, dist.Normal(0.0, 1.0), jnp.cumsum(drift, axis=-2))

    model = time_reparam(as_model(body), "haar")
    tr = get_trace(model, empty_covariates(T_OBS), jnp.zeros((T_OBS, 1)))
    assert tr["drift"]["type"] == "sample"
    assert "drift_haar" not in tr


def test_nesting_is_rejected() -> None:
    """Wrapping a wrapped model raises instead of silently applying only the inner transform."""
    wrapped = time_reparam(rw_model, "haar")
    with pytest.raises(ValueError, match="cannot be nested"):
        time_reparam(wrapped, "dct")


# --------------------------------------------------------------------------- exactness


@pytest.mark.parametrize("transform", ["haar", "dct"])
def test_auxiliary_is_the_transform_of_the_site_along_time(transform: TimeTransform) -> None:
    """``aux == T(drift)`` with ``T`` acting on axis ``-2``, so the time axis is the one rotated."""
    model, data, covariates = _split(time_reparam(rw_model, transform))
    tr = get_trace(model, covariates, data)

    expected = TRANSFORMS[transform](tr["drift"]["value"])
    assert jnp.allclose(tr[f"drift_{transform}"]["value"], expected, atol=1e-6)


@pytest.mark.parametrize("transform", ["haar", "dct"])
def test_log_density_is_unchanged(transform: TimeTransform) -> None:
    """Unit Jacobian: the wrapped model evaluated at ``aux`` equals the original at ``drift``."""
    wrapped, data, covariates = _split(time_reparam(rw_model, transform))
    train = covariates[:T_OBS]
    tr = get_trace(wrapped, train, data)
    scalars = {"drift_scale": tr["drift_scale"]["value"], "sigma": tr["sigma"]["value"]}

    wrapped_ld, _ = log_density(
        wrapped,
        (train, data),
        {},
        {**scalars, f"drift_{transform}": tr[f"drift_{transform}"]["value"]},
    )
    plain_ld, _ = log_density(
        rw_model, (train, data), {}, {**scalars, "drift": tr["drift"]["value"]}
    )
    assert jnp.allclose(wrapped_ld, plain_ld, rtol=1e-6)


# --------------------------------------------------------------------------- drivers


def test_predictive_reads_the_auxiliary_and_ignores_a_stale_deterministic() -> None:
    """``predict_in_sample`` rebuilds ``drift`` from ``drift_haar``; a stale ``drift`` entry is inert."""
    model, _, covariates = _split(time_reparam(rw_model, "haar"))
    train = covariates[:T_OBS]
    num_samples = 3
    aux = random.normal(random.PRNGKey(1), (num_samples, T_OBS, 1))
    posterior = {
        "drift_scale": jnp.ones((num_samples,)),
        "sigma": jnp.full((num_samples,), 1e-6),
        "drift_haar": aux,
    }

    draws = predict_in_sample(random.PRNGKey(2), model, posterior, train)
    expected = jnp.cumsum(HaarTransform(dim=-2).inv(aux), axis=-2)
    assert jnp.allclose(draws, expected, atol=1e-4)

    stale = {**posterior, "drift": jnp.zeros((num_samples, T_OBS, 1))}
    assert jnp.allclose(predict_in_sample(random.PRNGKey(2), model, stale, train), draws)


def test_wrapped_model_is_a_stable_jit_static_argument(
    count_compilations: Callable[[], AbstractContextManager[SimpleNamespace]],
) -> None:
    """One wrapped object compiles ``forecast`` once; rewrapping is a new static arg."""
    model, data, covariates = _split(time_reparam(rw_model, "dct"))
    num_samples = 2
    posterior = {
        "drift_scale": jnp.ones((num_samples,)),
        "sigma": jnp.ones((num_samples,)),
        "drift_dct": jnp.zeros((num_samples, T_OBS, 1)),
    }
    jax.block_until_ready(forecast(random.PRNGKey(0), model, posterior, data, covariates))

    with count_compilations() as tally:
        jax.block_until_ready(forecast(random.PRNGKey(0), model, posterior, data, covariates))
    assert tally.count == 0

    with count_compilations() as tally:
        rewrapped = time_reparam(rw_model, "dct")
        jax.block_until_ready(forecast(random.PRNGKey(0), rewrapped, posterior, data, covariates))
    assert tally.count >= 1


@pytest.mark.parametrize("transform", ["haar", "dct"])
def test_svi_fit_draws_and_forecasts(transform: TimeTransform) -> None:
    """AutoNormal fits the aux site; ``draw_posterior`` and ``forecast`` need no extra plumbing."""
    model, data, covariates = _split(time_reparam(rw_model, transform))
    guide = AutoNormal(model)
    svi = SVI(model, guide, Adam(0.05), Trace_ELBO())
    result = svi.run(random.PRNGKey(0), 50, covariates[:T_OBS], data, progress_bar=False)
    assert jnp.isfinite(result.losses[-1])

    posterior = draw_posterior(random.PRNGKey(1), guide, result.params, 4)
    assert posterior[f"drift_{transform}"].shape == (4, T_OBS, 1)
    assert posterior["drift"].shape == (4, T_OBS, 1)

    draws = forecast(random.PRNGKey(2), model, posterior, data, covariates)
    assert draws.shape == (4, FUTURE, 1)
    assert bool(jnp.isfinite(draws).all())


def test_nuts_samples_the_auxiliary_and_forecasts() -> None:
    """NUTS runs on the reparameterized potential and its samples feed ``forecast`` directly."""
    model, data, covariates = _split(time_reparam(rw_model, "haar"))
    mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=5, progress_bar=False)
    mcmc.run(random.PRNGKey(0), covariates[:T_OBS], data)
    samples = mcmc.get_samples()

    assert samples["drift_haar"].shape == (5, T_OBS, 1)
    assert samples["drift"].shape == (5, T_OBS, 1)
    draws = forecast(random.PRNGKey(1), model, samples, data, covariates)
    assert draws.shape == (5, FUTURE, 1)
    assert bool(jnp.isfinite(draws).all())
