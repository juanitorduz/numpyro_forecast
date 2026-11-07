"""Tests for drawing posterior samples from a fitted guide (``numpyro_forecast.predictive``)."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from conftest import assert_host_resident, empty_covariates, rw_model, svi_guide_params
from jax import random
from jaxtyping import TypeCheckError
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoBNAFNormal, AutoDelta, AutoGuide, AutoIAFNormal, AutoNormal

from numpyro_forecast.convert import to_datatree
from numpyro_forecast.exceptions import DeviceMemoryError
from numpyro_forecast.predictive import draw_posterior, forecast
from numpyro_forecast.typing import Array


def test_draw_posterior_svi_leading_sample_axis() -> None:
    guide, params = svi_guide_params(t=30)
    post = draw_posterior(random.PRNGKey(2), guide, params, 8)
    assert post["drift"].shape == (8, 30, 1)


def test_draw_posterior_rejects_non_positive() -> None:
    guide, params = svi_guide_params(t=30, num_steps=20)
    with pytest.raises(ValueError, match="num_samples must be positive"):
        draw_posterior(random.PRNGKey(2), guide, params, 0)


def test_draw_posterior_autodelta_tiled_sample_axis() -> None:
    # AutoDelta is a MAP point estimate: it carries no posterior spread of its
    # own, so it is drawn once and tiled to the leading sample axis (dispatch by
    # guide type, never shape inspection).
    data = jnp.zeros((20, 1))
    covariates = empty_covariates(20)
    guide = AutoDelta(rw_model)
    svi = SVI(rw_model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    result = svi.run(random.PRNGKey(0), 10, covariates, data, progress_bar=False)
    post = draw_posterior(random.PRNGKey(1), guide, result.params, 7)
    assert post["sigma"].shape[0] == 7
    assert jnp.allclose(post["sigma"][0], post["sigma"][1])


# --- Normalizing-flow guides: params are pytrees, not flat arrays (#282) --------

_FLOW_T = 12
_FLOW_GUIDES = [
    pytest.param(AutoIAFNormal, id="AutoIAFNormal"),
    pytest.param(AutoBNAFNormal, id="AutoBNAFNormal"),
]


def _flow_fit(guide_cls: type[AutoGuide]) -> tuple[AutoGuide, dict, Array]:
    """Fit a flow guide on ``rw_model`` for a few steps; its params hold lists of arrays."""
    data = jnp.cumsum(0.1 * random.normal(random.PRNGKey(0), (_FLOW_T, 1)), axis=-2)
    guide = guide_cls(rw_model)
    svi = SVI(rw_model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    result = svi.run(random.PRNGKey(1), 5, empty_covariates(_FLOW_T), data, progress_bar=False)
    assert any(isinstance(v, list) for v in result.params.values())
    return guide, result.params, data


@pytest.mark.parametrize("guide_cls", _FLOW_GUIDES)
@pytest.mark.parametrize(
    ("device", "batch_size"), [(None, None), ("host", None), (None, 4), ("host", 4)]
)
def test_draw_posterior_flow_guide_params(
    guide_cls: type[AutoGuide], device: str | None, batch_size: int | None
) -> None:
    guide, params, _ = _flow_fit(guide_cls)
    post = draw_posterior(
        random.PRNGKey(2), guide, params, 10, batch_size=batch_size, device=device
    )
    assert {"drift", "drift_scale", "sigma"} <= set(post)
    assert post["drift"].shape == (10, _FLOW_T, 1)
    assert post["sigma"].shape == (10,)
    assert all(bool(np.all(np.isfinite(np.asarray(leaf)))) for leaf in post.values())
    if device == "host":
        assert_host_resident(post)


@pytest.mark.parametrize("guide_cls", _FLOW_GUIDES)
def test_flow_guide_draws_feed_forecast_and_datatree(guide_cls: type[AutoGuide]) -> None:
    guide, params, data = _flow_fit(guide_cls)
    post = draw_posterior(random.PRNGKey(2), guide, params, 10, batch_size=4, device="host")
    covariates = empty_covariates(_FLOW_T + 3)
    fc = forecast(random.PRNGKey(3), rw_model, post, data, covariates)
    assert fc.shape == (10, 3, 1)
    tree = to_datatree(random.PRNGKey(4), rw_model, post, data, covariates)
    assert tree["posterior"].sizes["draw"] == 10
    assert "drift" in tree["posterior"].dataset


@pytest.mark.parametrize("bad", ["not an array", ["not an array"]], ids=["str", "list-of-str"])
def test_draw_posterior_rejects_non_array_param_leaf(bad: object) -> None:
    # The pytree hint still type-checks the leaves: a string is not a parameter.
    # beartype checks one randomly chosen item of a dict, so the dict holds one key.
    guide, _ = svi_guide_params(t=30, num_steps=5)
    with pytest.raises(TypeCheckError):
        draw_posterior(random.PRNGKey(2), guide, {"bad": bad}, 4)


# --- Chunked drawing with host offload (GPU OOM in draw_posterior) --------------


def _trees_equal(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)


def test_draw_posterior_chunked_shape_finite_and_truncated() -> None:
    # 10 draws in chunks of 4: the last chunk overdraws and is cut back to 10.
    guide, params = svi_guide_params(t=30)
    post = draw_posterior(random.PRNGKey(2), guide, params, 10, batch_size=4)
    assert post["drift"].shape == (10, 30, 1)
    assert bool(jnp.all(jnp.isfinite(post["drift"])))


def test_draw_posterior_batch_ge_samples_bitwise_matches_default() -> None:
    # At or above the draw count the single-shot path runs: same key, same draws.
    guide, params = svi_guide_params(t=30)
    plain = draw_posterior(random.PRNGKey(2), guide, params, 8)
    batched = draw_posterior(random.PRNGKey(2), guide, params, 8, batch_size=8)
    oversized = draw_posterior(random.PRNGKey(2), guide, params, 8, batch_size=64)
    assert _trees_equal(plain, batched)
    assert _trees_equal(plain, oversized)


def test_draw_posterior_chunked_deterministic_given_key_and_batch() -> None:
    guide, params = svi_guide_params(t=30)
    first = draw_posterior(random.PRNGKey(2), guide, params, 10, batch_size=4)
    second = draw_posterior(random.PRNGKey(2), guide, params, 10, batch_size=4)
    assert _trees_equal(first, second)


def test_draw_posterior_host_is_host_resident_bitwise_match() -> None:
    # device is a placement knob, never a draws knob.
    guide, params = svi_guide_params(t=30)
    plain = draw_posterior(random.PRNGKey(2), guide, params, 10, batch_size=4)
    hosted = draw_posterior(random.PRNGKey(2), guide, params, 10, batch_size=4, device="host")
    assert_host_resident(hosted)
    assert _trees_equal(plain, hosted)


def test_draw_posterior_chunked_calls_guide_per_fixed_size_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory contract: the guide is sampled ceil(n/b) times with exactly b draws each.

    Fixed-size chunks keep the guide sampling at a single compiled shape, and
    per-chunk transfers (tested via the host path elsewhere) bound accelerator
    memory by one chunk of the posterior instead of all of it.
    """
    import numpyro_forecast.predictive as predictive_mod

    calls: list[tuple[int, Array]] = []
    real_jitted = predictive_mod._jitted_sample_posterior

    def spy_jitted(guide):  # type: ignore[no-untyped-def]
        real_sample = real_jitted(guide)

        def wrapped(key: Array, params: dict, *, sample_shape: tuple[int, ...]):  # type: ignore[no-untyped-def]
            calls.append((sample_shape[0], key))
            return real_sample(key, params, sample_shape=sample_shape)

        return wrapped

    monkeypatch.setattr(predictive_mod, "_jitted_sample_posterior", spy_jitted)
    guide, params = svi_guide_params(t=30)
    draw_posterior(random.PRNGKey(2), guide, params, 10, batch_size=4)
    assert [n for n, _ in calls] == [4, 4, 4]  # ceil(10 / 4) chunks, fixed size
    raw = [tuple(int(x) for x in jnp.ravel(random.key_data(k))) for _, k in calls]
    assert len(set(raw)) == len(raw)  # one distinct subkey per chunk


def test_draw_posterior_chunked_single_compile(count_compilations) -> None:
    """The jitted guide sampling compiles once per call and is reused for every chunk.

    Every chunk shares the fixed ``batch_size`` shape, so a call builds one
    jitted ``sample_posterior`` and compiles it once. Nothing is cached across
    calls (issue #167), so after a warm-up (which compiles the eager chunking
    helpers) each call costs exactly one compilation: below 3 and 2 chunks.
    """
    guide, params = svi_guide_params(t=30)

    def _sweep() -> list[int]:
        counts = []
        for n in (10, 7):
            with count_compilations() as tally:
                jax.block_until_ready(
                    draw_posterior(random.PRNGKey(n), guide, params, n, batch_size=4)
                )
            counts.append(tally.count)
        return counts

    _sweep()  # warm-up
    assert _sweep() == [1, 1]


@pytest.mark.parametrize("batch_size", [None, 3])
def test_draw_posterior_reads_mutable_model_state_at_call_time(batch_size: int | None) -> None:
    """Issue #167: a deterministic site recomputed by the guide is not frozen.

    ``AutoNormal.sample_posterior`` re-runs the model to recompute deterministic
    sites, so a program cached on the guide would replay the first trace of a
    model that reads mutable state.
    """
    settings = {"offset": 0.0}

    def model(covariates: Array, data: Array | None = None) -> None:
        level = jnp.asarray(numpyro.sample("level", dist.Normal(0.0, 1.0)))
        numpyro.deterministic("shifted", level + settings["offset"])
        numpyro.sample("obs", dist.Normal(level, 1.0), obs=data)

    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.05), Trace_ELBO())
    result = svi.run(
        random.PRNGKey(0), 10, empty_covariates(5), jnp.zeros((5, 1)), progress_bar=False
    )

    def shift() -> np.ndarray:
        post = draw_posterior(random.PRNGKey(1), guide, result.params, 4, batch_size=batch_size)
        return np.asarray(post["shifted"]) - np.asarray(post["level"])

    assert np.allclose(shift(), 0.0, atol=1e-4)
    settings["offset"] = 100.0
    assert np.allclose(shift(), 100.0, atol=1e-3)


def test_draw_posterior_rejects_non_positive_batch_size() -> None:
    guide, params = svi_guide_params(t=30, num_steps=10)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        draw_posterior(random.PRNGKey(0), guide, params, 5, batch_size=0)


# --- Self-diagnosing device OOM errors -------------------------------------------


def _raise_oom(guide):  # type: ignore[no-untyped-def]
    def _raise(key: Array, params: dict, *, sample_shape: tuple[int, ...]):  # type: ignore[no-untyped-def]
        msg = "RESOURCE_EXHAUSTED: Out of memory while trying to allocate 3800000000 bytes."
        raise RuntimeError(msg)

    return _raise


def test_draw_posterior_chunked_oom_reports_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """A device OOM is re-raised with the budget and the batch-size lever."""
    import numpyro_forecast.predictive as predictive_mod

    monkeypatch.setattr(predictive_mod, "_jitted_sample_posterior", _raise_oom)
    guide, params = svi_guide_params(t=30)
    with pytest.raises(
        DeviceMemoryError, match=r"posterior drawing.*lower batch_size \(currently 4\)"
    ) as excinfo:
        draw_posterior(random.PRNGKey(2), guide, params, 10, batch_size=4)
    assert "RESOURCE_EXHAUSTED" in str(excinfo.value.__cause__)


def test_draw_posterior_unchunked_oom_says_set_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpyro_forecast.predictive as predictive_mod

    monkeypatch.setattr(predictive_mod, "_jitted_sample_posterior", _raise_oom)
    guide, params = svi_guide_params(t=30)
    with pytest.raises(DeviceMemoryError, match="set batch_size to sample in chunks"):
        draw_posterior(random.PRNGKey(2), guide, params, 10)


def test_draw_posterior_non_oom_errors_propagate_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpyro_forecast.predictive as predictive_mod

    def raise_other(guide):  # type: ignore[no-untyped-def]
        def _raise(key: Array, params: dict, *, sample_shape: tuple[int, ...]):  # type: ignore[no-untyped-def]
            msg = "boom"
            raise ValueError(msg)

        return _raise

    monkeypatch.setattr(predictive_mod, "_jitted_sample_posterior", raise_other)
    guide, params = svi_guide_params(t=30)
    with pytest.raises(ValueError, match="boom"):
        draw_posterior(random.PRNGKey(2), guide, params, 10, batch_size=4)
