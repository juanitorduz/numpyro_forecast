"""Permanent invariant suite ported from ``spikes/spike_scan_replay.py`` (S-M).

Each test pins one NumPyro-internals behavior that ``markov_time_series``
(roadmap §7) relies on; the spike script is kept verbatim for provenance
(recorded on jax 0.10.2 / numpyro 0.21.0) and these are its ⚙ canaries for
K8/K10. A JAX or NumPyro upgrade that breaks a scan behavior fails a named
test whose docstring explains exactly which design assumption just broke.
"""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import Array, random
from numpyro.contrib.control_flow import scan
from numpyro.handlers import reparam, seed, trace
from numpyro.infer import Predictive
from numpyro.infer.reparam import LocScaleReparam

PHI = 0.7
DRIFT_SIGMA = 0.1
T_OBS = 12
FUTURE = 6


def _scan_body(
    site: str, phi: float, drift_sigma: float
) -> Callable[[Array, None], tuple[Any, Any]]:
    def body(carry: Array, _: None) -> tuple[Any, Any]:
        z = numpyro.sample(site, dist.Normal(phi * carry, drift_sigma))
        return z, z

    return body


def _ar1_latent(covariates: Array, data: Array | None = None) -> Array:
    """AR(1) latent over the full horizon: two-scan (in-sample + future)."""
    duration = covariates.shape[-2]
    t_obs = duration if data is None else data.shape[-2]
    future = duration - t_obs
    init = jnp.zeros((1,))
    final, zs = scan(_scan_body("z", PHI, DRIFT_SIGMA), init, None, length=t_obs)
    if future == 0:
        return jnp.moveaxis(zs, 0, -2)
    _, zf = scan(_scan_body("z_future", PHI, DRIFT_SIGMA), final, None, length=future)
    return jnp.moveaxis(jnp.concatenate([zs, zf], axis=0), 0, -2)


def _ar1_model(covariates: Array, data: Array | None = None) -> None:
    level = _ar1_latent(covariates, data)
    numpyro.sample("obs", dist.Normal(level, 0.05), obs=data)


def test_scan_site_stored_time_leading() -> None:
    """S-M1: a scan sample site is stored time-leading ``(t, obs)``.

    The scan layout keeps time at axis 0; conversion to the package contract
    (time at ``-2``) happens only in the primitive's return (invariant I5).
    """
    cov = jnp.zeros((T_OBS, 1))
    tr = trace(seed(lambda: _ar1_latent(cov), random.PRNGKey(0))).get_trace()
    assert tr["z"]["value"].shape == (T_OBS, 1)


def test_predictive_replays_scan_posterior_unmodified() -> None:
    """S-M1b: ``Predictive`` replays a scan posterior dict without reshaping.

    The forecast horizon draws a separate ``z_future`` site from the prior
    while the in-sample ``z`` site is substituted verbatim.
    """
    z_last = 5.0
    n_samples = 4
    in_sample = jnp.broadcast_to(jnp.full((T_OBS, 1), z_last), (n_samples, T_OBS, 1))
    posterior = {"z": in_sample}
    full_cov = jnp.zeros((T_OBS + FUTURE, 1))
    data = jnp.zeros((T_OBS, 1))

    pred = Predictive(_ar1_model, posterior_samples=posterior, return_sites=["z_future"])
    zf = pred(random.PRNGKey(1), full_cov, data)["z_future"]
    assert zf.shape == (n_samples, FUTURE, 1)


def test_carry_threading_tracks_closed_form_extreme_state() -> None:
    """S-M2: the forecast scan is seeded by the final in-sample carry.

    With an extreme last latent (far from the unconditional mean 0), the
    horizon means track ``phi**(k+1) * z_last`` rather than decaying to 0.
    This is the loud regression: a broken carry hand-off forecasts the mean.
    """
    z_last = 5.0
    n_samples = 4
    in_sample = jnp.broadcast_to(jnp.full((T_OBS, 1), z_last), (n_samples, T_OBS, 1))
    posterior = {"z": in_sample}
    full_cov = jnp.zeros((T_OBS + FUTURE, 1))
    data = jnp.zeros((T_OBS, 1))

    pred = Predictive(_ar1_model, posterior_samples=posterior, return_sites=["z_future"])
    zf = pred(random.PRNGKey(1), full_cov, data)["z_future"]
    mean_zf = zf.mean(axis=0)[:, 0]
    closed = jnp.array([PHI ** (k + 1) * z_last for k in range(FUTURE)])
    assert float(jnp.abs(mean_zf - closed).max()) < 0.1
    assert float(mean_zf[0]) > 3.0


def _batched_body(site: str, size: int) -> Callable[[Array, None], tuple[Any, Any]]:
    def body(carry: Array, _: None) -> tuple[Any, Any]:
        # The batch plate sits at dim=-2 so the trailing obs dim (-1) stays put:
        # per-step shape is (*plate_batch, obs) = (size, 1).
        with numpyro.plate("series", size, dim=-2):
            z = numpyro.sample(site, dist.Normal(PHI * carry, DRIFT_SIGMA))
        return z, z

    return body


def _batched_latent(covariates: Array, data: Array | None = None, *, size: int = 3) -> Array:
    duration = covariates.shape[-2]
    t_obs = duration if data is None else data.shape[-2]
    init = jnp.zeros((size, 1))
    _, zs = scan(_batched_body("z", size), init, None, length=t_obs)
    return jnp.moveaxis(zs, 0, -2)


def test_batched_carry_threading_stores_t_b_obs() -> None:
    """S-M3: a plate inside the scan body yields a site stored ``(t, B, obs)``.

    Batched carry threading keeps time leading with the plate batch to its
    right, so the return ``moveaxis`` produces ``(B, duration, obs)``.
    """
    cov = jnp.zeros((T_OBS, 1))
    tr = trace(seed(lambda: _batched_latent(cov), random.PRNGKey(2))).get_trace()
    assert tr["z"]["value"].shape == (T_OBS, 3, 1)


def _reparam_body(site: str) -> Callable[[Array, None], tuple[Any, Any]]:
    def body(carry: Array, _: None) -> tuple[Any, Any]:
        with reparam(config={site: LocScaleReparam(0)}):
            z = numpyro.sample(site, dist.Normal(PHI * carry, DRIFT_SIGMA))
        return z, z

    return body


def _reparam_inside(covariates: Array, data: Array | None = None) -> Array:
    _, zs = scan(_reparam_body("z"), jnp.zeros((1,)), None, length=covariates.shape[-2])
    return jnp.moveaxis(zs, 0, -2)


def test_reparam_inside_scan_body_works() -> None:
    """S-M4a: ``LocScaleReparam`` applied INSIDE the scan body works.

    The trace exposes the decentered site ``z_decentered`` and the original
    ``z`` becomes deterministic: the supported placement for reparam configs.
    """
    cov = jnp.zeros((T_OBS, 1))
    tr = trace(seed(_reparam_inside, random.PRNGKey(3))).get_trace(cov)
    assert "z_decentered" in tr
    assert tr["z"]["type"] == "deterministic"


def _reparam_outside(covariates: Array, data: Array | None = None) -> Array:
    _, zs = scan(
        _scan_body("z", PHI, DRIFT_SIGMA), jnp.zeros((1,)), None, length=covariates.shape[-2]
    )
    return jnp.moveaxis(zs, 0, -2)


def test_reparam_wrapping_scan_raises() -> None:
    """S-M4b: wrapping the scan with ``reparam`` from OUTSIDE raises.

    Reparam config must live inside the body; a handler wrapping the whole
    scan is not a supported placement and must fail loudly.
    """
    cov = jnp.zeros((T_OBS, 1))
    with pytest.raises(Exception):  # noqa: B017 - documents that *some* error occurs
        with reparam(config={"z": LocScaleReparam(0)}):
            trace(seed(_reparam_outside, random.PRNGKey(4))).get_trace(cov)


def _plate_wraps_scan(covariates: Array, data: Array | None = None) -> Array:
    with numpyro.plate("series", 3):
        _, zs = scan(
            _scan_body("z", PHI, DRIFT_SIGMA),
            jnp.zeros((1,)),
            None,
            length=covariates.shape[-2],
        )
    return jnp.moveaxis(zs, 0, -2)


def test_plate_wrapping_scan_rejected() -> None:
    """S-M5: a plate WRAPPING the scan is rejected by NumPyro.

    Enclosing plates are unsupported: batching must happen with a plate INSIDE
    the body (S-M3), which is why ``markov_time_series`` rejects enclosing
    plates with actionable guidance.
    """
    cov = jnp.zeros((T_OBS, 1))
    with pytest.raises(Exception):  # noqa: B017 - documents rejection
        trace(seed(_plate_wraps_scan, random.PRNGKey(5))).get_trace(cov)
