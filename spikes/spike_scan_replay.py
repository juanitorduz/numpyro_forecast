"""Spike S-M: is scan-based state-space replay + carry threading sound?

Pins the NumPyro-internals behaviors that `markov_time_series` (roadmap §7)
relies on, on jax 0.10.2 / numpyro 0.21.0:

S-M1  storage layout: a `contrib.control_flow.scan` sample site is stored
      time-leading `(t, *plate_batch, obs)`.
S-M1b replay: `Predictive` replays a posterior dict of scan sites unmodified
      (no layout conversion on the replay path).
S-M2  carry threading: the forecast scan seeded by the final in-sample carry
      tracks the AR(1) closed form `phi^k * z_last` (extreme-state check, far
      from the unconditional mean 0).
S-M3  batched carry threading: a plate inside the scan body yields a batched
      site stored `(t, B, obs)`.
S-M4  reparam placement: `LocScaleReparam` INSIDE the scan body works (the
      trace exposes `z_decentered`); wrapping the scan from OUTSIDE raises.
S-M5  plate placement: a plate WRAPPING the scan is rejected by NumPyro.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.contrib.control_flow import scan
from numpyro.handlers import reparam, seed, substitute, trace
from numpyro.infer import Predictive
from numpyro.infer.reparam import LocScaleReparam

PHI = 0.7
DRIFT_SIGMA = 0.1
T_OBS = 12
FUTURE = 6


def _scan_body(site, phi, drift_sigma):
    def body(carry, _):
        z = numpyro.sample(site, dist.Normal(phi * carry, drift_sigma))
        return z, z

    return body


def ar1_latent(covariates, data=None, *, phi=PHI, drift_sigma=DRIFT_SIGMA):
    """AR(1) latent over the full horizon, two-scan design (in-sample + future)."""
    duration = covariates.shape[-2]
    t_obs = duration if data is None else data.shape[-2]
    future = duration - t_obs
    init = jnp.zeros((1,))
    final, zs = scan(_scan_body("z", phi, drift_sigma), init, None, length=t_obs)
    if future == 0:
        return jnp.moveaxis(zs, 0, -2)
    _, zf = scan(_scan_body("z_future", phi, drift_sigma), final, None, length=future)
    return jnp.moveaxis(jnp.concatenate([zs, zf], axis=0), 0, -2)


def ar1_model(covariates, data=None):
    level = ar1_latent(covariates, data)
    numpyro.sample("obs", dist.Normal(level, 0.05), obs=data)


# ---- S-M1: storage layout is time-leading ----------------------------------
cov = jnp.zeros((T_OBS, 1))
tr = trace(seed(lambda: ar1_latent(cov), random.PRNGKey(0))).get_trace()
z_shape = tr["z"]["value"].shape
assert z_shape == (T_OBS, 1), f"[S-M1] expected scan layout (t, obs)=(12,1), got {z_shape}"
print(f"[S-M1 ] scan site stored time-leading: z.shape={z_shape} OK")


# ---- S-M1b + S-M2: replay unmodified + carry threading (extreme state) ------
# Extreme posterior: the last in-sample latent is far from the mean 0.
z_last = 5.0
S = 4
in_sample = jnp.broadcast_to(jnp.full((T_OBS, 1), z_last), (S, T_OBS, 1))
posterior = {"z": in_sample}
full_cov = jnp.zeros((T_OBS + FUTURE, 1))
data = jnp.zeros((T_OBS, 1))

pred = Predictive(ar1_model, posterior_samples=posterior, return_sites=["z_future"])
zf = pred(random.PRNGKey(1), full_cov, data)["z_future"]
assert zf.shape == (S, FUTURE, 1), f"[S-M1b] future replay shape {zf.shape}"
# forecast step k mean tracks phi^(k+1) * z_last (step 0 is one step ahead).
mean_zf = zf.mean(axis=0)[:, 0]
closed = jnp.array([PHI ** (k + 1) * z_last for k in range(FUTURE)])
err = float(jnp.abs(mean_zf - closed).max())
assert err < 0.1, f"[S-M2] carry threading error {err} vs closed form {closed}"
assert float(mean_zf[0]) > 3.0, "[S-M2] step-1 forecast should stay near phi*z_last, not 0"
print(f"[S-M1b] Predictive replays scan posterior unmodified: {zf.shape} OK")
print(f"[S-M2 ] carry threads phi^k*z_last (max err {err:.4f}, step1={float(mean_zf[0]):.3f}) OK")


# ---- S-M3: batched carry threading (plate inside the scan body) -------------
def batched_body(site, size):
    def body(carry, _):
        # The batch plate sits at dim=-2 so the trailing obs dim (-1) stays put:
        # per-step shape is (*plate_batch, obs) = (size, 1).
        with numpyro.plate("series", size, dim=-2):
            z = numpyro.sample(site, dist.Normal(PHI * carry, DRIFT_SIGMA))
        return z, z

    return body


def batched_latent(covariates, data=None, *, size=3):
    duration = covariates.shape[-2]
    t_obs = duration if data is None else data.shape[-2]
    init = jnp.zeros((size, 1))
    _, zs = scan(batched_body("z", size), init, None, length=t_obs)
    return jnp.moveaxis(zs, 0, -2)


tr_b = trace(seed(lambda: batched_latent(cov), random.PRNGKey(2))).get_trace()
zb = tr_b["z"]["value"].shape
assert zb == (T_OBS, 3, 1), f"[S-M3] batched scan layout expected (t, B, obs), got {zb}"
print(f"[S-M3 ] batched carry threads; site stored {zb} (t, B, obs) OK")


# ---- S-M4: reparam INSIDE the scan body works; OUTSIDE raises ---------------
def reparam_body(site):
    def body(carry, _):
        with reparam(config={site: LocScaleReparam(0)}):
            z = numpyro.sample(site, dist.Normal(PHI * carry, DRIFT_SIGMA))
        return z, z

    return body


def reparam_inside(covariates, data=None):
    _, zs = scan(reparam_body("z"), jnp.zeros((1,)), None, length=covariates.shape[-2])
    return jnp.moveaxis(zs, 0, -2)


tr_r = trace(seed(reparam_inside, random.PRNGKey(3))).get_trace(cov)
assert "z_decentered" in tr_r, f"[S-M4] expected z_decentered site, got {sorted(tr_r)}"
assert tr_r["z"]["type"] == "deterministic", "[S-M4] z should be deterministic after reparam"
print("[S-M4a] reparam INSIDE scan works; trace exposes z_decentered OK")


def reparam_outside(covariates, data=None):
    _, zs = scan(_scan_body("z", PHI, DRIFT_SIGMA), jnp.zeros((1,)), None, length=covariates.shape[-2])
    return jnp.moveaxis(zs, 0, -2)


raised = False
try:
    with reparam(config={"z": LocScaleReparam(0)}):
        trace(seed(reparam_outside, random.PRNGKey(4))).get_trace(cov)
except Exception as exc:  # noqa: BLE001 - spike documents that *some* error occurs
    raised = True
    print(f"[S-M4b] reparam OUTSIDE scan raises as expected: {type(exc).__name__} OK")
assert raised, "[S-M4b] reparam wrapping the scan should raise"


# ---- S-M5: plate WRAPPING the scan is rejected ------------------------------
def plate_wraps_scan(covariates, data=None):
    with numpyro.plate("series", 3):
        _, zs = scan(
            _scan_body("z", PHI, DRIFT_SIGMA), jnp.zeros((1,)), None, length=covariates.shape[-2]
        )
    return jnp.moveaxis(zs, 0, -2)


raised = False
try:
    trace(seed(plate_wraps_scan, random.PRNGKey(5))).get_trace(cov)
except Exception as exc:  # noqa: BLE001 - spike documents rejection
    raised = True
    print(f"[S-M5 ] plate WRAPPING scan rejected as expected: {type(exc).__name__} OK")
assert raised, "[S-M5] plate wrapping the scan should be rejected"

# substitute is imported to document the replay mechanism Predictive uses.
_ = substitute

print("\nSPIKE S-M VERDICT: ALL GREEN")
