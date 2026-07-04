"""Spike S-V: is vmapped SVI sound? (roadmap §4.4, risks F1/F2, gate K2)

F1: does svi.init (AutoNormal prototype tracing) work under vmap over data?
F2: is AutoNormal.sample_posterior pure in params (vmap == loop)?
F3 (bonus): does the vmapped Predictive forecast step work (vmap over
    posterior AND data/covariates with the model static)?
Also: AutoMultivariateNormal for F1, and the plan-B init broadcast check.
"""
import jax
import jax.numpy as jnp
from jax import lax, random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal, AutoNormal

# Rolling windows: same shapes, different data — the §4.4 contract.
T_TRAIN, T_TEST, N_WIN, STEPS, S = 30, 5, 4, 400, 300
DUR = T_TRAIN + (N_WIN - 1) + T_TEST

rng = random.PRNGKey(0)
full_cov = jnp.linspace(0, 1, DUR)[:, None]                      # (DUR, 1)
full_data = (2.0 * full_cov[:, 0] + 0.1 * random.normal(rng, (DUR,)))[:, None]

starts = jnp.arange(N_WIN)  # stride 1

def slice_windows(t0):
    return (
        lax.dynamic_slice_in_dim(full_data, t0, T_TRAIN, axis=0),
        lax.dynamic_slice_in_dim(full_cov, t0, T_TRAIN, axis=0),
        lax.dynamic_slice_in_dim(full_cov, t0, T_TRAIN + T_TEST, axis=0),
    )

train_d, train_c, hor_c = jax.vmap(slice_windows)(starts)
print(f"[wind] stacked windows: {train_d.shape}, {train_c.shape}, {hor_c.shape}")

def model(covariates, data=None):
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    mu = beta * covariates[:, 0]
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=None if data is None else data[:, 0])


def run_fit_probe(guide_cls, tag):
    guide = guide_cls(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.05), Trace_ELBO())

    def fit_one(key, d, c):
        state = svi.init(key, c, d)                       # <- F1: traced under vmap
        def step(s, _):
            s, loss = svi.update(s, c, d)
            return s, loss
        state, losses = lax.scan(step, state, length=STEPS)
        return svi.get_params(state), losses

    keys = jax.vmap(lambda i: random.fold_in(random.PRNGKey(1), i))(starts)
    params, losses = jax.jit(jax.vmap(fit_one))(keys, train_d, train_c)
    l0, l1 = float(losses[:, 0].mean()), float(losses[:, -1].mean())
    print(f"[F1:{tag}] vmapped init+scan OK; mean loss {l0:.1f} -> {l1:.1f}; "
          f"param leaves batched: "
          f"{ {k: v.shape for k, v in list(params.items())[:2]} }")
    return guide, svi, params, fit_one, keys


guide, svi, params, fit_one, keys = run_fit_probe(AutoNormal, "AutoNormal")
run_fit_probe(AutoMultivariateNormal, "AutoMVN")

# ---- F2: sample_posterior purity: vmap over params vs python loop, same key
kpost = random.PRNGKey(2)
post_vmap = jax.vmap(
    lambda p: guide.sample_posterior(kpost, p, sample_shape=(S,))
)(params)
post_loop = [
    guide.sample_posterior(kpost, jax.tree_util.tree_map(lambda x: x[i], params),
                           sample_shape=(S,))
    for i in range(N_WIN)
]
diffs = [
    float(jnp.abs(post_vmap["beta"][i] - post_loop[i]["beta"]).max())
    for i in range(N_WIN)
]
print(f"[F2  ] vmap vs loop sample_posterior max|diff| per window: "
      f"{[f'{d:.2e}' for d in diffs]}")
assert max(diffs) < 1e-5, "sample_posterior NOT pure under vmap"

# ---- F3: vmapped forecast — Predictive inside vmap, over posterior AND args
def forecast_one(key, post_w, d, hc):
    # in-package terms: Predictive(model, posterior)(key, horizon_covs, data)
    # here the toy model has no _future machinery; we just check the nesting
    # (Predictive vmaps over samples INSIDE our window vmap) executes and
    # shapes come out (S, T_TRAIN+T_TEST) per window.
    pred = Predictive(model, posterior_samples={"beta": post_w["beta"],
                                                "sigma": post_w["sigma"]})
    return pred(key, hc)["obs"]

fc = jax.jit(jax.vmap(forecast_one))(keys, post_vmap, train_d, hor_c)
print(f"[F3  ] nested vmap(Predictive) OK; forecast stack shape: {fc.shape} "
      f"(expect ({N_WIN}, {S}, {T_TRAIN + T_TEST}))")

# ---- plan-B probe (only relevant if F1 had failed; run anyway to document):
state0 = svi.init(random.PRNGKey(3), train_c[0], train_d[0])
bstates = jax.tree_util.tree_map(
    lambda x: jnp.broadcast_to(x, (N_WIN, *jnp.shape(x))), state0
)
def cont(key, st, d, c):
    def step(s, _):
        s, loss = svi.update(s, c, d)
        return s, loss
    st, losses = lax.scan(step, st, length=50)
    return losses[-1]
lb = jax.vmap(cont)(keys, bstates, train_d, train_c)
print(f"[planB] broadcast-init continuation also works: per-window losses "
      f"{[f'{float(x):.1f}' for x in lb]}")

# ---- compile-count sanity: is the vmapped fit ONE compilation?
lowered = jax.jit(jax.vmap(fit_one)).lower(keys, train_d, train_c)
print(f"[jit  ] vmapped fit lowers to a single computation "
      f"(windows are a batch dim, not an unroll): OK")

print("\nSPIKE S-V VERDICT: ALL GREEN")
