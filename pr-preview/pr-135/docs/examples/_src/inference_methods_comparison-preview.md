# Comparing inference methods: NUTS, SVI, Pathfinder, and MCLMC with `numpyro_forecast`


One advantage of writing a forecasting model once is that you can fit it with different inference engines without touching the model code. In this notebook we take the weekly BART ridership model from the [univariate forecasting example](forecasting_univariate.md) (a random-walk local level, Fourier seasonality, and a Student-T likelihood) and fit it four ways: with **NUTS** (Markov chain Monte Carlo), with **SVI** (stochastic variational inference, using a custom [optax](https://optax.readthedocs.io/en/latest/) optimizer), with **multi-path Pathfinder** (quasi-Newton variational inference run over several parallel L-BFGS paths, from [BlackJAX](https://blackjax-devs.github.io/blackjax/)), and with **MCLMC** (microcanonical Langevin Monte Carlo, a BlackJAX sampler that plugs into the same MCMC entry point through a kernel adapter).

Every engine ends in the same shape: a dict of posterior samples with a leading sample axis, either read straight off an `MCMC` run (`mcmc.get_samples()`) or drawn from a fitted variational approximation ([draw_posterior](../../../reference/predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior), [multipathfinder_samples](../../../reference/contrib.blackjax.multipathfinder_samples.md#numpyro_forecast.contrib.blackjax.multipathfinder_samples)). A single [to_datatree](../../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) call then turns any of them into an ArviZ `DataTree` holding both the in-sample posterior predictive and the forecast over the test horizon, which powers the plots and the evaluation alike. We compare the four engines on the continuous ranked probability score (CRPS) over the training and test windows, and on wall-clock time.


# Prepare notebook


    In [1]:


``` python
from collections.abc import Mapping
from time import perf_counter

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import pandas as pd
import xarray as xr
from jax import random
from numpyro.diagnostics import effective_sample_size
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.reparam import LocScaleReparam
from numpyro.optim import optax_to_numpyro

from numpyro_forecast import (
    Horizon,
    draw_posterior,
    eval_crps,
    innovations,
    predict,
    time_reparam,
    to_datatree,
)
from numpyro_forecast.contrib.blackjax import (
    BlackjaxMCLMCKernel,
    fit_multipathfinder,
    multipathfinder_samples,
)
from numpyro_forecast.datasets import load_bart_weekly
from numpyro_forecast.features import fourier_features
from numpyro_forecast.typing import Array, ForecastModel

az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"

numpyro.set_host_device_count(n=4)

rng_key = random.PRNGKey(seed=42)

%load_ext autoreload
%autoreload 2
%load_ext jaxtyping
%jaxtyping.typechecker beartype.beartype
%config InlineBackend.figure_format = "retina"
```


# Read data

We work with total weekly BART ridership on the log scale, exactly as in the univariate example. Throughout the package, time lives at axis `-2` and the observation dimension at axis `-1`, so the series has shape `(weeks, 1)`.


    In [2]:


``` python
data = load_bart_weekly()  # (weeks, 1), log scale
duration = data.shape[0]
print("data shape:", data.shape)
```


    data shape: (469, 1)


# Train-test split

We hold out the last `52` weeks (one full year) as the test set and train on the preceding `417` weeks, so the test window covers a complete seasonal cycle.


    In [3]:


``` python
T0 = 0
T2 = duration  # 469
T1 = T2 - 52  # 417: train / test split

y_train = data[T0:T1]
y_test = data[T1:T2]

time = np.arange(T2)
time_train = time[T0:T1]
time_test = time[T1:T2]
print("train:", y_train.shape, "test:", y_test.shape)

fig, ax = plt.subplots()
ax.plot(time_train, np.asarray(y_train[:, 0]), color="C0", label="train")
ax.plot(time_test, np.asarray(y_test[:, 0]), color="C1", label="test")
ax.axvline(T1, color="gray", ls="--", label="train/test split")
ax.legend()
ax.set(title="Train / test split", xlabel="week", ylabel="log(# rides)");
```


    train: (417, 1) test: (52, 1)


<figure class="figure">
<p><img src="inference_methods_comparison_files/figure-html/cell-4-output-2.png" class="figure-img" width="1011" height="611" /></p>
</figure>


# Seasonal features

The annual cycle enters through a Fourier design matrix built with [fourier_features](../../../reference/features.fourier_features.md#numpyro_forecast.features.fourier_features): `26` harmonics (so `52` sine and cosine columns) at a period of `365.25 / 7` weeks.


    In [4]:


``` python
num_terms = 26
covariates = fourier_features(duration, period=365.25 / 7, num_terms=num_terms)
covariates_train = covariates[T0:T1]
print("covariates shape:", covariates.shape)
```


    covariates shape: (469, 52)


# Model specification

The model is the same *local level with seasonality* as in the [univariate forecasting example](forecasting_univariate.md): a global bias, a random-walk level, and a Fourier regression for the annual cycle, with a heavy-tailed Student-T likelihood to absorb outlier weeks. See that notebook for the full mathematical specification, the priors, and a rendering of the model graph. Here we only restate the code, written once as a plain `(covariates, data=None)` function, so all four inference engines below consume exactly the same object.


    In [5]:


``` python
def univariate_model(covariates: Array, data: Array | None = None) -> None:
    """Local level + Fourier regression with Student-T observations."""
    h = Horizon.from_data(covariates, data)
    num_features = covariates.shape[-1]

    bias = numpyro.sample("bias", dist.Normal(0.0, 10.0))
    weight = numpyro.sample("weight", dist.Normal(0.0, 0.1).expand([num_features]).to_event(1))
    drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-20.0, 5.0))
    nu = numpyro.sample("nu", dist.Gamma(10.0, 2.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-5.0, 5.0))
    centered = numpyro.sample("centered", dist.Uniform(0.0, 1.0))

    drift = innovations(
        h,
        "drift",
        lambda: dist.Normal(0.0, drift_scale),
        reparam=LocScaleReparam(centered=centered),
    )
    level = jnp.cumsum(drift, axis=-2)
    regression = (weight * covariates).sum(axis=-1, keepdims=True)
    prediction = level + bias + regression

    predict(h, dist.StudentT(df=nu, loc=0.0, scale=sigma), prediction)
```


# Inference

We now fit the same model four times, once per inference engine, using plain NumPyro (`MCMC`/`SVI`) or BlackJAX entry points directly against `univariate_model`; nothing about the model changes, only how we draw a posterior from it. In brief:

- **NUTS** (the No-U-Turn Sampler) is gradient-based Markov chain Monte Carlo. It draws asymptotically exact samples from the posterior, which makes it our reference here, usually at the highest computational cost of the four, though the discussion below shows that ranking is not absolute.
- **SVI** (stochastic variational inference) turns inference into optimization: it fits the parameters of an approximating guide distribution (here `AutoNormal`, a diagonal Gaussian) by maximizing the evidence lower bound (ELBO). It is much faster than MCMC, and its accuracy is bounded by how well the guide family can match the true posterior.
- **Multi-path Pathfinder** is quasi-Newton variational inference, run several times over. Each of several independent L-BFGS paths (vectorized under `vmap`) recycles its own optimization trajectory into a normal approximation and an ELBO estimate; instead of keeping only the best-ELBO path, all of them are kept and combined when the draws are taken, either by Pareto-smoothed importance sampling (PSIS) over the pooled draws or by weighting whole paths by their ELBO, with the `pareto_k` diagnostic deciding which of the two is trustworthy. It is often used for fast approximate posteriors or to initialize MCMC.
- **MCLMC** (microcanonical Langevin Monte Carlo) is MCMC of a different flavor: it simulates energy-preserving isokinetic dynamics with stochastic momentum refreshment and skips the Metropolis accept/reject correction entirely. Every draw costs a fixed two gradient evaluations, far below the cost of a NUTS trajectory, in exchange for a small step-size-controlled bias in the stationary distribution.

Each engine below hands us either raw posterior samples (`mcmc.get_samples()` for NUTS and MCLMC) or a fitted guide/approximation that a small drawing function turns into samples of the same shape ([draw_posterior](../../../reference/predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior) for SVI, [multipathfinder_samples](../../../reference/contrib.blackjax.multipathfinder_samples.md#numpyro_forecast.contrib.blackjax.multipathfinder_samples) for Pathfinder), and [to_datatree](../../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) accepts either form, so the export below is one call whichever engine produced the posterior.


## NUTS

We run `4` chains in parallel with `2_000` warmup steps and `1_000` posterior draws each. The posterior includes one drift increment per training week (417 of them), so this is an expensive fit: of the four engines here, only the eight-path Pathfinder run below takes longer.


    In [6]:


``` python
rng_key, rng_subkey = random.split(rng_key)

start = perf_counter()
nuts_mcmc = MCMC(
    NUTS(univariate_model),
    num_warmup=2_000,
    num_samples=1_000,
    num_chains=4,
    chain_method="parallel",
    progress_bar=False,
)
nuts_mcmc.run(rng_subkey, covariates_train, y_train, extra_fields=("num_steps",))
nuts_samples = nuts_mcmc.get_samples()
jax.block_until_ready(nuts_samples)
nuts_seconds = perf_counter() - start
print(f"NUTS: 4 chains x 1_000 draws in {nuts_seconds:.1f}s")
```


    NUTS: 4 chains x 1_000 draws in 40.9s


## NUTS with time-axis reparameterization

The `417` drift increments are the expensive part of this posterior, and not only because there are many of them: the level is their cumulative sum, so the data constrain sums of neighboring increments far more tightly than any single one, and the posterior over the block is a long thin ellipse along the time axis. [time_reparam](../../../reference/reparam.time_reparam.md#numpyro_forecast.reparam.time_reparam) wraps the model with a `numpyro.handlers.reparam` handler that rotates every in-sample latent under the `time` plate into an orthonormal basis, here the discrete cosine basis (`"dct"`; `"haar"` picks a wavelet basis instead). The rotation leaves the log density unchanged and only changes the coordinates inference sees: the sampler now draws the auxiliary site `drift_decentered_dct`, and both `drift_decentered` and `drift` become deterministic sites recomputed from it. We build the wrapped model once (it is a static argument of the jit-compiled drivers, so wrapping inside a loop would recompile) and reuse it in every cell below that needs it, including the ArviZ export.

The run uses the same `MCMC(NUTS(...))` configuration as above. We collect `num_steps`, the number of leapfrog steps per iteration, for both runs and compare wall time, mean leapfrog steps per draw, and the smallest bulk effective sample size over the `417` drift increments, computed with `numpyro.diagnostics.effective_sample_size` from the chain-grouped samples.


    In [7]:


``` python
univariate_model_dct = time_reparam(univariate_model, "dct")

rng_key, rng_subkey = random.split(rng_key)

start = perf_counter()
nuts_dct_mcmc = MCMC(
    NUTS(univariate_model_dct),
    num_warmup=2_000,
    num_samples=1_000,
    num_chains=4,
    chain_method="parallel",
    progress_bar=False,
)
nuts_dct_mcmc.run(rng_subkey, covariates_train, y_train, extra_fields=("num_steps",))
nuts_dct_samples = nuts_dct_mcmc.get_samples()
jax.block_until_ready(nuts_dct_samples)
nuts_dct_seconds = perf_counter() - start
print(f"NUTS + DCT: 4 chains x 1_000 draws in {nuts_dct_seconds:.1f}s")
```


    NUTS + DCT: 4 chains x 1_000 draws in 55.1s


    In [8]:


``` python
def nuts_report(mcmc: MCMC, seconds: float) -> dict[str, float]:
    """Wall time, mean leapfrog steps per draw, and min bulk ESS over ``drift`` for a run."""
    drift = mcmc.get_samples(group_by_chain=True)["drift"]
    return {
        "walltime (s)": seconds,
        "leapfrog / draw": float(mcmc.get_extra_fields()["num_steps"].mean()),
        "min ESS (drift)": float(effective_sample_size(drift).min()),
    }


nuts_comparison = pd.DataFrame(
    {
        "NUTS": nuts_report(nuts_mcmc, nuts_seconds),
        "NUTS + DCT": nuts_report(nuts_dct_mcmc, nuts_dct_seconds),
    }
).T
nuts_comparison.round(1)
```


|            | walltime (s) | leapfrog / draw | min ESS (drift) |
|------------|--------------|-----------------|-----------------|
| NUTS       | 40.9         | 446.4           | 45.8            |
| NUTS + DCT | 55.1         | 314.4           | 558.9           |


## SVI

NumPyro's `SVI` expects a NumPyro optimizer, so any optax `GradientTransformation` needs one line of glue, `numpyro.optim.optax_to_numpyro`, before it can be passed in. We use that to run a custom optimizer built from two pieces:

- A **one-cycle** learning-rate schedule (`optax.linear_onecycle_schedule`): a linear warmup to a peak followed by a long annealing phase. The warmup lets the optimizer pass through a much higher mid-run learning rate than a fixed setting could tolerate, and the final annealing polishes the optimum.
- **Reduce-on-plateau** (`optax.contrib.reduce_on_plateau`): an adaptive safeguard that scales the updates down by `factor=0.8` whenever the ELBO, averaged over `accumulation_size=100` steps, stops improving for `patience=20` consecutive windows. NumPyro forwards the per-step ELBO value to the optimizer chain, which is exactly the signal this transformation monitors.

The univariate example needs `50_000` steps at a fixed `Adam(0.005)`; cycling up to a peak of `0.01` reaches a comparable ELBO in `20_000` steps, less than half the budget.


    In [9]:


``` python
num_steps = 20_000

scheduler = optax.linear_onecycle_schedule(
    transition_steps=num_steps,
    peak_value=0.01,
    pct_start=0.3,
    pct_final=0.85,
    div_factor=2,
    final_div_factor=3,
)

optimizer = optax.chain(
    optax.adam(learning_rate=scheduler),
    optax.contrib.reduce_on_plateau(
        factor=0.8,
        patience=20,
        accumulation_size=100,
    ),
)
optim = optax_to_numpyro(optimizer)

fig, ax = plt.subplots()
ax.plot(np.asarray(jax.vmap(scheduler)(jnp.arange(num_steps))), color="C0")
ax.set(title="One-cycle learning rate schedule", xlabel="SVI step", ylabel="learning rate");
```


<figure class="figure">
<p><img src="inference_methods_comparison_files/figure-html/cell-10-output-1.png" class="figure-img" width="1011" height="611" /></p>
</figure>


    In [10]:


``` python
guide = AutoNormal(univariate_model)
svi = SVI(univariate_model, guide, optim, Trace_ELBO())

rng_key, rng_subkey = random.split(rng_key)

start = perf_counter()
svi_result = svi.run(rng_subkey, num_steps, covariates_train, y_train, progress_bar=False)
jax.block_until_ready(svi_result.losses)
svi_seconds = perf_counter() - start
print(f"SVI: {num_steps:_} steps in {svi_seconds:.1f}s")

fig, ax = plt.subplots()
ax.plot(svi_result.losses)
ax.set(title="ELBO loss", xlabel="SVI step", ylabel="loss");
```


    SVI: 20_000 steps in 4.9s


<figure class="figure">
<p><img src="inference_methods_comparison_files/figure-html/cell-11-output-2.png" class="figure-img" width="1011" height="611" /></p>
</figure>


`SVI.run` fits `guide`, but a guide by itself is not yet a set of posterior samples: [draw_posterior](../../../reference/predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior) draws `2_000` samples of the latent sites from the fitted `guide`/`params` pair, in the same leading-sample-axis shape [to_datatree](../../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) and [forecast](../../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast) expect.


    In [11]:


``` python
rng_key, rng_subkey = random.split(rng_key)
svi_posterior = draw_posterior(rng_subkey, guide, svi_result.params, 2_000)
```


## SVI with time-axis reparameterization

A diagonal-Gaussian guide is exactly the approximation that the long thin drift posterior defeats: `AutoNormal` has no off-diagonal terms with which to represent the coupling between neighboring increments, so it either shrinks every marginal to fit the ridge or overstates the joint uncertainty. In the DCT basis that coupling is largely gone, so the same guide family is fitted to a rotated posterior that is closer to diagonal. `univariate_model_dct` is the model handed to `AutoNormal`, to `SVI`, and later to the export, and [draw_posterior](../../../reference/predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior) returns a posterior dict that now carries the sampled `drift_decentered_dct` alongside the deterministic `drift_decentered` and `drift`, which [to_datatree](../../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) and [forecast](../../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast) consume unchanged.

We first reuse the baseline's optimizer, one-cycle schedule, and `20_000` step budget unchanged, so that the two runs differ only in the coordinates the guide sees.


    In [12]:


``` python
guide_dct_same = AutoNormal(univariate_model_dct)
svi_dct_same = SVI(univariate_model_dct, guide_dct_same, optim, Trace_ELBO())

rng_key, rng_subkey = random.split(rng_key)
svi_dct_same_result = svi_dct_same.run(
    rng_subkey, num_steps, covariates_train, y_train, progress_bar=False
)
print(f"final ELBO loss, SVI + DCT (peak 0.01): {np.mean(svi_dct_same_result.losses[-100:]):.1f}")
```


    final ELBO loss, SVI + DCT (peak 0.01): 197.1


The rotated run descends far faster over the first few thousand steps and then stalls well above the baseline. It has not converged to a different optimum: run for longer at a fixed learning rate it keeps descending and ends below the baseline (the [univariate example](forecasting_univariate.md) shows this under a fixed `Adam(0.005)` for `50_000` steps). What stalls it is the intercept. Early in the run the level holds part of the intercept and `bias` has to take it over, and in the original coordinates that hand-over is a two-parameter ridge between `bias` and the first increment, whereas in the DCT basis a level offset is spread over all `417` cosine coefficients, so Adam moves it one small per-coordinate step at a time. The one-cycle schedule, whose peak of `0.01` was tuned for the original coordinates, starts annealing at `30%` of the budget, long before that hand-over is complete.

The rotated posterior is better conditioned, so it tolerates a larger step. Tripling the peak to `0.03` and keeping everything else identical lets the run finish within the same `20_000` steps, and it does so for every seed we tried (a peak of `0.02` finishes only for some seeds, and the baseline itself gets worse at higher peaks, so the re-tuning is specific to the new coordinates). This second fit is the `SVI + DCT` column in the comparison below.


    In [13]:


``` python
scheduler_dct = optax.linear_onecycle_schedule(
    transition_steps=num_steps,
    peak_value=0.03,
    pct_start=0.3,
    pct_final=0.85,
    div_factor=2,
    final_div_factor=3,
)
optim_dct = optax_to_numpyro(
    optax.chain(
        optax.adam(learning_rate=scheduler_dct),
        optax.contrib.reduce_on_plateau(factor=0.8, patience=20, accumulation_size=100),
    )
)

guide_dct = AutoNormal(univariate_model_dct)
svi_dct = SVI(univariate_model_dct, guide_dct, optim_dct, Trace_ELBO())

rng_key, rng_subkey = random.split(rng_key)

start = perf_counter()
svi_dct_result = svi_dct.run(rng_subkey, num_steps, covariates_train, y_train, progress_bar=False)
jax.block_until_ready(svi_dct_result.losses)
svi_dct_seconds = perf_counter() - start
print(f"SVI + DCT: {num_steps:_} steps in {svi_dct_seconds:.1f}s")

fig, ax = plt.subplots()
ax.plot(svi_result.losses, color="C0", label="SVI (peak 0.01)")
ax.plot(svi_dct_same_result.losses, color="C2", label="SVI + DCT (peak 0.01)")
ax.plot(svi_dct_result.losses, color="C3", label="SVI + DCT (peak 0.03)")
ax.set_yscale("symlog")
ax.legend()
ax.set(title="ELBO loss", xlabel="SVI step", ylabel="loss");
```


    SVI + DCT: 20_000 steps in 2.4s


<figure class="figure">
<p><img src="inference_methods_comparison_files/figure-html/cell-14-output-2.png" class="figure-img" width="1011" height="611" /></p>
</figure>


    In [14]:


``` python
def svi_report(losses: Array, median: Mapping[str, Array]) -> dict[str, float]:
    """Report the final ELBO loss (mean over the last ``100`` steps) and scalar guide medians."""
    return {
        "final ELBO loss": float(jnp.mean(losses[-100:])),
        "bias": float(median["bias"]),
        "drift_scale": float(median["drift_scale"]),
        "centered": float(median["centered"]),
    }


svi_comparison = pd.DataFrame(
    {
        "SVI (peak 0.01)": svi_report(svi_result.losses, guide.median(svi_result.params)),
        "SVI + DCT (peak 0.01)": svi_report(
            svi_dct_same_result.losses, guide_dct_same.median(svi_dct_same_result.params)
        ),
        "SVI + DCT (peak 0.03)": svi_report(
            svi_dct_result.losses, guide_dct.median(svi_dct_result.params)
        ),
    }
).T
svi_comparison.round(4)
```


|                       | final ELBO loss | bias    | drift_scale | centered |
|-----------------------|-----------------|---------|-------------|----------|
| SVI (peak 0.01)       | -435.7541       | 14.5341 | 0.0014      | 0.0761   |
| SVI + DCT (peak 0.01) | 197.1361        | 5.6526  | 0.1576      | 0.7438   |
| SVI + DCT (peak 0.03) | -499.3820       | 14.5140 | 0.0039      | 0.0707   |


The guide medians show the mechanism directly. The stalled run still carries part of the intercept in the level (a lower `bias` and a `drift_scale` two orders of magnitude larger than the baseline's, which inflates every increment's marginal and, through the cumulative sum, the whole in-sample band), while the re-tuned run recovers the baseline's `bias` and a small `drift_scale` and ends at a lower ELBO loss than the baseline. The model's log density is unchanged by the rotation; what changed is the optimization problem the guide solves, and its schedule has to be re-tuned for the new coordinates rather than carried over.


    In [15]:


``` python
rng_key, rng_subkey = random.split(rng_key)
svi_dct_posterior = draw_posterior(rng_subkey, guide_dct, svi_dct_result.params, 2_000)
```


## Pathfinder

[fit_multipathfinder](../../../reference/contrib.blackjax.fit_multipathfinder.md#numpyro_forecast.contrib.blackjax.fit_multipathfinder) lives in `numpyro_forecast.contrib.blackjax` and needs the optional BlackJAX backend (install it with `pip install "numpyro_forecast[blackjax]"`). It runs several independent L-BFGS paths toward the posterior mode, each inducing its own normal approximation and its own ELBO estimate, and keeps every one of them instead of returning only the best-ELBO path.

Three settings matter here. The first is `maxiter`, the L-BFGS iteration budget: the default (`30`) suits posteriors with a handful of parameters, but ours has one drift increment per training week and needs a few hundred iterations to approach the high-density region, so we set it to `500`. The second is `maxcor`, the L-BFGS history size: it caps the rank of the low-rank-plus-diagonal covariance correction at roughly twice its value, so the default of `10` gives a correction of rank about `20` on a posterior with roughly `474` parameters here, and we raise it to `50` to let the approximation capture more of that covariance structure. The third is `num_paths`, the number of independent L-BFGS paths: they run vectorized under `vmap`, so all eight share one compilation and advance together rather than one after another, and all of their approximations survive into the drawing step below rather than being discarded in favor of a single winner.

`num_elbo_samples` is the memory knob rather than a quality knob. The fit estimates an ELBO at every L-BFGS iterate of every path, so it materializes on the order of `num_paths * maxiter * num_elbo_samples * 474` numbers at once; we keep it at `100` so that the `maxiter=500` this posterior needs stays affordable on a laptop. It also sets the size of the pool that the fit-time `pareto_k` diagnostic printed below is computed over, which is a separate pool from the draws taken in the next cell.

This section also owns its own `PRNGKey`, split from a fixed seed rather than threaded through the notebook's running `rng_key`, so edits earlier in the notebook cannot reshuffle its random stream.


    In [16]:


``` python
key_pathfinder_fit, key_pathfinder_draw = random.split(random.PRNGKey(seed=2_025))

start = perf_counter()
pathfinder_fit = fit_multipathfinder(
    key_pathfinder_fit,
    univariate_model,
    y_train,
    covariates_train,
    num_paths=8,
    num_elbo_samples=100,
    maxiter=500,
    maxcor=50,
)
pathfinder_seconds = perf_counter() - start

print("per-path ELBO:", [round(elbo, 1) for elbo in pathfinder_fit.elbos])
print(f"pareto_k: {pathfinder_fit.pareto_k:.2f}")
print(f"Pathfinder: {len(pathfinder_fit.elbos)} paths in {pathfinder_seconds:.1f}s")
```


    /var/folders/cm/3dzy9rdd5s3672z0s1brjkvh0000gn/T/ipykernel_7288/3865784160.py:4: UserWarning: pareto_k=12.35 > 0.7: PSIS importance weights over the pooled draws are unreliable, so multipathfinder_samples(..., resample="auto") falls back to ELBO-weighted path sampling instead of PSIS resampling; increase num_paths/maxiter/maxcor or fall back to MCMC.
      pathfinder_fit = fit_multipathfinder(


    per-path ELBO: [-1272.3, -741.4, -524.7, -597.5, -805.8, -1532.1, -497.5, -192.3]
    pareto_k: 12.35
    Pathfinder: 8 paths in 68.3s


[multipathfinder_samples](../../../reference/contrib.blackjax.multipathfinder_samples.md#numpyro_forecast.contrib.blackjax.multipathfinder_samples) draws fresh samples from every path's fitted approximation on each call, `2_000` per path here, and then combines the `8` paths into the `2_000` returned draws. How it combines them is the `resample` argument. With `resample="psis"` all `8 * 2_000` fresh draws are pooled, scored both under the model and under the approximation that produced them, and importance-resampled with Pareto smoothing, which is the textbook multi-path Pathfinder estimator. With `resample="elbo"` each returned draw instead picks a whole path with probability proportional to `softmax` of the per-path ELBOs and takes one fresh draw from it, so a path that fits several hundred nats better than the rest simply takes over.

The default, `resample="auto"`, chooses between the two using the `pareto_k` printed above: PSIS when `pareto_k` is at most `0.7`, and ELBO-weighted path sampling otherwise. The gate matters because importance weights degenerate in high dimensions. On a posterior with hundreds of parameters the log ratio between the target and the approximation is dominated by a handful of draws, `pareto_k` climbs far above `0.7`, and PSIS resampling collapses the answer onto those few draws; weighting whole paths cannot concentrate that way, because it reweights `8` well-separated numbers rather than thousands of individual draws. Reading the `pareto_k` printed above therefore tells you which branch the cell below took: below `0.5` the PSIS weights are reliable, `0.5` to `0.7` is borderline, and above `0.7` the draws come from ELBO-weighted path sampling instead, which is exactly what the warning printed by the fit cell above is telling you.

The output contract is unchanged either way: `2_000` samples, leading sample axis, ready for [to_datatree](../../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) exactly like [draw_posterior](../../../reference/predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior).


    In [17]:


``` python
pathfinder_posterior = multipathfinder_samples(key_pathfinder_draw, pathfinder_fit, 2_000)
```


## MCLMC

The `MCMC` entry point used for NUTS accepts any NumPyro-compatible kernel, and `numpyro_forecast.contrib.blackjax` provides adapters for BlackJAX samplers (the same optional dependency as Pathfinder above). [BlackjaxMCLMCKernel](../../../reference/contrib.blackjax.BlackjaxMCLMCKernel.md#numpyro_forecast.contrib.blackjax.BlackjaxMCLMCKernel) wraps microcanonical Langevin Monte Carlo: the kernel tunes the step size, the trajectory length L, and a diagonal preconditioner once inside its `init`, and every subsequent MCMC step is a single tuned MCLMC step. Because that tuning replaces warmup, we pass `num_warmup=0` (the adapter warns that warmup steps would be discarded work), and because the adapter must run chains sequentially we pass `chain_method="sequential"` and draw one long chain instead of four parallel ones.

The flip side of skipping the Metropolis correction is that nothing rejects a bad step: the draws carry a small discretization bias controlled by the tuned step size, and an unlucky tuning run degrades the samples silently instead of showing up as divergences the way it would in NUTS. In practice one validates MCLMC against a proper score like the CRPS below or against a short NUTS reference run. We use a generous tuning budget, which costs little because a tuning step is as cheap as a sampling step.


    In [18]:


``` python
rng_key, rng_subkey = random.split(rng_key)

start = perf_counter()
mclmc_mcmc = MCMC(
    BlackjaxMCLMCKernel(univariate_model, num_tuning_steps=10_000),
    num_warmup=0,
    num_samples=10_000,
    chain_method="sequential",
    progress_bar=False,
)
mclmc_mcmc.run(rng_subkey, covariates_train, y_train)
mclmc_samples = mclmc_mcmc.get_samples()
jax.block_until_ready(mclmc_samples)
mclmc_seconds = perf_counter() - start
print(f"MCLMC: 1 chain x 10_000 draws in {mclmc_seconds:.1f}s")
```


    MCLMC: 1 chain x 10_000 draws in 5.3s


# Exporting fits to ArviZ

[to_datatree](../../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) is posterior-first: it never fits anything itself, so every engine above hands it a plain dict of latent-site draws with a leading sample axis, either `mcmc.get_samples()` for the two MCMC engines or the [draw_posterior](../../../reference/predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior)/[multipathfinder_samples](../../../reference/contrib.blackjax.multipathfinder_samples.md#numpyro_forecast.contrib.blackjax.multipathfinder_samples) output for the two variational ones. The `num_chains` argument reshapes that flat sample axis into `(chain, draw)`: `4` for the NUTS posterior (matching its `4` parallel chains) and the default `1` (a single pseudo chain) for the other three, whose draws carry no chain structure of their own. Because we pass the full-length covariates (longer than the training data, the package-wide shape convention for a forecast horizon), the same call also draws the forecast over the held-out year and stores it in the `predictions` and `predictions_constant_data` groups, continuing the in-sample time coordinate. If you need finer control over the forecast draws, [add_forecast_groups](../../../reference/convert.add_forecast_groups.md#numpyro_forecast.convert.add_forecast_groups) attaches them step by step.

The export is one call, identical for the four posteriors. The only post-processing we add is cosmetic, for plotting: this series is univariate, so we drop the singleton observation dimension and expose the week index as a variable that `az.plot_lm` can use as the x axis.


    In [19]:


``` python
def build_tree(
    rng_key: Array,
    posterior: Mapping[str, Array | np.ndarray],
    *,
    num_chains: int = 1,
    model: ForecastModel = univariate_model,
) -> xr.DataTree:
    """Export a posterior to an ArviZ ``DataTree`` with in-sample and forecast groups."""
    tree = to_datatree(
        rng_key,
        model,
        posterior,
        y_train,
        covariates,
        num_chains=num_chains,
        posterior_dims={"drift": ["time"]},
    )
    for group in ("posterior_predictive", "observed_data", "predictions"):
        tree[group] = tree[group].dataset.isel(obs_dim=0)
    tree["constant_data"] = tree["constant_data"].dataset.assign(
        week=("time", time_train.astype(float))
    )
    tree["predictions_constant_data"] = tree["predictions_constant_data"].dataset.assign(
        week=("time", time_test.astype(float))
    )
    return tree


rng_key, key_nuts, key_svi, key_pf, key_mclmc = random.split(rng_key, 5)
nuts_tree = build_tree(key_nuts, nuts_samples, num_chains=4)
svi_tree = build_tree(key_svi, svi_posterior)
pathfinder_tree = build_tree(key_pf, pathfinder_posterior)
mclmc_tree = build_tree(key_mclmc, mclmc_samples)

rng_key, key_nuts_dct, key_svi_dct = random.split(rng_key, 3)
nuts_dct_tree = build_tree(
    key_nuts_dct, nuts_dct_samples, num_chains=4, model=univariate_model_dct
)
svi_dct_tree = build_tree(key_svi_dct, svi_dct_posterior, model=univariate_model_dct)

nuts_tree
```


![](data:image/svg+xml;base64,PHN2ZyBzdHlsZT0icG9zaXRpb246IGFic29sdXRlOyB3aWR0aDogMDsgaGVpZ2h0OiAwOyBvdmVyZmxvdzogaGlkZGVuIj4KPGRlZnM+CjxzeW1ib2wgaWQ9Imljb24tZGF0YWJhc2UiIHZpZXdib3g9IjAgMCAzMiAzMiI+CjxwYXRoIGQ9Ik0xNiAwYy04LjgzNyAwLTE2IDIuMjM5LTE2IDV2NGMwIDIuNzYxIDcuMTYzIDUgMTYgNXMxNi0yLjIzOSAxNi01di00YzAtMi43NjEtNy4xNjMtNS0xNi01eiIgLz4KPHBhdGggZD0iTTE2IDE3Yy04LjgzNyAwLTE2LTIuMjM5LTE2LTV2NmMwIDIuNzYxIDcuMTYzIDUgMTYgNXMxNi0yLjIzOSAxNi01di02YzAgMi43NjEtNy4xNjMgNS0xNiA1eiIgLz4KPHBhdGggZD0iTTE2IDI2Yy04LjgzNyAwLTE2LTIuMjM5LTE2LTV2NmMwIDIuNzYxIDcuMTYzIDUgMTYgNXMxNi0yLjIzOSAxNi01di02YzAgMi43NjEtNy4xNjMgNS0xNiA1eiIgLz4KPC9zeW1ib2w+CjxzeW1ib2wgaWQ9Imljb24tZmlsZS10ZXh0MiIgdmlld2JveD0iMCAwIDMyIDMyIj4KPHBhdGggZD0iTTI4LjY4MSA3LjE1OWMtMC42OTQtMC45NDctMS42NjItMi4wNTMtMi43MjQtMy4xMTZzLTIuMTY5LTIuMDMwLTMuMTE2LTIuNzI0Yy0xLjYxMi0xLjE4Mi0yLjM5My0xLjMxOS0yLjg0MS0xLjMxOWgtMTUuNWMtMS4zNzggMC0yLjUgMS4xMjEtMi41IDIuNXYyN2MwIDEuMzc4IDEuMTIyIDIuNSAyLjUgMi41aDIzYzEuMzc4IDAgMi41LTEuMTIyIDIuNS0yLjV2LTE5LjVjMC0wLjQ0OC0wLjEzNy0xLjIzLTEuMzE5LTIuODQxek0yNC41NDMgNS40NTdjMC45NTkgMC45NTkgMS43MTIgMS44MjUgMi4yNjggMi41NDNoLTQuODExdi00LjgxMWMwLjcxOCAwLjU1NiAxLjU4NCAxLjMwOSAyLjU0MyAyLjI2OHpNMjggMjkuNWMwIDAuMjcxLTAuMjI5IDAuNS0wLjUgMC41aC0yM2MtMC4yNzEgMC0wLjUtMC4yMjktMC41LTAuNXYtMjdjMC0wLjI3MSAwLjIyOS0wLjUgMC41LTAuNSAwIDAgMTUuNDk5LTAgMTUuNSAwdjdjMCAwLjU1MiAwLjQ0OCAxIDEgMWg3djE5LjV6IiAvPgo8cGF0aCBkPSJNMjMgMjZoLTE0Yy0wLjU1MiAwLTEtMC40NDgtMS0xczAuNDQ4LTEgMS0xaDE0YzAuNTUyIDAgMSAwLjQ0OCAxIDFzLTAuNDQ4IDEtMSAxeiIgLz4KPHBhdGggZD0iTTIzIDIyaC0xNGMtMC41NTIgMC0xLTAuNDQ4LTEtMXMwLjQ0OC0xIDEtMWgxNGMwLjU1MiAwIDEgMC40NDggMSAxcy0wLjQ0OCAxLTEgMXoiIC8+CjxwYXRoIGQ9Ik0yMyAxOGgtMTRjLTAuNTUyIDAtMS0wLjQ0OC0xLTFzMC40NDgtMSAxLTFoMTRjMC41NTIgMCAxIDAuNDQ4IDEgMXMtMC40NDggMS0xIDF6IiAvPgo8L3N5bWJvbD4KPC9kZWZzPgo8L3N2Zz4=) <style>/* CSS stylesheet for displaying xarray objects in notebooks */

:root {
  --xr-font-color0: var(
    --jp-content-font-color0,
    var(--pst-color-text-base rgba(0, 0, 0, 1))
  );
  --xr-font-color2: var(
    --jp-content-font-color2,
    var(--pst-color-text-base, rgba(0, 0, 0, 0.54))
  );
  --xr-font-color3: var(
    --jp-content-font-color3,
    var(--pst-color-text-base, rgba(0, 0, 0, 0.38))
  );
  --xr-border-color: var(
    --jp-border-color2,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 10))
  );
  --xr-disabled-color: var(
    --jp-layout-color3,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 40))
  );
  --xr-background-color: var(
    --jp-layout-color0,
    var(--pst-color-on-background, white)
  );
  --xr-background-color-row-even: var(
    --jp-layout-color1,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 5))
  );
  --xr-background-color-row-odd: var(
    --jp-layout-color2,
    hsl(from var(--pst-color-on-background, white) h s calc(l - 15))
  );
}

html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --xr-font-color0: var(
    --jp-content-font-color0,
    var(--pst-color-text-base, rgba(255, 255, 255, 1))
  );
  --xr-font-color2: var(
    --jp-content-font-color2,
    var(--pst-color-text-base, rgba(255, 255, 255, 0.54))
  );
  --xr-font-color3: var(
    --jp-content-font-color3,
    var(--pst-color-text-base, rgba(255, 255, 255, 0.38))
  );
  --xr-border-color: var(
    --jp-border-color2,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 10))
  );
  --xr-disabled-color: var(
    --jp-layout-color3,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 40))
  );
  --xr-background-color: var(
    --jp-layout-color0,
    var(--pst-color-on-background, #111111)
  );
  --xr-background-color-row-even: var(
    --jp-layout-color1,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 5))
  );
  --xr-background-color-row-odd: var(
    --jp-layout-color2,
    hsl(from var(--pst-color-on-background, #111111) h s calc(l + 15))
  );
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
  line-height: 1.6;
  padding-bottom: 4px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
}

.xr-header {
  border-bottom: solid 1px var(--xr-border-color);
  margin-bottom: 4px;
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-obj-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type,
.xr-group-box-contents > label {
  color: var(--xr-font-color2);
  display: block;
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
  margin-block-start: 0;
  margin-block-end: 0;
}

.xr-section-item {
  display: contents;
}

.xr-section-item > input,
.xr-group-box-contents > input,
.xr-array-wrap > input {
  display: block;
  opacity: 0;
  height: 0;
  margin: 0;
}

.xr-section-item > input + label,
.xr-var-item > input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item > input:enabled + label,
.xr-var-item > input:enabled + label,
.xr-array-wrap > input:enabled + label,
.xr-group-box-contents > input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item > input:focus-visible + label,
.xr-var-item > input:focus-visible + label,
.xr-array-wrap > input:focus-visible + label,
.xr-group-box-contents > input:focus-visible + label {
  outline: auto;
}

.xr-section-item > input:enabled + label:hover,
.xr-var-item > input:enabled + label:hover,
.xr-array-wrap > input:enabled + label:hover,
.xr-group-box-contents > input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
  white-space: nowrap;
}

.xr-section-summary > em {
  font-weight: normal;
}

.xr-span-grid {
  grid-column-end: -1;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.3em;
}

.xr-group-box-contents > input:checked + label > span {
  display: inline-block;
  padding-left: 0.6em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: "►";
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: "▼";
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details,
.xr-group-box-contents > label {
  padding-top: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  grid-column: 1 / -1;
  margin-top: 4px;
  margin-bottom: 5px;
}

.xr-section-summary-in ~ .xr-section-details {
  display: none;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-children {
  display: inline-grid;
  grid-template-columns: 100%;
  grid-column: 1 / -1;
  padding-top: 4px;
}

.xr-group-box {
  display: inline-grid;
  grid-template-columns: 0px 30px auto;
}

.xr-group-box-vline {
  grid-column-start: 1;
  border-right: 0.2em solid;
  border-color: var(--xr-border-color);
  width: 0px;
}

.xr-group-box-hline {
  grid-column-start: 2;
  grid-row-start: 1;
  height: 1em;
  width: 26px;
  border-bottom: 0.2em solid;
  border-color: var(--xr-border-color);
}

.xr-group-box-contents {
  grid-column-start: 3;
  padding-bottom: 4px;
}

.xr-group-box-contents > label::before {
  content: "📂";
  padding-right: 0.3em;
}

.xr-group-box-contents > input:checked + label::before {
  content: "📁";
}

.xr-group-box-contents > input:checked + label {
  padding-bottom: 0px;
}

.xr-group-box-contents > input:checked ~ .xr-sections {
  display: none;
}

.xr-group-box-contents > input + label > span {
  display: none;
}

.xr-group-box-ellipsis {
  font-size: 1.4em;
  font-weight: 900;
  color: var(--xr-font-color2);
  letter-spacing: 0.15em;
  cursor: default;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: "(";
}

.xr-dim-list:after {
  content: ")";
}

.xr-dim-list li:not(:last-child):after {
  content: ",";
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  border-color: var(--xr-background-color-row-odd);
  margin-bottom: 0;
  padding-top: 2px;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
  border-color: var(--xr-background-color-row-even);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  border-top: 2px dotted var(--xr-background-color);
  padding-bottom: 20px !important;
  padding-top: 10px !important;
}

.xr-var-attrs-in + label,
.xr-var-data-in + label,
.xr-index-data-in + label {
  padding: 0 1px;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-data > pre,
.xr-index-data > pre,
.xr-var-data > table > tbody > tr {
  background-color: transparent !important;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}

.xr-var-attrs-in:checked + label > .xr-icon-file-text2,
.xr-var-data-in:checked + label > .xr-icon-database,
.xr-index-data-in:checked + label > .xr-icon-database {
  color: var(--xr-font-color0);
  filter: drop-shadow(1px 1px 5px var(--xr-font-color2));
  stroke-width: 0.8px;
}
</style>

``` xr-text-repr-fallback
<xarray.DataTree>
Group: /
│   Attributes:
│       inference_library:  numpyro
│       creation_library:   numpyro_forecast
│       sample_dims:        ['chain', 'draw']
├── Group: /posterior
│       Dimensions:                 (chain: 4, draw: 1000, time: 417, drift_dim_0: 1,
│                                    drift_decentered_dim_0: 417,
│                                    drift_decentered_dim_1: 1, weight_dim_0: 52)
│       Coordinates:
│         * chain                   (chain) int64 32B 0 1 2 3
│         * draw                    (draw) int64 8kB 0 1 2 3 4 5 ... 995 996 997 998 999
│         * time                    (time) int64 3kB 0 1 2 3 4 5 ... 412 413 414 415 416
│         * drift_dim_0             (drift_dim_0) int64 8B 0
│         * drift_decentered_dim_0  (drift_decentered_dim_0) int64 3kB 0 1 2 ... 415 416
│         * drift_decentered_dim_1  (drift_decentered_dim_1) int64 8B 0
│         * weight_dim_0            (weight_dim_0) int64 416B 0 1 2 3 4 ... 48 49 50 51
│       Data variables:
│           bias                    (chain, draw) float32 16kB 14.52 14.52 ... 14.52
│           centered                (chain, draw) float32 16kB 0.2751 0.2891 ... 0.2175
│           drift                   (chain, draw, time, drift_dim_0) float32 7MB -0.0...
│           drift_decentered        (chain, draw, drift_decentered_dim_0, drift_decentered_dim_1) float32 7MB ...
│           drift_scale             (chain, draw) float32 16kB 0.004127 ... 0.003585
│           nu                      (chain, draw) float32 16kB 1.92 2.022 ... 1.758 1.63
│           sigma                   (chain, draw) float32 16kB 0.01877 ... 0.02009
│           weight                  (chain, draw, weight_dim_0) float32 832kB 5.609e-...
│       Attributes:
│           created_at:                 2026-09-25T16:34:53.605231+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
├── Group: /posterior_predictive
│       Dimensions:  (chain: 4, draw: 1000, time: 417)
│       Coordinates:
│         * chain    (chain) int64 32B 0 1 2 3
│         * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│         * time     (time) int64 3kB 0 1 2 3 4 5 6 7 ... 410 411 412 413 414 415 416
│           obs_dim  int64 8B 0
│       Data variables:
│           obs      (chain, draw, time) float32 7MB 14.39 14.47 14.44 ... 14.68 14.25
│       Attributes:
│           created_at:                 2026-09-25T16:34:54.230682+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
├── Group: /observed_data
│       Dimensions:  (time: 417)
│       Coordinates:
│         * time     (time) int64 3kB 0 1 2 3 4 5 6 7 ... 410 411 412 413 414 415 416
│           obs_dim  int64 8B 0
│       Data variables:
│           obs      (time) float32 2kB 14.41 14.45 14.42 14.53 ... 14.71 14.65 14.04
│       Attributes:
│           created_at:                 2026-09-25T16:34:54.230946+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /constant_data
│       Dimensions:        (time: 417, covariate_dim: 52)
│       Coordinates:
│         * time           (time) int64 3kB 0 1 2 3 4 5 6 ... 411 412 413 414 415 416
│         * covariate_dim  (covariate_dim) int64 416B 0 1 2 3 4 5 ... 46 47 48 49 50 51
│       Data variables:
│           covariates     (time, covariate_dim) float32 87kB 0.0 0.0 ... -0.2376
│           week           (time) float64 3kB 0.0 1.0 2.0 3.0 ... 414.0 415.0 416.0
│       Attributes:
│           created_at:                 2026-09-25T16:34:54.231125+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /predictions
│       Dimensions:  (chain: 4, draw: 1000, time: 52)
│       Coordinates:
│         * chain    (chain) int64 32B 0 1 2 3
│         * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│         * time     (time) int64 416B 417 418 419 420 421 422 ... 464 465 466 467 468
│           obs_dim  int64 8B 0
│       Data variables:
│           obs      (chain, draw, time) float32 832kB 14.5 14.66 14.59 ... 14.69 14.45
│       Attributes:
│           created_at:                 2026-09-25T16:34:54.765667+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
└── Group: /predictions_constant_data
        Dimensions:        (time: 52, covariate_dim: 52)
        Coordinates:
          * time           (time) int64 416B 417 418 419 420 421 ... 464 465 466 467 468
          * covariate_dim  (covariate_dim) int64 416B 0 1 2 3 4 5 ... 46 47 48 49 50 51
        Data variables:
            covariates     (time, covariate_dim) float32 11kB -0.05158 -0.103 ... 0.3138
            week           (time) float64 416B 417.0 418.0 419.0 ... 466.0 467.0 468.0
        Attributes:
            created_at:                 2026-09-25T16:34:54.765918+00:00
            creation_library:           ArviZ
            creation_library_version:   1.2.0
            creation_library_language:  Python
            sample_dims:                []
```


xarray.DataTree


/posterior(20)

Dimensions:


- chain: 4
- draw: 1000
- time: 417
- drift_dim_0: 1
- drift_decentered_dim_0: 417
- drift_decentered_dim_1: 1
- weight_dim_0: 52


Coordinates: (7)


chain


(chain)


int64


0 1 2 3


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0, 1, 2, 3])


draw


(draw)


int64


0 1 2 3 4 5 ... 995 996 997 998 999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))


time


(time)


int64


0 1 2 3 4 5 ... 412 413 414 415 416


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 414, 415, 416], shape=(417,))


drift_dim_0


(drift_dim_0)


int64


0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0])


drift_decentered_dim_0


(drift_decentered_dim_0)


int64


0 1 2 3 4 5 ... 412 413 414 415 416


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 414, 415, 416], shape=(417,))


drift_decentered_dim_1


(drift_decentered_dim_1)


int64


0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0])


weight_dim_0


(weight_dim_0)


int64


0 1 2 3 4 5 6 ... 46 47 48 49 50 51


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51])


Data variables: (8)


bias


(chain, draw)


float32


14.52 14.52 14.52 ... 14.52 14.52


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[14.522769, 14.522823, 14.520166, ..., 14.520412, 14.520885,14.521676],[14.499089, 14.52957 , 14.534934, ..., 14.506197, 14.50647 ,14.506622],[14.506336, 14.507917, 14.515212, ..., 14.500494, 14.517703,14.543724],[14.52655 , 14.513382, 14.520092, ..., 14.50972 , 14.515064,14.522751]], shape=(4, 1000), dtype=float32)


centered


(chain, draw)


float32


0.2751 0.2891 ... 0.1912 0.2175


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.2750971 , 0.28909233, 0.28737545, ..., 0.35976836, 0.39899406,0.39417627],[0.03827449, 0.05611218, 0.03514542, ..., 0.13271827, 0.13168453,0.13051099],[0.00952784, 0.00094242, 0.00330084, ..., 0.08791041, 0.08305194,0.09896431],[0.25255182, 0.23863766, 0.2638188 , ..., 0.19323346, 0.19124207,0.2174563 ]], shape=(4, 1000), dtype=float32)


drift


(chain, draw, time, drift_dim_0)


float32


-0.001281 -0.008677 ... 0.002467


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[-1.28113339e-03],[-8.67695268e-03],[-2.56780698e-03],...,[ 5.72073739e-03],[-6.34413678e-04],[ 1.89317472e-03]],[[-4.22187662e-03],[ 3.51827289e-03],[-6.53890078e-04],...,[-4.48802114e-03],[-2.57837004e-04],[-2.21994147e-03]],[[ 5.08349529e-03],[-7.02615036e-03],[-3.15580619e-05],...,......,[ 4.94974246e-03],[-2.41020834e-03],[-4.35515190e-04]],[[ 3.15782148e-03],[-4.58589615e-03],[-2.26436835e-03],...,[-4.21552127e-03],[ 9.56872944e-04],[-2.96346057e-04]],[[ 2.72325845e-03],[ 3.73887876e-03],[ 9.69161629e-04],...,[ 3.14563792e-03],[ 3.27162468e-03],[ 2.46736407e-03]]]], shape=(4, 1000, 417, 1), dtype=float32)


drift_decentered


(chain, draw, drift_decentered_dim_0, drift_decentered_dim_1)


float32


-0.06856 -0.4643 ... 0.2682 0.2023


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[-6.8557315e-02],[-4.6432993e-01],[-1.3741110e-01],...,[ 3.0613393e-01],[-3.3949390e-02],[ 1.0130949e-01]],[[-2.3094329e-01],[ 1.9245507e-01],[-3.5768818e-02],...,[-2.4550183e-01],[-1.4104090e-02],[-1.2143429e-01]],[[ 2.7954641e-01],[-3.8637492e-01],[-1.7354088e-03],...,......,[ 4.0571499e-01],[-1.9755729e-01],[-3.5697825e-02]],[[ 2.7307606e-01],[-3.9657038e-01],[-1.9581373e-01],...,[-3.6454180e-01],[ 8.2746632e-02],[-2.5626849e-02]],[[ 2.2325416e-01],[ 3.0651525e-01],[ 7.9452381e-02],...,[ 2.5788105e-01],[ 2.6820952e-01],[ 2.0227580e-01]]]], shape=(4, 1000, 417, 1), dtype=float32)


drift_scale


(chain, draw)


float32


0.004127 0.003591 ... 0.003585


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.00412666, 0.00359119, 0.00361337, ..., 0.0049215 , 0.00510939,0.00513298],[0.00496192, 0.00530479, 0.00454442, ..., 0.00399491, 0.00400425,0.00401533],[0.00352766, 0.00529676, 0.00463715, ..., 0.00440683, 0.00500706,0.00497631],[0.00408733, 0.00333039, 0.00587115, ..., 0.00424634, 0.00402808,0.00358509]], shape=(4, 1000), dtype=float32)


nu


(chain, draw)


float32


1.92 2.022 1.685 ... 1.758 1.63


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[1.9200256, 2.021819 , 1.6846199, ..., 1.5598605, 1.5857873,1.5822036],[1.4172397, 1.3539988, 1.2175179, ..., 1.8145773, 1.8183025,1.8215802],[1.8014345, 1.6629344, 1.7579145, ..., 1.5171518, 1.490285 ,1.4078312],[1.5656185, 1.4457428, 2.0473208, ..., 1.6257255, 1.7579772,1.6296672]], shape=(4, 1000), dtype=float32)


sigma


(chain, draw)


float32


0.01877 0.01865 ... 0.02062 0.02009


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.01877156, 0.01864916, 0.02021584, ..., 0.01456721, 0.01449602,0.01463227],[0.01364388, 0.01613893, 0.01193974, ..., 0.01613054, 0.01616515,0.0160995 ],[0.02000512, 0.01803317, 0.02076497, ..., 0.01753979, 0.01697182,0.01484893],[0.01735437, 0.01619705, 0.01908069, ..., 0.01954363, 0.02062156,0.02009047]], shape=(4, 1000), dtype=float32)


weight


(chain, draw, weight_dim_0)


float32


5.609e-05 0.01054 ... -0.004804


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 5.60902990e-05,  1.05387019e-02,  2.02174876e-02, ...,2.67900876e-03, -1.63362350e-03, -1.77067390e-03],[-4.40761121e-03,  1.82757489e-02,  1.99259501e-02, ...,-2.00856826e-03,  3.54994809e-05,  8.38923806e-05],[-6.25193352e-03,  1.66887268e-02,  2.22379081e-02, ...,-1.60070788e-03, -1.26389530e-03, -1.21591403e-03],...,[-5.05910628e-03,  1.55774867e-02,  1.81173831e-02, ...,6.12005708e-04,  3.25440941e-03, -1.66183023e-03],[-5.94916008e-03,  1.43654319e-02,  1.81213133e-02, ...,1.42741017e-03,  3.90699878e-03, -1.36421679e-03],[-6.50457758e-03,  1.47754457e-02,  1.79222710e-02, ...,1.13217556e-03,  3.61243542e-03, -1.07539131e-03]],[[ 1.49329286e-03,  1.84352808e-02,  2.07221210e-02, ...,3.30449175e-03,  3.14481417e-03, -2.58685177e-04],[-1.20504349e-02,  1.23043088e-02,  2.23089289e-02, ...,-1.43004442e-03,  1.15904619e-03, -3.19378823e-03],[-4.60186228e-03,  1.25086494e-02,  1.92744192e-02, ...,4.26429138e-03, -1.00374210e-03, -1.22730632e-03],...1.16043119e-03,  6.83947990e-04, -3.54305864e-03],[-4.97591496e-03,  1.47501286e-02,  1.67379752e-02, ...,1.99005776e-03,  2.59308377e-04, -1.56565558e-03],[-6.25985488e-03,  1.37124583e-02,  2.06016153e-02, ...,1.48496867e-04, -2.13907124e-03, -1.70285115e-03]],[[-3.32618388e-03,  1.30844293e-02,  2.10040770e-02, ...,1.44521426e-03,  3.74361477e-03, -1.77029148e-03],[-7.85565283e-03,  1.47795770e-02,  2.27643792e-02, ...,1.07048289e-03, -9.64646097e-05, -1.43316819e-03],[-7.64129590e-03,  1.25230625e-02,  2.06889361e-02, ...,4.94041620e-03,  1.17838103e-03, -8.14449694e-03],...,[-1.08907940e-02,  1.43212629e-02,  1.91459041e-02, ...,-1.52057456e-03,  1.85902102e-03, -1.34972704e-03],[-5.67942020e-03,  1.04669137e-02,  2.38878839e-02, ...,1.95006107e-03,  7.24634156e-04, -6.33124169e-03],[-7.55386753e-03,  2.01332290e-02,  1.88305620e-02, ...,-6.40851853e-04,  5.85369125e-04, -4.80440585e-03]]],shape=(4, 1000, 52), dtype=float32)


Attributes: (5)


created_at :  
2026-09-25T16:34:53.605231+00:00

creation_library :  
ArviZ

creation_library_version :  
1.2.0

creation_library_language :  
Python

sample_dims :  
\['chain', 'draw'\]


/posterior_predictive(10)

Dimensions:


- chain: 4
- draw: 1000
- time: 417


Coordinates: (4)


chain


(chain)


int64


0 1 2 3


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0, 1, 2, 3])


draw


(draw)


int64


0 1 2 3 4 5 ... 995 996 997 998 999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))


time


(time)


int64


0 1 2 3 4 5 ... 412 413 414 415 416


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 414, 415, 416], shape=(417,))


obs_dim


()


int64


0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(0)


Data variables: (1)


obs


(chain, draw, time)


float32


14.39 14.47 14.44 ... 14.68 14.25


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[14.3876095, 14.465941 , 14.436863 , ..., 14.710745 ,14.641898 , 14.297402 ],[14.393753 , 14.329216 , 14.447812 , ..., 14.709267 ,14.673056 , 14.231346 ],[14.359555 , 14.503365 , 14.481538 , ..., 14.724676 ,14.634289 , 14.204009 ],...,[14.406814 , 14.445917 , 14.421692 , ..., 14.677764 ,14.643    , 14.329033 ],[14.445397 , 14.523464 , 14.422801 , ..., 14.665564 ,14.691609 , 14.249859 ],[14.399201 , 14.539675 , 14.414224 , ..., 14.678984 ,14.664358 , 14.3237915]],[[14.525057 , 14.465582 , 14.433795 , ..., 14.662696 ,14.614962 , 14.223475 ],[14.378104 , 14.535701 , 14.429616 , ..., 14.6709   ,14.678733 , 14.248827 ],[14.381308 , 14.433356 , 14.433701 , ..., 14.680571 ,14.676933 , 14.220161 ],...[14.38232  , 14.457669 , 14.439581 , ..., 14.667486 ,14.658976 , 14.207481 ],[14.423708 , 14.449862 , 14.434088 , ..., 14.698764 ,14.704353 , 14.223276 ],[14.350008 , 14.448785 , 14.3399515, ..., 14.642634 ,15.004046 , 14.282252 ]],[[14.46995  , 14.491674 , 14.408989 , ..., 14.625055 ,14.667366 , 14.221833 ],[14.365867 , 14.41265  , 14.416937 , ..., 14.565806 ,14.634846 , 14.185261 ],[14.405733 , 14.551271 , 14.4295635, ..., 14.652109 ,14.647446 , 14.191732 ],...,[14.414277 , 14.452101 , 14.446161 , ..., 14.660893 ,14.657175 , 14.20624  ],[14.248542 , 14.474654 , 14.363992 , ..., 14.7876625,14.653933 , 14.219271 ],[14.377328 , 14.427375 , 14.349741 , ..., 14.694173 ,14.681542 , 14.254318 ]]], shape=(4, 1000, 417), dtype=float32)


Attributes: (5)


created_at :  
2026-09-25T16:34:54.230682+00:00

creation_library :  
ArviZ

creation_library_version :  
1.2.0

creation_library_language :  
Python

sample_dims :  
\['chain', 'draw'\]


/observed_data(8)

Dimensions:


- time: 417


Coordinates: (2)


time


(time)


int64


0 1 2 3 4 5 ... 412 413 414 415 416


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 414, 415, 416], shape=(417,))


obs_dim


()


int64


0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(0)


Data variables: (1)


obs


(time)


float32


14.41 14.45 14.42 ... 14.65 14.04


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([14.409758 , 14.452444 , 14.424588 , 14.529594 , 14.516827 ,14.536647 , 14.499719 , 14.4457035, 14.527379 , 14.518174 ,14.519266 , 14.47356  , 14.518399 , 14.548611 , 14.561788 ,14.541207 , 14.522805 , 14.558051 , 14.555797 , 14.568521 ,14.525397 , 14.405628 , 14.540501 , 14.553799 , 14.578104 ,14.609343 , 14.459167 , 14.550219 , 14.573015 , 14.563639 ,14.562765 , 14.5578785, 14.563554 , 14.593687 , 14.603224 ,14.490224 , 14.611021 , 14.611185 , 14.620996 , 14.626418 ,14.458051 , 14.415224 , 14.610413 , 14.592795 , 14.480232 ,14.593957 , 14.303107 , 14.579532 , 14.612829 , 14.589255 ,14.540648 , 14.176248 , 14.456273 , 14.5345335, 14.47762  ,14.570745 , 14.594236 , 14.585067 , 14.618121 , 14.574687 ,14.586572 , 14.613272 , 14.572606 , 14.588731 , 14.553821 ,14.5904255, 14.563532 , 14.630408 , 14.615726 , 14.62211  ,14.620018 , 14.618416 , 14.624354 , 14.512324 , 14.596832 ,14.567822 , 14.646883 , 14.707247 , 14.48131  , 14.611531 ,14.633596 , 14.633275 , 14.637484 , 14.629257 , 14.650006 ,14.651122 , 14.643528 , 14.567774 , 14.684897 , 14.69442  ,14.6914215, 14.719228 , 14.768465 , 14.693457 , 14.704176 ,14.74647  , 14.589776 , 14.6485615, 14.389623 , 14.626623 ,...14.597787 , 14.71515  , 14.708497 , 14.736562 , 14.680824 ,14.651695 , 14.675075 , 14.653225 , 14.702647 , 14.725265 ,14.718654 , 14.693803 , 14.711188 , 14.700039 , 14.549411 ,14.695086 , 14.724124 , 14.708156 , 14.764582 , 14.474372 ,14.704191 , 14.723518 , 14.716631 , 14.713915 , 14.701516 ,14.693797 , 14.703767 , 14.690833 , 14.546381 , 14.709186 ,14.721344 , 14.725226 , 14.729357 , 14.716863 , 14.718071 ,14.714689 , 14.698227 , 14.706976 , 14.716086 , 14.366756 ,14.698102 , 14.730522 , 14.7270155, 14.644983 , 14.120207 ,14.40623  , 14.6119   , 14.546069 , 14.697077 , 14.683383 ,14.685147 , 14.677837 , 14.569544 , 14.668326 , 14.667568 ,14.6571865, 14.660634 , 14.7006445, 14.631047 , 14.679982 ,14.703481 , 14.698523 , 14.701732 , 14.696913 , 14.689038 ,14.693262 , 14.552782 , 14.688538 , 14.706831 , 14.689468 ,14.773753 , 14.48263  , 14.706127 , 14.70885  , 14.714022 ,14.68153  , 14.675425 , 14.6790285, 14.703018 , 14.654665 ,14.550027 , 14.735775 , 14.730156 , 14.714912 , 14.726917 ,14.707603 , 14.7097645, 14.720431 , 14.708797 , 14.707592 ,14.594886 , 14.177667 , 14.655781 , 14.69477  , 14.711192 ,14.645604 , 14.038745 ], dtype=float32)


Attributes: (5)


created_at :  
2026-09-25T16:34:54.230946+00:00

creation_library :  
ArviZ

creation_library_version :  
1.2.0

creation_library_language :  
Python

sample_dims :  
\[\]


/constant_data(9)

Dimensions:


- time: 417
- covariate_dim: 52


Coordinates: (2)


time


(time)


int64


0 1 2 3 4 5 ... 412 413 414 415 416


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 414, 415, 416], shape=(417,))


covariate_dim


(covariate_dim)


int64


0 1 2 3 4 5 6 ... 46 47 48 49 50 51


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51])


Data variables: (2)


covariates


(time, covariate_dim)


float32


0.0 0.0 0.0 ... -0.4002 -0.2376


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[ 0.        ,  0.        ,  0.        , ...,  1.        ,1.        ,  1.        ],[ 0.12012617,  0.23851258,  0.35344467, ..., -0.968519  ,-0.9914097 , -0.9999422 ],[ 0.23851258,  0.46325794,  0.66126347, ...,  0.876058  ,0.9657865 ,  0.99976885],...,[-0.4012302 , -0.7350354 , -0.9453188 , ..., -0.8852392 ,-0.6241668 , -0.25839978],[-0.28829038, -0.5521009 , -0.7690279 , ...,  0.7415851 ,0.51668113,  0.24789807],[-0.17117523, -0.33729756, -0.49347657, ..., -0.55114007,-0.40021673, -0.23760432]], shape=(417, 52), dtype=float32)


week


(time)


float64


0.0 1.0 2.0 ... 414.0 415.0 416.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,22.,  23.,  24.,  25.,  26.,  27.,  28.,  29.,  30.,  31.,  32.,33.,  34.,  35.,  36.,  37.,  38.,  39.,  40.,  41.,  42.,  43.,44.,  45.,  46.,  47.,  48.,  49.,  50.,  51.,  52.,  53.,  54.,55.,  56.,  57.,  58.,  59.,  60.,  61.,  62.,  63.,  64.,  65.,66.,  67.,  68.,  69.,  70.,  71.,  72.,  73.,  74.,  75.,  76.,77.,  78.,  79.,  80.,  81.,  82.,  83.,  84.,  85.,  86.,  87.,88.,  89.,  90.,  91.,  92.,  93.,  94.,  95.,  96.,  97.,  98.,99., 100., 101., 102., 103., 104., 105., 106., 107., 108., 109.,110., 111., 112., 113., 114., 115., 116., 117., 118., 119., 120.,121., 122., 123., 124., 125., 126., 127., 128., 129., 130., 131.,132., 133., 134., 135., 136., 137., 138., 139., 140., 141., 142.,143., 144., 145., 146., 147., 148., 149., 150., 151., 152., 153.,154., 155., 156., 157., 158., 159., 160., 161., 162., 163., 164.,165., 166., 167., 168., 169., 170., 171., 172., 173., 174., 175.,176., 177., 178., 179., 180., 181., 182., 183., 184., 185., 186.,187., 188., 189., 190., 191., 192., 193., 194., 195., 196., 197.,198., 199., 200., 201., 202., 203., 204., 205., 206., 207., 208.,209., 210., 211., 212., 213., 214., 215., 216., 217., 218., 219.,220., 221., 222., 223., 224., 225., 226., 227., 228., 229., 230.,231., 232., 233., 234., 235., 236., 237., 238., 239., 240., 241.,242., 243., 244., 245., 246., 247., 248., 249., 250., 251., 252.,253., 254., 255., 256., 257., 258., 259., 260., 261., 262., 263.,264., 265., 266., 267., 268., 269., 270., 271., 272., 273., 274.,275., 276., 277., 278., 279., 280., 281., 282., 283., 284., 285.,286., 287., 288., 289., 290., 291., 292., 293., 294., 295., 296.,297., 298., 299., 300., 301., 302., 303., 304., 305., 306., 307.,308., 309., 310., 311., 312., 313., 314., 315., 316., 317., 318.,319., 320., 321., 322., 323., 324., 325., 326., 327., 328., 329.,330., 331., 332., 333., 334., 335., 336., 337., 338., 339., 340.,341., 342., 343., 344., 345., 346., 347., 348., 349., 350., 351.,352., 353., 354., 355., 356., 357., 358., 359., 360., 361., 362.,363., 364., 365., 366., 367., 368., 369., 370., 371., 372., 373.,374., 375., 376., 377., 378., 379., 380., 381., 382., 383., 384.,385., 386., 387., 388., 389., 390., 391., 392., 393., 394., 395.,396., 397., 398., 399., 400., 401., 402., 403., 404., 405., 406.,407., 408., 409., 410., 411., 412., 413., 414., 415., 416.])


Attributes: (5)


created_at :  
2026-09-25T16:34:54.231125+00:00

creation_library :  
ArviZ

creation_library_version :  
1.2.0

creation_library_language :  
Python

sample_dims :  
\[\]


/predictions(10)

Dimensions:


- chain: 4
- draw: 1000
- time: 52


Coordinates: (4)


chain


(chain)


int64


0 1 2 3


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0, 1, 2, 3])


draw


(draw)


int64


0 1 2 3 4 5 ... 995 996 997 998 999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))


time


(time)


int64


417 418 419 420 ... 465 466 467 468


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430,431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444,445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458,459, 460, 461, 462, 463, 464, 465, 466, 467, 468])


obs_dim


()


int64


0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(0)


Data variables: (1)


obs


(chain, draw, time)


float32


14.5 14.66 14.59 ... 14.69 14.45


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[14.503907 , 14.659236 , 14.588355 , ..., 14.692155 ,14.684695 , 14.262318 ],[14.368981 , 14.57186  , 14.542863 , ..., 14.668153 ,14.73518  , 14.290753 ],[14.399597 , 14.563817 , 14.554544 , ..., 14.635212 ,14.627603 , 14.258506 ],...,[14.357595 , 14.587165 , 14.551408 , ..., 14.73449  ,14.762737 , 14.364089 ],[14.366856 , 14.613092 , 14.454675 , ..., 14.619513 ,14.688612 , 14.286854 ],[14.41882  , 14.626106 , 14.561189 , ..., 14.622261 ,14.690737 , 14.278607 ]],[[14.378604 , 14.5820675, 14.617068 , ..., 14.68907  ,14.611128 , 14.272454 ],[14.371564 , 14.561808 , 14.590638 , ..., 14.630689 ,14.645187 , 14.261362 ],[14.406833 , 14.582171 , 14.557283 , ..., 14.697556 ,14.738623 , 14.385211 ],...[14.412874 , 14.596779 , 14.661889 , ..., 14.607026 ,14.581307 , 14.23718  ],[14.508021 , 14.631997 , 14.600297 , ..., 14.690496 ,14.740638 , 14.280039 ],[14.421268 , 14.615503 , 14.549985 , ..., 14.683498 ,13.560177 , 14.273614 ]],[[14.424391 , 14.63654  , 14.573101 , ..., 14.637129 ,14.694934 , 14.289777 ],[14.381046 , 14.5393505, 14.584018 , ..., 14.715045 ,14.733766 , 14.365591 ],[14.312335 , 14.59347  , 14.57439  , ..., 14.720205 ,14.764447 , 14.329182 ],...,[14.419512 , 14.630643 , 14.580336 , ..., 14.659102 ,14.6970215, 14.238585 ],[14.380308 , 14.598778 , 14.652709 , ..., 14.693976 ,14.685219 , 14.273905 ],[14.411222 , 14.618261 , 14.560053 , ..., 14.696238 ,14.690015 , 14.449202 ]]], shape=(4, 1000, 52), dtype=float32)


Attributes: (5)


created_at :  
2026-09-25T16:34:54.765667+00:00

creation_library :  
ArviZ

creation_library_version :  
1.2.0

creation_library_language :  
Python

sample_dims :  
\['chain', 'draw'\]


/predictions_constant_data(9)

Dimensions:


- time: 52
- covariate_dim: 52


Coordinates: (2)


time


(time)


int64


417 418 419 420 ... 465 466 467 468


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430,431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444,445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458,459, 460, 461, 462, 463, 464, 465, 466, 467, 468])


covariate_dim


(covariate_dim)


int64


0 1 2 3 4 5 6 ... 46 47 48 49 50 51


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51])


Data variables: (2)


covariates


(time, covariate_dim)


float32


-0.05158 -0.103 ... 0.1256 0.3138


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[-0.05158474, -0.10303211, -0.15421268, ...,  0.3260781 ,0.27698866,  0.22704606],[ 0.06875662,  0.13718781,  0.20495847, ..., -0.08048676,-0.14888641, -0.21658021],[ 0.18810216,  0.36948887,  0.53767157, ..., -0.17017189,0.01822436,  0.20608942],...,[-0.42083025, -0.7635034 , -0.9643768 , ..., -0.5404368 ,-0.13612227,  0.29336202],[-0.3088138 , -0.58743954, -0.8086423 , ...,  0.3138603 ,0.00532949, -0.30372226],[-0.19232132, -0.37746212, -0.5485164 , ..., -0.06762641,0.12555493,  0.3138149 ]], shape=(52, 52), dtype=float32)


week


(time)


float64


417.0 418.0 419.0 ... 467.0 468.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([417., 418., 419., 420., 421., 422., 423., 424., 425., 426., 427.,428., 429., 430., 431., 432., 433., 434., 435., 436., 437., 438.,439., 440., 441., 442., 443., 444., 445., 446., 447., 448., 449.,450., 451., 452., 453., 454., 455., 456., 457., 458., 459., 460.,461., 462., 463., 464., 465., 466., 467., 468.])


Attributes: (5)


created_at :  
2026-09-25T16:34:54.765918+00:00

creation_library :  
ArviZ

creation_library_version :  
1.2.0

creation_library_language :  
Python

sample_dims :  
\[\]


Attributes: (3)


inference_library :  
numpyro

creation_library :  
numpyro_forecast

sample_dims :  
\['chain', 'draw'\]


## NUTS diagnostics

Because the NUTS tree keeps its `4` chains, the standard MCMC diagnostics apply directly to it: `az.summary` reports posterior summaries, effective sample sizes, and \hat{R} for the scalar parameters. Values of \hat{R} close to `1` indicate that the chains mixed well.


    In [20]:


``` python
az.summary(nuts_tree, var_names=["bias", "drift_scale", "nu", "sigma", "centered"])
```


|  | mean | sd | eti89_lb | eti89_ub | ess_bulk | ess_tail | r_hat | mcse_mean | mcse_sd |
|----|----|----|----|----|----|----|----|----|----|
| bias | 14.5136 | 0.0103 | 14 | 15 | 488 | 456 | 1.02 | 0.00047 | 0.00035 |
| drift_scale | 0.00441 | 0.00071 | 0.0034 | 0.0056 | 271 | 432 | 1.02 | 4.2e-05 | 3.3e-05 |
| nu | 1.57 | 0.22 | 1.3 | 1.9 | 188 | 511 | 1.02 | 0.016 | 0.014 |
| sigma | 0.0171 | 0.0021 | 0.014 | 0.021 | 87 | 424 | 1.04 | 0.00022 | 0.00016 |
| centered | 0.2 | 0.16 | 0.02 | 0.5 | 5 | 10 | 1.82 | 0.076 | 0.042 |


The scalar parameters that shape the forecast (`bias`, `drift_scale`, `nu`, `sigma`) mix well. The exception is `centered`, and it is worth understanding why: this site only selects the drift's parameterization, so the joint density over the data is the same for every value of `centered` and its exact posterior equals its \text{Uniform}(0, 1) prior. NUTS explores that flat direction slowly, which is exactly what the large \hat{R} flags, but none of it leaks into the forecasts, which consume only the implied drift.


# CRPS on train and test

We score each engine with the **continuous ranked probability score** (CRPS), a proper scoring rule that compares a single observed value against the whole forecast distribution, rewarding forecasts that are both sharp and calibrated (lower is better). The in-sample score comes from the `posterior_predictive` group and the out-of-sample score from the `predictions` group, so the metrics are computed from the very same draws the plots below display.


    In [21]:


``` python
def compute_crps(tree: xr.DataTree) -> dict[str, float]:
    """Score the in-sample and forecast draws in ``tree`` against the observed data."""
    pred_train = jnp.asarray(tree["posterior_predictive"]["obs"].values).reshape(-1, T1 - T0)
    pred_test = jnp.asarray(tree["predictions"]["obs"].values).reshape(-1, T2 - T1)
    return {
        "train": float(eval_crps(pred_train, y_train[:, 0])),
        "test": float(eval_crps(pred_test, y_test[:, 0])),
    }


crps_results = {
    "NUTS": compute_crps(nuts_tree),
    "SVI": compute_crps(svi_tree),
    "Pathfinder": compute_crps(pathfinder_tree),
    "MCLMC": compute_crps(mclmc_tree),
    "NUTS + DCT": compute_crps(nuts_dct_tree),
    "SVI + DCT": compute_crps(svi_dct_tree),
}
walltimes = {
    "NUTS": nuts_seconds,
    "SVI": svi_seconds,
    "Pathfinder": pathfinder_seconds,
    "MCLMC": mclmc_seconds,
    "NUTS + DCT": nuts_dct_seconds,
    "SVI + DCT": svi_dct_seconds,
}

comparison = pd.DataFrame(crps_results).T
comparison.columns = ["train CRPS", "test CRPS"]
comparison["walltime (s)"] = [walltimes[method] for method in comparison.index]
comparison.round(4)
```


|            | train CRPS | test CRPS | walltime (s) |
|------------|------------|-----------|--------------|
| NUTS       | 0.0242     | 0.0302    | 40.8726      |
| SVI        | 0.0271     | 0.0371    | 4.9253       |
| Pathfinder | 0.0276     | 0.0318    | 68.3311      |
| MCLMC      | 0.0242     | 0.0302    | 5.3368       |
| NUTS + DCT | 0.0242     | 0.0304    | 55.0582      |
| SVI + DCT  | 0.0258     | 0.0304    | 2.3789       |


# Forecast visualization

For each engine we overlay the in-sample posterior predictive (blue) and the forecast over the held-out year (orange), each with 50\\ and 94\\ HDI bands, on the observed series. The `DataTree` layout makes this a two-call `az.plot_lm` pattern: one call for the `posterior_predictive` group and one for the `predictions` group, sharing a single plot collection.


    In [22]:


``` python
def crps_title(name: str) -> str:
    """Format a plot title with the method's train and test CRPS."""
    scores = crps_results[name]
    return f"{name} (train CRPS: {scores['train']:.4f}, test CRPS: {scores['test']:.4f})"


def plot_forecast(tree: xr.DataTree, title: str) -> None:
    """Overlay the in-sample and forecast HDI bands on the observed series."""
    pc = az.plot_lm(
        tree,
        y="obs",
        x="week",
        group="posterior_predictive",
        ci_kind="hdi",
        ci_prob=(0.5, 0.94),
        smooth=False,
        visuals={"ci_band": {"color": "C0"}, "observed_scatter": False, "pe_line": False},
        figure_kwargs={"figsize": (10, 6)},
    )
    train_bands = pc.viz["ci_band"]["week"]
    band_train_94 = train_bands.sel(prob=0.94).item()
    band_train_50 = train_bands.sel(prob=0.5).item()
    az.plot_lm(
        tree,
        y="obs",
        x="week",
        group="predictions",
        plot_collection=pc,
        ci_kind="hdi",
        ci_prob=(0.5, 0.94),
        smooth=False,
        visuals={"ci_band": {"color": "C1"}, "observed_scatter": False, "pe_line": False},
    )
    test_bands = pc.viz["ci_band"]["week"]
    band_test_94 = test_bands.sel(prob=0.94).item()
    band_test_50 = test_bands.sel(prob=0.5).item()
    ax = pc.viz["figure"].item().axes[0]
    band_train_94.set_label(r"in-sample $94\%$ HDI")
    band_train_50.set_label(r"in-sample $50\%$ HDI")
    band_test_94.set_label(r"forecast $94\%$ HDI")
    band_test_50.set_label(r"forecast $50\%$ HDI")
    (obs_line,) = ax.plot(time, np.asarray(data[:, 0]), color="black", lw=1, label="observed")
    split_line = ax.axvline(T1, color="gray", ls="--", label="train/test split")
    ax.legend(
        handles=[band_train_94, band_train_50, band_test_94, band_test_50, obs_line, split_line],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )
    ax.set(title=title, ylabel="log(# rides)")


plot_forecast(nuts_tree, title=crps_title("NUTS"))
```


<figure class="figure">
<p><img src="inference_methods_comparison_files/figure-html/cell-23-output-1.png" class="figure-img" width="1011" height="611" /></p>
</figure>


    In [23]:


``` python
plot_forecast(svi_tree, title=crps_title("SVI"))
```


<figure class="figure">
<p><img src="inference_methods_comparison_files/figure-html/cell-24-output-1.png" class="figure-img" width="1011" height="611" /></p>
</figure>


    In [24]:


``` python
plot_forecast(svi_dct_tree, title=crps_title("SVI + DCT"))
```


<figure class="figure">
<p><img src="inference_methods_comparison_files/figure-html/cell-25-output-1.png" class="figure-img" width="1011" height="611" /></p>
</figure>


    In [25]:


``` python
plot_forecast(pathfinder_tree, title=crps_title("Pathfinder"))
```


<figure class="figure">
<p><img src="inference_methods_comparison_files/figure-html/cell-26-output-1.png" class="figure-img" width="1011" height="611" /></p>
</figure>


    In [26]:


``` python
plot_forecast(mclmc_tree, title=crps_title("MCLMC"))
```


<figure class="figure">
<p><img src="inference_methods_comparison_files/figure-html/cell-27-output-1.png" class="figure-img" width="1011" height="611" /></p>
</figure>


# Trade-offs

The summary plot puts the four engines side by side, together with the two runs that fit the same model through the DCT time-axis reparameterization.


    In [27]:


``` python
methods = list(crps_results)
train_scores = [crps_results[m]["train"] for m in methods]
test_scores = [crps_results[m]["test"] for m in methods]

x = np.arange(len(methods))
fig, ax = plt.subplots()
ax.bar(x - 0.2, train_scores, width=0.4, color="C0", label="train CRPS")
ax.bar(x + 0.2, test_scores, width=0.4, color="C1", label="test CRPS")
ax.set_xticks(x, methods)
ax.legend()
ax.set(title="CRPS by inference method", xlabel="inference method", ylabel="CRPS");
```


<figure class="figure">
<p><img src="inference_methods_comparison_files/figure-html/cell-28-output-1.png" class="figure-img" width="1011" height="611" /></p>
</figure>


Scores alone hide the cost side, so we also chart the wall time each engine needed to produce its posterior (the fit for the variational methods and the sampling run for the MCMC ones, measured on the same laptop CPU, one run each, so treat small differences as noise).


    In [28]:


``` python
fig, ax = plt.subplots()
bars = ax.bar(methods, [walltimes[m] for m in methods], color="C0")
ax.bar_label(bars, fmt="%.1f s")
ax.set(title="Training time by inference method", xlabel="inference method", ylabel="seconds");
```


<figure class="figure">
<p><img src="inference_methods_comparison_files/figure-html/cell-29-output-1.png" class="figure-img" width="1011" height="611" /></p>
</figure>


NUTS sets the reference score on both windows (train CRPS `0.0242`, test CRPS `0.0302`) in `40.9`s across its `4` chains. MCLMC (`5.3`s) and SVI (`4.9`s) both come in at a small fraction of that cost: MCLMC matches the reference to four decimals on this run by spending two fixed gradient evaluations per draw instead of a full NUTS trajectory, though its silent-failure mode still deserves the validation that NUTS's divergence diagnostics provide for free, and SVI's `0.0271`/`0.0371` reflects a diagonal-Gaussian guide that cannot bend to the true posterior's shape as faithfully as sampling does.Multi-path Pathfinder is the interesting case, because its forecasts and its own diagnostic point in opposite directions. Its scores are close to the reference, `0.0276` train and `0.0318` test CRPS, while its `pareto_k` came out at `12.35`, more than an order of magnitude above the `0.7` reliability threshold. Both readings are correct, because they describe different objects. The `pareto_k` printed by the fit cell measures whether importance weights over the pooled per-path draws can be trusted, and on a `474`-parameter posterior they cannot: the log ratio between the target and the approximation is dominated by a handful of draws. The samples that produced the CRPS above never passed through those weights, because `resample="auto"` read the same `pareto_k`, fell back to ELBO-weighted path sampling, and let the best path take over.The cost side is less flattering. At `68.3`s for `8` paths, Pathfinder is the slowest fit in the notebook, slower than NUTS's `40.9`s, because the ELBO is estimated at every one of the `500` L-BFGS iterates of all `8` paths. Most of that budget buys the diversity that makes the ELBO comparison meaningful rather than accuracy as such: fewer paths, or a smaller `maxiter`, would be much cheaper at the price of not knowing whether any path had converged. Read the table together with the diagnostic rather than either alone. Pathfinder here is an accurate, well-diagnosed fit that is not yet a cheap one, and tuning it down toward its usual role, a fast approximate posterior or an MCMC initializer, is the obvious next experiment.The two DCT columns show what [time_reparam](../../../reference/reparam.time_reparam.md#numpyro_forecast.reparam.time_reparam) changes. NUTS + DCT reproduces the NUTS scores (`0.0242`/`0.0304`), as it must for a unit-Jacobian change of coordinates, and it changes the sampler's work: `314` leapfrog steps per draw instead of `446` and a minimum bulk ESS over the drift increments of `559` instead of `46`, a twelvefold gain per draw, at `55.1`s against `40.9`s of wall time because each leapfrog step now pays for the transform. Per effective sample it is by far the cheapest MCMC run in the notebook. SVI + DCT, once its learning-rate peak is re-tuned for the rotated coordinates, is the best variational fit in the table: train CRPS `0.0258` and test CRPS `0.0304`, within a hair of the NUTS reference and well ahead of the plain SVI's `0.0371` test score, at a final ELBO loss of `-499` against `-436`, in `2.4`s. The same fit with the baseline's schedule stalled at an ELBO loss of `197`, so the rotation is free for the model and for the sampler, but for a variational fit it is a change of optimization problem whose schedule has to be re-tuned rather than carried over.## Next stepsA single train/test split is only one view of forecasting skill; the [univariate example](forecasting_univariate.md) shows how to score these same models with rolling-origin backtesting, including a fully vectorized variant. From here you can also swap guides (`AutoMultivariateNormal` captures posterior correlations that `AutoNormal` ignores, at the cost of an `O(n^2)` covariance), or swap kernels ([BlackjaxNUTSKernel](../../../reference/contrib.blackjax.BlackjaxNUTSKernel.md#numpyro_forecast.contrib.blackjax.BlackjaxNUTSKernel) runs BlackJAX's own NUTS through the same adapter MCLMC used above, and [BlackjaxCustomKernel](../../../reference/contrib.blackjax.BlackjaxCustomKernel.md#numpyro_forecast.contrib.blackjax.BlackjaxCustomKernel) accepts any BlackJAX sampler through a small build function) without touching `univariate_model` or the [to_datatree](../../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) export itself.## References- Orduz, J. [*Univariate time series forecasting with NumPyro*](https://juanitorduz.github.io/numpyro_forecasting-univariate/).- Pyro. [*Forecasting I: Univariate, Heavy Tailed*](https://pyro.ai/examples/forecasting_i.html).- Hoffman, M. D., & Gelman, A. (2014). [*The No-U-Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo*](https://jmlr.org/papers/v15/hoffman14a.html). JMLR.- Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). [*Stochastic variational inference*](https://jmlr.org/papers/v14/hoffman13a.html). JMLR.- Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2022). [*Pathfinder: Parallel quasi-Newton variational inference*](https://jmlr.org/papers/v23/21-0889.html). JMLR.- Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024). [*Pareto smoothed importance sampling*](https://jmlr.org/papers/v25/19-556.html). JMLR.- Robnik, J., De Luca, G. B., Silverstein, E., & Seljak, U. (2023). [*Microcanonical Hamiltonian Monte Carlo*](https://jmlr.org/papers/v24/22-1450.html). JMLR.- Robnik, J., & Seljak, U. (2024). [*Fluctuation without dissipation: Microcanonical Langevin Monte Carlo*](https://arxiv.org/abs/2303.18221).- Smith, L. N., & Topin, N. (2019). [*Super-convergence: Very fast training of neural networks using large learning rates*](https://arxiv.org/abs/1708.07120).
