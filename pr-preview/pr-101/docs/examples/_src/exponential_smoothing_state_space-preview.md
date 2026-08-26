# Exponential Smoothing in State Space Form with `numpyro_forecast`


Exponential smoothing is one of the most widely used forecasting techniques. In its classical (component) form it is a set of recursive update equations for a *level*, a *trend*, and a *seasonal* component. A more powerful way to write the same idea is the **innovations state space form** (also known as the single source of error, or SSOE, model), which turns exponential smoothing into a proper generative stochastic process. The key consequence is that forecast uncertainty is propagated correctly: the prediction interval widens with the horizon instead of collapsing to the observation noise.

This notebook ports the blog post [*Exponential Smoothing with NumPyro: State Space Form*](https://juanitorduz.github.io/exponential_smoothing_numpyro_ssm/) (with material from its predecessor [*Notes on Exponential Smoothing with NumPyro*](https://juanitorduz.github.io/exponential_smoothing_numpyro/)) into a `numpyro_forecast` example. We show how to write the damped Holt-Winters model in state space form as a plain NumPyro model on the package's [`ssoe`](https://juanitorduz.github.io/numpyro_forecast/reference/models.ssoe.html) building block (the single-source-of-error recursion is exactly what the block implements), fit it with the NUTS sampler, and reuse the package's forecasting and evaluation machinery. Along the way we introduce the JAX `scan` operation that rolls the latent state forward.

A practical note on the design: the [`innovations`](https://juanitorduz.github.io/numpyro_forecast/reference/models.innovations.html) and [`predict`](https://juanitorduz.github.io/numpyro_forecast/reference/models.predict.html) building blocks assume a deterministic mean plus independent per-step noise, which is not how an innovations model behaves: the error of one step drives the state of the next. That error feedback is what [ssoe](../../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) provides. It takes the driving series as an argument, and because the package's [predict_in_sample](../../../reference/predictive.predict_in_sample.md#numpyro_forecast.predictive.predict_in_sample) and [to_datatree](../../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) call the model with `data=None`, the observed series has to travel through the `covariates` argument; the model reads only its first `t_obs` rows, which the block checks.


# Prepare notebook


    In [1]:


``` python
import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import preliz as pz
import xarray as xr
from jax import random
from numpyro.infer import MCMC, NUTS

from numpyro_forecast import (
    Horizon,
    eval_coverage,
    eval_crps,
    eval_mae,
    eval_rmse,
    predictions_to_datatree,
    ssoe,
    to_datatree,
)
from numpyro_forecast.arrays import concat_future
from numpyro_forecast.typing import Array

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


    /Users/juanitorduz/Documents/numpyro_forecast/.claude/worktrees/refactor3-pr-e1/.venv/lib/python3.14/site-packages/preliz/ppls/pymc_io.py:16: UserWarning: PyMC not installed. PyMC related functions will not work.
      warnings.warn("PyMC not installed. PyMC related functions will not work.")
    /Users/juanitorduz/Documents/numpyro_forecast/.claude/worktrees/refactor3-pr-e1/.venv/lib/python3.14/site-packages/preliz/ppls/agnostic.py:34: UserWarning: PyMC not installed. PyMC related functions will not work.
      warnings.warn("PyMC not installed. PyMC related functions will not work.")


# Generate synthetic data

We use the same synthetic series as the source posts: a seasonal cosine wave with period m = 15, a slow logarithmic trend, and additive Gaussian noise,

y_t = \cos(2 \pi t) + \log(t + 1) + 0.2 \\ \varepsilon_t, \qquad \varepsilon_t \sim \text{Normal}(0, 1).

This gives us a trend, a clear seasonality, and enough noise to make the inference interesting. We hold out the last 20\\ of the series as a test set.


    In [2]:


``` python
n_seasons = 15
t = jnp.linspace(0, n_seasons + 1, (n_seasons + 1) * n_seasons)

rng_key, rng_subkey = random.split(rng_key)
y = jnp.cos(2 * jnp.pi * t) + jnp.log(t + 1) + 0.2 * random.normal(rng_subkey, t.shape)

n = y.shape[0]
n_train = int(0.8 * n)
future = n - n_train

t_train, t_test = t[:n_train], t[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# The package expects time at axis -2 and the observation dimension at axis -1.
train_data = y_train[:, None]
test_data = y_test[:, None]

# The observed series doubles as the covariate: the model reads its history from
# here (only the first t_obs rows are ever read), and the trailing zero rows just
# fix the forecast horizon.
covariates_train = train_data
covariates_full = concat_future(train_data, jnp.zeros((future, 1)))

print(f"total: {n}, train: {n_train}, test (forecast horizon): {future}")
```


    total: 240, train: 192, test (forecast horizon): 48


We can visualize the series:


    In [3]:


``` python
fig, ax = plt.subplots()
ax.plot(t_train, y_train, color="C0", label="train")
ax.plot(t_test, y_test, color="C1", label="test")
ax.axvline(float(t_test[0]), color="gray", linestyle="--", label="train/test split")
ax.legend(loc="upper left")
ax.set(title="Synthetic time series", xlabel="time", ylabel="y");
```


<figure class="figure">
<p><img src="exponential_smoothing_state_space_files/figure-html/cell-4-output-1.png" class="figure-img" width="1011" height="611" /></p>
</figure>


# A short detour on `scan`

Exponential smoothing is defined by a recursion: each state depends on the previous one. In JAX we express such recursions with [`jax.lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html) rather than a Python `for` loop, because `scan` compiles to a single efficient, differentiable operation. Conceptually `scan` is equivalent to the following pure-Python function (from the JAX documentation):

``` python
def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, np.stack(ys)
```

It threads a *carry* (the running state) through a step function `f`, and stacks the per-step outputs. The [ssoe](../../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) building block runs two such scans for us, but the state update it threads through them is ours to write, so it pays to see the mechanics once.


## A simple example

As a warm-up, we use `scan` to compute the geometric damping sum \varphi_h = \varphi + \varphi^2 + \cdots + \varphi^h that appears in the damped-trend forecast formula below. The carry holds the running sum and the current power of \varphi.


    In [4]:


``` python
def damping_sum(phi, h):
    def step(carry, _):
        running_sum, power = carry
        power = power * phi
        running_sum = running_sum + power
        return (running_sum, power), running_sum

    (total, _), partial_sums = jax.lax.scan(step, (0.0, 1.0), xs=None, length=h)
    return total, partial_sums


phi_example = 0.8
total, partial_sums = damping_sum(phi_example, 5)
closed_form = sum(phi_example**i for i in range(1, 6))
print(f"scan result:  {float(total):.5f}")
print(f"closed form:  {closed_form:.5f}")
print(f"partial sums: {np.asarray(partial_sums).round(5)}")
```


    scan result:  2.68928
    closed form:  2.68928
    partial sums: [0.8     1.44    1.952   2.3616  2.68928]


# From component form to state space form

The classical **damped Holt-Winters** method with additive seasonality of period m is a set of recursive updates for the level \ell_t, the trend b_t, and the seasonal component s_t, together with an h-step forecast,

 \begin{align\*} \hat{y}\_{t+h \mid t} &= \ell_t + \varphi_h \\ b_t + s\_{t + h - m(k+1)}, \\ \ell_t &= \alpha (y_t - s\_{t-m}) + (1 - \alpha)(\ell\_{t-1} + \varphi \\ b\_{t-1}), \\ b_t &= \beta^{\*} (\ell_t - \ell\_{t-1}) + (1 - \beta^{\*}) \varphi \\ b\_{t-1}, \\ s_t &= \gamma (y_t - \ell\_{t-1} - \varphi \\ b\_{t-1}) + (1 - \gamma) s\_{t-m}, \end{align\*} 

where \alpha, \beta^{\*}, \gamma \in (0, 1) are smoothing parameters, \varphi \in (0, 1) is the damping factor, \varphi_h = \varphi + \varphi^2 + \cdots + \varphi^h, and k = \lfloor (h-1)/m \rfloor.

The **innovations state space form** (SSOE) rewrites this as a generative model driven by a *single* error term \varepsilon_t shared across all equations,

 \begin{align\*} y_t &= \underbrace{\ell\_{t-1} + \varphi \\ b\_{t-1} + s\_{t-m}}\_{\mu_t} + \varepsilon_t, \qquad \varepsilon_t \sim \text{Normal}(0, \sigma), \\ \ell_t &= \ell\_{t-1} + \varphi \\ b\_{t-1} + \alpha \\ \varepsilon_t, \\ b_t &= \varphi \\ b\_{t-1} + \beta \\ \varepsilon_t, \\ s_t &= s\_{t-m} + \gamma \\ \varepsilon_t, \end{align\*} 

with the coefficient map \beta = \beta^{\*} \alpha and \gamma = \gamma^{\*} (1 - \alpha). The two forms are mathematically equivalent, but the SSOE form is the one we want for probabilistic forecasting. In sample, the innovation is exactly the one-step-ahead forecast error \varepsilon_t = y_t - \mu_t, so the whole state trajectory is a deterministic function of the observed data and the parameters. Out of sample there is no data, so \varepsilon_t is *sampled* and fed back into the level, trend, and seasonal updates. Because a single innovation drives every component, the forecast uncertainty compounds and the prediction interval widens with the horizon, which is the behavior we expect from a genuine stochastic process.


# The model

The model is a plain NumPyro function `(covariates, data=None)`. Its first line derives the per-call [`Horizon`](https://juanitorduz.github.io/numpyro_forecast/reference/models.Horizon.html) from the shapes (the observed data `h.data`, the number of in-sample steps `h.t_obs`, and the forecast length `h.future`), and the recursion goes to the [`ssoe`](https://juanitorduz.github.io/numpyro_forecast/reference/models.ssoe.html) building block. The block takes the driving series `y` (sliced from the covariates, see the design note above), the initial state, a `step` function, and the innovation distribution, and it owns the two scans, neither of which contains a NumPyro sample site:

1.  **In sample.** A deterministic filter consumes the observed series: at each step `step(carry, x_t)` returns the one-step-ahead mean \mu_t and a `carry_fn(y_t, eps_t)` that advances the state with the innovation \varepsilon_t = y_t - \mu_t. The means come back as `r.mu`; the whole in-sample likelihood is then a single `Normal` observation site `"obs"` against them, and we also expose \mu_t as the deterministic site `"mu"` for the in-sample fit plot.
2.  **Out of sample.** When `h.future > 0` the block draws the horizon innovations from the prior at a separate `"eps_future"` site (under its own `time_future` plate), rolls the state forward from the final in-sample state feeding those innovations back through `carry_fn`, and returns the trajectory as `r.y_future`, which we register as the deterministic `"forecast"` site the package's [forecast](../../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast) driver reads. Because `"eps_future"` does not exist while training, `Predictive` draws it from the prior at forecast time, exactly like the built-in `_future` sites.

The state update `advance` is shared by both scans and is the SSOE update above, one innovation driving level, trend, and seasonality. One shape convention to know: rows carry the observation axis, so the scalar state emits a `(1,)` mean (`mu[None]`) and reads the scalar innovation back out of the `(1,)` error (`eps_t[0]`); the block checks these shapes so a mismatch fails loudly instead of broadcasting silently.

The priors follow the source post: \text{Beta}(5, 5) on the level, trend, and seasonal smoothing parameters (flat enough near the boundaries to avoid a funnel-shaped posterior), \text{Beta}(2, 5) on the damping factor (favoring some damping), a tight \text{HalfNormal}(0.5) on the noise, and weakly informative priors on the initial states.


    In [5]:


``` python
def exponential_smoothing_ssm(covariates: Array, data: Array | None = None) -> None:
    """Damped Holt-Winters exponential smoothing in innovations state space form.

    Parameters
    ----------
    covariates
        The observed series itself, with time at axis ``-2``; only the first
        ``h.t_obs`` rows are read, the trailing rows fix the forecast horizon.
    data
        Observed data with time at axis ``-2``, or ``None`` when the drivers
        sample the observation site.
    """
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :]  # observed history only; never reads beyond t_obs

    # Smoothing parameters, damping, initial states, and observation noise.
    level_smoothing = numpyro.sample("level_smoothing", dist.Beta(5, 5))
    level_init = numpyro.sample("level_init", dist.Normal(y[0, 0], 1))
    trend_smoothing = numpyro.sample("trend_smoothing", dist.Beta(5, 5))
    trend_init = numpyro.sample("trend_init", dist.Normal(0, 0.1))
    seasonality_smoothing = numpyro.sample("seasonality_smoothing", dist.Beta(5, 5))
    phi = numpyro.sample("phi", dist.Beta(2, 5))
    with numpyro.plate("n_seasons", n_seasons):
        seasonality_init = numpyro.sample("seasonality_init", dist.Normal(0, 1))
    noise = numpyro.sample("noise", dist.HalfNormal(0.5))

    # Component form to SSOE coefficient map.
    beta = trend_smoothing * level_smoothing
    gamma = seasonality_smoothing * (1 - level_smoothing)

    def advance(carry, innovation):
        # Shared state update: one innovation drives level, trend, and seasonality.
        level, trend, seasonality = carry
        level = level + phi * trend + level_smoothing * innovation
        trend = phi * trend + beta * innovation
        new_season = seasonality[0] + gamma * innovation
        seasonality = jnp.concatenate([seasonality[1:], new_season[None]])
        return (level, trend, seasonality)

    def step(carry, _):
        level, trend, seasonality = carry
        mu = level + phi * trend + seasonality[0]
        # Rows carry the observation axis: emit a (1,) mean, read the scalar error back.
        return mu[None], lambda y_t, eps_t: advance(carry, eps_t[0])

    init_state = (level_init, trend_init, seasonality_init)
    r = ssoe(h, "eps", y, init_state, step, dist.Normal(0, noise))

    numpyro.deterministic("mu", r.mu)
    numpyro.sample("obs", dist.Normal(r.mu, noise), obs=h.data)
    if h.future > 0:
        numpyro.deterministic("forecast", r.y_future)
```


# Priors

Before fitting, it is worth looking at the priors on the bounded parameters. The \text{Beta}(5, 5) prior on the smoothing parameters is symmetric and concentrated away from 0 and 1, which keeps the sampler away from the boundary regions where the posterior geometry degenerates. The \text{Beta}(2, 5) prior on the damping factor \varphi puts more mass below 0.5, encoding a mild preference for damped (non-explosive) trends.


    In [6]:


``` python
fig, (ax_smoothing, ax_noise) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 9),
    sharex=False,
    sharey=True,
    layout="constrained",
)
pz.Beta(5, 5).plot_pdf(ax=ax_smoothing, color="C0")
pz.Beta(2, 5).plot_pdf(ax=ax_smoothing, color="C1")
ax_smoothing.set(
    title="Priors on the bounded parameters",
    xlabel=None,
    ylabel="density",
)

pz.HalfNormal(0.5).plot_pdf(ax=ax_noise, color="C2")
ax_noise.set(
    title="Prior on the observation noise",
    xlabel="value",
    ylabel="density",
);
```


<figure class="figure">
<p><img src="exponential_smoothing_state_space_files/figure-html/cell-7-output-1.png" class="figure-img" width="1131" height="788" /></p>
</figure>


# Inference

We fit the model with plain NumPyro: the NUTS sampler through `MCMC`, running 4 chains of 2{,}000 warmup and 2{,}000 sampling steps each on the training window. The model is an ordinary NumPyro callable, so nothing package-specific happens here; `mcmc.get_samples()` returns the posterior draws as a plain dictionary with the chains flattened together, which is the format every package driver consumes.

We then export the draws into an ArviZ-schema `xarray.DataTree` with [`to_datatree`](https://juanitorduz.github.io/numpyro_forecast/reference/convert.to_datatree.html): a single call restores the `(chain, draw)` structure (we pass `num_chains=4`), samples the in-sample one-step-ahead posterior predictive from the same draws, and, because we hand it the *full-horizon* covariates, also runs the forecast and stores it in the `predictions` group. Everything downstream (diagnostics, trace plots, the in-sample fit, the forecast, the metrics) reads from this one object.


    In [7]:


``` python
rng_key, rng_subkey = random.split(rng_key)
mcmc = MCMC(
    NUTS(exponential_smoothing_ssm),
    num_warmup=2_000,
    num_samples=2_000,
    num_chains=4,
    chain_method="sequential",
    progress_bar=False,
)
mcmc.run(rng_subkey, covariates_train, train_data)
posterior = mcmc.get_samples()

rng_key, rng_subkey = random.split(rng_key)
tree = to_datatree(
    rng_subkey,
    exponential_smoothing_ssm,
    posterior,
    train_data,
    covariates_full,
    num_chains=4,
    posterior_dims={"mu": ["time", "obs_dim"]},
)
tree
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
│       Dimensions:                 (chain: 4, draw: 2000, time: 192, obs_dim: 1,
│                                    seasonality_init_dim_0: 15)
│       Coordinates:
│         * chain                   (chain) int64 32B 0 1 2 3
│         * draw                    (draw) int64 16kB 0 1 2 3 4 ... 1996 1997 1998 1999
│         * time                    (time) int64 2kB 0 1 2 3 4 5 ... 187 188 189 190 191
│         * obs_dim                 (obs_dim) int64 8B 0
│         * seasonality_init_dim_0  (seasonality_init_dim_0) int64 120B 0 1 2 ... 13 14
│       Data variables:
│           level_init              (chain, draw) float32 32kB 0.3102 0.4143 ... 0.5364
│           level_smoothing         (chain, draw) float32 32kB 0.2118 0.207 ... 0.212
│           mu                      (chain, draw, time, obs_dim) float32 6MB 0.9767 ....
│           noise                   (chain, draw) float32 32kB 0.233 0.2251 ... 0.2407
│           phi                     (chain, draw) float32 32kB 0.3755 0.3285 ... 0.3153
│           seasonality_init        (chain, draw, seasonality_init_dim_0) float32 480kB ...
│           seasonality_smoothing   (chain, draw) float32 32kB 0.2746 0.1679 ... 0.3157
│           trend_init              (chain, draw) float32 32kB 0.09367 ... -0.05799
│           trend_smoothing         (chain, draw) float32 32kB 0.5881 0.5905 ... 0.3572
│       Attributes:
│           created_at:                 2026-08-26T16:35:47.762252+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.3.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
├── Group: /posterior_predictive
│       Dimensions:  (chain: 4, draw: 2000, time: 192, obs_dim: 1)
│       Coordinates:
│         * chain    (chain) int64 32B 0 1 2 3
│         * draw     (draw) int64 16kB 0 1 2 3 4 5 6 ... 1994 1995 1996 1997 1998 1999
│         * time     (time) int64 2kB 0 1 2 3 4 5 6 7 ... 185 186 187 188 189 190 191
│         * obs_dim  (obs_dim) int64 8B 0
│       Data variables:
│           obs      (chain, draw, time, obs_dim) float32 6MB 0.689 1.007 ... 2.481
│       Attributes:
│           created_at:                 2026-08-26T16:35:47.907858+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.3.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
├── Group: /observed_data
│       Dimensions:  (time: 192, obs_dim: 1)
│       Coordinates:
│         * time     (time) int64 2kB 0 1 2 3 4 5 6 7 ... 185 186 187 188 189 190 191
│         * obs_dim  (obs_dim) int64 8B 0
│       Data variables:
│           obs      (time, obs_dim) float32 768B 1.121 1.137 0.6104 ... 2.527 2.841
│       Attributes:
│           created_at:                 2026-08-26T16:35:47.908092+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.3.0
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /constant_data
│       Dimensions:        (time: 192, covariate_dim: 1)
│       Coordinates:
│         * time           (time) int64 2kB 0 1 2 3 4 5 6 ... 186 187 188 189 190 191
│         * covariate_dim  (covariate_dim) int64 8B 0
│       Data variables:
│           covariates     (time, covariate_dim) float32 768B 1.121 1.137 ... 2.841
│       Attributes:
│           created_at:                 2026-08-26T16:35:47.908261+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.3.0
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /predictions
│       Dimensions:  (chain: 4, draw: 2000, time: 48, obs_dim: 1)
│       Coordinates:
│         * chain    (chain) int64 32B 0 1 2 3
│         * draw     (draw) int64 16kB 0 1 2 3 4 5 6 ... 1994 1995 1996 1997 1998 1999
│         * time     (time) int64 384B 192 193 194 195 196 197 ... 235 236 237 238 239
│         * obs_dim  (obs_dim) int64 8B 0
│       Data variables:
│           obs      (chain, draw, time, obs_dim) float32 2MB 3.364 3.11 ... 3.689 4.017
│       Attributes:
│           created_at:                 2026-08-26T16:35:48.088411+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.3.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
└── Group: /predictions_constant_data
        Dimensions:        (time: 48, covariate_dim: 1)
        Coordinates:
          * time           (time) int64 384B 192 193 194 195 196 ... 235 236 237 238 239
          * covariate_dim  (covariate_dim) int64 8B 0
        Data variables:
            covariates     (time, covariate_dim) float32 192B 0.0 0.0 0.0 ... 0.0 0.0
        Attributes:
            created_at:                 2026-08-26T16:35:48.088629+00:00
            creation_library:           ArviZ
            creation_library_version:   1.3.0
            creation_library_language:  Python
            sample_dims:                []
```


xarray.DataTree


/posterior(19)

Dimensions:


- chain: 4
- draw: 2000
- time: 192
- obs_dim: 1
- seasonality_init_dim_0: 15


Coordinates: (5)


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


0 1 2 3 4 ... 1996 1997 1998 1999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([   0,    1,    2, ..., 1997, 1998, 1999], shape=(2000,))


time


(time)


int64


0 1 2 3 4 5 ... 187 188 189 190 191


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,182, 183, 184, 185, 186, 187, 188, 189, 190, 191])


obs_dim


(obs_dim)


int64


0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0])


seasonality_init_dim_0


(seasonality_init_dim_0)


int64


0 1 2 3 4 5 6 7 8 9 10 11 12 13 14


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])


Data variables: (9)


level_init


(chain, draw)


float32


0.3102 0.4143 ... 0.6464 0.5364


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[ 0.3102441 ,  0.4143369 ,  0.6325664 , ...,  0.428628  ,0.5188503 ,  0.44129032],[ 0.23185168,  0.26650435, -0.37129322, ...,  0.16399278,0.239105  , -0.00179972],[ 0.38834456,  0.5037876 ,  0.84100366, ...,  0.38611057,0.2578325 ,  0.23527595],[-0.18569404, -0.28162962, -0.1490242 , ...,  0.523722  ,0.64644605,  0.536389  ]], shape=(4, 2000), dtype=float32)


level_smoothing


(chain, draw)


float32


0.2118 0.207 ... 0.2118 0.212


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.21180885, 0.20700541, 0.20735084, ..., 0.15499593, 0.21554367,0.2452756 ],[0.15743446, 0.1563927 , 0.16128917, ..., 0.17726235, 0.2528454 ,0.15303738],[0.29045674, 0.17839259, 0.17678964, ..., 0.30907997, 0.20649406,0.20125927],[0.23124462, 0.21992955, 0.22099063, ..., 0.2672789 , 0.21176444,0.21197107]], shape=(4, 2000), dtype=float32)


mu


(chain, draw, time, obs_dim)


float32


0.9767 0.9038 ... 2.108 2.628


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[0.97674537],[0.9037795 ],[0.7298924 ],...,[1.6078115 ],[2.116275  ],[2.635385  ]],[[1.2524488 ],[0.9502721 ],[0.8070518 ],...,[1.5674657 ],[2.0896082 ],[2.6750996 ]],[[1.460968  ],[1.2301044 ],[0.9015404 ],...,......,[1.5880554 ],[2.117119  ],[2.646013  ]],[[1.2054992 ],[0.9732424 ],[0.929857  ],...,[1.60047   ],[2.1053421 ],[2.6341052 ]],[[0.9961096 ],[1.1050782 ],[0.8551133 ],...,[1.6048017 ],[2.107566  ],[2.6277716 ]]]], shape=(4, 2000, 192, 1), dtype=float32)


noise


(chain, draw)


float32


0.233 0.2251 ... 0.2333 0.2407


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.23295596, 0.2250819 , 0.2368775 , ..., 0.2707622 , 0.22862327,0.21895857],[0.23950501, 0.2183373 , 0.21666214, ..., 0.20968458, 0.23674776,0.2454313 ],[0.20722628, 0.22318353, 0.22304565, ..., 0.24168764, 0.2678586 ,0.2648568 ],[0.24556258, 0.23259634, 0.23021604, ..., 0.23836681, 0.23333827,0.24066891]], shape=(4, 2000), dtype=float32)


phi


(chain, draw)


float32


0.3755 0.3285 ... 0.2849 0.3153


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.3755373 , 0.32846394, 0.22724172, ..., 0.23776004, 0.4451672 ,0.16674133],[0.4246999 , 0.3532729 , 0.34437832, ..., 0.36900154, 0.23214853,0.43913049],[0.07934931, 0.05168606, 0.02625992, ..., 0.10842285, 0.24650073,0.29173326],[0.07110062, 0.4732713 , 0.16116785, ..., 0.06381722, 0.2849329 ,0.31532416]], shape=(4, 2000), dtype=float32)


seasonality_init


(chain, draw, seasonality_init_dim_0)


float32


0.6313 0.5078 0.266 ... 0.5299 0.81


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 0.63132477,  0.5078063 ,  0.26599875, ..., -0.00247833,0.45693517,  0.575393  ],[ 0.82355404,  0.54904604,  0.3597271 , ...,  0.4279541 ,0.6529156 ,  0.8680943 ],[ 0.8047464 ,  0.6471148 ,  0.34061605, ...,  0.06205944,0.48821434,  0.81348497],...,[ 0.63374007,  0.7369648 ,  0.302432  , ...,  0.3795026 ,0.34037435,  0.7106715 ],[ 0.61370116,  0.40098652,  0.2513078 , ...,  0.08412582,0.588745  ,  0.5920451 ],[ 0.7206711 ,  0.27203247,  0.1894665 , ...,  0.2789006 ,0.31283894,  0.88871574]],[[ 0.96456546,  0.93327475,  0.3684165 , ...,  0.632422  ,0.79877055,  0.9758999 ],[ 0.9407794 ,  1.0736065 ,  0.56424934, ...,  0.52303416,0.8610957 ,  1.1943538 ],[ 1.0912931 ,  1.1844621 ,  0.70343727, ...,  0.7682459 ,0.81798065,  1.2376496 ],...[ 0.59521157,  0.73905504,  0.31363192, ...,  0.24442632,0.46798927,  0.9713884 ],[ 1.0893528 ,  0.7957339 ,  0.64128464, ...,  0.58853436,0.85764366,  0.75745505],[ 1.1496885 ,  0.71419543,  0.61484206, ...,  0.5831527 ,0.88147783,  0.914384  ]],[[ 1.406563  ,  1.364231  ,  1.3505374 , ...,  1.0180956 ,1.1907846 ,  1.5561029 ],[ 1.2795255 ,  1.3756386 ,  0.88317853, ...,  0.81733286,1.4390804 ,  1.1839983 ],[ 1.3795034 ,  1.1911049 ,  1.0726407 , ...,  0.99758416,0.9708299 ,  1.6331786 ],...,[ 0.49653697,  0.5532248 ,  0.0899976 , ...,  0.20212635,0.32169744,  0.5954196 ],[ 0.5956105 ,  0.39341512,  0.3152626 , ...,  0.08289178,0.52967566,  0.66843176],[ 0.47800586,  0.56324923,  0.3065285 , ...,  0.18145472,0.52994084,  0.8100125 ]]], shape=(4, 2000, 15), dtype=float32)


seasonality_smoothing


(chain, draw)


float32


0.2746 0.1679 ... 0.2785 0.3157


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.2745696 , 0.1679367 , 0.23027821, ..., 0.42047608, 0.25508708,0.19790621],[0.23300436, 0.19097768, 0.25639692, ..., 0.12793216, 0.2969641 ,0.17143218],[0.19962682, 0.23564851, 0.21722025, ..., 0.2075097 , 0.23786122,0.26215643],[0.16842502, 0.30570108, 0.21334799, ..., 0.2795805 , 0.27852777,0.31565687]], shape=(4, 2000), dtype=float32)


trend_init


(chain, draw)


float32


0.09367 0.04432 ... -0.05799


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[ 0.09366985,  0.04432118,  0.10409731, ..., -0.03882837,0.08496471, -0.0477034 ],[ 0.09536485, -0.25388274,  0.19554064, ..., -0.05217663,0.04778647,  0.0881134 ],[-0.16613765, -0.0596179 , -0.02964345, ...,  0.00174198,0.07353854,  0.1108966 ],[ 0.14057688, -0.02974648, -0.01118814, ..., -0.05722304,-0.12830149, -0.05798854]], shape=(4, 2000), dtype=float32)


trend_smoothing


(chain, draw)


float32


0.5881 0.5905 ... 0.3505 0.3572


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.588146  , 0.5904923 , 0.50878656, ..., 0.5717677 , 0.44782993,0.6288765 ],[0.5159084 , 0.4802067 , 0.47103935, ..., 0.41554493, 0.6417081 ,0.6008914 ],[0.7153275 , 0.67368335, 0.7855745 , ..., 0.4263487 , 0.2496242 ,0.2690621 ],[0.5558435 , 0.4393846 , 0.5796549 , ..., 0.7138077 , 0.3504516 ,0.35721207]], shape=(4, 2000), dtype=float32)


Attributes: (5)


created_at :  
2026-08-26T16:35:47.762252+00:00

creation_library :  
ArviZ

creation_library_version :  
1.3.0

creation_library_language :  
Python

sample_dims :  
\['chain', 'draw'\]


/posterior_predictive(10)

Dimensions:


- chain: 4
- draw: 2000
- time: 192
- obs_dim: 1


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


0 1 2 3 4 ... 1996 1997 1998 1999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([   0,    1,    2, ..., 1997, 1998, 1999], shape=(2000,))


time


(time)


int64


0 1 2 3 4 5 ... 187 188 189 190 191


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,182, 183, 184, 185, 186, 187, 188, 189, 190, 191])


obs_dim


(obs_dim)


int64


0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0])


Data variables: (1)


obs


(chain, draw, time, obs_dim)


float32


0.689 1.007 0.6744 ... 1.955 2.481


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[0.68895787],[1.0071591 ],[0.67442954],...,[1.7267246 ],[2.2576191 ],[2.5051014 ]],[[1.0209818 ],[1.1097883 ],[0.9442604 ],...,[1.3690847 ],[2.154669  ],[2.002819  ]],[[1.4431775 ],[1.0084363 ],[1.0690532 ],...,......,[1.8580503 ],[2.2101762 ],[2.9321227 ]],[[1.0313734 ],[0.9196002 ],[1.1770512 ],...,[1.4583066 ],[2.1498528 ],[2.760774  ]],[[0.9746825 ],[1.146706  ],[0.9472305 ],...,[1.5435624 ],[1.9549606 ],[2.480648  ]]]], shape=(4, 2000, 192, 1), dtype=float32)


Attributes: (5)


created_at :  
2026-08-26T16:35:47.907858+00:00

creation_library :  
ArviZ

creation_library_version :  
1.3.0

creation_library_language :  
Python

sample_dims :  
\['chain', 'draw'\]


/observed_data(8)

Dimensions:


- time: 192
- obs_dim: 1


Coordinates: (2)


time


(time)


int64


0 1 2 3 4 5 ... 187 188 189 190 191


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,182, 183, 184, 185, 186, 187, 188, 189, 190, 191])


obs_dim


(obs_dim)


int64


0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0])


Data variables: (1)


obs


(time, obs_dim)


float32


1.121 1.137 0.6104 ... 2.527 2.841


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[ 1.1211528 ],[ 1.13744   ],[ 0.6103915 ],[ 0.35997933],[-0.11876036],[-0.38529447],[-0.57232076],[-0.84638906],[-0.5815945 ],[-0.33790883],[-0.05460303],[ 0.38775223],[ 1.1809202 ],[ 1.3550934 ],[ 1.666842  ],[ 1.6369815 ],[ 1.4368426 ],[ 1.4964561 ],[ 0.9815164 ],[ 0.49746913],...[ 1.7027713 ],[ 1.55514   ],[ 1.6696285 ],[ 2.1702137 ],[ 2.339507  ],[ 2.9180605 ],[ 3.2757707 ],[ 3.1773505 ],[ 3.600819  ],[ 3.4696782 ],[ 3.138224  ],[ 2.3520281 ],[ 2.118178  ],[ 1.7963666 ],[ 1.5095379 ],[ 1.6045712 ],[ 1.3837999 ],[ 1.7123247 ],[ 2.5271583 ],[ 2.841226  ]], dtype=float32)


Attributes: (5)


created_at :  
2026-08-26T16:35:47.908092+00:00

creation_library :  
ArviZ

creation_library_version :  
1.3.0

creation_library_language :  
Python

sample_dims :  
\[\]


/constant_data(8)

Dimensions:


- time: 192
- covariate_dim: 1


Coordinates: (2)


time


(time)


int64


0 1 2 3 4 5 ... 187 188 189 190 191


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,182, 183, 184, 185, 186, 187, 188, 189, 190, 191])


covariate_dim


(covariate_dim)


int64


0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0])


Data variables: (1)


covariates


(time, covariate_dim)


float32


1.121 1.137 0.6104 ... 2.527 2.841


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[ 1.1211528 ],[ 1.13744   ],[ 0.6103915 ],[ 0.35997933],[-0.11876036],[-0.38529447],[-0.57232076],[-0.84638906],[-0.5815945 ],[-0.33790883],[-0.05460303],[ 0.38775223],[ 1.1809202 ],[ 1.3550934 ],[ 1.666842  ],[ 1.6369815 ],[ 1.4368426 ],[ 1.4964561 ],[ 0.9815164 ],[ 0.49746913],...[ 1.7027713 ],[ 1.55514   ],[ 1.6696285 ],[ 2.1702137 ],[ 2.339507  ],[ 2.9180605 ],[ 3.2757707 ],[ 3.1773505 ],[ 3.600819  ],[ 3.4696782 ],[ 3.138224  ],[ 2.3520281 ],[ 2.118178  ],[ 1.7963666 ],[ 1.5095379 ],[ 1.6045712 ],[ 1.3837999 ],[ 1.7123247 ],[ 2.5271583 ],[ 2.841226  ]], dtype=float32)


Attributes: (5)


created_at :  
2026-08-26T16:35:47.908261+00:00

creation_library :  
ArviZ

creation_library_version :  
1.3.0

creation_library_language :  
Python

sample_dims :  
\[\]


/predictions(10)

Dimensions:


- chain: 4
- draw: 2000
- time: 48
- obs_dim: 1


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


0 1 2 3 4 ... 1996 1997 1998 1999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([   0,    1,    2, ..., 1997, 1998, 1999], shape=(2000,))


time


(time)


int64


192 193 194 195 ... 236 237 238 239


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205,206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,234, 235, 236, 237, 238, 239])


obs_dim


(obs_dim)


int64


0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0])


Data variables: (1)


obs


(chain, draw, time, obs_dim)


float32


3.364 3.11 3.152 ... 3.689 4.017


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[3.3638828],[3.1097145],[3.1518757],...,[2.9554002],[2.390232 ],[2.9109173]],[[2.7011962],[3.1653354],[3.771706 ],...,[2.0777855],[2.6018271],[2.671361 ]],[[2.8148594],[3.181801 ],[3.3359663],...,......,[2.5725627],[3.1331372],[3.293423 ]],[[2.690995 ],[3.6425638],[3.8258421],...,[2.8313189],[3.0710776],[3.5152655]],[[2.7133482],[3.363191 ],[3.7255483],...,[3.4982908],[3.6888971],[4.0173273]]]], shape=(4, 2000, 48, 1), dtype=float32)


Attributes: (5)


created_at :  
2026-08-26T16:35:48.088411+00:00

creation_library :  
ArviZ

creation_library_version :  
1.3.0

creation_library_language :  
Python

sample_dims :  
\['chain', 'draw'\]


/predictions_constant_data(8)

Dimensions:


- time: 48
- covariate_dim: 1


Coordinates: (2)


time


(time)


int64


192 193 194 195 ... 236 237 238 239


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205,206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,234, 235, 236, 237, 238, 239])


covariate_dim


(covariate_dim)


int64


0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0])


Data variables: (1)


covariates


(time, covariate_dim)


float32


0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],...[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]], dtype=float32)


Attributes: (5)


created_at :  
2026-08-26T16:35:48.088629+00:00

creation_library :  
ArviZ

creation_library_version :  
1.3.0

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


# Diagnostics

With the chains restored in the tree, ArviZ's convergence diagnostics apply directly: \hat{R} and the bulk and tail effective sample sizes for the scalar parameters.


    In [8]:


``` python
scalar_vars = [
    "level_smoothing",
    "trend_smoothing",
    "seasonality_smoothing",
    "phi",
    "noise",
    "level_init",
    "trend_init",
]
rhat = az.rhat(tree, var_names=scalar_vars)
ess_bulk = az.ess(tree, var_names=scalar_vars)
ess_tail = az.ess(tree, var_names=scalar_vars, method="tail")
diagnostics = pd.DataFrame(
    {
        "r_hat": [float(rhat[name].item()) for name in scalar_vars],
        "ess_bulk": [float(ess_bulk[name].item()) for name in scalar_vars],
        "ess_tail": [float(ess_tail[name].item()) for name in scalar_vars],
    },
    index=scalar_vars,
)
diagnostics.round({"r_hat": 3, "ess_bulk": 0, "ess_tail": 0})
```


|                       | r_hat | ess_bulk | ess_tail |
|-----------------------|-------|----------|----------|
| level_smoothing       | 1.001 | 3880.0   | 3947.0   |
| trend_smoothing       | 1.001 | 4702.0   | 4784.0   |
| seasonality_smoothing | 1.001 | 2849.0   | 3518.0   |
| phi                   | 1.002 | 3594.0   | 3534.0   |
| noise                 | 1.001 | 3586.0   | 4176.0   |
| level_init            | 1.005 | 855.0    | 1586.0   |
| trend_init            | 1.000 | 4391.0   | 4760.0   |


The \hat{R} values are close to 1 and the effective sample sizes are healthy, which indicates that the chains have mixed well. This is the payoff of the state space parameterization together with the tuned priors: the posterior geometry is well behaved and the sampler explores it without trouble. The trace plots below confirm the good mixing.


    In [9]:


``` python
pc_trace = az.plot_trace_dist(
    tree,
    var_names=scalar_vars,
    figure_kwargs={"figsize": (10, 16)},
    compact=True,
)
pc_trace.viz["figure"].item().suptitle(
    "Trace plots",
    fontsize=18,
    fontweight="bold",
    y=1.03,
);
```


<figure class="figure">
<p><img src="exponential_smoothing_state_space_files/figure-html/cell-10-output-1.png" class="figure-img" width="1011" height="1663" /></p>
</figure>


# Forecast

The tree already holds both predictive ensembles, one draw per posterior sample. The `posterior_predictive` group is the in-sample one-step-ahead predictive of the `"obs"` site: the fitted mean \mu_t plus observation noise. The `predictions` group holds the forecast over the test horizon: for each posterior draw the model replayed the in-sample filter, then rolled the state forward while sampling fresh innovations. We stack the `(chain, draw)` dimensions of each into a single sample axis to get the draws-first layout the plotting and scoring helpers expect. One consequence worth noting: the bands and the metrics below use all 8{,}000 forecast paths, one per posterior draw, rather than a thinned subset, so the plotted ensemble and the scored ensemble are the same.


    In [10]:


``` python
def stacked_draws(group: xr.DataTree | xr.DataArray, var: str) -> np.ndarray:
    """Stack a tree variable's ``(chain, draw)`` dims into a leading sample axis.

    Parameters
    ----------
    group
        A tree group holding ``var`` with dims ``(chain, draw, time, obs_dim)``
        (typed as the union ``tree[...]`` returns; a group always arrives here).
    var
        Name of the variable to extract.

    Returns
    -------
    np.ndarray
        The draws with shape ``(sample, time, obs_dim)``.
    """
    return (
        group.dataset[var]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "time", "obs_dim")
        .to_numpy()
    )


in_sample_pp = stacked_draws(tree["posterior_predictive"], "obs")
forecast_draws = stacked_draws(tree["predictions"], "obs")

print(f"in-sample posterior predictive: {in_sample_pp.shape}")
print(f"forecast samples: {forecast_draws.shape}")
```


    in-sample posterior predictive: (8000, 192, 1)
    forecast samples: (8000, 48, 1)


We visualize both the in-sample fit and the forecast with `az.plot_lm`, showing the 50\\ and 94\\ HDI bands (packing each ensemble with the package's [predictions_to_datatree](../../../reference/convert.predictions_to_datatree.md#numpyro_forecast.convert.predictions_to_datatree)). The forecast band (in orange) clearly fans out as the horizon grows: this is the calibrated uncertainty that the innovations state space form provides.


    In [11]:


``` python
crps_train = eval_crps(in_sample_pp, train_data)
crps_test = eval_crps(forecast_draws, test_data)

hdi_probs = (0.5, 0.94)
pc = az.plot_lm(
    predictions_to_datatree(in_sample_pp, np.asarray(t_train), ["y"], observed=train_data),
    y="obs",
    x="t",
    plot_dim="time",
    ci_kind="hdi",
    ci_prob=hdi_probs,
    smooth=False,
    visuals={"ci_band": {"color": "C0"}, "observed_scatter": False, "pe_line": False},
    figure_kwargs={"figsize": (12, 7)},
)
in_sample_bands = pc.viz["ci_band"]["t"]
band_in_94 = in_sample_bands.sel(prob=0.94).item()
band_in_50 = in_sample_bands.sel(prob=0.5).item()

az.plot_lm(
    predictions_to_datatree(forecast_draws, np.asarray(t_test), ["y"], observed=test_data),
    y="obs",
    x="t",
    plot_dim="time",
    plot_collection=pc,
    ci_kind="hdi",
    ci_prob=hdi_probs,
    smooth=False,
    visuals={"ci_band": {"color": "C1"}, "observed_scatter": False, "pe_line": False},
)
forecast_bands = pc.viz["ci_band"]["t"]
band_fc_94 = forecast_bands.sel(prob=0.94).item()
band_fc_50 = forecast_bands.sel(prob=0.5).item()

ax = pc.viz["figure"].item().axes[0]
band_in_94.set_label(r"in-sample $94\%$ HDI")
band_in_50.set_label(r"in-sample $50\%$ HDI")
band_fc_94.set_label(r"forecast $94\%$ HDI")
band_fc_50.set_label(r"forecast $50\%$ HDI")
(observed_line,) = ax.plot(np.asarray(t), np.asarray(y), color="black", lw=1, label="observed")
split_line = ax.axvline(float(t_test[0]), color="gray", linestyle="--", label="train/test split")
ax.legend(
    handles=[band_in_94, band_in_50, band_fc_94, band_fc_50, observed_line, split_line],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=3,
)
ax.set(
    title=f"Exponential smoothing forecast (train CRPS: {crps_train:.3f}, test CRPS: {crps_test:.3f})",
    xlabel="time",
    ylabel="y",
);
```


<figure class="figure">
<p><img src="exponential_smoothing_state_space_files/figure-html/cell-12-output-1.png" class="figure-img" width="1211" height="711" /></p>
</figure>


# Evaluation

Finally, we score the forecast against the held-out test set with the package's evaluation metrics: mean absolute error and root mean squared error (point-forecast accuracy), the continuous ranked probability score (a proper score for the whole predictive distribution), and the empirical coverage of the central 90\\ interval (calibration).


    In [12]:


``` python
metrics = {
    "MAE": eval_mae(forecast_draws, test_data),
    "RMSE": eval_rmse(forecast_draws, test_data),
    "CRPS": eval_crps(forecast_draws, test_data),
    "coverage (90%)": eval_coverage(forecast_draws, test_data, alpha=0.9),
}
for name, value in metrics.items():
    print(f"{name:>16}: {value:.4f}")
```


                 MAE: 0.2436
                RMSE: 0.2821
                CRPS: 0.1657
      coverage (90%): 0.9583


The coverage of the central 90\\ interval sits close to its nominal level, confirming that the forecast is well calibrated. For a systematic assessment over multiple origins you would reach for `numpyro_forecast.backtest`, which refits the model on a moving window (the [ARMA example](https://juanitorduz.github.io/numpyro_forecast/examples/arma.html) does exactly that with the same building block); we omit it here because it retrains the full sampler for every window.


# References

- Hyndman, R. J., & Athanasopoulos, G. (2021). [*Forecasting: Principles and Practice*](https://otexts.com/fpp3/), 3rd edition. Chapters on exponential smoothing and ETS models.
- Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
- Orduz, J. [*Exponential Smoothing with NumPyro: State Space Form*](https://juanitorduz.github.io/exponential_smoothing_numpyro_ssm/).
- Orduz, J. [*Notes on Exponential Smoothing with NumPyro*](https://juanitorduz.github.io/exponential_smoothing_numpyro/).
