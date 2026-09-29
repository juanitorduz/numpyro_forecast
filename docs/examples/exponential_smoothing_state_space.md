# Exponential Smoothing in State Space Form


Exponential Smoothing in State Space Form with `numpyro_forecast`

Exponential smoothing is one of the most widely used forecasting techniques. In its classical (component) form it is a set of recursive update equations for a *level*, a *trend*, and a *seasonal* component. A more powerful way to write the same idea is the **innovations state space form** (also known as the single source of error, or SSOE, model), which turns exponential smoothing into a proper generative stochastic process. The key consequence is that forecast uncertainty is propagated correctly: the prediction interval widens with the horizon instead of collapsing to the observation noise.

This notebook ports the blog post [*Exponential Smoothing with NumPyro: State Space Form*](https://juanitorduz.github.io/exponential_smoothing_numpyro_ssm/) (with material from its predecessor [*Notes on Exponential Smoothing with NumPyro*](https://juanitorduz.github.io/exponential_smoothing_numpyro/)) into a `numpyro_forecast` example. We show how to write the damped Holt-Winters model in state space form as a plain NumPyro model on the package's [`ssoe`](https://juanitorduz.github.io/numpyro_forecast/reference/models.ssoe.html) building block (the single-source-of-error recursion is exactly what the block implements), fit it with the NUTS sampler, and reuse the package's forecasting and evaluation machinery. Along the way we introduce the JAX `scan` operation that rolls the latent state forward.

A practical note on the design: the [`innovations`](https://juanitorduz.github.io/numpyro_forecast/reference/models.innovations.html) and [`predict`](https://juanitorduz.github.io/numpyro_forecast/reference/models.predict.html) building blocks assume a deterministic mean plus independent per-step noise, which is not how an innovations model behaves: the error of one step drives the state of the next. That error feedback is what [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) provides. It takes the driving series as an argument, and because the package's [predict_in_sample](../../reference/predictive.predict_in_sample.md#numpyro_forecast.predictive.predict_in_sample) and [to_datatree](../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) call the model with `data=None`, the observed series has to travel through the `covariates` argument; the model reads only its first `t_obs` rows, which the block checks.


# Prepare notebook


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


# Generate synthetic data

We use the same synthetic series as the source posts: a seasonal cosine wave with period m = 15, a slow logarithmic trend, and additive Gaussian noise,

y_t = \cos(2 \pi t) + \log(t + 1) + 0.2 \\ \varepsilon_t, \qquad \varepsilon_t \sim \text{Normal}(0, 1).

This gives us a trend, a clear seasonality, and enough noise to make the inference interesting. We hold out the last 20\\ of the series as a test set.


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


``` python
fig, ax = plt.subplots()
ax.plot(t_train, y_train, color="C0", label="train")
ax.plot(t_test, y_test, color="C1", label="test")
ax.axvline(float(t_test[0]), color="gray", linestyle="--", label="train/test split")
ax.legend(loc="upper left")
ax.set(title="Synthetic time series", xlabel="time", ylabel="y");
```


<figure class="figure">
<p><img src="exponential_smoothing_state_space_files/figure-html/_src-exponential_smoothing_state_space-cell-4-output-1.png" class="figure-img" width="1011" height="611" /></p>
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

It threads a *carry* (the running state) through a step function `f`, and stacks the per-step outputs. The [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) building block runs two such scans for us, but the state update it threads through them is ours to write, so it pays to see the mechanics once.


## A simple example

As a warm-up, we use `scan` to compute the geometric damping sum \varphi_h = \varphi + \varphi^2 + \cdots + \varphi^h that appears in the damped-trend forecast formula below. The carry holds the running sum and the current power of \varphi.


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

The model is a plain NumPyro function `(covariates, data=None)`. Its first line derives the per-call [`Horizon`](https://juanitorduz.github.io/numpyro_forecast/reference/models.Horizon.html) from the shapes (the observed data `h.data`, the number of in-sample steps `h.t_obs`, and the forecast length `h.future`), and the recursion goes to the [`ssoe`](https://juanitorduz.github.io/numpyro_forecast/reference/models.ssoe.html) building block. The block takes the driving series `y` (sliced from the covariates, see the design note above), the initial state, a [mean](../../reference/typing.Array.md#numpyro_forecast.typing.Array.mean) function, an `update` function, and the innovation distribution, and it owns the two scans, neither of which contains a NumPyro sample site:

1.  **In sample.** A deterministic filter consumes the observed series: at each step `mean(carry, x_t)` returns the one-step-ahead mean \mu_t and `update(carry, y_t, eps_t, x_t)` advances the state with the innovation \varepsilon_t = y_t - \mu_t. The means come back as `r.mu`; the whole in-sample likelihood is then a single `Normal` observation site `"obs"` against them, and we also expose \mu_t as the deterministic site `"mu"` for the in-sample fit plot.
2.  **Out of sample.** When `h.future > 0` the block draws the horizon innovations from the prior at a separate `"eps_future"` site (under its own `time_future` plate), rolls the state forward from the final in-sample state feeding those innovations back through `update`, and returns the trajectory as `r.y_future`, which we register as the deterministic `"forecast"` site the package's [forecast](../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast) driver reads. Because `"eps_future"` does not exist while training, `Predictive` draws it from the prior at forecast time, exactly like the built-in `_future` sites.

The state update `advance` is shared by both scans and is the SSOE update above, one innovation driving level, trend, and seasonality. One shape convention to know: rows carry the observation axis, so the scalar state emits a `(1,)` mean (`mu[None]`) and reads the scalar innovation back out of the `(1,)` error (`eps_t[0]`); the block checks these shapes so a mismatch fails loudly instead of broadcasting silently.

The priors follow the source post: \text{Beta}(5, 5) on the level, trend, and seasonal smoothing parameters (flat enough near the boundaries to avoid a funnel-shaped posterior), \text{Beta}(2, 5) on the damping factor (favoring some damping), a tight \text{HalfNormal}(0.5) on the noise, and weakly informative priors on the initial states.


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

    def mean(carry, _):
        level, trend, seasonality = carry
        # Rows carry the observation axis: emit a (1,) mean.
        return (level + phi * trend + seasonality[0])[None]

    def update(carry, y_t, eps_t, _):
        # Read the scalar error back from the (1,) row.
        return advance(carry, eps_t[0])

    init_state = (level_init, trend_init, seasonality_init)
    r = ssoe(h, "eps", y, init_state, mean, update, dist.Normal(0, noise))

    numpyro.deterministic("mu", r.mu)
    numpyro.sample("obs", dist.Normal(r.mu, noise), obs=h.data)
    if h.future > 0:
        numpyro.deterministic("forecast", r.y_future)
```


# Priors

Before fitting, it is worth looking at the priors on the bounded parameters. The \text{Beta}(5, 5) prior on the smoothing parameters is symmetric and concentrated away from 0 and 1, which keeps the sampler away from the boundary regions where the posterior geometry degenerates. The \text{Beta}(2, 5) prior on the damping factor \varphi puts more mass below 0.5, encoding a mild preference for damped (non-explosive) trends.


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
<p><img src="exponential_smoothing_state_space_files/figure-html/_src-exponential_smoothing_state_space-cell-7-output-1.png" class="figure-img" width="1131" height="788" /></p>
</figure>


# Inference

We fit the model with plain NumPyro: the NUTS sampler through `MCMC`, running 4 chains of 2{,}000 warmup and 2{,}000 sampling steps each on the training window. The model is an ordinary NumPyro callable, so nothing package-specific happens here; `mcmc.get_samples()` returns the posterior draws as a plain dictionary with the chains flattened together, which is the format every package driver consumes.

We then export the draws into an ArviZ-schema `xarray.DataTree` with [`to_datatree`](https://juanitorduz.github.io/numpyro_forecast/reference/convert.to_datatree.html): a single call restores the `(chain, draw)` structure (we pass `num_chains=4`), samples the in-sample one-step-ahead posterior predictive from the same draws, and, because we hand it the *full-horizon* covariates, also runs the forecast and stores it in the `predictions` group. Everything downstream (diagnostics, trace plots, the in-sample fit, the forecast, the metrics) reads from this one object.


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
│           level_init              (chain, draw) float32 32kB 0.1042 0.2372 ... 0.5154
│           level_smoothing         (chain, draw) float32 32kB 0.2262 0.2147 ... 0.2145
│           mu                      (chain, draw, time, obs_dim) float32 6MB 0.9221 ....
│           noise                   (chain, draw) float32 32kB 0.2476 0.2276 ... 0.2401
│           phi                     (chain, draw) float32 32kB 0.3645 0.2896 ... 0.288
│           seasonality_init        (chain, draw, seasonality_init_dim_0) float32 480kB ...
│           seasonality_smoothing   (chain, draw) float32 32kB 0.2684 0.1636 ... 0.3114
│           trend_init              (chain, draw) float32 32kB 0.1338 ... -0.06602
│           trend_smoothing         (chain, draw) float32 32kB 0.52 0.549 ... 0.3691
│       Attributes:
│           created_at:                 2026-09-29T18:38:34.452157+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
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
│           obs      (chain, draw, time, obs_dim) float32 6MB 0.6162 1.077 ... 2.481
│       Attributes:
│           created_at:                 2026-09-29T18:38:34.615487+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
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
│           created_at:                 2026-09-29T18:38:34.615766+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
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
│           created_at:                 2026-09-29T18:38:34.615949+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
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
│           obs      (chain, draw, time, obs_dim) float32 2MB 3.382 3.099 ... 4.016
│       Attributes:
│           created_at:                 2026-09-29T18:38:34.818801+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
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
            created_at:                 2026-09-29T18:38:34.819034+00:00
            creation_library:           ArviZ
            creation_library_version:   1.2.0
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


0.1042 0.2372 ... 0.6006 0.5154


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[ 0.10419866,  0.23723318,  0.4159305 , ...,  0.38825178,0.01595362,  0.34321716],[-0.00274092,  0.14366099,  0.15499333, ...,  0.25324833,0.45404178,  0.09265403],[ 0.36121386,  0.25289512,  0.6647955 , ...,  0.49127114,0.12309081,  0.09964179],[-0.0728039 ,  0.00129045, -0.20990974, ...,  0.54472375,0.6005941 ,  0.51536757]], shape=(4, 2000), dtype=float32)


level_smoothing


(chain, draw)


float32


0.2262 0.2147 ... 0.2143 0.2145


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.22624171, 0.21470328, 0.2660817 , ..., 0.16232786, 0.17668948,0.22966792],[0.15618388, 0.3592594 , 0.3393285 , ..., 0.2553746 , 0.22751172,0.16329992],[0.2611294 , 0.16206685, 0.16251494, ..., 0.31208825, 0.16539477,0.1609887 ],[0.28040612, 0.28333038, 0.26458067, ..., 0.26773453, 0.2143182 ,0.21445972]], shape=(4, 2000), dtype=float32)


mu


(chain, draw, time, obs_dim)


float32


0.9221 0.9667 ... 2.106 2.628


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[0.9220529 ],[0.9667024 ],[0.7625237 ],...,[1.6081431 ],[2.1148975 ],[2.6381013 ]],[[1.2196057 ],[1.0338279 ],[0.7646363 ],...,[1.5784531 ],[2.0995333 ],[2.6718802 ]],[[1.0701003 ],[0.9431573 ],[0.7200458 ],...,......,[1.5881068 ],[2.1150856 ],[2.6450348 ]],[[1.1869941 ],[0.91616917],[0.91862464],...,[1.6005028 ],[2.1049044 ],[2.6346605 ]],[[0.9921522 ],[1.0696038 ],[0.8507924 ],...,[1.604358  ],[2.1064298 ],[2.6281335 ]]]], shape=(4, 2000, 192, 1), dtype=float32)


noise


(chain, draw)


float32


0.2476 0.2276 ... 0.2327 0.2401


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.24760577, 0.22757246, 0.23447123, ..., 0.26424727, 0.24297984,0.22624843],[0.24174379, 0.24331711, 0.23262066, ..., 0.24136879, 0.22960313,0.24420048],[0.2099352 , 0.22936922, 0.22941123, ..., 0.24278316, 0.20342404,0.20337866],[0.24996549, 0.24944374, 0.23631328, ..., 0.23824202, 0.23274848,0.2400895 ]], shape=(4, 2000), dtype=float32)


phi


(chain, draw)


float32


0.3645 0.2896 ... 0.2591 0.288


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.36446515, 0.28956103, 0.23259558, ..., 0.28556445, 0.47843373,0.18105975],[0.39809066, 0.08358195, 0.12337512, ..., 0.14110307, 0.17085293,0.5510389 ],[0.18778245, 0.0940506 , 0.04929074, ..., 0.10497456, 0.37408975,0.4272081 ],[0.0769428 , 0.06880829, 0.0742364 , ..., 0.05830554, 0.25911584,0.28797135]], shape=(4, 2000), dtype=float32)


seasonality_init


(chain, draw, seasonality_init_dim_0)


float32


0.7691 0.7424 ... 0.5333 0.7794


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.7690977 , 0.7423962 , 0.48268154, ..., 0.17750677,0.60482   , 0.8651934 ],[0.97234535, 0.80816233, 0.5133209 , ..., 0.66336286,0.8345956 , 1.0314555 ],[0.65797466, 0.51701146, 0.23707618, ..., 0.4230631 ,0.56319827, 0.6827384 ],...,[0.9675257 , 0.9212727 , 0.7016902 , ..., 0.47644156,0.82938576, 0.9455309 ],[0.62027717, 0.84675044, 0.41220933, ..., 0.66492724,0.5823811 , 0.7575279 ],[1.079485  , 0.37168446, 0.48320892, ..., 0.40553328,0.6471511 , 1.1545092 ]],[[1.020512  , 0.9204883 , 0.45346698, ..., 0.7121212 ,0.8844024 , 1.0034366 ],[1.1039293 , 1.1254457 , 0.57052124, ..., 0.78920054,1.047892  , 1.2724224 ],[1.1238862 , 1.0006001 , 0.6613307 , ..., 0.660245  ,1.1096838 , 1.1712556 ],...[0.6079955 , 0.64003414, 0.28757003, ..., 0.33240077,0.5583231 , 0.8702638 ],[0.999418  , 0.9290164 , 0.6641747 , ..., 0.67095464,1.0940216 , 1.3097427 ],[1.0867814 , 0.844012  , 0.65076435, ..., 0.668195  ,1.095214  , 1.4083985 ]],[[1.2985042 , 1.2902628 , 0.62111026, ..., 0.61751497,1.2478014 , 1.4559258 ],[1.2429271 , 1.3439109 , 0.6987881 , ..., 0.5927794 ,1.3236308 , 1.3063539 ],[1.3101493 , 1.020987  , 1.0945846 , ..., 1.0620263 ,0.9038289 , 1.4255853 ],...,[0.49432114, 0.5656797 , 0.08171522, ..., 0.1875057 ,0.31611827, 0.62714475],[0.6208315 , 0.3743747 , 0.32757807, ..., 0.10014091,0.5356124 , 0.6185468 ],[0.49579635, 0.54811656, 0.31394035, ..., 0.1840206 ,0.53331816, 0.77944636]]], shape=(4, 2000, 15), dtype=float32)


seasonality_smoothing


(chain, draw)


float32


0.2684 0.1636 ... 0.2743 0.3114


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.26841366, 0.1636042 , 0.3275245 , ..., 0.39006045, 0.36041015,0.22131774],[0.23869939, 0.26719034, 0.26048726, ..., 0.25659588, 0.3151462 ,0.16643165],[0.35547313, 0.29899967, 0.27875587, ..., 0.22429663, 0.2924037 ,0.31231588],[0.18488838, 0.19805086, 0.23869461, ..., 0.27799624, 0.27426144,0.31144494]], shape=(4, 2000), dtype=float32)


trend_init


(chain, draw)


float32


0.1338 0.03463 ... -0.1329 -0.06602


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[ 0.13377564,  0.03462896, -0.01635815, ...,  0.03220195,0.08328503, -0.05337352],[ 0.08519433,  0.1680065 ,  0.23231424, ..., -0.06383547,0.05015887,  0.08788078],[ 0.10370061,  0.07952344,  0.09260137, ..., -0.00105917,-0.01138956,  0.02945486],[ 0.03098879,  0.03413826,  0.00725304, ..., -0.0540663 ,-0.13288099, -0.06601943]], shape=(4, 2000), dtype=float32)


trend_smoothing


(chain, draw)


float32


0.52 0.549 0.4181 ... 0.3652 0.3691


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.51995736, 0.54897714, 0.41810217, ..., 0.51259893, 0.49682853,0.683832  ],[0.49396294, 0.51195127, 0.5175259 , ..., 0.5915873 , 0.50900275,0.595831  ],[0.6272169 , 0.61239886, 0.7523629 , ..., 0.44114438, 0.66640556,0.6754959 ],[0.72191775, 0.72342455, 0.7026536 , ..., 0.70629436, 0.36519745,0.36912572]], shape=(4, 2000), dtype=float32)


Attributes: (5)


created_at :  
2026-09-29T18:38:34.452157+00:00

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


0.6162 1.077 0.7036 ... 1.954 2.481


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[0.6161675 ],[1.0765831 ],[0.703573  ],...,[1.7345343 ],[2.2651303 ],[2.4996247 ]],[[0.98557746],[1.1951092 ],[0.9033631 ],...,[1.377877  ],[2.1653142 ],[1.9921607 ]],[[1.0524905 ],[0.72374094],[0.885857  ],...,......,[1.8579603 ],[2.208094  ],[2.9309947 ]],[[1.0133085 ],[0.86266255],[1.1651939 ],...,[1.4586989 ],[2.1493025 ],[2.761009  ]],[[0.9707767 ],[1.1111313 ],[0.9426878 ],...,[1.5432662 ],[1.9541917 ],[2.4813643 ]]]], shape=(4, 2000, 192, 1), dtype=float32)


Attributes: (5)


created_at :  
2026-09-29T18:38:34.615487+00:00

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
2026-09-29T18:38:34.615766+00:00

creation_library :  
ArviZ

creation_library_version :  
1.2.0

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
2026-09-29T18:38:34.615949+00:00

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


3.382 3.099 3.126 ... 3.688 4.016


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[3.3820145],[3.099333 ],[3.1263812],...,[2.9227772],[2.3228955],[2.861484 ]],[[2.7037864],[3.1599677],[3.761728 ],...,[2.0886521],[2.600681 ],[2.658366 ]],[[2.8767614],[3.2048326],[3.3531222],...,......,[2.5722287],[3.1317437],[3.294363 ]],[[2.6926036],[3.643505 ],[3.8255873],...,[2.8328743],[3.0727634],[3.514503 ]],[[2.7136898],[3.363443 ],[3.726202 ],...,[3.4968035],[3.6884425],[4.015752 ]]]], shape=(4, 2000, 48, 1), dtype=float32)


Attributes: (5)


created_at :  
2026-09-29T18:38:34.818801+00:00

creation_library :  
ArviZ

creation_library_version :  
1.2.0

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
2026-09-29T18:38:34.819034+00:00

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


# Diagnostics

With the chains restored in the tree, ArviZ's convergence diagnostics apply directly: \hat{R} and the bulk and tail effective sample sizes for the scalar parameters.


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
| level_smoothing       | 1.001 | 3309.0   | 3644.0   |
| trend_smoothing       | 1.001 | 4573.0   | 4570.0   |
| seasonality_smoothing | 1.000 | 2706.0   | 3398.0   |
| phi                   | 1.002 | 3449.0   | 3634.0   |
| noise                 | 1.001 | 3417.0   | 4104.0   |
| level_init            | 1.008 | 825.0    | 1547.0   |
| trend_init            | 1.001 | 4043.0   | 4609.0   |


The \hat{R} values are close to 1 and the effective sample sizes are healthy, which indicates that the chains have mixed well. This is the payoff of the state space parameterization together with the tuned priors: the posterior geometry is well behaved and the sampler explores it without trouble. The trace plots below confirm the good mixing.


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
<p><img src="exponential_smoothing_state_space_files/figure-html/_src-exponential_smoothing_state_space-cell-10-output-1.png" class="figure-img" width="1011" height="1663" /></p>
</figure>


# Forecast

The tree already holds both predictive ensembles, one draw per posterior sample. The `posterior_predictive` group is the in-sample one-step-ahead predictive of the `"obs"` site: the fitted mean \mu_t plus observation noise. The `predictions` group holds the forecast over the test horizon: for each posterior draw the model replayed the in-sample filter, then rolled the state forward while sampling fresh innovations. We stack the `(chain, draw)` dimensions of each into a single sample axis to get the draws-first layout the plotting and scoring helpers expect. One consequence worth noting: the bands and the metrics below use all 8{,}000 forecast paths, one per posterior draw, rather than a thinned subset, so the plotted ensemble and the scored ensemble are the same.


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


We visualize both the in-sample fit and the forecast with `az.plot_lm`, showing the 50\\ and 94\\ HDI bands (packing each ensemble with the package's [predictions_to_datatree](../../reference/convert.predictions_to_datatree.md#numpyro_forecast.convert.predictions_to_datatree)). The forecast band (in orange) clearly fans out as the horizon grows: this is the calibrated uncertainty that the innovations state space form provides.


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
<p><img src="exponential_smoothing_state_space_files/figure-html/_src-exponential_smoothing_state_space-cell-12-output-1.png" class="figure-img" width="1211" height="711" /></p>
</figure>


# Evaluation

Finally, we score the forecast against the held-out test set with the package's evaluation metrics: mean absolute error and root mean squared error (point-forecast accuracy), the continuous ranked probability score (a proper score for the whole predictive distribution), and the empirical coverage of the central 90\\ interval (calibration).


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


                 MAE: 0.2432
                RMSE: 0.2820
                CRPS: 0.1657
      coverage (90%): 0.9583


The coverage of the central 90\\ interval sits close to its nominal level, confirming that the forecast is well calibrated. For a systematic assessment over multiple origins you would reach for `numpyro_forecast.backtest`, which refits the model on a moving window (the [ARMA example](https://juanitorduz.github.io/numpyro_forecast/docs/examples/arma.html) does exactly that with the same building block); we omit it here because it retrains the full sampler for every window.


# References

- Hyndman, R. J., & Athanasopoulos, G. (2021). [*Forecasting: Principles and Practice*](https://otexts.com/fpp3/), 3rd edition. Chapters on exponential smoothing and ETS models.
- Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
- Orduz, J. [*Exponential Smoothing with NumPyro: State Space Form*](https://juanitorduz.github.io/exponential_smoothing_numpyro_ssm/).
- Orduz, J. [*Notes on Exponential Smoothing with NumPyro*](https://juanitorduz.github.io/exponential_smoothing_numpyro/).

[Source: Exponential Smoothing in State Space Form with `numpyro_forecast`](_src/exponential_smoothing_state_space-preview.html#9b8886b7)
