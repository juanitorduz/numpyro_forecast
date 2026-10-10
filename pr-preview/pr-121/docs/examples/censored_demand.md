# Demand Forecasting with Censored Likelihood


Demand Forecasting with Censored Likelihood with `numpyro_forecast`

This notebook ports the blog post [**Demand Forecasting with Censored Likelihood**](https://juanitorduz.github.io/demand/) to the [`numpyro_forecast`](https://github.com/juanitorduz/numpyro_forecast) package. The subject is a fact of retail life: recorded sales understate demand in two distinct ways. When the product is out of stock, a day of genuine demand is recorded as zero sales. And when demand exceeds what the shelf (or the supply) can carry, the register stops counting at the capacity cap, so the recorded number is a *lower bound* on what customers actually wanted. A model trained naively on such a series learns to forecast *sales*, but replenishment and capacity planning need a forecast of *demand*: order against sales and you bake yesterday's stockouts into tomorrow's assortment, systematically under-serving your best days.

We simulate a demand series from an AR(2) process with weekly seasonality, corrupt it into observed sales through random stockouts and a hard capacity cap, and then fit an AR(2) model with Fourier seasonality **whose likelihood knows about the censoring**: below the cap an observation contributes the usual \text{Normal} density, and at the cap it contributes the *survival mass* P(\text{demand} \geq \text{cap}), the probability that latent demand was at least as large as the recorded bound. Days with the product off the shelf are masked out of the likelihood entirely. Because the data are simulated, the true demand is known and the claim "the censored likelihood recovers demand" can be checked against ground truth rather than asserted.

This example completes a trio of availability mechanisms in this documentation. The [availability TSB example](https://juanitorduz.github.io/numpyro_forecast/docs/examples/availability_tsb.html) freezes its recursion updates when the product is off the shelf, and the [fresh retail stockout example](https://juanitorduz.github.io/numpyro_forecast/docs/examples/fresh_retail_stockout.html) scales the mean by a saturating availability factor; its next-steps list asks for precisely the model built here. The closing section compares the three mechanisms side by side.


# Prepare notebook


``` python
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import xarray as xr
from jax import random
from jaxtyping import Float
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from numpyro.infer import MCMC, NUTS

from numpyro_forecast import (
    Horizon,
    eval_coverage,
    eval_crps,
    eval_mae,
    eval_rmse,
    evaluate_forecast,
    predictions_to_datatree,
    ssoe,
    to_datatree,
)
from numpyro_forecast.arrays import pad_future
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


# Generate data

We reproduce the blog post's data generating process. Latent demand follows an AR(2) recursion with a weekly sinusoid, clipped at zero:

d_t = \max\left(0, \\ \phi_1 \\ d\_{t-1} + \phi_2 \\ d\_{t-2} + \gamma \sin\left(\frac{2\pi t}{7}\right) + \alpha + \varepsilon^d_t\right), \qquad \varepsilon^d_t \sim \text{Normal}(0, \sigma_d).

Sales are demand minus a friction term and noise, never exceeding demand and never negative:

s_t = \max\left(0, \\ \min\left(d_t + \varepsilon^s_t - \delta, \\ d_t\right)\right), \qquad \varepsilon^s_t \sim \text{Normal}(0, \sigma_s).

Observed sales gate the sales through an availability coin flip and cap them at the shelf capacity:

a_t \sim \text{Bernoulli}(0.8), \qquad y_t = \min\left(a_t \\ s_t, \\ y\_{\max}\right), \qquad y\_{\max} = 2.2.

It pays to keep the four quantities straight, because the whole example is about the gaps between them:

- **Latent demand** d_t is what customers want on day t. It is never observed directly, and it is the number planning cares about.
- **Sales** s_t are what would sell with the product fully available: demand minus real-world friction (a customer walks away, a basket is abandoned), which is why s_t \leq d_t always.
- **Observed sales** y_t are the only column a transaction database records: sales zeroed out on stockout days and truncated at the capacity cap.
- **Availability** a_t says whether the product was on the shelf at all. Retail systems typically know this (or can reconstruct it from inventory snapshots), which is what makes the model below feasible in practice.

From the observed series we also derive the **censoring indicator** c_t = \mathbb{1}\\y_t = y\_{\max}\\: on those days the register hit the cap, so the recorded value is a lower bound on sales rather than a measurement of them. Production data would flag y_t \geq y\_{\max} instead; the exact equality is safe here only because the simulation's `minimum` returns the cap bit-exactly.


``` python
@dataclass(frozen=True)
class DemandParams:
    """Parameters of the demand and sales data generating process.

    Attributes
    ----------
    n_periods
        Number of days to simulate.
    phi_1, phi_2
        AR(2) coefficients of the latent demand recursion.
    seasonal_amplitude
        Amplitude of the weekly sinusoid in the demand recursion.
    seasonal_period
        Seasonal period in days.
    intercept
        Constant term of the demand recursion.
    demand_init
        Initial demand level seeding both lags of the recursion.
    demand_noise, sales_noise
        Standard deviations of the demand and sales noise terms.
    demand_sales_delta
        Friction subtracted from demand when generating sales.
    availability_rate
        Probability that the product is on the shelf on a given day.
    max_capacity
        Hard cap on recorded daily sales.
    """

    n_periods: int = 180
    phi_1: float = 0.6
    phi_2: float = 0.3
    seasonal_amplitude: float = 0.6
    seasonal_period: int = 7
    intercept: float = 0.2
    demand_init: float = 2.0
    demand_noise: float = 0.3
    sales_noise: float = 0.5
    demand_sales_delta: float = 0.25
    availability_rate: float = 0.8
    max_capacity: float = 2.2


class DemandData(NamedTuple):
    """Simulated series of the demand and sales process.

    Attributes
    ----------
    demand
        Latent demand series.
    sales
        Sales under full availability.
    sales_obs
        Observed sales: availability-gated and capacity-capped.
    is_available
        Availability indicator (1 if the product was on the shelf).
    """

    demand: Float[Array, " t"]
    sales: Float[Array, " t"]
    sales_obs: Float[Array, " t"]
    is_available: Float[Array, " t"]


def generate_demand_sales(rng_key: Array, params: DemandParams) -> DemandData:
    """Simulate latent demand, sales, and observed sales.

    Parameters
    ----------
    rng_key
        PRNG key for the noise terms and the availability draws.
    params
        Parameters of the data generating process.

    Returns
    -------
    DemandData
        The four simulated series, each of length ``params.n_periods``.
    """
    key_demand, key_sales, key_avail = random.split(rng_key, 3)
    noise_demand = params.demand_noise * random.normal(key_demand, (params.n_periods,))
    noise_sales = params.sales_noise * random.normal(key_sales, (params.n_periods,))
    is_available = random.bernoulli(
        key_avail, params.availability_rate, (params.n_periods,)
    ).astype(jnp.float32)
    t_grid = jnp.arange(params.n_periods, dtype=jnp.float32)

    def dgp_step(carry, xs):
        demand_prev_1, demand_prev_2 = carry
        t, eps_demand, eps_sales = xs
        seasonal = params.seasonal_amplitude * jnp.sin(2 * jnp.pi * t / params.seasonal_period)
        demand_t = jnp.clip(
            params.phi_1 * demand_prev_1
            + params.phi_2 * demand_prev_2
            + seasonal
            + params.intercept
            + eps_demand,
            min=0.0,
        )
        sales_t = jnp.clip(
            jnp.minimum(demand_t + eps_sales - params.demand_sales_delta, demand_t),
            min=0.0,
        )
        return (demand_t, demand_prev_1), (demand_t, sales_t)

    init = (jnp.asarray(params.demand_init), jnp.asarray(params.demand_init))
    _, (demand, sales) = jax.lax.scan(dgp_step, init, (t_grid, noise_demand, noise_sales))
    sales_obs = jnp.minimum(is_available * sales, params.max_capacity)
    return DemandData(demand=demand, sales=sales, sales_obs=sales_obs, is_available=is_available)


params = DemandParams()
rng_key, rng_subkey = random.split(rng_key)
demand, sales, sales_obs, is_available = generate_demand_sales(rng_subkey, params)

censored = (sales_obs == params.max_capacity).astype(jnp.float32)
time = np.arange(params.n_periods)

n_stockout = int((1 - is_available).sum())
n_censored = int(censored.sum())
print(f"periods: {params.n_periods}")
print(f"stockout days: {n_stockout} ({n_stockout / params.n_periods:.0%})")
print(f"capacity-censored days: {n_censored} ({n_censored / params.n_periods:.0%})")
```


    periods: 180
    stockout days: 49 (27%)
    capacity-censored days: 36 (20%)


The plot splits the story into its two steps. The top panel is the market: latent demand (black) with sales (blue) hugging it from below by the friction term. The bottom panel is the database: observed sales (orange) collapse to zero wherever the gray availability line (right axis) drops off the shelf, and flatline at the dashed capacity cap wherever sales would have exceeded it, with the censored days marked.


``` python
fig, (ax_top, ax_bot) = plt.subplots(
    nrows=2, ncols=1, sharex=True, sharey=True, figsize=(12, 8), layout="constrained"
)
ax_top.plot(time, np.asarray(demand), color="black", lw=1.5, label="latent demand")
ax_top.plot(time, np.asarray(sales), color="C0", lw=1, label="sales")
ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
ax_top.set(ylabel="units")

ax_bot.plot(time, np.asarray(sales), color="C0", lw=1, label="sales")
ax_bot.plot(time, np.asarray(sales_obs), color="C1", lw=1, label="observed sales")
ax_bot.scatter(
    time[np.asarray(censored) == 1],
    np.asarray(sales_obs)[np.asarray(censored) == 1],
    color="C3",
    s=20,
    zorder=5,
    label="censored at capacity",
)
ax_bot.axhline(params.max_capacity, color="C3", ls="--", lw=1.5, label="capacity cap")
ax_twin = ax_bot.twinx()
ax_twin.plot(
    time,
    np.asarray(is_available),
    color="gray",
    lw=1,
    alpha=0.7,
    drawstyle="steps-mid",
    label="availability",
)
ax_twin.grid(False)
bot_handles, bot_labels = ax_bot.get_legend_handles_labels()
twin_handles, twin_labels = ax_twin.get_legend_handles_labels()
ax_bot.legend(
    bot_handles + twin_handles,
    bot_labels + twin_labels,
    loc="center left",
    bbox_to_anchor=(1.06, 0.5),
)
ax_bot.set(xlabel="time", ylabel="units")
fig.suptitle("Demand and sales simulation", fontsize=16, fontweight="bold");
```


<figure class="figure">
<p><img src="censored_demand_files/figure-html/_src-censored_demand-cell-4-output-1.png" class="figure-img" width="1211" height="811" /></p>
</figure>


# Train-test split and covariates

We hold out the last 30 days as the test window. Throughout the package, time lives at axis `-2` and the observation dimension at axis `-1`, so the training data has shape `(150, 1)`.

The covariates carry everything the model needs over the **full** duration, in a `(180, 7)` tensor. The package infers the forecast horizon from the shapes: covariates longer than the data by 30 rows means a 30-step forecast.

- Column `0` is the observed sales history the AR recursion filters. The history has to travel through the covariates rather than the data argument: the package's [predict_in_sample](../../reference/predictive.predict_in_sample.md#numpyro_forecast.predictive.predict_in_sample) and [to_datatree](../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) call the model with `data=None` (they sample the observation site), while the recursion always needs the observed series. Only the first `t_obs` rows are ever read (the [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) building block below checks this), so the future rows are zeroed out and no information leaks.
- Column `1` is the availability mask and column `2` the censoring indicator, read only over the training rows. One honest caveat versus the availability TSB and fresh retail examples, whose recursions consume these inputs over the horizon too: here the model pads its own horizon inputs (availability one, censoring zero), so the forecast is *structurally* uncensored demand-scale sales, the number a planner should order against, regardless of what the trailing rows contain. We still pin the trailing 30 rows to the same values so the scenario travels with the forecast into the tree's `predictions_constant_data`, as documentation rather than as a model input.
- Columns `3:7` are weekly Fourier features (two harmonics, sines then cosines) from the package's [`fourier_features`](https://juanitorduz.github.io/numpyro_forecast/reference/features.fourier_features.html) helper, the only columns the model reads over the horizon.


``` python
forecast_horizon = 30
n_train = params.n_periods - forecast_horizon

train_data = sales_obs[:n_train][:, None]
demand_test = demand[n_train:][:, None]
sales_obs_test = sales_obs[n_train:][:, None]
t_test = time[n_train:]

n_order = 2
fourier = fourier_features(params.n_periods, float(params.seasonal_period), n_order)

in_train = jnp.arange(params.n_periods) < n_train
covariates = jnp.concatenate(
    [
        jnp.where(in_train, sales_obs, 0.0)[:, None],  # history; future rows never read
        jnp.where(in_train, is_available, 1.0)[:, None],  # scenario: fully available
        jnp.where(in_train, censored, 0.0)[:, None],  # scenario: uncensored
        fourier,
    ],
    axis=-1,
)
covariates_train = covariates[:n_train]
print(f"train data shape: {train_data.shape}, full covariates shape: {covariates.shape}")
```


    train data shape: (150, 1), full covariates shape: (180, 7)


# Model specification

The mean recursion is an AR(2) on *filtered* lags \tilde{y}\_t (defined below) plus a Fourier seasonal term:

\hat{y}\_t = \mu + \phi_1 \\ \tilde{y}\_{t-1} + \phi_2 \\ \tilde{y}\_{t-2} + \mathbf{f}\_t^\top \boldsymbol{\beta},

with priors

\begin{align\*} \mu & \sim \text{Normal}(1, 1), \\ \phi_1, \phi_2 & \sim \text{Normal}(0, 1), \\ \boldsymbol{\beta} & \sim \text{Normal}(0, 1), \\ \sigma & \sim \text{HalfNormal}(1). \end{align\*}

The likelihood is where the censoring lives. An uncensored day contributes the usual density; a censored day only tells us that latent sales were *at least* the cap, so it contributes the survival mass above the recorded value:

p(y_t \mid \hat{y}\_t, \sigma) = \text{Normal}(y_t \mid \hat{y}\_t, \sigma)^{1 - c_t} \left\[1 - \Phi\left(\frac{y_t - \hat{y}\_t}{\sigma}\right)\right\]^{c_t},

where \Phi is the standard \text{Normal} CDF, and the whole term is masked out on stockout days (a_t = 0), which carry no demand information. NumPyro ships this construction (from version `0.20.0`) as [`RightCensoredDistribution`](https://num.pyro.ai/en/stable/distributions.html#censored-distributions): its `log_prob` is exactly the expression above (with a numerical-stability clip on the CDF), and its `sample` draws from the *uncensored* base distribution, which is precisely what we want posterior predictive draws to describe. The whole likelihood is then a single vectorized `"obs"` site.

**Filtering through the gaps.** Masking the likelihood is only half the treatment of a corrupted day, because an autoregression also consumes every day as a *lagged value*. A recorded stockout zero fed into the carry masquerades as a demand crash: it drags the next day's prediction down, attenuates the AR coefficients (post-gap days look like violent rebounds no moderate \phi can explain), and inflates \sigma to absorb the damage. The recursion therefore carries a filtered series instead of the raw observations:

\tilde{y}\_t = a_t \left\[(1 - c_t) \\ y_t + c_t \max\left(y_t, \hat{y}\_t\right)\right\] + (1 - a_t) \\ \hat{y}\_t.

Clean on-shelf days pass the observation through; capped days floor the lag at the model's own prediction, since the truth is at least the cap; off-shelf days carry the prediction itself, the model's best estimate of the demand nobody could express. This is the same one-step-ahead logic a state space filter applies to missing observations, done with a plug-in mean instead of a full state distribution.

The model is a plain NumPyro function `(covariates, data=None)` that derives its train/forecast split from the shapes with [`Horizon.from_data`](https://juanitorduz.github.io/numpyro_forecast/reference/models.Horizon.html) and hands the recursion to the [`ssoe`](https://juanitorduz.github.io/numpyro_forecast/reference/models.ssoe.html) building block, the single-source-of-error recursion shared with the [ARMA](https://juanitorduz.github.io/numpyro_forecast/docs/examples/arma.html) and [exponential smoothing](https://juanitorduz.github.io/numpyro_forecast/docs/examples/exponential_smoothing_state_space.html) examples. [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) takes the driving series, an initial carry, a [mean](../../reference/typing.Array.md#numpyro_forecast.typing.Array.mean) function, an `update` function, and the innovation distribution. `mean(carry, x_t)` returns the one-step-ahead mean \hat{y}\_t and `update(carry, y_t, eps_t, x_t)` builds the next carry from the day's value and its error; calling [mean](../../reference/typing.Array.md#numpyro_forecast.typing.Array.mean) again inside `update` (the same expression on the same inputs, so exactly the filter's prediction) is what lets the lag filter above floor capped days at \hat{y}\_t. Rows carry the observation axis, so the two placeholder lags are `y[0]` with shape `(1,)`, the mean has shape `(1,)`, and a scalar state would emit `mu[None]`; the block checks these shapes. The block owns two scans, neither containing a sample site:

1.  **In sample.** A deterministic `jax.lax.scan` runs [mean](../../reference/typing.Array.md#numpyro_forecast.typing.Array.mean) over the observed history, feeding each observation and its error \varepsilon_t = y_t - \hat{y}\_t through `update`, and returns the one-step-ahead means as `r.mu` (exposed as the deterministic site `"pred_mean"`); the `"obs"` site conditions the data on them through the censored likelihood. The AR(2) needs two lags, so the first two steps run on placeholder lags and are masked out of the likelihood.
2.  **Out of sample.** When `h.future > 0` the block draws the horizon innovations at the `"eps_future"` site (under its own `time_future` plate), rolls the recursion forward from the final filtered lags with each *sampled* value \hat{y}\_t + \varepsilon_t fed back through `update`, and returns the trajectory as `r.y_future`, which we register (clipped at zero, since demand is nonnegative) as the deterministic `"forecast"` site the package reads. Since `"eps_future"` does not exist during training, `Predictive` draws it from the prior at forecast time and the uncertainty compounds over the horizon exactly as the generative process says it should. The availability and censoring inputs are padded over the horizon with [`pad_future`](https://juanitorduz.github.io/numpyro_forecast/reference/arrays.pad_future.html) (available, uncensored), so no gate or cap applies there: the forecast is of latent demand-scale sales, unconstrained by the cap.

One small difference from the blog post's hand-rolled scans: the same `update` serves both scans, so the clip at zero now applies to the filtered lag in sample as well, which only matters on stockout days whose prediction is negative.


``` python
def ar2_seasonal(covariates: Array, data: Array | None = None) -> None:
    """Censored AR(2) model with weekly Fourier seasonality.

    Parameters
    ----------
    covariates
        Seven-input tensor ``(duration, 7)`` spanning the full horizon: column
        ``0`` is the observed sales history (only the first ``h.t_obs`` rows
        are read), column ``1`` the availability mask, column ``2`` the
        censoring indicator, and columns ``3:7`` the weekly Fourier features
        (the only columns read over the forecast horizon).
    data
        Observed sales with time at axis ``-2``, or ``None`` when the drivers
        sample the observation site.
    """
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, 0:1]  # observed history only; never reads beyond t_obs
    available = covariates[..., : h.t_obs, 1:2]
    censored = covariates[..., : h.t_obs, 2:3]
    fourier = covariates[..., 3:]

    mu = numpyro.sample("mu", dist.Normal(loc=1, scale=1))
    phi_1 = numpyro.sample("phi_1", dist.Normal(loc=0, scale=1))
    phi_2 = numpyro.sample("phi_2", dist.Normal(loc=0, scale=1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=1))
    with numpyro.plate("fourier_modes", fourier.shape[-1]):
        beta_seasonal = numpyro.sample("beta_seasonal", dist.Normal(loc=0, scale=1))
    seasonal = (fourier @ beta_seasonal)[..., None]

    def mean(carry, x_t):
        seasonal_t, _, _ = x_t
        lag_1, lag_2 = carry
        return mu + phi_1 * lag_1 + phi_2 * lag_2 + seasonal_t

    def update(carry, y_t, eps_t, x_t):
        _, available_t, censored_t = x_t
        lag_1, _ = carry
        # The filtered lag: pass clean observations through, floor capped days at the
        # prediction, and substitute the prediction on stockout days. Calling mean again
        # reproduces the filter's prediction exactly (same expression, same inputs).
        pred = mean(carry, x_t)
        on_shelf = jnp.where(censored_t == 1, jnp.maximum(y_t, pred), y_t)
        y_filtered = jnp.where(available_t == 1, on_shelf, pred)
        return jnp.clip(y_filtered, min=0.0), lag_1

    # Over the horizon the product is available and uncensored: no gate, no cap.
    xs = (seasonal, pad_future(available, h.future, value=1.0), pad_future(censored, h.future))
    init_carry = (y[0], y[0])  # placeholder lags; the first two steps are masked below
    r = ssoe(h, "eps", y, init_carry, mean, update, dist.Normal(loc=0, scale=sigma), xs=xs)
    pred_mean = numpyro.deterministic("pred_mean", r.mu)

    valid = (jnp.arange(h.t_obs)[:, None] >= 2) & (available == 1)
    numpyro.sample(
        "obs",
        dist.RightCensoredDistribution(
            dist.Normal(loc=pred_mean, scale=sigma), censored=censored
        ).mask(valid),
        obs=h.data,
    )
    if h.future > 0:
        numpyro.deterministic("forecast", jnp.clip(r.y_future, min=0.0))
```


# Inference with NUTS

We fit the model on the training window with plain NumPyro: the No-U-Turn Sampler through `MCMC`, running 4 chains of 1{,}000 warmup and 1{,}000 sampling steps each with `target_accept_prob=0.9` (the survival term gives the likelihood a slightly harder geometry near the cap). A modest budget is plenty because the posterior is tiny: the in-sample filter is deterministic, so the only latents are the eight parameters (\mu, \phi_1, \phi_2, \sigma, and four Fourier coefficients). The small `fit_nuts` helper wraps the call so the naive baseline below fits with exactly the same settings.

We then export the posterior draws into an ArviZ-schema `xarray.DataTree` with [`to_datatree`](https://juanitorduz.github.io/numpyro_forecast/reference/convert.to_datatree.html), which restores the `(chain, draw)` structure (we pass `num_chains=4`) and samples the in-sample posterior predictive from the same draws. Because we pass the *extended* covariates, the tree automatically carries `predictions` groups with the out-of-sample draws of the `"forecast"` site, and the trailing scenario rows of the covariates land verbatim in `predictions_constant_data`, so the tree documents that this forecast describes a full-availability, uncensored scenario.


``` python
def fit_nuts(
    rng_key: Array, model: ForecastModel, data: Array, covariates: Array
) -> dict[str, Array]:
    """Fit ``model`` with NUTS at ``target_accept_prob=0.9`` (4 chains, 1,000 warmup and 1,000 draws each)."""
    mcmc = MCMC(
        NUTS(model, target_accept_prob=0.9),
        num_warmup=1_000,
        num_samples=1_000,
        num_chains=4,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(rng_key, covariates, data)
    return mcmc.get_samples()


rng_key, rng_subkey = random.split(rng_key)
posterior = fit_nuts(rng_subkey, ar2_seasonal, train_data, covariates_train)

rng_key, rng_subkey = random.split(rng_key)
tree = to_datatree(
    rng_subkey,
    ar2_seasonal,
    posterior,
    train_data,
    covariates,
    num_chains=4,
    posterior_dims={"pred_mean": ["time", "obs_dim"]},
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
│       Dimensions:              (chain: 4, draw: 1000, beta_seasonal_dim_0: 4,
│                                 time: 150, obs_dim: 1)
│       Coordinates:
│         * chain                (chain) int64 32B 0 1 2 3
│         * draw                 (draw) int64 8kB 0 1 2 3 4 5 ... 995 996 997 998 999
│         * beta_seasonal_dim_0  (beta_seasonal_dim_0) int64 32B 0 1 2 3
│         * time                 (time) int64 1kB 0 1 2 3 4 5 ... 145 146 147 148 149
│         * obs_dim              (obs_dim) int64 8B 0
│       Data variables:
│           beta_seasonal        (chain, draw, beta_seasonal_dim_0) float32 64kB 0.55...
│           mu                   (chain, draw) float32 16kB 0.3071 0.1814 ... 0.47
│           phi_1                (chain, draw) float32 16kB 0.5045 0.551 ... 0.4947
│           phi_2                (chain, draw) float32 16kB 0.3653 0.3969 ... 0.304
│           pred_mean            (chain, draw, time, obs_dim) float32 2MB 0.2106 ... ...
│           sigma                (chain, draw) float32 16kB 0.4931 0.5094 ... 0.5975
│       Attributes:
│           created_at:                 2026-09-29T18:38:24.653214+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
├── Group: /posterior_predictive
│       Dimensions:  (chain: 4, draw: 1000, time: 150, obs_dim: 1)
│       Coordinates:
│         * chain    (chain) int64 32B 0 1 2 3
│         * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│         * time     (time) int64 1kB 0 1 2 3 4 5 6 7 ... 143 144 145 146 147 148 149
│         * obs_dim  (obs_dim) int64 8B 0
│       Data variables:
│           obs      (chain, draw, time, obs_dim) float32 2MB -0.3986 1.011 ... 2.352
│       Attributes:
│           created_at:                 2026-09-29T18:38:24.841588+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
├── Group: /observed_data
│       Dimensions:  (time: 150, obs_dim: 1)
│       Coordinates:
│         * time     (time) int64 1kB 0 1 2 3 4 5 6 7 ... 143 144 145 146 147 148 149
│         * obs_dim  (obs_dim) int64 8B 0
│       Data variables:
│           obs      (time, obs_dim) float32 600B 0.0 2.2 0.0 2.2 ... 0.0 0.0 0.0 2.2
│       Attributes:
│           created_at:                 2026-09-29T18:38:24.841858+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /constant_data
│       Dimensions:        (time: 150, covariate_dim: 7)
│       Coordinates:
│         * time           (time) int64 1kB 0 1 2 3 4 5 6 ... 144 145 146 147 148 149
│         * covariate_dim  (covariate_dim) int64 56B 0 1 2 3 4 5 6
│       Data variables:
│           covariates     (time, covariate_dim) float32 4kB 0.0 0.0 ... -0.2225 -0.901
│       Attributes:
│           created_at:                 2026-09-29T18:38:24.842051+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /predictions
│       Dimensions:  (chain: 4, draw: 1000, time: 30, obs_dim: 1)
│       Coordinates:
│         * chain    (chain) int64 32B 0 1 2 3
│         * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│         * time     (time) int64 240B 150 151 152 153 154 155 ... 175 176 177 178 179
│         * obs_dim  (obs_dim) int64 8B 0
│       Data variables:
│           obs      (chain, draw, time, obs_dim) float32 480kB 2.935 1.728 ... 2.956
│       Attributes:
│           created_at:                 2026-09-29T18:38:25.072903+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
└── Group: /predictions_constant_data
        Dimensions:        (time: 30, covariate_dim: 7)
        Coordinates:
          * time           (time) int64 240B 150 151 152 153 154 ... 175 176 177 178 179
          * covariate_dim  (covariate_dim) int64 56B 0 1 2 3 4 5 6
        Data variables:
            covariates     (time, covariate_dim) float32 840B 0.0 1.0 ... -0.901 0.6235
        Attributes:
            created_at:                 2026-09-29T18:38:25.073158+00:00
            creation_library:           ArviZ
            creation_library_version:   1.2.0
            creation_library_language:  Python
            sample_dims:                []
```


xarray.DataTree


/posterior(16)

Dimensions:


- chain: 4
- draw: 1000
- beta_seasonal_dim_0: 4
- time: 150
- obs_dim: 1


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


0 1 2 3 4 5 ... 995 996 997 998 999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))


beta_seasonal_dim_0


(beta_seasonal_dim_0)


int64


0 1 2 3


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0, 1, 2, 3])


time


(time)


int64


0 1 2 3 4 5 ... 145 146 147 148 149


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,140, 141, 142, 143, 144, 145, 146, 147, 148, 149])


obs_dim


(obs_dim)


int64


0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0])


Data variables: (6)


beta_seasonal


(chain, draw, beta_seasonal_dim_0)


float32


0.5591 -0.04451 ... -0.104 0.08106


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 0.5591447 , -0.04451339, -0.0435505 , -0.05296631],[ 0.50053763,  0.02363414, -0.11516415, -0.12392239],[ 0.52372295, -0.02785258, -0.03731854, -0.03956132],...,[ 0.5538658 ,  0.16943866, -0.147026  , -0.11494731],[ 0.44866744, -0.13500886, -0.17105387, -0.04619906],[ 0.60577625,  0.01823926, -0.10054317, -0.10063499]],[[ 0.6205467 ,  0.01782155, -0.18363437, -0.14535704],[ 0.57380253,  0.06064026, -0.1871667 , -0.13565852],[ 0.42232248, -0.06123746, -0.21855116, -0.06765711],...,[ 0.4812488 , -0.02617146, -0.19471568,  0.0187237 ],[ 0.5069468 , -0.0204498 , -0.18748449, -0.13136683],[ 0.43057382,  0.00127634, -0.13999394, -0.04048444]],[[ 0.4361102 ,  0.03546958, -0.2600437 , -0.19619665],[ 0.5608652 , -0.15059811, -0.17698061,  0.0708751 ],[ 0.56642526, -0.15753235, -0.17939986,  0.07767019],...,[ 0.46088463,  0.07546823, -0.18841942, -0.08627583],[ 0.5765393 , -0.0761444 , -0.18463863, -0.10120564],[ 0.4873596 , -0.07105602, -0.13299415,  0.07869115]],[[ 0.5853917 ,  0.01217251, -0.23374048, -0.17007391],[ 0.43343633, -0.0648467 , -0.13560477, -0.07640368],[ 0.49266383, -0.09396556, -0.26622376,  0.02199454],...,[ 0.5637018 , -0.01471817, -0.3504508 , -0.14122047],[ 0.45757222, -0.09071962, -0.38600525, -0.19014944],[ 0.39729753,  0.06511508, -0.10400075,  0.08106254]]],shape=(4, 1000, 4), dtype=float32)


mu


(chain, draw)


float32


0.3071 0.1814 ... 0.4997 0.47


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.30709398, 0.1813979 , 0.0294814 , ..., 0.2054264 , 0.36257753,0.24456458],[0.29377237, 0.28215963, 0.27413657, ..., 0.36178887, 0.25161844,0.33355546],[0.30191213, 0.18754052, 0.12328263, ..., 0.25756016, 0.29162434,0.4906252 ],[0.20203269, 0.44792038, 0.454789  , ..., 0.43671784, 0.49966422,0.46999785]], shape=(4, 1000), dtype=float32)


phi_1


(chain, draw)


float32


0.5045 0.551 ... 0.3272 0.4947


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.50453156, 0.5509679 , 0.5215353 , ..., 0.5630168 , 0.35437065,0.6180924 ],[0.5481041 , 0.58149683, 0.3149006 , ..., 0.24707673, 0.55200005,0.49419814],[0.47505814, 0.45960435, 0.42674872, ..., 0.42347756, 0.52024734,0.40309146],[0.4526161 , 0.35811916, 0.3625146 , ..., 0.3941012 , 0.32724807,0.49469966]], shape=(4, 1000), dtype=float32)


phi_2


(chain, draw)


float32


0.3653 0.3969 ... 0.435 0.304


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.36528105, 0.3969317 , 0.4856398 , ..., 0.35383826, 0.48157853,0.30446592],[0.32242957, 0.30507368, 0.54730034, ..., 0.5704784 , 0.3435947 ,0.33521488],[0.36143845, 0.4606942 , 0.5201615 , ..., 0.43138513, 0.3654835 ,0.369668  ],[0.46915537, 0.39259803, 0.4560692 , ..., 0.41066253, 0.43504438,0.30397215]], shape=(4, 1000), dtype=float32)


pred_mean


(chain, draw, time, obs_dim)


float32


0.2106 0.7917 2.116 ... 1.753 2.054


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 0.21057716],[ 0.7917292 ],[ 2.1158345 ],...,[ 1.1862314 ],[ 1.6740947 ],[ 2.2068863 ]],[[-0.05768864],[ 0.5515472 ],[ 2.0085375 ],...,[ 0.9665399 ],[ 1.4799981 ],[ 1.9954886 ]],[[-0.04739846],[ 0.39732575],[ 1.7434835 ],...,......,[ 0.8032944 ],[ 1.3751795 ],[ 2.069734  ]],[[-0.07649049],[ 0.57060546],[ 1.9622847 ],...,[ 0.80315465],[ 1.3112259 ],[ 2.0208454 ]],[[ 0.44705963],[ 0.98237884],[ 2.0034225 ],...,[ 1.3429445 ],[ 1.7526968 ],[ 2.0544653 ]]]], shape=(4, 1000, 150, 1), dtype=float32)


sigma


(chain, draw)


float32


0.4931 0.5094 ... 0.4832 0.5975


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.49308538, 0.5094279 , 0.46797103, ..., 0.5567375 , 0.4820939 ,0.61383414],[0.56280285, 0.5348111 , 0.5499795 , ..., 0.58857095, 0.48579007,0.50594354],[0.4769226 , 0.56439877, 0.57691705, ..., 0.5203696 , 0.5623889 ,0.59324354],[0.5233938 , 0.55117965, 0.57999796, ..., 0.55965006, 0.48323324,0.59746796]], shape=(4, 1000), dtype=float32)


Attributes: (5)


created_at :  
2026-09-29T18:38:24.653214+00:00

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
- time: 150
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


0 1 2 3 4 5 ... 995 996 997 998 999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))


time


(time)


int64


0 1 2 3 4 5 ... 145 146 147 148 149


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,140, 141, 142, 143, 144, 145, 146, 147, 148, 149])


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


-0.3986 1.011 ... 0.9982 2.352


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[-0.39856717],[ 1.0105472 ],[ 1.9984392 ],...,[ 1.2253861 ],[ 1.2730609 ],[ 1.5872067 ]],[[-0.58156794],[ 0.91258043],[ 2.3190818 ],...,[ 1.6704911 ],[ 1.481953  ],[ 1.450954  ]],[[-0.08254524],[-0.0405979 ],[ 2.074419  ],...,......,[ 0.06873332],[ 1.4605557 ],[ 2.3085876 ]],[[-0.82510144],[ 0.60835123],[ 2.8057435 ],...,[ 1.2824519 ],[ 1.4238085 ],[ 2.0017612 ]],[[ 0.5885088 ],[ 1.4391588 ],[ 1.8851713 ],...,[ 0.34820682],[ 0.99819535],[ 2.35159   ]]]], shape=(4, 1000, 150, 1), dtype=float32)


Attributes: (5)


created_at :  
2026-09-29T18:38:24.841588+00:00

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


- time: 150
- obs_dim: 1


Coordinates: (2)


time


(time)


int64


0 1 2 3 4 5 ... 145 146 147 148 149


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,140, 141, 142, 143, 144, 145, 146, 147, 148, 149])


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


0.0 2.2 0.0 2.2 ... 0.0 0.0 0.0 2.2


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.        ],[2.2       ],[0.        ],[2.2       ],[1.5981454 ],[1.330448  ],[1.4032364 ],[0.        ],[0.        ],[2.2       ],[0.        ],[2.2       ],[2.0323782 ],[2.2       ],[2.2       ],[1.4770805 ],[2.2       ],[2.2       ],[0.        ],[1.6346474 ],...[1.0327187 ],[1.2768031 ],[0.38543403],[0.        ],[1.3336072 ],[1.1269362 ],[0.89154816],[1.3728547 ],[1.0591812 ],[0.7849984 ],[0.        ],[1.0683882 ],[1.6382626 ],[0.        ],[1.5513194 ],[1.1958487 ],[0.        ],[0.        ],[0.        ],[2.2       ]], dtype=float32)


Attributes: (5)


created_at :  
2026-09-29T18:38:24.841858+00:00

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


- time: 150
- covariate_dim: 7


Coordinates: (2)


time


(time)


int64


0 1 2 3 4 5 ... 145 146 147 148 149


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,140, 141, 142, 143, 144, 145, 146, 147, 148, 149])


covariate_dim


(covariate_dim)


int64


0 1 2 3 4 5 6


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0, 1, 2, 3, 4, 5, 6])


Data variables: (1)


covariates


(time, covariate_dim)


float32


0.0 0.0 0.0 ... -0.2225 -0.901


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00, ...,0.0000000e+00,  1.0000000e+00,  1.0000000e+00],[ 2.2000000e+00,  1.0000000e+00,  1.0000000e+00, ...,9.7492790e-01,  6.2348974e-01, -2.2252107e-01],[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00, ...,-4.3388399e-01, -2.2252107e-01, -9.0096873e-01],...,[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00, ...,1.5926582e-05,  1.0000000e+00,  1.0000000e+00],[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00, ...,9.7492421e-01,  6.2348330e-01, -2.2253717e-01],[ 2.2000000e+00,  1.0000000e+00,  1.0000000e+00, ...,-4.3389940e-01, -2.2252941e-01, -9.0096134e-01]],shape=(150, 7), dtype=float32)


Attributes: (5)


created_at :  
2026-09-29T18:38:24.842051+00:00

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
- time: 30
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


0 1 2 3 4 5 ... 995 996 997 998 999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))


time


(time)


int64


150 151 152 153 ... 176 177 178 179


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177,178, 179])


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


2.935 1.728 0.7197 ... 2.969 2.956


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[2.934748  ],[1.7277291 ],[0.71968544],...,[2.580685  ],[2.6530724 ],[2.6935244 ]],[[1.2972183 ],[1.3374043 ],[1.5682148 ],...,[0.61445415],[0.19393057],[0.01684931]],[[1.6828265 ],[1.6043894 ],[0.7684723 ],...,......,[2.9057727 ],[2.3267024 ],[1.9578154 ]],[[2.5491533 ],[2.8074162 ],[1.932171  ],...,[1.6900384 ],[2.6519265 ],[3.1344223 ]],[[2.4273353 ],[2.7563295 ],[2.1162052 ],...,[1.8705219 ],[2.969018  ],[2.9555528 ]]]], shape=(4, 1000, 30, 1), dtype=float32)


Attributes: (5)


created_at :  
2026-09-29T18:38:25.072903+00:00

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


- time: 30
- covariate_dim: 7


Coordinates: (2)


time


(time)


int64


150 151 152 153 ... 176 177 178 179


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177,178, 179])


covariate_dim


(covariate_dim)


int64


0 1 2 3 4 5 6


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0, 1, 2, 3, 4, 5, 6])


Data variables: (1)


covariates


(time, covariate_dim)


float32


0.0 1.0 0.0 ... -0.901 0.6235


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00,  4.3387559e-01,-7.8182018e-01, -9.0097278e-01,  6.2350398e-01],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00, -4.3389225e-01,7.8184325e-01, -9.0096480e-01,  6.2347507e-01],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00, -9.7493011e-01,4.3386611e-01, -2.2251140e-01, -9.0097737e-01],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00, -7.8182513e-01,-9.7493243e-01,  6.2349772e-01, -2.2250113e-01],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00,  1.0522323e-05,2.1044645e-05,  1.0000000e+00,  1.0000000e+00],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00,  7.8183824e-01,9.7492307e-01,  6.2348127e-01, -2.2254217e-01],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00,  9.7492540e-01,-4.3390402e-01, -2.2253190e-01, -9.0095907e-01],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00,  4.3387327e-01,-7.8181696e-01, -9.0097392e-01,  6.2350798e-01],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00, -4.3389454e-01,7.8184640e-01, -9.0096366e-01,  6.2347108e-01],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00, -9.7493064e-01,4.3386149e-01, -2.2250889e-01, -9.0097958e-01],...[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00,  9.7492766e-01,-4.3388575e-01, -2.2252202e-01, -9.0096790e-01],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00,  4.3388242e-01,-7.8182966e-01, -9.0096951e-01,  6.2349212e-01],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00, -4.3388540e-01,7.8183377e-01, -9.0096807e-01,  6.2348694e-01],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00, -9.7493184e-01,4.3385229e-01, -2.2250390e-01, -9.0098405e-01],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00, -7.8182036e-01,-9.7493589e-01,  6.2350374e-01, -2.2248617e-01],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00,  1.8199416e-05,3.6398833e-05,  1.0000000e+00,  1.0000000e+00],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00,  7.8184307e-01,9.7491968e-01,  6.2347531e-01, -2.2255713e-01],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00,  9.7492713e-01,-4.3389037e-01, -2.2252451e-01, -9.0096569e-01],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00,  4.3388012e-01,-7.8182644e-01, -9.0097064e-01,  6.2349612e-01],[ 0.0000000e+00,  1.0000000e+00,  0.0000000e+00, -4.3388772e-01,7.8183699e-01, -9.0096694e-01,  6.2348294e-01]], dtype=float32)


Attributes: (5)


created_at :  
2026-09-29T18:38:25.073158+00:00

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

`az.summary` on the parameters gives the convergence picture in one call: posterior means and standard deviations, the 94\\ HDIs, effective sample sizes, and \hat{R}.

A word on reading the parameter table: the data generating process used \phi_1 = 0.6, \phi_2 = 0.3, and intercept 0.2. Exact recovery is not on the table, because the model's recursion runs on (filtered) sales rather than demand and the process clips at zero, but the estimates land in the right neighborhood, and they do so *because of* the lag filter. Feed the recorded stockout zeros in as lagged values instead and the AR coefficients collapse toward zero while \sigma inflates: every post-stockout day then looks like a violent rebound from a demand crash that never happened, and shrinking \phi is the likelihood's only way to explain it. The forecast quality check that ultimately matters, recovering latent demand out of sample, comes further below.


``` python
scalar_vars = ["mu", "phi_1", "phi_2", "sigma", "beta_seasonal"]
az.summary(tree, var_names=scalar_vars, ci_kind="hdi", ci_prob=0.94)
```


|  | mean | sd | hdi94_lb | hdi94_ub | ess_bulk | ess_tail | r_hat | mcse_mean | mcse_sd |
|----|----|----|----|----|----|----|----|----|----|
| mu | 0.294 | 0.133 | 0.059 | 0.56 | 2762 | 2353 | 1.00 | 0.0026 | 0.0019 |
| phi_1 | 0.447 | 0.112 | 0.24 | 0.66 | 1909 | 1815 | 1.00 | 0.0025 | 0.0017 |
| phi_2 | 0.419 | 0.112 | 0.21 | 0.62 | 2129 | 2416 | 1.00 | 0.0024 | 0.0017 |
| sigma | 0.543 | 0.046 | 0.47 | 0.64 | 3217 | 2743 | 1.00 | 0.00081 | 0.00059 |
| beta_seasonal\[0\] | 0.506 | 0.081 | 0.36 | 0.66 | 3447 | 2594 | 1.00 | 0.0014 | 0.001 |
| beta_seasonal\[1\] | -0.021 | 0.086 | -0.18 | 0.14 | 3794 | 2479 | 1.00 | 0.0014 | 0.00098 |
| beta_seasonal\[2\] | -0.184 | 0.084 | -0.35 | -0.028 | 2432 | 2694 | 1.00 | 0.0017 | 0.0012 |
| beta_seasonal\[3\] | -0.074 | 0.085 | -0.23 | 0.087 | 3420 | 2752 | 1.00 | 0.0015 | 0.001 |


``` python
pc_trace = az.plot_trace_dist(
    tree,
    var_names=scalar_vars,
    compact=True,
    figure_kwargs={"figsize": (12, 9)},
)
pc_trace.viz["figure"].item().suptitle(
    "Trace plots",
    fontsize=18,
    fontweight="bold",
    y=1.03,
);
```


<figure class="figure">
<p><img src="censored_demand_files/figure-html/_src-censored_demand-cell-9-output-1.png" class="figure-img" width="1211" height="942" /></p>
</figure>


# In-sample fit

The tree's `posterior_predictive` group holds the one-step-ahead predictive of the `"obs"` site over the training window (the first two steps run on placeholder lags and are dropped from the plot). Because `RightCensoredDistribution` samples from its *base* distribution, these draws describe latent demand-scale sales, so the two places where they deliberately disagree with the observed series are the whole point: on capacity days the bands ride **above** the dashed cap that pins the orange line down, and through the gray stockout gaps the predictive mean carries the filter's demand estimate instead of chasing the recorded zeros. The latent demand curve (black), which the model never saw, runs along the upper half of the bands (the predictive describes *sales*, which sit below demand by the friction term) and escapes them only at the strongest peaks. The lower band also dips below zero in the troughs: a \text{Normal} likelihood pays for its simplicity with predictive mass on negative sales, a compromise the next steps revisit (the forecast recursion, by contrast, clips its sampled trajectories at zero). This cell also defines the small plotting helpers (`stacked_draws` and `plot_band_forecast`) shared with the remaining band plots.


``` python
def hdi_label(prob: float, prefix: str = "") -> str:
    r"""Legend label for an HDI band, e.g. ``$94\%$ HDI``."""
    percent = f"{prob:.0%}".replace("%", r"\%")
    return f"{prefix}${percent}$ HDI"


hdi_probs = (0.5, 0.94)
hdi_alphas = [0.4, 0.2]  # 50% band darker, 94% band lighter; both light enough to read overlays


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


def plot_band_forecast(
    draws: np.ndarray,
    x: np.ndarray,
    color: str,
    label_prefix: str = "",
    observed: Array | np.ndarray | None = None,
    figsize: tuple[float, float] = (12.0, 6.0),
) -> tuple[Axes, list[Artist]]:
    r"""Plot the posterior mean line and the $50\%$/$94\%$ HDI bands of ``draws``.

    Wraps ``predictions_to_datatree`` and ``az.plot_lm`` with the notebook-wide
    band styling (inner band darker via ``hdi_alphas``) and labels the artists.
    Overlays (observed series, split lines, extra reference lines) and the
    legend are the caller's responsibility.

    Parameters
    ----------
    draws
        Predictive draws with shape ``(sample, time, 1)``.
    x
        Numeric x values of length ``time``.
    color
        Matplotlib color for the bands and the mean line.
    label_prefix
        Prefix for the legend labels, e.g. ``"forecast "``.
    observed
        Optional observed data stored alongside the draws.
    figsize
        Figure size passed to ``plot_lm``.

    Returns
    -------
    tuple[Axes, list[Artist]]
        The axes and the labeled band and mean-line handles for the legend.
    """
    idata = predictions_to_datatree(draws, x, ["y"], observed=observed)
    pc = az.plot_lm(
        idata,
        y="obs",
        x="t",
        plot_dim="time",
        ci_kind="hdi",
        ci_prob=hdi_probs,
        smooth=False,
        point_estimate="mean",
        visuals={
            "ci_band": {"color": color},
            "observed_scatter": False,
            "pe_line": {"color": color, "alpha": 1.0, "width": 1.5},
        },
        aes={"alpha": ["prob"]},
        alpha=hdi_alphas,
        figure_kwargs={"figsize": figsize},
    )
    bands = pc.viz["ci_band"]["t"]
    band_94, band_50 = bands.sel(prob=0.94).item(), bands.sel(prob=0.5).item()
    band_94.set_label(hdi_label(0.94, prefix=label_prefix))
    band_50.set_label(hdi_label(0.5, prefix=label_prefix))
    pe_line = pc.viz["pe_line"]["t"].item()
    pe_line.set_label(f"{label_prefix}posterior mean")
    ax = pc.viz["figure"].item().axes[0]
    return ax, [band_94, band_50, pe_line]


train_pp = stacked_draws(tree["posterior_predictive"], "obs")[:, 2:, :]

ax, handles = plot_band_forecast(
    train_pp,
    time[2:n_train].astype(float),
    "C2",
    observed=train_data[2:],
)
ax.fill_between(
    time[:n_train],
    0,
    1,
    where=(np.asarray(is_available[:n_train]) == 0).tolist(),
    transform=ax.get_xaxis_transform(),
    color="gray",
    alpha=0.2,
    step="mid",
)
(demand_line,) = ax.plot(
    time[:n_train], np.asarray(demand[:n_train]), color="black", lw=1.2, label="latent demand"
)
(obs_line,) = ax.plot(
    time[:n_train], np.asarray(sales_obs[:n_train]), color="C1", lw=1, label="observed sales"
)
cap_line = ax.axhline(params.max_capacity, color="C3", ls="--", lw=1, label="capacity cap")
ax.legend(
    handles=[*handles, demand_line, obs_line, cap_line],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=3,
)
ax.set(title="One-step-ahead in-sample fit", xlabel="time", ylabel="units");
```


<figure class="figure">
<p><img src="censored_demand_files/figure-html/_src-censored_demand-cell-10-output-1.png" class="figure-img" width="1211" height="611" /></p>
</figure>


# Forecasting demand

The `predictions` group already holds the out-of-sample draws of the `"forecast"` site, structurally uncensored as discussed in the covariates section. We plot them against the **latent demand**, the series the model never saw, and score the forecast with the CRPS against that ground truth.

This is the money plot of the example: on its seasonal peaks the demand runs above the capacity cap, into territory where not a single observation exists, and the forecast mean follows it across the line. A model of *sales* cannot do this, because sales above the cap were never once recorded; the survival terms in the likelihood are what taught the model that capped days were floors, not values.


``` python
forecast_pp = stacked_draws(tree["predictions"], "obs")
crps_demand = eval_crps(forecast_pp, demand_test)

t_zoom = time[90:]
ax, handles = plot_band_forecast(forecast_pp, t_test.astype(float), "C2", label_prefix="forecast ")
(demand_line,) = ax.plot(
    t_zoom, np.asarray(demand[90:]), color="black", lw=1.5, label="latent demand"
)
(sales_line,) = ax.plot(
    t_zoom, np.asarray(sales_obs[90:]), ":", color="C1", lw=1.5, label="observed sales"
)
cap_line = ax.axhline(params.max_capacity, color="C3", ls="--", lw=1, label="capacity cap")
split_line = ax.axvline(n_train, color="gray", ls="--", label="train/test split")
ax.legend(
    handles=[*handles, demand_line, sales_line, cap_line, split_line],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=4,
)
ax.set(
    title=f"Censored-likelihood demand forecast (test CRPS vs latent demand: {crps_demand:.3f})",
    xlabel="time",
    ylabel="units",
);
```


<figure class="figure">
<p><img src="censored_demand_files/figure-html/_src-censored_demand-cell-11-output-1.png" class="figure-img" width="1211" height="611" /></p>
</figure>


# The naive comparison: ignoring the cap

How much of that is the censored likelihood, and how much just the AR(2) structure? With the censoring indicator at zero everywhere, `RightCensoredDistribution` *is* the plain \text{Normal}: every capped day is treated as an exact observation of 2.2. So the naive baseline is the same model fit on a covariates tensor whose censoring column is zeroed, one line of code. Stockout days remain masked and lag-filtered in both models; what changes is only how the capacity days enter, in the likelihood (exact value versus survival mass) and in the lag filter (trusted versus floored at the prediction).


``` python
covariates_naive = covariates.at[:, 2].set(0.0)

rng_key, rng_subkey = random.split(rng_key)
posterior_naive = fit_nuts(rng_subkey, ar2_seasonal, train_data, covariates_naive[:n_train])

rng_key, rng_subkey = random.split(rng_key)
tree_naive = to_datatree(
    rng_subkey,
    ar2_seasonal,
    posterior_naive,
    train_data,
    covariates_naive,
    num_chains=4,
    posterior_dims={"pred_mean": ["time", "obs_dim"]},
)
az.summary(tree_naive, var_names=scalar_vars, ci_kind="hdi", ci_prob=0.94)
```


|  | mean | sd | hdi94_lb | hdi94_ub | ess_bulk | ess_tail | r_hat | mcse_mean | mcse_sd |
|----|----|----|----|----|----|----|----|----|----|
| mu | 0.452 | 0.13 | 0.22 | 0.7 | 2957 | 2801 | 1.00 | 0.0024 | 0.0016 |
| phi_1 | 0.405 | 0.097 | 0.23 | 0.59 | 2226 | 2419 | 1.00 | 0.0021 | 0.0015 |
| phi_2 | 0.316 | 0.098 | 0.13 | 0.49 | 2765 | 2849 | 1.00 | 0.0019 | 0.0013 |
| sigma | 0.434 | 0.0294 | 0.38 | 0.49 | 3495 | 2923 | 1.00 | 0.0005 | 0.00038 |
| beta_seasonal\[0\] | 0.344 | 0.061 | 0.23 | 0.46 | 3397 | 2782 | 1.00 | 0.001 | 0.00076 |
| beta_seasonal\[1\] | 0.013 | 0.062 | -0.11 | 0.13 | 4445 | 2790 | 1.00 | 0.00093 | 0.00066 |
| beta_seasonal\[2\] | -0.157 | 0.063 | -0.28 | -0.036 | 2895 | 2563 | 1.00 | 0.0012 | 0.00083 |
| beta_seasonal\[3\] | -0.086 | 0.063 | -0.2 | 0.033 | 4459 | 2820 | 1.00 | 0.00094 | 0.00068 |


Side by side, the two forecasts tell the whole story. The censored model tracks the latent demand above the cap; the naive model, trained to believe demand *was* 2.2 on every capped day, pulls its level down toward the cap and undershoots the demand it is supposed to inform.


``` python
forecast_naive_pp = stacked_draws(tree_naive["predictions"], "obs")

series_names = ["censored likelihood", "plain Normal likelihood"]
both_draws = np.concatenate([forecast_pp, forecast_naive_pp], axis=-1)
idata_both = predictions_to_datatree(both_draws, t_test.astype(float), series_names)
pc = az.plot_lm(
    idata_both,
    y="obs",
    x="t",
    plot_dim="time",
    ci_kind="hdi",
    ci_prob=hdi_probs,
    smooth=False,
    point_estimate="mean",
    visuals={
        "ci_band": {"color": "C2"},
        "observed_scatter": False,
        "pe_line": {"color": "C2", "alpha": 1.0, "width": 1.5},
    },
    aes={"alpha": ["prob"]},
    alpha=hdi_alphas,
    figure_kwargs={"figsize": (12, 5), "sharex": True, "sharey": True},
)
axes = pc.viz["plot"]["t"]
for label in series_names:
    ax = axes.sel(series=label).item()
    ax.plot(t_test, np.asarray(demand[n_train:]), color="black", lw=1.5, label="latent demand")
    ax.axhline(params.max_capacity, color="C3", ls="--", lw=1, label="capacity cap")
    ax.set(title=label, xlabel="time", ylabel="")
axes.sel(series=series_names[0]).item().set(ylabel="units")
axes.sel(series=series_names[0]).item().legend(loc="upper left")
fig = pc.viz["figure"].item()
fig.suptitle(
    "Demand forecast: censored vs plain likelihood",
    fontsize=16,
    fontweight="bold",
    y=1.05,
);
```


<figure class="figure">
<p><img src="censored_demand_files/figure-html/_src-censored_demand-cell-13-output-1.png" class="figure-img" width="1211" height="540" /></p>
</figure>


# Evaluation

We score both models on the held-out 30 days with [`evaluate_forecast`](https://juanitorduz.github.io/numpyro_forecast/reference/evaluate.evaluate_forecast.html), against two different notions of truth: the **latent demand** (the business quantity, known here because we simulated it) and the **observed sales** (the only truth available in production data, itself gated and capped over the test window). Alongside the point metrics and the CRPS we track the empirical coverage of the central 50\\ and 94\\ intervals.


``` python
metrics = {
    "mae": eval_mae,
    "rmse": eval_rmse,
    "crps": eval_crps,
    "coverage_50": partial(eval_coverage, alpha=0.5),
    "coverage_94": partial(eval_coverage, alpha=0.94),
}

rows = {
    ("censored likelihood", "latent demand"): evaluate_forecast(
        forecast_pp, demand_test, metrics=metrics
    ),
    ("plain Normal likelihood", "latent demand"): evaluate_forecast(
        forecast_naive_pp, demand_test, metrics=metrics
    ),
    ("censored likelihood", "observed sales"): evaluate_forecast(
        forecast_pp, sales_obs_test, metrics=metrics
    ),
    ("plain Normal likelihood", "observed sales"): evaluate_forecast(
        forecast_naive_pp, sales_obs_test, metrics=metrics
    ),
}
results_df = pd.DataFrame(rows).T.round(3)
results_df.index.names = ["model", "truth"]
results_df
```


|  |  | mae | rmse | crps | coverage_50 | coverage_94 |
|----|----|----|----|----|----|----|
| model | truth |  |  |  |  |  |
| censored likelihood | latent demand | 0.487 | 0.636 | 0.369 | 0.567 | 1.000 |
| plain Normal likelihood | latent demand | 0.488 | 0.566 | 0.329 | 0.467 | 0.967 |
| censored likelihood | observed sales | 1.334 | 1.620 | 0.996 | 0.267 | 0.600 |
| plain Normal likelihood | observed sales | 0.995 | 1.204 | 0.774 | 0.300 | 0.467 |


Two readings, one per truth:

- **Against latent demand**, the aggregate point metrics are nearly a wash: the two MAEs differ by three thousandths, and the naive model is ahead on RMSE and CRPS. This is expected. Both models target the *sales* scale, which sits below demand by the friction \delta = 0.25 in expectation, so a modest downward offset against this truth is structural and shared. On the two thirds of test days where demand sits below the cap the two likelihoods largely agree, and there the naive model's tighter, lower predictive scores well, offsetting its losses at the peaks. Calibration is where they separate: the censored model's central intervals cover at or above their nominal levels at both widths, while the naive model's 50\\ interval falls short, because treating capped days as exact observations drags its mean down *and* shrinks its fitted noise scale (compare the \sigma posteriors in the two summary tables).
- **Against observed sales**, the naive model wins the three point metrics (MAE, RMSE, CRPS), and that is not a defect but the fresh retail example's lesson restated: the test window's observed sales are themselves gated and capped, so a *correct* demand forecast is penalized for sitting above the caps and the stockout zeros. Scoring against recorded sales systematically favors models that repeat the corruption, though only on point accuracy: both models' intervals sit far below nominal against this truth (the censored model covers 0.267 at 50\\ and 0.600 at 94\\, the naive model 0.300 and 0.467), and neither is clearly the better calibrated one. In production, where latent demand is unavailable, this is an argument for evaluating on periods or stores with clean availability.

Neither aggregate row answers the operational question, "how much should we stock for the strong days?", so we re-score on exactly the days that drive that decision: the test days whose latent demand exceeds the cap, the days a planner would under-stock by trusting the naive model.


``` python
peak_days = np.asarray(demand_test[:, 0] > params.max_capacity)
print(f"test days with latent demand above the cap: {int(peak_days.sum())} of {forecast_horizon}")

peak_rows = {
    "censored likelihood": evaluate_forecast(
        forecast_pp[:, peak_days, :], demand_test[peak_days], metrics=metrics
    ),
    "plain Normal likelihood": evaluate_forecast(
        forecast_naive_pp[:, peak_days, :], demand_test[peak_days], metrics=metrics
    ),
}
peak_df = pd.DataFrame(peak_rows).T.round(3)
peak_df.index.name = "model"
peak_df
```


    test days with latent demand above the cap: 10 of 30


|                         | mae   | rmse  | crps  | coverage_50 | coverage_94 |
|-------------------------|-------|-------|-------|-------------|-------------|
| model                   |       |       |       |             |             |
| censored likelihood     | 0.317 | 0.343 | 0.273 | 0.8         | 1.0         |
| plain Normal likelihood | 0.673 | 0.734 | 0.450 | 0.3         | 0.9         |


On the peak days the ambiguity disappears: the censored model roughly halves the point errors and its central 50\\ interval covers most of the peak-day demand, while the naive model's forecast mean tops out below the cap it mistook for data (visible in the side-by-side plot above), so its errors there are structural, not noise. The censored model's advantage is exactly where the money is: the days when there was more demand than shelf.


# Three ways to model censoring

The fresh retail stockout example closes its next-steps list with:

> Replace the mean-level factor with a censored likelihood: treat sales as latent demand right-censored by the available stock, which uses the same availability feature but models the mechanism instead of its average effect.

This notebook is that item made concrete, on a synthetic process where the recovered demand can be checked against ground truth. The three examples form a progression of mechanisms for the same business problem:

| Example | Censoring pattern | Mechanism | Demand forecast |
|----|----|----|----|
| [Availability TSB](https://juanitorduz.github.io/numpyro_forecast/docs/examples/availability_tsb.html) | Binary: off-shelf days record zero | Freeze the probability-recursion updates where the product is off the shelf | Feed a full-availability scenario input |
| [Fresh retail stockouts](https://juanitorduz.github.io/numpyro_forecast/docs/examples/fresh_retail_stockout.html) | Fractional: noisy within-day availability | Multiplicative availability factor on the mean, with a learned floor | Pin availability to one over the horizon |
| This notebook | Hard capacity cap plus stockout mask | Censored likelihood: density below the cap, survival mass at it | The generative recursion, unconstrained by the cap |

The choice between them is driven by what the data records and how the censoring operates:

- **Gating the recursion** (TSB) is the right tool when availability is binary and the model is a recursive smoother: off-shelf periods simply carry no information, so the estimate should freeze rather than decay.
- **A multiplicative factor** shines when availability is *fractional* and *noisy* (a reconstructed share of the day on the shelf): it models the average effect of partial availability on the mean, and its learned floor absorbs label noise such as recorded sales on supposedly out-of-stock days.
- **A censored likelihood** is the sharpest instrument when the censoring point is *known* per observation (a shelf capacity, an inventory level, a purchase limit): it models the mechanism itself rather than its average effect, at the price of trusting the recorded censoring indicator. All three share the same practical payoff: the corruption mechanism enters the model explicitly rather than as noise, so forecasting demand amounts to running the model with the mechanism switched off, whether through scenario covariate rows (TSB, fresh retail) or through a forecast recursion that never applies the cap (this notebook).

In practice the mechanisms compose: a retailer with hourly stockout labels *and* known shelf capacities could use the fresh retail example's availability feature for partial days and this notebook's survival terms for capped ones.


# Next steps

- Replace the capped-day plug-in \max(y_t, \hat{y}\_t) in the lag filter with the censored conditional mean \hat{y}\_t + \sigma \\ \varphi(z_t) / \left(1 - \Phi(z_t)\right) with z_t = (y_t - \hat{y}\_t) / \sigma, the exact expectation of the latent value given that it exceeds the cap.
- Let the capacity cap vary by day, read from inventory snapshots, instead of a single constant. `RightCensoredDistribution` censors at each *recorded* value, so only the data preparation changes, not the model.
- Swap the \text{Normal} base for a strictly nonnegative observation model (for example a truncated \text{Normal}), removing the predictive mass on negative sales visible in the in-sample bands.
- Replace the fixed train-test split with rolling-origin evaluation via the package's [backtest](../../reference/evaluate.backtest.md#numpyro_forecast.evaluate.backtest) helper, as in the [Croston example](https://juanitorduz.github.io/numpyro_forecast/docs/examples/croston.html).


# References

- Orduz, J. [*Demand Forecasting with Censored Likelihood*](https://juanitorduz.github.io/demand/). The blog post this notebook ports.
- NumPyro documentation: [Censored distributions](https://num.pyro.ai/en/stable/distributions.html#censored-distributions) (`RightCensoredDistribution` and friends, available from NumPyro `0.20.0`).
- Tobin, J. (1958). [*Estimation of Relationships for Limited Dependent Variables*](https://doi.org/10.2307/1907382). Econometrica, 26(1), 24-36. The classic censored-regression (Tobit) model.
- The [ARMA example](https://juanitorduz.github.io/numpyro_forecast/docs/examples/arma.html) in this documentation, which introduces the [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) building block for AR-on-observations models.
- The [availability TSB example](https://juanitorduz.github.io/numpyro_forecast/docs/examples/availability_tsb.html) and the [fresh retail stockout example](https://juanitorduz.github.io/numpyro_forecast/docs/examples/fresh_retail_stockout.html) in this documentation: the sibling availability mechanisms compared above.

[Source: Demand Forecasting with Censored Likelihood with `numpyro_forecast`](_src/censored_demand-preview.html#679c0a19)
