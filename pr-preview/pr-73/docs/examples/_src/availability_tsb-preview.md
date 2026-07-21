# TSB with Availability Constraints for Intermittent Demand with `numpyro_forecast`


This notebook ports the blog post [**Hacking the TSB Model for Intermittent Time Series to Accommodate for Availability Constraints**](https://juanitorduz.github.io/availability_tsb/) to the [`numpyro_forecast`](https://github.com/juanitorduz/numpyro_forecast) package. It closes the intermittent-demand trilogy started by the [Croston example](https://juanitorduz.github.io/numpyro_forecast/examples/croston.html) and the [TSB example](https://juanitorduz.github.io/numpyro_forecast/examples/tsb.html), and like those notebooks it focuses on the *one* structural change the method makes and why that change matters.

The motivation is a fact of retail life that the classical intermittent-demand methods ignore: a sales series contains **two kinds of zeros**. Some periods are zero because nobody wanted the product (no demand), and some are zero because nobody *could* buy it (a stock-out, a delisting, a closed store). What we observe is censored demand, \\y_t = a_t \cdot d^{\ast}\_t\\, where \\d^{\ast}\_t\\ is the demand that would have materialized and \\a_t \in \\0, 1\\\\ says whether the product was on the shelf.

Plain TSB cannot tell these zeros apart. Its demand probability decays at *every* zero, so a stretch of stock-outs is read as demand fading away, and the estimate converges to \\P(\text{available}) \cdot P(\text{demand} \mid \text{available})\\: biased low by the availability rate, and biased differently for every series depending on its stock-out history. The fix from the blog post is a **one-line change**: gate the probability update with the availability mask, so that off-shelf periods, which carry no demand information whatsoever, leave the estimate frozen instead of decaying it. The estimate then targets the uncensored \\P(\text{demand} \mid \text{available})\\, and because availability becomes a model *input*, the forecast turns into a **scenario tool**: feed a full-availability future to forecast unconstrained demand (the number replenishment planning needs), or feed any planned availability path.

Two practical notes on the port:

- We reuse the sibling notebooks' reusable level model (one `jax.lax.scan` per exponential smoothing recursion, with a boolean gate deciding *when* the level updates), promoted from a single series to a `(time, series)` panel. Croston gates on demand events, TSB gates on every period, and the availability-aware variant gates on the availability mask. The entire method is that one argument.
- The covariates carry **two stacked inputs** in a `(covariate, time, series)` tensor: the **sales history** the recursions consume, and the **availability mask**. Because the forecast reads its future availability from the covariates, choosing a scenario is just choosing the trailing rows of the availability input. Everything plugs straight into [fit_svi](../../../reference/functional.svi.fit_svi.md#numpyro_forecast.functional.svi.fit_svi), [to_datatree](../../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree), [forecast](../../../reference/functional.prediction.forecast.md#numpyro_forecast.functional.prediction.forecast), and [add_forecast_groups](../../../reference/convert.add_forecast_groups.md#numpyro_forecast.convert.add_forecast_groups).


# Prepare notebook


    In [1]:


``` python
from typing import NamedTuple, cast

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import preliz as pz
import xarray as xr
from jax import random
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from numpyro.handlers import scope
from numpyro.infer import Predictive

from numpyro_forecast import (
    add_forecast_groups,
    eval_coverage,
    eval_crps,
    forecasting_model,
    predictions_to_datatree,
    to_datatree,
)
from numpyro_forecast.functional import Horizon, draw_posterior, fit_svi, forecast
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


    /Users/juanitorduz/Documents/numpyro_forecast/.venv/lib/python3.14/site-packages/preliz/ppls/pymc_io.py:16: UserWarning: PyMC not installed. PyMC related functions will not work.
      warnings.warn("PyMC not installed. PyMC related functions will not work.")
    /Users/juanitorduz/Documents/numpyro_forecast/.venv/lib/python3.14/site-packages/preliz/ppls/agnostic.py:34: UserWarning: PyMC not installed. PyMC related functions will not work.
      warnings.warn("PyMC not installed. PyMC related functions will not work.")


# Generate data

We use the blog post's synthetic panel: \\1{,}000\\ series over \\60\\ periods. Each series draws a rate \\\lambda_i \sim \text{Gamma}(2.5)\\, its latent demand is \\d^{\ast}\_{t, i} \sim \text{Poisson}(\lambda_i)\\, availability is an independent coin flip \\a\_{t, i} \sim \text{Bernoulli}(0.6)\\, and the observed sales are the censored product \\y\_{t, i} = a\_{t, i} \cdot d^{\ast}\_{t, i}\\. The last \\10\\ periods are held out as a test window.

The one deliberate extension over the blog post is that the generator also *returns* the uncensored demand and the true rates. The data-generating process knows the ground truth, so later sections can score the recovered demand probabilities against \\P(d^{\ast} \> 0) = 1 - e^{-\lambda}\\ instead of eyeballing them.


    In [2]:


``` python
class IntermittentPanel(NamedTuple):
    """Synthetic intermittent-demand panel with an availability mask.

    Attributes
    ----------
    demand : Array
        Uncensored latent demand counts, shape ``(t_max, n_series)``.
    available : Array
        Availability mask in ``{0.0, 1.0}``, shape ``(t_max, n_series)``.
    sales : Array
        Observed sales ``available * demand``, shape ``(t_max, n_series)``.
    lambdas : Array
        True per-series Poisson rates, shape ``(1, n_series)``.
    """

    demand: Array
    available: Array
    sales: Array
    lambdas: Array


def generate_intermittent_counts(
    rng_key: Array,
    n_series: int,
    t_max: int,
    a: float = 1.0,
    p: float = 0.5,
) -> IntermittentPanel:
    """Generate an intermittent-demand panel with availability censoring.

    Follows the blog post's generator: per-series Poisson rates from a global
    Gamma, Poisson demand counts, and an independent Bernoulli availability
    mask; observed sales are the censored product. The uncensored demand and
    the true rates are returned as well, for ground-truth comparisons.

    Parameters
    ----------
    rng_key
        PRNG key consumed by the three draws.
    n_series
        Number of series in the panel.
    t_max
        Number of time periods.
    a
        Shape parameter of the Gamma distribution the rates are drawn from.
    p
        Probability that a product is available in a given period.

    Returns
    -------
    IntermittentPanel
        The generated panel.
    """
    rng_key, rng_subkey = random.split(rng_key)
    lambdas = random.gamma(rng_subkey, a=a, shape=(1, n_series))

    rng_key, rng_subkey = random.split(rng_key)
    demand = random.poisson(rng_subkey, lam=lambdas, shape=(t_max, n_series))
    demand = demand.astype(jnp.float32)

    rng_key, rng_subkey = random.split(rng_key)
    available = random.bernoulli(rng_subkey, p=p, shape=demand.shape).astype(jnp.float32)

    return IntermittentPanel(
        demand=demand, available=available, sales=available * demand, lambdas=lambdas
    )


n_series = 1_000
t_max = 60
availability_rate = 0.6

rng_key, rng_subkey = random.split(rng_key)
panel = generate_intermittent_counts(
    rng_subkey, n_series=n_series, t_max=t_max, a=2.5, p=availability_rate
)
lam = np.asarray(panel.lambdas[0])
print(f"sales shape: {panel.sales.shape}, lambda range: [{lam.min():.2f}, {lam.max():.2f}]")
```


    sales shape: (60, 1000), lambda range: [0.17, 9.98]


Throughout the package, time lives at axis `-2` and the observation dimension at axis `-1`; for a panel the series axis *is* the observation axis, so the data are simply `(time, series)` arrays. The covariates stack the two inputs in front, giving the `(covariate, time, series)` tensor described above. For the fixed-origin forecast we extend the sales input over the horizon with zeros (leak-free, because the model never reads it past [t_obs](../../../reference/forecaster.ForecastingModel.md#numpyro_forecast.forecaster.ForecastingModel.t_obs)) and the availability input with the *realized* test availability: unlike future sales, future availability is a legitimate input, since in practice assortment and replenishment plans are known ahead of time.


    In [3]:


``` python
t_max_train = 50
train_data = panel.sales[:t_max_train, :]
test_data = panel.sales[t_max_train:, :]
available_train = panel.available[:t_max_train, :]
available_test = panel.available[t_max_train:, :]
t = np.arange(t_max)
t_train, t_test = t[:t_max_train], t[t_max_train:]

covariates_train = jnp.stack([train_data, available_train], axis=0)
sales_input_full = jnp.concatenate([train_data, jnp.zeros_like(test_data)], axis=0)
covariates_full = jnp.stack([sales_input_full, panel.available], axis=0)
print(f"train data shape: {train_data.shape}, full covariates shape: {covariates_full.shape}")
```


    train data shape: (50, 1000), full covariates shape: (2, 60, 1000)


## Two kinds of zeros

Before modeling anything, it is worth quantifying how badly the zeros conflate the two stories. In the training window, roughly \\40\\\\ of all periods are stock-outs by construction, and they turn a substantial share of periods with genuine demand into observed zeros (lost sales). A method that reads every zero as "no demand" is fitting to all of them.


    In [4]:


``` python
demand_train = panel.demand[:t_max_train, :]
share_zero_sales = float(jnp.mean(train_data == 0))
share_unavailable = float(jnp.mean(available_train == 0))
share_censored = float(jnp.mean((demand_train > 0) & (available_train == 0)))
share_true_zero = float(jnp.mean((demand_train == 0) & (available_train == 1)))

print(f"share of zero-sales periods:                {share_zero_sales:.2f}")
print(f"  of which stock-out periods:               {share_unavailable:.2f}")
print(f"  of which on-shelf periods with no demand: {share_true_zero:.2f}")
print(f"share of periods with demand lost to a stock-out: {share_censored:.2f}")
```


    share of zero-sales periods:                0.50
      of which stock-out periods:               0.40
      of which on-shelf periods with no demand: 0.11
    share of periods with demand lost to a stock-out: 0.33


We plot ten representative series, spanning the panel from the fastest movers to the slowest ones, with the stock-out periods shaded. The shaded zeros are exactly the ones plain TSB misreads.


    In [5]:


``` python
def shade_stockouts(ax: Axes, t_axis: np.ndarray, available: np.ndarray) -> Artist:
    """Shade the periods where the product is off the shelf.

    Parameters
    ----------
    ax
        The axes to draw on.
    t_axis
        Time values of length ``time``.
    available
        The 0/1 availability values along ``t_axis``.

    Returns
    -------
    Artist
        The shading artist, labeled ``"stock-out"`` for legends.
    """
    return ax.fill_between(
        t_axis,
        0,
        1,
        where=np.asarray(available) == 0,
        step="mid",
        transform=ax.get_xaxis_transform(),
        color="C3",
        alpha=0.15,
        lw=0,
        label="stock-out",
    )


order = np.argsort(lam)[::-1]
example_series = [int(order[10]), int(order[500]), int(order[900])]
display_series = [int(order[rank]) for rank in np.linspace(5, n_series - 10, 10, dtype=int)]
print(
    f"display series {display_series} with rates "
    f"{[round(float(lam[i]), 2) for i in display_series]}"
)

fig, axes = plt.subplots(
    nrows=10, ncols=1, figsize=(12, 22), sharex=True, sharey=False, layout="constrained"
)
for ax, i in zip(axes, display_series, strict=True):
    ax.plot(t_train, train_data[:, i], "o-", color="black", lw=1, ms=3, label="train")
    ax.plot(t_test, test_data[:, i], "o-", color="C1", lw=1, ms=3, label="test")
    shade = shade_stockouts(ax, t, np.asarray(panel.available[:, i]))
    split_line = ax.axvline(t_max_train, color="gray", ls="--", label="train/test split")
    ax.set(title=rf"series {i} ($\lambda = {lam[i]:.2f}$)", ylabel="sales")
axes[-1].set(xlabel="time")
train_line, test_line = axes[-1].get_lines()[:2]
fig.legend(
    handles=[train_line, test_line, split_line, shade],
    loc="outside lower center",
    ncol=4,
)
fig.suptitle("Observed sales and stock-outs (ten example series)", fontsize=18, fontweight="bold");
```


    display series [25, 91, 430, 526, 806, 241, 46, 330, 803, 261] with rates [8.19, 4.35, 3.46, 2.84, 2.35, 1.93, 1.61, 1.27, 0.91, 0.34]


<figure class="figure">
<p><img src="availability_tsb_files/figure-html/cell-6-output-2.png" class="figure-img" width="1211" height="2206" /></p>
</figure>


# From Croston to TSB to availability constraints

All three methods in this trilogy decompose the sparse series into a **demand size** and an **occurrence** component and run simple exponential smoothing on each; they differ only in what the occurrence component is and *when* it updates. Writing \\\ell_t\\ for a component level, every recursion below is the same masked update \\\ell_t = \ell\_{t-1} + g_t \\ \alpha \\ (x_t - \ell\_{t-1})\\ with a different gate \\g_t\\:

| method | occurrence component | update gate \\g_t\\ | what \\\hat{p}\\ estimates under stock-outs |
|----|----|----|----|
| Croston | inverse inter-demand interval | demand events only | interval-based, availability inflates intervals |
| TSB | demand indicator \\d_t\\ | every period | \\P(\text{available}) \cdot P(\text{demand} \mid \text{available})\\ |
| availability TSB | demand indicator \\d_t\\ | available periods \\a_t = 1\\ | \\P(\text{demand} \mid \text{available})\\ |

[Croston's method](https://juanitorduz.github.io/numpyro_forecast/examples/croston.html) updates both components only at demand events, so a stock-out run simply freezes it, but it also *stretches the measured inter-demand intervals*: the drought caused by the stock-out is booked as demand slowing down, and there is no natural place in the interval bookkeeping to discount it. [TSB](https://juanitorduz.github.io/numpyro_forecast/examples/tsb.html) replaces the intervals with the demand indicator \\d_t = \mathbf{1}\[y_t \> 0\]\\ smoothed at every period:

\\ \hat{p}\_t = \begin{cases} \beta + (1 - \beta) \\ \hat{p}\_{t-1} & \text{if } y_t \> 0, \\ (1 - \beta) \\ \hat{p}\_{t-1} & \text{if } y_t = 0. \end{cases} \\

This is the method's strength on genuinely fading demand and its weakness under censoring: the second branch fires on stock-out zeros too. The blog post's hack rewrites the zero branch as

\\ \hat{p}\_t = (1 - a_t \\ \beta) \\ \hat{p}\_{t-1}, \\

so an on-shelf zero (\\a_t = 1\\) decays the probability exactly as in TSB, while an off-shelf period (\\a_t = 0\\) leaves it untouched. Since a sale requires the product on the shelf (\\y_t \> 0 \Rightarrow a_t = 1\\), all branches collapse into the single gated recursion

\\ \hat{p}\_t = \hat{p}\_{t-1} + a_t \\ \beta \\ (d_t - \hat{p}\_{t-1}): \\

simple exponential smoothing of the demand indicator, updated **only when the product is available**. The point forecast becomes \\\hat{y}\_{t+h} = a\_{t+h} \\ \hat{z}\_t \\ \hat{p}\_t\\ with the *future* availability \\a\_{t+h}\\ chosen by the forecaster, which is what turns the model into a scenario tool. And because plain TSB is recovered exactly by setting \\a_t \equiv 1\\, the comparison at the end of this notebook needs no second model: it just feeds the same model an all-ones availability input.


# Prior for the smoothing parameters

Both smoothing parameters get a \\\text{Beta}(2, 8)\\ prior. It keeps the classical center (mean \\2/10 = 0.2\\, matching the standard practice of smoothing parameters roughly in \\\[0.1, 0.3\]\\) but is deliberately *wider* than the blog post's \\\text{Beta}(10, 40)\\: same mean, far fatter tails, so a series whose data genuinely call for a very reactive or very stiff level can reach it. With \\1{,}000\\ series each contributing its own posterior, there is no reason to constrain them tightly a priori.


    In [6]:


``` python
prior_mean = 2 / 10

fig, ax = plt.subplots(figsize=(9, 5))
pz.Beta(2, 8).plot_pdf(ax=ax, color="C0")
pz.Beta(10, 40).plot_pdf(ax=ax, alpha=0.7)
ax.axvline(prior_mean, color="C1", ls="--", label="prior mean")
ax.legend()
ax.set(
    title=r"Smoothing parameter prior: $\text{Beta}(2, 8)$ vs the blog post's $\text{Beta}(10, 40)$",
    xlabel="smoothing parameter",
    ylabel="density",
);
```


<figure class="figure">
<p><img src="availability_tsb_files/figure-html/cell-7-output-1.png" class="figure-img" width="793" height="481" /></p>
</figure>


# Model specification

The model is the TSB notebook's two-component construction promoted to a panel, with one structural change. The reusable `panel_level_model` runs the gated level recursion for all series at once: the per-series parameters (sites `smoothing` and `noise`) are sampled inside a `numpyro.plate` over series, one `jax.lax.scan` over the calendar axis carries the whole `(series,)` level vector, and, when forecasting, the component draws its flat predictive at a site named [future](../../../reference/forecaster.ForecastingModel.md#numpyro_forecast.forecaster.ForecastingModel.future). Composing with NumPyro's [`scope`](https://num.pyro.ai/en/stable/handlers.html#scope) handler under the prefixes `z` and `p` yields the parameter names `z_smoothing`, …, `p_future`, just like the siblings.

The remaining choices, and where their numbers come from:

- **Level inits.** Following the blog post, the levels start deterministically rather than sampled as in the sibling notebooks: the demand-size level starts at the first observation, \\\ell^z_0 = y_0\\, and the demand probability starts at \\\hat{p}\_0 = 0.5\\, the agnostic "no idea whether this period sees demand" value that the data then pull toward each series' true probability.
- **Noise priors.** The demand-size noise is hierarchical: a global scale \\\sigma\_{\text{scale}} \sim \text{LogNormal}(\log 5, 0.5)\\ (centered on the blog post's value of \\5\\, slightly wider than its \\0.3\\ log-scale) with per-series \\\sigma_i \sim \text{HalfNormal}(\sigma\_{\text{scale}})\\, which shares strength across \\1{,}000\\ series that individually see only a handful of demand events. The probability component instead gets a fixed weakly informative \\\sigma_i \sim \text{HalfNormal}(1)\\: its observations live in \\\[0, 1\]\\, so a scale of order one is already essentially flat and there is nothing for a hierarchy to learn.
- **Noise floors.** One pragmatic addition over the blog post: each component's observation scale gets a small constant floor (\\0.1\\ on the sizes, \\0.05\\ on the indicator, well below any scale the data support). With this many series, some have every training demand equal (all \\1\\s is common for slow movers) or no on-shelf demand at all, and without the floor SVI drives those series' scales toward zero until the ELBO turns NaN late in the optimization.

The `availability_tsb` body then does what is specific to this method:

1.  **Bookkeeping.** From the covariates it reads the observed sales prefix (input `0`), the availability mask (input `1`), and the *future* availability rows, and builds the demand indicator.
2.  **The one-line innovation.** The demand-size component is gated by `is_demand`, exactly as in Croston and TSB. The demand-probability component smooths the indicator gated by `available`: where the TSB notebook passes an all-true `every_period` gate, this model passes the availability mask. That single argument is the whole method.
3.  **In sample.** The size likelihood `"obs"` is masked to demand events, as in the siblings. The probability likelihood `"obs_prob"` is masked to *available* periods: an off-shelf indicator observation carries no demand information, so it contributes no likelihood either. The deterministic sites expose the uncensored `"demand_rate"` (\\\hat{z}\_{t-1} \hat{p}\_{t-1}\\), the censored `"rate"` (\\a_t \hat{z}\_{t-1} \hat{p}\_{t-1}\\, the expected *sales*), and the probability path `"prob"`.
4.  **Out of sample.** The `"forecast"` site is the component predictives' product times the future availability read from the covariates, \\a \cdot \hat{z} \cdot \hat{p}\\, so the same fitted model forecasts any availability scenario.


    In [7]:


``` python
def panel_level_model(
    values: Array,
    is_event: Array,
    future: int,
    init: Array,
    noise_scale: Array | float,
    noise_floor: float,
) -> tuple[Array, Array, Array | None]:
    """Gated simple exponential smoothing level model on a ``(time, series)`` panel.

    Samples the per-series component priors (sites ``smoothing``, ``noise``)
    inside a plate over series, runs the where-gated level recursion along the
    calendar axis, and, when ``future > 0``, draws the flat forecast predictive
    at the site ``future``. Meant to be called under
    :func:`numpyro.handlers.scope`, which prefixes the site names per component.
    This is the sibling notebooks' ``level_model`` promoted to a panel, with
    deterministic inits and a hierarchical noise scale following the blog post.

    Parameters
    ----------
    values
        Observed component values, shape ``(time, series)``; read only where
        ``is_event`` is true.
    is_event
        Boolean update gate, shape ``(time, series)``; the level only updates
        where it is true (the availability mask for the demand-probability
        component).
    future
        Number of forecast steps (``0`` while training).
    init
        Initial level per series, shape ``(series,)``.
    noise_scale
        Scale of the ``HalfNormal`` prior on the per-series observation noise.
    noise_floor
        Constant added to the sampled noise, keeping the observation scale
        away from zero for series whose component values are constant.

    Returns
    -------
    tuple[Array, Array, Array | None]
        The one-step-ahead means (the pre-update levels), the observation noise
        scale, and the forecast predictive draws (``None`` when ``future == 0``).
    """
    n_series = values.shape[-1]

    with numpyro.plate("series", n_series):
        # cast() only narrows numpyro's union return type for the type checker.
        smoothing = cast(
            Array, numpyro.sample("smoothing", dist.Beta(concentration1=2, concentration0=8))
        )
        noise = noise_floor + cast(
            Array, numpyro.sample("noise", dist.HalfNormal(scale=noise_scale))
        )

    def transition_fn(carry, inputs):
        x_t, event_t = inputs
        level = jnp.where(event_t, smoothing * x_t + (1 - smoothing) * carry, carry)
        # Emit the pre-update level: the one-step-ahead mean.
        return level, carry

    last_level, mu = jax.lax.scan(transition_fn, init, (values, is_event))

    future_draws = None
    if future > 0:
        future_draws = cast(
            Array,
            numpyro.sample(
                "future",
                dist.Normal(loc=last_level, scale=noise).expand([future, n_series]).to_event(2),
            ),
        )
    return mu, noise, future_draws


def availability_tsb(h: Horizon, covariates: Array) -> None:
    """TSB with an availability-gated demand-probability component, on a series panel.

    Identical to the TSB body except for the demand-probability component's
    update gate: the availability mask (input ``1`` of the covariates) instead
    of an all-true every-period gate. Plain TSB is recovered exactly by
    feeding an all-ones availability input.

    Parameters
    ----------
    h
        The train/forecast horizon for the current model call.
    covariates
        Two-input tensor ``(covariate, time, series)`` spanning the full
        horizon: input ``0`` is the observed sales history (only the first
        ``h.t_obs`` rows are read), input ``1`` the availability mask (its
        trailing rows define the forecast's availability scenario).
    """
    y = covariates[0, : h.t_obs, :]
    available = covariates[1, : h.t_obs, :] > 0
    available_future = covariates[1, h.t_obs :, :]
    is_demand = y > 0
    demand_indicator = is_demand.astype(y.dtype)

    # cast() only narrows numpyro's union return type for the type checker.
    noise_scale = cast(
        Array, numpyro.sample("noise_scale", dist.LogNormal(loc=jnp.log(5), scale=0.5))
    )

    # Demand-size component: identical to Croston/TSB (updates only at demand events).
    z_mu, z_noise, z_future = scope(panel_level_model, "z", divider="_")(
        y, is_demand, h.future, init=y[0], noise_scale=noise_scale, noise_floor=0.1
    )
    # Demand-probability component: THE one-line innovation. TSB passes an
    # all-true gate here; the availability mask freezes the update off the shelf.
    p_mu, p_noise, p_future = scope(panel_level_model, "p", divider="_")(
        demand_indicator,
        available,
        h.future,
        init=0.5 * jnp.ones_like(y[0]),
        noise_scale=1.0,
        noise_floor=0.05,
    )

    numpyro.deterministic("demand_rate", z_mu * p_mu)
    numpyro.deterministic("rate", available * z_mu * p_mu)
    numpyro.deterministic("prob", p_mu)
    numpyro.sample("obs", dist.Normal(loc=z_mu, scale=z_noise).mask(is_demand), obs=h.data)
    numpyro.sample(
        "obs_prob",
        dist.Normal(loc=p_mu, scale=p_noise).mask(available),  # off-shelf: no likelihood
        obs=demand_indicator,
    )

    if z_future is not None and p_future is not None:  # exactly when h.future > 0
        numpyro.deterministic("z_forecast", z_future)
        numpyro.deterministic("p_forecast", p_future)
        numpyro.deterministic("forecast", available_future * z_future * p_future)


model = forecasting_model(availability_tsb)
```


## Prior predictive check

Before fitting we draw from the prior predictive with NumPyro's `Predictive` and look at the implied `"rate"` paths for one example series. The recursions are driven by the observed history through the covariates, so even under the prior the rate follows the data's rough shape; what the prior controls is how strongly each observation moves the levels and how wide the bands are. This cell also defines the small band-plot helpers (`hdi_label`, `stacked_draws`, and `plot_band_forecast`, shared with the sibling notebooks) used by every band plot below.


    In [8]:


``` python
def hdi_label(prob: float, prefix: str = "") -> str:
    r"""Legend label for an HDI band, e.g. ``$94\%$ HDI``."""
    percent = f"{prob:.0%}".replace("%", r"\%")
    return f"{prefix}${percent}$ HDI"


hdi_probs = (0.5, 0.94)
hdi_alphas = [0.6, 0.3]  # 50% band darker, 94% band lighter


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
    Overlays (observed series, split lines, stock-out shading) and the legend
    are the caller's responsibility.

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


rng_key, rng_subkey = random.split(rng_key)
prior_predictive = Predictive(model, num_samples=500)
prior_samples = prior_predictive(rng_subkey, covariates_train)
prior_rate = np.asarray(prior_samples["rate"])

i = example_series[0]
ax, handles = plot_band_forecast(
    prior_rate[:, :, [i]],
    t_train.astype(float),
    "C0",
    label_prefix="prior rate ",
    figsize=(10.0, 6.0),
)
(obs_line,) = ax.plot(
    t_train, np.asarray(train_data[:, i]), "o-", color="black", lw=1, ms=4, label="observed"
)
shade = shade_stockouts(ax, t_train, np.asarray(available_train[:, i]))
handles[2].set_label("prior rate mean")
ax.legend(
    handles=[*handles, obs_line, shade],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=3,
)
ax.set(title=f"Prior predictive rate (series {i})", xlabel="time", ylabel="sales");
```


    /Users/juanitorduz/Documents/numpyro_forecast/.venv/lib/python3.14/site-packages/arviz_plots/plots/lm_plot.py:360: UserWarning: When multiple credible intervals are plotted, it is recommended to map 'alpha' aesthetic to 'prob' dimension to differentiate between intervals.
      warnings.warn(


<figure class="figure">
<p><img src="availability_tsb_files/figure-html/cell-9-output-2.png" class="figure-img" width="1011" height="611" /></p>
</figure>


# Inference with SVI

With \\1{,}000\\ series the posterior has about \\4{,}000\\ latent dimensions (four per-series parameters, the two components' smoothing and noise, plus the global noise scale), which is exactly the regime where the sibling notebooks' NUTS setup stops being the right tool and stochastic variational inference shines. We fit with the functional [`fit_svi`](https://juanitorduz.github.io/numpyro_forecast/reference/functional.svi.fit_svi.html), following the blog post's configuration: an `AutoNormal` guide (the default) and `Adam` with learning rate \\0.001\\ for \\10{,}000\\ steps. The ELBO loss settles well before the end of the run.


    In [9]:


``` python
num_steps = 10_000

rng_key, rng_subkey = random.split(rng_key)
fit = fit_svi(
    rng_subkey,
    model,
    train_data,
    covariates_train,
    optim=0.001,
    num_steps=num_steps,
)
print(f"mean ELBO loss over the last 100 steps: {float(jnp.mean(fit.losses[-100:])):,.0f}")

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(np.asarray(fit.losses))
ax.set(title="ELBO loss", xlabel="SVI step", ylabel="loss");
```


    mean ELBO loss over the last 100 steps: 57,127


<figure class="figure">
<p><img src="availability_tsb_files/figure-html/cell-10-output-2.png" class="figure-img" width="911" height="611" /></p>
</figure>


Is the plain constant-rate `Adam` leaving anything on the table? The [fresh-retail example](https://juanitorduz.github.io/numpyro_forecast/examples/fresh_retail_stockout.html) gets a real benefit from a fancier `optax` recipe (a one-cycle learning-rate schedule chained with a reduce-on-plateau backoff), so we run the same recipe here and compare where the ELBO lands. On this model the two optimizers end in the same place, so the schedule buys nothing that the simple configuration does not already deliver, and we keep `Adam` for the rest of the notebook. The comparison is worth keeping around, though: it is a one-argument change ([fit_svi](../../../reference/functional.svi.fit_svi.md#numpyro_forecast.functional.svi.fit_svi) accepts any `optax` optimizer via `optim=`), and on harder posterior geometries, like the retail example's hierarchical model on real data, it is the difference between converging and stalling.


    In [10]:


``` python
scheduler = optax.linear_onecycle_schedule(
    transition_steps=num_steps,
    peak_value=0.01,
    pct_start=0.3,
    pct_final=0.85,
    div_factor=2,
    final_div_factor=3,
)
custom_optimizer = optax.chain(
    optax.adam(learning_rate=scheduler),
    optax.contrib.reduce_on_plateau(factor=0.8, patience=20, accumulation_size=100),
)

rng_key, rng_subkey = random.split(rng_key)
fit_onecycle = fit_svi(
    rng_subkey,
    model,
    train_data,
    covariates_train,
    optim=custom_optimizer,
    num_steps=num_steps,
)

loss_adam = float(jnp.mean(fit.losses[-100:]))
loss_onecycle = float(jnp.mean(fit_onecycle.losses[-100:]))
print(f"last-100 mean ELBO loss, constant-rate Adam:        {loss_adam:,.1f}")
print(f"last-100 mean ELBO loss, one-cycle + plateau chain: {loss_onecycle:,.1f}")
print(f"relative difference: {abs(loss_onecycle - loss_adam) / loss_adam:.2%}")
```


    last-100 mean ELBO loss, constant-rate Adam:        57,127.2
    last-100 mean ELBO loss, one-cycle + plateau chain: 57,130.9
    relative difference: 0.01%


# Diagnostics

We export the fit into an ArviZ-schema `xarray.DataTree` with [`to_datatree`](https://juanitorduz.github.io/numpyro_forecast/reference/convert.to_datatree.html). Because we pass the *extended* covariates (whose availability input carries the realized test availability), the tree automatically gains `predictions` groups holding the out-of-sample forecast draws for that scenario. We register the three per-timestep deterministics so they share the tree-wide `time` coordinate, name the covariate axes explicitly (the covariates are `3`-D here, so the default two-name layout does not apply), and bound the accelerator memory of the predictive pass with `predictive_batch_size`, since every stored site on this panel is a `(draws, time, series)` block.


    In [11]:


``` python
rng_key, rng_subkey = random.split(rng_key)
tree = to_datatree(
    rng_subkey,
    fit,
    model,
    train_data,
    covariates_full,
    num_predictive_samples=1_000,
    predictive_batch_size=250,
    posterior_dims={
        "rate": ["time", "obs_dim"],
        "demand_rate": ["time", "obs_dim"],
        "prob": ["time", "obs_dim"],
    },
    covariate_dims=["covariate", "time", "obs_dim"],
    coords={"covariate": ["sales", "availability"]},
)
tree
```


![](data:image/svg+xml;base64,PHN2ZyBzdHlsZT0icG9zaXRpb246IGFic29sdXRlOyB3aWR0aDogMDsgaGVpZ2h0OiAwOyBvdmVyZmxvdzogaGlkZGVuIj4KPGRlZnM+CjxzeW1ib2wgaWQ9Imljb24tZGF0YWJhc2UiIHZpZXdib3g9IjAgMCAzMiAzMiI+CjxwYXRoIGQ9Ik0xNiAwYy04LjgzNyAwLTE2IDIuMjM5LTE2IDV2NGMwIDIuNzYxIDcuMTYzIDUgMTYgNXMxNi0yLjIzOSAxNi01di00YzAtMi43NjEtNy4xNjMtNS0xNi01eiIgLz4KPHBhdGggZD0iTTE2IDE3Yy04LjgzNyAwLTE2LTIuMjM5LTE2LTV2NmMwIDIuNzYxIDcuMTYzIDUgMTYgNXMxNi0yLjIzOSAxNi01di02YzAgMi43NjEtNy4xNjMgNS0xNiA1eiIgLz4KPHBhdGggZD0iTTE2IDI2Yy04LjgzNyAwLTE2LTIuMjM5LTE2LTV2NmMwIDIuNzYxIDcuMTYzIDUgMTYgNXMxNi0yLjIzOSAxNi01di02YzAgMi43NjEtNy4xNjMgNS0xNiA1eiIgLz4KPC9zeW1ib2w+CjxzeW1ib2wgaWQ9Imljb24tZmlsZS10ZXh0MiIgdmlld2JveD0iMCAwIDMyIDMyIj4KPHBhdGggZD0iTTI4LjY4MSA3LjE1OWMtMC42OTQtMC45NDctMS42NjItMi4wNTMtMi43MjQtMy4xMTZzLTIuMTY5LTIuMDMwLTMuMTE2LTIuNzI0Yy0xLjYxMi0xLjE4Mi0yLjM5My0xLjMxOS0yLjg0MS0xLjMxOWgtMTUuNWMtMS4zNzggMC0yLjUgMS4xMjEtMi41IDIuNXYyN2MwIDEuMzc4IDEuMTIyIDIuNSAyLjUgMi41aDIzYzEuMzc4IDAgMi41LTEuMTIyIDIuNS0yLjV2LTE5LjVjMC0wLjQ0OC0wLjEzNy0xLjIzLTEuMzE5LTIuODQxek0yNC41NDMgNS40NTdjMC45NTkgMC45NTkgMS43MTIgMS44MjUgMi4yNjggMi41NDNoLTQuODExdi00LjgxMWMwLjcxOCAwLjU1NiAxLjU4NCAxLjMwOSAyLjU0MyAyLjI2OHpNMjggMjkuNWMwIDAuMjcxLTAuMjI5IDAuNS0wLjUgMC41aC0yM2MtMC4yNzEgMC0wLjUtMC4yMjktMC41LTAuNXYtMjdjMC0wLjI3MSAwLjIyOS0wLjUgMC41LTAuNSAwIDAgMTUuNDk5LTAgMTUuNSAwdjdjMCAwLjU1MiAwLjQ0OCAxIDEgMWg3djE5LjV6IiAvPgo8cGF0aCBkPSJNMjMgMjZoLTE0Yy0wLjU1MiAwLTEtMC40NDgtMS0xczAuNDQ4LTEgMS0xaDE0YzAuNTUyIDAgMSAwLjQ0OCAxIDFzLTAuNDQ4IDEtMSAxeiIgLz4KPHBhdGggZD0iTTIzIDIyaC0xNGMtMC41NTIgMC0xLTAuNDQ4LTEtMXMwLjQ0OC0xIDEtMWgxNGMwLjU1MiAwIDEgMC40NDggMSAxcy0wLjQ0OCAxLTEgMXoiIC8+CjxwYXRoIGQ9Ik0yMyAxOGgtMTRjLTAuNTUyIDAtMS0wLjQ0OC0xLTFzMC40NDgtMSAxLTFoMTRjMC41NTIgMCAxIDAuNDQ4IDEgMXMtMC40NDggMS0xIDF6IiAvPgo8L3N5bWJvbD4KPC9kZWZzPgo8L3N2Zz4=)

``` xr-text-repr-fallback
<xarray.DataTree>
Group: /
│   Attributes:
│       inference_library:  numpyro
│       creation_library:   numpyro_forecast
│       sample_dims:        ['chain', 'draw']
├── Group: /posterior
│       Dimensions:            (chain: 1, draw: 1000, time: 50, obs_dim: 1000,
│                               p_noise_dim_0: 1000, p_smoothing_dim_0: 1000,
│                               z_noise_dim_0: 1000, z_smoothing_dim_0: 1000)
│       Coordinates:
│         * chain              (chain) int64 8B 0
│         * draw               (draw) int64 8kB 0 1 2 3 4 5 ... 994 995 996 997 998 999
│         * time               (time) int64 400B 0 1 2 3 4 5 6 ... 43 44 45 46 47 48 49
│         * obs_dim            (obs_dim) int64 8kB 0 1 2 3 4 5 ... 995 996 997 998 999
│         * p_noise_dim_0      (p_noise_dim_0) int64 8kB 0 1 2 3 4 ... 996 997 998 999
│         * p_smoothing_dim_0  (p_smoothing_dim_0) int64 8kB 0 1 2 3 ... 996 997 998 999
│         * z_noise_dim_0      (z_noise_dim_0) int64 8kB 0 1 2 3 4 ... 996 997 998 999
│         * z_smoothing_dim_0  (z_smoothing_dim_0) int64 8kB 0 1 2 3 ... 996 997 998 999
│       Data variables:
│           demand_rate        (chain, draw, time, obs_dim) float32 200MB 0.5 ... 3.188
│           noise_scale        (chain, draw) float32 4kB 1.642 1.596 ... 1.604 1.663
│           p_noise            (chain, draw, p_noise_dim_0) float32 4MB 0.2356 ... 0....
│           p_smoothing        (chain, draw, p_smoothing_dim_0) float32 4MB 0.3565 .....
│           prob               (chain, draw, time, obs_dim) float32 200MB 0.5 ... 0.9987
│           rate               (chain, draw, time, obs_dim) float32 200MB 0.5 ... 0.0
│           z_noise            (chain, draw, z_noise_dim_0) float32 4MB 2.273 ... 1.783
│           z_smoothing        (chain, draw, z_smoothing_dim_0) float32 4MB 0.07928 ....
│       Attributes:
│           created_at:                 2026-07-21T13:34:21.325380+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
│           variational:                True
├── Group: /posterior_predictive
│       Dimensions:  (chain: 1, draw: 1000, time: 50, obs_dim: 1000)
│       Coordinates:
│         * chain    (chain) int64 8B 0
│         * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│         * time     (time) int64 400B 0 1 2 3 4 5 6 7 8 ... 41 42 43 44 45 46 47 48 49
│         * obs_dim  (obs_dim) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│       Data variables:
│           obs      (chain, draw, time, obs_dim) float32 200MB -2.585 0.9524 ... 4.154
│       Attributes:
│           created_at:                 2026-07-21T13:34:21.754916+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
├── Group: /observed_data
│       Dimensions:  (time: 50, obs_dim: 1000)
│       Coordinates:
│         * time     (time) int64 400B 0 1 2 3 4 5 6 7 8 ... 41 42 43 44 45 46 47 48 49
│         * obs_dim  (obs_dim) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│       Data variables:
│           obs      (time, obs_dim) float32 200kB 1.0 1.0 2.0 3.0 ... 2.0 0.0 0.0 0.0
│       Attributes:
│           created_at:                 2026-07-21T13:34:21.755161+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /constant_data
│       Dimensions:     (covariate: 2, time: 50, obs_dim: 1000)
│       Coordinates:
│         * covariate   (covariate) <U12 96B 'sales' 'availability'
│         * time        (time) int64 400B 0 1 2 3 4 5 6 7 8 ... 42 43 44 45 46 47 48 49
│         * obs_dim     (obs_dim) int64 8kB 0 1 2 3 4 5 6 ... 994 995 996 997 998 999
│       Data variables:
│           covariates  (covariate, time, obs_dim) float32 400kB 1.0 1.0 2.0 ... 1.0 0.0
│       Attributes:
│           created_at:                 2026-07-21T13:34:21.755812+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /predictions
│       Dimensions:  (chain: 1, draw: 1000, time: 10, obs_dim: 1000)
│       Coordinates:
│         * chain    (chain) int64 8B 0
│         * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│         * time     (time) int64 80B 50 51 52 53 54 55 56 57 58 59
│         * obs_dim  (obs_dim) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│       Data variables:
│           obs      (chain, draw, time, obs_dim) float32 40MB -0.0 1.134 ... -0.0 0.0
│       Attributes:
│           created_at:                 2026-07-21T13:34:22.008860+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
└── Group: /predictions_constant_data
        Dimensions:     (covariate: 2, time: 10, obs_dim: 1000)
        Coordinates:
          * covariate   (covariate) <U12 96B 'sales' 'availability'
          * time        (time) int64 80B 50 51 52 53 54 55 56 57 58 59
          * obs_dim     (obs_dim) int64 8kB 0 1 2 3 4 5 6 ... 994 995 996 997 998 999
        Data variables:
            covariates  (covariate, time, obs_dim) float32 80kB 0.0 0.0 0.0 ... 0.0 0.0
        Attributes:
            created_at:                 2026-07-21T13:34:22.009184+00:00
            creation_library:           ArviZ
            creation_library_version:   1.2.0
            creation_library_language:  Python
            sample_dims:                []
```


xarray.DataTree


/posterior(22)

Dimensions:


- chain: 1
- draw: 1000
- time: 50
- obs_dim: 1000
- p_noise_dim_0: 1000
- p_smoothing_dim_0: 1000
- z_noise_dim_0: 1000
- z_smoothing_dim_0: 1000


Coordinates: (8)


chain


(chain)


int64


0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0])


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


0 1 2 3 4 5 6 ... 44 45 46 47 48 49


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])


obs_dim


(obs_dim)


int64


0 1 2 3 4 5 ... 995 996 997 998 999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))


p_noise_dim_0


(p_noise_dim_0)


int64


0 1 2 3 4 5 ... 995 996 997 998 999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))


p_smoothing_dim_0


(p_smoothing_dim_0)


int64


0 1 2 3 4 5 ... 995 996 997 998 999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))


z_noise_dim_0


(z_noise_dim_0)


int64


0 1 2 3 4 5 ... 995 996 997 998 999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))


z_smoothing_dim_0


(z_smoothing_dim_0)


int64


0 1 2 3 4 5 ... 995 996 997 998 999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))


Data variables: (8)


demand_rate


(chain, draw, time, obs_dim)


float32


0.5 0.5 1.0 ... 2.512 0.04121 3.188


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[0.5       , 0.5       , 1.        , ..., 0.        ,0.        , 1.5       ],[0.6782645 , 0.5416363 , 1.259012  , ..., 0.        ,0.        , 1.821347  ],[1.1701968 , 0.49653283, 0.93291295, ..., 0.48845178,0.        , 2.196816  ],...,[2.7286556 , 1.4210991 , 1.7483305 , ..., 1.895539  ,0.10322234, 3.5041564 ],[2.7286556 , 1.4210991 , 1.7483305 , ..., 2.1258059 ,0.07261866, 3.5945382 ],[2.7286556 , 1.4210991 , 1.8080542 , ..., 2.543216  ,0.07261866, 3.1444442 ]],[[0.5       , 0.5       , 1.        , ..., 0.        ,0.        , 1.5       ],[0.67434454, 0.55141246, 1.2266889 , ..., 0.        ,0.        , 1.831261  ],[1.4757882 , 0.4947135 , 0.9486122 , ..., 0.68103886,0.        , 2.1647592 ],...0.03651327, 3.2493646 ],[2.9599526 , 1.4673674 , 1.5857991 , ..., 2.1478982 ,0.02816564, 3.3343325 ],[2.9599526 , 1.4673674 , 1.6226114 , ..., 2.5577505 ,0.02816564, 3.1081858 ]],[[0.5       , 0.5       , 1.        , ..., 0.        ,0.        , 1.5       ],[0.5861883 , 0.5657135 , 1.1837469 , ..., 0.        ,0.        , 2.1095319 ],[1.3870059 , 0.49136347, 0.96623707, ..., 0.46575168,0.        , 2.5309641 ],...,[2.9010303 , 1.4277786 , 1.7093551 , ..., 1.8990767 ,0.0670123 , 3.3007855 ],[2.9010303 , 1.4277786 , 1.7093551 , ..., 2.1139655 ,0.04121009, 3.3553944 ],[2.9010303 , 1.4277786 , 1.7621943 , ..., 2.512274  ,0.04121009, 3.187746  ]]]],shape=(1, 1000, 50, 1000), dtype=float32)


noise_scale


(chain, draw)


float32


1.642 1.596 1.633 ... 1.604 1.663


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[1.6416291, 1.5958658, 1.6325939, 1.6945783, 1.6003406, 1.6255283,1.6421682, 1.6443455, 1.5555668, 1.614861 , 1.6523478, 1.646847 ,1.6432052, 1.6258223, 1.6975629, 1.6614513, 1.664088 , 1.6003448,1.6012199, 1.5659268, 1.6225545, 1.6466433, 1.60414  , 1.6503952,1.6611876, 1.5827991, 1.7345903, 1.6313452, 1.6109694, 1.672122 ,1.6361485, 1.6599686, 1.6631761, 1.5659422, 1.6108747, 1.6323732,1.6656111, 1.712672 , 1.659887 , 1.688221 , 1.6199827, 1.6414831,1.5965594, 1.6727933, 1.6683942, 1.602181 , 1.6751714, 1.67705  ,1.6301814, 1.6369519, 1.6429139, 1.6195647, 1.641255 , 1.5977068,1.6112276, 1.6104451, 1.6465298, 1.659197 , 1.6195042, 1.6359935,1.715927 , 1.6339296, 1.6720576, 1.607311 , 1.5854268, 1.6463037,1.6791978, 1.6211045, 1.6283805, 1.6229887, 1.6421996, 1.6847124,1.6719561, 1.6599553, 1.6641331, 1.6528298, 1.6591628, 1.6796407,1.6346507, 1.5726473, 1.6558863, 1.6582421, 1.6563594, 1.6812468,1.677731 , 1.6375397, 1.6150467, 1.643502 , 1.6205966, 1.6443094,1.5775449, 1.6493683, 1.6408716, 1.6682272, 1.6853138, 1.585379 ,1.6400694, 1.6736987, 1.6644564, 1.6701779, 1.6790502, 1.5774459,1.6959455, 1.6780759, 1.6057984, 1.6543074, 1.6432146, 1.6565053,1.6687175, 1.571016 , 1.6412761, 1.5717098, 1.6543851, 1.6208785,1.618209 , 1.7078304, 1.6749426, 1.6338297, 1.6305742, 1.6717522,...1.6156881, 1.6516421, 1.5987262, 1.6158215, 1.6227918, 1.6319647,1.7263533, 1.6160934, 1.6532023, 1.6532918, 1.652952 , 1.7065803,1.6541291, 1.6081972, 1.6582243, 1.69543  , 1.6990416, 1.6532215,1.6684816, 1.5904355, 1.6092715, 1.64985  , 1.6278578, 1.6102451,1.64433  , 1.6162637, 1.6274749, 1.6124511, 1.5611887, 1.6266475,1.6727331, 1.656884 , 1.6655791, 1.6261462, 1.6851727, 1.7133818,1.636466 , 1.5766335, 1.691194 , 1.6445113, 1.6533564, 1.6542022,1.6348943, 1.6810308, 1.6593655, 1.7476557, 1.6539026, 1.6339725,1.713681 , 1.6616988, 1.6369412, 1.6439741, 1.6024175, 1.6339896,1.6413014, 1.6423017, 1.6298608, 1.6303184, 1.6531131, 1.5821995,1.6607072, 1.6472602, 1.6942983, 1.6366313, 1.6229575, 1.6195221,1.6737407, 1.7155664, 1.6439842, 1.679561 , 1.5966047, 1.5795963,1.5994356, 1.6327971, 1.6480588, 1.5943656, 1.6719537, 1.6916468,1.6427773, 1.6357688, 1.671581 , 1.6469822, 1.6624147, 1.6527021,1.6147665, 1.6641214, 1.6202685, 1.6304166, 1.6411166, 1.623867 ,1.5806018, 1.5871994, 1.6749038, 1.6542323, 1.6025615, 1.729628 ,1.6453216, 1.6759707, 1.620002 , 1.6143054, 1.6428082, 1.5931753,1.6199028, 1.6883078, 1.6644399, 1.6931082, 1.6182313, 1.5921946,1.6162312, 1.7244012, 1.5856706, 1.6379843, 1.6421332, 1.7258818,1.6155782, 1.6036739, 1.6041226, 1.6627   ]], dtype=float32)


p_noise


(chain, draw, p_noise_dim_0)


float32


0.2356 0.3296 ... 0.3635 0.2637


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.23557004, 0.3295964 , 0.28722733, ..., 0.33739036,0.3955534 , 0.27733847],[0.23538832, 0.41975355, 0.3417156 , ..., 0.38350692,0.41343474, 0.4003674 ],[0.36853442, 0.39166048, 0.29544455, ..., 0.4357567 ,0.24930838, 0.30580193],...,[0.2871821 , 0.35036743, 0.22920299, ..., 0.39317298,0.34532732, 0.36878356],[0.19114229, 0.48188245, 0.38271156, ..., 0.42993554,0.42688757, 0.3214646 ],[0.27185494, 0.38835335, 0.37892863, ..., 0.4895899 ,0.3634896 , 0.26367855]]], shape=(1, 1000, 1000), dtype=float32)


p_smoothing


(chain, draw, p_smoothing_dim_0)


float32


0.3565 0.08327 ... 0.385 0.4064


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.35652906, 0.08327255, 0.2590119 , ..., 0.19484755,0.2964831 , 0.21423137],[0.34868917, 0.10282499, 0.2266888 , ..., 0.36005324,0.29112014, 0.2208406 ],[0.20488365, 0.07814995, 0.1328513 , ..., 0.06689196,0.11641613, 0.23009804],...,[0.16498864, 0.19421962, 0.42516562, ..., 0.19061652,0.1618794 , 0.13161285],[0.27692997, 0.06892153, 0.11122721, ..., 0.24989341,0.228619  , 0.16223253],[0.17237663, 0.13142703, 0.18374693, ..., 0.1762564 ,0.38503695, 0.40635452]]], shape=(1, 1000, 1000), dtype=float32)


prob


(chain, draw, time, obs_dim)


float32


0.5 0.5 0.5 ... 0.05805 0.9987


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[0.5       , 0.5       , 0.5       , ..., 0.5       ,0.5       , 0.5       ],[0.6782645 , 0.5416363 , 0.629506  , ..., 0.5       ,0.5       , 0.6071157 ],[0.79297256, 0.49653283, 0.46645647, ..., 0.5974238 ,0.5       , 0.6912838 ],...,[0.99956113, 0.7600253 , 0.8896235 , ..., 0.77201736,0.11591072, 0.97138715],[0.99956113, 0.7600253 , 0.8896235 , ..., 0.81643915,0.08154514, 0.9775169 ],[0.99956113, 0.7600253 , 0.9182124 , ..., 0.8522055 ,0.08154514, 0.9823335 ]],[[0.5       , 0.5       , 0.5       , ..., 0.5       ,0.5       , 0.5       ],[0.67434454, 0.55141246, 0.61334443, ..., 0.5       ,0.5       , 0.61042035],[0.78789705, 0.4947135 , 0.4743061 , ..., 0.68002665,0.5       , 0.69645536],...0.12834577, 0.9528761 ],[0.9975666 , 0.75083727, 0.8383683 , ..., 0.8290444 ,0.09900349, 0.9605211 ],[0.9975666 , 0.75083727, 0.85634613, ..., 0.8717651 ,0.09900349, 0.96692586]],[[0.5       , 0.5       , 0.5       , ..., 0.5       ,0.5       , 0.5       ],[0.5861883 , 0.5657135 , 0.59187347, ..., 0.5       ,0.5       , 0.7031772 ],[0.6575198 , 0.49136347, 0.48311853, ..., 0.5881282 ,0.5       , 0.82379246],...,[0.9829037 , 0.76581156, 0.8582117 , ..., 0.7730285 ,0.0943976 , 0.99625576],[0.9829037 , 0.76581156, 0.8582117 , ..., 0.8130337 ,0.05805104, 0.9977772 ],[0.9829037 , 0.76581156, 0.8842649 , ..., 0.84598774,0.05805104, 0.9986805 ]]]],shape=(1, 1000, 50, 1000), dtype=float32)


rate


(chain, draw, time, obs_dim)


float32


0.5 0.5 1.0 1.5 ... 0.0 0.04121 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[0.5       , 0.5       , 1.        , ..., 0.        ,0.        , 1.5       ],[0.6782645 , 0.5416363 , 1.259012  , ..., 0.        ,0.        , 1.821347  ],[0.        , 0.        , 0.93291295, ..., 0.        ,0.        , 0.        ],...,[0.        , 0.        , 0.        , ..., 1.895539  ,0.10322234, 3.5041564 ],[0.        , 0.        , 1.7483305 , ..., 2.1258059 ,0.        , 3.5945382 ],[0.        , 1.4210991 , 1.8080542 , ..., 0.        ,0.07261866, 0.        ]],[[0.5       , 0.5       , 1.        , ..., 0.        ,0.        , 1.5       ],[0.67434454, 0.55141246, 1.2266889 , ..., 0.        ,0.        , 1.831261  ],[0.        , 0.        , 0.9486122 , ..., 0.        ,0.        , 0.        ],...0.03651327, 3.2493646 ],[0.        , 0.        , 1.5857991 , ..., 2.1478982 ,0.        , 3.3343325 ],[0.        , 1.4673674 , 1.6226114 , ..., 0.        ,0.02816564, 0.        ]],[[0.5       , 0.5       , 1.        , ..., 0.        ,0.        , 1.5       ],[0.5861883 , 0.5657135 , 1.1837469 , ..., 0.        ,0.        , 2.1095319 ],[0.        , 0.        , 0.96623707, ..., 0.        ,0.        , 0.        ],...,[0.        , 0.        , 0.        , ..., 1.8990767 ,0.0670123 , 3.3007855 ],[0.        , 0.        , 1.7093551 , ..., 2.1139655 ,0.        , 3.3553944 ],[0.        , 1.4277786 , 1.7621943 , ..., 0.        ,0.04121009, 0.        ]]]],shape=(1, 1000, 50, 1000), dtype=float32)


z_noise


(chain, draw, z_noise_dim_0)


float32


2.273 0.9714 ... 0.9517 1.783


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[2.2729077 , 0.97144175, 0.7048762 , ..., 1.2990873 ,0.77874327, 1.6654072 ],[2.1058667 , 0.9474447 , 0.844502  , ..., 1.2889745 ,1.089972  , 1.3748813 ],[2.4379141 , 0.84468067, 0.7863216 , ..., 1.2037662 ,0.38981688, 1.5715235 ],...,[2.1181762 , 0.83416605, 0.6763507 , ..., 1.1266961 ,0.69756234, 1.3436103 ],[1.9629935 , 1.0041883 , 0.9009639 , ..., 1.1429585 ,1.123892  , 1.6957167 ],[1.9611555 , 0.9789245 , 0.961785  , ..., 1.1008172 ,0.9516987 , 1.7827178 ]]], shape=(1, 1000, 1000), dtype=float32)


z_smoothing


(chain, draw, z_smoothing_dim_0)


float32


0.07928 0.2907 ... 0.2661 0.07233


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.07928485, 0.2907415 , 0.11092173, ..., 0.27253225,0.4247977 , 0.17787854],[0.14551206, 0.09308542, 0.07054768, ..., 0.3338295 ,0.22210976, 0.1082524 ],[0.07821747, 0.08379131, 0.11445527, ..., 0.46371633,0.1380882 , 0.22935659],...,[0.31724772, 0.12656786, 0.12887691, ..., 0.19936539,0.12140953, 0.06226573],[0.21951218, 0.2192169 , 0.03021587, ..., 0.24352941,0.08028406, 0.10394039],[0.18490852, 0.29427147, 0.1300808 , ..., 0.263974  ,0.26609617, 0.07233211]]], shape=(1, 1000, 1000), dtype=float32)


Attributes: (6)


created_at :  
2026-07-21T13:34:21.325380+00:00

creation_library :  
ArviZ

creation_library_version :  
1.2.0

creation_library_language :  
Python

sample_dims :  
\['chain', 'draw'\]

variational :  
True


/posterior_predictive(10)

Dimensions:


- chain: 1
- draw: 1000
- time: 50
- obs_dim: 1000


Coordinates: (4)


chain


(chain)


int64


0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0])


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


0 1 2 3 4 5 6 ... 44 45 46 47 48 49


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])


obs_dim


(obs_dim)


int64


0 1 2 3 4 5 ... 995 996 997 998 999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))


Data variables: (1)


obs


(chain, draw, time, obs_dim)


float32


-2.585 0.9524 ... -0.4725 4.154


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[-2.584829  ,  0.9524088 ,  1.6452434 , ..., -2.101253  ,-0.23768803,  2.6046705 ],[-1.4616485 ,  0.04638965,  2.773092  , ..., -3.4859781 ,0.8573891 ,  2.568034  ],[ 1.7744092 ,  2.863715  ,  1.0430939 , ..., -1.2135471 ,-0.14930597,  4.333962  ],...,[-1.2050627 ,  1.7277703 ,  2.5311892 , ..., -1.2706459 ,0.60986936,  2.1594825 ],[ 2.5552487 ,  1.2631192 ,  1.6882025 , ...,  0.6074698 ,0.13905123,  5.567868  ],[ 1.6200869 , -0.21489447,  2.183492  , ...,  4.6327214 ,-0.39420646,  3.6504116 ]],[[-1.7072934 ,  1.8867687 ,  2.4083626 , ..., -3.1491606 ,0.6865282 ,  2.8707554 ],[ 2.3502629 ,  1.4805038 ,  2.066377  , ..., -2.7371752 ,0.25684795,  4.187906  ],[ 2.8947086 ,  2.136424  ,  3.636254  , ..., -1.0803797 ,1.2342826 ,  1.9288054 ],...0.52896786,  6.298092  ],[-0.22268045,  3.0086749 ,  3.1160135 , ...,  3.1890354 ,0.7740424 ,  4.2279406 ],[ 0.04493713,  3.6145422 ,  3.2841437 , ...,  4.3561387 ,-2.4431305 ,  6.317762  ]],[[-1.5166011 ,  2.1837714 ,  2.946469  , ..., -2.7062275 ,1.5667833 ,  4.287099  ],[ 2.9663105 ,  1.8921396 ,  2.12064   , ...,  0.03020589,-0.36844075,  3.8513367 ],[ 1.7266587 , -0.21004365,  1.8016311 , ...,  2.3834224 ,-1.0077399 ,  1.3293755 ],...,[-1.5501094 , -0.6320512 ,  0.54482424, ...,  3.278456  ,-0.93964976,  4.5965323 ],[ 3.9174888 ,  0.30723456,  1.521188  , ...,  2.2805521 ,1.5582681 ,  0.02582051],[ 5.7725286 ,  1.301194  ,  1.6545142 , ...,  4.4337497 ,-0.47245854,  4.1535535 ]]]],shape=(1, 1000, 50, 1000), dtype=float32)


Attributes: (5)


created_at :  
2026-07-21T13:34:21.754916+00:00

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


- time: 50
- obs_dim: 1000


Coordinates: (2)


time


(time)


int64


0 1 2 3 4 5 6 ... 44 45 46 47 48 49


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])


obs_dim


(obs_dim)


int64


0 1 2 3 4 5 ... 995 996 997 998 999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))


Data variables: (1)


obs


(time, obs_dim)


float32


1.0 1.0 2.0 3.0 ... 2.0 0.0 0.0 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[1., 1., 2., ..., 0., 0., 3.],[7., 0., 0., ..., 3., 0., 4.],[0., 0., 1., ..., 0., 0., 0.],...,[0., 0., 0., ..., 3., 0., 4.],[0., 0., 2., ..., 4., 0., 1.],[0., 1., 1., ..., 0., 0., 0.]], shape=(50, 1000), dtype=float32)


Attributes: (5)


created_at :  
2026-07-21T13:34:21.755161+00:00

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


- covariate: 2
- time: 50
- obs_dim: 1000


Coordinates: (3)


covariate


(covariate)


\<U12


'sales' 'availability'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['sales', 'availability'], dtype='<U12')


time


(time)


int64


0 1 2 3 4 5 6 ... 44 45 46 47 48 49


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])


obs_dim


(obs_dim)


int64


0 1 2 3 4 5 ... 995 996 997 998 999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))


Data variables: (1)


covariates


(covariate, time, obs_dim)


float32


1.0 1.0 2.0 3.0 ... 1.0 0.0 1.0 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[1., 1., 2., ..., 0., 0., 3.],[7., 0., 0., ..., 3., 0., 4.],[0., 0., 1., ..., 0., 0., 0.],...,[0., 0., 0., ..., 3., 0., 4.],[0., 0., 2., ..., 4., 0., 1.],[0., 1., 1., ..., 0., 0., 0.]],[[1., 1., 1., ..., 0., 0., 1.],[1., 1., 1., ..., 1., 0., 1.],[0., 0., 1., ..., 0., 0., 0.],...,[0., 0., 0., ..., 1., 1., 1.],[0., 0., 1., ..., 1., 0., 1.],[0., 1., 1., ..., 0., 1., 0.]]],shape=(2, 50, 1000), dtype=float32)


Attributes: (5)


created_at :  
2026-07-21T13:34:21.755812+00:00

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


- chain: 1
- draw: 1000
- time: 10
- obs_dim: 1000


Coordinates: (4)


chain


(chain)


int64


0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0])


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


50 51 52 53 54 55 56 57 58 59


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59])


obs_dim


(obs_dim)


int64


0 1 2 3 4 5 ... 995 996 997 998 999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))


Data variables: (1)


obs


(chain, draw, time, obs_dim)


float32


-0.0 1.134 0.0 ... 4.038 -0.0 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[-0.00000000e+00,  1.13425064e+00,  0.00000000e+00, ...,-2.76441574e-01,  7.85750389e-01,  3.51096129e+00],[-0.00000000e+00,  0.00000000e+00,  1.53906143e+00, ...,0.00000000e+00,  9.11785603e-01,  3.66306877e+00],[ 0.00000000e+00, -0.00000000e+00,  1.42802417e+00, ...,6.83469677e+00, -1.37476969e+00,  0.00000000e+00],...,[ 0.00000000e+00,  1.14195740e+00,  9.80820775e-01, ...,3.55617189e+00, -0.00000000e+00,  0.00000000e+00],[ 1.18154919e+00,  0.00000000e+00,  1.85884058e+00, ...,0.00000000e+00, -2.95453388e-02,  2.32469559e+00],[ 3.53652716e+00,  0.00000000e+00,  1.08322167e+00, ...,1.32379735e+00,  0.00000000e+00,  0.00000000e+00]],[[ 0.00000000e+00,  9.57378149e-01,  0.00000000e+00, ...,8.62511253e+00,  3.13758075e-01,  1.20065804e+01],[ 0.00000000e+00,  0.00000000e+00,  1.38504457e+00, ...,0.00000000e+00, -1.30518861e-02,  1.94735932e+00],[ 0.00000000e+00,  0.00000000e+00,  1.18639715e-01, ...,2.21687675e+00, -2.95517355e-01,  0.00000000e+00],...1.21884310e+00, -0.00000000e+00,  0.00000000e+00],[ 2.90715480e+00,  0.00000000e+00,  7.05792010e-01, ...,0.00000000e+00,  5.17736256e-01,  3.35584593e+00],[ 1.97415721e+00,  0.00000000e+00,  1.07672542e-01, ...,5.27291596e-01,  0.00000000e+00,  0.00000000e+00]],[[ 0.00000000e+00, -1.14590682e-01,  0.00000000e+00, ...,1.77340770e+00, -2.22006872e-01,  8.96911621e-01],[ 0.00000000e+00,  0.00000000e+00, -1.14924334e-01, ...,0.00000000e+00, -1.56876259e-02,  3.59755921e+00],[ 0.00000000e+00,  0.00000000e+00, -6.41032815e-01, ...,4.07854509e+00, -1.05620794e-01,  0.00000000e+00],...,[ 0.00000000e+00,  1.38957596e+00,  8.15236986e-01, ...,3.02757716e+00,  0.00000000e+00,  0.00000000e+00],[ 5.00996590e+00,  0.00000000e+00, -5.27322367e-02, ...,0.00000000e+00, -8.76562655e-01,  3.15467000e+00],[ 6.07991505e+00,  0.00000000e+00,  2.36341929e+00, ...,4.03817368e+00, -0.00000000e+00,  0.00000000e+00]]]],shape=(1, 1000, 10, 1000), dtype=float32)


Attributes: (5)


created_at :  
2026-07-21T13:34:22.008860+00:00

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


- covariate: 2
- time: 10
- obs_dim: 1000


Coordinates: (3)


covariate


(covariate)


\<U12


'sales' 'availability'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['sales', 'availability'], dtype='<U12')


time


(time)


int64


50 51 52 53 54 55 56 57 58 59


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59])


obs_dim


(obs_dim)


int64


0 1 2 3 4 5 ... 995 996 997 998 999


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 997, 998, 999], shape=(1000,))


Data variables: (1)


covariates


(covariate, time, obs_dim)


float32


0.0 0.0 0.0 0.0 ... 1.0 1.0 0.0 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0., 0., 0., ..., 0., 0., 0.],[0., 0., 0., ..., 0., 0., 0.],[0., 0., 0., ..., 0., 0., 0.],...,[0., 0., 0., ..., 0., 0., 0.],[0., 0., 0., ..., 0., 0., 0.],[0., 0., 0., ..., 0., 0., 0.]],[[0., 1., 0., ..., 1., 1., 1.],[0., 0., 1., ..., 0., 1., 1.],[0., 0., 1., ..., 1., 1., 0.],...,[0., 1., 1., ..., 1., 0., 0.],[1., 0., 1., ..., 0., 1., 1.],[1., 0., 1., ..., 1., 0., 0.]]],shape=(2, 10, 1000), dtype=float32)


Attributes: (5)


created_at :  
2026-07-21T13:34:22.009184+00:00

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


A variational fit has no chains to converge, so the MCMC diagnostics of the sibling notebooks (\\\hat{R}\\, effective sample sizes) do not apply; the ELBO curve above plays their role. What we can inspect is the fitted posterior itself. `az.summary` on the global noise scale checks the one shared parameter, and for the \\1{,}000\\-dimensional per-series sites we look at the *distribution* of posterior-mean smoothing parameters across series against the prior mean.


    In [12]:


``` python
az.summary(tree, var_names=["noise_scale"], ci_kind="hdi", ci_prob=0.94, kind="stats")
```


|             | mean | sd   | hdi94_lb | hdi94_ub |
|-------------|------|------|----------|----------|
| noise_scale | 1.6  | 0.04 | 1.6      | 1.7      |


    In [13]:


``` python
posterior = tree["posterior"].dataset
z_sm = posterior["z_smoothing"].mean(dim=("chain", "draw")).to_numpy()
p_sm = posterior["p_smoothing"].mean(dim=("chain", "draw")).to_numpy()

smoothing_means = xr.DataTree.from_dict(
    {"posterior": xr.Dataset({"z_smoothing": ("series", z_sm), "p_smoothing": ("series", p_sm)})}
)
pc = az.plot_dist(
    smoothing_means, sample_dims=["series"], kind="hist", figure_kwargs={"figsize": (12, 4)}
)
for name in ["z_smoothing", "p_smoothing"]:
    ax = pc.viz["plot"][name].item()
    ax.axvline(prior_mean, color="C1", ls="--", label="prior mean")
    ax.legend()
    ax.set(xlabel="posterior mean")
fig = pc.viz["plot"]["z_smoothing"].item().figure
fig.suptitle(
    "Per-series posterior-mean smoothing parameters", fontsize=18, fontweight="bold", y=1.05
);
```


<figure class="figure">
<p><img src="availability_tsb_files/figure-html/cell-14-output-1.png" class="figure-img" width="1211" height="435" /></p>
</figure>


    In [14]:


``` python
high_sm = p_sm > 0.3
p_true_all = 1 - np.exp(-lam)
print(f"series with posterior-mean p_smoothing above 0.3: {int(high_sm.sum())}")
print(f"  their mean true demand probability:      {float(p_true_all[high_sm].mean()):.2f}")
print(f"  remaining series' mean true probability: {float(p_true_all[~high_sm].mean()):.2f}")
```


    series with posterior-mean p_smoothing above 0.3: 307
      their mean true demand probability:      0.97
      remaining series' mean true probability: 0.76


The size component's posterior means stay in the sensible \\\[0.1, 0.5\]\\ range, centered a little above the prior mean (the annotated mean is \\0.24\\): demand is i.i.d. within each series, so there is no genuine trend in the sizes for a reactive smoothing parameter to chase, and fifty periods per series let the data shift the prior only mildly. The probability component is more interesting: its distribution is clearly bimodal, and the printed split shows that the series in the upper mode are the fastest movers, whose true demand probability is essentially \\1\\. For them the demand indicator is a near-constant string of ones, and the quickest way to explain it is to escape the \\\hat{p}\_0 = 0.5\\ initialization fast, so their likelihood rewards a much larger smoothing parameter.


# In-sample fit

For the in-sample story we plot the posterior of the `"rate"` site, the expected *sales* per period \\a_t \hat{z}\_{t-1} \hat{p}\_{t-1}\\, for five example series spanning the panel from fast to slow movers. The availability mask is visible twice in every panel: the rate drops to exactly zero in every shaded stock-out (no sales can happen off the shelf), and between stock-outs it moves gently as the smoothed components track the data.


    In [15]:


``` python
rate_draws = stacked_draws(tree["posterior"], "rate")

insample_series = display_series[::2]
insample_labels = [f"series {i}" for i in insample_series]
idata_rate = predictions_to_datatree(
    rate_draws[:, :, insample_series], t_train.astype(float), insample_labels
)
pc = az.plot_lm(
    idata_rate,
    y="obs",
    x="t",
    plot_dim="time",
    ci_kind="hdi",
    ci_prob=hdi_probs,
    smooth=False,
    point_estimate="mean",
    visuals={
        "ci_band": {"color": "C0"},
        "observed_scatter": False,
        "pe_line": {"color": "C0", "alpha": 1.0, "width": 1.5},
    },
    aes={"alpha": ["prob"]},
    alpha=hdi_alphas,
    col_wrap=1,
    figure_kwargs={"figsize": (12, 16), "sharex": True},
)
axes = pc.viz["plot"]["t"]
for label, i in zip(insample_labels, insample_series, strict=True):
    ax = axes.sel(series=label).item()
    (obs_line,) = ax.plot(
        t_train, np.asarray(train_data[:, i]), "o-", color="black", lw=1, ms=4, label="observed"
    )
    shade = shade_stockouts(ax, t_train, np.asarray(available_train[:, i]))
    ax.set(title=rf"series {i} ($\lambda = {lam[i]:.2f}$)", xlabel="time", ylabel="sales")
bands = pc.viz["ci_band"]["t"]
band_94 = bands.sel(series=insample_labels[0], prob=0.94).item()
band_50 = bands.sel(series=insample_labels[0], prob=0.5).item()
band_94.set_label(hdi_label(0.94, prefix="rate "))
band_50.set_label(hdi_label(0.5, prefix="rate "))
pe_line = pc.viz["pe_line"]["t"].sel(series=insample_labels[0]).item()
pe_line.set_label("rate posterior mean")
fig = pc.viz["figure"].item()
fig.legend(
    handles=[band_94, band_50, pe_line, obs_line, shade],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.04),
    ncol=5,
)
fig.suptitle(
    "In-sample expected sales (five example series)", fontsize=16, fontweight="bold", y=1.02
);
```


    /Users/juanitorduz/Documents/numpyro_forecast/.venv/lib/python3.14/site-packages/arviz_plots/plots/lm_plot.py:360: UserWarning: When multiple credible intervals are plotted, it is recommended to map 'alpha' aesthetic to 'prob' dimension to differentiate between intervals.
      warnings.warn(


<figure class="figure">
<p><img src="availability_tsb_files/figure-html/cell-16-output-2.png" class="figure-img" width="1211" height="1706" /></p>
</figure>


## The demand probability

The probability path is where the innovation lives, so we look at it for the series with the *longest* stock-out run in the training window. Through the shaded run the availability-gated estimate stays **frozen**: plain TSB would decay it by a factor \\(1 - \beta)\\ per period across the same stretch, reading the enforced silence as vanishing demand. Between stock-outs the path does exactly what TSB should do, decaying through on-shelf zeros and jumping at demands: it moves around the series' true demand probability \\1 - e^{-\lambda}\\ (dashed line, never seen by the model), overshooting during the long mid-sample streak of demand events and falling back through the on-shelf zeros that follow.


    In [16]:


``` python
avail_train_np = np.asarray(available_train)
run = np.zeros(n_series)
longest = np.zeros(n_series)
for step in range(t_max_train):
    run = np.where(avail_train_np[step] == 0, run + 1, 0.0)
    longest = np.maximum(longest, run)
j = int(np.argmax(longest))
print(
    f"series {j}: longest stock-out run in train = {int(longest[j])} periods, "
    f"lambda = {lam[j]:.2f}"
)

prob_draws = stacked_draws(tree["posterior"], "prob")

ax, handles = plot_band_forecast(
    prob_draws[:, :, [j]],
    t_train.astype(float),
    "C4",
    label_prefix="probability ",
    figsize=(10.0, 6.0),
)
p_true_j = 1 - np.exp(-lam[j])
true_line = ax.axhline(p_true_j, color="black", ls="--", lw=1, label="true demand probability")
event_times = t_train[np.asarray(train_data[:, j] > 0)]
(rug,) = ax.plot(
    event_times,
    np.zeros_like(event_times, dtype=float),
    "|",
    color="black",
    ms=14,
    label="demand events",
)
shade = shade_stockouts(ax, t_train, avail_train_np[:, j])
ax.legend(
    handles=[*handles, true_line, rug, shade],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=3,
)
ax.set(
    title=f"In-sample demand probability (series {j})",
    xlabel="time",
    ylabel="demand probability",
);
```


    series 178: longest stock-out run in train = 9 periods, lambda = 2.07


    /Users/juanitorduz/Documents/numpyro_forecast/.venv/lib/python3.14/site-packages/arviz_plots/plots/lm_plot.py:360: UserWarning: When multiple credible intervals are plotted, it is recommended to map 'alpha' aesthetic to 'prob' dimension to differentiate between intervals.
      warnings.warn(


<figure class="figure">
<p><img src="availability_tsb_files/figure-html/cell-17-output-3.png" class="figure-img" width="1011" height="611" /></p>
</figure>


# Forecast

The `predictions` group of the tree already holds the out-of-sample draws of the `"forecast"` site under the **realized-availability scenario**: the test window's actual stock-out pattern rode in on the covariates, so the forecast predicts zero sales in the periods the product is genuinely off the shelf and \\\hat{z} \hat{p}\\ plus noise elsewhere. That is the right object to score against the observed test sales, which we do panel-wide with the CRPS and the central-interval coverages.


    In [17]:


``` python
forecast_pp = stacked_draws(tree["predictions"], "obs")
crps_test = float(eval_crps(forecast_pp, np.asarray(test_data)))
cov_50 = float(eval_coverage(forecast_pp, np.asarray(test_data), alpha=0.5))
cov_94 = float(eval_coverage(forecast_pp, np.asarray(test_data), alpha=0.94))
print(f"panel test CRPS (realized availability): {crps_test:.4f}")
print(f"empirical 50% coverage: {cov_50:.2f}  (nominal 0.50)")
print(f"empirical 94% coverage: {cov_94:.2f}  (nominal 0.94)")
```


    panel test CRPS (realized availability): 0.5385
    empirical 50% coverage: 0.70  (nominal 0.50)
    empirical 94% coverage: 0.98  (nominal 0.94)


    In [18]:


``` python
def add_availability_axis(ax: Axes, t_axis: np.ndarray, available: np.ndarray) -> Artist:
    """Draw the availability series on a right-hand twin axis, as in the blog post.

    Parameters
    ----------
    ax
        The axes holding the sales-scale plot.
    t_axis
        Time values of length ``time``.
    available
        The 0/1 availability values along ``t_axis``.

    Returns
    -------
    Artist
        The availability line, labeled ``"availability"`` for legends.
    """
    ax_avail = ax.twinx()
    (avail_line,) = ax_avail.plot(
        t_axis, np.asarray(available), color="C2", lw=1.5, alpha=0.7, label="availability"
    )
    ax_avail.set(ylabel="availability", yticks=[0, 1])
    ax_avail.grid(False)
    return avail_line


i = example_series[0]
ax, handles = plot_band_forecast(
    forecast_pp[:, :, [i]],
    t_test.astype(float),
    "C1",
    label_prefix="forecast ",
    observed=np.asarray(test_data[:, [i]]),
)
(obs_line,) = ax.plot(
    t, np.asarray(panel.sales[:, i]), "o-", color="black", lw=1, ms=4, label="observed"
)
split_line = ax.axvline(t_max_train, color="gray", ls="--", label="train/test split")
shade = shade_stockouts(ax, t, np.asarray(panel.available[:, i]))
avail_line = add_availability_axis(ax, t, np.asarray(panel.available[:, i]))
ax.legend(
    handles=[*handles, obs_line, split_line, shade, avail_line],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=4,
)
ax.set(
    title=f"Realized-availability forecast, series {i} (panel test CRPS: {crps_test:.4f})",
    xlabel="time",
    ylabel="sales",
);
```


    /Users/juanitorduz/Documents/numpyro_forecast/.venv/lib/python3.14/site-packages/arviz_plots/plots/lm_plot.py:360: UserWarning: When multiple credible intervals are plotted, it is recommended to map 'alpha' aesthetic to 'prob' dimension to differentiate between intervals.
      warnings.warn(


<figure class="figure">
<p><img src="availability_tsb_files/figure-html/cell-19-output-2.png" class="figure-img" width="1211" height="611" /></p>
</figure>


The forecast band pinches to zero inside the shaded test stock-outs and re-opens when the product returns to the shelf: the availability input is doing the work directly in the forecast path.


## Zeros at the forecast origin do not drag the forecast down

Here is the single most important picture in this notebook. Consider a series whose training window *ends* in a stock-out run: the last thing the model sees before forecasting is a string of zeros. Every classical intermittent-demand method reads that string as evidence that demand is dying, plain TSB decays its demand probability by \\(1 - \beta)\\ per period through the entire run, and its forecast opens *low* accordingly. But these zeros carry no demand information at all, because the product was off the shelf. We select the series with the longest such trailing run among the fast movers whose test window is mostly on the shelf, so the difference is visible in the observed data.


    In [19]:


``` python
trailing_run = np.cumprod(avail_train_np[::-1] == 0, axis=0).sum(axis=0)
hero_candidates = (lam > np.median(lam)) & (np.asarray(available_test).mean(axis=0) >= 0.8)
k = int(np.argmax(np.where(hero_candidates, trailing_run, -1)))
run_k = int(trailing_run[k])
lost_k = float(panel.demand[t_max_train - run_k : t_max_train, k].sum())
print(
    f"series {k}: last {run_k} training periods are stock-outs, lambda = {lam[k]:.2f}, "
    f"latent demand lost in the run = {lost_k:.0f} units"
)

ax, handles = plot_band_forecast(
    forecast_pp[:, :, [k]],
    t_test.astype(float),
    "C1",
    label_prefix="forecast ",
    observed=np.asarray(test_data[:, [k]]),
)
(train_line,) = ax.plot(
    t_train, np.asarray(train_data[:, k]), "o-", color="C0", lw=1, ms=4, label="train"
)
(test_line,) = ax.plot(
    t_test, np.asarray(test_data[:, k]), "o-", color="black", lw=1, ms=4, label="test (observed)"
)
split_line = ax.axvline(t_max_train, color="gray", ls="--", label="train/test split")
shade = shade_stockouts(ax, t, np.asarray(panel.available[:, k]))
avail_line = add_availability_axis(ax, t, np.asarray(panel.available[:, k]))
ax.annotate(
    f"last {run_k} training periods:\nzeros from stock-outs",
    xy=(t_max_train - run_k / 2, 0.04),
    xycoords=ax.get_xaxis_transform(),
    xytext=(0.55, 0.85),
    textcoords="axes fraction",
    ha="center",
    fontsize=11,
    arrowprops={"arrowstyle": "->", "color": "C3"},
    color="C3",
)
ax.legend(
    handles=[*handles, train_line, test_line, split_line, shade, avail_line],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=4,
)
ax.set(
    title=f"The forecast opens at the demand level (series {k})",
    xlabel="time",
    ylabel="sales",
);
```


    /Users/juanitorduz/Documents/numpyro_forecast/.venv/lib/python3.14/site-packages/arviz_plots/plots/lm_plot.py:360: UserWarning: When multiple credible intervals are plotted, it is recommended to map 'alpha' aesthetic to 'prob' dimension to differentiate between intervals.
      warnings.warn(


    series 41: last 5 training periods are stock-outs, lambda = 5.12, latent demand lost in the run = 33 units


<figure class="figure">
<p><img src="availability_tsb_files/figure-html/cell-20-output-3.png" class="figure-img" width="1211" height="611" /></p>
</figure>


The last observations before the forecast origin are all zeros, and yet the forecast does **not** open at zero: it opens right at the series' demand level, and the observed test sales (black) immediately confirm it. The model can do this because the availability gate froze the demand-probability estimate through the shaded run, so at the origin \\\hat{p}\\ still remembers what demand looked like the last time the product was actually on the shelf. The green availability line makes the mechanism visible: the forecast bands pinch toward zero exactly where availability drops, and nowhere else. We return to this series at the end of the notebook to show what plain TSB would have done in its place.


## Component forecasts

As in the sibling notebooks, we sample the two component predictives directly with `Predictive`, requesting the `"z_forecast"` and `"p_forecast"` deterministic sites, and plot them side by side with a single faceted `plot_lm` call. The posterior draws come from [`draw_posterior`](https://juanitorduz.github.io/numpyro_forecast/reference/functional.posterior.draw_posterior.html) (they are reused for the scenario forecasts below), chunked and moved to host for the same memory reasons as the tree export. The demand-size component predicts the size of the next demand; the demand-probability component predicts the chance an *on-shelf* period sees demand, which is precisely what makes multiplying by a chosen future availability meaningful.


    In [20]:


``` python
rng_key, rng_subkey = random.split(rng_key)
post = draw_posterior(rng_subkey, fit, num_samples=1_000, batch_size=250, device="host")

rng_key, rng_subkey = random.split(rng_key)
predictive = Predictive(
    model,
    posterior_samples={k: v[:500] for k, v in post.items()},
    return_sites=["z_forecast", "p_forecast"],
)
component_draws = predictive(rng_subkey, covariates_full, train_data)

i = example_series[0]
components = np.concatenate(
    [
        np.asarray(component_draws["z_forecast"])[:, :, [i]],
        np.asarray(component_draws["p_forecast"])[:, :, [i]],
    ],
    axis=-1,
)

idata_components = predictions_to_datatree(
    components, t_test.astype(float), ["demand size", "demand probability"]
)
pc = az.plot_lm(
    idata_components,
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
    figure_kwargs={"figsize": (12, 5), "sharex": True},
)
axes = pc.viz["plot"]["t"]
axes.sel(series="demand size").item().set(
    title="Demand size forecast", xlabel="time", ylabel="demand size"
)
axes.sel(series="demand probability").item().set(
    title="Demand probability forecast", xlabel="time", ylabel="demand probability"
)
bands = pc.viz["ci_band"]["t"]
band_94 = bands.sel(series="demand size", prob=0.94).item()
band_50 = bands.sel(series="demand size", prob=0.5).item()
band_94.set_label(hdi_label(0.94))
band_50.set_label(hdi_label(0.5))
pe_line = pc.viz["pe_line"]["t"].sel(series="demand size").item()
pe_line.set_label("posterior mean")
axes.sel(series="demand size").item().legend(handles=[band_94, band_50, pe_line], loc="upper left")
fig = pc.viz["figure"].item()
fig.suptitle(f"Component forecasts (series {i})", fontsize=16, fontweight="bold", y=1.05);
```


    /Users/juanitorduz/Documents/numpyro_forecast/.venv/lib/python3.14/site-packages/arviz_plots/plots/lm_plot.py:360: UserWarning: When multiple credible intervals are plotted, it is recommended to map 'alpha' aesthetic to 'prob' dimension to differentiate between intervals.
      warnings.warn(


<figure class="figure">
<p><img src="availability_tsb_files/figure-html/cell-21-output-2.png" class="figure-img" width="1211" height="540" /></p>
</figure>


## Scenario planning: full availability

Here is the payoff of modeling availability as an input. The tree above answers "what will we *sell* given the availability we actually had"; replenishment planning needs "what would we sell **if the product were always on the shelf**". With the functional API that is one more [`forecast`](https://juanitorduz.github.io/numpyro_forecast/reference/functional.prediction.forecast.html) call on the *same* posterior draws, feeding covariates whose future availability rows are all ones, and one [`add_forecast_groups`](https://juanitorduz.github.io/numpyro_forecast/reference/convert.add_forecast_groups.html) call to package the draws as the `predictions` group of a sibling tree (the groups it copies from `tree` are shared, so this costs no memory beyond the new forecast).


    In [21]:


``` python
covariates_full_ones = jnp.stack(
    [
        sales_input_full,
        jnp.concatenate([available_train, jnp.ones_like(available_test)], axis=0),
    ],
    axis=0,
)

rng_key, rng_subkey = random.split(rng_key)
fc_full = forecast(
    rng_subkey, model, post, train_data, covariates_full_ones, batch_size=250, device="host"
)
tree_full = add_forecast_groups(
    tree, np.asarray(fc_full), covariates_full_ones[:, t_max_train:, :]
)
forecast_full = stacked_draws(tree_full["predictions"], "obs")
print(f"full-availability forecast draws: {forecast_full.shape}")
```


    full-availability forecast draws: (1000, 10, 1000)


We compare the two scenarios where they differ most visibly: the total demand across the whole panel, period by period. Under realized availability the forecast tracks the observed total *sales*; under full availability it recovers the total latent *demand*, whose ground truth (the sum of the \\\lambda_i\\, dashed line) the model has never seen. Roughly \\40\\\\ of demand is invisible in any single period's sales, and the availability-aware model reconstructs it.


    In [22]:


``` python
totals = np.stack(
    [forecast_pp.sum(axis=-1), forecast_full.sum(axis=-1)], axis=-1
)  # (sample, time, scenario)
expected_total = float(lam.sum())
scenario_labels = ["realized availability", "full availability"]

idata_totals = predictions_to_datatree(totals, t_test.astype(float), scenario_labels)
pc = az.plot_lm(
    idata_totals,
    y="obs",
    x="t",
    plot_dim="time",
    ci_kind="hdi",
    ci_prob=hdi_probs,
    smooth=False,
    point_estimate="mean",
    visuals={
        "ci_band": {"color": "C1"},
        "observed_scatter": False,
        "pe_line": {"color": "C1", "alpha": 1.0, "width": 1.5},
    },
    aes={"alpha": ["prob"]},
    alpha=hdi_alphas,
    figure_kwargs={"figsize": (12, 5), "sharex": True, "sharey": True},
)
axes = pc.viz["plot"]["t"]
ax_realized = axes.sel(series="realized availability").item()
ax_full = axes.sel(series="full availability").item()
(sales_line,) = ax_realized.plot(
    t_test,
    np.asarray(test_data.sum(axis=-1)),
    "o-",
    color="black",
    lw=1,
    ms=4,
    label="observed total sales",
)
(demand_line,) = ax_full.plot(
    t_test,
    np.asarray(panel.demand[t_max_train:].sum(axis=-1)),
    "o-",
    color="C0",
    lw=1,
    ms=4,
    label="latent total demand",
)
truth_line = ax_full.axhline(
    expected_total, color="black", ls="--", lw=1, label="true expected demand"
)
ax_realized.set(title="Realized availability", xlabel="time", ylabel="panel total")
ax_full.set(title="Full availability", xlabel="time", ylabel="")
bands = pc.viz["ci_band"]["t"]
band_94 = bands.sel(series="full availability", prob=0.94).item()
band_50 = bands.sel(series="full availability", prob=0.5).item()
band_94.set_label(hdi_label(0.94))
band_50.set_label(hdi_label(0.5))
ax_realized.legend(handles=[sales_line], loc="lower left")
ax_full.legend(handles=[band_94, band_50, demand_line, truth_line], loc="lower left")
fig = pc.viz["figure"].item()
fig.suptitle(
    "Panel-total forecasts under two availability scenarios",
    fontsize=16,
    fontweight="bold",
    y=1.05,
);
```


    /Users/juanitorduz/Documents/numpyro_forecast/.venv/lib/python3.14/site-packages/arviz_plots/plots/lm_plot.py:360: UserWarning: When multiple credible intervals are plotted, it is recommended to map 'alpha' aesthetic to 'prob' dimension to differentiate between intervals.
      warnings.warn(


<figure class="figure">
<p><img src="availability_tsb_files/figure-html/cell-23-output-2.png" class="figure-img" width="1211" height="540" /></p>
</figure>


    In [23]:


``` python
mean_total_realized = float(totals[..., 0].mean())
mean_total_full = float(totals[..., 1].mean())
print(f"true expected demand per period (sum of lambdas):  {expected_total:,.1f}")
print(
    f"expected sales per period at 60% availability:     {availability_rate * expected_total:,.1f}"
)
print(
    f"forecast total per period, realized availability:  {mean_total_realized:,.1f} "
    f"({mean_total_realized / expected_total:.0%} of expected demand)"
)
print(
    f"forecast total per period, full availability:      {mean_total_full:,.1f} "
    f"({mean_total_full / expected_total:.0%} of expected demand)"
)
```


    true expected demand per period (sum of lambdas):  2,471.8
    expected sales per period at 60% availability:     1,483.1
    forecast total per period, realized availability:  1,485.9 (60% of expected demand)
    forecast total per period, full availability:      2,476.1 (100% of expected demand)


How to read this figure:

- **Left panel (realized availability).** The forecast answers "how much will we *sell* under the stock-out pattern the test window actually had". The panel total wiggles period by period because a different random \\40\\\\ of the assortment is off the shelf each period, and the band tracks the observed total sales (black), which is the quantity this scenario should predict. Note that the model gets the *level* right without ever being told the availability rate: it learned each series' on-shelf demand and the covariates supply who is on the shelf when.
- **Right panel (full availability).** Same posterior, one covariate change: every product on the shelf over the whole horizon. The forecast jumps to the level of the *latent demand* (blue), the sales that would materialize with nothing censored, and its mean sits essentially on the true expected demand \\\sum_i \lambda_i\\ (dashed), a ground-truth quantity the model has never observed. This is the number a replenishment plan actually needs, and no amount of post-processing of the left panel produces it: scaling censored forecasts up by a global availability rate would miss which series were censored and by how much.
- **The gap between the panels** is the roughly \\40\\\\ of demand that stock-outs make invisible in any single period's sales. The printed totals below the figure quantify it: the realized-availability total sits near the expected *sales* level, while the full-availability total recovers the expected *demand* within a few percent.
- **Band widths.** The bands are much narrower, relative to the mean, than in the single-series forecasts above: summing \\1{,}000\\ series averages away the independent per-series noise, so what remains is mostly the (small, well-pooled) parameter uncertainty plus the availability pattern itself.


# Comparison with plain TSB

Since plain TSB is the special case \\a_t \equiv 1\\, comparing against it requires no second model: we refit the *same* model with an all-ones availability input, so its demand-probability component decays on every zero, stock-out or not. The synthetic setup then lets us do something a real dataset never allows: score both fits against the **known truth**. Each series' true on-shelf demand probability is \\1 - e^{-\lambda_i}\\, and we compare it with each fit's posterior-mean probability path, time-averaged over the second half of the training window to wash out the \\\hat{p}\_0 = 0.5\\ transient.


    In [24]:


``` python
covariates_train_ones = jnp.stack([train_data, jnp.ones_like(train_data)], axis=0)

rng_key, rng_subkey = random.split(rng_key)
fit_plain = fit_svi(
    rng_subkey,
    model,
    train_data,
    covariates_train_ones,
    optim=0.001,
    num_steps=10_000,
)
print(f"mean ELBO loss over the last 100 steps: {float(jnp.mean(fit_plain.losses[-100:])):,.0f}")

rng_key, rng_subkey = random.split(rng_key)
post_plain = draw_posterior(
    rng_subkey, fit_plain, num_samples=1_000, batch_size=250, device="host"
)
```


    mean ELBO loss over the last 100 steps: 87,131


    In [25]:


``` python
p_true = 1 - np.exp(-lam)
p_hat_aware = np.asarray(post["prob"])[:, t_max_train // 2 :, :].mean(axis=(0, 1))
p_hat_plain = np.asarray(post_plain["prob"])[:, t_max_train // 2 :, :].mean(axis=(0, 1))

ratio_aware = float(p_hat_aware.mean() / p_true.mean())
ratio_plain = float(p_hat_plain.mean() / p_true.mean())
print(f"mean estimated / true demand probability, availability-aware: {ratio_aware:.2f}")
print(f"mean estimated / true demand probability, plain TSB:          {ratio_plain:.2f}")

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(p_true, p_hat_aware, s=8, alpha=0.3, color="C0", label="availability-aware TSB")
ax.scatter(p_true, p_hat_plain, s=8, alpha=0.3, color="C1", label="plain TSB")
grid = np.linspace(0.0, 1.0, 100)
ax.plot(grid, grid, color="black", ls="--", lw=1, label="identity")
ax.plot(
    grid,
    availability_rate * grid,
    color="gray",
    ls=":",
    lw=2,
    label=r"$0.6 \times$ identity (availability rate)",
)
ax.legend(loc="upper left")
ax.set(
    title="Recovering the true demand probability",
    xlabel=r"true $P(\text{demand} \mid \text{available}) = 1 - e^{-\lambda}$",
    ylabel="posterior-mean probability",
);
```


    mean estimated / true demand probability, availability-aware: 0.99
    mean estimated / true demand probability, plain TSB:          0.60


<figure class="figure">
<p><img src="availability_tsb_files/figure-html/cell-26-output-2.png" class="figure-img" width="811" height="711" /></p>
</figure>


The scatter is the whole argument in one picture: the availability-aware estimates line up with the identity, while the plain TSB estimates line up with the \\0.6 \times\\ line, exactly the \\P(\text{available}) \cdot P(\text{demand} \mid \text{available})\\ bias predicted in the comparison section. On observed *sales* under the historical availability regime that bias partly cancels (a censored probability times an uncensored future is roughly right on average), but the moment we ask the scenario question, plain TSB has no way to answer: its forecast of unconstrained demand inherits the bias in full.

One subtlety in setting the comparison up honestly: [forecast](../../../reference/functional.prediction.forecast.md#numpyro_forecast.functional.prediction.forecast) reruns the model's recursions from the covariates it is given, so plain TSB's covariates must carry the all-ones availability input over the *whole* horizon, training window included. Feeding it the gated history would smuggle the availability information back into a method that, by definition, never sees it.


    In [26]:


``` python
covariates_plain_full = jnp.stack([sales_input_full, jnp.ones_like(panel.available)], axis=0)

rng_key, rng_subkey = random.split(rng_key)
fc_full_plain = forecast(
    rng_subkey, model, post_plain, train_data, covariates_plain_full, batch_size=250, device="host"
)

total_aware = float(forecast_full.sum(axis=-1).mean())
total_plain = float(np.asarray(fc_full_plain).sum(axis=-1).mean())
print(f"true expected demand per period (sum of lambdas):     {expected_total:,.1f}")
print(
    f"full-availability forecast, availability-aware TSB:   {total_aware:,.1f} "
    f"({total_aware / expected_total:.0%} of truth)"
)
print(
    f"full-availability forecast, plain TSB:                {total_plain:,.1f} "
    f"({total_plain / expected_total:.0%} of truth)"
)
```


    true expected demand per period (sum of lambdas):     2,471.8
    full-availability forecast, availability-aware TSB:   2,476.1 (100% of truth)
    full-availability forecast, plain TSB:                1,480.6 (60% of truth)


Finally, we return to the hero series from the forecast section, the fast mover whose training window ends in a stock-out run, and ask both fits the same scenario question: how much demand would there be with the product always on the shelf? For plain TSB the trailing run is a double blow. Its demand-probability estimate is biased low on *every* series (the \\0.6 \times\\ line above), and on this series it decayed further through each zero of the trailing run right before the forecast origin. The availability-aware fit froze through the same run.


    In [27]:


``` python
hero_draws = np.concatenate(
    [forecast_full[:, :, [k]], np.asarray(fc_full_plain)[:, :, [k]]], axis=-1
)
hero_labels = ["availability-aware TSB", "plain TSB"]
idata_hero = predictions_to_datatree(hero_draws, t_test.astype(float), hero_labels)
pc = az.plot_lm(
    idata_hero,
    y="obs",
    x="t",
    plot_dim="time",
    ci_kind="hdi",
    ci_prob=hdi_probs,
    smooth=False,
    point_estimate="mean",
    visuals={
        "ci_band": {"color": "C1"},
        "observed_scatter": False,
        "pe_line": {"color": "C1", "alpha": 1.0, "width": 1.5},
    },
    aes={"alpha": ["prob"]},
    alpha=hdi_alphas,
    figure_kwargs={"figsize": (12, 5), "sharex": True, "sharey": True},
)
axes = pc.viz["plot"]["t"]
for label in hero_labels:
    ax = axes.sel(series=label).item()
    (demand_line,) = ax.plot(
        t_test,
        np.asarray(panel.demand[t_max_train:, k]),
        "o-",
        color="black",
        lw=1,
        ms=4,
        label="latent demand (truth)",
    )
    lam_line = ax.axhline(lam[k], color="black", ls="--", lw=1, label=r"expected demand $\lambda$")
    ax.set(title=label, xlabel="time", ylabel="demand" if label == hero_labels[0] else "")
bands = pc.viz["ci_band"]["t"]
band_94 = bands.sel(series=hero_labels[0], prob=0.94).item()
band_50 = bands.sel(series=hero_labels[0], prob=0.5).item()
band_94.set_label(hdi_label(0.94, prefix="forecast "))
band_50.set_label(hdi_label(0.5, prefix="forecast "))
pe_line = pc.viz["pe_line"]["t"].sel(series=hero_labels[0]).item()
pe_line.set_label("forecast mean")
fig = pc.viz["figure"].item()
fig.legend(
    handles=[band_94, band_50, pe_line, demand_line, lam_line],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=3,
)
fig.suptitle(
    f"Full-availability forecast for series {k} (training ends in {run_k} stock-out periods)",
    fontsize=16,
    fontweight="bold",
    y=1.05,
);
```


    /Users/juanitorduz/Documents/numpyro_forecast/.venv/lib/python3.14/site-packages/arviz_plots/plots/lm_plot.py:360: UserWarning: When multiple credible intervals are plotted, it is recommended to map 'alpha' aesthetic to 'prob' dimension to differentiate between intervals.
      warnings.warn(


<figure class="figure">
<p><img src="availability_tsb_files/figure-html/cell-28-output-2.png" class="figure-img" width="1211" height="595" /></p>
</figure>


The two facets share the y axis, so the level gap *is* the story: the availability-aware forecast opens at the series' true demand level (dashed line) while the plain TSB forecast opens well below it, still paying for zeros that were never about demand. Multiply this picture by every series and every stock-out run in the panel and you get the aggregate shortfall printed above.

Asked how much the panel would sell with everything on the shelf, the availability-aware model lands close to the true expected demand while plain TSB misses low by roughly the availability rate: at scale, that is the difference between stocking for demand and stocking for last year's stock-outs.

As for Croston, the comparison stays conceptual: its occurrence bookkeeping lives on the event axis (inter-demand intervals), where a stock-out is indistinguishable from slow demand because it simply stretches the interval in progress. There is no per-period update to gate, which is why the availability hack needs TSB's calendar-axis demand-probability component as its starting point.


# A final note: what the availability mask buys you

It is worth collecting what the one-line change delivered, because each piece showed up in a different section:

- **Unbiased demand estimates.** The demand-probability component recovers \\P(\text{demand} \mid \text{available})\\ instead of the censored product, as the recovery scatter shows against ground truth.
- **Scenario forecasts.** Availability enters as an input (the trailing rows of the availability covariate), so the same posterior answers "what will we sell under the planned availability" and "what would demand be with everything on the shelf", the number replenishment actually needs. Plain TSB can only extrapolate the censored history.
- **No stock-out death spiral.** A forecast that decays with every stock-out under-forecasts, which under-stocks, which causes more stock-out zeros: a feedback loop the frozen update never enters.
- **Nearly free.** One extra input series and one gated update; plain TSB is recovered exactly at \\a_t \equiv 1\\, so nothing is lost where availability data does not exist.

The same caveats as in the sibling notebooks apply to the likelihood choices: Gaussian likelihoods for a count size and a \\0/1\\ indicator are the blog post's pragmatic simplification, and \\\text{Bernoulli}\\ occurrence or truncated size likelihoods are the natural refinements. The hack also treats availability as *exogenous*; when stock-outs correlate with demand (best-sellers sell out), the censoring is informative and the frozen update, while far better than the decaying one, is no longer the full story.


# References

- Orduz, J. [*Hacking the TSB Model for Intermittent Time Series to Accommodate for Availability Constraints*](https://juanitorduz.github.io/availability_tsb/). The blog post this notebook ports.
- The [TSB example](https://juanitorduz.github.io/numpyro_forecast/examples/tsb.html) in this documentation, whose two-component level-model construction this notebook promotes to a panel, and the blog post it ports: Orduz, J. [*TSB Method for Intermittent Time Series Forecasting in NumPyro*](https://juanitorduz.github.io/tsb_numpyro/).
- The [Croston example](https://juanitorduz.github.io/numpyro_forecast/examples/croston.html) in this documentation, the first notebook of the intermittent-demand trilogy.
- Teunter, R. H., Syntetos, A. A., & Babai, M. Z. (2011). *Intermittent demand: Linking forecasting to inventory obsolescence*. European Journal of Operational Research, 214(3), 606-615. The paper that introduces the TSB method.
- Croston, J. D. (1972). *Forecasting and stock control for intermittent demands*. Operational Research Quarterly, 23(3), 289-303.
- statsforecast documentation: [`TSB`](https://nixtlaverse.nixtla.io/statsforecast/docs/models/tsb.html), the classical TSB baseline.
