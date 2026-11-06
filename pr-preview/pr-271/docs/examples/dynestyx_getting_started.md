# Getting Started with `dynestyx` State Space Models


Getting Started with `dynestyx` State Space Models

*This example fits the BART weekly ridership model of the [comparing inference methods](inference_methods_comparison.md) example and the ARMA model of the [ARMA](arma.md) example as [`dynestyx`](https://github.com/BasisResearch/dynestyx) state space models. It is the entry point to `dynestyx` in `numpyro_forecast`. The [advanced example](dynestyx_integration.md) continues with filtering versus smoothing, the cost of a gradient, variational inference and backtesting.*

The [`dynestyx` repository](https://github.com/BasisResearch/dynestyx) describes the library as follows:

> `dynestyx` is a library designed for Bayesian modeling and inference of dynamical systems. It is an extension of [NumPyro](https://num.pyro.ai/en/stable/), and incorporates a wide variety of state-of-the-art inference methods for state space models.

This notebook uses these components of `dynestyx`:

- **`DynamicalModel`**: holds the initial condition, the state evolution and the observation model of a state space model.
- **`dynestyx.sample`**: adds a `DynamicalModel` to a NumPyro model. The effect handlers around the call decide what it does with the model.
- **`Filter` and `Smoother`**: handlers that integrate the latent path out and add the marginal log likelihood to the NumPyro trace.
- **`LatentPathBuilder`**: a handler that samples the latent path.
- **`Simulator`**: a handler that rolls the state forward in time.

The [`state_space`](../../reference/contrib.dynestyx.state_space.md) building block of `numpyro_forecast.contrib.dynestyx` (extra `dynestyx`) connects a `DynamicalModel` to the drivers of the package: [`forecast`](../../reference/predictive.forecast.md), [`predict_in_sample`](../../reference/predictive.predict_in_sample.md), [`to_datatree`](../../reference/convert.to_datatree.md) and [`backtest`](../../reference/evaluate.backtest.md).

We proceed as follows:

1.  **How [state_space](../../reference/contrib.dynestyx.state_space.md#numpyro_forecast.contrib.dynestyx.state_space) works.** The three modes of the block and where the `dynestyx` handler goes.
2.  **Local level with a Kalman filter.** The BART model with Normal noise: the Kalman filter integrates the level out, and we compare the fit with the [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) model of the package.
3.  **In-sample predictive and forecast.** We export the Kalman fit with [to_datatree](../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) under a Kalman smoother, without a refit.
4.  **Student-T noise with an explicit path.** No Kalman filter is exact here, so `dynestyx` samples the level.
5.  **Missing data.** We fit the explicit-path model through a gap of 20 weeks.
6.  **A two-dimensional state.** We write ARMA(1,1) as a state space model.


# Prepare Notebook


``` python
import numpyro

numpyro.set_host_device_count(n=4)  # before any import that initializes jax

import warnings
from time import perf_counter
from typing import Literal

import arviz as az
import dynestyx as dsx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import polars as pl
import xarray as xr
from dynestyx import DiscreteTimeSimulator, DynamicalModel, Filter, LatentPathBuilder, Smoother
from dynestyx.inference.configs.smoother import KFSmootherConfig
from dynestyx.inference.filters import KFConfig
from dynestyx.models import LinearGaussianObservation, LinearGaussianStateEvolution
from jax import random
from jaxtyping import Float
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast import (
    Horizon,
    eval_crps,
    forecast,
    innovations,
    predict,
    predict_in_sample,
    predictions_to_datatree,
    ssoe,
    to_datatree,
)
from numpyro_forecast.contrib.dynestyx import state_space
from numpyro_forecast.datasets import load_bart_weekly
from numpyro_forecast.features import fourier_features
from numpyro_forecast.typing import Array, ForecastModel

# arviz-plots warns whenever `aes` maps `alpha`, even to `["prob"]` as the band plots below do
warnings.filterwarnings(
    "ignore", message="When multiple credible intervals are plotted", category=UserWarning
)

az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"

rng_key = random.PRNGKey(seed=42)

print(f"dynestyx {dsx.__version__}, numpyro {numpyro.__version__}, jax {jax.__version__}")

%load_ext autoreload
%autoreload 2
%load_ext jaxtyping
%jaxtyping.typechecker beartype.beartype
%config InlineBackend.figure_format = "retina"
```


    dynestyx 0.8.0, numpyro 0.22.0, jax 0.11.2


# How [state_space](../../reference/contrib.dynestyx.state_space.md#numpyro_forecast.contrib.dynestyx.state_space) Works

`state_space(h, name, y, dynamics, controls=...)` goes inside a model function `(covariates, data=None)`, like the other building blocks of the package. It turns the [`Horizon`](../../reference/models.Horizon.md) `h` into the time grids of `dynestyx.sample`, calls it once, and registers the sites the drivers read. `h` decides the mode of the call:

- **Training** (`data` given, no horizon): the handler adds its likelihood to the trace, which is what NUTS or SVI fits.
- **Forecasting** (`covariates` longer than `data`): a `Simulator` rolls the conditioned state at the last observation forward, and the block registers the draws as the `"forecast"` site.
- **In-sample predictive** (`data=None`): the block draws the in-sample states, from the smoothing distribution under a `Smoother` or the sampled path under a `LatentPathBuilder`. It draws one observation per step from the observation model and registers them as the `"obs"` site. This is not a prior predictive, because the block still conditions on the observed window.

The block reads the observed window `y` from the covariates, as [`ssoe`](../../reference/models.ssoe.md) does, because the drivers call the model without data for the in-sample predictive.

The `dynestyx` handler goes around each call that runs the model, as in the `dynestyx` examples, so the same model function serves every inference strategy:

``` python
with Filter(filter_config=KFConfig()):
    mcmc.run(key_fit, covariates_train, data_train)

# the parameters only, without the filtered states that the filter fit recorded
posterior = {k: v for k, v in mcmc.get_samples().items() if not k.startswith("f_")}
with DiscreteTimeSimulator(n_simulations=1), Smoother(smoother_config=KFSmootherConfig()):
    tree = to_datatree(key_export, model, posterior, data_train, covariates)
```

The block raises an error when no `Filter`, `Smoother` or `LatentPathBuilder` is around the call. Without one, `dynestyx` would pick an inference method itself and only log the choice.


# Local Level with a Kalman Filter

The [comparing inference methods](inference_methods_comparison.md) example fits a random walk level plus a Fourier regression to the weekly BART ridership. In this section the noise is Normal:

 \begin{align\*} \ell_t &= \ell\_{t-1} + w_t, \qquad w_t \sim \text{Normal}(0, s), \\ y_t &= b + \ell_t + \beta^\top u_t + v_t, \qquad v_t \sim \text{Normal}(0, \sigma), \end{align\*} 

where y_t is the log ridership of week t, u_t holds the 52 Fourier terms of the yearly cycle, and s is the `drift_scale`. The priors are those of the comparing inference methods example. We hold out the last 52 weeks.

The covariate array carries the series in column 0 and the Fourier terms in columns `1:`. The `dynestyx` model reads its observed window from column 0, and the `numpyro_forecast` model ignores it.


``` python
bart = load_bart_weekly()
duration = bart.shape[0]
t_obs = duration - 52
bart_train, bart_test = bart[:t_obs], bart[t_obs:]
week = np.arange(duration)

fourier = fourier_features(duration, period=365.25 / 7, num_terms=26)
bart_covariates = jnp.concatenate([bart, fourier], axis=-1)  # (duration, 53)

print(f"covariates {bart_covariates.shape}, train {bart_train.shape}, test {bart_test.shape}")
```


    covariates (469, 53), train (417, 1), test (52, 1)


The model is linear and Gaussian, so the Kalman filter integrates the level out exactly, and NUTS explores the 55 parameters only. In `dynestyx` terms, the regression is the control matrix `D` of the observation model and the intercept is its `bias`. The Fourier terms enter as controls over the whole horizon, so the forecast reads their future rows from the covariates.


``` python
def gaussian_ssm(covariates: Array, data: Array | None = None) -> None:
    """Random walk level plus Fourier regression with Normal noise, as a dynestyx model."""
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :1]  # the observed window, (t_obs, 1)
    controls = covariates[..., 1:]  # the Fourier terms over the whole horizon, (duration, 52)
    bias = numpyro.sample("bias", dist.Normal(0.0, 10.0))
    weight = numpyro.sample(
        "weight", dist.Normal(0.0, 0.1).expand([controls.shape[-1]]).to_event(1)
    )
    drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-20.0, 5.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-5.0, 5.0))
    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), drift_scale**2 * jnp.eye(1)),
        state_evolution=LinearGaussianStateEvolution(
            A=jnp.eye(1), cov=drift_scale**2 * jnp.eye(1)
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.eye(1), R=sigma**2 * jnp.eye(1), D=weight[None, :], bias=bias[None]
        ),
        control_dim=controls.shape[-1],
    )
    state_space(h, "f", y, dynamics, controls=controls)
```


Next, let's render the model with `numpyro.render_model`. The render traces the model, so the block needs a `dynestyx` handler around the call. We use the Kalman filter of the fit below, on the `cd_dynamax` backend. We pass `KFConfig` explicitly: without a configuration, `Filter()` uses an ensemble Kalman filter, which is approximate.


``` python
kalman = Filter(filter_config=KFConfig(filter_source="cd_dynamax"))

with kalman:
    graph = numpyro.render_model(
        gaussian_ssm,
        model_args=(bart_covariates[:t_obs], bart_train),
        render_distributions=True,
    )

graph
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-5-output-1.svg" class="img-fluid figure-img" /></p>
</figure>


The graph has no `time` plate: the Kalman filter integrates the level out, so NUTS samples the four parameter sites only. The gray node `f_marginal_log_likelihood` is the factor that the filter adds to the trace. The dashed nodes are deterministic sites that `dynestyx` records, such as the filtered state means `f_filtered_states_mean`.

As a reference we fit the same model with the building blocks of the package: [`innovations`](../../reference/models.innovations.md) samples the 417 weekly steps of the level, and [`predict`](../../reference/models.predict.md) adds the likelihood. The comparing inference methods example also samples a `centered` site for its `LocScaleReparam`. That site only selects a parameterization, so its posterior equals its prior and NUTS mixes poorly on it. Here we fix the non-centered form instead.


``` python
def gaussian_model(covariates: Array, data: Array | None = None) -> None:
    """Random walk level plus Fourier regression with Normal noise, with ``innovations``."""
    h = Horizon.from_data(covariates, data)
    fourier = covariates[..., 1:]  # column 0 is the series, read only by the dynestyx model
    bias = numpyro.sample("bias", dist.Normal(0.0, 10.0))
    weight = numpyro.sample(
        "weight", dist.Normal(0.0, 0.1).expand([fourier.shape[-1]]).to_event(1)
    )
    drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-20.0, 5.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-5.0, 5.0))
    drift = innovations(
        h, "drift", dist.Normal(0.0, drift_scale), reparam=LocScaleReparam(centered=0)
    )
    level = jnp.cumsum(drift, axis=-2)
    regression = (weight * fourier).sum(axis=-1, keepdims=True)
    predict(h, dist.Normal(0.0, sigma), level + bias + regression)
```


This model has no `dynestyx` site, so we render it without a handler:


``` python
numpyro.render_model(
    gaussian_model,
    model_args=(bart_covariates[:t_obs], bart_train),
    render_distributions=True,
)
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-7-output-1.svg" class="img-fluid figure-img" /></p>
</figure>


Here the level is explicit. The `time` plate holds the non-centered weekly steps `drift_decentered` and the steps `drift`, which enter the likelihood of the observed `obs` site.

We fit both models with NUTS: 4 chains, a target acceptance probability of 0.9 and parallel chains. The fits under a `LatentPathBuilder` below are the exception: they run their chains one after the other. Every fit in this notebook goes through `fit_nuts`, which also returns the wall time of the run, warmup and compilation included.

For every fit we check the convergence over every sample site, the vectors and the latent path included. `diagnostics` reports the number of scalar parameters NUTS explores, the smallest bulk effective sample size (ESS), the largest \hat{R}, the number of divergent transitions, the leapfrog steps per draw and the wall time. `fits_table` collects the diagnostics of several fits into one table. `posterior_tree` collects the draws of a list of sites into an ArviZ tree, which is where the diagnostics and the posterior plots below start.


``` python
def fit_nuts(
    rng_key: Array,
    model: ForecastModel,
    covariates: Array,
    data: Array,
    *,
    num_warmup: int,
    chain_method: Literal["parallel", "sequential", "vectorized"],
    num_samples: int = 1_000,
) -> tuple[MCMC, float]:
    """Fit ``model`` with NUTS (4 chains) and return the sampler and the wall time in seconds."""
    mcmc = MCMC(
        NUTS(model, target_accept_prob=0.9),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=4,
        chain_method=chain_method,
        progress_bar=False,
    )
    start = perf_counter()
    mcmc.run(rng_key, covariates, data, extra_fields=("num_steps", "diverging"))
    jax.block_until_ready(mcmc.get_samples())
    return mcmc, perf_counter() - start


def posterior_tree(mcmc: MCMC, names: list[str]) -> xr.DataTree:
    """Return the ``(chain, draw)`` posterior of the sites ``names`` as a DataTree."""
    grouped = mcmc.get_samples(group_by_chain=True)
    return az.from_dict({"posterior": {name: np.asarray(grouped[name]) for name in names}})


def diagnostics(mcmc: MCMC, seconds: float) -> dict[str, int | float]:
    """Convergence over every sample site of the fit, and its sampler cost."""
    sites = sorted(mcmc.last_state.z)  # the sample sites NUTS explores, deterministic ones out
    tree = posterior_tree(mcmc, sites)
    # one value per scalar parameter; min and max propagate a NaN
    ess = np.concatenate(
        [v.to_numpy().ravel() for v in az.ess(tree, method="bulk").dataset.data_vars.values()]
    )
    r_hat = np.concatenate(
        [v.to_numpy().ravel() for v in az.rhat(tree).dataset.data_vars.values()]
    )
    extra = mcmc.get_extra_fields()
    return {
        "parameters": ess.size,
        "min bulk ESS": round(float(ess.min())),  # a NaN raises
        "max R-hat": round(float(r_hat.max()), 3),
        "divergences": int(np.asarray(extra["diverging"]).sum()),
        "leapfrog steps per draw": round(float(np.asarray(extra["num_steps"]).mean()), 1),
        "wall time (s)": round(seconds, 1),
    }


def fits_table(fits: dict[str, dict[str, int | float]]) -> pl.DataFrame:
    """One row of ``diagnostics`` per fit."""
    return pl.DataFrame([{"fit": label} | row for label, row in fits.items()])
```


The `dynestyx` fit runs under the Kalman filter `kalman` of the graph above.


``` python
rng_key, key_kalman, key_innovations = random.split(rng_key, 3)

with kalman:
    mcmc_kalman, seconds = fit_nuts(
        key_kalman,
        gaussian_ssm,
        bart_covariates[:t_obs],
        bart_train,
        num_warmup=2_000,
        chain_method="parallel",
    )

gaussian_fits = {"dynestyx Kalman filter": diagnostics(mcmc_kalman, seconds)}

mcmc_innovations, seconds = fit_nuts(
    key_innovations,
    gaussian_model,
    bart_covariates[:t_obs],
    bart_train,
    num_warmup=2_000,
    chain_method="parallel",
)
gaussian_fits["innovations"] = diagnostics(mcmc_innovations, seconds)

fits_table(gaussian_fits)
```


shape: (2, 7)

| fit | parameters | min bulk ESS | max R-hat | divergences | leapfrog steps per draw | wall time (s) |
|----|----|----|----|----|----|----|
| str | i64 | i64 | f64 | i64 | f64 | f64 |
| "dynestyx Kalman filter" | 55 | 4245 | 1.004 | 0 | 11.5 | 72.5 |
| "innovations" | 472 | 1684 | 1.008 | 0 | 127.0 | 11.9 |


Both fits converge over all their parameters:

- The largest \hat{R} is 1.008 and the smallest bulk ESS is 1{,}684. There are no divergent transitions.
- NUTS explores 55 parameters with the Kalman filter and 472 with [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations), the 417 weekly steps of the level included.
- The filter takes 11.5 leapfrog steps per draw, against 127.0 for [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations). Each of those steps runs the filter as a sequential scan over the 417 weeks. That costs much more than the cumulative sum of [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations), so on the CPU that executed this notebook the Kalman fit took longer in wall time.

We overlay the posteriors of the three scalar parameters that both models share, one parameter per panel.


``` python
def plot_posteriors(
    fits: dict[str, MCMC], names: list[str], title: str, truth: dict[str, float] | None = None
) -> None:
    """Overlay the posterior densities of ``names`` for every fit, one panel per parameter."""
    trees = {label: posterior_tree(mcmc, names) for label, mcmc in fits.items()}
    with warnings.catch_warnings():
        # fits with different numbers of draws: arviz-plots pads the shorter one with NaN
        # (which the KDE drops), and xarray warns about the padding join
        warnings.filterwarnings("ignore", message="In a future version of xarray")
        pc = az.plot_dist(
            trees,
            var_names=names,
            kind="kde",
            visuals={
                "credible_interval": False,
                "point_estimate": False,
                "point_estimate_text": False,
            },
            figure_kwargs={"figsize": (12, 3.5)},
        )
    pc.add_legend("model")
    for name in names:
        ax = pc.get_target(name, {})
        ax.set(xlabel=name, ylabel="density")
        if truth is not None:
            ax.axvline(truth[name], color="black", ls="--", lw=1)
    pc.viz["figure"].item().suptitle(title, fontsize=14, fontweight="bold")


plot_posteriors(
    {"dynestyx Kalman filter": mcmc_kalman, "innovations": mcmc_innovations},
    ["bias", "drift_scale", "sigma"],
    title="Local level and Fourier regression posteriors, Normal noise",
)
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-10-output-1.png" class="figure-img" width="1206" height="361" /></p>
</figure>


The posteriors of the intercept, of the scale of the weekly steps and of the noise scale coincide.


# In-Sample Predictive and Forecast

[to_datatree](../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) draws the in-sample predictive by calling the model without data, and the forecast by calling it with the full-horizon covariates. For the in-sample predictive the block needs the distribution of the states given the whole window, p(x_t \mid y\_{1:T}, \theta). A `Filter` does not have it: it carries the filtering distribution p(x_t \mid y\_{1:t}, \theta) only. A Kalman smoother has the smoothing distribution, and it adds the same marginal likelihood to the trace as the Kalman filter. The draws of the filter fit are therefore draws of the posterior under the smoother too, so we export them under the smoother, with the `Simulator` outside it for the forecast, and do not refit.

The block draws every in-sample state from its smoothing distribution and pushes it through the observation model. These draws are exact for every week, so the bands and the CRPS below are correct. They are independent across weeks, while [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) replays a joint posterior path.


``` python
smoother = Smoother(smoother_config=KFSmootherConfig(filter_source="cd_dynamax"))
# the parameters only: the filter fit also recorded its filtered states (f_filtered_states_*),
# which the export under the smoother does not need
kalman_parameters = {
    name: value for name, value in mcmc_kalman.get_samples().items() if not name.startswith("f_")
}
rng_key, key_tree_kalman, key_tree = random.split(rng_key, 3)

with DiscreteTimeSimulator(n_simulations=1), smoother:
    trees = {
        "dynestyx Kalman smoother": to_datatree(
            key_tree_kalman,
            gaussian_ssm,
            kalman_parameters,
            bart_train,
            bart_covariates,
            num_chains=4,
            batch_size=500,
        )
    }

trees["innovations"] = to_datatree(
    key_tree,
    gaussian_model,
    mcmc_innovations.get_samples(),
    bart_train,
    bart_covariates,
    num_chains=4,
    batch_size=500,
)

trees["dynestyx Kalman smoother"]
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
│       Dimensions:       (chain: 4, draw: 1000, weight_dim_0: 52)
│       Coordinates:
│         * chain         (chain) int64 32B 0 1 2 3
│         * draw          (draw) int64 8kB 0 1 2 3 4 5 6 ... 993 994 995 996 997 998 999
│         * weight_dim_0  (weight_dim_0) int64 416B 0 1 2 3 4 5 6 ... 46 47 48 49 50 51
│       Data variables:
│           bias          (chain, draw) float32 16kB 14.51 14.5 14.5 ... 14.52 14.52
│           drift_scale   (chain, draw) float32 16kB 0.004764 0.008118 ... 0.008298
│           sigma         (chain, draw) float32 16kB 0.07794 0.07916 ... 0.07391 0.07532
│           weight        (chain, draw, weight_dim_0) float32 832kB -0.01708 ... -0.0102
│       Attributes:
│           created_at:                 2026-11-06T09:49:05.462838+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.3.1
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
├── Group: /posterior_predictive
│       Dimensions:  (chain: 4, draw: 1000, time: 417, obs_dim: 1)
│       Coordinates:
│         * chain    (chain) int64 32B 0 1 2 3
│         * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│         * time     (time) int64 3kB 0 1 2 3 4 5 6 7 ... 410 411 412 413 414 415 416
│         * obs_dim  (obs_dim) int64 8B 0
│       Data variables:
│           obs      (chain, draw, time, obs_dim) float32 7MB 14.36 14.41 ... 13.96
│       Attributes:
│           created_at:                 2026-11-06T09:49:08.284511+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.3.1
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
├── Group: /observed_data
│       Dimensions:  (time: 417, obs_dim: 1)
│       Coordinates:
│         * time     (time) int64 3kB 0 1 2 3 4 5 6 7 ... 410 411 412 413 414 415 416
│         * obs_dim  (obs_dim) int64 8B 0
│       Data variables:
│           obs      (time, obs_dim) float32 2kB 14.41 14.45 14.42 ... 14.71 14.65 14.04
│       Attributes:
│           created_at:                 2026-11-06T09:49:08.284769+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.3.1
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /constant_data
│       Dimensions:        (time: 417, covariate_dim: 53)
│       Coordinates:
│         * time           (time) int64 3kB 0 1 2 3 4 5 6 ... 411 412 413 414 415 416
│         * covariate_dim  (covariate_dim) int64 424B 0 1 2 3 4 5 ... 47 48 49 50 51 52
│       Data variables:
│           covariates     (time, covariate_dim) float32 88kB 14.41 0.0 ... -0.2375
│       Attributes:
│           created_at:                 2026-11-06T09:49:08.284956+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.3.1
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /predictions
│       Dimensions:  (chain: 4, draw: 1000, time: 52, obs_dim: 1)
│       Coordinates:
│         * chain    (chain) int64 32B 0 1 2 3
│         * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│         * time     (time) int64 416B 417 418 419 420 421 422 ... 464 465 466 467 468
│         * obs_dim  (obs_dim) int64 8B 0
│       Data variables:
│           obs      (chain, draw, time, obs_dim) float32 832kB 14.41 14.6 ... 14.29
│       Attributes:
│           created_at:                 2026-11-06T09:49:09.363806+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.3.1
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
└── Group: /predictions_constant_data
        Dimensions:        (time: 52, covariate_dim: 53)
        Coordinates:
          * time           (time) int64 416B 417 418 419 420 421 ... 464 465 466 467 468
          * covariate_dim  (covariate_dim) int64 424B 0 1 2 3 4 5 ... 47 48 49 50 51 52
        Data variables:
            covariates     (time, covariate_dim) float32 11kB 14.27 -0.05158 ... 0.3139
        Attributes:
            created_at:                 2026-11-06T09:49:09.364051+00:00
            creation_library:           ArviZ
            creation_library_version:   1.3.1
            creation_library_language:  Python
            sample_dims:                []
```


xarray.DataTree


/posterior(12)

Dimensions:


- chain: 4
- draw: 1000
- weight_dim_0: 52


Coordinates: (3)


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


weight_dim_0


(weight_dim_0)


int64


0 1 2 3 4 5 6 ... 46 47 48 49 50 51


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51])


Data variables: (4)


bias


(chain, draw)


float32


14.51 14.5 14.5 ... 14.52 14.52


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[14.511663 , 14.496328 , 14.501753 , ..., 14.485701 , 14.506138 ,14.490269 ],[14.522337 , 14.507599 , 14.512726 , ..., 14.513311 , 14.499488 ,14.5277605],[14.516909 , 14.519018 , 14.525178 , ..., 14.49705  , 14.518607 ,14.543873 ],[14.484605 , 14.525157 , 14.487849 , ..., 14.504019 , 14.522483 ,14.521395 ]], shape=(4, 1000), dtype=float32)


drift_scale


(chain, draw)


float32


0.004764 0.008118 ... 0.008298


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.00476403, 0.0081178 , 0.00607936, ..., 0.00597177, 0.00802004,0.00583311],[0.00572998, 0.00482184, 0.00569915, ..., 0.00686037, 0.00486369,0.00548739],[0.00684307, 0.01067492, 0.00402754, ..., 0.00746749, 0.00421162,0.0064944 ],[0.0093593 , 0.00420232, 0.00746909, ..., 0.00842299, 0.00483023,0.0082981 ]], shape=(4, 1000), dtype=float32)


sigma


(chain, draw)


float32


0.07794 0.07916 ... 0.07391 0.07532


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.07794318, 0.07915986, 0.08285472, ..., 0.07946339, 0.07861351,0.07384983],[0.0719737 , 0.07739149, 0.07493311, ..., 0.07870835, 0.07493472,0.08225714],[0.07794587, 0.07797195, 0.07565617, ..., 0.07426681, 0.0840572 ,0.07901019],[0.07419652, 0.07747736, 0.0770191 , ..., 0.07289953, 0.07390758,0.07532408]], shape=(4, 1000), dtype=float32)


weight


(chain, draw, weight_dim_0)


float32


-0.01708 0.01173 ... -0.0102


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[-1.70792006e-02,  1.17295915e-02,  1.09228669e-02, ...,2.99552293e-03, -1.57177541e-02, -1.59237646e-02],[ 5.33571607e-03,  1.88299213e-02,  1.30581241e-02, ...,-1.11647174e-02,  1.25638032e-02,  4.25685383e-03],[-9.41296760e-03,  1.96120404e-02,  1.02538234e-02, ...,-1.08037954e-02, -6.64213207e-03,  5.67571005e-05],...,[-1.74430255e-02,  1.25159798e-02,  5.52712940e-03, ...,-9.00391489e-03,  3.60206142e-03, -1.89831178e-03],[-9.30078980e-03,  1.75893828e-02,  2.10291110e-02, ...,-1.36014549e-02, -2.15209229e-03, -9.87699907e-03],[ 5.99312643e-03,  8.50297417e-03,  1.43908095e-02, ...,-1.24223512e-02,  6.61426457e-05, -8.78377445e-03]],[[ 1.30025269e-02,  1.41966557e-02,  1.88691448e-02, ...,-7.61931809e-03,  5.93922660e-03, -4.31574741e-03],[ 1.44832884e-03,  1.84091367e-02,  1.78093370e-02, ...,-1.16584767e-02,  9.72373970e-03, -1.95182627e-03],[-2.43998412e-03,  6.97969832e-03,  7.10012065e-03, ...,-9.10386723e-03,  5.75704454e-03, -9.36545152e-03],...-6.90387515e-03, -4.25489014e-03, -1.46892830e-03],[-8.00314732e-03,  1.60404369e-02,  9.65169072e-03, ...,-5.67891262e-03,  9.58627788e-04, -1.79984327e-02],[-5.95459947e-03,  1.23650879e-02,  1.31863859e-02, ...,-1.02719003e-02,  2.05270550e-03, -1.12079561e-03]],[[-3.27120558e-03,  1.06119243e-02,  1.33349448e-02, ...,-8.98781791e-03, -2.57896842e-03, -7.26590445e-03],[ 2.08539283e-03,  2.12756507e-02,  1.73892453e-02, ...,-3.61431856e-03,  7.50784879e-04, -8.74056946e-03],[-6.73347106e-03,  1.31008811e-02,  1.31547106e-02, ...,-1.32679678e-02, -4.24881745e-03, -6.21252833e-03],...,[ 1.32482182e-02,  2.02995557e-02,  1.38544301e-02, ...,-4.65570856e-03, -2.81463191e-03, -5.91926277e-03],[-1.06875645e-03,  2.00882144e-02,  1.93031821e-02, ...,3.18677933e-03, -1.24367718e-02, -1.15447100e-02],[-1.31893614e-02,  1.33326966e-02,  1.81114320e-02, ...,-4.52471897e-03, -1.08022038e-02, -1.01980157e-02]]],shape=(4, 1000, 52), dtype=float32)


Attributes: (5)


created_at :  
2026-11-06T09:49:05.462838+00:00

creation_library :  
ArviZ

creation_library_version :  
1.3.1

creation_library_language :  
Python

sample_dims :  
\['chain', 'draw'\]


/posterior_predictive(10)

Dimensions:


- chain: 4
- draw: 1000
- time: 417
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


0 1 2 3 4 5 ... 412 413 414 415 416


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 414, 415, 416], shape=(417,))


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


14.36 14.41 14.34 ... 14.73 13.96


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[14.363709 ],[14.413323 ],[14.335875 ],...,[14.559734 ],[14.670741 ],[14.262251 ]],[[14.279418 ],[14.419488 ],[14.434615 ],...,[14.503041 ],[14.667353 ],[14.130747 ]],[[14.319824 ],[14.449744 ],[14.420914 ],...,......,[14.690656 ],[14.778281 ],[14.171055 ]],[[14.431041 ],[14.484094 ],[14.4392395],...,[14.543997 ],[14.745484 ],[14.282976 ]],[[14.382096 ],[14.336303 ],[14.481818 ],...,[14.757873 ],[14.730902 ],[13.961009 ]]]], shape=(4, 1000, 417, 1), dtype=float32)


Attributes: (5)


created_at :  
2026-11-06T09:49:08.284511+00:00

creation_library :  
ArviZ

creation_library_version :  
1.3.1

creation_library_language :  
Python

sample_dims :  
\['chain', 'draw'\]


/observed_data(8)

Dimensions:


- time: 417
- obs_dim: 1


Coordinates: (2)


time


(time)


int64


0 1 2 3 4 5 ... 412 413 414 415 416


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 414, 415, 416], shape=(417,))


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


14.41 14.45 14.42 ... 14.65 14.04


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[14.409758 ],[14.452444 ],[14.424588 ],[14.529594 ],[14.516827 ],[14.536647 ],[14.499719 ],[14.4457035],[14.527379 ],[14.518174 ],[14.519266 ],[14.47356  ],[14.518399 ],[14.548611 ],[14.561788 ],[14.541207 ],[14.522805 ],[14.558051 ],[14.555797 ],[14.568521 ],...[14.6790285],[14.703018 ],[14.654665 ],[14.550027 ],[14.735775 ],[14.730156 ],[14.714912 ],[14.726917 ],[14.707603 ],[14.7097645],[14.720431 ],[14.708797 ],[14.707592 ],[14.594886 ],[14.177667 ],[14.655781 ],[14.69477  ],[14.711192 ],[14.645604 ],[14.038745 ]], dtype=float32)


Attributes: (5)


created_at :  
2026-11-06T09:49:08.284769+00:00

creation_library :  
ArviZ

creation_library_version :  
1.3.1

creation_library_language :  
Python

sample_dims :  
\[\]


/constant_data(8)

Dimensions:


- time: 417
- covariate_dim: 53


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


0 1 2 3 4 5 6 ... 47 48 49 50 51 52


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52])


Data variables: (1)


covariates


(time, covariate_dim)


float32


14.41 0.0 0.0 ... -0.4002 -0.2375


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[14.409758  ,  0.        ,  0.        , ...,  1.        ,1.        ,  1.        ],[14.452444  ,  0.12012617,  0.23851258, ..., -0.968519  ,-0.9914097 , -0.9999422 ],[14.424588  ,  0.23851258,  0.4632579 , ...,  0.8760579 ,0.9657866 ,  0.9997688 ],...,[14.711192  , -0.401229  , -0.7350337 , ..., -0.8852253 ,-0.62421006, -0.25829974],[14.645604  , -0.28829065, -0.5521013 , ...,  0.74156004,0.5166653 ,  0.24789836],[14.038745  , -0.17117706, -0.33730108, ..., -0.55120456,-0.40024394, -0.23746832]], shape=(417, 53), dtype=float32)


Attributes: (5)


created_at :  
2026-11-06T09:49:08.284956+00:00

creation_library :  
ArviZ

creation_library_version :  
1.3.1

creation_library_language :  
Python

sample_dims :  
\[\]


/predictions(10)

Dimensions:


- chain: 4
- draw: 1000
- time: 52
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


417 418 419 420 ... 465 466 467 468


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430,431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444,445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458,459, 460, 461, 462, 463, 464, 465, 466, 467, 468])


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


14.41 14.6 14.5 ... 14.71 14.29


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[14.414149 ],[14.60431  ],[14.502378 ],...,[14.680561 ],[14.704657 ],[14.152498 ]],[[14.312315 ],[14.50558  ],[14.51213  ],...,[14.576787 ],[14.719574 ],[14.247818 ]],[[14.22094  ],[14.739264 ],[14.766057 ],...,......,[14.679101 ],[14.582156 ],[14.145496 ]],[[14.432067 ],[14.568735 ],[14.59151  ],...,[14.749332 ],[14.806824 ],[14.316496 ]],[[14.35282  ],[14.59074  ],[14.641648 ],...,[14.662271 ],[14.712215 ],[14.290674 ]]]], shape=(4, 1000, 52, 1), dtype=float32)


Attributes: (5)


created_at :  
2026-11-06T09:49:09.363806+00:00

creation_library :  
ArviZ

creation_library_version :  
1.3.1

creation_library_language :  
Python

sample_dims :  
\['chain', 'draw'\]


/predictions_constant_data(8)

Dimensions:


- time: 52
- covariate_dim: 53


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


0 1 2 3 4 5 6 ... 47 48 49 50 51 52


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52])


Data variables: (1)


covariates


(time, covariate_dim)


float32


14.27 -0.05158 ... 0.1255 0.3139


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[ 1.4270954e+01, -5.1584367e-02, -1.0303138e-01, ...,3.2614416e-01,  2.7694622e-01,  2.2701082e-01],[ 1.4592882e+01,  6.8755411e-02,  1.3718541e-01, ...,-8.0549002e-02, -1.4889039e-01, -2.1652709e-01],[ 1.4619793e+01,  1.8809941e-01,  3.6948368e-01, ...,-1.7011768e-01,  1.8276557e-02,  2.0601831e-01],...,[ 1.4705699e+01, -4.2083099e-01, -7.6350451e-01, ...,-5.4039413e-01, -1.3612033e-01,  2.9343396e-01],[ 1.4669233e+01, -3.0881247e-01, -5.8743727e-01, ...,3.1392100e-01,  5.3757317e-03, -3.0369502e-01],[ 1.3986477e+01, -1.9232151e-01, -3.7746248e-01, ...,-6.7682765e-02,  1.2546124e-01,  3.1392100e-01]],shape=(52, 53), dtype=float32)


Attributes: (5)


created_at :  
2026-11-06T09:49:09.364051+00:00

creation_library :  
ArviZ

creation_library_version :  
1.3.1

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


Both trees carry the `posterior_predictive` group (the in-sample draws) and the `predictions` group (the forecast). To score them we merge the `(chain, draw)` dimensions into one sample axis. `plot_crps` then draws the CRPS of each model as a bar, with its value on top.


``` python
def stacked(tree: xr.DataTree, group: str) -> np.ndarray:
    """Draws of ``tree[group]["obs"]`` with ``(chain, draw)`` merged into ``(sample, time, 1)``."""
    return (
        tree[group]
        .dataset["obs"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "time", "obs_dim")
        .to_numpy()
    )


def plot_crps(scores: dict[str, dict[str, float]], title: str, xlabel: str = "model") -> None:
    """Bars of the CRPS, ``scores[window][model]``: one group per model, one color per window."""
    models = list(next(iter(scores.values())))
    x = np.arange(len(models))
    width = 0.8 / len(scores)
    _, ax = plt.subplots(figsize=(10, 5))
    for i, (window, by_model) in enumerate(scores.items()):
        offset = (i - (len(scores) - 1) / 2) * width
        bars = ax.bar(
            x + offset, [by_model[m] for m in models], width=width, color=f"C{i}", label=window
        )
        ax.bar_label(bars, fmt="%.4f")
    ax.set_xticks(x, models)
    ax.margins(y=0.25)
    if len(scores) > 1:
        ax.legend(title="scored on", loc="upper left")
    ax.set(title=title, xlabel=xlabel, ylabel="CRPS")


gaussian_crps = {
    "training window": {
        label: float(eval_crps(stacked(tree, "posterior_predictive"), bart_train))
        for label, tree in trees.items()
    },
    "test year": {
        label: float(eval_crps(stacked(tree, "predictions"), bart_test))
        for label, tree in trees.items()
    },
}
plot_crps(gaussian_crps, title="BART weekly ridership: CRPS of the Normal noise models");
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-12-output-1.png" class="figure-img" width="1011" height="511" /></p>
</figure>


The two models give the same in-sample CRPS (0.0315 and 0.0316) and the same test CRPS (0.0361 and 0.0360). Next we look at the in-sample predictive and the forecast of the `dynestyx` model over the last two years of the training window and the test year.


``` python
def hdi_label(prob: float, prefix: str = "") -> str:
    r"""Legend label for an HDI band, e.g. ``$94\%$ HDI``."""
    percent = f"{prob:.0%}".replace("%", r"\%")
    return f"{prefix}${percent}$ HDI"


hdi_probs = (0.5, 0.94)
hdi_alphas = [0.6, 0.3]  # 50% band darker, 94% band lighter
start = t_obs - 104  # the last two years of the training window
tree = trees["dynestyx Kalman smoother"]

pc = az.plot_lm(
    predictions_to_datatree(
        stacked(tree, "posterior_predictive")[:, start:], week[start:t_obs], ["bart"]
    ),
    y="obs",
    x="t",
    plot_dim="time",
    ci_kind="hdi",
    ci_prob=hdi_probs,
    smooth=False,
    aes={"alpha": ["prob"]},
    alpha=hdi_alphas,
    visuals={"ci_band": {"color": "C0"}, "observed_scatter": False, "pe_line": False},
    figure_kwargs={"figsize": (12, 6)},
)
in_sample_bands = {prob: pc.viz["ci_band"]["t"].sel(prob=prob).item() for prob in hdi_probs}
az.plot_lm(
    predictions_to_datatree(stacked(tree, "predictions"), week[t_obs:], ["bart"]),
    y="obs",
    x="t",
    plot_dim="time",
    plot_collection=pc,
    ci_kind="hdi",
    ci_prob=hdi_probs,
    smooth=False,
    aes={"alpha": ["prob"]},
    alpha=hdi_alphas,
    visuals={"ci_band": {"color": "C1"}, "observed_scatter": False, "pe_line": False},
)
forecast_bands = {prob: pc.viz["ci_band"]["t"].sel(prob=prob).item() for prob in hdi_probs}
for prob in hdi_probs:
    in_sample_bands[prob].set_label(hdi_label(prob, prefix="in-sample "))
    forecast_bands[prob].set_label(hdi_label(prob, prefix="forecast "))

ax = pc.viz["figure"].item().axes[0]
(obs_line,) = ax.plot(week[start:], bart[start:, 0], color="black", lw=1, label="observed")
split_line = ax.axvline(t_obs, color="gray", ls="--", label="train/test split")
ax.legend(
    handles=[*in_sample_bands.values(), *forecast_bands.values(), obs_line, split_line],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=3,
)
ax.set(
    title="BART weekly ridership: Kalman smoother in-sample predictive and forecast",
    xlabel="week",
    ylabel="log(# rides)",
);
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-13-output-1.png" class="figure-img" width="1211" height="611" /></p>
</figure>


The in-sample band is the observation noise around the smoothed level. The forecast band is about as wide: the weekly steps of the level are small, so one year of them adds little to the observation noise. A few of the sharpest dips, such as the last week of the test year, fall outside the 94\\ bands. The next section replaces the Normal noise with Student-T noise.


# Student-T Noise with an Explicit Path

The comparing inference methods example uses Student-T noise, whose heavy tails allow for the sharp dips of the series:

 y_t = b + \ell_t + \beta^\top u_t + v_t, \qquad v_t \sim \text{StudentT}(\nu, 0, \sigma). 

Student-T noise rules out an exact Kalman filter. A particle filter does not combine well with NUTS here: with resampling, the likelihood estimate is not continuous in the parameters, and without it the weights collapse over 417 weeks. The `dynestyx` model therefore samples the level with a `LatentPathBuilder`, as [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) does. The observation model is a function of the state x_t, the controls u_t and the time t.

The builder samples the states x_t = b + \ell_t directly (the centered form) rather than their steps. With the intercept b as a separate parameter, a change of b must move all 417 states the other way. In this form the intercept is the initial state, so the `dynestyx` model has no `bias` site and gives the initial state the \text{Normal}(0, 10) prior of the intercept. The two models differ only in the variance s^2 that the first step adds to this prior, which is negligible.


``` python
def level_regression(covariates: Array, data: Array | None = None) -> None:
    """Random walk level plus Fourier regression with Student-T noise, as a dynestyx model."""
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :1]
    controls = covariates[..., 1:]
    weight = numpyro.sample(
        "weight", dist.Normal(0.0, 0.1).expand([controls.shape[-1]]).to_event(1)
    )
    drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-20.0, 5.0))
    nu = numpyro.sample("nu", dist.Gamma(10.0, 2.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-5.0, 5.0))

    def observation(x: Array, u: Array | None, t: Array) -> dist.Distribution:
        regression = 0.0 if u is None else u @ weight
        return dist.StudentT(nu, x + regression, sigma).to_event(1)

    dynamics = DynamicalModel(
        initial_condition=dist.Normal(0.0, 10.0).expand([1]).to_event(1),  # the intercept
        state_evolution=lambda x, u, t_now, t_next: dist.Normal(x, drift_scale).to_event(1),
        observation_model=observation,
        control_dim=controls.shape[-1],
    )
    state_space(h, "f", y, dynamics, controls=controls)
```


We render the `dynestyx` model after its fit, because the `LatentPathBuilder` needs the concrete data of the fit (see below). As a reference, we write the same model with [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) and Student-T noise in [predict](../../reference/models.predict.md#numpyro_forecast.models.predict):


``` python
def univariate_model(covariates: Array, data: Array | None = None) -> None:
    """Random walk level plus Fourier regression with Student-T noise, with ``innovations``."""
    h = Horizon.from_data(covariates, data)
    fourier = covariates[..., 1:]
    bias = numpyro.sample("bias", dist.Normal(0.0, 10.0))
    weight = numpyro.sample(
        "weight", dist.Normal(0.0, 0.1).expand([fourier.shape[-1]]).to_event(1)
    )
    drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-20.0, 5.0))
    nu = numpyro.sample("nu", dist.Gamma(10.0, 2.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-5.0, 5.0))
    drift = innovations(
        h, "drift", dist.Normal(0.0, drift_scale), reparam=LocScaleReparam(centered=0)
    )
    level = jnp.cumsum(drift, axis=-2)
    regression = (weight * fourier).sum(axis=-1, keepdims=True)
    predict(h, dist.StudentT(df=nu, loc=0.0, scale=sigma), level + bias + regression)
```


Next, we render the [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) model:


``` python
numpyro.render_model(
    univariate_model,
    model_args=(bart_covariates[:t_obs], bart_train),
    render_distributions=True,
)
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-16-output-1.svg" class="img-fluid figure-img" /></p>
</figure>


Compared with the Normal model, the graph has one more parameter: the degrees of freedom `nu` of the Student-T noise.

Both fits run their chains one after the other (`chain_method="sequential"`). Parallel chains under a `LatentPathBuilder` crashed the Python process with `dynestyx` 0.8.0 and Python 3.14 on the machine that executed this notebook (`dynestyx` declares support for Python 3.12 and 3.13). The explicit path runs 4{,}000 warmup steps and 4{,}000 draws per chain, against 2{,}000 and 1{,}000 for [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations), because it mixes more slowly. The per-parameter table below shows its slowest parameter.

We keep the `LatentPathBuilder` instance of the fit. The builder records the pattern of missing observations from the concrete data of the fit. The drivers later run the model under `jax.jit`, where the data are traced, so every later call must use the same instance.


``` python
latent_path = LatentPathBuilder()
rng_key, key_latent, key_univariate = random.split(rng_key, 3)

with latent_path:
    mcmc_latent, seconds = fit_nuts(
        key_latent,
        level_regression,
        bart_covariates[:t_obs],
        bart_train,
        num_warmup=4_000,
        num_samples=4_000,
        chain_method="sequential",
    )

student_fits = {"dynestyx latent path": diagnostics(mcmc_latent, seconds)}

mcmc_univariate, seconds = fit_nuts(
    key_univariate,
    univariate_model,
    bart_covariates[:t_obs],
    bart_train,
    num_warmup=2_000,
    chain_method="sequential",
)
student_fits["innovations"] = diagnostics(mcmc_univariate, seconds)

fits_table(student_fits)
```


shape: (2, 7)

| fit | parameters | min bulk ESS | max R-hat | divergences | leapfrog steps per draw | wall time (s) |
|----|----|----|----|----|----|----|
| str | i64 | i64 | f64 | i64 | f64 | f64 |
| "dynestyx latent path" | 472 | 439 | 1.005 | 0 | 135.2 | 54.8 |
| "innovations" | 473 | 946 | 1.006 | 0 | 255.0 | 73.5 |


With these chain lengths both fits converge over all their parameters:

- The largest \hat{R} is 1.006 and the smallest bulk ESS is 439. There are no divergent transitions.
- The two fits ran chains of different lengths, so their wall times do not compare their cost.
- Per draw, the explicit path takes 135.2 leapfrog steps and [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) 255.0, the length of a NUTS tree of depth 8.

Now we can render the `dynestyx` model. `numpyro.render_model` traces the model, as the drivers do, so it needs the builder of the fit, which has recorded the pattern of missing observations:


``` python
with latent_path:
    graph = numpyro.render_model(
        level_regression,
        model_args=(bart_covariates[:t_obs], bart_train),
        render_distributions=True,
    )

graph
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-18-output-1.svg" class="img-fluid figure-img" /></p>
</figure>


The builder adds the latent path as one sample site, `f_state_path_params`, which holds the 417 weekly states. The parameters and the path feed the gray factor `f_joint_log_prob_factor`: the joint log density of the path and the observations. The dashed nodes are deterministic sites that `dynestyx` records, such as the path `f_state_path`.

Next, we look at the diagnostics of the three scalar parameters that both models share:


``` python
student_names = ["drift_scale", "nu", "sigma"]  # both models also share the 52 weights
student_summaries = {
    label: az.summary(posterior_tree(mcmc, student_names), fmt="xarray", round_to="none")
    for label, mcmc in (("dynestyx latent path", mcmc_latent), ("innovations", mcmc_univariate))
}
pl.DataFrame(
    [
        {"fit": label, "parameter": name}
        | {
            stat: float(summary[name].sel(summary=stat))
            for stat in ("mean", "sd", "ess_bulk", "r_hat")
        }
        for label, summary in student_summaries.items()
        for name in student_names
    ]
).with_columns(
    pl.col("mean", "sd").round(4),
    pl.col("ess_bulk").round(0).cast(pl.Int64),
    pl.col("r_hat").round(3),
)
```


shape: (6, 6)

| fit                    | parameter     | mean   | sd     | ess_bulk | r_hat |
|------------------------|---------------|--------|--------|----------|-------|
| str                    | str           | f64    | f64    | i64      | f64   |
| "dynestyx latent path" | "drift_scale" | 0.0044 | 0.0008 | 439      | 1.005 |
| "dynestyx latent path" | "nu"          | 1.5749 | 0.2216 | 9972     | 1.0   |
| "dynestyx latent path" | "sigma"       | 0.0172 | 0.002  | 6682     | 1.001 |
| "innovations"          | "drift_scale" | 0.0045 | 0.0007 | 1335     | 1.004 |
| "innovations"          | "nu"          | 1.5674 | 0.2233 | 1228     | 1.001 |
| "innovations"          | "sigma"       | 0.0171 | 0.002  | 946      | 1.001 |


`drift_scale` is the slowest parameter of the explicit path: its bulk ESS is 439 from 16{,}000 draws, against 1{,}335 from 4{,}000 draws for [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations). The builder samples the states in centered form and has no reparameterization option, while [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) samples the non-centered steps.


``` python
plot_posteriors(
    {"dynestyx latent path": mcmc_latent, "innovations": mcmc_univariate},
    student_names,
    title="Local level and Fourier regression posteriors, Student-T noise",
)
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-20-output-1.png" class="figure-img" width="1206" height="361" /></p>
</figure>


The posteriors of the three shared parameters coincide.

Both models forecast through [forecast](../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast). The `LatentPathBuilder` around the call is the instance that fitted the model.


``` python
rng_key, key_latent, key_univariate = random.split(rng_key, 3)

with DiscreteTimeSimulator(n_simulations=1), latent_path:
    student_forecasts = {
        "dynestyx latent path": forecast(
            key_latent,
            level_regression,
            mcmc_latent.get_samples(),
            bart_train,
            bart_covariates,
            batch_size=500,
        )
    }

student_forecasts["innovations"] = forecast(
    key_univariate,
    univariate_model,
    mcmc_univariate.get_samples(),
    bart_train,
    bart_covariates,
    batch_size=500,
)

student_crps = {
    label: float(eval_crps(draws, bart_test)) for label, draws in student_forecasts.items()
}
plot_crps(
    {"test year": student_crps},
    title="BART weekly ridership: test CRPS of the Student-T noise models",
);
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-21-output-1.png" class="figure-img" width="1011" height="511" /></p>
</figure>


The test CRPS is 0.0302 for the explicit path and 0.0303 for [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations). Both score better than the Normal models of the previous section (test CRPS 0.0361 and 0.0360). We plot the two forecasts one above the other. The point estimate is the median of the draws: with \nu near 1.6 the Student-T noise has no finite variance, so a few extreme draws move the mean. The explicit path has four times as many draws as [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations), so we thin them to the same number for the figure.


``` python
num_draws = min(len(draws) for draws in student_forecasts.values())
thinned = [np.asarray(draws)[:: len(draws) // num_draws] for draws in student_forecasts.values()]
pc = az.plot_lm(
    predictions_to_datatree(
        np.concatenate(thinned, axis=-1),
        week[t_obs:],
        list(student_forecasts),
    ),
    y="obs",
    x="t",
    plot_dim="time",
    ci_kind="hdi",
    ci_prob=hdi_probs,
    smooth=False,
    point_estimate="median",
    col_wrap=1,
    aes={"color": ["series"], "alpha": ["prob"]},
    alpha=hdi_alphas,
    aes_by_visuals={"pe_line": ["color"]},
    visuals={
        "observed_scatter": False,
        "pe_line": {"alpha": 1.0, "width": 1.5},
        "xlabel": False,
        "ylabel": False,
    },
    figure_kwargs={"figsize": (12, 10), "sharex": True, "sharey": True},
)
history = slice(t_obs - 104, t_obs)
for label in student_forecasts:
    ax = pc.get_target("t", {"series": label})
    ax.plot(week[history], bart_train[history, 0], color="black", lw=1, label="train")
    ax.plot(week[t_obs:], bart_test[:, 0], color="black", lw=1, ls="--", label="test")
    ax.axvline(t_obs, color="gray", ls="--", lw=1)
    ax.set(title=f"{label} (test CRPS: {student_crps[label]:.4f})", ylabel="log(# rides)")
    for prob in hdi_probs:
        pc.viz["ci_band"]["t"].sel(series=label, prob=prob).item().set_label(hdi_label(prob))
    pc.viz["pe_line"]["t"].sel(series=label).item().set_label("forecast median")
    ax.legend(loc="lower left", ncol=5)
ax.set(xlabel="week")
pc.viz["figure"].item().suptitle(
    "BART weekly ridership: Student-T forecasts of the last 52 weeks",
    fontsize=14,
    fontweight="bold",
);
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-22-output-1.png" class="figure-img" width="1211" height="1011" /></p>
</figure>


The two panels show the same forecast: over the test year the medians and the HDI bands of the two models agree closely.


# Missing Data

The `LatentPathBuilder` skips missing observations, so the Student-T model fits through a gap with no change to the model function. (The Kalman filter and smoother of the `cuthbert` backend skip them too, while the `cd_dynamax` backend rejects them.) We remove 20 weeks of the training window by setting them to `NaN` in two places. The block reads the observed window from column 0 of the covariates, so the gap has to go there to reach the likelihood. A `NaN` in `data` alone would not reach it. We also put the gap into `data`, which [to_datatree](../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) would store as the observed data.

The gap fit gets its own `LatentPathBuilder` and the longer chains of the explicit path. A builder keeps one pattern of missing observations per site name and shape. A second fit with the same builder and a different pattern replaces it, and a later call that evaluates the likelihood of the first model (a refit or `log_density`) then fails.


``` python
gap = slice(300, 320)
bart_gap = bart_train.at[gap].set(jnp.nan)
bart_covariates_gap = bart_covariates.at[gap, 0].set(jnp.nan)
gap_builder = LatentPathBuilder()
rng_key, key_gap = random.split(rng_key)

with gap_builder:
    mcmc_gap, seconds = fit_nuts(
        key_gap,
        level_regression,
        bart_covariates_gap[:t_obs],
        bart_gap,
        num_warmup=4_000,
        num_samples=4_000,
        chain_method="sequential",
    )

fits_table({"gap fit": diagnostics(mcmc_gap, seconds)})
```


shape: (1, 7)

| fit | parameters | min bulk ESS | max R-hat | divergences | leapfrog steps per draw | wall time (s) |
|----|----|----|----|----|----|----|
| str | i64 | i64 | f64 | i64 | f64 | f64 |
| "gap fit" | 472 | 355 | 1.013 | 0 | 163.6 | 55.5 |


The gap fit nearly converges: the largest \hat{R} is 1.013, just above the usual threshold of 1.01, the smallest bulk ESS is 355, and there are no divergent transitions. The conclusions below are qualitative and unlikely to change with longer chains.

We draw the in-sample predictive of the gap fit, which covers the gap weeks, under the builder of that fit. As a reference we draw the in-sample predictive of the fit on all weeks from the previous section. That fit has seen the weeks of the gap, so we expect it to score better there.


``` python
rng_key, key_gap_pred, key_full_pred = random.split(rng_key, 3)
in_sample = {}

with gap_builder:
    in_sample["fit with the gap"] = predict_in_sample(
        key_gap_pred,
        level_regression,
        mcmc_gap.get_samples(),
        bart_covariates_gap[:t_obs],
        batch_size=500,
    )

with latent_path:
    in_sample["fit on all weeks"] = predict_in_sample(
        key_full_pred,
        level_regression,
        mcmc_latent.get_samples(),
        bart_covariates[:t_obs],
        batch_size=500,
    )

gap_crps = {
    label: float(eval_crps(draws[:, gap], bart_train[gap])) for label, draws in in_sample.items()
}
plot_crps(
    {"gap weeks": gap_crps}, title="In-sample CRPS on the 20 weeks of the gap", xlabel="fit"
);
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-24-output-1.png" class="figure-img" width="1011" height="511" /></p>
</figure>


On the 20 held-out weeks the CRPS of the gap fit is 0.0265, against 0.0229 for the fit that saw them. We plot the in-sample predictive of the gap fit around the gap, with the median of the fit on all weeks as a reference.


``` python
window = slice(250, t_obs)
pc = az.plot_lm(
    predictions_to_datatree(
        np.asarray(in_sample["fit with the gap"])[:, window], week[window], ["bart"]
    ),
    y="obs",
    x="t",
    plot_dim="time",
    ci_kind="hdi",
    ci_prob=hdi_probs,
    smooth=False,
    aes={"alpha": ["prob"]},
    alpha=hdi_alphas,
    point_estimate="median",
    visuals={
        "ci_band": {"color": "C1"},
        "observed_scatter": False,
        "pe_line": {"color": "C1", "alpha": 1.0, "width": 1.5},
    },
    figure_kwargs={"figsize": (12, 6)},
)
for prob in hdi_probs:
    pc.viz["ci_band"]["t"].sel(prob=prob).item().set_label(hdi_label(prob, "fit with the gap "))
pc.viz["pe_line"]["t"].item().set_label("fit with the gap, median")
ax = pc.viz["figure"].item().axes[0]
ax.plot(
    week[window],
    np.median(np.asarray(in_sample["fit on all weeks"]), axis=0)[window, 0],
    color="C0",
    ls="--",
    label="fit on all weeks, median",
)
ax.plot(week[window], bart_gap[window, 0], "o", color="black", ms=3, label="observed")
ax.plot(week[gap], bart_train[gap, 0], "o", color="C2", ms=4, mfc="none", label="held out")
ax.axvspan(gap.start, gap.stop - 1, color="gray", alpha=0.15, label="gap")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
ax.set(
    title="In-sample posterior predictive through a 20-week gap",
    xlabel="week",
    ylabel="log(# rides)",
);
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-25-output-1.png" class="figure-img" width="1211" height="611" /></p>
</figure>


The band of the gap fit follows the Fourier regression through the gap, including the two dips of the yearly cycle inside it. The median of the fit on all weeks stays inside the band.


# A Two-Dimensional State: ARMA(1,1)

The local level has a one-dimensional state. The [ARMA example](arma.md) writes the ARMA(1,1) model

 y_t = \mu + \phi \\ y\_{t-1} + \theta \\ \varepsilon\_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim \text{Normal}(0, \sigma) 

with [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe), which conditions on y\_{-1} = \mu and \varepsilon\_{-1} = 0. The same model is a linear-Gaussian state space model with a two-dimensional state, so the Kalman filter gives the exact likelihood instead. With the deviation a_t = y_t - c from the process mean c = \mu / (1 - \phi), the state x_t = (a_t, \theta \varepsilon_t) follows

 x_t = A x\_{t-1} + g \\ \varepsilon_t, \qquad A = \begin{pmatrix} \phi & 1 \\ 0 & 0 \end{pmatrix}, \qquad g = \begin{pmatrix} 1 \\ \theta \end{pmatrix}, 

so the state covariance is Q = \sigma^2 g g^\top, and the observation is y_t = H x_t + c with H = (1, 0). The initial state has the stationary covariance P_0, the solution of P_0 = A P_0 A^\top + Q. The state noise has rank one, so we add a small jitter \delta = 10^{-4} \sigma^2 to Q, to P_0 and as the observation variance, which keeps the Cholesky factors defined in single precision.

We simulate 100 steps with \phi = 0.4, \theta = 0.7 and \sigma = 0.5, as the ARMA example does.


``` python
phi_true, theta_true, noise_scale, n_steps = 0.4, 0.7, 0.5, 100


def simulate_arma(
    rng_key: Array, n: int, phi: float, theta: float, scale: float
) -> Float[Array, " time"]:
    """Simulate ``n`` steps of a zero-mean ARMA(1,1) process."""
    error = scale * random.normal(rng_key, (n + 1,))

    def step(carry: tuple[Array, Array], noise: Array) -> tuple[tuple[Array, Array], Array]:
        y_prev, error_prev = carry
        y_t = phi * y_prev + theta * error_prev + noise
        return (y_t, noise), y_t

    return jax.lax.scan(step, (jnp.zeros(()), error[0]), error[1:])[1]


rng_key, rng_subkey = random.split(rng_key)
arma_data = simulate_arma(rng_subkey, n_steps, phi_true, theta_true, noise_scale)[:, None]
arma_covariates = arma_data  # both models read the observed series from the covariates
```


`arma_dynamics` builds the matrices above:


``` python
def arma_dynamics(mu: Array, phi: Array, theta: Array, sigma: Array) -> DynamicalModel:
    """ARMA(1,1) as a linear-Gaussian state space model with a stationary initial state."""
    A = jnp.array([[phi, 1.0], [0.0, 0.0]])
    g = jnp.array([1.0, theta])
    Q = sigma**2 * jnp.outer(g, g)
    jitter = 1e-4 * sigma**2 * jnp.eye(2)
    P0 = jnp.linalg.solve(jnp.eye(4) - jnp.kron(A, A), Q.reshape(-1)).reshape(2, 2)
    return DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(2), 0.5 * (P0 + P0.T) + jitter),
        state_evolution=LinearGaussianStateEvolution(A=A, cov=Q + jitter),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0, 0.0]]), R=jitter[:1, :1], bias=(mu / (1.0 - phi))[None]
        ),
    )
```


The two model functions share the priors of the ARMA example and read the series from the covariates. `arma_ssm` passes the dynamics to the block:


``` python
def arma_ssm(covariates: Array, data: Array | None = None) -> None:
    """ARMA(1,1) as a dynestyx state space model."""
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :]
    mu = numpyro.sample("mu", dist.Normal(loc=0, scale=1))
    phi = numpyro.sample("phi", dist.Uniform(low=-1, high=1))
    theta = numpyro.sample("theta", dist.Uniform(low=-1, high=1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=1))
    state_space(h, "f", y, arma_dynamics(mu, phi, theta, sigma))
```


We render it under the Kalman filter of the local level section:


``` python
with kalman:
    graph = numpyro.render_model(
        arma_ssm, model_args=(arma_covariates, arma_data), render_distributions=True
    )

graph
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-29-output-1.svg" class="img-fluid figure-img" /></p>
</figure>


As for the local level, the filter integrates the state out: the four parameters feed the factor `f_marginal_log_likelihood`, and the graph has no `time` plate.

`arma_ssoe` is the model of the ARMA example:


``` python
def arma_ssoe(covariates: Array, data: Array | None = None) -> None:
    """ARMA(1,1) in innovations form with ``ssoe``."""
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :]
    mu = numpyro.sample("mu", dist.Normal(loc=0, scale=1))
    phi = numpyro.sample("phi", dist.Uniform(low=-1, high=1))
    theta = numpyro.sample("theta", dist.Uniform(low=-1, high=1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=1))

    def mean(carry: tuple[Array, Array], _: Array | None) -> Array:
        y_prev, error_prev = carry
        return mu + phi * y_prev + theta * error_prev

    def update(
        carry: tuple[Array, Array], y_t: Array, eps_t: Array, _: Array | None
    ) -> tuple[Array, Array]:
        return y_t, eps_t

    result = ssoe(h, "eps", y, (mu[None], jnp.zeros((1,))), mean, update, dist.Normal(0, sigma))
    numpyro.sample("obs", dist.Normal(loc=result.mu, scale=sigma), obs=h.data)
    if h.future > 0:
        numpyro.deterministic("forecast", result.y_future)
```


It has no `dynestyx` site, so we render it without a handler:


``` python
numpyro.render_model(arma_ssoe, model_args=(arma_covariates, arma_data), render_distributions=True)
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-31-output-1.svg" class="img-fluid figure-img" /></p>
</figure>


The four parameters enter the likelihood of the observed `obs` site directly, through the [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) recursion.

The `dynestyx` fit runs under the Kalman filter of the local level section.


``` python
rng_key, key_arma_kalman, key_arma_ssoe = random.split(rng_key, 3)

with kalman:
    mcmc_arma_kalman, seconds = fit_nuts(
        key_arma_kalman,
        arma_ssm,
        arma_covariates,
        arma_data,
        num_warmup=1_000,
        chain_method="parallel",
    )

arma_fits = {"dynestyx Kalman filter": diagnostics(mcmc_arma_kalman, seconds)}

mcmc_arma_ssoe, seconds = fit_nuts(
    key_arma_ssoe, arma_ssoe, arma_covariates, arma_data, num_warmup=1_000, chain_method="parallel"
)
arma_fits["ssoe"] = diagnostics(mcmc_arma_ssoe, seconds)

fits_table(arma_fits)
```


shape: (2, 7)

| fit | parameters | min bulk ESS | max R-hat | divergences | leapfrog steps per draw | wall time (s) |
|----|----|----|----|----|----|----|
| str | i64 | i64 | f64 | i64 | f64 | f64 |
| "dynestyx Kalman filter" | 4 | 3633 | 1.003 | 0 | 6.2 | 7.8 |
| "ssoe" | 4 | 3476 | 1.0 | 0 | 6.0 | 0.8 |


Both fits converge (largest \hat{R} 1.003, smallest bulk ESS 3{,}476, no divergent transitions), with about six leapfrog steps per draw each: neither model samples a path.


``` python
plot_posteriors(
    {"dynestyx Kalman filter": mcmc_arma_kalman, "ssoe": mcmc_arma_ssoe},
    ["mu", "phi", "theta", "sigma"],
    title="ARMA(1,1) posteriors (dashed: true value)",
    truth={"mu": 0.0, "phi": phi_true, "theta": theta_true, "sigma": noise_scale},
)
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-33-output-1.png" class="figure-img" width="1206" height="361" /></p>
</figure>


The two posteriors are close, and the true values lie inside the bulk of both. The [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) likelihood conditions on y\_{-1} = \mu and \varepsilon\_{-1} = 0, and the Kalman filter likelihood is exact up to the small jitter. The small differences between the two posteriors can come from these two likelihoods or from Monte Carlo error, and this run does not separate the two.


# Conclusion


## Key Findings

- **One model function, any handler.** [state_space](../../reference/contrib.dynestyx.state_space.md#numpyro_forecast.contrib.dynestyx.state_space) turns a `DynamicalModel` into a `numpyro_forecast` model. The handler around each call selects the inference: a Kalman filter or smoother integrates the level out, a `LatentPathBuilder` samples it. [forecast](../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast), [predict_in_sample](../../reference/predictive.predict_in_sample.md#numpyro_forecast.predictive.predict_in_sample) and [to_datatree](../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) work unchanged with the handler around the call, and the block raises an error when the handler is missing.
- **The Kalman filter integrates the level out.** For the Normal model it gives the same posteriors and the same in-sample and test CRPS as [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations), with 55 parameters for NUTS instead of 472. A Kalman smoother exports the filter fit with [to_datatree](../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree), without a refit.
- **The explicit path covers Student-T noise and gaps.** Its posteriors and forecast match [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations), but it samples the states in centered form and needs longer chains. The fit through the gap nearly converges.
- **The Kalman filter gives the exact ARMA(1,1) likelihood**, up to a small jitter. Its posteriors are close to those of the [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) model of the ARMA example.


## Limitations

- The in-sample draws under a smoother are independent across weeks. For joint paths, use [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) or a `LatentPathBuilder`.
- The `LatentPathBuilder` has no reparameterization option, and every later call must use the builder instance of the fit. Parallel chains under it crashed the process with Python 3.14 on the machine that executed this notebook.
- The block models one series per model function, with the step index as its time grid.


## Next Steps

The [advanced `dynestyx` example](dynestyx_integration.md) compares filtering and smoothing on a local level model and measures the cost of a full evaluation against a gradient. It also fits a seasonal model with SVI and backtests a `dynestyx` model with expanding windows.
