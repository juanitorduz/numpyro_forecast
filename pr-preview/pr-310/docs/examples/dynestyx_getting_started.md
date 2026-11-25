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
7.  **Reusing a filtered history.** We forecast from new weeks without a second filter pass over the training window.


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
from dynestyx.configs import KFConfig, KFSmootherConfig
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
from numpyro_forecast.contrib.dynestyx import end_state, start_state, state_space
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


    dynestyx 0.9.0, numpyro 0.22.0, jax 0.12.0.1


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
| "dynestyx Kalman filter" | 55 | 5268 | 1.005 | 0 | 13.6 | 86.6 |
| "innovations" | 472 | 1870 | 1.007 | 0 | 127.0 | 9.1 |


Both fits converge over all their parameters:

- The largest \hat{R} is 1.007 and the smallest bulk ESS is 1{,}870. There are no divergent transitions.
- NUTS explores 55 parameters with the Kalman filter and 472 with [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations), the 417 weekly steps of the level included.
- The filter takes 13.6 leapfrog steps per draw, against 127.0 for [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations). Each of those steps runs the filter as a sequential scan over the 417 weeks. That costs much more than the cumulative sum of [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations), so on the CPU that executed this notebook the Kalman fit took longer in wall time.

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
│           bias          (chain, draw) float32 16kB 14.51 14.5 14.5 ... 14.52 14.51
│           drift_scale   (chain, draw) float32 16kB 0.004726 0.00821 ... 0.007556
│           sigma         (chain, draw) float32 16kB 0.07841 0.07886 ... 0.0738 0.07753
│           weight        (chain, draw, weight_dim_0) float32 832kB -0.01736 ... -0.0...
│       Attributes:
│           created_at:                 2026-11-25T20:33:47.654473+00:00
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
│           obs      (chain, draw, time, obs_dim) float32 7MB 14.36 14.41 ... 13.97
│       Attributes:
│           created_at:                 2026-11-25T20:33:50.037480+00:00
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
│           created_at:                 2026-11-25T20:33:50.037743+00:00
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
│           created_at:                 2026-11-25T20:33:50.037934+00:00
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
│           obs      (chain, draw, time, obs_dim) float32 832kB 14.41 14.6 ... 14.7 14.3
│       Attributes:
│           created_at:                 2026-11-25T20:33:51.093098+00:00
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
            created_at:                 2026-11-25T20:33:51.093345+00:00
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


14.51 14.5 14.5 ... 14.52 14.51


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[14.51325  , 14.496571 , 14.500757 , ..., 14.488217 , 14.506786 ,14.49062  ],[14.517092 , 14.500598 , 14.513791 , ..., 14.519422 , 14.496621 ,14.5242605],[14.563289 , 14.525236 , 14.505895 , ..., 14.500653 , 14.50991  ,14.552671 ],[14.551686 , 14.480451 , 14.509668 , ..., 14.503634 , 14.522429 ,14.506742 ]], shape=(4, 1000), dtype=float32)


drift_scale


(chain, draw)


float32


0.004726 0.00821 ... 0.007556


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.00472574, 0.00821038, 0.00627228, ..., 0.00575402, 0.00778546,0.00585143],[0.00600302, 0.00487772, 0.00699058, ..., 0.00718063, 0.0092594 ,0.00503238],[0.0070326 , 0.01182093, 0.00348751, ..., 0.00544998, 0.00471292,0.00552508],[0.00594941, 0.00592584, 0.00646334, ..., 0.00827641, 0.00472978,0.00755565]], shape=(4, 1000), dtype=float32)


sigma


(chain, draw)


float32


0.07841 0.07886 ... 0.0738 0.07753


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.07840638, 0.07886042, 0.08284794, ..., 0.07790546, 0.07768471,0.07348783],[0.07556581, 0.08146802, 0.07786708, ..., 0.07745885, 0.07619439,0.07981718],[0.0780354 , 0.07807573, 0.07620809, ..., 0.07419238, 0.08403311,0.07783786],[0.07599479, 0.07624798, 0.07802195, ..., 0.07281648, 0.07380091,0.07752902]], shape=(4, 1000), dtype=float32)


weight


(chain, draw, weight_dim_0)


float32


-0.01736 0.01326 ... -0.005573


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[-1.73618849e-02,  1.32646412e-02,  1.07884845e-02, ...,2.94772373e-03, -1.57449637e-02, -1.54059874e-02],[ 5.57414908e-03,  1.80592611e-02,  1.53072290e-02, ...,-1.25026442e-02,  1.27683785e-02,  2.88921921e-03],[-9.32747126e-03,  1.97908115e-02,  1.00216195e-02, ...,-1.10817151e-02, -6.09394722e-03,  1.64533150e-03],...,[-1.74090005e-02,  1.17448661e-02,  8.31314921e-03, ...,-6.81386748e-03,  4.30344371e-03,  1.29253452e-03],[-9.07925330e-03,  1.69784687e-02,  2.15858128e-02, ...,-1.22634461e-02, -1.71422795e-03, -7.61844963e-03],[ 6.00126013e-03,  8.55708029e-03,  1.49964960e-02, ...,-1.23800235e-02,  7.60230469e-05, -8.83628335e-03]],[[ 7.49800354e-03,  1.98826808e-02,  1.85982343e-02, ...,-1.95372826e-03,  4.50334465e-03, -4.71643545e-03],[-1.41890403e-02,  1.68383084e-02,  1.32821351e-02, ...,-1.41145233e-02,  6.01238920e-04, -5.40984329e-03],[ 4.09626029e-03,  2.43002307e-02,  2.36802399e-02, ...,-3.48075852e-03, -9.00015328e-03, -3.34264315e-03],...4.03936341e-04, -2.31856969e-03, -6.94129476e-03],[-7.56813865e-03,  7.43227545e-03,  1.04455967e-02, ...,-7.46514974e-03,  1.34189858e-03, -1.53792324e-02],[-7.34553952e-03,  1.62489023e-02,  9.56790149e-03, ...,-9.71644465e-03,  5.01982262e-03, -2.50129122e-03]],[[-1.12222135e-02,  7.74421403e-03,  1.14418520e-02, ...,-1.04364278e-02, -4.11937898e-03, -1.43722743e-02],[ 6.46366598e-03,  2.32083946e-02,  1.86469108e-02, ...,-2.51584873e-03,  1.90873235e-03, -3.09185870e-03],[-8.31114408e-03,  1.29612368e-02,  1.25042312e-02, ...,-1.43380361e-02, -5.36817266e-03, -1.05273146e-02],...,[ 1.31252082e-02,  2.03733537e-02,  1.41429249e-02, ...,-4.46986128e-03, -3.04257148e-03, -5.89711592e-03],[-1.08664494e-03,  2.02494748e-02,  1.93038899e-02, ...,3.19263060e-03, -1.25076408e-02, -1.14839301e-02],[-8.84233415e-03,  1.23416046e-02,  9.55662411e-03, ...,-1.63127519e-02,  1.00592542e-02, -5.57316048e-03]]],shape=(4, 1000, 52), dtype=float32)


Attributes: (5)


created_at :  
2026-11-25T20:33:47.654473+00:00

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


14.36 14.41 14.34 ... 14.71 13.97


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[14.363575 ],[14.413854 ],[14.344045 ],...,[14.561677 ],[14.678786 ],[14.258564 ]],[[14.277088 ],[14.423851 ],[14.422554 ],...,[14.499625 ],[14.65388  ],[14.133951 ]],[[14.318539 ],[14.444922 ],[14.422164 ],...,......,[14.690333 ],[14.778405 ],[14.17052  ]],[[14.429705 ],[14.484902 ],[14.439282 ],...,[14.545208 ],[14.74539  ],[14.28304  ]],[[14.392194 ],[14.277823 ],[14.503437 ],...,[14.861187 ],[14.714692 ],[13.968086 ]]]], shape=(4, 1000, 417, 1), dtype=float32)


Attributes: (5)


created_at :  
2026-11-25T20:33:50.037480+00:00

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
2026-11-25T20:33:50.037743+00:00

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
2026-11-25T20:33:50.037934+00:00

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


14.41 14.6 14.51 ... 14.7 14.3


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[14.412554 ],[14.602383 ],[14.507754 ],...,[14.682039 ],[14.713711 ],[14.150381 ]],[[14.311101 ],[14.509119 ],[14.506822 ],...,[14.5743885],[14.705229 ],[14.24733  ]],[[14.217356 ],[14.735993 ],[14.7646265],...,......,[14.677975 ],[14.582447 ],[14.143853 ]],[[14.431208 ],[14.568812 ],[14.592839 ],...,[14.749383 ],[14.806357 ],[14.315938 ]],[[14.361811 ],[14.573044 ],[14.617075 ],...,[14.786309 ],[14.698449 ],[14.302065 ]]]], shape=(4, 1000, 52, 1), dtype=float32)


Attributes: (5)


created_at :  
2026-11-25T20:33:51.093098+00:00

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
2026-11-25T20:33:51.093345+00:00

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

Both fits run their chains one after the other (`chain_method="sequential"`). In an earlier run with `dynestyx` 0.8.0, jax 0.11.2 and Python 3.14, parallel chains under a `LatentPathBuilder` crashed the Python process (`dynestyx` declares support for Python 3.12 and 3.13). This run did not try them again. The explicit path runs 4{,}000 warmup steps and 4{,}000 draws per chain, against 2{,}000 and 1{,}000 for [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations), because it mixes more slowly. The per-parameter table below shows its slowest parameter.

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
| "dynestyx latent path" | 472 | 372 | 1.012 | 0 | 128.6 | 52.4 |
| "innovations" | 473 | 888 | 1.006 | 0 | 261.0 | 69.9 |


With these chain lengths the [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) fit converges and the explicit path nearly converges:

- The largest \hat{R} of the explicit path is 1.012, just above the usual threshold of 1.01, and its smallest bulk ESS is 372.
- The largest \hat{R} of [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) is 1.006 and its smallest bulk ESS is 888.
- Neither fit has a divergent transition.
- The two fits ran chains of different lengths, so their wall times do not compare their cost.
- Per draw, the explicit path takes 128.6 leapfrog steps and [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) 261.0, about the 255 steps of a NUTS tree of depth 8.

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
| "dynestyx latent path" | "drift_scale" | 0.0045 | 0.0008 | 372      | 1.012 |
| "dynestyx latent path" | "nu"          | 1.5708 | 0.2192 | 10459    | 1.0   |
| "dynestyx latent path" | "sigma"       | 0.0172 | 0.002  | 7251     | 1.0   |
| "innovations"          | "drift_scale" | 0.0045 | 0.0007 | 1412     | 1.003 |
| "innovations"          | "nu"          | 1.5661 | 0.2214 | 1238     | 1.002 |
| "innovations"          | "sigma"       | 0.0171 | 0.002  | 888      | 1.001 |


`drift_scale` is the slowest parameter of the explicit path, and the one with the largest \hat{R} (1.012): its bulk ESS is 372 from 16{,}000 draws, against 1{,}412 from 4{,}000 draws for [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations). The builder samples the states in centered form and has no reparameterization option, while [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) samples the non-centered steps.


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


The test CRPS is 0.0302 for both the explicit path and [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations). Both score better than the Normal models of the previous section (test CRPS 0.0361 and 0.0360). We plot the two forecasts one above the other. The point estimate is the median of the draws: with \nu near 1.6 the Student-T noise has no finite variance, so a few extreme draws move the mean. The explicit path has four times as many draws as [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations), so we thin them to the same number for the figure.


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
| "gap fit" | 472 | 407 | 1.01 | 0 | 177.2 | 59.8 |


The gap fit converges, at the limit: the largest \hat{R} is 1.01, which is the usual threshold, the smallest bulk ESS is 407, and there are no divergent transitions. The conclusions below are qualitative and unlikely to change with longer chains.

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
| "dynestyx Kalman filter" | 4 | 3633 | 1.003 | 0 | 6.2 | 7.4 |
| "ssoe" | 4 | 3476 | 1.0 | 0 | 6.0 | 1.1 |


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


# Reusing a Filtered History

This section forecasts from new observations without a second pass over the history. Every [forecast](../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast) call conditions on the whole observed window again, for every posterior draw. Between two refits, new weeks arrive while the parameters stay at the posterior draws of the last fit, and only the level moves. The filter of the history is then the same computation in every call, so we run it once and keep its result.

The result to keep is the **filtering distribution at the last week** T of the training window. It is the distribution of the level given the observations up to that week and the parameters \theta of one posterior draw:

 \ell_T \mid y\_{1:T}, \theta \sim \text{Normal}(m_T, P_T). 

The Kalman filter computes the mean m_T and the variance P_T in closed form, so there is one pair (m_T, P_T) per posterior draw. Two functions of `numpyro_forecast.contrib.dynestyx` carry this pair from one window to the next:

- [`end_state`](../../reference/contrib.dynestyx.end_state.md) takes the pair from the draws of a run under a filter and returns it as two posterior sites.
- [`start_state`](../../reference/contrib.dynestyx.start_state.md) reads the two sites inside a model function.

`dynestyx` puts the initial condition of a model at its first observation time, which is week T + 1 for the new window. The model must therefore move the level one random walk step forward before it builds the initial condition. The step keeps the mean and adds the variance of one step, with s the `drift_scale` as in *Local Level with a Kalman Filter*:

 \ell\_{T+1} \mid y\_{1:T}, \theta \sim \text{Normal}(m_T, P_T + s^2) 

In these two formulas the second argument of the Normal distribution is a variance, while s and \sigma in the model of that section are standard deviations.

`resumed_ssm` is `gaussian_ssm` with this initial condition. The Fourier terms enter the observation model only, so the step into the new window reads no control.


``` python
def resumed_ssm(covariates: Array, data: Array | None = None) -> None:
    """Random walk level plus Fourier regression, resumed from the filtered state of a fit."""
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :1]  # the new observations, (t_obs, 1)
    controls = covariates[..., 1:]  # the Fourier terms of the new window, (duration, 52)
    bias = numpyro.sample("bias", dist.Normal(0.0, 10.0))
    weight = numpyro.sample(
        "weight", dist.Normal(0.0, 0.1).expand([controls.shape[-1]]).to_event(1)
    )
    drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-20.0, 5.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-5.0, 5.0))
    mean, cov = start_state("f", 1)  # the filtered level at the last week: (1,) and (1, 1)
    dynamics = DynamicalModel(
        # one random walk step into the new window: the same mean, one step more of variance
        initial_condition=dist.MultivariateNormal(mean, cov + drift_scale**2 * jnp.eye(1)),
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


The first stage, the filter of the history, needs no new computation here. The fit of *Local Level with a Kalman Filter* ran under the Kalman filter, and `dynestyx` recorded the filtered mean and variance of every week for every draw, as the sites `f_filtered_states_mean` and `f_filtered_states_cov`. [end_state](../../reference/contrib.dynestyx.end_state.md#numpyro_forecast.contrib.dynestyx.end_state) takes their last week, so no extra pass over the history is needed. We add the two sites that it returns to the parameter draws `kalman_parameters`.

`dynestyx` records such a site by default only while it has at most 100{,}000 elements per draw. Here each site has 417. A larger site needs `record_filtered_states_mean=True` and `record_filtered_states_cov=True` in the filter configuration.


``` python
# the parameter draws plus the two start sites f_start_mean and f_start_cov
posterior_resumed = kalman_parameters | end_state(mcmc_kalman.get_samples(), "f")

assert posterior_resumed["f_start_mean"].shape == (4_000, 1)
assert posterior_resumed["f_start_cov"].shape == (4_000, 1, 1)
```


Now we check that the resumed model gives the forecast of the whole window. Suppose that the first 8 weeks of the test year arrive. We forecast the remaining 44 weeks in two ways, with the same random key:

- **Whole window:** `forecast_whole` conditions `gaussian_ssm` on the window of 417 + 8 = 425 weeks.
- **Resumed:** `forecast_resumed` conditions `resumed_ssm` on the 8 new weeks only, and starts from the filtered state of the fit.

The figure shows the 50\\ and 94\\ HDI bands of the resumed forecast, the mean of the whole-window forecast, and the observed series from 26 weeks before the end of the training window. Its title gives the largest absolute difference between the draws of the two forecasts.


``` python
new_weeks = 8
origin = t_obs + new_weeks  # the first week of the forecast


def forecast_whole(rng_key: Array, new: Array) -> Array:
    """Forecast from ``origin``, conditioned on the training window and the weeks ``new``."""
    window = jnp.concatenate([bart_train[:, 0], new])  # (t_obs + new_weeks,)
    # column 0: the series, with the new weeks in place; columns 1: the Fourier terms
    covariates = bart_covariates.at[:origin, 0].set(window)  # (duration, 53)
    with DiscreteTimeSimulator(n_simulations=1), kalman:
        draws = forecast(rng_key, gaussian_ssm, kalman_parameters, window[:, None], covariates)
    return jnp.asarray(draws)  # (4_000, 44, 1)


def forecast_resumed(rng_key: Array, new: Array) -> Array:
    """Forecast from ``origin``, resumed from the filtered state with the weeks ``new`` only."""
    # the rows of the test year only; column 0: the new weeks, columns 1: the Fourier terms
    covariates = bart_covariates[t_obs:].at[:new_weeks, 0].set(new)  # (52, 53)
    with DiscreteTimeSimulator(n_simulations=1), kalman:
        draws = forecast(rng_key, resumed_ssm, posterior_resumed, new[:, None], covariates)
    return jnp.asarray(draws)  # (4_000, 44, 1)


# both forms with the same key and the 8 observed weeks, shape (new_weeks,)
rng_key, key_check = random.split(rng_key)
whole = forecast_whole(key_check, bart_test[:new_weeks, 0])
resumed = forecast_resumed(key_check, bart_test[:new_weeks, 0])

# the series is on the log scale, so 1e-2 is a relative difference of about 1% in ridership
largest_gap = float(jnp.abs(resumed - whole).max())
assert largest_gap < 1e-2

# the HDI bands of the resumed forecast
pc = az.plot_lm(
    predictions_to_datatree(np.asarray(resumed), week[origin:], ["bart"]),
    y="obs",
    x="t",
    plot_dim="time",
    ci_kind="hdi",
    ci_prob=hdi_probs,
    smooth=False,
    aes={"alpha": ["prob"]},
    alpha=hdi_alphas,
    visuals={"ci_band": {"color": "C1"}, "observed_scatter": False, "pe_line": False},
    figure_kwargs={"figsize": (12, 6)},
)

for prob in hdi_probs:
    pc.viz["ci_band"]["t"].sel(prob=prob).item().set_label(hdi_label(prob, "resumed forecast "))

# the mean of the whole-window forecast, the observed series and the ends of the two windows
ax = pc.viz["figure"].item().axes[0]
ax.plot(week[origin:], whole.mean(axis=0)[:, 0], color="C0", ls="--", label="whole-window mean")
history = slice(t_obs - 26, None)
ax.plot(week[history], bart[history, 0], color="black", lw=1, label="observed")
ax.axvline(t_obs, color="gray", ls="--", label="end of the training window")
ax.axvline(origin, color="gray", ls=":", label="end of the new weeks")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax.set(
    title=f"Resumed and whole-window forecasts: largest absolute difference {largest_gap:.1e}",
    xlabel="week",
    ylabel="log(# rides)",
);
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-36-output-1.png" class="figure-img" width="1211" height="611" /></p>
</figure>


The two forecasts coincide: the largest absolute difference between their draws is 9.5 \times 10^{-7} on the log scale, which is the rounding error of single precision. The dashed mean of the whole-window forecast runs through the middle of the resumed bands. There are two reasons for the agreement:

- The Kalman filter is exact for this model, so the filtered state at the last week of the training window holds all that the earlier observations say about the level.
- A start site that the posterior fills consumes no random number, so the two forms draw the same noise from the same key.

Next we measure what the resumed form saves.

Per draw, the resumed form runs 8 filter steps and the whole-window form 425. To measure the saving we time many forecasts. A plain [forecast](../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast) call compiles its program on every call, and that time would hide the difference. We therefore put each function through one `jax.jit(jax.vmap(...))`, which compiles once for all the forecasts of the same shapes, and we time the second call. Inside `jax.jit`, [forecast](../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast) needs `batch_size` and `device` at their defaults.

The forecasts are 32 **scenarios**. Each scenario is a possible first 8 weeks of the test year: one draw of the forecast of the fit from *In-Sample Predictive and Forecast*. Each scenario gets its own key. The figure shows the wall time of the 32 forecasts, with 4{,}000 draws each, for the two forms.


``` python
num_scenarios = 32
rng_key, key_scenarios = random.split(rng_key)
keys = random.split(key_scenarios, num_scenarios)

# every 125th forecast draw of the fit, cut to its first 8 test weeks
fit_forecast = stacked(trees["dynestyx Kalman smoother"], "predictions")  # (4_000, 52, 1)
stride = fit_forecast.shape[0] // num_scenarios
scenarios = jnp.asarray(fit_forecast[::stride, :new_weeks, 0])  # (32, new_weeks)

seconds_by_form = {}
draws_by_form = {}

for label, one_forecast in (("whole window", forecast_whole), ("resumed", forecast_resumed)):
    many_forecasts = jax.jit(jax.vmap(one_forecast))
    jax.block_until_ready(many_forecasts(keys, scenarios))  # the first call compiles
    start = perf_counter()
    draws_by_form[label] = jax.block_until_ready(many_forecasts(keys, scenarios))
    seconds_by_form[label] = perf_counter() - start

# (32, 4_000, 44, 1) for both forms
assert draws_by_form["whole window"].shape == draws_by_form["resumed"].shape

_, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(list(seconds_by_form), list(seconds_by_form.values()), color=["C0", "C1"])
ax.bar_label(bars, fmt="%.2f s")
ax.margins(y=0.15)
ratio = seconds_by_form["whole window"] / seconds_by_form["resumed"]
ax.set(
    title=f"{num_scenarios} forecasts in one compiled call: resumed is {ratio:.1f} times faster",
    xlabel="form of the forecast",
    ylabel="wall time (s)",
);
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-37-output-1.png" class="figure-img" width="1011" height="511" /></p>
</figure>


The 32 forecasts took 0.40 seconds in the whole-window form and 0.26 seconds in the resumed form, so the resumed form is 1.5 times faster. The saving is much smaller than the ratio of the filter steps, 425 against 8 per draw. The 417 filter steps that the resumed form skips cost the difference of 0.14 seconds. After the filter, both forms roll the level forward over the same 44 weeks and draw the observations, and this rollout is most of the time of the resumed form.

The recipe has these limits:

- **The step into the new window is the job of the model.** It has a closed form for a linear Gaussian evolution, as here. A nonlinear evolution needs a linearization.
- **An evolution that reads the control** needs the control of the last week of the earlier window, which the covariates of the new window do not hold.
- **The time grid restarts at 0** in the new window, so the dynamics must not read the absolute time.
- **The resumed model function is for the predictive drivers with a fixed posterior**: [forecast](../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast), [predict_in_sample](../../reference/predictive.predict_in_sample.md#numpyro_forecast.predictive.predict_in_sample) and [to_datatree](../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree). It is not for a fit, where the two start sites would be latent variables.
- **The helpers are tested with the Kalman filter and the extended Kalman filter** (`KFConfig`, `EKFConfig`); a `LatentPathBuilder` and a `Smoother` record no filtered state. A particle filter records the moments of its particles under the same site names, so resuming from them is a Gaussian approximation that is not tested.


# Conclusion


## Key Findings

- **One model function, any handler.** [state_space](../../reference/contrib.dynestyx.state_space.md#numpyro_forecast.contrib.dynestyx.state_space) turns a `DynamicalModel` into a `numpyro_forecast` model. The handler around each call selects the inference: a Kalman filter or smoother integrates the level out, a `LatentPathBuilder` samples it. [forecast](../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast), [predict_in_sample](../../reference/predictive.predict_in_sample.md#numpyro_forecast.predictive.predict_in_sample) and [to_datatree](../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) work unchanged with the handler around the call, and the block raises an error when the handler is missing.
- **The Kalman filter integrates the level out.** For the Normal model it gives the same posteriors and the same in-sample and test CRPS as [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations), with 55 parameters for NUTS instead of 472. A Kalman smoother exports the filter fit with [to_datatree](../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree), without a refit.
- **A filtered history is reused.** [end_state](../../reference/contrib.dynestyx.end_state.md#numpyro_forecast.contrib.dynestyx.end_state) and [start_state](../../reference/contrib.dynestyx.start_state.md#numpyro_forecast.contrib.dynestyx.start_state) resume the Kalman filter from the filtered level of the fit. The resumed forecast equals the whole-window forecast, and 32 forecasts in one compiled call were 1.5 times faster.
- **The explicit path covers Student-T noise and gaps.** Its posteriors and forecast match [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations), but it samples the states in centered form and needs longer chains. With 16{,}000 draws the fit on all weeks nearly converges (largest \hat{R} 1.012, just above the usual threshold of 1.01), and the fit through the gap converges (largest \hat{R} 1.01).
- **The Kalman filter gives the exact ARMA(1,1) likelihood**, up to a small jitter. Its posteriors are close to those of the [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) model of the ARMA example.


## Limitations

- The in-sample draws under a smoother are independent across weeks. For joint paths, use [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) or a `LatentPathBuilder`.
- The `LatentPathBuilder` has no reparameterization option, and every later call must use the builder instance of the fit. In an earlier run with `dynestyx` 0.8.0 and jax 0.11.2, parallel chains under it crashed the Python process. This run did not try them again.
- The block models one series per model function, with the step index as its time grid.


## Next Steps

The [advanced `dynestyx` example](dynestyx_integration.md) compares filtering and smoothing on a local level model and measures the cost of a full evaluation against a gradient. It also fits a seasonal model with SVI and backtests a `dynestyx` model with expanding windows.
