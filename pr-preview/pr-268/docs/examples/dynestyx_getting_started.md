# Getting Started with `dynestyx` State Space Models


Getting Started with `dynestyx` State Space Models

*This example reproduces the [ARMA](arma.md) and [comparing inference methods](inference_methods_comparison.md) examples with [`dynestyx`](https://github.com/BasisResearch/dynestyx) models. It is the entry point to `dynestyx` in `numpyro_forecast`. The [advanced example](dynestyx_integration.md) continues with filtering versus smoothing, the cost of a gradient, variational inference and backtesting.*

`dynestyx` is a probabilistic programming library for dynamical systems, built on NumPyro. A `DynamicalModel` holds an initial condition, a state evolution and an observation model. Effect handlers decide what `dynestyx.sample` does with it: a `Filter` or a `Smoother` integrates the latent path out and adds the marginal log likelihood to the NumPyro trace, a `LatentPathBuilder` samples the path, and a `Simulator` rolls the state forward in time.

The [`state_space`](../../reference/contrib.dynestyx.state_space.md) building block of `numpyro_forecast.contrib.dynestyx` (extra `dynestyx`) connects a `DynamicalModel` to the drivers of the package. In a model function `(covariates, data=None)`, it turns the [`Horizon`](../../reference/models.Horizon.md) into the time grids of `dynestyx.sample` and registers the `"obs"` and `"forecast"` sites that [`forecast`](../../reference/predictive.forecast.md), [`predict_in_sample`](../../reference/predictive.predict_in_sample.md) and [`to_datatree`](../../reference/convert.to_datatree.md) read. The block reads the observed series from the covariates, as [`ssoe`](../../reference/models.ssoe.md) does, because the drivers call the model without data for the in-sample predictive.

The `dynestyx` handler goes around each call that runs the model, as in the `dynestyx` examples. The same model function works under every handler, so the handler selects the inference strategy:

``` python
with Filter(filter_config=KFConfig()):
    mcmc.run(rng_key, covariates_train, data_train)

with DiscreteTimeSimulator(n_simulations=1), Smoother(smoother_config=KFSmootherConfig()):
    tree = to_datatree(rng_key, model, mcmc.get_samples(), data_train, covariates)
```

We proceed as follows:

1.  **ARMA(1,1).** We fit an ARMA model with [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) and as a state space model with the Kalman filter, which gives the exact likelihood.
2.  **Local level and Fourier regression.** We fit the BART weekly ridership model with [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) and with an explicit latent path from `dynestyx`, and forecast the last year with both.
3.  **Integrating out the level.** With Normal noise, the Kalman filter integrates the level out, so the sampler explores only the parameters.
4.  **In-sample posterior predictive.** We export the Kalman model with [to_datatree](../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) under a Kalman smoother, without a refit.
5.  **Missing data.** We fit the explicit-path model through a gap of 20 weeks.
6.  **Sampler cost.** We compare the cost of the three pairs of fits.


# Prepare Notebook


``` python
import numpyro

numpyro.set_host_device_count(n=4)  # before any import that initializes jax

from time import perf_counter
from typing import Literal

import arviz as az
import dynestyx as dsx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import xarray as xr
from dynestyx import DiscreteTimeSimulator, DynamicalModel, Filter, LatentPathBuilder, Smoother
from dynestyx.inference.configs.smoother import KFSmootherConfig
from dynestyx.inference.filters import KFConfig
from dynestyx.models import LinearGaussianObservation, LinearGaussianStateEvolution
from jax import random
from jax.flatten_util import ravel_pytree
from jaxtyping import Float
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import LocScaleReparam
from numpyro.infer.util import initialize_model

from numpyro_forecast import (
    Horizon,
    eval_crps,
    forecast,
    innovations,
    predict,
    predict_in_sample,
    ssoe,
    to_datatree,
)
from numpyro_forecast.contrib.dynestyx import state_space
from numpyro_forecast.datasets import load_bart_weekly
from numpyro_forecast.features import fourier_features
from numpyro_forecast.typing import Array, ForecastModel

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


Every comparison fits both models with the same NUTS settings: 4 chains of 1{,}000 draws, the same warmup length and the same chain method. The helpers below run a fit, measure the cost of one gradient of the log density, and collect the diagnostics.

- `fit_nuts` returns the sampler and its wall time, which includes the compilation.
- `gradient_cost` measures the gradient of the potential energy, which NUTS evaluates at every leapfrog step. It evaluates 1{,}000 gradients at nearby points inside one compiled loop and returns the milliseconds per gradient, without the compilation and without the Python overhead of one call per gradient.
- `convergence` reports, over the scalar parameters, the smallest bulk effective sample size (ESS), the largest \hat{R} and the number of divergent transitions.
- `diagnostics` adds the wall time, the leapfrog steps per draw and the gradient evaluations of the sampling phase. The ratio of gradient evaluations to the smallest ESS measures how hard the posterior is for NUTS.


``` python
def fit_nuts(
    rng_key: Array,
    model: ForecastModel,
    covariates: Array,
    data: Array,
    *,
    num_warmup: int,
    chain_method: Literal["parallel", "sequential", "vectorized"],
) -> tuple[MCMC, float]:
    """Fit ``model`` with NUTS (4 chains, 1,000 draws each) and return the wall time in seconds."""
    mcmc = MCMC(
        NUTS(model),
        num_warmup=num_warmup,
        num_samples=1_000,
        num_chains=4,
        chain_method=chain_method,
        progress_bar=False,
    )
    start = perf_counter()
    mcmc.run(rng_key, covariates, data, extra_fields=("num_steps", "diverging"))
    jax.block_until_ready(mcmc.get_samples())
    return mcmc, perf_counter() - start


def gradient_cost(
    rng_key: Array, model: ForecastModel, covariates: Array, data: Array, num_evals: int = 1_000
) -> float:
    """Milliseconds per gradient of the potential energy of ``model``, in one compiled loop."""
    model_info = initialize_model(rng_key, model, model_args=(covariates, data))
    flat, unravel = ravel_pytree(model_info.param_info.z)
    # distinct points, so that the compiler cannot reuse one gradient for every iteration
    points = flat + 1e-3 * random.normal(rng_key, (num_evals, flat.size))
    value_and_grad = jax.value_and_grad(lambda z: model_info.potential_fn(unravel(z)))
    run = jax.jit(lambda z: jax.lax.map(value_and_grad, z))
    jax.block_until_ready(run(points))  # compile
    start = perf_counter()
    jax.block_until_ready(run(points))
    return (perf_counter() - start) / num_evals * 1e3


def posterior_tree(mcmc: MCMC, names: list[str]) -> xr.DataTree:
    """Return the ``(chain, draw)`` posterior of the scalar parameters ``names`` as a DataTree."""
    grouped = mcmc.get_samples(group_by_chain=True)
    return az.from_dict({"posterior": {name: np.asarray(grouped[name]) for name in names}})


def convergence(mcmc: MCMC, names: list[str]) -> dict[str, float]:
    """Smallest bulk ESS, largest R-hat and divergent transitions over the parameters ``names``."""
    summary = az.summary(posterior_tree(mcmc, names), var_names=names, round_to="none")
    return {
        "min bulk ESS": float(summary["ess_bulk"].min()),
        "max R-hat": float(summary["r_hat"].max()),
        "divergences": float(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
    }


def diagnostics(
    mcmc: MCMC, seconds: float, ms_per_gradient: float, names: list[str]
) -> dict[str, float]:
    """Sampler cost and convergence of one fit over the scalar parameters ``names``."""
    steps = np.asarray(mcmc.get_extra_fields()["num_steps"])
    checks = convergence(mcmc, names)
    return {
        "wall time (s)": seconds,
        "leapfrog steps per draw": float(steps.mean()),
        "gradient evaluations": float(steps.sum()),
        **checks,
        "ms per gradient": ms_per_gradient,
        "gradients per effective draw": float(steps.sum()) / checks["min bulk ESS"],
    }


def cost_table(cost: dict[str, dict[str, float]]) -> pd.DataFrame:
    """One row per fit, rounded per column, with the counts as integers."""
    digits = {"wall time (s)": 1, "leapfrog steps per draw": 1, "max R-hat": 3}
    digits |= {"ms per gradient": 4, "gradients per effective draw": 1}
    counts = {"gradient evaluations": int, "min bulk ESS": int, "divergences": int}
    return pd.DataFrame(cost).T.round(digits | dict.fromkeys(counts, 0)).astype(counts)
```


Three plotting helpers: one overlays the posteriors of two fits, one parameter per panel, one computes the HDI bounds of forecast draws, and one writes the legend label of an HDI band.


``` python
def plot_posteriors(
    fits: dict[str, MCMC], names: list[str], title: str, truth: dict[str, float] | None = None
) -> None:
    """Overlay the posterior densities of ``names`` for every fit, one panel per parameter."""
    trees = {label: posterior_tree(mcmc, names) for label, mcmc in fits.items()}
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


def hdi_bounds(
    draws: Float[Array | np.ndarray, " sample time 1"], prob: float
) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper HDI bounds over the sample axis of ``(sample, time, 1)`` draws."""
    da = xr.DataArray(np.asarray(draws)[..., 0], dims=["sample", "time"])
    bounds = az.hdi(da, prob=prob, dim="sample")
    return bounds.sel(ci_bound="lower").to_numpy(), bounds.sel(ci_bound="upper").to_numpy()


def hdi_label(prob: float, prefix: str = "") -> str:
    r"""Legend label for an HDI band, e.g. ``$94\%$ HDI``."""
    percent = f"{prob:.0%}".replace("%", r"\%")
    return f"{prefix}${percent}$ HDI"
```


# ARMA(1,1)

The [ARMA example](arma.md) writes the ARMA(1,1) model

 y_t = \mu + \phi \\ y\_{t-1} + \theta \\ \varepsilon\_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim \text{Normal}(0, \sigma) 

with [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe), which conditions on y\_{-1} = \mu and \varepsilon\_{-1} = 0. The same model is a linear-Gaussian state space model, so the Kalman filter gives the exact likelihood instead. With the deviation a_t = y_t - c from the process mean c = \mu / (1 - \phi), the state x_t = (a_t, \theta \varepsilon_t) follows

 x_t = A x\_{t-1} + g \\ \varepsilon_t, \qquad A = \begin{pmatrix} \phi & 1 \\ 0 & 0 \end{pmatrix}, \qquad g = \begin{pmatrix} 1 \\ \theta \end{pmatrix}, 

so the state covariance is Q = \sigma^2 g g^\top, and the observation is y_t = H x_t + c with H = (1, 0). The initial state has the stationary covariance P_0, the solution of P_0 = A P_0 A^\top + Q. The state noise has rank one, so we add a small jitter \delta = 10^{-4} \sigma^2 to Q, to P_0 and as the observation variance, which keeps the Cholesky factors defined in single precision.

We simulate 100 steps with \phi = 0.4, \theta = 0.7 and \sigma = 0.5. The priors are those of the ARMA example, which also checks them with prior predictive samples.


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


The `numpyro_forecast` model, as in the ARMA example:


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


The `dynestyx` model has the same priors. `arma_dynamics` builds the matrices above, and the model function reads the series from the covariates and hands the [Horizon](../../reference/models.Horizon.md#numpyro_forecast.models.Horizon), the window and the dynamics to [state_space](../../reference/contrib.dynestyx.state_space.md#numpyro_forecast.contrib.dynestyx.state_space).


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


def arma_ssm(covariates: Array, data: Array | None = None) -> None:
    """ARMA(1,1) as a dynestyx state space model."""
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :]  # the observed window travels in the covariates
    mu = numpyro.sample("mu", dist.Normal(loc=0, scale=1))
    phi = numpyro.sample("phi", dist.Uniform(low=-1, high=1))
    theta = numpyro.sample("theta", dist.Uniform(low=-1, high=1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=1))
    state_space(h, "f", y, arma_dynamics(mu, phi, theta, sigma))
```


We fit both models with parallel chains. The `dynestyx` fit runs under a Kalman filter on the `cd_dynamax` backend. We pass `KFConfig` explicitly: without a configuration, `Filter()` uses an ensemble Kalman filter, which is approximate.


``` python
arma_names = ["mu", "phi", "theta", "sigma"]
kalman = Filter(filter_config=KFConfig(filter_source="cd_dynamax"))
rng_key, key_ssoe, key_kalman, key_grad = random.split(rng_key, 4)

mcmc_arma_ssoe, seconds = fit_nuts(
    key_ssoe, arma_ssoe, arma_covariates, arma_data, num_warmup=1_000, chain_method="parallel"
)
ms = gradient_cost(key_grad, arma_ssoe, arma_covariates, arma_data)
arma_cost = {"ssoe": diagnostics(mcmc_arma_ssoe, seconds, ms, arma_names)}

with kalman:
    mcmc_arma_kalman, seconds = fit_nuts(
        key_kalman, arma_ssm, arma_covariates, arma_data, num_warmup=1_000, chain_method="parallel"
    )
    ms = gradient_cost(key_grad, arma_ssm, arma_covariates, arma_data)

arma_cost["dynestyx Kalman filter"] = diagnostics(mcmc_arma_kalman, seconds, ms, arma_names)

cost_table(arma_cost)
```


|  | wall time (s) | leapfrog steps per draw | gradient evaluations | min bulk ESS | max R-hat | divergences | ms per gradient | gradients per effective draw |
|----|----|----|----|----|----|----|----|----|
| ssoe | 1.5 | 7.8 | 31264 | 2206 | 1.002 | 0 | 0.0055 | 14.2 |
| dynestyx Kalman filter | 18.9 | 7.5 | 29844 | 1866 | 1.002 | 0 | 0.7659 | 16.0 |


Both fits converge: the largest \hat{R} is 1.002, the smallest bulk ESS is above 1{,}800 of the 4{,}000 draws, and there are no divergent transitions. Neither model samples a path, so NUTS takes about the same number of leapfrog steps per draw (7.8 and 7.5). A gradient of the Kalman filter costs about 140 times more (0.7659 against 0.0055 milliseconds), because every step propagates and updates a 2 \times 2 state covariance, where the [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) recursion is one multiply-add. The wall time (1.5 against 18.9 seconds) also includes the warmup and the compilation.


``` python
plot_posteriors(
    {"ssoe": mcmc_arma_ssoe, "dynestyx Kalman filter": mcmc_arma_kalman},
    arma_names,
    title="ARMA(1,1) posteriors (dashed: true value)",
    truth={"mu": 0.0, "phi": phi_true, "theta": theta_true, "sigma": noise_scale},
)
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-9-output-1.png" class="figure-img" width="1206" height="361" /></p>
</figure>


The two posteriors agree. The true values of \mu, \phi and \theta lie inside the bulk of both, and the true \sigma = 0.5 lies in the upper tail of both, so that difference comes from the simulated data, not from the method. The [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) likelihood conditions on the first observation and the Kalman filter likelihood is exact, but with 100 observations the difference does not show.


# Local Level and Fourier Regression

The [comparing inference methods](inference_methods_comparison.md) example fits a random walk level plus a Fourier regression with Student-T noise to the weekly BART ridership:

 \begin{align\*} \ell_t &= \ell\_{t-1} + w_t, \qquad w_t \sim \text{Normal}(0, s), \\ y_t &= b + \ell_t + \beta^\top u_t + v_t, \qquad v_t \sim \text{StudentT}(\nu, 0, \sigma), \end{align\*} 

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


The `numpyro_forecast` model samples the 417 weekly steps of the level with [`innovations`](../../reference/models.innovations.md). The comparing inference methods example also samples a `centered` site for its `LocScaleReparam`. That site only selects a parameterization, so its posterior equals its prior and NUTS mixes poorly on it. Here we fix the non-centered form instead.


``` python
def univariate_model(covariates: Array, data: Array | None = None) -> None:
    """Random walk level plus Fourier regression with Student-T noise, with ``innovations``."""
    h = Horizon.from_data(covariates, data)
    fourier = covariates[..., 1:]  # column 0 is the series, read only by the dynestyx model
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


Student-T noise rules out an exact Kalman filter. A particle filter does not combine well with NUTS here: with resampling, the likelihood estimate is not continuous in the parameters, and without it the weights collapse over 417 weeks. The `dynestyx` model therefore samples the level with a `LatentPathBuilder`, as [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) does. The observation model is a function of the state x_t, the controls u_t and the time t, and the Fourier terms enter as controls over the whole horizon, so the forecast reads their future rows from the covariates.

The builder samples the states x_t = b + \ell_t directly (the centered form), not their steps. With the intercept b as a separate parameter, a change of b must move all 417 states the other way. In this form the intercept is the initial state, so the `dynestyx` model has no `bias` site and gives the initial state the \text{Normal}(0, 10) prior of the intercept. The two models differ only in the variance s^2 that the first step adds to this prior, which is negligible.


``` python
def level_regression(covariates: Array, data: Array | None = None) -> None:
    """Random walk level plus Fourier regression with Student-T noise, as a dynestyx model."""
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :1]  # the observed window, (t_obs, 1)
    controls = covariates[..., 1:]  # the Fourier terms over the whole horizon, (duration, 52)
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


Both fits run their chains one after the other (`chain_method="sequential"`). Parallel chains under a `LatentPathBuilder` crashed the Python process with `dynestyx` 0.8.0 on the machine that executed this notebook, and a fair wall time needs the same chain method on both sides.

We keep the `LatentPathBuilder` instance of the fit. The builder records the pattern of missing observations from the concrete data of the fit, and the drivers later run the model under `jax.jit`, where the data are traced, so every later call must use the same instance.


``` python
bart_names = ["drift_scale", "nu", "sigma"]  # the parameters both models share
latent_path = LatentPathBuilder()
rng_key, key_innovations, key_latent, key_grad = random.split(rng_key, 4)

mcmc_innovations, seconds = fit_nuts(
    key_innovations,
    univariate_model,
    bart_covariates[:t_obs],
    bart_train,
    num_warmup=2_000,
    chain_method="sequential",
)
ms = gradient_cost(key_grad, univariate_model, bart_covariates[:t_obs], bart_train)
bart_cost = {"innovations": diagnostics(mcmc_innovations, seconds, ms, bart_names)}

with latent_path:
    mcmc_latent, seconds = fit_nuts(
        key_latent,
        level_regression,
        bart_covariates[:t_obs],
        bart_train,
        num_warmup=2_000,
        chain_method="sequential",
    )
    ms = gradient_cost(key_grad, level_regression, bart_covariates[:t_obs], bart_train)

bart_cost["dynestyx latent path"] = diagnostics(mcmc_latent, seconds, ms, bart_names)

cost_table(bart_cost)
```


|  | wall time (s) | leapfrog steps per draw | gradient evaluations | min bulk ESS | max R-hat | divergences | ms per gradient | gradients per effective draw |
|----|----|----|----|----|----|----|----|----|
| innovations | 111.7 | 255.0 | 1020000 | 1206 | 1.002 | 0 | 0.0191 | 845.8 |
| dynestyx latent path | 20.6 | 107.0 | 427904 | 117 | 1.057 | 0 | 0.0161 | 3646.5 |


The explicit path takes fewer leapfrog steps per draw than [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) (107 against 255), its gradients are slightly cheaper (0.0161 against 0.0191 milliseconds), and it needs less than a fifth of the wall time (20.6 against 111.7 seconds). But it does not converge as well. Its largest \hat{R} is 1.057, above the usual threshold of 1.01, and its smallest bulk ESS is 117, so per effective draw it needs about four times as many gradients (3{,}646 against 846). Neither fit has divergent transitions. The diagnostics per parameter show which parameter is slow:


``` python
pd.concat(
    {
        label: az.summary(posterior_tree(mcmc, bart_names), var_names=bart_names, round_to="none")
        for label, mcmc in (
            ("innovations", mcmc_innovations),
            ("dynestyx latent path", mcmc_latent),
        )
    },
    names=["fit", "parameter"],
)[["mean", "sd", "ess_bulk", "r_hat"]].round({"mean": 4, "sd": 4, "ess_bulk": 0, "r_hat": 3})
```


<table class="dataframe table table-sm table-striped small">
<thead>
<tr>
<th></th>
<th></th>
<th>mean</th>
<th>sd</th>
<th>ess_bulk</th>
<th>r_hat</th>
</tr>
<tr>
<th>fit</th>
<th>parameter</th>
<th></th>
<th></th>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<th rowspan="3" data-valign="top">innovations</th>
<th>drift_scale</th>
<td>0.0044</td>
<td>0.0007</td>
<td>1372.0</td>
<td>1.000</td>
</tr>
<tr>
<th>nu</th>
<td>1.5781</td>
<td>0.2211</td>
<td>1873.0</td>
<td>1.001</td>
</tr>
<tr>
<th>sigma</th>
<td>0.0173</td>
<td>0.0020</td>
<td>1206.0</td>
<td>1.002</td>
</tr>
<tr>
<th rowspan="3" data-valign="top">dynestyx latent path</th>
<th>drift_scale</th>
<td>0.0047</td>
<td>0.0008</td>
<td>117.0</td>
<td>1.057</td>
</tr>
<tr>
<th>nu</th>
<td>1.5632</td>
<td>0.2231</td>
<td>2633.0</td>
<td>1.002</td>
</tr>
<tr>
<th>sigma</th>
<td>0.0171</td>
<td>0.0020</td>
<td>1954.0</td>
<td>1.003</td>
</tr>
</tbody>
</table>


`drift_scale`, the scale of the weekly steps of the level, is the slow parameter of the explicit path: its bulk ESS is 117, against 1{,}372 under [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations), while `nu` and `sigma` mix well under both. The builder samples the states in centered form and has no reparameterization option, while [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) samples the non-centered steps, so the leapfrog counts compare two parameterizations as well as two libraries. The summaries of the explicit path are approximate, and longer chains would make them more reliable.


``` python
plot_posteriors(
    {"innovations": mcmc_innovations, "dynestyx latent path": mcmc_latent},
    bart_names,
    title="Local level and Fourier regression posteriors",
)
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-15-output-1.png" class="figure-img" width="1206" height="361" /></p>
</figure>


The densities of `nu` and `sigma` coincide. The `drift_scale` density of the explicit path sits a little to the right (posterior mean 0.0047 against 0.0044). That fit has not converged on `drift_scale`, so the shift may come from its slow mixing.

Both models forecast through [forecast](../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast). For the `dynestyx` model, the block reads the forecast from the rollout of a `DiscreteTimeSimulator` around the call. `dynestyx` places the initial state of the rollout at the first prediction time, so the block asks for one extra step at the last observation and drops it: the first forecast week is a transition of the final level, not a copy of it. The `LatentPathBuilder` around the call is the instance that fitted the model.


``` python
rng_key, key_innovations, key_latent = random.split(rng_key, 3)
bart_forecasts = {
    "innovations": forecast(
        key_innovations,
        univariate_model,
        mcmc_innovations.get_samples(),
        bart_train,
        bart_covariates,
        batch_size=500,
    )
}

with DiscreteTimeSimulator(n_simulations=1), latent_path:
    bart_forecasts["dynestyx latent path"] = forecast(
        key_latent,
        level_regression,
        mcmc_latent.get_samples(),
        bart_train,
        bart_covariates,
        batch_size=500,
    )

pd.Series(
    {label: float(eval_crps(draws, bart_test)) for label, draws in bart_forecasts.items()},
    name="test CRPS",
).round(4)
```


    innovations             0.0301
    dynestyx latent path    0.0303
    Name: test CRPS, dtype: float64


We plot the `dynestyx` forecast with its 50\\ and 94\\ HDI bands, and the [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) forecast as its median and the bounds of its 50\\ and 94\\ HDI bands. We show medians, not means: with \nu close to 1.5 the Student-T noise has no finite variance, so the mean of the draws moves with a few extreme draws.


``` python
fig, ax = plt.subplots(figsize=(12, 6))
history = slice(t_obs - 104, t_obs)
draws_latent = bart_forecasts["dynestyx latent path"]
draws_innovations = bart_forecasts["innovations"]

for prob, alpha, ls in ((0.94, 0.3, ":"), (0.5, 0.5, "-.")):
    lower, upper = hdi_bounds(draws_latent, prob)
    ax.fill_between(
        week[t_obs:],
        lower,
        upper,
        color="C1",
        alpha=alpha,
        label=hdi_label(prob, prefix="dynestyx latent path "),
    )
    lower, upper = hdi_bounds(draws_innovations, prob)
    ax.plot(
        week[t_obs:], lower, color="C0", ls=ls, label=hdi_label(prob, "innovations ") + " bounds"
    )
    ax.plot(week[t_obs:], upper, color="C0", ls=ls)

ax.plot(
    week[t_obs:],
    np.median(np.asarray(draws_latent), axis=0)[:, 0],
    color="C1",
    label="dynestyx latent path median",
)
ax.plot(
    week[t_obs:],
    np.median(np.asarray(draws_innovations), axis=0)[:, 0],
    color="C0",
    ls="--",
    label="innovations median",
)
ax.plot(week[history], bart_train[history, 0], color="black", lw=1, label="train")
ax.plot(week[t_obs:], bart_test[:, 0], color="black", lw=1, ls="--", label="test")
ax.axvline(t_obs, color="gray", ls="--", lw=1)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=3)
ax.set(
    title="BART weekly ridership: forecast of the last 52 weeks",
    xlabel="week",
    ylabel="log(# rides)",
);
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-17-output-1.png" class="figure-img" width="1211" height="611" /></p>
</figure>


The two forecasts agree. The test CRPS is 0.0301 for [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) and 0.0303 for the explicit path, and the medians and the 94\\ HDI bounds of the two models lie on top of each other over the whole test year. The slow mixing of `drift_scale` does not show in the forecast.


# Integrating Out the Level

With Normal noise the same model is linear and Gaussian. The Kalman filter then integrates the level out exactly, and NUTS explores the 55 parameters only, not the 417 weekly steps of the level as well. In `dynestyx` terms, the regression is the control matrix `D` of the observation model and the bias is its offset.


``` python
def gaussian_model(covariates: Array, data: Array | None = None) -> None:
    """Random walk level plus Fourier regression with Normal noise, with ``innovations``."""
    h = Horizon.from_data(covariates, data)
    fourier = covariates[..., 1:]
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


def gaussian_ssm(covariates: Array, data: Array | None = None) -> None:
    """Random walk level plus Fourier regression with Normal noise, as a dynestyx model."""
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :1]
    controls = covariates[..., 1:]
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


Both fits run parallel chains, which the Kalman filter supports. This model keeps the `bias` site of the [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) model: the filter integrates the states out, so no sampled path couples the intercept to the states.


``` python
gaussian_names = ["bias", "drift_scale", "sigma"]
rng_key, key_gaussian, key_gaussian_kalman, key_grad = random.split(rng_key, 4)

mcmc_gaussian, seconds = fit_nuts(
    key_gaussian,
    gaussian_model,
    bart_covariates[:t_obs],
    bart_train,
    num_warmup=2_000,
    chain_method="parallel",
)
ms = gradient_cost(key_grad, gaussian_model, bart_covariates[:t_obs], bart_train)
gaussian_cost = {"innovations": diagnostics(mcmc_gaussian, seconds, ms, gaussian_names)}

with kalman:
    mcmc_gaussian_kalman, seconds = fit_nuts(
        key_gaussian_kalman,
        gaussian_ssm,
        bart_covariates[:t_obs],
        bart_train,
        num_warmup=2_000,
        chain_method="parallel",
    )
    ms = gradient_cost(key_grad, gaussian_ssm, bart_covariates[:t_obs], bart_train)

gaussian_cost["dynestyx Kalman filter"] = diagnostics(
    mcmc_gaussian_kalman, seconds, ms, gaussian_names
)

cost_table(gaussian_cost)
```


|  | wall time (s) | leapfrog steps per draw | gradient evaluations | min bulk ESS | max R-hat | divergences | ms per gradient | gradients per effective draw |
|----|----|----|----|----|----|----|----|----|
| innovations | 26.6 | 114.5 | 458105 | 1599 | 1.001 | 11 | 0.0163 | 286.4 |
| dynestyx Kalman filter | 101.2 | 7.8 | 31272 | 4234 | 1.002 | 0 | 1.3012 | 7.4 |


The largest \hat{R} is 1.002 and the smallest bulk ESS is 1{,}599. The [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) fit reports 11 divergent transitions in 4{,}000 draws, a sign that it does not explore every region of the posterior; the Kalman filter reports none. The Kalman filter needs 7.8 leapfrog steps per draw against 114.5, and about 40 times fewer gradients per effective draw (7.4 against 286.4). Each of its gradients runs the filter over the 417 weeks and costs about 80 times more (1.3012 against 0.0163 milliseconds). The Sampler Cost section compares the two effects. The wall time, which includes the warmup and the compilation, is longer for the filter (101.2 against 26.6 seconds).


``` python
plot_posteriors(
    {"innovations": mcmc_gaussian, "dynestyx Kalman filter": mcmc_gaussian_kalman},
    gaussian_names,
    title="Gaussian local level and Fourier regression posteriors",
)
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-20-output-1.png" class="figure-img" width="1206" height="361" /></p>
</figure>


The posteriors agree for the three parameters, `bias` included.


# In-Sample Posterior Predictive

[predict_in_sample](../../reference/predictive.predict_in_sample.md#numpyro_forecast.predictive.predict_in_sample) and [to_datatree](../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) call the model without data and read the `"obs"` site. For a `dynestyx` model, the block needs the distribution of the states given the whole window, p(x_t \mid y\_{1:T}, \theta), which a `Filter` does not have: it carries the filtering distribution p(x_t \mid y\_{1:t}, \theta) only. A Kalman smoother has the smoothing distribution, and it adds the same marginal likelihood to the trace as the Kalman filter. The draws of the filter fit above are therefore draws of the posterior under the smoother too. We export them under the smoother, with the `Simulator` outside it for the forecast, and do not refit.

The block draws every in-sample state from its smoothing distribution and pushes it through the observation model. These draws are exact for every week, so the bands and the CRPS below are correct, but they are independent across weeks, not a joint path. [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) and `LatentPathBuilder` replay a joint posterior path instead.


``` python
smoother = Smoother(smoother_config=KFSmootherConfig(filter_source="cd_dynamax"))
# the parameters only: the filter fit also recorded its filtered states (f_filtered_states_*),
# which the export under the smoother does not need
kalman_parameters = {
    name: value
    for name, value in mcmc_gaussian_kalman.get_samples().items()
    if not name.startswith("f_")
}
rng_key, key_tree, key_tree_kalman = random.split(rng_key, 3)
trees = {
    "innovations": to_datatree(
        key_tree,
        gaussian_model,
        mcmc_gaussian.get_samples(),
        bart_train,
        bart_covariates,
        num_chains=4,
        batch_size=500,
    )
}

with DiscreteTimeSimulator(n_simulations=1), smoother:
    trees["dynestyx Kalman smoother"] = to_datatree(
        key_tree_kalman,
        gaussian_ssm,
        kalman_parameters,
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
│           bias          (chain, draw) float32 16kB 14.52 14.52 14.51 ... 14.51 14.5
│           drift_scale   (chain, draw) float32 16kB 0.004938 0.005319 ... 0.005393
│           sigma         (chain, draw) float32 16kB 0.08004 0.07986 ... 0.07506 0.07872
│           weight        (chain, draw, weight_dim_0) float32 832kB -0.005858 ... -0....
│       Attributes:
│           created_at:                 2026-11-03T11:41:06.391198+00:00
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
│           obs      (chain, draw, time, obs_dim) float32 7MB 14.33 14.52 ... 14.31
│       Attributes:
│           created_at:                 2026-11-03T11:41:12.208919+00:00
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
│           created_at:                 2026-11-03T11:41:12.211769+00:00
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
│           created_at:                 2026-11-03T11:41:12.212560+00:00
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
│           obs      (chain, draw, time, obs_dim) float32 832kB 14.36 14.74 ... 14.25
│       Attributes:
│           created_at:                 2026-11-03T11:41:15.142764+00:00
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
            created_at:                 2026-11-03T11:41:15.145473+00:00
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


14.52 14.52 14.51 ... 14.51 14.5


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[14.519246 , 14.516442 , 14.512387 , ..., 14.519242 , 14.506527 ,14.4972925],[14.511379 , 14.527223 , 14.54096  , ..., 14.507469 , 14.542439 ,14.495682 ],[14.47496  , 14.548789 , 14.491268 , ..., 14.532587 , 14.525666 ,14.545436 ],[14.510921 , 14.4973755, 14.53815  , ..., 14.513752 , 14.512789 ,14.500986 ]], shape=(4, 1000), dtype=float32)


drift_scale


(chain, draw)


float32


0.004938 0.005319 ... 0.005393


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.0049378 , 0.00531932, 0.00644292, ..., 0.00643193, 0.00671805,0.00503278],[0.00441848, 0.00717565, 0.006998  , ..., 0.00546303, 0.00861842,0.00481792],[0.00500589, 0.00599688, 0.00582266, ..., 0.00694658, 0.00543406,0.0067784 ],[0.00696305, 0.00513607, 0.0061319 , ..., 0.00688266, 0.00599816,0.00539256]], shape=(4, 1000), dtype=float32)


sigma


(chain, draw)


float32


0.08004 0.07986 ... 0.07506 0.07872


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.08003634, 0.07986301, 0.07670652, ..., 0.07465477, 0.07548851,0.07964594],[0.08016146, 0.07935506, 0.08037973, ..., 0.08287556, 0.078803  ,0.0806475 ],[0.07431714, 0.07471378, 0.07792508, ..., 0.07914142, 0.07836846,0.07566297],[0.07389297, 0.08280987, 0.07144713, ..., 0.08095629, 0.07505613,0.0787205 ]], shape=(4, 1000), dtype=float32)


weight


(chain, draw, weight_dim_0)


float32


-0.005858 0.0076 ... -0.007153


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[-0.00585787,  0.00760042,  0.00866246, ..., -0.00868301,-0.00612059, -0.01878156],[-0.00580781,  0.01005891,  0.01026686, ..., -0.001551  ,-0.01373828, -0.00755534],[ 0.00167421,  0.02422465,  0.02194338, ..., -0.01378076,0.00706787, -0.00563175],...,[-0.01349587,  0.02493137,  0.00547346, ..., -0.01025648,-0.00369903, -0.01810753],[-0.00983371,  0.02284557,  0.00629351, ..., -0.00566878,-0.00312235, -0.01461021],[ 0.00235304,  0.00587661,  0.02077279, ..., -0.01061299,-0.00383982, -0.00572169]],[[-0.01299441,  0.02179356,  0.01422252, ..., -0.00198424,-0.00559818, -0.00322273],[-0.00789142,  0.01992228,  0.01221653, ..., -0.00566545,-0.00846449, -0.00485475],[-0.00396394,  0.02864434,  0.01848001, ..., -0.01109787,0.00040784,  0.00039967],...[ 0.00904724,  0.01019117,  0.00500332, ..., -0.00146126,-0.01135969, -0.001834  ],[-0.00255136,  0.01602039,  0.0144409 , ...,  0.00317105,-0.00063728,  0.00953371],[-0.00839752,  0.02023406,  0.00725985, ..., -0.01446875,-0.00181892, -0.0075899 ]],[[-0.00364783,  0.01270764,  0.01361839, ..., -0.00100669,-0.00024967, -0.0033693 ],[ 0.00346355,  0.02242982,  0.01731548, ..., -0.01472155,-0.00508634, -0.00936676],[-0.01058297,  0.01127716,  0.00972288, ...,  0.00220954,-0.00117612, -0.01438886],...,[-0.01001048,  0.01701409,  0.01443807, ..., -0.02229672,-0.00619818, -0.01313522],[-0.00351085,  0.00038952,  0.0109113 , ..., -0.00527818,-0.00267419, -0.00335253],[-0.00544331,  0.00663255,  0.01134535, ..., -0.00257967,-0.00312361, -0.00715288]]], shape=(4, 1000, 52), dtype=float32)


Attributes: (5)


created_at :  
2026-11-03T11:41:06.391198+00:00

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


14.33 14.52 14.34 ... 14.68 14.31


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[14.331059 ],[14.516977 ],[14.34348  ],...,[14.68126  ],[14.574733 ],[14.3008375]],[[14.608049 ],[14.517512 ],[14.312527 ],...,[14.752906 ],[14.506334 ],[14.060438 ]],[[14.326659 ],[14.331408 ],[14.54306  ],...,......,[14.709993 ],[14.580296 ],[14.11863  ]],[[14.470848 ],[14.56616  ],[14.417576 ],...,[14.701998 ],[14.542353 ],[14.027803 ]],[[14.442721 ],[14.459551 ],[14.491458 ],...,[14.53258  ],[14.684449 ],[14.314935 ]]]], shape=(4, 1000, 417, 1), dtype=float32)


Attributes: (5)


created_at :  
2026-11-03T11:41:12.208919+00:00

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
2026-11-03T11:41:12.211769+00:00

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
2026-11-03T11:41:12.212560+00:00

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


14.36 14.74 14.51 ... 14.74 14.25


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[14.362419 ],[14.743162 ],[14.513835 ],...,[14.725538 ],[14.673572 ],[14.421175 ]],[[14.318442 ],[14.711215 ],[14.654705 ],...,[14.561908 ],[14.833518 ],[14.308677 ]],[[14.459128 ],[14.640042 ],[14.411572 ],...,......,[14.5847645],[14.688045 ],[14.413139 ]],[[14.424269 ],[14.509407 ],[14.452668 ],...,[14.613519 ],[14.651874 ],[14.191654 ]],[[14.296536 ],[14.537654 ],[14.442961 ],...,[14.706479 ],[14.738222 ],[14.254447 ]]]], shape=(4, 1000, 52, 1), dtype=float32)


Attributes: (5)


created_at :  
2026-11-03T11:41:15.142764+00:00

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
2026-11-03T11:41:15.145473+00:00

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


Both trees carry the `posterior_predictive` group (the in-sample draws) and the `predictions` group (the forecast). We merge the `(chain, draw)` dimensions into one sample axis to score and plot them.


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


pd.DataFrame(
    {
        "in-sample CRPS": {
            label: float(eval_crps(stacked(tree, "posterior_predictive"), bart_train))
            for label, tree in trees.items()
        },
        "test CRPS": {
            label: float(eval_crps(stacked(tree, "predictions"), bart_test))
            for label, tree in trees.items()
        },
    }
).round(4)
```


|                          | in-sample CRPS | test CRPS |
|--------------------------|----------------|-----------|
| innovations              | 0.0316         | 0.0362    |
| dynestyx Kalman smoother | 0.0315         | 0.0363    |


The in-sample CRPS is 0.0316 for [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) and 0.0315 for the Kalman smoother, and the test CRPS is 0.0362 against 0.0363. The Normal models forecast the test year worse than the Student-T models of the previous section (test CRPS 0.0301 and 0.0303). Those models put the degrees of freedom \nu near 1.6 (the table of the previous section), so the noise of this series has much heavier tails than a Normal distribution.


``` python
fig, axes = plt.subplots(
    nrows=2, ncols=1, figsize=(12, 9), sharex=True, sharey=True, layout="constrained"
)
start = t_obs - 104  # the last two years of the training window, then the test year

for ax, (label, tree) in zip(axes, trees.items(), strict=True):
    for group, color, span in (
        ("posterior_predictive", "C0", slice(start, t_obs)),
        ("predictions", "C1", slice(None)),
    ):
        draws = stacked(tree, group)
        name = "in-sample" if group == "posterior_predictive" else "forecast"
        weeks = week[start:t_obs] if group == "posterior_predictive" else week[t_obs:]
        for prob, alpha in ((0.94, 0.3), (0.5, 0.5)):
            lower, upper = hdi_bounds(draws, prob)
            ax.fill_between(
                weeks,
                lower[span],
                upper[span],
                color=color,
                alpha=alpha,
                label=hdi_label(prob, prefix=f"{name} "),
            )

    ax.plot(week[start:], bart[start:, 0], color="black", lw=1, label="observed")
    ax.axvline(t_obs, color="gray", ls="--", label="train/test split")
    ax.set(title=label, ylabel="log(# rides)")

axes[-1].set(xlabel="week")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncols=3)
fig.suptitle(
    "In-sample posterior predictive and forecast from to_datatree", fontsize=14, fontweight="bold"
);
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-23-output-1.png" class="figure-img" width="1211" height="906" /></p>
</figure>


The two models give the same bands. The in-sample band is the observation noise around the level, and the forecast band is only a little wider: the weekly steps of the level are small, so one year of them adds little to the observation noise. The `dynestyx` band comes from per-week smoothing draws and the [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) band from joint paths, and for every single week the two are the same distribution, which is all these bands show.


# Missing Data

The `LatentPathBuilder` skips missing observations, so the Student-T model fits through a gap with no change to the model function. (The Kalman filter and smoother of the `cuthbert` backend skip them too, while the `cd_dynamax` backend rejects them.) We remove 20 weeks of the training window. The block reads the observed window from column 0 of the covariates, so the gap goes there as `NaN`. A `NaN` in `data` alone would not reach the likelihood. We also put the gap into `data`, which [to_datatree](../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) would store as the observed data.

The gap fit gets its own `LatentPathBuilder`. A builder keeps one pattern of missing observations per site name and shape, so a second fit with the same builder would replace the pattern that the drivers of the first fit read.


``` python
gap = slice(300, 320)
bart_gap = bart_train.at[gap].set(jnp.nan)
bart_covariates_gap = bart_covariates.at[gap, 0].set(jnp.nan)
gap_builder = LatentPathBuilder()
rng_key, key_gap = random.split(rng_key)

with gap_builder:
    mcmc_gap, _ = fit_nuts(
        key_gap,
        level_regression,
        bart_covariates_gap[:t_obs],
        bart_gap,
        num_warmup=2_000,
        chain_method="sequential",
    )

pd.DataFrame({"gap fit": convergence(mcmc_gap, bart_names)}).T.round(
    {"min bulk ESS": 0, "max R-hat": 3, "divergences": 0}
)
```


|         | min bulk ESS | max R-hat | divergences |
|---------|--------------|-----------|-------------|
| gap fit | 65.0         | 1.045     | 0.0         |


The gap fit mixes even more slowly than the explicit path on all weeks (smallest bulk ESS 65 against 117, largest \hat{R} 1.045), with no divergent transitions, so its summaries are approximate too.

We draw the in-sample predictive of the gap fit, which covers the gap weeks, under the builder of that fit. As a reference we use the in-sample predictive of the fit on all weeks. That fit has seen the weeks of the gap, so we expect it to score better there.


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

before = slice(gap.start - 20, gap.start)
gap_scores = {}

for label, draws in in_sample.items():
    lower, upper = hdi_bounds(draws, 0.94)
    gap_scores[label] = {
        "CRPS on the gap weeks": float(eval_crps(draws[:, gap], bart_train[gap])),
        "94% HDI width, gap weeks": float((upper - lower)[gap].mean()),
        "94% HDI width, 20 weeks before": float((upper - lower)[before].mean()),
    }

pd.DataFrame(gap_scores).T.round(4)
```


|  | CRPS on the gap weeks | 94% HDI width, gap weeks | 94% HDI width, 20 weeks before |
|----|----|----|----|
| fit with the gap | 0.0256 | 0.1944 | 0.1758 |
| fit on all weeks | 0.0229 | 0.1791 | 0.1787 |


Inside the gap, the 94\\ band of the gap fit is about 11\\ wider than in the 20 weeks before (0.194 against 0.176), while the band of the fit on all weeks keeps its width (0.179 in both windows). On the held-out weeks the CRPS of the gap fit is 0.0256, against 0.0229 for the fit that saw them.


``` python
fig, ax = plt.subplots(figsize=(12, 6))
window = slice(250, t_obs)

for prob, alpha in ((0.94, 0.3), (0.5, 0.5)):
    lower, upper = hdi_bounds(in_sample["fit with the gap"], prob)
    ax.fill_between(
        week[window],
        lower[window],
        upper[window],
        color="C1",
        alpha=alpha,
        label=hdi_label(prob, prefix="fit with the gap "),
    )

ax.plot(
    week[window],
    np.median(np.asarray(in_sample["fit on all weeks"]), axis=0)[window, 0],
    color="C0",
    ls="--",
    label="fit on all weeks, median",
)
ax.plot(week[window], bart_gap[window, 0], "o", color="black", ms=3, label="observed")
ax.plot(week[gap], bart_train[gap, 0], "o", color="gray", ms=4, mfc="none", label="held out")
ax.axvspan(gap.start, gap.stop - 1, color="gray", alpha=0.15, label="gap")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax.set(
    title="In-sample posterior predictive through a 20-week gap",
    xlabel="week",
    ylabel="log(# rides)",
);
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-26-output-1.png" class="figure-img" width="1211" height="611" /></p>
</figure>


The band follows the Fourier regression through the gap, including the two dips of the yearly cycle inside it, and the median of the fit on all weeks stays inside the band. The band widens only a little: 20 weeks of a random walk with the small weekly steps of this series add little to the uncertainty of the level.


# Sampler Cost

The three comparisons above measure the same three quantities. The number of gradient evaluations per effective draw depends on the shape of the posterior that NUTS explores. The milliseconds per gradient depend on the model code. Their product is the compute time per effective draw, without the compilation and the overhead that the wall time includes.


``` python
cost = pd.concat(
    {
        "ARMA(1,1)": pd.DataFrame(arma_cost).T,
        "local level, Student-T": pd.DataFrame(bart_cost).T,
        "local level, Normal": pd.DataFrame(gaussian_cost).T,
    },
    names=["comparison", "fit"],
)
cost["ms per effective draw"] = cost["gradients per effective draw"] * cost["ms per gradient"]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4.5), layout="constrained")
comparisons = list(cost.index.get_level_values("comparison").unique())
positions = np.arange(len(comparisons))

panels = {
    "gradients per effective draw": "Shape of the posterior",
    "ms per gradient": "Cost of one gradient",
    "ms per effective draw": "Compute per effective draw",
}

for ax, (column, title) in zip(axes, panels.items(), strict=True):
    for offset, (library, color) in zip(
        (-0.2, 0.2), (("package building block", "C0"), ("dynestyx", "C1")), strict=True
    ):
        values = [
            cost.loc[name, column].iloc[0 if library == "package building block" else 1]
            for name in comparisons
        ]
        ax.bar(positions + offset, values, width=0.4, color=color, label=library)

    ax.set_yscale("log")
    ax.set_xticks(positions, [name.replace(", ", "\n") for name in comparisons])
    ax.set(title=title, xlabel="comparison", ylabel=column)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncols=2)
fig.suptitle("Sampler cost of the three comparisons", fontsize=14, fontweight="bold");
```


<figure class="figure">
<p><img src="dynestyx_getting_started_files/figure-html/_src-dynestyx_getting_started-cell-27-output-1.png" class="figure-img" width="1211" height="456" /></p>
</figure>


In the ARMA comparison both models sample the same four parameters, so they need about the same number of gradients per effective draw, and the more expensive gradient of the Kalman filter decides: the [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) model is about two orders of magnitude cheaper per effective draw. For the Student-T level, the explicit path of `dynestyx` needs about four times as many gradients per effective draw as [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations), because of the slow `drift_scale`, and its slightly cheaper gradients do not make up for that. For the Normal level, the Kalman filter needs about 40 times fewer gradients per effective draw, but its gradients cost about 80 times more, so it needs about twice the compute per effective draw. The filter makes the posterior easy for NUTS. Whether it also saves time depends on the cost of its gradient, which grows with the length of the series and the size of the state.


# Conclusion


## Key Findings

- **One model function, any handler.** [state_space](../../reference/contrib.dynestyx.state_space.md#numpyro_forecast.contrib.dynestyx.state_space) turns a `DynamicalModel` into a `numpyro_forecast` model. The handler around each call selects the inference: a Kalman filter or smoother integrates the level out, a `LatentPathBuilder` samples it. [forecast](../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast), [predict_in_sample](../../reference/predictive.predict_in_sample.md#numpyro_forecast.predictive.predict_in_sample) and [to_datatree](../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) work unchanged with the handler around the call.
- **Same posteriors, same forecasts.** Where both fits converge, the `dynestyx` models and the building blocks of the package give the same posteriors. The forecasts of the two BART comparisons agree, with test CRPS values within 0.0002 of each other, and so do the in-sample bands of the two Normal models.
- **Integrating out moves the cost.** With the Kalman filter, NUTS needs few leapfrog steps per draw and few gradients per effective draw, has no divergent transitions, and each gradient costs more. For the Normal level on the CPU, the filter needs about twice the compute per effective draw of [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations).
- **The explicit path is general.** The `LatentPathBuilder` accepts non-Gaussian observation models and missing observations. It samples the states in centered form, and here `drift_scale` mixes slowly under it.


## Limitations

- The in-sample draws under a smoother are independent across weeks. For joint paths, use [innovations](../../reference/models.innovations.md#numpyro_forecast.models.innovations) or a `LatentPathBuilder`.
- The `LatentPathBuilder` has no reparameterization option. Every later call must use the builder instance of the fit, and parallel chains under it crashed the process on the machine that executed this notebook.
- The block models one series per model function, with the step index as its time grid.
- Without a handler around a call, `dynestyx` picks one itself and does not warn, so always pass the handler and its configuration.


## Next Steps

The [advanced `dynestyx` example](dynestyx_integration.md) compares filtering and smoothing on a local level model, measures the cost of a full evaluation against a gradient, fits a seasonal model with SVI, and backtests a `dynestyx` model with expanding windows.
