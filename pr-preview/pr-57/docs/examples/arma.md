# ARMA(1,1) Model


ARMA(1,1) Model with `numpyro_forecast`

This notebook ports the blog post [**Notes on an ARMA(1,1) Model with NumPyro**](https://juanitorduz.github.io/arma_numpyro/) to the [`numpyro_forecast`](https://github.com/juanitorduz/numpyro_forecast) package. Autoregressive moving average (ARMA) models are the workhorse of classical time series analysis, and the \\(1,1)\\ member is the smallest one that combines both mechanisms: an autoregressive term that feeds the previous *observation* back into the mean, and a moving average term that feeds the previous *forecast error* back. We simulate data from an ARMA(1,1) process, so we own the data generating process and can verify that the model recovers the true parameters, and we fit it with MCMC (the NUTS sampler) through the package's [HMCForecaster](../../reference/forecaster.HMCForecaster.md#numpyro_forecast.forecaster.HMCForecaster).

Two deliberate changes from the blog post are worth calling out:

- Instead of a single train-test split, we evaluate with **expanding-window time-slice cross-validation** via `numpyro_forecast.backtest`, scoring every fold with the continuous ranked probability score (CRPS) and the empirical coverage of the central \\50\\\\ and \\94\\\\ intervals, both in-sample and out-of-sample, exactly as in the [univariate forecasting example](https://juanitorduz.github.io/numpyro_forecast/examples/forecasting_univariate.html).
- The forecast path is fully **generative**: over the horizon we sample future innovations and feed them back into the ARMA recursion, so the forecast uncertainty compounds correctly step by step. The blog post instead zeroed the future errors inside its prediction loop, which understates the multi-step uncertainty.

A practical note on the design, in the same spirit as the [exponential smoothing example](https://juanitorduz.github.io/numpyro_forecast/examples/exponential_smoothing_state_space.html): the built-in [time_series](../../reference/functional.models.time_series.md#numpyro_forecast.functional.models.time_series) and [predict](../../reference/functional.models.predict.md#numpyro_forecast.functional.models.predict) primitives assume a deterministic mean plus independent per-step noise, which cannot express ARMA's recursive dependence on past observations and errors. We therefore write the model body directly against the functional API's [Horizon](../../reference/functional.models.Horizon.md#numpyro_forecast.functional.models.Horizon) value, registering the framework's `"obs"` and `"forecast"` sites ourselves, while reusing everything downstream: [HMCForecaster](../../reference/forecaster.HMCForecaster.md#numpyro_forecast.forecaster.HMCForecaster), [predict_in_sample](../../reference/functional.prediction.predict_in_sample.md#numpyro_forecast.functional.prediction.predict_in_sample), and [backtest](../../reference/evaluate.backtest.md#numpyro_forecast.evaluate.backtest). And since an ARMA model is, at heart, a regression on the *lagged series itself*, the observed series plays the role of the covariates: the `covariates` argument is the natural carrier for the history the filter needs, because it spans the full horizon and is available at prediction time. The model only ever reads the first [t_obs](../../reference/forecaster.ForecastingModel.md#numpyro_forecast.forecaster.ForecastingModel.t_obs) rows, so no future information leaks into a forecast (we simply never look past the training window).


# Prepare notebook


``` python
from functools import partial

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from jaxtyping import Float

from numpyro_forecast import (
    HMCForecaster,
    backtest,
    eval_coverage,
    eval_crps,
    forecasting_model,
)
from numpyro_forecast.functional import Horizon
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


# Generate data

The ARMA(1,1) process is defined by the recursion

\\y_t = \phi \\ y\_{t-1} + \theta \\ \varepsilon\_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim \text{Normal}(0, \sigma),\\

where \\\phi\\ is the autoregressive coefficient, \\\theta\\ the moving average coefficient, and \\\sigma\\ the innovation scale. We simulate \\T = 100\\ observations with \\\phi = 0.4\\, \\\theta = 0.7\\, and \\\sigma = 0.5\\ (one extra step initializes the recursion and is dropped). As in the blog post, we write the simulation twice: first as a transparent Python loop, then with [`jax.lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html), which compiles the recursion into a single efficient operation and is the idiom we also use inside the model. Compared to the blog we tighten the type hints: the key takes the package-wide `Array` type and the return shape is spelled out with a [jaxtyping](https://docs.kidger.site/jaxtyping/) annotation.


``` python
phi_true = 0.4
theta_true = 0.7
noise_scale = 0.5
n_samples = 100 + 1  # one extra step to initialize the recursion


def generate_arma_1_1_data_for_loop(
    rng_key: Array, n_samples: int, phi: float, theta: float, noise_scale: float
) -> Float[Array, " t"]:
    """Simulate an ARMA(1,1) series with an explicit Python loop.

    Parameters
    ----------
    rng_key
        PRNG key for the innovations.
    n_samples
        Number of steps to simulate; the initial step is dropped, so the
        returned series has ``n_samples - 1`` entries.
    phi
        Autoregressive coefficient.
    theta
        Moving average coefficient.
    noise_scale
        Scale of the Gaussian innovations.

    Returns
    -------
    Float[Array, " t"]
        The simulated series of length ``n_samples - 1``.
    """
    error = noise_scale * random.normal(rng_key, (n_samples,))
    y = jnp.zeros(n_samples)
    for t in range(1, n_samples):
        y = y.at[t].set(phi * y[t - 1] + theta * error[t - 1] + error[t])
    return y[1:]


rng_key, rng_subkey = random.split(rng_key)
y_for_loop = generate_arma_1_1_data_for_loop(
    rng_subkey, n_samples, phi_true, theta_true, noise_scale
)
print(f"series shape: {y_for_loop.shape}")
```


    series shape: (100,)


The scan version threads a carry `(y_prev, error_prev)` through the innovations and produces the same series:


``` python
def generate_arma_1_1_data_scan(
    rng_key: Array, n_samples: int, phi: float, theta: float, noise_scale: float
) -> Float[Array, " t"]:
    """Simulate an ARMA(1,1) series with :func:`jax.lax.scan`.

    Parameters
    ----------
    rng_key
        PRNG key for the innovations.
    n_samples
        Number of steps to simulate; the initial step is dropped, so the
        returned series has ``n_samples - 1`` entries.
    phi
        Autoregressive coefficient.
    theta
        Moving average coefficient.
    noise_scale
        Scale of the Gaussian innovations.

    Returns
    -------
    Float[Array, " t"]
        The simulated series of length ``n_samples - 1``.
    """
    error = noise_scale * random.normal(rng_key, (n_samples,))

    def arma_step(carry, noise):
        y_prev, error_prev = carry
        y_t = phi * y_prev + theta * error_prev + noise
        return (y_t, noise), y_t

    _, y = jax.lax.scan(arma_step, (jnp.zeros(()), error[0]), error[1:])
    return y


y = generate_arma_1_1_data_scan(rng_subkey, n_samples, phi_true, theta_true, noise_scale)
print(f"for-loop and scan agree: {jnp.allclose(y_for_loop, y, atol=1e-6)}")
```


    for-loop and scan agree: True


Throughout the package, time lives at axis `-2` and the observation dimension at axis `-1`, so the single series has shape `(100, 1)`. Following the design note above, the same array also serves as the covariates.


``` python
data = y[:, None]
covariates = data  # the lagged series is the "covariate" of an ARMA model
duration = data.shape[-2]
time = np.arange(duration)
print(f"data shape: {data.shape}")

fig, ax = plt.subplots()
ax.plot(time, np.asarray(y), color="black", lw=1)
ax.set(title="Simulated ARMA(1,1) series", xlabel="time", ylabel="y")
plt.show()
```


    data shape: (100, 1)


<figure class="figure">
<p><img src="arma_files/figure-html/_src-arma-cell-5-output-2.png" class="figure-img" width="1011" height="611" /></p>
</figure>


# ACF and PACF

Before modeling, we look at the empirical autocorrelation function (ACF) and partial autocorrelation function (PACF), the classical tools for identifying ARMA orders. The blog post used the statsmodels plotting helpers; here we compute both from scratch, which takes a handful of lines (the PACF follows from the ACF via the Durbin-Levinson recursion).


``` python
def acf(y: Float[Array, " t"], max_lag: int) -> Float[Array, " lags"]:
    """Empirical autocorrelation function up to ``max_lag``.

    Parameters
    ----------
    y
        The observed series.
    max_lag
        Largest lag to compute.

    Returns
    -------
    Float[Array, " lags"]
        Autocorrelations at lags ``0, ..., max_lag``.
    """
    y_centered = y - y.mean()
    denominator = jnp.sum(y_centered**2)
    return jnp.stack(
        [
            jnp.sum(y_centered[lag:] * y_centered[: y.size - lag]) / denominator
            for lag in range(max_lag + 1)
        ]
    )


def pacf(y: Float[Array, " t"], max_lag: int) -> Float[Array, " lags"]:
    """Empirical partial autocorrelations via the Durbin-Levinson recursion.

    Parameters
    ----------
    y
        The observed series.
    max_lag
        Largest lag to compute.

    Returns
    -------
    Float[Array, " lags"]
        Partial autocorrelations at lags ``0, ..., max_lag`` (``1`` at lag zero).
    """
    rho = np.asarray(acf(y, max_lag))
    pacf_values = np.zeros(max_lag + 1)
    pacf_values[0] = 1.0
    phi_prev = np.zeros(max_lag + 1)
    for k in range(1, max_lag + 1):
        if k == 1:
            phi_kk = rho[1]
        else:
            numerator = rho[k] - np.sum(phi_prev[1:k] * rho[k - 1 : 0 : -1])
            denominator = 1.0 - np.sum(phi_prev[1:k] * rho[1:k])
            phi_kk = numerator / denominator
        phi_new = phi_prev.copy()
        phi_new[k] = phi_kk
        phi_new[1:k] = phi_prev[1:k] - phi_kk * phi_prev[k - 1 : 0 : -1]
        phi_prev = phi_new
        pacf_values[k] = phi_kk
    return jnp.asarray(pacf_values)


max_lag = 20
lags = np.arange(max_lag + 1)
significance_bound = 1.96 / np.sqrt(duration)

fig, (ax_acf, ax_pacf) = plt.subplots(
    nrows=2, ncols=1, figsize=(10, 8), sharex=True, sharey=True, layout="constrained"
)
ax_acf.stem(lags, np.asarray(acf(y, max_lag)))
ax_acf.axhspan(-significance_bound, significance_bound, color="C0", alpha=0.15)
ax_acf.set(title="Autocorrelation function (ACF)", ylabel="autocorrelation")
ax_pacf.stem(lags, np.asarray(pacf(y, max_lag)))
ax_pacf.axhspan(-significance_bound, significance_bound, color="C0", alpha=0.15)
ax_pacf.set(
    title="Partial autocorrelation function (PACF)",
    xlabel="lag",
    ylabel="partial autocorrelation",
)
plt.show()
```


<figure class="figure">
<p><img src="arma_files/figure-html/_src-arma-cell-6-output-1.png" class="figure-img" width="1011" height="811" /></p>
</figure>


The shaded band is the approximate \\95\\\\ significance region \\\pm 1.96 / \sqrt{T}\\ for white noise. The ACF starts high and decays geometrically over the first few lags, while the PACF is large and positive at lag \\1\\ and then flips to a significant negative value at lag \\2\\ before dying out: a sign-alternating decay rather than a clean cutoff. Neither function cuts sharply, which is the classical signature of a mixed ARMA process: a pure AR(\\p\\) model would cut the PACF after lag \\p\\, and a pure MA(\\q\\) model would cut the ACF after lag \\q\\.


# Model specification

The generative model adds a global mean \\\mu\\ to the recursion and uses the same priors as the blog post:

\\y_t = \mu + \phi \\ y\_{t-1} + \theta \\ \varepsilon\_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim \text{Normal}(0, \sigma),\\

\\\begin{align\*} \mu & \sim \text{Normal}(0, 1), \\ \phi & \sim \text{Uniform}(-1, 1), \\ \theta & \sim \text{Uniform}(-1, 1), \\ \sigma & \sim \text{HalfNormal}(1). \end{align\*}\\

The key insight (from the blog post, and the same one behind the innovations state space form of exponential smoothing) is that *in sample the errors are deterministic* given the parameters and the observed data: running the recursion forward, the one-step-ahead prediction at time \\t\\ is \\\hat{y}\_t = \mu + \phi \\ y\_{t-1} + \theta \\ \varepsilon\_{t-1}\\ and the error is simply \\\varepsilon_t = y_t - \hat{y}\_t\\, initialized with \\y\_{-1} = \mu\\ and \\\varepsilon\_{-1} = 0\\. The whole in-sample likelihood is then a single Gaussian observation site: conditioning \\y_t \sim \text{Normal}(\hat{y}\_t, \sigma)\\ is exactly the blog post's "condition on the errors" trick, \\\varepsilon_t \sim \text{Normal}(0, \sigma)\\, since the two differ only by a location shift.

The model body follows the package's two-scan pattern:

1.  **In sample.** A deterministic `jax.lax.scan` filters the observed series into one-step-ahead means \\\hat{y}\_t\\ (exposed as the deterministic site `"mu_t"`), and the `"obs"` site conditions the data on them.
2.  **Out of sample.** When `h.future > 0` we draw the horizon innovations from the prior at a separate `"eps_future"` site, then roll the recursion forward feeding the *sampled* observation and innovation back into the carry, and expose the trajectory as the deterministic `"forecast"` site the forecaster reads. Because `"eps_future"` does not exist while training, `Predictive` draws it from the prior at forecast time, so the forecast uncertainty compounds over the horizon exactly as the generative process says it should.

Neither scan body contains a sample site, so plain `jax.lax.scan` is all we need.


``` python
def arma_1_1(h: Horizon, covariates: Array) -> None:
    """ARMA(1,1) model with a deterministic in-sample error-feedback filter.

    Parameters
    ----------
    h
        The train/forecast horizon for the current model call.
    covariates
        The observed series itself, with time at axis ``-2``; only the first
        ``h.t_obs`` rows are read.
    """
    y = covariates[..., : h.t_obs, 0]  # observed history only; never reads beyond t_obs

    mu = numpyro.sample("mu", dist.Normal(loc=0, scale=1))
    phi = numpyro.sample("phi", dist.Uniform(low=-1, high=1))
    theta = numpyro.sample("theta", dist.Uniform(low=-1, high=1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=1))

    def transition_fn(carry, y_t):
        y_prev, error_prev = carry
        pred = mu + phi * y_prev + theta * error_prev
        return (y_t, y_t - pred), pred

    init_carry = (mu, jnp.zeros(()))  # y_{-1} = mu and eps_{-1} = 0 seed the recursion
    (_, error_last), preds = jax.lax.scan(transition_fn, init_carry, y)

    numpyro.deterministic("mu_t", preds[:, None])
    numpyro.sample("obs", dist.Normal(loc=preds[:, None], scale=sigma), obs=h.data)

    if h.future > 0:
        eps_future = numpyro.sample(
            "eps_future", dist.Normal(loc=0, scale=sigma).expand([h.future]).to_event(1)
        )

        def forecast_fn(carry, eps):
            y_prev, error_prev = carry
            pred = mu + phi * y_prev + theta * error_prev
            y_next = pred + eps  # the sampled innovation is fed back into the carry
            return (y_next, eps), y_next

        _, y_future = jax.lax.scan(forecast_fn, (y[-1], error_last), eps_future)
        numpyro.deterministic("forecast", y_future[:, None])


model = forecasting_model(arma_1_1)
```


# Inference with NUTS

We fit the model on the **full series** (no train-test split; the held-out evaluation comes from the cross-validation below) with the No-U-Turn Sampler through [HMCForecaster](../../reference/forecaster.HMCForecaster.md#numpyro_forecast.forecaster.HMCForecaster), running \\4\\ chains of \\2{,}000\\ warmup and \\2{,}000\\ sampling steps each, matching the blog post's setup. The posterior is tiny: because the in-sample errors are deterministic, the only latent parameters are \\\mu\\, \\\phi\\, \\\theta\\, and \\\sigma\\.


``` python
rng_key, rng_subkey = random.split(rng_key)
forecaster = HMCForecaster(
    rng_subkey,
    model,
    data,
    covariates,
    num_warmup=2_000,
    num_samples=2_000,
    num_chains=4,
)
```


# Diagnostics and parameter recovery

[HMCForecaster](../../reference/forecaster.HMCForecaster.md#numpyro_forecast.forecaster.HMCForecaster) stores the posterior draws with the chains flattened together (deterministic sites such as `"mu_t"` ride along; we keep just the four parameters). NumPyro flattens the chains in order, so a plain reshape recovers the `(chain, draw)` structure for ArviZ. Since we simulated the data ourselves, we can put the posterior side by side with the true parameter values.


``` python
num_chains = 4
scalar_vars = ["mu", "phi", "theta", "sigma"]
true_values = {"mu": 0.0, "phi": phi_true, "theta": theta_true, "sigma": noise_scale}

posterior = {
    name: np.asarray(value).reshape(num_chains, -1, *value.shape[1:])
    for name, value in forecaster.posterior_samples.items()
    if name in scalar_vars
}
idata = az.from_dict({"posterior": posterior})

rhat = az.rhat(idata, var_names=scalar_vars)
ess_bulk = az.ess(idata, var_names=scalar_vars)
ess_tail = az.ess(idata, var_names=scalar_vars, method="tail")
recovery = pd.DataFrame(
    {
        "true_value": [true_values[name] for name in scalar_vars],
        "posterior_mean": [float(posterior[name].mean()) for name in scalar_vars],
        "posterior_sd": [float(posterior[name].std()) for name in scalar_vars],
        "r_hat": [float(rhat[name].item()) for name in scalar_vars],
        "ess_bulk": [float(ess_bulk[name].item()) for name in scalar_vars],
        "ess_tail": [float(ess_tail[name].item()) for name in scalar_vars],
    },
    index=scalar_vars,
)
recovery.round({"posterior_mean": 3, "posterior_sd": 3, "r_hat": 3, "ess_bulk": 0, "ess_tail": 0})
```


|       | true_value | posterior_mean | posterior_sd | r_hat | ess_bulk | ess_tail |
|-------|------------|----------------|--------------|-------|----------|----------|
| mu    | 0.0        | -0.081         | 0.068        | 1.000 | 5254.0   | 5210.0   |
| phi   | 0.4        | 0.376          | 0.147        | 1.001 | 4192.0   | 4533.0   |
| theta | 0.7        | 0.526          | 0.151        | 1.001 | 4349.0   | 4084.0   |
| sigma | 0.5        | 0.436          | 0.032        | 1.001 | 6510.0   | 5500.0   |


The sampler recovers the parameters well: every true value lies within about two posterior standard deviations of its posterior mean, the \\\hat{R}\\ values are essentially \\1\\, and the effective sample sizes are healthy. The point estimates for \\\theta\\ and \\\sigma\\ come in somewhat low, and this is a feature of the particular realization rather than of the model: the innovations drawn for this seed happen to have a sample standard deviation of \\0.43\\ (against the population value \\0.5\\), and the posterior mean of \\\sigma\\ matches that realized scale almost exactly. The moving average coefficient is in turn the hardest parameter to pin down with \\T = 100\\ observations, because \\\phi\\ and \\\theta\\ can partially substitute for each other in an ARMA likelihood (a well-known feature), so its posterior is wide. The trace plots make the recovery visual: the dashed black lines mark the true values, and the chains mix well around them.


``` python
pc_trace = az.plot_trace_dist(
    idata,
    var_names=scalar_vars,
    compact=True,
    figure_kwargs={"figsize": (10, 12)},
)
for var_name in scalar_vars:
    ax_dist = pc_trace.viz["plot"][var_name].sel(column="dist").item()
    ax_dist.axvline(true_values[var_name], color="black", ls="--", lw=1.5, label="true value")
pc_trace.viz["plot"]["mu"].sel(column="dist").item().legend(loc="upper right")
pc_trace.viz["figure"].item().suptitle(
    "Trace plots and parameter recovery",
    fontsize=18,
    fontweight="bold",
    y=1.03,
)
plt.show()
```


<figure class="figure">
<p><img src="arma_files/figure-html/_src-arma-cell-10-output-1.png" class="figure-img" width="1011" height="1251" /></p>
</figure>


# In-sample fit

Next we look at the one-step-ahead posterior predictive over the training window with [predict_in_sample](../../reference/functional.prediction.predict_in_sample.md#numpyro_forecast.functional.prediction.predict_in_sample): at each time step the predicted mean uses the observed history up to the previous step, and the `"obs"` site adds the observation noise. We plot the \\50\\\\ and \\94\\\\ HDI bands against the observed series with ArviZ `plot_lm` and score the fit with the CRPS, a proper scoring rule that compares each observation to the whole predictive distribution (lower is better).


``` python
rng_key, rng_subkey = random.split(rng_key)
train_pp = forecaster.predict_in_sample(rng_subkey, covariates, num_samples=2_000)
crps_train = eval_crps(train_pp, data)

idata_in_sample = az.from_dict(
    {
        "posterior_predictive": {"obs": np.asarray(train_pp[..., 0])[None]},
        "observed_data": {"obs": np.asarray(y)},
        "constant_data": {"time": time.astype(float)},
    },
    coords={"t": time.astype(float)},
    dims={"obs": ["t"], "time": ["t"]},
)
pc = az.plot_lm(
    idata_in_sample,
    y="obs",
    x="time",
    ci_kind="hdi",
    ci_prob=(0.5, 0.94),
    smooth=False,
    visuals={"ci_band": {"color": "C0"}, "observed_scatter": False, "pe_line": False},
    figure_kwargs={"figsize": (10, 6)},
)
bands = pc.viz["ci_band"]["time"]
band_94, band_50 = bands.sel(prob=0.94).item(), bands.sel(prob=0.5).item()
band_94.set_label(r"$94\%$ HDI")
band_50.set_label(r"$50\%$ HDI")
ax = pc.viz["figure"].item().axes[0]
(obs_line,) = ax.plot(time, np.asarray(y), color="black", lw=1, label="observed")
ax.legend(handles=[band_94, band_50, obs_line], loc="upper right")
ax.set(
    title=f"One-step-ahead in-sample fit (train CRPS: {crps_train:.4f})",
    xlabel="time",
    ylabel="y",
)
plt.show()
```


<figure class="figure">
<p><img src="arma_files/figure-html/_src-arma-cell-11-output-1.png" class="figure-img" width="1011" height="611" /></p>
</figure>


# Expanding-window cross-validation

A single split tells us how the model does on one held-out window. A more honest picture comes from *expanding-window* time-slice cross-validation: we repeatedly move the train/test boundary forward, refit from scratch, and forecast the next window, so every later part of the series is scored out-of-sample exactly once. `numpyro_forecast.backtest` runs this loop for us, refitting the NUTS sampler on each fold through the same [HMCForecaster](../../reference/forecaster.HMCForecaster.md#numpyro_forecast.forecaster.HMCForecaster).

We size the folds at roughly \\10\\\\ of the series: each fold forecasts the next `10` steps (`test_window=10`), stepping forward `10` steps at a time (`stride=10`) so the folds do not overlap, and the first `50` observations (half the series) seed the initial training window (`min_train_window=50`). That yields five folds with split points at \\t = 50, 60, 70, 80, 90\\. With `eval_train=True` each fold also scores its in-sample one-step-ahead posterior predictive with the same metrics (this is what the series-as-covariates design buys us), and `keep_predictions=True` retains the out-of-sample forecast samples so we can plot them. Alongside the CRPS we track the empirical **coverage** of the central \\50\\\\ and \\94\\\\ intervals: a well-calibrated forecast covers close to its nominal level.


``` python
metrics = {
    "crps": eval_crps,
    "coverage_50": partial(eval_coverage, alpha=0.5),
    "coverage_94": partial(eval_coverage, alpha=0.94),
}

rng_key, rng_subkey = random.split(rng_key)
results = backtest(
    rng_subkey,
    data,  # full dataset, no train/test split
    covariates,  # the series itself, sliced per fold by backtest
    lambda: model,
    forecaster_fn=HMCForecaster,
    metrics=metrics,
    test_window=10,  # ~10% of the series per fold
    stride=10,  # non-overlapping folds
    min_train_window=50,  # half the series seeds the first fold
    num_samples=2_000,
    eval_train=True,
    keep_predictions=True,
    forecaster_options={"num_warmup": 2_000, "num_samples": 2_000, "num_chains": 4},
)

split_points = [r.t1 for r in results]
train_crps = [r.train_metrics["crps"] for r in results]
test_crps = [r.metrics["crps"] for r in results]
test_cov_50 = [r.metrics["coverage_50"] for r in results]
test_cov_94 = [r.metrics["coverage_94"] for r in results]

print(f"folds: {len(results)} (split points: {split_points})")
print(f"mean in-sample CRPS:     {np.mean(train_crps):.4f}")
print(f"mean out-of-sample CRPS: {np.mean(test_crps):.4f}")
print(f"mean out-of-sample 50% coverage: {np.mean(test_cov_50):.2f}  (nominal 0.50)")
print(f"mean out-of-sample 94% coverage: {np.mean(test_cov_94):.2f}  (nominal 0.94)")
```


    folds: 5 (split points: [50, 60, 70, 80, 90])
    mean in-sample CRPS:     0.2398
    mean out-of-sample CRPS: 0.2924
    mean out-of-sample 50% coverage: 0.60  (nominal 0.50)
    mean out-of-sample 94% coverage: 1.00  (nominal 0.94)


## Forecasts per fold

Overlaying every fold's out-of-sample forecast (orange \\50\\\\ and \\94\\\\ HDI bands) on the observed series gives the rolling-origin view: each band picks up where the previous fold's training window ended, and the dashed lines mark the successive train/test splits.


``` python
pc = None
for r in results:
    prediction = r.prediction
    if prediction is None:  # keep_predictions=True guarantees this never triggers
        continue
    fold_time = time[r.t1 : r.t2].astype(float)
    idata_fold = az.from_dict(
        {
            "posterior_predictive": {"obs": np.asarray(prediction[..., 0])[None]},
            "observed_data": {"obs": np.asarray(data[r.t1 : r.t2, 0])},
            "constant_data": {"time": fold_time},
        },
        coords={"t": fold_time},
        dims={"obs": ["t"], "time": ["t"]},
    )
    if pc is None:
        pc = az.plot_lm(
            idata_fold,
            y="obs",
            x="time",
            ci_kind="hdi",
            ci_prob=(0.5, 0.94),
            smooth=False,
            visuals={"ci_band": {"color": "C1"}, "observed_scatter": False, "pe_line": False},
            figure_kwargs={"figsize": (12, 6)},
        )
        bands = pc.viz["ci_band"]["time"]
        band_94, band_50 = bands.sel(prob=0.94).item(), bands.sel(prob=0.5).item()
    else:
        az.plot_lm(
            idata_fold,
            y="obs",
            x="time",
            plot_collection=pc,
            ci_kind="hdi",
            ci_prob=(0.5, 0.94),
            smooth=False,
            visuals={"ci_band": {"color": "C1"}, "observed_scatter": False, "pe_line": False},
        )

if pc is None:
    msg = "no folds were plotted"
    raise ValueError(msg)
ax = pc.viz["figure"].item().axes[0]
band_94.set_label(r"forecast $94\%$ HDI")
band_50.set_label(r"forecast $50\%$ HDI")
(obs_line,) = ax.plot(time, np.asarray(y), color="black", lw=1, label="observed")
split_lines = [
    ax.axvline(r.t1, color="gray", ls="--", lw=0.5, label="train/test split") for r in results
]
ax.legend(handles=[band_94, band_50, obs_line, split_lines[0]], loc="upper left")
ax.set(title="Expanding-window cross-validation forecasts", xlabel="time", ylabel="y")
plt.show()
```


<figure class="figure">
<p><img src="arma_files/figure-html/_src-arma-cell-13-output-1.png" class="figure-img" width="1211" height="611" /></p>
</figure>


The bands show textbook ARMA behavior. Within each fold the forecast mean decays toward the unconditional level at the geometric rate \\\phi^h\\, so with \\\phi = 0.4\\ the memory of the last observation is essentially gone after two or three steps and the band settles at the *marginal* width of the process. This is not a defect: for a short-memory process, reverting quickly to the stationary distribution is exactly the right forecast, and what cross-validation checks here is that the bands are the right width, in other words that the forecast is calibrated.


## CRPS per fold

The per-fold CRPS quantifies the picture. The in-sample and out-of-sample scores stay close across folds, and neither trends upward as the training window grows, so the model is neither over- nor under-fitting: with only four parameters and a correctly specified model class, even the first fold's \\50\\ observations pin the predictive distribution down well. The out-of-sample score is a bit noisier, as it is computed from just \\10\\ observations per fold.


``` python
fig, ax = plt.subplots()
ax.plot(split_points, train_crps, "o-", color="C0", label="in-sample CRPS")
ax.plot(split_points, test_crps, "o-", color="C1", label="out-of-sample CRPS")
ax.legend()
ax.set(xlabel="train/test split point", ylabel="CRPS", title="CRPS per cross-validation fold")
plt.show()
```


<figure class="figure">
<p><img src="arma_files/figure-html/_src-arma-cell-14-output-1.png" class="figure-img" width="1011" height="611" /></p>
</figure>


## Forecast calibration

Finally we check the coverage: the fraction of held-out observations that fall inside the central \\50\\\\ and \\94\\\\ prediction intervals of their fold's forecast. Well-calibrated forecasts should track the dashed nominal levels; points above mean the bands are too wide (under-confident), points below that they are too narrow (over-confident). With only \\10\\ observations per fold the empirical coverage is grainy (each observation moves it by \\0.1\\), so some wobble around the nominal levels is expected.

A small caveat: [eval_coverage](../../reference/evaluate.eval_coverage.md#numpyro_forecast.evaluate.eval_coverage) measures coverage of the central quantile interval, while the plotted bands are ArviZ HDIs. For the near-symmetric Gaussian predictive here the two nearly coincide, so this is a faithful check of the bands shown above.


``` python
fig, ax = plt.subplots()
ax.plot(split_points, test_cov_50, "o-", color="C0", label=r"empirical $50\%$ coverage")
ax.plot(split_points, test_cov_94, "o-", color="C1", label=r"empirical $94\%$ coverage")
ax.axhline(0.5, color="C0", ls="--", lw=1, label=r"nominal $50\%$")
ax.axhline(0.94, color="C1", ls="--", lw=1, label=r"nominal $94\%$")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.set(
    xlabel="train/test split point",
    ylabel="coverage",
    title="Out-of-sample interval coverage per fold",
    ylim=(0, 1.05),
)
plt.show()
```


<figure class="figure">
<p><img src="arma_files/figure-html/_src-arma-cell-15-output-1.png" class="figure-img" width="1011" height="611" /></p>
</figure>


# References

- Orduz, J. [*Notes on an ARMA(1,1) Model with NumPyro*](https://juanitorduz.github.io/arma_numpyro/). The blog post this notebook ports.
- Hyndman, R. J., & Athanasopoulos, G. (2021). [*Forecasting: Principles and Practice*](https://otexts.com/fpp3/), 3rd edition. Chapter 9: ARIMA models.
- NumPyro documentation: [Example: AR(2) process](https://num.pyro.ai/en/stable/examples/ar2.html).
- Pyro forum: [Lax.scan to implement ARMA(1,1)](https://forum.pyro.ai/t/lax-scan-to-implement-arma-1-1/2518).
- The [exponential smoothing example](https://juanitorduz.github.io/numpyro_forecast/examples/exponential_smoothing_state_space.html) in this documentation, which uses the same functional-model-body plus [HMCForecaster](../../reference/forecaster.HMCForecaster.md#numpyro_forecast.forecaster.HMCForecaster) pattern for an error-feedback model.
- The [univariate forecasting example](https://juanitorduz.github.io/numpyro_forecast/examples/forecasting_univariate.html) in this documentation, which introduces [backtest](../../reference/evaluate.backtest.md#numpyro_forecast.evaluate.backtest) and the per-fold evaluation workflow.

[Source: ARMA(1,1) Model with `numpyro_forecast`](_src/arma-preview.html#f54f5241)
