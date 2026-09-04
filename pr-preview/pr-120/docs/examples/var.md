# Vector Autoregression (VAR)


Vector Autoregression (VAR) with `numpyro_forecast`

This notebook ports the blog post [**Bayesian VAR in NumPyro**](https://juanitorduz.github.io/var_numpyro/) to the [`numpyro_forecast`](https://github.com/juanitorduz/numpyro_forecast) package. A vector autoregression (VAR) models several time series jointly: each series is regressed on the past values of all series, and the shocks are correlated across series. We fit a VAR with two lags to the quarterly growth rates of US real GDP, consumption and investment, sample the posterior with NUTS, forecast 30 quarters ahead, and compute impulse response functions (IRFs), the standard tool to read a VAR.

The package provides the VAR pieces as reusable components. You do not write the lag recursion, the forecast loop or the IRF recursion yourself:

- [`var_step`](https://juanitorduz.github.io/numpyro_forecast/reference/var.var_step.html) turns sampled coefficients into a step for the [`ssoe`](https://juanitorduz.github.io/numpyro_forecast/reference/models.ssoe.html) building block. The block runs the in-sample recursion and the generative forecast.
- [`impulse_response`](https://juanitorduz.github.io/numpyro_forecast/reference/var.impulse_response.html) computes the responses for all posterior draws at once, with optional orthogonalization and cumulation.
- [`companion_matrix`](https://juanitorduz.github.io/numpyro_forecast/reference/var.companion_matrix.html) gives the stability check.
- [`minnesota_prior`](https://juanitorduz.github.io/numpyro_forecast/reference/priors.minnesota_prior.html) returns the moments of the Minnesota shrinkage prior. It lives in a separate module and is independent of the VAR code: the prior is always your own `numpyro.sample` call.

Some deliberate changes from the blog post:

- We model the growth rates in **percent** (`100 * diff(log)`), so the `Normal(0, 1)` and `HalfNormal(1)` priors are on a sensible scale.
- The in-sample recursion and the forecast run through [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) and [var_step](../../reference/var.var_step.md#numpyro_forecast.var.var_step) instead of a hand-written `scan`. The likelihood still conditions on the first two observations, as in the blog post.
- The IRF section adds **orthogonalized** and **cumulative** responses.
- A final section refits the model with a **Minnesota prior** and compares the two fits.
- We drop the comparison with `statsmodels`, which is not a dependency of the package.

The components are deliberately minimal. If you need a complete Bayesian VAR toolkit (identification schemes, variance decompositions, lag selection), see [Impulso](https://github.com/thomaspinder/impulso) by Thomas Pinder. Its `MinnesotaPrior` parameterization and its batched moving-average recursion inspired the two helpers used here.


# Prepare notebook


``` python
import warnings

import arviz as az
import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import xarray as xr
from jax import random
from jaxtyping import Float
from numpyro.infer import MCMC, NUTS

from numpyro_forecast import Horizon, eval_crps, predictions_to_datatree, ssoe, to_datatree
from numpyro_forecast.arrays import pad_future
from numpyro_forecast.priors import minnesota_prior
from numpyro_forecast.typing import Array, ForecastModel
from numpyro_forecast.var import companion_matrix, impulse_response, var_step

az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"
warnings.filterwarnings(
    "ignore", message="When multiple credible intervals are plotted", category=UserWarning
)

numpyro.set_host_device_count(n=4)

rng_key = random.PRNGKey(seed=42)

%load_ext autoreload
%autoreload 2
%load_ext jaxtyping
%jaxtyping.typechecker beartype.beartype
```


# Read data

We use the `macrodata` dataset shipped with `statsmodels`: quarterly US macroeconomic data from 1959Q1 to 2009Q3, compiled from the Federal Reserve Bank of St. Louis (FRED) and released in the public domain. We read the CSV from the `statsmodels` repository and keep three series in billions of chained 2005 dollars: real GDP, real personal consumption and real gross private domestic investment.

The levels trend upward and are not stationary. We take log differences, which turn levels into quarter-on-quarter growth rates, and multiply by 100 to read them in percent. A VAR assumes stationarity, and growth rates are a standard way to get there for macro aggregates.


``` python
url = (
    "https://raw.githubusercontent.com/statsmodels/statsmodels/main/"
    "statsmodels/datasets/macrodata/macrodata.csv"
)
macro_df = pd.read_csv(url)
dates_all = pd.PeriodIndex.from_fields(
    year=macro_df["year"], quarter=macro_df["quarter"], freq="Q"
).to_timestamp()
names = ["realgdp", "realcons", "realinv"]
levels = macro_df[names].set_index(dates_all)
y_pct = (100 * np.log(levels).diff().dropna()).rename_axis("date")
print(
    f"shape: {y_pct.shape}, from {y_pct.index[0].to_period('Q')} to {y_pct.index[-1].to_period('Q')}"
)
y_pct.head()
```


    shape: (202, 3), from 1959Q2 to 2009Q3


|            | realgdp   | realcons | realinv    |
|------------|-----------|----------|------------|
| date       |           |          |            |
| 1959-04-01 | 2.494213  | 1.528611 | 8.021268   |
| 1959-07-01 | -0.119295 | 1.038598 | -7.213104  |
| 1959-10-01 | 0.349453  | 0.108401 | 3.442511   |
| 1960-01-01 | 2.219018  | 0.953415 | 10.266377  |
| 1960-04-01 | -0.468455 | 1.257243 | -10.669385 |


``` python
y_pct.describe().T[["mean", "std", "min", "max"]].round(3)
```


|          | mean  | std   | min     | max    |
|----------|-------|-------|---------|--------|
| realgdp  | 0.776 | 0.880 | -2.071  | 3.859  |
| realcons | 0.837 | 0.694 | -2.296  | 2.773  |
| realinv  | 0.814 | 4.685 | -19.316 | 12.209 |


Investment growth is about five times more volatile than GDP or consumption growth. Keep this in mind for the Minnesota prior section: the three series are not on a common scale.


``` python
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 8), layout="constrained")
for ax, name, color in zip(axes, names, ("C0", "C1", "C2"), strict=True):
    ax.plot(y_pct.index, y_pct[name], color=color, lw=1.2, label=name)
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set(ylabel="percent")
    ax.legend(loc="upper right")
axes[-1].set(xlabel="date")
fig.suptitle("Quarterly growth rates (100 x log difference)", fontsize=16, fontweight="bold");
```


<figure class="figure">
<p><img src="var_files/figure-html/_src-var-cell-5-output-1.png" class="img-fluid figure-img" /></p>
</figure>


# Model specification

Let y_t \in \mathbb{R}^k be the vector of the k = 3 growth rates in quarter t. A VAR with p lags is

 y_t = c + \sum\_{l=1}^{p} \Phi_l \\ y\_{t-l} + \varepsilon_t, \qquad \varepsilon_t \sim \text{MultivariateNormal}(0, \Sigma), 

with an intercept vector c, one k \times k coefficient matrix \Phi_l per lag, and shocks that are independent over time but correlated across series through \Sigma. We parameterize the covariance through its Cholesky factor, \Sigma = L L^\top with L = \text{diag}(\sigma) \\ L\_\Omega, where \sigma holds the shock standard deviations and L\_\Omega is the Cholesky factor of the correlation matrix. This separates scales from correlations and gives each a natural prior:

\begin{align\*} c_i & \sim \text{Normal}(0, 1), \\ \sigma_i & \sim \text{HalfNormal}(1), \\ L\_\Omega & \sim \text{LKJCholesky}(\eta = 1), \\ \Phi\_{l, ij} & \sim \text{Normal}(0, 1). \end{align\*}

The LKJ prior with \eta = 1 is uniform over correlation matrices. We call the `Normal(0, 1)` prior on the 2 \times 3 \times 3 = 18 coefficients *weakly informative* rather than diffuse: on percent-scaled growth rates it already rules out wild dynamics.


## The VAR as an innovations state-space model

Stack the last p observations into a state s\_{t-1} = \[y\_{t-1}; \dots; y\_{t-p}\]. The observation equation is y_t = c + \[\Phi_1 \cdots \Phi_p\] \\ s\_{t-1} + \varepsilon_t, and the state update shifts the window and appends y_t. The same error vector \varepsilon_t drives both equations, and there is no separate state noise. In sample, given the parameters and the data, the state is known and the one-step-ahead mean \mu_t is deterministic; the error is the residual \varepsilon_t = y_t - \mu_t. This is the single-source-of-error form, and it is exactly the contract of the [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) building block:

1.  **In sample**, [ssoe](../../reference/models.ssoe.md#numpyro_forecast.models.ssoe) runs a deterministic `jax.lax.scan` over the observed rows, computing \mu_t from the lag window and pushing the observed y_t into the window. It returns the means as `r.mu`, and we write the likelihood `obs ~ MultivariateNormal(r.mu, L)` ourselves.
2.  **Out of sample**, when `covariates` extend beyond `data`, the block draws the future shocks \varepsilon\_{T+h} from the noise distribution at a separate `eps_future` site, feeds y\_{T+h} = \mu\_{T+h} + \varepsilon\_{T+h} back into the window, and returns the sampled paths as `r.y_future`.

The noise distribution is a `MultivariateNormal` over the series axis, so the future shocks are correlated across series exactly as the in-sample residuals are. `var_step(phi, intercept)` builds the step function from the sampled coefficients: the carry is the lag window with shape `(lags, series)` in natural time order (most recent row last), the mean is c + \sum_l \Phi_l y\_{t-l}, and the carry update drops the oldest row and appends the new one.


## Data layout and the first p observations

Time lives at axis `-2` and the series at axis `-1`, so the data is a `(time, 3)` array. The likelihood conditions on the first p = 2 rows, which seed the lag window (this is the conditional likelihood used by the blog post and by `statsmodels`): `y_init` holds these two rows, `data` holds the remaining 200 rows, and the forecast horizon is fixed by padding `data` with 30 zero rows through [pad_future](../../reference/arrays.pad_future.md#numpyro_forecast.arrays.pad_future). The model reads only the first `h.t_obs` rows of `covariates` (the block checks this), so the padding rows are never used as data. They only set the horizon.

We write the model as a **factory** that takes the prior on \Phi as an argument. The VAR code below never changes when we swap the prior in the last section.


``` python
p = 2
y_all = jnp.asarray(y_pct.to_numpy())  # (202, 3), float32
y_init = y_all[:p]  # the two rows that seed the lag window
data = y_all[p:]  # the 200 rows in the likelihood
future = 30
covariates_train = data  # fitting: no horizon
covariates_full = pad_future(data, future)  # forecasting: 30 unread rows fix the horizon
dates = y_pct.index[p:]
future_dates = pd.date_range(dates[-1], periods=future + 1, freq="QS")[1:]
time_coord = list(dates.append(future_dates))
print(f"y_init: {y_init.shape}, data: {data.shape}, covariates_full: {covariates_full.shape}")
print(f"forecast window: {future_dates[0].to_period('Q')} to {future_dates[-1].to_period('Q')}")
```


    y_init: (2, 3), data: (200, 3), covariates_full: (230, 3)
    forecast window: 2009Q4 to 2017Q1


``` python
def make_var_model(phi_prior: dist.Distribution, y_init: Array) -> ForecastModel:
    """Build an observed VAR model whose prior on the coefficients is ``phi_prior``.

    Parameters
    ----------
    phi_prior
        Prior distribution of the coefficient tensor, with event shape
        ``(lags, series, series)``.
    y_init
        The first ``lags`` rows of the series, which seed the lag window.

    Returns
    -------
    ForecastModel
        A plain ``(covariates, data=None)`` model function.
    """
    k = y_init.shape[-1]

    def var_model(covariates: Array, data: Array | None = None) -> None:
        h = Horizon.from_data(covariates, data)
        y = covariates[..., : h.t_obs, :]  # observed history only; never reads beyond t_obs

        # jnp.asarray only narrows numpyro's union return type for the type checker.
        intercept = jnp.asarray(
            numpyro.sample("intercept", dist.Normal(0.0, 1.0).expand([k]).to_event(1))
        )
        sigma = jnp.asarray(numpyro.sample("sigma", dist.HalfNormal(1.0).expand([k]).to_event(1)))
        l_omega = jnp.asarray(numpyro.sample("l_omega", dist.LKJCholesky(k, concentration=1.0)))
        phi = jnp.asarray(numpyro.sample("phi", phi_prior))
        scale_tril = sigma[..., :, None] * l_omega

        noise = dist.MultivariateNormal(jnp.zeros(k), scale_tril=scale_tril)
        r = ssoe(h, "eps", y, y_init, var_step(phi, intercept), noise)

        numpyro.deterministic("mu_t", r.mu)
        numpyro.sample("obs", dist.MultivariateNormal(r.mu, scale_tril=scale_tril), obs=h.data)
        if h.future > 0:
            numpyro.deterministic("forecast", r.y_future)

    return var_model


k = len(names)
weak_prior = dist.Normal(0.0, 1.0).expand([p, k, k]).to_event(3)
var_model = make_var_model(weak_prior, y_init)
```


# Inference with NUTS

We fit with four sequential NUTS chains of 1,000 warmup and 1,000 draws each. Fitting uses `covariates_train`, which has the same length as `data`, so the posterior holds only the parameters and the in-sample means. We pass the padded `covariates_full` to [to_datatree](../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree), which runs the posterior predictive for the 200 in-sample rows and the 30 forecast rows in one call and names every dimension.


``` python
def fit_nuts(rng_key: Array, model: ForecastModel, data: Array, covariates: Array) -> MCMC:
    """Fit ``model`` with NUTS (4 chains, 1,000 warmup and 1,000 draws each)."""
    mcmc = MCMC(
        NUTS(model),
        num_warmup=1_000,
        num_samples=1_000,
        num_chains=4,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(rng_key, covariates, data, extra_fields=("diverging",))
    return mcmc


def n_divergences(mcmc: MCMC) -> int:
    """Total number of divergent transitions across chains."""
    return int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())


# ``phi`` gets its own dimension names: ``az.summary`` mislabels the rows of a 3-D variable
# that shares a dimension (``series``) with 1-D variables (arviz 1.2).
coords = {
    "series": names,
    "equation": names,
    "lagged_series": names,
    "obs_dim": names,
    "lag": list(range(1, p + 1)),
}
posterior_dims = {
    "mu_t": ["time", "obs_dim"],
    "phi": ["lag", "equation", "lagged_series"],
    "intercept": ["series"],
    "sigma": ["series"],
}


def export(rng_key: Array, model: ForecastModel, posterior: dict[str, Array]) -> xr.DataTree:
    """Posterior, in-sample predictive and forecast draws as a labeled ArviZ tree."""
    return to_datatree(
        rng_key,
        model,
        posterior,
        data,
        covariates_full,
        num_chains=4,
        coords=coords,
        posterior_dims=posterior_dims,
        time_coord=time_coord,
    )
```


``` python
rng_key, rng_subkey = random.split(rng_key)
mcmc = fit_nuts(rng_subkey, var_model, data, covariates_train)
posterior = mcmc.get_samples()
print(f"divergences: {n_divergences(mcmc)}")

rng_key, rng_subkey = random.split(rng_key)
tree = export(rng_subkey, var_model, posterior)
tree
```


    divergences: 0


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
│       Dimensions:        (chain: 4, draw: 1000, series: 3, l_omega_dim_0: 3,
│                           l_omega_dim_1: 3, time: 200, obs_dim: 3, lag: 2,
│                           equation: 3, lagged_series: 3)
│       Coordinates:
│         * chain          (chain) int64 32B 0 1 2 3
│         * draw           (draw) int64 8kB 0 1 2 3 4 5 6 ... 994 995 996 997 998 999
│         * series         (series) <U8 96B 'realgdp' 'realcons' 'realinv'
│         * l_omega_dim_0  (l_omega_dim_0) int64 24B 0 1 2
│         * l_omega_dim_1  (l_omega_dim_1) int64 24B 0 1 2
│         * time           (time) datetime64[us] 2kB 1959-10-01 ... 2009-07-01
│         * obs_dim        (obs_dim) <U8 96B 'realgdp' 'realcons' 'realinv'
│         * lag            (lag) int64 16B 1 2
│         * equation       (equation) <U8 96B 'realgdp' 'realcons' 'realinv'
│         * lagged_series  (lagged_series) <U8 96B 'realgdp' 'realcons' 'realinv'
│       Data variables:
│           intercept      (chain, draw, series) float32 48kB 0.1031 0.4838 ... -1.274
│           l_omega        (chain, draw, l_omega_dim_0, l_omega_dim_1) float32 144kB ...
│           mu_t           (chain, draw, time, obs_dim) float32 10MB 1.23 ... -0.5934
│           phi            (chain, draw, lag, equation, lagged_series) float32 288kB ...
│           sigma          (chain, draw, series) float32 48kB 0.7407 0.6767 ... 3.953
│       Attributes:
│           created_at:                 2026-09-04T06:35:23.810579+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
├── Group: /posterior_predictive
│       Dimensions:  (chain: 4, draw: 1000, time: 200, obs_dim: 3)
│       Coordinates:
│         * chain    (chain) int64 32B 0 1 2 3
│         * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│         * time     (time) datetime64[us] 2kB 1959-10-01 1960-01-01 ... 2009-07-01
│         * obs_dim  (obs_dim) <U8 96B 'realgdp' 'realcons' 'realinv'
│       Data variables:
│           obs      (chain, draw, time, obs_dim) float32 10MB 2.553 1.829 ... -2.806
│       Attributes:
│           created_at:                 2026-09-04T06:35:24.034747+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
├── Group: /observed_data
│       Dimensions:  (time: 200, obs_dim: 3)
│       Coordinates:
│         * time     (time) datetime64[us] 2kB 1959-10-01 1960-01-01 ... 2009-07-01
│         * obs_dim  (obs_dim) <U8 96B 'realgdp' 'realcons' 'realinv'
│       Data variables:
│           obs      (time, obs_dim) float32 2kB 0.3495 0.1084 3.443 ... 0.7265 2.02
│       Attributes:
│           created_at:                 2026-09-04T06:35:24.035308+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /constant_data
│       Dimensions:        (time: 200, covariate_dim: 3)
│       Coordinates:
│         * time           (time) datetime64[us] 2kB 1959-10-01 ... 2009-07-01
│         * covariate_dim  (covariate_dim) int64 24B 0 1 2
│       Data variables:
│           covariates     (time, covariate_dim) float32 2kB 0.3495 0.1084 ... 2.02
│       Attributes:
│           created_at:                 2026-09-04T06:35:24.035800+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /predictions
│       Dimensions:  (chain: 4, draw: 1000, time: 30, obs_dim: 3)
│       Coordinates:
│         * chain    (chain) int64 32B 0 1 2 3
│         * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│         * time     (time) datetime64[us] 240B 2009-10-01 2010-01-01 ... 2017-01-01
│         * obs_dim  (obs_dim) <U8 96B 'realgdp' 'realcons' 'realinv'
│       Data variables:
│           obs      (chain, draw, time, obs_dim) float32 1MB 1.018 1.055 ... -2.136
│       Attributes:
│           created_at:                 2026-09-04T06:35:24.296777+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
└── Group: /predictions_constant_data
        Dimensions:        (time: 30, covariate_dim: 3)
        Coordinates:
          * time           (time) datetime64[us] 240B 2009-10-01 ... 2017-01-01
          * covariate_dim  (covariate_dim) int64 24B 0 1 2
        Data variables:
            covariates     (time, covariate_dim) float32 360B 0.0 0.0 0.0 ... 0.0 0.0
        Attributes:
            created_at:                 2026-09-04T06:35:24.297133+00:00
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
- series: 3
- l_omega_dim_0: 3
- l_omega_dim_1: 3
- time: 200
- obs_dim: 3
- lag: 2
- equation: 3
- lagged_series: 3


Coordinates: (10)


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


series


(series)


\<U8


'realgdp' 'realcons' 'realinv'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['realgdp', 'realcons', 'realinv'], dtype='<U8')


l_omega_dim_0


(l_omega_dim_0)


int64


0 1 2


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0, 1, 2])


l_omega_dim_1


(l_omega_dim_1)


int64


0 1 2


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0, 1, 2])


time


(time)


datetime64\[us\]


1959-10-01 ... 2009-07-01


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['1959-10-01T00:00:00.000000', '1960-01-01T00:00:00.000000','1960-04-01T00:00:00.000000', '1960-07-01T00:00:00.000000','1960-10-01T00:00:00.000000', '1961-01-01T00:00:00.000000','1961-04-01T00:00:00.000000', '1961-07-01T00:00:00.000000','1961-10-01T00:00:00.000000', '1962-01-01T00:00:00.000000','1962-04-01T00:00:00.000000', '1962-07-01T00:00:00.000000','1962-10-01T00:00:00.000000', '1963-01-01T00:00:00.000000','1963-04-01T00:00:00.000000', '1963-07-01T00:00:00.000000','1963-10-01T00:00:00.000000', '1964-01-01T00:00:00.000000','1964-04-01T00:00:00.000000', '1964-07-01T00:00:00.000000','1964-10-01T00:00:00.000000', '1965-01-01T00:00:00.000000','1965-04-01T00:00:00.000000', '1965-07-01T00:00:00.000000','1965-10-01T00:00:00.000000', '1966-01-01T00:00:00.000000','1966-04-01T00:00:00.000000', '1966-07-01T00:00:00.000000','1966-10-01T00:00:00.000000', '1967-01-01T00:00:00.000000','1967-04-01T00:00:00.000000', '1967-07-01T00:00:00.000000','1967-10-01T00:00:00.000000', '1968-01-01T00:00:00.000000','1968-04-01T00:00:00.000000', '1968-07-01T00:00:00.000000','1968-10-01T00:00:00.000000', '1969-01-01T00:00:00.000000','1969-04-01T00:00:00.000000', '1969-07-01T00:00:00.000000','1969-10-01T00:00:00.000000', '1970-01-01T00:00:00.000000','1970-04-01T00:00:00.000000', '1970-07-01T00:00:00.000000','1970-10-01T00:00:00.000000', '1971-01-01T00:00:00.000000','1971-04-01T00:00:00.000000', '1971-07-01T00:00:00.000000','1971-10-01T00:00:00.000000', '1972-01-01T00:00:00.000000','1972-04-01T00:00:00.000000', '1972-07-01T00:00:00.000000','1972-10-01T00:00:00.000000', '1973-01-01T00:00:00.000000','1973-04-01T00:00:00.000000', '1973-07-01T00:00:00.000000','1973-10-01T00:00:00.000000', '1974-01-01T00:00:00.000000','1974-04-01T00:00:00.000000', '1974-07-01T00:00:00.000000','1974-10-01T00:00:00.000000', '1975-01-01T00:00:00.000000','1975-04-01T00:00:00.000000', '1975-07-01T00:00:00.000000','1975-10-01T00:00:00.000000', '1976-01-01T00:00:00.000000','1976-04-01T00:00:00.000000', '1976-07-01T00:00:00.000000','1976-10-01T00:00:00.000000', '1977-01-01T00:00:00.000000','1977-04-01T00:00:00.000000', '1977-07-01T00:00:00.000000','1977-10-01T00:00:00.000000', '1978-01-01T00:00:00.000000','1978-04-01T00:00:00.000000', '1978-07-01T00:00:00.000000','1978-10-01T00:00:00.000000', '1979-01-01T00:00:00.000000','1979-04-01T00:00:00.000000', '1979-07-01T00:00:00.000000','1979-10-01T00:00:00.000000', '1980-01-01T00:00:00.000000','1980-04-01T00:00:00.000000', '1980-07-01T00:00:00.000000','1980-10-01T00:00:00.000000', '1981-01-01T00:00:00.000000','1981-04-01T00:00:00.000000', '1981-07-01T00:00:00.000000','1981-10-01T00:00:00.000000', '1982-01-01T00:00:00.000000','1982-04-01T00:00:00.000000', '1982-07-01T00:00:00.000000','1982-10-01T00:00:00.000000', '1983-01-01T00:00:00.000000','1983-04-01T00:00:00.000000', '1983-07-01T00:00:00.000000','1983-10-01T00:00:00.000000', '1984-01-01T00:00:00.000000','1984-04-01T00:00:00.000000', '1984-07-01T00:00:00.000000','1984-10-01T00:00:00.000000', '1985-01-01T00:00:00.000000','1985-04-01T00:00:00.000000', '1985-07-01T00:00:00.000000','1985-10-01T00:00:00.000000', '1986-01-01T00:00:00.000000','1986-04-01T00:00:00.000000', '1986-07-01T00:00:00.000000','1986-10-01T00:00:00.000000', '1987-01-01T00:00:00.000000','1987-04-01T00:00:00.000000', '1987-07-01T00:00:00.000000','1987-10-01T00:00:00.000000', '1988-01-01T00:00:00.000000','1988-04-01T00:00:00.000000', '1988-07-01T00:00:00.000000','1988-10-01T00:00:00.000000', '1989-01-01T00:00:00.000000','1989-04-01T00:00:00.000000', '1989-07-01T00:00:00.000000','1989-10-01T00:00:00.000000', '1990-01-01T00:00:00.000000','1990-04-01T00:00:00.000000', '1990-07-01T00:00:00.000000','1990-10-01T00:00:00.000000', '1991-01-01T00:00:00.000000','1991-04-01T00:00:00.000000', '1991-07-01T00:00:00.000000','1991-10-01T00:00:00.000000', '1992-01-01T00:00:00.000000','1992-04-01T00:00:00.000000', '1992-07-01T00:00:00.000000','1992-10-01T00:00:00.000000', '1993-01-01T00:00:00.000000','1993-04-01T00:00:00.000000', '1993-07-01T00:00:00.000000','1993-10-01T00:00:00.000000', '1994-01-01T00:00:00.000000','1994-04-01T00:00:00.000000', '1994-07-01T00:00:00.000000','1994-10-01T00:00:00.000000', '1995-01-01T00:00:00.000000','1995-04-01T00:00:00.000000', '1995-07-01T00:00:00.000000','1995-10-01T00:00:00.000000', '1996-01-01T00:00:00.000000','1996-04-01T00:00:00.000000', '1996-07-01T00:00:00.000000','1996-10-01T00:00:00.000000', '1997-01-01T00:00:00.000000','1997-04-01T00:00:00.000000', '1997-07-01T00:00:00.000000','1997-10-01T00:00:00.000000', '1998-01-01T00:00:00.000000','1998-04-01T00:00:00.000000', '1998-07-01T00:00:00.000000','1998-10-01T00:00:00.000000', '1999-01-01T00:00:00.000000','1999-04-01T00:00:00.000000', '1999-07-01T00:00:00.000000','1999-10-01T00:00:00.000000', '2000-01-01T00:00:00.000000','2000-04-01T00:00:00.000000', '2000-07-01T00:00:00.000000','2000-10-01T00:00:00.000000', '2001-01-01T00:00:00.000000','2001-04-01T00:00:00.000000', '2001-07-01T00:00:00.000000','2001-10-01T00:00:00.000000', '2002-01-01T00:00:00.000000','2002-04-01T00:00:00.000000', '2002-07-01T00:00:00.000000','2002-10-01T00:00:00.000000', '2003-01-01T00:00:00.000000','2003-04-01T00:00:00.000000', '2003-07-01T00:00:00.000000','2003-10-01T00:00:00.000000', '2004-01-01T00:00:00.000000','2004-04-01T00:00:00.000000', '2004-07-01T00:00:00.000000','2004-10-01T00:00:00.000000', '2005-01-01T00:00:00.000000','2005-04-01T00:00:00.000000', '2005-07-01T00:00:00.000000','2005-10-01T00:00:00.000000', '2006-01-01T00:00:00.000000','2006-04-01T00:00:00.000000', '2006-07-01T00:00:00.000000','2006-10-01T00:00:00.000000', '2007-01-01T00:00:00.000000','2007-04-01T00:00:00.000000', '2007-07-01T00:00:00.000000','2007-10-01T00:00:00.000000', '2008-01-01T00:00:00.000000','2008-04-01T00:00:00.000000', '2008-07-01T00:00:00.000000','2008-10-01T00:00:00.000000', '2009-01-01T00:00:00.000000','2009-04-01T00:00:00.000000', '2009-07-01T00:00:00.000000'],dtype='datetime64[us]')


obs_dim


(obs_dim)


\<U8


'realgdp' 'realcons' 'realinv'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['realgdp', 'realcons', 'realinv'], dtype='<U8')


lag


(lag)


int64


1 2


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([1, 2])


equation


(equation)


\<U8


'realgdp' 'realcons' 'realinv'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['realgdp', 'realcons', 'realinv'], dtype='<U8')


lagged_series


(lagged_series)


\<U8


'realgdp' 'realcons' 'realinv'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['realgdp', 'realcons', 'realinv'], dtype='<U8')


Data variables: (5)


intercept


(chain, draw, series)


float32


0.1031 0.4838 ... 0.5024 -1.274


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 0.10312091,  0.48383945, -2.5184457 ],[ 0.08537946,  0.41749915, -2.0667188 ],[ 0.3015044 ,  0.5285377 , -1.1337879 ],...,[ 0.22957109,  0.53891104, -1.15649   ],[ 0.20048736,  0.58098143, -1.8298321 ],[ 0.11979293,  0.4759067 , -2.026041  ]],[[ 0.387278  ,  0.63216156, -1.1776836 ],[ 0.34054992,  0.5441619 , -0.9577487 ],[ 0.15526173,  0.5583605 , -2.132862  ],...,[ 0.14866434,  0.45134223, -2.133964  ],[ 0.25439677,  0.5560107 , -1.618279  ],[ 0.34654117,  0.6562711 , -1.7046548 ]],[[ 0.35008496,  0.58374935, -1.5304705 ],[ 0.26071352,  0.5124433 , -2.0216498 ],[ 0.33600745,  0.62275714, -1.4704986 ],...,[ 0.01731343,  0.38538572, -2.4541547 ],[ 0.0379752 ,  0.42113495, -2.724741  ],[ 0.420493  ,  0.6779011 , -0.8690811 ]],[[ 0.20681006,  0.491232  , -1.6401768 ],[ 0.25413808,  0.5938929 , -2.0477955 ],[ 0.12585647,  0.5517798 , -2.1647048 ],...,[ 0.0847002 ,  0.36319292, -2.5300589 ],[ 0.4411324 ,  0.6868283 , -0.98628974],[ 0.27970895,  0.5024341 , -1.2744097 ]]],shape=(4, 1000, 3), dtype=float32)


l_omega


(chain, draw, l_omega_dim_0, l_omega_dim_1)


float32


1.0 0.0 0.0 ... -0.4702 0.5248


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 1.        ,  0.        ,  0.        ],[ 0.6417907 ,  0.76687986,  0.        ],[ 0.75763804, -0.40630215,  0.51078683]],[[ 1.        ,  0.        ,  0.        ],[ 0.53907204,  0.84225965,  0.        ],[ 0.774807  , -0.38447943,  0.5018463 ]],[[ 1.        ,  0.        ,  0.        ],[ 0.61685836,  0.78707415,  0.        ],[ 0.70640534, -0.46272662,  0.53560764]],...,[[ 1.        ,  0.        ,  0.        ],[ 0.5747251 ,  0.81834656,  0.        ],[ 0.7569417 , -0.35954192,  0.5456821 ]],[[ 1.        ,  0.        ,  0.        ],[ 0.6026404 ,  0.79801285,  0.        ],...[ 0.7224597 , -0.3626164 ,  0.58869463]],[[ 1.        ,  0.        ,  0.        ],[ 0.5924085 ,  0.8056378 ,  0.        ],[ 0.73225075, -0.39898995,  0.5519202 ]],...,[[ 1.        ,  0.        ,  0.        ],[ 0.5700733 ,  0.8215938 ,  0.        ],[ 0.680265  , -0.45527256,  0.574427  ]],[[ 1.        ,  0.        ,  0.        ],[ 0.5168506 ,  0.8560756 ,  0.        ],[ 0.7579728 , -0.43410873,  0.48685405]],[[ 1.        ,  0.        ,  0.        ],[ 0.5734395 ,  0.8192479 ,  0.        ],[ 0.7096048 , -0.47015053,  0.5248043 ]]]],shape=(4, 1000, 3, 3), dtype=float32)


mu_t


(chain, draw, time, obs_dim)


float32


1.23 0.937 2.923 ... 0.2422 -0.5934


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 1.22957206e+00,  9.36951041e-01,  2.92324996e+00],[ 4.09523547e-01,  5.84592879e-01, -1.49780476e+00],[ 6.80289149e-01,  9.14911807e-01,  5.82918644e-01],...,[-5.63477278e-01,  6.59755468e-02, -5.77602768e+00],[-2.15152189e-01,  8.62606764e-02, -3.74728346e+00],[ 1.07512325e-02,  1.71357602e-01, -3.92661452e+00]],[[ 1.15002215e+00,  1.02036440e+00,  2.25750756e+00],[ 4.17224407e-01,  6.66727960e-01, -8.49488258e-01],[ 6.09809577e-01,  8.04276705e-01,  6.65876389e-01],...,[-6.06445909e-01, -7.09824860e-02, -5.24986076e+00],[-3.88945013e-01, -1.59536630e-01, -3.38467455e+00],[-2.54566520e-01,  4.28846180e-02, -3.57793760e+00]],[[ 9.26127553e-01,  8.57550144e-01,  9.52251911e-01],[ 8.25535417e-01,  8.25240314e-01,  1.14423037e+00],[ 8.07917237e-01,  9.57043290e-01,  1.81004906e+00],...,...[-6.71539724e-01, -1.72713101e-01, -6.04820919e+00],[-2.49773532e-01,  3.52184176e-02, -2.92290854e+00],[-1.99985653e-01,  1.19639874e-01, -2.49314642e+00]],[[ 1.03318667e+00,  1.03149843e+00,  1.57637000e+00],[ 6.17027521e-01,  8.11185420e-01, -2.11775422e-01],[ 8.29617202e-01,  9.59127665e-01,  1.45605183e+00],...,[-6.51000440e-02,  2.87214845e-01, -3.69212294e+00],[ 2.73332238e-01,  1.63424850e-01, -1.77464402e+00],[ 6.36986792e-02, -7.55756497e-02, -2.90524244e+00]],[[ 1.00546420e+00,  9.81077909e-01,  1.61773241e+00],[ 6.31943345e-01,  5.92866421e-01,  3.05088401e-01],[ 8.25121164e-01,  8.24242294e-01,  1.58839142e+00],...,[-2.75592357e-01,  1.65059656e-01, -3.90044546e+00],[-1.05553180e-01,  1.50159180e-01, -2.42465210e+00],[ 4.24223453e-01,  2.42201090e-01, -5.93417764e-01]]]],shape=(4, 1000, 200, 3), dtype=float32)


phi


(chain, draw, lag, equation, lagged_series)


float32


0.1593 0.3327 ... -0.4188 -0.2306


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[[ 1.59264028e-01,  3.32699150e-01, -1.34624327e-02],[ 1.07292481e-01,  1.18322954e-01,  3.79041128e-04],[ 3.95922083e-03,  2.82214737e+00,  1.98339317e-02]],[[ 2.26703167e-01,  1.70558125e-01, -1.53789883e-02],[-4.01742682e-02,  1.84553012e-01,  2.04267744e-02],[ 1.05739319e+00,  3.25832337e-01, -6.00002594e-02]]],[[[-1.41351623e-02,  3.96661490e-01,  9.47317760e-03],[-8.28321874e-02,  2.50918269e-01,  2.69565582e-02],[-3.09514701e-02,  2.30549884e+00,  2.28634123e-02]],[[ 9.32457894e-02,  2.94214964e-01,  4.61282488e-03],[ 1.17474094e-01,  1.58482254e-01, -1.05245854e-03],[-4.26157206e-01,  1.52689993e+00,  1.02210209e-01]]],[[[-9.30558071e-02,  4.89522547e-01,  2.60236524e-02],[-8.86477083e-02,  2.97262341e-01,  3.36582996e-02],...[ 1.90733954e-01,  1.05727708e+00, -5.76436855e-02]]],[[[-9.54011306e-02,  4.19936031e-01,  1.01843560e-02],[-4.06555235e-01,  4.81446862e-01,  6.07424416e-02],[-3.60088348e-01,  2.33947492e+00,  5.48449531e-02]],[[-3.15289855e-01,  4.11998808e-01,  4.67013195e-02],[-1.68386012e-01,  2.37609491e-01,  3.62860560e-02],[-1.85074055e+00,  1.99934947e+00,  2.55002826e-01]]],[[[ 1.99789032e-01,  1.93346709e-01, -5.50857233e-03],[ 8.34673047e-02,  8.26116949e-02, -4.42230783e-04],[ 2.26533547e-01,  2.18620157e+00,  5.98717704e-02]],[[ 3.20655614e-01,  2.37151906e-02, -4.07652408e-02],[ 6.42801300e-02,  1.14650473e-01,  7.98210874e-03],[ 1.43132532e+00, -4.18848872e-01, -2.30553299e-01]]]]],shape=(4, 1000, 2, 3, 3), dtype=float32)


sigma


(chain, draw, series)


float32


0.7407 0.6767 ... 0.6743 3.953


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.7406692 , 0.67667633, 3.6273296 ],[0.7374479 , 0.6556411 , 3.9118762 ],[0.7568117 , 0.6377874 , 4.032502  ],...,[0.768365  , 0.65113413, 4.11802   ],[0.71377975, 0.6631159 , 3.5993726 ],[0.7242116 , 0.67774135, 3.7518559 ]],[[0.7694906 , 0.6708486 , 4.0801773 ],[0.72124636, 0.63928235, 4.02017   ],[0.77961993, 0.66288424, 3.9700599 ],...,[0.6961043 , 0.60509014, 3.7456017 ],[0.73252666, 0.6553079 , 3.8596523 ],[0.7212443 , 0.5878971 , 3.9361038 ]],[[0.77223045, 0.6382302 , 4.0157895 ],[0.6717635 , 0.6186336 , 3.761094  ],[0.72607267, 0.6460416 , 3.670212  ],...,[0.7017207 , 0.64243066, 3.6822202 ],[0.7222378 , 0.6235499 , 3.6844697 ],[0.7777623 , 0.6508724 , 4.0989356 ]],[[0.7354738 , 0.6551204 , 3.7963169 ],[0.77205557, 0.6512871 , 3.9435248 ],[0.70560795, 0.6392633 , 3.6499646 ],...,[0.73183626, 0.6721806 , 4.0497727 ],[0.7197319 , 0.62728393, 3.8620677 ],[0.7030298 , 0.67425275, 3.95273   ]]],shape=(4, 1000, 3), dtype=float32)


Attributes: (5)


created_at :  
2026-09-04T06:35:23.810579+00:00

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
- time: 200
- obs_dim: 3


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


datetime64\[us\]


1959-10-01 ... 2009-07-01


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['1959-10-01T00:00:00.000000', '1960-01-01T00:00:00.000000','1960-04-01T00:00:00.000000', '1960-07-01T00:00:00.000000','1960-10-01T00:00:00.000000', '1961-01-01T00:00:00.000000','1961-04-01T00:00:00.000000', '1961-07-01T00:00:00.000000','1961-10-01T00:00:00.000000', '1962-01-01T00:00:00.000000','1962-04-01T00:00:00.000000', '1962-07-01T00:00:00.000000','1962-10-01T00:00:00.000000', '1963-01-01T00:00:00.000000','1963-04-01T00:00:00.000000', '1963-07-01T00:00:00.000000','1963-10-01T00:00:00.000000', '1964-01-01T00:00:00.000000','1964-04-01T00:00:00.000000', '1964-07-01T00:00:00.000000','1964-10-01T00:00:00.000000', '1965-01-01T00:00:00.000000','1965-04-01T00:00:00.000000', '1965-07-01T00:00:00.000000','1965-10-01T00:00:00.000000', '1966-01-01T00:00:00.000000','1966-04-01T00:00:00.000000', '1966-07-01T00:00:00.000000','1966-10-01T00:00:00.000000', '1967-01-01T00:00:00.000000','1967-04-01T00:00:00.000000', '1967-07-01T00:00:00.000000','1967-10-01T00:00:00.000000', '1968-01-01T00:00:00.000000','1968-04-01T00:00:00.000000', '1968-07-01T00:00:00.000000','1968-10-01T00:00:00.000000', '1969-01-01T00:00:00.000000','1969-04-01T00:00:00.000000', '1969-07-01T00:00:00.000000','1969-10-01T00:00:00.000000', '1970-01-01T00:00:00.000000','1970-04-01T00:00:00.000000', '1970-07-01T00:00:00.000000','1970-10-01T00:00:00.000000', '1971-01-01T00:00:00.000000','1971-04-01T00:00:00.000000', '1971-07-01T00:00:00.000000','1971-10-01T00:00:00.000000', '1972-01-01T00:00:00.000000','1972-04-01T00:00:00.000000', '1972-07-01T00:00:00.000000','1972-10-01T00:00:00.000000', '1973-01-01T00:00:00.000000','1973-04-01T00:00:00.000000', '1973-07-01T00:00:00.000000','1973-10-01T00:00:00.000000', '1974-01-01T00:00:00.000000','1974-04-01T00:00:00.000000', '1974-07-01T00:00:00.000000','1974-10-01T00:00:00.000000', '1975-01-01T00:00:00.000000','1975-04-01T00:00:00.000000', '1975-07-01T00:00:00.000000','1975-10-01T00:00:00.000000', '1976-01-01T00:00:00.000000','1976-04-01T00:00:00.000000', '1976-07-01T00:00:00.000000','1976-10-01T00:00:00.000000', '1977-01-01T00:00:00.000000','1977-04-01T00:00:00.000000', '1977-07-01T00:00:00.000000','1977-10-01T00:00:00.000000', '1978-01-01T00:00:00.000000','1978-04-01T00:00:00.000000', '1978-07-01T00:00:00.000000','1978-10-01T00:00:00.000000', '1979-01-01T00:00:00.000000','1979-04-01T00:00:00.000000', '1979-07-01T00:00:00.000000','1979-10-01T00:00:00.000000', '1980-01-01T00:00:00.000000','1980-04-01T00:00:00.000000', '1980-07-01T00:00:00.000000','1980-10-01T00:00:00.000000', '1981-01-01T00:00:00.000000','1981-04-01T00:00:00.000000', '1981-07-01T00:00:00.000000','1981-10-01T00:00:00.000000', '1982-01-01T00:00:00.000000','1982-04-01T00:00:00.000000', '1982-07-01T00:00:00.000000','1982-10-01T00:00:00.000000', '1983-01-01T00:00:00.000000','1983-04-01T00:00:00.000000', '1983-07-01T00:00:00.000000','1983-10-01T00:00:00.000000', '1984-01-01T00:00:00.000000','1984-04-01T00:00:00.000000', '1984-07-01T00:00:00.000000','1984-10-01T00:00:00.000000', '1985-01-01T00:00:00.000000','1985-04-01T00:00:00.000000', '1985-07-01T00:00:00.000000','1985-10-01T00:00:00.000000', '1986-01-01T00:00:00.000000','1986-04-01T00:00:00.000000', '1986-07-01T00:00:00.000000','1986-10-01T00:00:00.000000', '1987-01-01T00:00:00.000000','1987-04-01T00:00:00.000000', '1987-07-01T00:00:00.000000','1987-10-01T00:00:00.000000', '1988-01-01T00:00:00.000000','1988-04-01T00:00:00.000000', '1988-07-01T00:00:00.000000','1988-10-01T00:00:00.000000', '1989-01-01T00:00:00.000000','1989-04-01T00:00:00.000000', '1989-07-01T00:00:00.000000','1989-10-01T00:00:00.000000', '1990-01-01T00:00:00.000000','1990-04-01T00:00:00.000000', '1990-07-01T00:00:00.000000','1990-10-01T00:00:00.000000', '1991-01-01T00:00:00.000000','1991-04-01T00:00:00.000000', '1991-07-01T00:00:00.000000','1991-10-01T00:00:00.000000', '1992-01-01T00:00:00.000000','1992-04-01T00:00:00.000000', '1992-07-01T00:00:00.000000','1992-10-01T00:00:00.000000', '1993-01-01T00:00:00.000000','1993-04-01T00:00:00.000000', '1993-07-01T00:00:00.000000','1993-10-01T00:00:00.000000', '1994-01-01T00:00:00.000000','1994-04-01T00:00:00.000000', '1994-07-01T00:00:00.000000','1994-10-01T00:00:00.000000', '1995-01-01T00:00:00.000000','1995-04-01T00:00:00.000000', '1995-07-01T00:00:00.000000','1995-10-01T00:00:00.000000', '1996-01-01T00:00:00.000000','1996-04-01T00:00:00.000000', '1996-07-01T00:00:00.000000','1996-10-01T00:00:00.000000', '1997-01-01T00:00:00.000000','1997-04-01T00:00:00.000000', '1997-07-01T00:00:00.000000','1997-10-01T00:00:00.000000', '1998-01-01T00:00:00.000000','1998-04-01T00:00:00.000000', '1998-07-01T00:00:00.000000','1998-10-01T00:00:00.000000', '1999-01-01T00:00:00.000000','1999-04-01T00:00:00.000000', '1999-07-01T00:00:00.000000','1999-10-01T00:00:00.000000', '2000-01-01T00:00:00.000000','2000-04-01T00:00:00.000000', '2000-07-01T00:00:00.000000','2000-10-01T00:00:00.000000', '2001-01-01T00:00:00.000000','2001-04-01T00:00:00.000000', '2001-07-01T00:00:00.000000','2001-10-01T00:00:00.000000', '2002-01-01T00:00:00.000000','2002-04-01T00:00:00.000000', '2002-07-01T00:00:00.000000','2002-10-01T00:00:00.000000', '2003-01-01T00:00:00.000000','2003-04-01T00:00:00.000000', '2003-07-01T00:00:00.000000','2003-10-01T00:00:00.000000', '2004-01-01T00:00:00.000000','2004-04-01T00:00:00.000000', '2004-07-01T00:00:00.000000','2004-10-01T00:00:00.000000', '2005-01-01T00:00:00.000000','2005-04-01T00:00:00.000000', '2005-07-01T00:00:00.000000','2005-10-01T00:00:00.000000', '2006-01-01T00:00:00.000000','2006-04-01T00:00:00.000000', '2006-07-01T00:00:00.000000','2006-10-01T00:00:00.000000', '2007-01-01T00:00:00.000000','2007-04-01T00:00:00.000000', '2007-07-01T00:00:00.000000','2007-10-01T00:00:00.000000', '2008-01-01T00:00:00.000000','2008-04-01T00:00:00.000000', '2008-07-01T00:00:00.000000','2008-10-01T00:00:00.000000', '2009-01-01T00:00:00.000000','2009-04-01T00:00:00.000000', '2009-07-01T00:00:00.000000'],dtype='datetime64[us]')


obs_dim


(obs_dim)


\<U8


'realgdp' 'realcons' 'realinv'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['realgdp', 'realcons', 'realinv'], dtype='<U8')


Data variables: (1)


obs


(chain, draw, time, obs_dim)


float32


2.553 1.829 4.206 ... 0.1912 -2.806


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 2.55257416e+00,  1.82891822e+00,  4.20636845e+00],[ 8.53146672e-01,  2.60665327e-01,  1.82302225e+00],[ 6.20439112e-01,  9.39159095e-01, -8.61024976e-01],...,[-3.10105145e-01,  1.90878034e-01,  3.18040848e-02],[ 8.75406444e-01,  3.71257335e-01,  4.11027431e-01],[ 2.72654414e-01,  8.46521497e-01, -4.98561192e+00]],[[-4.46777344e-02, -1.34004831e-01, -2.79577494e-01],[ 1.06404674e+00,  1.16493189e+00,  3.95359325e+00],[ 9.20493066e-01,  1.55999768e+00,  3.00515151e+00],...,[-1.77613413e+00,  2.61677474e-01, -1.04782982e+01],[ 3.94291431e-01,  3.14649373e-01, -1.15585995e+00],[ 1.73926353e-04,  3.44963968e-01,  4.24232006e-01]],[[ 9.77861881e-02, -9.16321874e-02, -2.01237440e+00],[-2.95656919e-03,  3.80850524e-01, -3.88230276e+00],[ 1.12679172e+00,  1.19775343e+00, -4.63624239e-01],...,...[-7.85625279e-01,  7.17735946e-01, -8.73674393e+00],[-3.05579066e-01,  2.19690353e-01, -5.61023712e+00],[-1.03030455e+00,  8.42356741e-01, -1.04273682e+01]],[[ 1.13297510e+00,  1.16376901e+00,  9.57052588e-01],[ 1.25400400e+00,  1.60922110e-01,  6.12456274e+00],[ 7.14956641e-01,  1.22657061e+00,  1.54611742e+00],...,[ 3.97115946e-03,  4.28220212e-01, -5.45101929e+00],[-2.11458504e-02,  5.01457676e-02, -6.76644802e-01],[ 6.92328691e-01,  8.06323707e-01, -5.01065493e-01]],[[ 2.60350943e+00,  2.61257887e+00,  3.84846926e+00],[ 6.75936937e-02,  7.36723423e-01, -6.39577675e+00],[ 1.91653252e-01,  7.21629381e-01, -3.08792830e+00],...,[ 1.43772638e+00,  1.20329142e+00,  7.97855377e-01],[ 1.28967571e+00,  1.81867361e+00,  2.63391066e+00],[ 8.78272951e-02,  1.91163883e-01, -2.80562782e+00]]]],shape=(4, 1000, 200, 3), dtype=float32)


Attributes: (5)


created_at :  
2026-09-04T06:35:24.034747+00:00

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


- time: 200
- obs_dim: 3


Coordinates: (2)


time


(time)


datetime64\[us\]


1959-10-01 ... 2009-07-01


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['1959-10-01T00:00:00.000000', '1960-01-01T00:00:00.000000','1960-04-01T00:00:00.000000', '1960-07-01T00:00:00.000000','1960-10-01T00:00:00.000000', '1961-01-01T00:00:00.000000','1961-04-01T00:00:00.000000', '1961-07-01T00:00:00.000000','1961-10-01T00:00:00.000000', '1962-01-01T00:00:00.000000','1962-04-01T00:00:00.000000', '1962-07-01T00:00:00.000000','1962-10-01T00:00:00.000000', '1963-01-01T00:00:00.000000','1963-04-01T00:00:00.000000', '1963-07-01T00:00:00.000000','1963-10-01T00:00:00.000000', '1964-01-01T00:00:00.000000','1964-04-01T00:00:00.000000', '1964-07-01T00:00:00.000000','1964-10-01T00:00:00.000000', '1965-01-01T00:00:00.000000','1965-04-01T00:00:00.000000', '1965-07-01T00:00:00.000000','1965-10-01T00:00:00.000000', '1966-01-01T00:00:00.000000','1966-04-01T00:00:00.000000', '1966-07-01T00:00:00.000000','1966-10-01T00:00:00.000000', '1967-01-01T00:00:00.000000','1967-04-01T00:00:00.000000', '1967-07-01T00:00:00.000000','1967-10-01T00:00:00.000000', '1968-01-01T00:00:00.000000','1968-04-01T00:00:00.000000', '1968-07-01T00:00:00.000000','1968-10-01T00:00:00.000000', '1969-01-01T00:00:00.000000','1969-04-01T00:00:00.000000', '1969-07-01T00:00:00.000000','1969-10-01T00:00:00.000000', '1970-01-01T00:00:00.000000','1970-04-01T00:00:00.000000', '1970-07-01T00:00:00.000000','1970-10-01T00:00:00.000000', '1971-01-01T00:00:00.000000','1971-04-01T00:00:00.000000', '1971-07-01T00:00:00.000000','1971-10-01T00:00:00.000000', '1972-01-01T00:00:00.000000','1972-04-01T00:00:00.000000', '1972-07-01T00:00:00.000000','1972-10-01T00:00:00.000000', '1973-01-01T00:00:00.000000','1973-04-01T00:00:00.000000', '1973-07-01T00:00:00.000000','1973-10-01T00:00:00.000000', '1974-01-01T00:00:00.000000','1974-04-01T00:00:00.000000', '1974-07-01T00:00:00.000000','1974-10-01T00:00:00.000000', '1975-01-01T00:00:00.000000','1975-04-01T00:00:00.000000', '1975-07-01T00:00:00.000000','1975-10-01T00:00:00.000000', '1976-01-01T00:00:00.000000','1976-04-01T00:00:00.000000', '1976-07-01T00:00:00.000000','1976-10-01T00:00:00.000000', '1977-01-01T00:00:00.000000','1977-04-01T00:00:00.000000', '1977-07-01T00:00:00.000000','1977-10-01T00:00:00.000000', '1978-01-01T00:00:00.000000','1978-04-01T00:00:00.000000', '1978-07-01T00:00:00.000000','1978-10-01T00:00:00.000000', '1979-01-01T00:00:00.000000','1979-04-01T00:00:00.000000', '1979-07-01T00:00:00.000000','1979-10-01T00:00:00.000000', '1980-01-01T00:00:00.000000','1980-04-01T00:00:00.000000', '1980-07-01T00:00:00.000000','1980-10-01T00:00:00.000000', '1981-01-01T00:00:00.000000','1981-04-01T00:00:00.000000', '1981-07-01T00:00:00.000000','1981-10-01T00:00:00.000000', '1982-01-01T00:00:00.000000','1982-04-01T00:00:00.000000', '1982-07-01T00:00:00.000000','1982-10-01T00:00:00.000000', '1983-01-01T00:00:00.000000','1983-04-01T00:00:00.000000', '1983-07-01T00:00:00.000000','1983-10-01T00:00:00.000000', '1984-01-01T00:00:00.000000','1984-04-01T00:00:00.000000', '1984-07-01T00:00:00.000000','1984-10-01T00:00:00.000000', '1985-01-01T00:00:00.000000','1985-04-01T00:00:00.000000', '1985-07-01T00:00:00.000000','1985-10-01T00:00:00.000000', '1986-01-01T00:00:00.000000','1986-04-01T00:00:00.000000', '1986-07-01T00:00:00.000000','1986-10-01T00:00:00.000000', '1987-01-01T00:00:00.000000','1987-04-01T00:00:00.000000', '1987-07-01T00:00:00.000000','1987-10-01T00:00:00.000000', '1988-01-01T00:00:00.000000','1988-04-01T00:00:00.000000', '1988-07-01T00:00:00.000000','1988-10-01T00:00:00.000000', '1989-01-01T00:00:00.000000','1989-04-01T00:00:00.000000', '1989-07-01T00:00:00.000000','1989-10-01T00:00:00.000000', '1990-01-01T00:00:00.000000','1990-04-01T00:00:00.000000', '1990-07-01T00:00:00.000000','1990-10-01T00:00:00.000000', '1991-01-01T00:00:00.000000','1991-04-01T00:00:00.000000', '1991-07-01T00:00:00.000000','1991-10-01T00:00:00.000000', '1992-01-01T00:00:00.000000','1992-04-01T00:00:00.000000', '1992-07-01T00:00:00.000000','1992-10-01T00:00:00.000000', '1993-01-01T00:00:00.000000','1993-04-01T00:00:00.000000', '1993-07-01T00:00:00.000000','1993-10-01T00:00:00.000000', '1994-01-01T00:00:00.000000','1994-04-01T00:00:00.000000', '1994-07-01T00:00:00.000000','1994-10-01T00:00:00.000000', '1995-01-01T00:00:00.000000','1995-04-01T00:00:00.000000', '1995-07-01T00:00:00.000000','1995-10-01T00:00:00.000000', '1996-01-01T00:00:00.000000','1996-04-01T00:00:00.000000', '1996-07-01T00:00:00.000000','1996-10-01T00:00:00.000000', '1997-01-01T00:00:00.000000','1997-04-01T00:00:00.000000', '1997-07-01T00:00:00.000000','1997-10-01T00:00:00.000000', '1998-01-01T00:00:00.000000','1998-04-01T00:00:00.000000', '1998-07-01T00:00:00.000000','1998-10-01T00:00:00.000000', '1999-01-01T00:00:00.000000','1999-04-01T00:00:00.000000', '1999-07-01T00:00:00.000000','1999-10-01T00:00:00.000000', '2000-01-01T00:00:00.000000','2000-04-01T00:00:00.000000', '2000-07-01T00:00:00.000000','2000-10-01T00:00:00.000000', '2001-01-01T00:00:00.000000','2001-04-01T00:00:00.000000', '2001-07-01T00:00:00.000000','2001-10-01T00:00:00.000000', '2002-01-01T00:00:00.000000','2002-04-01T00:00:00.000000', '2002-07-01T00:00:00.000000','2002-10-01T00:00:00.000000', '2003-01-01T00:00:00.000000','2003-04-01T00:00:00.000000', '2003-07-01T00:00:00.000000','2003-10-01T00:00:00.000000', '2004-01-01T00:00:00.000000','2004-04-01T00:00:00.000000', '2004-07-01T00:00:00.000000','2004-10-01T00:00:00.000000', '2005-01-01T00:00:00.000000','2005-04-01T00:00:00.000000', '2005-07-01T00:00:00.000000','2005-10-01T00:00:00.000000', '2006-01-01T00:00:00.000000','2006-04-01T00:00:00.000000', '2006-07-01T00:00:00.000000','2006-10-01T00:00:00.000000', '2007-01-01T00:00:00.000000','2007-04-01T00:00:00.000000', '2007-07-01T00:00:00.000000','2007-10-01T00:00:00.000000', '2008-01-01T00:00:00.000000','2008-04-01T00:00:00.000000', '2008-07-01T00:00:00.000000','2008-10-01T00:00:00.000000', '2009-01-01T00:00:00.000000','2009-04-01T00:00:00.000000', '2009-07-01T00:00:00.000000'],dtype='datetime64[us]')


obs_dim


(obs_dim)


\<U8


'realgdp' 'realcons' 'realinv'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['realgdp', 'realcons', 'realinv'], dtype='<U8')


Data variables: (1)


obs


(time, obs_dim)


float32


0.3495 0.1084 3.443 ... 0.7265 2.02


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[ 3.49453270e-01,  1.08401097e-01,  3.44251108e+00],[ 2.21901798e+00,  9.53415096e-01,  1.02663765e+01],[-4.68455315e-01,  1.25724280e+00, -1.06693850e+01],[ 1.63288012e-01, -3.96792650e-01, -5.97787917e-01],[-1.29063594e+00,  1.34303316e-01, -1.31852016e+01],[ 5.92259109e-01, -2.79649887e-02,  2.52441812e+00],[ 1.85345340e+00,  1.47698414e+00,  7.18338728e+00],[ 1.60316396e+00,  4.83863056e-01,  8.04527092e+00],[ 2.01528168e+00,  1.98230624e+00,  1.67371130e+00],[ 1.77772593e+00,  1.05911660e+00,  5.79106426e+00],[ 1.09805155e+00,  1.22162342e+00, -9.71584797e-01],[ 9.20406699e-01,  8.06202650e-01,  1.77339709e+00],[ 2.42701873e-01,  1.40825522e+00, -3.41469789e+00],[ 1.29852104e+00,  6.71229422e-01,  5.40070963e+00],[ 1.24528348e+00,  9.50427711e-01,  1.44677019e+00],[ 1.86540413e+00,  1.35154164e+00,  3.20893407e+00],[ 7.57386208e-01,  8.34911942e-01,  1.22325003e+00],[ 2.21958637e+00,  1.95541751e+00,  4.02953768e+00],[ 1.14199162e+00,  1.74160087e+00, -4.60847944e-01],[ 1.34967840e+00,  1.81956387e+00,  2.34821105e+00],...[ 8.63886595e-01,  1.14353371e+00,  2.04034138e+00],[ 9.92864490e-01,  7.45980024e-01,  2.10216165e+00],[ 4.25307125e-01,  9.57666039e-01, -1.80540013e+00],[ 7.56753862e-01,  7.09740639e-01,  1.09561133e+00],[ 5.15464962e-01,  2.57968724e-01,  3.52174520e+00],[ 1.30328250e+00,  1.09762728e+00,  1.44670618e+00],[ 3.59558940e-01,  5.37134528e-01, -1.53514147e-01],[ 2.66426224e-02,  6.14598870e-01, -1.40780878e+00],[ 7.28204489e-01,  9.94956851e-01, -2.89718914e+00],[ 2.99855947e-01,  9.05317187e-01, -1.55203390e+00],[ 7.91339934e-01,  2.84535080e-01,  1.37865841e+00],[ 8.83184791e-01,  4.73504543e-01,  1.97611123e-01],[ 5.25151372e-01,  2.99478263e-01, -2.00779867e+00],[-1.82255045e-01, -1.49627030e-01, -1.92763901e+00],[ 3.61442894e-01,  1.49727818e-02, -2.74353838e+00],[-6.78136110e-01, -8.94805312e-01, -1.78362298e+00],[-1.38048303e+00, -7.84275293e-01, -6.91646481e+00],[-1.66119802e+00,  1.51050046e-01, -1.75598202e+01],[-1.85124770e-01, -2.19586790e-01, -6.75614691e+00],[ 6.86218739e-01,  7.26487339e-01,  2.01972437e+00]], dtype=float32)


Attributes: (5)


created_at :  
2026-09-04T06:35:24.035308+00:00

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


- time: 200
- covariate_dim: 3


Coordinates: (2)


time


(time)


datetime64\[us\]


1959-10-01 ... 2009-07-01


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['1959-10-01T00:00:00.000000', '1960-01-01T00:00:00.000000','1960-04-01T00:00:00.000000', '1960-07-01T00:00:00.000000','1960-10-01T00:00:00.000000', '1961-01-01T00:00:00.000000','1961-04-01T00:00:00.000000', '1961-07-01T00:00:00.000000','1961-10-01T00:00:00.000000', '1962-01-01T00:00:00.000000','1962-04-01T00:00:00.000000', '1962-07-01T00:00:00.000000','1962-10-01T00:00:00.000000', '1963-01-01T00:00:00.000000','1963-04-01T00:00:00.000000', '1963-07-01T00:00:00.000000','1963-10-01T00:00:00.000000', '1964-01-01T00:00:00.000000','1964-04-01T00:00:00.000000', '1964-07-01T00:00:00.000000','1964-10-01T00:00:00.000000', '1965-01-01T00:00:00.000000','1965-04-01T00:00:00.000000', '1965-07-01T00:00:00.000000','1965-10-01T00:00:00.000000', '1966-01-01T00:00:00.000000','1966-04-01T00:00:00.000000', '1966-07-01T00:00:00.000000','1966-10-01T00:00:00.000000', '1967-01-01T00:00:00.000000','1967-04-01T00:00:00.000000', '1967-07-01T00:00:00.000000','1967-10-01T00:00:00.000000', '1968-01-01T00:00:00.000000','1968-04-01T00:00:00.000000', '1968-07-01T00:00:00.000000','1968-10-01T00:00:00.000000', '1969-01-01T00:00:00.000000','1969-04-01T00:00:00.000000', '1969-07-01T00:00:00.000000','1969-10-01T00:00:00.000000', '1970-01-01T00:00:00.000000','1970-04-01T00:00:00.000000', '1970-07-01T00:00:00.000000','1970-10-01T00:00:00.000000', '1971-01-01T00:00:00.000000','1971-04-01T00:00:00.000000', '1971-07-01T00:00:00.000000','1971-10-01T00:00:00.000000', '1972-01-01T00:00:00.000000','1972-04-01T00:00:00.000000', '1972-07-01T00:00:00.000000','1972-10-01T00:00:00.000000', '1973-01-01T00:00:00.000000','1973-04-01T00:00:00.000000', '1973-07-01T00:00:00.000000','1973-10-01T00:00:00.000000', '1974-01-01T00:00:00.000000','1974-04-01T00:00:00.000000', '1974-07-01T00:00:00.000000','1974-10-01T00:00:00.000000', '1975-01-01T00:00:00.000000','1975-04-01T00:00:00.000000', '1975-07-01T00:00:00.000000','1975-10-01T00:00:00.000000', '1976-01-01T00:00:00.000000','1976-04-01T00:00:00.000000', '1976-07-01T00:00:00.000000','1976-10-01T00:00:00.000000', '1977-01-01T00:00:00.000000','1977-04-01T00:00:00.000000', '1977-07-01T00:00:00.000000','1977-10-01T00:00:00.000000', '1978-01-01T00:00:00.000000','1978-04-01T00:00:00.000000', '1978-07-01T00:00:00.000000','1978-10-01T00:00:00.000000', '1979-01-01T00:00:00.000000','1979-04-01T00:00:00.000000', '1979-07-01T00:00:00.000000','1979-10-01T00:00:00.000000', '1980-01-01T00:00:00.000000','1980-04-01T00:00:00.000000', '1980-07-01T00:00:00.000000','1980-10-01T00:00:00.000000', '1981-01-01T00:00:00.000000','1981-04-01T00:00:00.000000', '1981-07-01T00:00:00.000000','1981-10-01T00:00:00.000000', '1982-01-01T00:00:00.000000','1982-04-01T00:00:00.000000', '1982-07-01T00:00:00.000000','1982-10-01T00:00:00.000000', '1983-01-01T00:00:00.000000','1983-04-01T00:00:00.000000', '1983-07-01T00:00:00.000000','1983-10-01T00:00:00.000000', '1984-01-01T00:00:00.000000','1984-04-01T00:00:00.000000', '1984-07-01T00:00:00.000000','1984-10-01T00:00:00.000000', '1985-01-01T00:00:00.000000','1985-04-01T00:00:00.000000', '1985-07-01T00:00:00.000000','1985-10-01T00:00:00.000000', '1986-01-01T00:00:00.000000','1986-04-01T00:00:00.000000', '1986-07-01T00:00:00.000000','1986-10-01T00:00:00.000000', '1987-01-01T00:00:00.000000','1987-04-01T00:00:00.000000', '1987-07-01T00:00:00.000000','1987-10-01T00:00:00.000000', '1988-01-01T00:00:00.000000','1988-04-01T00:00:00.000000', '1988-07-01T00:00:00.000000','1988-10-01T00:00:00.000000', '1989-01-01T00:00:00.000000','1989-04-01T00:00:00.000000', '1989-07-01T00:00:00.000000','1989-10-01T00:00:00.000000', '1990-01-01T00:00:00.000000','1990-04-01T00:00:00.000000', '1990-07-01T00:00:00.000000','1990-10-01T00:00:00.000000', '1991-01-01T00:00:00.000000','1991-04-01T00:00:00.000000', '1991-07-01T00:00:00.000000','1991-10-01T00:00:00.000000', '1992-01-01T00:00:00.000000','1992-04-01T00:00:00.000000', '1992-07-01T00:00:00.000000','1992-10-01T00:00:00.000000', '1993-01-01T00:00:00.000000','1993-04-01T00:00:00.000000', '1993-07-01T00:00:00.000000','1993-10-01T00:00:00.000000', '1994-01-01T00:00:00.000000','1994-04-01T00:00:00.000000', '1994-07-01T00:00:00.000000','1994-10-01T00:00:00.000000', '1995-01-01T00:00:00.000000','1995-04-01T00:00:00.000000', '1995-07-01T00:00:00.000000','1995-10-01T00:00:00.000000', '1996-01-01T00:00:00.000000','1996-04-01T00:00:00.000000', '1996-07-01T00:00:00.000000','1996-10-01T00:00:00.000000', '1997-01-01T00:00:00.000000','1997-04-01T00:00:00.000000', '1997-07-01T00:00:00.000000','1997-10-01T00:00:00.000000', '1998-01-01T00:00:00.000000','1998-04-01T00:00:00.000000', '1998-07-01T00:00:00.000000','1998-10-01T00:00:00.000000', '1999-01-01T00:00:00.000000','1999-04-01T00:00:00.000000', '1999-07-01T00:00:00.000000','1999-10-01T00:00:00.000000', '2000-01-01T00:00:00.000000','2000-04-01T00:00:00.000000', '2000-07-01T00:00:00.000000','2000-10-01T00:00:00.000000', '2001-01-01T00:00:00.000000','2001-04-01T00:00:00.000000', '2001-07-01T00:00:00.000000','2001-10-01T00:00:00.000000', '2002-01-01T00:00:00.000000','2002-04-01T00:00:00.000000', '2002-07-01T00:00:00.000000','2002-10-01T00:00:00.000000', '2003-01-01T00:00:00.000000','2003-04-01T00:00:00.000000', '2003-07-01T00:00:00.000000','2003-10-01T00:00:00.000000', '2004-01-01T00:00:00.000000','2004-04-01T00:00:00.000000', '2004-07-01T00:00:00.000000','2004-10-01T00:00:00.000000', '2005-01-01T00:00:00.000000','2005-04-01T00:00:00.000000', '2005-07-01T00:00:00.000000','2005-10-01T00:00:00.000000', '2006-01-01T00:00:00.000000','2006-04-01T00:00:00.000000', '2006-07-01T00:00:00.000000','2006-10-01T00:00:00.000000', '2007-01-01T00:00:00.000000','2007-04-01T00:00:00.000000', '2007-07-01T00:00:00.000000','2007-10-01T00:00:00.000000', '2008-01-01T00:00:00.000000','2008-04-01T00:00:00.000000', '2008-07-01T00:00:00.000000','2008-10-01T00:00:00.000000', '2009-01-01T00:00:00.000000','2009-04-01T00:00:00.000000', '2009-07-01T00:00:00.000000'],dtype='datetime64[us]')


covariate_dim


(covariate_dim)


int64


0 1 2


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0, 1, 2])


Data variables: (1)


covariates


(time, covariate_dim)


float32


0.3495 0.1084 3.443 ... 0.7265 2.02


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[ 3.49453270e-01,  1.08401097e-01,  3.44251108e+00],[ 2.21901798e+00,  9.53415096e-01,  1.02663765e+01],[-4.68455315e-01,  1.25724280e+00, -1.06693850e+01],[ 1.63288012e-01, -3.96792650e-01, -5.97787917e-01],[-1.29063594e+00,  1.34303316e-01, -1.31852016e+01],[ 5.92259109e-01, -2.79649887e-02,  2.52441812e+00],[ 1.85345340e+00,  1.47698414e+00,  7.18338728e+00],[ 1.60316396e+00,  4.83863056e-01,  8.04527092e+00],[ 2.01528168e+00,  1.98230624e+00,  1.67371130e+00],[ 1.77772593e+00,  1.05911660e+00,  5.79106426e+00],[ 1.09805155e+00,  1.22162342e+00, -9.71584797e-01],[ 9.20406699e-01,  8.06202650e-01,  1.77339709e+00],[ 2.42701873e-01,  1.40825522e+00, -3.41469789e+00],[ 1.29852104e+00,  6.71229422e-01,  5.40070963e+00],[ 1.24528348e+00,  9.50427711e-01,  1.44677019e+00],[ 1.86540413e+00,  1.35154164e+00,  3.20893407e+00],[ 7.57386208e-01,  8.34911942e-01,  1.22325003e+00],[ 2.21958637e+00,  1.95541751e+00,  4.02953768e+00],[ 1.14199162e+00,  1.74160087e+00, -4.60847944e-01],[ 1.34967840e+00,  1.81956387e+00,  2.34821105e+00],...[ 8.63886595e-01,  1.14353371e+00,  2.04034138e+00],[ 9.92864490e-01,  7.45980024e-01,  2.10216165e+00],[ 4.25307125e-01,  9.57666039e-01, -1.80540013e+00],[ 7.56753862e-01,  7.09740639e-01,  1.09561133e+00],[ 5.15464962e-01,  2.57968724e-01,  3.52174520e+00],[ 1.30328250e+00,  1.09762728e+00,  1.44670618e+00],[ 3.59558940e-01,  5.37134528e-01, -1.53514147e-01],[ 2.66426224e-02,  6.14598870e-01, -1.40780878e+00],[ 7.28204489e-01,  9.94956851e-01, -2.89718914e+00],[ 2.99855947e-01,  9.05317187e-01, -1.55203390e+00],[ 7.91339934e-01,  2.84535080e-01,  1.37865841e+00],[ 8.83184791e-01,  4.73504543e-01,  1.97611123e-01],[ 5.25151372e-01,  2.99478263e-01, -2.00779867e+00],[-1.82255045e-01, -1.49627030e-01, -1.92763901e+00],[ 3.61442894e-01,  1.49727818e-02, -2.74353838e+00],[-6.78136110e-01, -8.94805312e-01, -1.78362298e+00],[-1.38048303e+00, -7.84275293e-01, -6.91646481e+00],[-1.66119802e+00,  1.51050046e-01, -1.75598202e+01],[-1.85124770e-01, -2.19586790e-01, -6.75614691e+00],[ 6.86218739e-01,  7.26487339e-01,  2.01972437e+00]], dtype=float32)


Attributes: (5)


created_at :  
2026-09-04T06:35:24.035800+00:00

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
- obs_dim: 3


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


datetime64\[us\]


2009-10-01 ... 2017-01-01


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['2009-10-01T00:00:00.000000', '2010-01-01T00:00:00.000000','2010-04-01T00:00:00.000000', '2010-07-01T00:00:00.000000','2010-10-01T00:00:00.000000', '2011-01-01T00:00:00.000000','2011-04-01T00:00:00.000000', '2011-07-01T00:00:00.000000','2011-10-01T00:00:00.000000', '2012-01-01T00:00:00.000000','2012-04-01T00:00:00.000000', '2012-07-01T00:00:00.000000','2012-10-01T00:00:00.000000', '2013-01-01T00:00:00.000000','2013-04-01T00:00:00.000000', '2013-07-01T00:00:00.000000','2013-10-01T00:00:00.000000', '2014-01-01T00:00:00.000000','2014-04-01T00:00:00.000000', '2014-07-01T00:00:00.000000','2014-10-01T00:00:00.000000', '2015-01-01T00:00:00.000000','2015-04-01T00:00:00.000000', '2015-07-01T00:00:00.000000','2015-10-01T00:00:00.000000', '2016-01-01T00:00:00.000000','2016-04-01T00:00:00.000000', '2016-07-01T00:00:00.000000','2016-10-01T00:00:00.000000', '2017-01-01T00:00:00.000000'],dtype='datetime64[us]')


obs_dim


(obs_dim)


\<U8


'realgdp' 'realcons' 'realinv'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['realgdp', 'realcons', 'realinv'], dtype='<U8')


Data variables: (1)


obs


(chain, draw, time, obs_dim)


float32


1.018 1.055 0.833 ... 2.243 -2.136


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 1.0180240e+00,  1.0551972e+00,  8.3303893e-01],[ 9.1139895e-01,  6.0690790e-01,  4.4680514e+00],[ 1.1280961e+00,  8.1243813e-01,  4.8446312e+00],...,[-1.2080957e+00, -6.5898126e-01, -1.0096832e+01],[-8.2774937e-02, -3.5957053e-01, -3.9933252e-01],[-1.0580705e+00, -4.7295451e-01, -5.4817877e+00]],[[ 8.3543730e-01,  8.3375609e-01,  1.2770636e+00],[ 6.5235347e-01,  5.6475532e-01,  5.4316339e+00],[-7.9993016e-01,  1.2189984e-02, -7.4347386e+00],...,[ 6.7232966e-01,  1.5714887e+00, -4.3597574e+00],[ 2.2152956e+00,  1.6034210e-01,  1.1773451e+01],[ 1.4017229e+00,  5.9148210e-01,  3.6501551e+00]],[[ 4.8091698e-01,  1.1788422e+00, -2.4966376e+00],[ 1.1629092e+00,  4.5376706e-01,  4.3294649e+00],[-3.5069132e-01, -3.8488686e-02, -1.7349412e+00],...,...[ 7.0160317e-01,  2.3713350e-01,  2.0843704e+00],[ 3.5343063e-01,  7.3891044e-01, -3.4922953e+00],[ 3.7721527e-01,  5.7609951e-01, -5.4384098e+00]],[[-7.4494177e-01, -1.3857440e+00, -4.1040821e+00],[-9.7574532e-01, -5.1348805e-01, -6.5439034e+00],[-1.4341778e+00,  7.2461677e-01, -1.5955865e+01],...,[ 6.7337042e-01,  1.1177440e+00,  8.8544309e-01],[ 2.3839116e-02,  7.7607942e-01, -3.3321176e+00],[ 2.0783362e+00,  1.1009558e+00,  5.7282581e+00]],[[ 7.4984151e-01, -2.6356649e-01,  4.8683777e+00],[-8.5763830e-01, -6.8324542e-01, -4.2654014e+00],[-8.1557572e-01, -3.0947536e-02, -3.3929849e+00],...,[ 1.1299040e+00,  1.6817294e+00,  4.3380919e+00],[ 1.1613892e+00,  1.4201896e+00,  4.4219656e+00],[ 1.7786182e+00,  2.2427478e+00, -2.1358867e+00]]]],shape=(4, 1000, 30, 3), dtype=float32)


Attributes: (5)


created_at :  
2026-09-04T06:35:24.296777+00:00

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
- covariate_dim: 3


Coordinates: (2)


time


(time)


datetime64\[us\]


2009-10-01 ... 2017-01-01


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['2009-10-01T00:00:00.000000', '2010-01-01T00:00:00.000000','2010-04-01T00:00:00.000000', '2010-07-01T00:00:00.000000','2010-10-01T00:00:00.000000', '2011-01-01T00:00:00.000000','2011-04-01T00:00:00.000000', '2011-07-01T00:00:00.000000','2011-10-01T00:00:00.000000', '2012-01-01T00:00:00.000000','2012-04-01T00:00:00.000000', '2012-07-01T00:00:00.000000','2012-10-01T00:00:00.000000', '2013-01-01T00:00:00.000000','2013-04-01T00:00:00.000000', '2013-07-01T00:00:00.000000','2013-10-01T00:00:00.000000', '2014-01-01T00:00:00.000000','2014-04-01T00:00:00.000000', '2014-07-01T00:00:00.000000','2014-10-01T00:00:00.000000', '2015-01-01T00:00:00.000000','2015-04-01T00:00:00.000000', '2015-07-01T00:00:00.000000','2015-10-01T00:00:00.000000', '2016-01-01T00:00:00.000000','2016-04-01T00:00:00.000000', '2016-07-01T00:00:00.000000','2016-10-01T00:00:00.000000', '2017-01-01T00:00:00.000000'],dtype='datetime64[us]')


covariate_dim


(covariate_dim)


int64


0 1 2


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([0, 1, 2])


Data variables: (1)


covariates


(time, covariate_dim)


float32


0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], dtype=float32)


Attributes: (5)


created_at :  
2026-09-04T06:35:24.297133+00:00

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

The summary table reports the posterior mean, standard deviation, 94\\ HDI, effective sample sizes and \hat{R} for every parameter. Rows of `phi` read `phi[lag, equation, lagged_series]`: the coefficient of the lagged series in the equation of the first series.


``` python
summary = az.summary(tree, var_names=["intercept", "sigma", "phi"], ci_kind="hdi", ci_prob=0.94)


def convergence_line(summary_: pd.DataFrame) -> str:
    """Worst r_hat and smallest bulk ESS of a summary table (its cells are formatted strings)."""
    r_hat = pd.to_numeric(summary_["r_hat"]).max()
    ess_bulk = int(pd.to_numeric(summary_["ess_bulk"]).min())
    return f"max r_hat: {r_hat:.3f}, min ess_bulk: {ess_bulk}"


print(convergence_line(summary))
summary
```


    max r_hat: 1.000, min ess_bulk: 1856


|  | mean | sd | hdi94_lb | hdi94_ub | ess_bulk | ess_tail | r_hat | mcse_mean | mcse_sd |
|----|----|----|----|----|----|----|----|----|----|
| intercept\[realgdp\] | 0.24 | 0.102 | 0.051 | 0.43 | 2501 | 2405 | 1.00 | 0.002 | 0.0014 |
| intercept\[realcons\] | 0.554 | 0.099 | 0.37 | 0.74 | 3479 | 3006 | 1.00 | 0.0017 | 0.0012 |
| intercept\[realinv\] | -1.74 | 0.49 | -2.7 | -0.84 | 2894 | 3068 | 1.00 | 0.009 | 0.0064 |
| sigma\[realgdp\] | 0.747 | 0.0362 | 0.68 | 0.82 | 2865 | 2450 | 1.00 | 0.00068 | 0.00047 |
| sigma\[realcons\] | 0.658 | 0.0334 | 0.6 | 0.72 | 4167 | 3304 | 1.00 | 0.00052 | 0.00037 |
| sigma\[realinv\] | 3.884 | 0.186 | 3.6 | 4.3 | 3641 | 3008 | 1.00 | 0.0031 | 0.0023 |
| phi\[1, realgdp, realgdp\] | -0.059 | 0.14 | -0.32 | 0.21 | 1922 | 2387 | 1.00 | 0.0032 | 0.0023 |
| phi\[1, realgdp, realcons\] | 0.443 | 0.114 | 0.23 | 0.65 | 2293 | 2217 | 1.00 | 0.0024 | 0.0016 |
| phi\[1, realgdp, realinv\] | 0.01 | 0.0227 | -0.032 | 0.054 | 1878 | 2259 | 1.00 | 0.00052 | 0.00037 |
| phi\[1, realcons, realgdp\] | -0.056 | 0.144 | -0.32 | 0.21 | 2295 | 2142 | 1.00 | 0.003 | 0.0021 |
| phi\[1, realcons, realcons\] | 0.228 | 0.113 | 0.015 | 0.44 | 2549 | 2649 | 1.00 | 0.0022 | 0.0016 |
| phi\[1, realcons, realinv\] | 0.0207 | 0.0223 | -0.021 | 0.063 | 2324 | 2309 | 1.00 | 0.00047 | 0.00032 |
| phi\[1, realinv, realgdp\] | -0.49 | 0.61 | -1.6 | 0.68 | 2268 | 2696 | 1.00 | 0.013 | 0.0091 |
| phi\[1, realinv, realcons\] | 2.82 | 0.5 | 1.9 | 3.7 | 3010 | 2855 | 1.00 | 0.0091 | 0.0064 |
| phi\[1, realinv, realinv\] | 0.071 | 0.104 | -0.13 | 0.26 | 2316 | 2712 | 1.00 | 0.0022 | 0.0015 |
| phi\[2, realgdp, realgdp\] | -0.008 | 0.141 | -0.27 | 0.26 | 1856 | 2344 | 1.00 | 0.0033 | 0.0023 |
| phi\[2, realgdp, realcons\] | 0.265 | 0.123 | 0.029 | 0.49 | 2125 | 2210 | 1.00 | 0.0027 | 0.0019 |
| phi\[2, realgdp, realinv\] | -0.001 | 0.0219 | -0.042 | 0.039 | 2026 | 2589 | 1.00 | 0.00049 | 0.00034 |
| phi\[2, realcons, realgdp\] | -0.117 | 0.144 | -0.39 | 0.15 | 2092 | 2427 | 1.00 | 0.0032 | 0.0021 |
| phi\[2, realcons, realcons\] | 0.224 | 0.123 | -0.013 | 0.46 | 2327 | 2562 | 1.00 | 0.0026 | 0.0018 |
| phi\[2, realcons, realinv\] | 0.0235 | 0.0216 | -0.018 | 0.063 | 2251 | 2365 | 1.00 | 0.00046 | 0.00031 |
| phi\[2, realinv, realgdp\] | 0.23 | 0.62 | -0.91 | 1.4 | 2223 | 2465 | 1.00 | 0.013 | 0.0092 |
| phi\[2, realinv, realcons\] | 0.64 | 0.54 | -0.37 | 1.7 | 2581 | 2361 | 1.00 | 0.011 | 0.0077 |
| phi\[2, realinv, realinv\] | -0.075 | 0.101 | -0.26 | 0.12 | 2659 | 2549 | 1.00 | 0.002 | 0.0014 |


``` python
pc_trace = az.plot_trace_dist(
    tree, var_names=["intercept", "sigma"], compact=True, figure_kwargs={"figsize": (12, 6)}
)
pc_trace.viz["figure"].item().suptitle(
    "Trace plots: intercept and shock scales", fontsize=16, fontweight="bold", y=1.03
);
```


<figure class="figure">
<p><img src="var_files/figure-html/_src-var-cell-11-output-1.png" class="img-fluid figure-img" /></p>
</figure>


## Stability

A VAR is stable when all eigenvalues of its companion matrix have modulus below one. Stability is what makes the forecast revert to a finite unconditional mean and the impulse responses die out. Because the impulse responses are a nonlinear function of the coefficients, a single explosive posterior draw would dominate their posterior mean at long horizons, so we check the share of stable draws before the IRF section and mask the unstable draws if there are any.


``` python
phi_draws = jnp.asarray(posterior["phi"])  # (4000, 2, 3, 3)
radius = np.abs(np.linalg.eigvals(np.asarray(companion_matrix(phi_draws)))).max(axis=-1)
stable = radius < 1.0
print(
    f"stable draws: {stable.mean():.3f}, median spectral radius: {np.median(radius):.3f}, "
    f"max: {radius.max():.3f}"
)
```


    stable draws: 1.000, median spectral radius: 0.597, max: 0.819


# In-sample fit

The posterior predictive of the `obs` site gives the one-step-ahead predictive distribution for every in-sample quarter. We plot the 50\\ and 94\\ HDI bands per series and score the fit with the continuous ranked probability score (CRPS).


``` python
def hdi_label(prob: float, prefix: str = "") -> str:
    r"""Legend label for an HDI band, e.g. ``$94\%$ HDI``."""
    percent = f"{prob:.0%}".replace("%", r"\%")
    return f"{prefix}${percent}$ HDI"


def stack_draws(group: str, tree_: xr.DataTree) -> Float[np.ndarray, " sample time series"]:
    """Flatten ``(chain, draw)`` of the ``obs`` variable of a tree group into a sample axis."""
    da = tree_[group].dataset["obs"]
    return da.stack(sample=("chain", "draw")).transpose("sample", "time", "obs_dim").to_numpy()


hdi_probs = (0.5, 0.94)
hdi_alphas = [0.6, 0.3]  # 50% band darker, 94% band lighter
dates_num = mdates.date2num(dates.to_pydatetime())
future_dates_num = mdates.date2num(future_dates.to_pydatetime())


def plot_fit_and_forecast(
    train_draws: Float[np.ndarray, " sample time series"],
    future_draws: Float[np.ndarray, " sample future series"] | None,
    title: str,
    n_last: int | None = None,
) -> None:
    """Facet the in-sample predictive (and optionally the forecast) per series.

    Parameters
    ----------
    train_draws
        In-sample posterior predictive draws ``(sample, time, series)``.
    future_draws
        Forecast draws ``(sample, future, series)``, or ``None`` for the in-sample plot only.
    title
        Figure title.
    n_last
        Plot only the last ``n_last`` in-sample quarters (``None`` for all).
    """
    start = 0 if n_last is None else train_draws.shape[1] - n_last
    x_train = dates_num[start:]
    observed = np.asarray(data)[start:]
    pc = az.plot_lm(
        predictions_to_datatree(train_draws[:, start:], x_train, names, observed=observed),
        y="obs",
        x="t",
        plot_dim="time",
        ci_kind="hdi",
        ci_prob=hdi_probs,
        smooth=False,
        col_wrap=1,
        visuals={
            "ci_band": {"color": "C0"},
            "observed_scatter": False,
            "pe_line": False,
            "xlabel": False,
            "ylabel": False,
        },
        aes={"alpha": ["prob"]},
        alpha=hdi_alphas,
        figure_kwargs={"figsize": (12, 9)},
    )
    train_bands = pc.viz["ci_band"]["t"].sel(series=names[0])
    handles = [train_bands.sel(prob=prob).item() for prob in (0.94, 0.5)]
    for handle, prob in zip(handles, (0.94, 0.5), strict=True):
        handle.set_label(hdi_label(prob, prefix="in-sample " if future_draws is not None else ""))
    if future_draws is not None:
        az.plot_lm(
            predictions_to_datatree(future_draws, future_dates_num, names),
            y="obs",
            x="t",
            plot_dim="time",
            plot_collection=pc,
            ci_kind="hdi",
            ci_prob=hdi_probs,
            smooth=False,
            visuals={
                "ci_band": {"color": "C1"},
                "observed_scatter": False,
                "pe_line": False,
                "xlabel": False,
                "ylabel": False,
            },
        )
        future_bands = pc.viz["ci_band"]["t"].sel(series=names[0])
        for prob in (0.94, 0.5):
            band = future_bands.sel(prob=prob).item()
            band.set_label(hdi_label(prob, prefix="forecast "))
            handles.append(band)
    for i, name in enumerate(names):
        ax = pc.get_target("t", {"series": name})
        (obs_line,) = ax.plot(x_train, observed[:, i], color="black", lw=1.2, label="observed")
        ax.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax.set_title(name, fontsize=11)
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    handles.append(obs_line)
    pc.get_target("t", {"series": names[0]}).legend(handles=handles, loc="upper left", fontsize=9)
    fig = pc.viz["figure"].item()
    fig.supxlabel("date")
    fig.supylabel("growth rate (percent)")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)


train_pp = stack_draws("posterior_predictive", tree)
crps_train = eval_crps(train_pp, data)
print(f"in-sample CRPS: {float(crps_train):.4f}")
```


    in-sample CRPS: 0.9674


``` python
plot_fit_and_forecast(
    train_pp, None, title=f"One-step-ahead in-sample fit (CRPS: {float(crps_train):.3f})"
)
```


<figure class="figure">
<p><img src="var_files/figure-html/_src-var-cell-14-output-1.png" class="img-fluid figure-img" /></p>
</figure>


# Forecast

The `predictions` group of the tree holds the 30-quarter forecast paths. Each path draws its own correlated shocks and feeds them back through the lag window, so the uncertainty compounds over the horizon. We show the last 40 in-sample quarters for context.


``` python
forecast_draws = stack_draws("predictions", tree)
plot_fit_and_forecast(train_pp, forecast_draws, title="VAR(2) forecast, 30 quarters", n_last=40)
```


<figure class="figure">
<p><img src="var_files/figure-html/_src-var-cell-15-output-1.png" class="img-fluid figure-img" /></p>
</figure>


The forecast bands widen over the first few quarters and then settle. For a stable VAR the forecast error covariance \sum\_{s \< h} \Psi_s \Sigma \Psi_s^\top converges to the unconditional covariance of the process, and the forecast mean converges to the unconditional mean (I - \sum_l \Phi_l)^{-1} c. The posterior bands also carry parameter uncertainty, so they are a mixture over draws, but with every draw stable the same picture holds. The table shows the width of the 94\\ HDI per series at a few horizons, and the printout compares the unconditional mean implied by the posterior means with the sample means and the mean forecast at the last horizon.


``` python
def hdi_width(draws: Float[np.ndarray, " sample time series"], prob: float) -> np.ndarray:
    """Width of the HDI of ``draws`` per time step and series."""
    da = xr.DataArray(np.asarray(draws), dims=["sample", "time", "series"])
    hdi = az.hdi(da, prob=prob, dim="sample")  # (time, series, ci_bound)
    return (hdi.sel(ci_bound="upper") - hdi.sel(ci_bound="lower")).to_numpy()


width_94 = hdi_width(forecast_draws, 0.94)
horizons = [1, 5, 10, 20, 30]
pd.DataFrame(
    width_94[[h - 1 for h in horizons]], index=pd.Index(horizons, name="h"), columns=names
).round(3)
```


|     | realgdp | realcons | realinv |
|-----|---------|----------|---------|
| h   |         |          |         |
| 1   | 2.880   | 2.512    | 14.786  |
| 5   | 3.172   | 2.730    | 16.646  |
| 10  | 3.222   | 2.671    | 16.764  |
| 20  | 3.165   | 2.728    | 16.068  |
| 30  | 3.126   | 2.760    | 16.045  |


``` python
phi_mean = np.asarray(posterior["phi"]).mean(axis=0)
c_mean = np.asarray(posterior["intercept"]).mean(axis=0)
unconditional_mean = np.linalg.solve(np.eye(k) - phi_mean.sum(axis=0), c_mean)
pd.DataFrame(
    {
        "unconditional mean": unconditional_mean,
        "sample mean": np.asarray(data).mean(axis=0),
        "mean forecast at h=30": forecast_draws.mean(axis=0)[-1],
    },
    index=names,
).round(3)
```


|          | unconditional mean | sample mean | mean forecast at h=30 |
|----------|--------------------|-------------|-----------------------|
| realgdp  | 0.789              | 0.772       | 0.790                 |
| realcons | 0.838              | 0.832       | 0.826                 |
| realinv  | 0.950              | 0.818       | 1.018                 |


# Impulse response functions

A forecast tells you where the system goes on average. An impulse response tells you how a shock to one series propagates to all series over time. For a stable VAR the moving-average (Wold) representation

 y_t = \mu + \sum\_{h=0}^{\infty} \Psi_h \\ \varepsilon\_{t-h} 

exists, and the coefficient matrices follow the recursion

 \Psi_0 = I, \qquad \Psi_h = \sum\_{j=1}^{\min(h, p)} \Phi_j \\ \Psi\_{h-j} \quad (h \geq 1). 

The entry \Psi_h\[i, j\] is the response of series i, h quarters after a unit shock to the reduced-form residual \varepsilon\_{t, j}, with the other residuals held at zero. `impulse_response(phi, horizon)` runs this recursion for all posterior draws at once (the draws pass through the leading batch axis; no `vmap` is needed) and returns an array of shape `(draws, horizon + 1, series, series)`, indexed as `[draw, h, response, shock]`.

The recursion exists for any coefficients, but the representation and the decay \Psi_h \to 0 need stability, which we checked above. If some draws were unstable we would mask them here; the mask below is the identity when all draws are stable.


``` python
n_irf_steps = 10
irf_labels = [f"{response} response to {shock} shock" for response in names for shock in names]
irf_draws = impulse_response(phi_draws[stable], n_irf_steps)  # (draws, 11, 3, 3)
print(f"irf_draws: {irf_draws.shape}")
print("posterior mean responses at h = 0, 1, 2 (rows: response, columns: shock):")
print(np.round(np.asarray(irf_draws.mean(axis=0)[:3]), 3))
```


    irf_draws: (4000, 11, 3, 3)
    posterior mean responses at h = 0, 1, 2 (rows: response, columns: shock):
    [[[ 1.     0.     0.   ]
      [ 0.     1.     0.   ]
      [ 0.     0.     1.   ]]

     [[-0.059  0.443  0.01 ]
      [-0.056  0.228  0.021]
      [-0.49   2.821  0.071]]

     [[-0.029  0.368  0.008]
      [-0.137  0.315  0.029]
      [ 0.066  1.271 -0.012]]]


``` python
def plot_irf_grid(
    irf: Float[Array, " sample steps series series"],
    title: str,
    ylabel: str,
    overlay: Float[Array, " sample steps series series"] | None = None,
    overlay_label: str = "",
) -> None:
    """Plot a ``series x series`` grid of impulse responses with HDI bands.

    Parameters
    ----------
    irf
        Impulse response draws ``(sample, steps, response, shock)``.
    title
        Figure title.
    ylabel
        Shared y-axis label.
    overlay
        Optional second set of draws whose posterior mean is overlaid as a line.
    overlay_label
        Legend label of the overlaid mean.
    """
    n_draws, n_steps = irf.shape[:2]
    steps = np.arange(n_steps, dtype=float)
    pc = az.plot_lm(
        predictions_to_datatree(np.asarray(irf).reshape(n_draws, n_steps, -1), steps, irf_labels),
        y="obs",
        x="t",
        plot_dim="time",
        ci_kind="hdi",
        ci_prob=hdi_probs,
        smooth=False,
        point_estimate="mean",
        col_wrap=3,
        visuals={
            "ci_band": {"color": "C0"},
            "observed_scatter": False,
            "pe_line": {"color": "C0", "alpha": 1.0, "width": 1.5},
            "xlabel": False,
            "ylabel": False,
        },
        aes={"alpha": ["prob"]},
        alpha=hdi_alphas,
        figure_kwargs={"figsize": (14, 10)},
    )
    bands = pc.viz["ci_band"]["t"].sel(series=irf_labels[0])
    handles = []
    for prob in (0.94, 0.5):
        band = bands.sel(prob=prob).item()
        band.set_label(hdi_label(prob))
        handles.append(band)
    mean_line = pc.viz["pe_line"]["t"].sel(series=irf_labels[0]).item()
    mean_line.set_label("posterior mean")
    handles.append(mean_line)
    if overlay is not None:
        az.plot_lm(
            predictions_to_datatree(
                np.asarray(overlay).reshape(overlay.shape[0], n_steps, -1), steps, irf_labels
            ),
            y="obs",
            x="t",
            plot_dim="time",
            plot_collection=pc,
            ci_kind="hdi",
            ci_prob=hdi_probs,
            smooth=False,
            point_estimate="mean",
            visuals={
                "ci_band": False,
                "observed_scatter": False,
                "pe_line": {"color": "C1", "alpha": 1.0, "width": 1.5},
                "xlabel": False,
                "ylabel": False,
            },
        )
        overlay_line = pc.viz["pe_line"]["t"].sel(series=irf_labels[0]).item()
        overlay_line.set_label(overlay_label)
        handles.append(overlay_line)
    for label in irf_labels:
        ax = pc.get_target("t", {"series": label})
        ax.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax.set_title(label, fontsize=10)
    pc.get_target("t", {"series": irf_labels[0]}).legend(
        handles=handles, loc="upper right", fontsize=8
    )
    fig = pc.viz["figure"].item()
    fig.supxlabel("quarters after the shock")
    fig.supylabel(ylabel)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)


plot_irf_grid(
    irf_draws,
    title="Impulse responses to a unit reduced-form shock",
    ylabel="response (percentage points)",
)
```


<figure class="figure">
<p><img src="var_files/figure-html/_src-var-cell-19-output-1.png" class="img-fluid figure-img" /></p>
</figure>


## Orthogonalized and cumulative responses

A unit shock to one reduced-form residual with the others held at zero is not an experiment we can observe when \Sigma is not diagonal: the residuals move together. The standard fix is to rewrite the shocks as \varepsilon_t = L \\ u_t with u_t \sim \text{MultivariateNormal}(0, I) and L the Cholesky factor of \Sigma, and to report the responses to the *orthogonalized* shocks u_t:

 \Theta_h = \Psi_h \\ L. 

A unit shock to u\_{t, j} is a one-standard-deviation shock. Because L is lower triangular, the first series in the ordering responds only to its own shock in the impact quarter, the second series to the first two shocks, and so on. This is the recursive identification, and the ordering `realgdp, realcons, realinv` is part of the model: a different ordering gives different orthogonalized responses. [impulse_response](../../reference/var.impulse_response.md#numpyro_forecast.var.impulse_response) takes the factor through `scale_tril`, here built from the posterior draws of \sigma and L\_\Omega.

Our series are growth rates in percent, g_t = 100 \\ \Delta \log Y_t, so the running sum \sum\_{s=0}^{h} \Theta_s is the response of the *log level* in percent, approximately the percent change of the level. We request it with `cumulative=True` and extend the horizon to 20 quarters. For a stable VAR the cumulative response converges to the long-run effect (I - \sum_l \Phi_l)^{-1} L.


``` python
sigma_draws = jnp.asarray(posterior["sigma"])[stable]
l_omega_draws = jnp.asarray(posterior["l_omega"])[stable]
scale_tril_draws = sigma_draws[..., :, None] * l_omega_draws  # (draws, 3, 3)
irf_level_draws = impulse_response(
    phi_draws[stable], 20, scale_tril=scale_tril_draws, cumulative=True
)
print(f"irf_level_draws: {irf_level_draws.shape}")
print("posterior mean cumulative response at h = 20 (rows: response, columns: shock):")
print(np.round(np.asarray(irf_level_draws.mean(axis=0)[-1]), 3))
```


    irf_level_draws: (4000, 21, 3, 3)
    posterior mean cumulative response at h = 20 (rows: response, columns: shock):
    [[1.259 0.613 0.139]
     [0.759 0.898 0.175]
     [5.186 1.382 2.667]]


``` python
plot_irf_grid(
    irf_level_draws,
    title="Cumulative responses to a one standard deviation orthogonalized shock",
    ylabel="level response (percent)",
)
```


<figure class="figure">
<p><img src="var_files/figure-html/_src-var-cell-21-output-1.png" class="img-fluid figure-img" /></p>
</figure>


# Minnesota prior

A VAR has many coefficients for its sample size: here 18 lag coefficients plus 3 intercepts for 200 quarters, and the count grows with p k^2. Unregularized fits overfit and forecast poorly. Litterman (1986) and Doan, Litterman and Sims (1984) proposed the *Minnesota prior*, which encodes three beliefs:

1.  Each series is close to a univariate process: the prior mean of the first own lag is m\_{\text{own}} and every other coefficient is centered at zero. For series in levels m\_{\text{own}} = 1 (a random walk); for differenced or otherwise stationary series, as here, m\_{\text{own}} = 0.
2.  Longer lags matter less: the prior standard deviation decays with the lag, d(l) = 1/l (harmonic) or 1/l^2.
3.  Other series matter less than the own past: cross-variable coefficients get a tighter prior by a factor \kappa \in \[0, 1\].

Together, for the coefficient of series j at lag l in the equation of series i,

 \Phi\_{l, ij} \sim \text{Normal}\left(m\_{l, ij}, \\ \lambda \\ d(l) \\ \kappa^{\[i \neq j\]}\right), \qquad m\_{l, ij} = m\_{\text{own}} \\ \[l = 1\] \\ \[i = j\], 

with an overall tightness \lambda. `minnesota_prior(n_lags, n_obs, tightness, cross_shrinkage, decay, own_lag_mean)` returns the `loc` and `scale` arrays in the `(lags, series, series)` layout of [var_step](../../reference/var.var_step.md#numpyro_forecast.var.var_step), and we pass them to `dist.Normal(...).to_event(3)`. Nothing else changes: the prior is an argument of the model factory.

This parameterization follows [Impulso's `MinnesotaPrior`](https://thomaspinder.github.io/Impulso/reference/generated/impulso.priors.MinnesotaPrior.html): the same three knobs (`tightness`, `decay`, `cross_shrinkage`) and a tightness that is fixed rather than estimated. There is no closed-form marginal likelihood for the independent-normal prior, so Impulso treats the tightness as a modeling choice, and so do we. Two differences: Impulso calls the 1/l^2 decay "geometric" (in Doan, Litterman and Sims it is the harmonic decay with exponent two), and it fixes the own-lag mean at one, while we expose `own_lag_mean` because differenced data call for zero.


## Series on different scales

The classic Litterman formulation multiplies the standard deviation of the cross-variable coefficients by \sigma_i / \sigma_j, the ratio of the residual standard deviations of the two series. Impulso omits this factor and asks for pre-scaled data. Our series are not on a common scale: investment growth is about five times more volatile than GDP or consumption growth, so the investment equation carries coefficients about five times larger, and a common tightness would shrink them five times too hard. We apply the classic correction with the sample standard deviations as a proxy for the residual scales. The table shows the ratios \sigma_i / \sigma_j (rows: equation, columns: lagged series).


``` python
sample_sd = y_pct.std().to_numpy()
scale_ratio = sample_sd[:, None] / sample_sd[None, :]
pd.DataFrame(scale_ratio, index=names, columns=names).round(2)
```


|          | realgdp | realcons | realinv |
|----------|---------|----------|---------|
| realgdp  | 1.00    | 1.27     | 0.19    |
| realcons | 0.79    | 1.00     | 0.15    |
| realinv  | 5.33    | 6.75     | 1.00    |


With \lambda = 0.5 and \kappa = 0.5 the prior standard deviation of a first-lag cross coefficient is 0.25 before scaling. The printout compares, for the investment equation at lag one, the unscaled (Impulso-style) prior standard deviations, the scaled ones we use, and the posterior under the weak prior: the coefficient on lagged consumption growth has a posterior mean near 2.8 with a standard deviation near 0.5, so an unscaled prior with standard deviation 0.25 sits more than ten prior standard deviations away from what the data say and would dominate the posterior, while the scaled prior is compatible with it.


``` python
tightness = 0.5
loc_mn, scale_unscaled = minnesota_prior(
    p, k, tightness=tightness, cross_shrinkage=0.5, own_lag_mean=0.0
)
scale_mn = scale_unscaled * jnp.asarray(scale_ratio, dtype=scale_unscaled.dtype)
minnesota = dist.Normal(loc_mn, scale_mn).to_event(3)

phi_weak_mean = np.asarray(posterior["phi"]).mean(axis=0)
phi_weak_sd = np.asarray(posterior["phi"]).std(axis=0)
pd.DataFrame(
    {
        "unscaled prior sd": np.asarray(scale_unscaled[0, 2]),
        "scaled prior sd": np.asarray(scale_mn[0, 2]),
        "weak prior posterior mean": phi_weak_mean[0, 2],
        "weak prior posterior sd": phi_weak_sd[0, 2],
    },
    index=pd.Index(names, name="lagged series (realinv equation, lag 1)"),
).round(3)
```


|  | unscaled prior sd | scaled prior sd | weak prior posterior mean | weak prior posterior sd |
|----|----|----|----|----|
| lagged series (realinv equation, lag 1) |  |  |  |  |
| realgdp | 0.25 | 1.331 | -0.490 | 0.608 |
| realcons | 0.25 | 1.687 | 2.821 | 0.498 |
| realinv | 0.50 | 0.500 | 0.071 | 0.104 |


``` python
var_model_mn = make_var_model(minnesota, y_init)

rng_key, rng_subkey = random.split(rng_key)
mcmc_mn = fit_nuts(rng_subkey, var_model_mn, data, covariates_train)
posterior_mn = mcmc_mn.get_samples()
print(f"divergences: {n_divergences(mcmc_mn)}")

rng_key, rng_subkey = random.split(rng_key)
tree_mn = export(rng_subkey, var_model_mn, posterior_mn)
summary_mn = az.summary(
    tree_mn, var_names=["intercept", "sigma", "phi"], ci_kind="hdi", ci_prob=0.94
)
print(convergence_line(summary_mn))
```


    divergences: 0
    max r_hat: 1.000, min ess_bulk: 2020


## Shrinkage of the coefficients

The table compares the posterior standard deviation of every coefficient under the two priors. The ratio column is below one where the Minnesota prior tightened the posterior. The effect is largest on the second lag, where the harmonic decay halves the prior standard deviation, and on the GDP and consumption equations. The investment equation at lag one is unchanged within Monte Carlo error: after the scale correction its prior is wide relative to what the data say, so the data decide.


``` python
phi_index = pd.MultiIndex.from_product(
    [list(range(1, p + 1)), names, names], names=["lag", "equation", "lagged series"]
)
phi_sd = pd.DataFrame(
    {
        "weak prior": phi_weak_sd.reshape(-1),
        "minnesota prior": np.asarray(posterior_mn["phi"]).std(axis=0).reshape(-1),
    },
    index=phi_index,
)
phi_sd["ratio"] = phi_sd["minnesota prior"] / phi_sd["weak prior"]
own = phi_sd.index.get_level_values("equation") == phi_sd.index.get_level_values("lagged series")
print(
    f"mean ratio on own lags: {phi_sd.loc[own, 'ratio'].mean():.2f}, "
    f"on cross lags: {phi_sd.loc[~own, 'ratio'].mean():.2f}"
)
phi_sd.round(3)
```


    mean ratio on own lags: 0.79, on cross lags: 0.76


<table class="dataframe table table-sm table-striped small">
<thead>
<tr>
<th></th>
<th></th>
<th></th>
<th>weak prior</th>
<th>minnesota prior</th>
<th>ratio</th>
</tr>
<tr>
<th>lag</th>
<th>equation</th>
<th>lagged series</th>
<th></th>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<th rowspan="9" data-valign="top">1</th>
<th rowspan="3" data-valign="top">realgdp</th>
<th>realgdp</th>
<td>0.140</td>
<td>0.119</td>
<td>0.846</td>
</tr>
<tr>
<th>realcons</th>
<td>0.114</td>
<td>0.101</td>
<td>0.889</td>
</tr>
<tr>
<th>realinv</th>
<td>0.023</td>
<td>0.018</td>
<td>0.803</td>
</tr>
<tr>
<th rowspan="3" data-valign="top">realcons</th>
<th>realgdp</th>
<td>0.144</td>
<td>0.105</td>
<td>0.730</td>
</tr>
<tr>
<th>realcons</th>
<td>0.113</td>
<td>0.094</td>
<td>0.831</td>
</tr>
<tr>
<th>realinv</th>
<td>0.022</td>
<td>0.016</td>
<td>0.727</td>
</tr>
<tr>
<th rowspan="3" data-valign="top">realinv</th>
<th>realgdp</th>
<td>0.608</td>
<td>0.650</td>
<td>1.069</td>
</tr>
<tr>
<th>realcons</th>
<td>0.498</td>
<td>0.542</td>
<td>1.088</td>
</tr>
<tr>
<th>realinv</th>
<td>0.104</td>
<td>0.101</td>
<td>0.971</td>
</tr>
<tr>
<th rowspan="9" data-valign="top">2</th>
<th rowspan="3" data-valign="top">realgdp</th>
<th>realgdp</th>
<td>0.141</td>
<td>0.085</td>
<td>0.600</td>
</tr>
<tr>
<th>realcons</th>
<td>0.123</td>
<td>0.083</td>
<td>0.673</td>
</tr>
<tr>
<th>realinv</th>
<td>0.022</td>
<td>0.014</td>
<td>0.617</td>
</tr>
<tr>
<th rowspan="3" data-valign="top">realcons</th>
<th>realgdp</th>
<td>0.144</td>
<td>0.070</td>
<td>0.488</td>
</tr>
<tr>
<th>realcons</th>
<td>0.123</td>
<td>0.086</td>
<td>0.696</td>
</tr>
<tr>
<th>realinv</th>
<td>0.022</td>
<td>0.011</td>
<td>0.532</td>
</tr>
<tr>
<th rowspan="3" data-valign="top">realinv</th>
<th>realgdp</th>
<td>0.616</td>
<td>0.445</td>
<td>0.722</td>
</tr>
<tr>
<th>realcons</th>
<td>0.543</td>
<td>0.433</td>
<td>0.798</td>
</tr>
<tr>
<th>realinv</th>
<td>0.101</td>
<td>0.079</td>
<td>0.780</td>
</tr>
</tbody>
</table>


## Forecast bands

Tighter coefficients mean less parameter uncertainty in the forecast. The table reports the mean width of the 94\\ HDI over the 30 forecast quarters, per series and per prior. The change is small: a few percent for GDP and consumption and none for investment. With 200 quarters for 18 coefficients, the forecast uncertainty comes from the shock covariance, not from the coefficients, and a prior of this tightness cannot move it much.


``` python
forecast_draws_mn = stack_draws("predictions", tree_mn)
pd.DataFrame(
    {
        "weak prior": width_94.mean(axis=0),
        "minnesota prior": hdi_width(forecast_draws_mn, 0.94).mean(axis=0),
    },
    index=names,
).round(3)
```


|          | weak prior | minnesota prior |
|----------|------------|-----------------|
| realgdp  | 3.205      | 3.161           |
| realcons | 2.700      | 2.626           |
| realinv  | 16.513     | 16.557          |


## Impulse responses

The grid overlays the posterior mean responses under the Minnesota prior (orange) on the bands and mean of the weak prior fit (blue). The two means agree closely and the orange line stays inside the 50\\ band of the weak prior fit in every panel. The Minnesota mean is smoother at two and three quarters after the shock, where the prior halves the standard deviation of the second-lag coefficients and irons out the wiggle that the weak prior fit shows there. With 200 quarters for 18 coefficients the data dominate a prior of this tightness; a smaller `tightness` trades this agreement for more shrinkage.


``` python
phi_draws_mn = jnp.asarray(posterior_mn["phi"])
radius_mn = np.abs(np.linalg.eigvals(np.asarray(companion_matrix(phi_draws_mn)))).max(axis=-1)
stable_mn = radius_mn < 1.0
print(f"stable draws (Minnesota prior): {stable_mn.mean():.3f}")
irf_draws_mn = impulse_response(phi_draws_mn[stable_mn], n_irf_steps)
plot_irf_grid(
    irf_draws,
    title="Impulse responses: weak prior (bands) vs Minnesota prior (orange mean)",
    ylabel="response (percentage points)",
    overlay=irf_draws_mn,
    overlay_label="posterior mean (Minnesota prior)",
)
```


    stable draws (Minnesota prior): 1.000


<figure class="figure">
<p><img src="var_files/figure-html/_src-var-cell-27-output-2.png" class="img-fluid figure-img" /></p>
</figure>


The tightness \lambda is a modeling choice, not an estimate: there is no closed-form marginal likelihood for the independent-normal prior to optimize it, so pick it from the scale of the coefficients you find plausible, or compare forecast scores across a few values with [backtest](../../reference/evaluate.backtest.md#numpyro_forecast.evaluate.backtest).


# References

- Orduz, J. [Bayesian VAR in NumPyro](https://juanitorduz.github.io/var_numpyro/). The source of this notebook.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. Chapters 2 and 5 cover the moving-average representation, impulse responses and the Minnesota prior.
- Litterman, R. B. (1986). Forecasting with Bayesian vector autoregressions: five years of experience. *Journal of Business & Economic Statistics*, 4(1), 25-38.
- Doan, T., Litterman, R. B. and Sims, C. A. (1984). Forecasting and conditional projection using realistic prior distributions. *Econometric Reviews*, 3(1), 1-100.
- Pinder, T. [Impulso](https://github.com/thomaspinder/impulso): a Bayesian VAR package for Python ([documentation](https://thomaspinder.github.io/Impulso/), [`MinnesotaPrior` reference](https://thomaspinder.github.io/Impulso/reference/generated/impulso.priors.MinnesotaPrior.html)). The Minnesota prior parameterization and the batched moving-average recursion used here follow its design.
- statsmodels. [macrodata](https://www.statsmodels.org/stable/datasets/generated/macrodata.html): United States macroeconomic data, 1959Q1 to 2009Q3, public domain.

[Source: Vector Autoregression (VAR) with `numpyro_forecast`](_src/var-preview.html#3c3b9b77)
