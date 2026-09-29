# Vector Autoregression (VAR)


Vector Autoregression (VAR) with `numpyro_forecast`

This notebook ports the blog post [**Bayesian VAR in NumPyro**](https://juanitorduz.github.io/var_numpyro/) to the [`numpyro_forecast`](https://github.com/juanitorduz/numpyro_forecast) package. A vector autoregression (VAR) models several time series jointly: each series is regressed on the past values of all series, and the shocks are correlated across series. We fit a VAR with two lags to the quarterly growth rates of US real GDP, consumption and investment, sample the posterior with NUTS, forecast 30 quarters ahead, and compute impulse response functions (IRFs), the standard tool to read a VAR.

The package provides the VAR pieces as reusable components. You do not write the lag recursion, the forecast loop or the IRF recursion yourself:

- [`var_step`](https://juanitorduz.github.io/numpyro_forecast/reference/var.var_step.html) turns sampled coefficients into the `mean` and `update` functions for the [`ssoe`](https://juanitorduz.github.io/numpyro_forecast/reference/models.ssoe.html) building block. The block runs the in-sample recursion and the generative forecast.
- [`impulse_response`](https://juanitorduz.github.io/numpyro_forecast/reference/var.impulse_response.html) computes the responses for all posterior draws at once, with optional orthogonalization and cumulation.
- [`companion_matrix`](https://juanitorduz.github.io/numpyro_forecast/reference/var.companion_matrix.html) gives the stability check.
- [`minnesota_prior`](https://juanitorduz.github.io/numpyro_forecast/reference/priors.minnesota_prior.html) returns the moments of the Minnesota shrinkage prior. It lives in a separate module and is independent of the VAR code: the prior is always your own `numpyro.sample` call.

The components are deliberately minimal. If you need a complete Bayesian VAR toolkit (identification schemes, variance decompositions, lag selection), see [Impulso](https://github.com/thomaspinder/impulso) by Thomas Pinder. Its `MinnesotaPrior` parameterization and its batched moving-average recursion inspired the two helpers used here.


# Prepare notebook


``` python
import datetime as dt
import itertools
import warnings

import arviz as az
import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
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
def quarter_label(d: dt.date) -> str:
    """Format a date as e.g. ``1959Q2``."""
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


url = (
    "https://raw.githubusercontent.com/statsmodels/statsmodels/main/"
    "statsmodels/datasets/macrodata/macrodata.csv"
)
macro_df = pl.read_csv(url)

names = ["realgdp", "realcons", "realinv"]

y_pct = macro_df.select(
    pl.date(pl.col("year"), (pl.col("quarter") - 1) * 3 + 1, 1).alias("date"),
    *[(pl.col(name).log().diff() * 100).alias(name) for name in names],
).drop_nulls()

print(
    f"shape: {y_pct.shape}, "
    f"from {quarter_label(y_pct['date'][0])} to {quarter_label(y_pct['date'][-1])}"
)
y_pct.head()
```


    shape: (202, 4), from 1959Q2 to 2009Q3


shape: (5, 4)

| date       | realgdp   | realcons | realinv    |
|------------|-----------|----------|------------|
| date       | f64       | f64      | f64        |
| 1959-04-01 | 2.494213  | 1.528611 | 8.021268   |
| 1959-07-01 | -0.119295 | 1.038598 | -7.213104  |
| 1959-10-01 | 0.349453  | 0.108401 | 3.442511   |
| 1960-01-01 | 2.219018  | 0.953415 | 10.266377  |
| 1960-04-01 | -0.468455 | 1.257243 | -10.669385 |


``` python
stats = ["mean", "std", "min", "max"]
y_pct.select(names).describe().filter(pl.col("statistic").is_in(stats)).with_columns(
    pl.col(names).round(3)
)
```


shape: (4, 4)

| statistic | realgdp | realcons | realinv |
|-----------|---------|----------|---------|
| str       | f64     | f64      | f64     |
| "mean"    | 0.776   | 0.837    | 0.814   |
| "std"     | 0.88    | 0.694    | 4.685   |
| "min"     | -2.071  | -2.296   | -19.316 |
| "max"     | 3.859   | 2.773    | 12.209  |


Investment growth is about five times more volatile than GDP or consumption growth. Keep this in mind for the Minnesota prior section: the three series are not on a common scale.


``` python
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 8), layout="constrained")

for ax, name, color in zip(axes, names, ("C0", "C1", "C2"), strict=True):
    ax.plot(y_pct["date"], y_pct[name], color=color, lw=1.2, label=name)
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

The noise distribution is a `MultivariateNormal` over the series axis, so the future shocks are correlated across series exactly as the in-sample residuals are. `var_step(phi, intercept)` builds the `mean` and `update` functions from the sampled coefficients: the carry is the lag window with shape `(lags, series)` in natural time order (most recent row last), the mean is c + \sum_l \Phi_l y\_{t-l}, and the carry update drops the oldest row and appends the new one.


## Data layout and the first p observations

Time lives at axis `-2` and the series at axis `-1`, so the data is a `(time, 3)` array. The likelihood conditions on the first p = 2 rows, which seed the lag window (this is the conditional likelihood used by the blog post and by `statsmodels`): `y_init` holds these two rows, `data` holds the remaining 200 rows, and the forecast horizon is fixed by padding `data` with 30 zero rows through [pad_future](../../reference/arrays.pad_future.md#numpyro_forecast.arrays.pad_future). The model reads only the first `h.t_obs` rows of `covariates` (the block checks this), so the padding rows are never used as data. They only set the horizon.

We write the model as a **factory** that takes the prior on \Phi as an argument. The VAR code below never changes when we swap the prior in the last section.


``` python
def add_quarters(d: dt.date, n: int) -> dt.date:
    """Advance a date by ``n`` quarters (calendar-quarter arithmetic, handles year rollover)."""
    month0 = d.month - 1 + 3 * n
    return dt.date(d.year + month0 // 12, month0 % 12 + 1, d.day)


p = 2
y_all = y_pct.select(names).to_jax()  # (202, 3), float32
y_init = y_all[:p]  # the two rows that seed the lag window
data = y_all[p:]  # the 200 rows in the likelihood
future = 30
covariates_train = data  # fitting: no horizon
covariates_full = pad_future(data, future)  # forecasting: 30 unread rows fix the horizon

dates = y_pct["date"][p:].to_list()
future_dates = [add_quarters(dates[-1], h) for h in range(1, future + 1)]
time_coord = dates + future_dates

print(f"y_init: {y_init.shape}, data: {data.shape}, covariates_full: {covariates_full.shape}")
print(f"forecast window: {quarter_label(future_dates[0])} to {quarter_label(future_dates[-1])}")
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

        intercept = numpyro.sample("intercept", dist.Normal(0.0, 1.0).expand([k]).to_event(1))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0).expand([k]).to_event(1))
        l_omega = numpyro.sample("l_omega", dist.LKJCholesky(k, concentration=1.0))
        phi = numpyro.sample("phi", phi_prior)
        scale_tril = sigma[..., :, None] * l_omega

        noise = dist.MultivariateNormal(jnp.zeros(k), scale_tril=scale_tril)
        mean, update = var_step(phi, intercept)
        r = ssoe(h, "eps", y, y_init, mean, update, noise)

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

We fit with four NUTS chains of 1,000 warmup and 1,000 draws each. Fitting uses `covariates_train`, which has the same length as `data`, so the posterior holds only the parameters and the in-sample means. We pass the padded `covariates_full` to [to_datatree](../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree), which runs the posterior predictive for the 200 in-sample rows and the 30 forecast rows in one call and names every dimension.


``` python
def fit_nuts(rng_key: Array, model: ForecastModel, data: Array, covariates: Array) -> MCMC:
    """Fit ``model`` with NUTS (4 chains, 1,000 warmup and 1,000 draws each)."""
    mcmc = MCMC(
        NUTS(model),
        num_warmup=1_000,
        num_samples=1_000,
        num_chains=4,
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
│         * time           (time) object 2kB 1959-10-01 1960-01-01 ... 2009-07-01
│         * obs_dim        (obs_dim) <U8 96B 'realgdp' 'realcons' 'realinv'
│         * lag            (lag) int64 16B 1 2
│         * equation       (equation) <U8 96B 'realgdp' 'realcons' 'realinv'
│         * lagged_series  (lagged_series) <U8 96B 'realgdp' 'realcons' 'realinv'
│       Data variables:
│           intercept      (chain, draw, series) float32 48kB 0.1953 0.434 ... -1.811
│           l_omega        (chain, draw, l_omega_dim_0, l_omega_dim_1) float32 144kB ...
│           mu_t           (chain, draw, time, obs_dim) float32 10MB 1.047 ... -0.6855
│           phi            (chain, draw, lag, equation, lagged_series) float32 288kB ...
│           sigma          (chain, draw, series) float32 48kB 0.7401 0.6598 ... 3.809
│       Attributes:
│           created_at:                 2026-09-29T18:37:56.716020+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
├── Group: /posterior_predictive
│       Dimensions:  (chain: 4, draw: 1000, time: 200, obs_dim: 3)
│       Coordinates:
│         * chain    (chain) int64 32B 0 1 2 3
│         * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│         * time     (time) object 2kB 1959-10-01 1960-01-01 ... 2009-04-01 2009-07-01
│         * obs_dim  (obs_dim) <U8 96B 'realgdp' 'realcons' 'realinv'
│       Data variables:
│           obs      (chain, draw, time, obs_dim) float32 10MB 2.369 1.81 ... -2.793
│       Attributes:
│           created_at:                 2026-09-29T18:37:57.733652+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
├── Group: /observed_data
│       Dimensions:  (time: 200, obs_dim: 3)
│       Coordinates:
│         * time     (time) object 2kB 1959-10-01 1960-01-01 ... 2009-04-01 2009-07-01
│         * obs_dim  (obs_dim) <U8 96B 'realgdp' 'realcons' 'realinv'
│       Data variables:
│           obs      (time, obs_dim) float32 2kB 0.3495 0.1084 3.443 ... 0.7265 2.02
│       Attributes:
│           created_at:                 2026-09-29T18:37:57.734550+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /constant_data
│       Dimensions:        (time: 200, covariate_dim: 3)
│       Coordinates:
│         * time           (time) object 2kB 1959-10-01 1960-01-01 ... 2009-07-01
│         * covariate_dim  (covariate_dim) int64 24B 0 1 2
│       Data variables:
│           covariates     (time, covariate_dim) float32 2kB 0.3495 0.1084 ... 2.02
│       Attributes:
│           created_at:                 2026-09-29T18:37:57.735227+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /predictions
│       Dimensions:  (chain: 4, draw: 1000, time: 30, obs_dim: 3)
│       Coordinates:
│         * chain    (chain) int64 32B 0 1 2 3
│         * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│         * time     (time) object 240B 2009-10-01 2010-01-01 ... 2016-10-01 2017-01-01
│         * obs_dim  (obs_dim) <U8 96B 'realgdp' 'realcons' 'realinv'
│       Data variables:
│           obs      (chain, draw, time, obs_dim) float32 1MB 1.207 1.29 ... -2.456
│       Attributes:
│           created_at:                 2026-09-29T18:37:58.159090+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
└── Group: /predictions_constant_data
        Dimensions:        (time: 30, covariate_dim: 3)
        Coordinates:
          * time           (time) object 240B 2009-10-01 2010-01-01 ... 2017-01-01
          * covariate_dim  (covariate_dim) int64 24B 0 1 2
        Data variables:
            covariates     (time, covariate_dim) float32 360B 0.0 0.0 0.0 ... 0.0 0.0
        Attributes:
            created_at:                 2026-09-29T18:37:58.159731+00:00
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


object


1959-10-01 ... 2009-07-01


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([datetime.date(1959, 10, 1), datetime.date(1960, 1, 1),datetime.date(1960, 4, 1), datetime.date(1960, 7, 1),datetime.date(1960, 10, 1), datetime.date(1961, 1, 1),datetime.date(1961, 4, 1), datetime.date(1961, 7, 1),datetime.date(1961, 10, 1), datetime.date(1962, 1, 1),datetime.date(1962, 4, 1), datetime.date(1962, 7, 1),datetime.date(1962, 10, 1), datetime.date(1963, 1, 1),datetime.date(1963, 4, 1), datetime.date(1963, 7, 1),datetime.date(1963, 10, 1), datetime.date(1964, 1, 1),datetime.date(1964, 4, 1), datetime.date(1964, 7, 1),datetime.date(1964, 10, 1), datetime.date(1965, 1, 1),datetime.date(1965, 4, 1), datetime.date(1965, 7, 1),datetime.date(1965, 10, 1), datetime.date(1966, 1, 1),datetime.date(1966, 4, 1), datetime.date(1966, 7, 1),datetime.date(1966, 10, 1), datetime.date(1967, 1, 1),datetime.date(1967, 4, 1), datetime.date(1967, 7, 1),datetime.date(1967, 10, 1), datetime.date(1968, 1, 1),datetime.date(1968, 4, 1), datetime.date(1968, 7, 1),datetime.date(1968, 10, 1), datetime.date(1969, 1, 1),datetime.date(1969, 4, 1), datetime.date(1969, 7, 1),datetime.date(1969, 10, 1), datetime.date(1970, 1, 1),datetime.date(1970, 4, 1), datetime.date(1970, 7, 1),datetime.date(1970, 10, 1), datetime.date(1971, 1, 1),datetime.date(1971, 4, 1), datetime.date(1971, 7, 1),datetime.date(1971, 10, 1), datetime.date(1972, 1, 1),datetime.date(1972, 4, 1), datetime.date(1972, 7, 1),datetime.date(1972, 10, 1), datetime.date(1973, 1, 1),datetime.date(1973, 4, 1), datetime.date(1973, 7, 1),datetime.date(1973, 10, 1), datetime.date(1974, 1, 1),datetime.date(1974, 4, 1), datetime.date(1974, 7, 1),datetime.date(1974, 10, 1), datetime.date(1975, 1, 1),datetime.date(1975, 4, 1), datetime.date(1975, 7, 1),datetime.date(1975, 10, 1), datetime.date(1976, 1, 1),datetime.date(1976, 4, 1), datetime.date(1976, 7, 1),datetime.date(1976, 10, 1), datetime.date(1977, 1, 1),datetime.date(1977, 4, 1), datetime.date(1977, 7, 1),datetime.date(1977, 10, 1), datetime.date(1978, 1, 1),datetime.date(1978, 4, 1), datetime.date(1978, 7, 1),datetime.date(1978, 10, 1), datetime.date(1979, 1, 1),datetime.date(1979, 4, 1), datetime.date(1979, 7, 1),datetime.date(1979, 10, 1), datetime.date(1980, 1, 1),datetime.date(1980, 4, 1), datetime.date(1980, 7, 1),datetime.date(1980, 10, 1), datetime.date(1981, 1, 1),datetime.date(1981, 4, 1), datetime.date(1981, 7, 1),datetime.date(1981, 10, 1), datetime.date(1982, 1, 1),datetime.date(1982, 4, 1), datetime.date(1982, 7, 1),datetime.date(1982, 10, 1), datetime.date(1983, 1, 1),datetime.date(1983, 4, 1), datetime.date(1983, 7, 1),datetime.date(1983, 10, 1), datetime.date(1984, 1, 1),datetime.date(1984, 4, 1), datetime.date(1984, 7, 1),datetime.date(1984, 10, 1), datetime.date(1985, 1, 1),datetime.date(1985, 4, 1), datetime.date(1985, 7, 1),datetime.date(1985, 10, 1), datetime.date(1986, 1, 1),datetime.date(1986, 4, 1), datetime.date(1986, 7, 1),datetime.date(1986, 10, 1), datetime.date(1987, 1, 1),datetime.date(1987, 4, 1), datetime.date(1987, 7, 1),datetime.date(1987, 10, 1), datetime.date(1988, 1, 1),datetime.date(1988, 4, 1), datetime.date(1988, 7, 1),datetime.date(1988, 10, 1), datetime.date(1989, 1, 1),datetime.date(1989, 4, 1), datetime.date(1989, 7, 1),datetime.date(1989, 10, 1), datetime.date(1990, 1, 1),datetime.date(1990, 4, 1), datetime.date(1990, 7, 1),datetime.date(1990, 10, 1), datetime.date(1991, 1, 1),datetime.date(1991, 4, 1), datetime.date(1991, 7, 1),datetime.date(1991, 10, 1), datetime.date(1992, 1, 1),datetime.date(1992, 4, 1), datetime.date(1992, 7, 1),datetime.date(1992, 10, 1), datetime.date(1993, 1, 1),datetime.date(1993, 4, 1), datetime.date(1993, 7, 1),datetime.date(1993, 10, 1), datetime.date(1994, 1, 1),datetime.date(1994, 4, 1), datetime.date(1994, 7, 1),datetime.date(1994, 10, 1), datetime.date(1995, 1, 1),datetime.date(1995, 4, 1), datetime.date(1995, 7, 1),datetime.date(1995, 10, 1), datetime.date(1996, 1, 1),datetime.date(1996, 4, 1), datetime.date(1996, 7, 1),datetime.date(1996, 10, 1), datetime.date(1997, 1, 1),datetime.date(1997, 4, 1), datetime.date(1997, 7, 1),datetime.date(1997, 10, 1), datetime.date(1998, 1, 1),datetime.date(1998, 4, 1), datetime.date(1998, 7, 1),datetime.date(1998, 10, 1), datetime.date(1999, 1, 1),datetime.date(1999, 4, 1), datetime.date(1999, 7, 1),datetime.date(1999, 10, 1), datetime.date(2000, 1, 1),datetime.date(2000, 4, 1), datetime.date(2000, 7, 1),datetime.date(2000, 10, 1), datetime.date(2001, 1, 1),datetime.date(2001, 4, 1), datetime.date(2001, 7, 1),datetime.date(2001, 10, 1), datetime.date(2002, 1, 1),datetime.date(2002, 4, 1), datetime.date(2002, 7, 1),datetime.date(2002, 10, 1), datetime.date(2003, 1, 1),datetime.date(2003, 4, 1), datetime.date(2003, 7, 1),datetime.date(2003, 10, 1), datetime.date(2004, 1, 1),datetime.date(2004, 4, 1), datetime.date(2004, 7, 1),datetime.date(2004, 10, 1), datetime.date(2005, 1, 1),datetime.date(2005, 4, 1), datetime.date(2005, 7, 1),datetime.date(2005, 10, 1), datetime.date(2006, 1, 1),datetime.date(2006, 4, 1), datetime.date(2006, 7, 1),datetime.date(2006, 10, 1), datetime.date(2007, 1, 1),datetime.date(2007, 4, 1), datetime.date(2007, 7, 1),datetime.date(2007, 10, 1), datetime.date(2008, 1, 1),datetime.date(2008, 4, 1), datetime.date(2008, 7, 1),datetime.date(2008, 10, 1), datetime.date(2009, 1, 1),datetime.date(2009, 4, 1), datetime.date(2009, 7, 1)], dtype=object)


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


0.1953 0.434 ... 0.4596 -1.811


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 0.19533381,  0.43399125, -1.3496677 ],[ 0.02001988,  0.47910476, -2.8849473 ],[ 0.37479222,  0.5478512 , -0.7046817 ],...,[ 0.21476845,  0.697626  , -1.8457513 ],[ 0.32380873,  0.59099406, -1.3261341 ],[ 0.11906592,  0.4923342 , -2.1622655 ]],[[ 0.18783535,  0.5830818 , -2.3680563 ],[ 0.17366871,  0.56143403, -1.9193435 ],[ 0.20822224,  0.5013449 , -1.6690441 ],...,[ 0.31148884,  0.5950913 , -1.3193196 ],[ 0.2378505 ,  0.5792743 , -1.5260807 ],[ 0.25894624,  0.5495803 , -1.5004284 ]],[[ 0.2775513 ,  0.57503265, -1.3967906 ],[ 0.39888707,  0.5401912 , -1.8103585 ],[ 0.22195393,  0.60488456, -1.3868365 ],...,[-0.01190321,  0.49469736, -3.6773021 ],[ 0.04653886,  0.48369208, -2.3646975 ],[ 0.38902363,  0.65436566, -1.1373843 ]],[[ 0.2555071 ,  0.4745961 , -1.8843788 ],[ 0.21253653,  0.6408943 , -1.8882742 ],[ 0.12132851,  0.53722805, -2.0711198 ],...,[ 0.4357751 ,  0.5734798 , -1.0294183 ],[ 0.53416586,  0.6669533 , -0.8117105 ],[ 0.07923683,  0.4596019 , -1.8113005 ]]],shape=(4, 1000, 3), dtype=float32)


l_omega


(chain, draw, l_omega_dim_0, l_omega_dim_1)


float32


1.0 0.0 0.0 ... -0.5064 0.5422


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 1.        ,  0.        ,  0.        ],[ 0.6075029 ,  0.7943175 ,  0.        ],[ 0.6817714 , -0.42048037,  0.5986518 ]],[[ 1.        ,  0.        ,  0.        ],[ 0.5642413 ,  0.8256099 ,  0.        ],[ 0.73811966, -0.3834894 ,  0.55508125]],[[ 1.        ,  0.        ,  0.        ],[ 0.60843265,  0.7936055 ,  0.        ],[ 0.7569262 , -0.43272614,  0.4897048 ]],...,[[ 1.        ,  0.        ,  0.        ],[ 0.53031605,  0.8478    ,  0.        ],[ 0.7705926 , -0.3788765 ,  0.51248384]],[[ 1.        ,  0.        ,  0.        ],[ 0.5598801 ,  0.82857364,  0.        ],...[ 0.7299583 , -0.36309686,  0.57906955]],[[ 1.        ,  0.        ,  0.        ],[ 0.614576  ,  0.78885764,  0.        ],[ 0.7368513 , -0.42001972,  0.5297487 ]],...,[[ 1.        ,  0.        ,  0.        ],[ 0.6136306 ,  0.7895932 ,  0.        ],[ 0.73827595, -0.40280867,  0.5410118 ]],[[ 1.        ,  0.        ,  0.        ],[ 0.49490193,  0.8689488 ,  0.        ],[ 0.7647471 , -0.43038535,  0.4795105 ]],[[ 1.        ,  0.        ,  0.        ],[ 0.6002639 ,  0.799802  ,  0.        ],[ 0.67050534, -0.50638175,  0.54221785]]]],shape=(4, 1000, 3, 3), dtype=float32)


mu_t


(chain, draw, time, obs_dim)


float32


1.047 0.9763 ... 0.3063 -0.6855


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 1.04745233e+00,  9.76337552e-01,  1.06093490e+00],[ 5.69742441e-01,  7.02627540e-01,  2.50433564e-01],[ 4.59447443e-01,  7.41379142e-01,  2.78019309e-01],...,[-2.26395041e-01,  6.44006729e-02, -3.13409781e+00],[ 6.07372880e-01,  4.13133860e-01,  8.11961770e-01],[ 7.49149323e-01,  6.54044628e-01,  1.57844841e+00]],[[ 7.91435301e-01,  7.04959333e-01,  6.39264345e-01],[ 5.00586987e-01,  6.80265129e-01, -9.58739519e-01],[ 5.99307179e-01,  1.06219971e+00,  3.09139967e-01],...,[-7.04783440e-01, -6.59930706e-03, -6.39864445e+00],[-4.39643115e-01, -1.49276555e-01, -4.32815838e+00],[-4.77473661e-02,  2.12279320e-01, -3.33214068e+00]],[[ 1.17406380e+00,  9.24458444e-01,  2.26464534e+00],[ 7.18153179e-01,  8.32193971e-01,  5.37357152e-01],[ 3.54117930e-01,  7.10481763e-01, -7.28923500e-01],...,...[-3.01060677e-02,  2.23856956e-01, -3.86805296e+00],[ 2.37419531e-01,  4.73336041e-01, -2.95266771e+00],[ 3.14729959e-01,  4.57594275e-01, -3.33029461e+00]],[[ 9.26073730e-01,  8.93298388e-01,  1.35402191e+00],[ 6.19408250e-01,  7.40119934e-01,  2.30540395e-01],[ 1.12317324e+00,  1.17056656e+00,  2.13252783e+00],...,[-4.87371087e-02,  2.03608811e-01, -3.71786022e+00],[-1.32866502e-01, -1.80790424e-01, -2.19478393e+00],[-2.97352076e-01, -1.91844165e-01, -2.13460112e+00]],[[ 1.21989322e+00,  9.79644418e-01,  1.94146252e+00],[ 4.71203089e-01,  6.00906491e-01,  1.08475804e-01],[ 3.90349060e-01,  7.56774068e-01,  2.67541409e-02],...,[-4.43469852e-01,  7.96978474e-02, -4.16768789e+00],[ 2.10705608e-01,  2.39707559e-01, -1.70855451e+00],[ 3.98215353e-01,  3.06282699e-01, -6.85485601e-01]]]],shape=(4, 1000, 200, 3), dtype=float32)


phi


(chain, draw, lag, equation, lagged_series)


float32


0.1282 0.4035 ... -0.21 -0.2661


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[[ 1.28241867e-01,  4.03513074e-01, -3.39466520e-02],[ 1.11296795e-01,  2.16696650e-01, -1.20512648e-02],[ 4.72965628e-01,  2.25463629e+00, -1.05234131e-01]],[[ 4.63260934e-02,  2.01933041e-01, -2.75212731e-02],[ 8.81871060e-02,  1.15843296e-01, -1.91242769e-02],[-1.64897311e-02,  4.21785444e-01, -1.54254511e-01]]],[[[ 8.70535709e-03,  3.96550030e-01,  1.55264558e-02],[ 4.43829782e-02,  2.00385064e-01,  2.73950528e-02],[-6.19165115e-02,  2.64165878e+00,  7.28246272e-02]],[[-1.08462516e-02,  3.43568653e-01, -3.18378326e-03],[ 9.01626721e-02,  3.17228213e-02, -6.57547591e-03],[-3.89418423e-01,  1.41117024e+00,  1.40449489e-02]]],[[[-3.40301991e-01,  5.48127651e-01,  2.30244230e-02],[-1.45874277e-01,  2.66937256e-01,  2.17545237e-02],...[-5.18218949e-02,  2.44060650e-01,  7.44785517e-02]]],[[[-1.55467764e-01,  3.96123916e-01,  4.09754105e-02],[-2.69884646e-01,  3.66314113e-01,  6.50940463e-02],[ 1.41603589e-01,  1.93368053e+00,  4.66697700e-02]],[[-2.82782286e-01,  3.26395929e-01,  5.78330979e-02],[ 2.71870499e-03,  6.32355213e-02,  2.24132445e-02],[-1.39512885e+00,  1.67606688e+00,  1.78106219e-01]]],[[[ 1.18741803e-01,  3.68676722e-01, -2.85019241e-02],[ 1.82769716e-01,  6.51104189e-03, -2.04964988e-02],[-6.89141452e-01,  3.05417442e+00,  8.69374722e-02]],[[ 1.92490652e-01,  2.10971773e-01, -2.94565484e-02],[-1.75995797e-01,  3.60211670e-01,  3.43571566e-02],[ 1.43581283e+00, -2.09971935e-01, -2.66127020e-01]]]]],shape=(4, 1000, 2, 3, 3), dtype=float32)


sigma


(chain, draw, series)


float32


0.7401 0.6598 3.86 ... 0.6915 3.809


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.7400705 , 0.65980196, 3.8600922 ],[0.7749313 , 0.6594389 , 4.137488  ],[0.7606299 , 0.6661901 , 3.697686  ],...,[0.7663137 , 0.6382282 , 4.2874846 ],[0.740348  , 0.65771025, 4.0772166 ],[0.7066104 , 0.6778794 , 3.5848584 ]],[[0.70194113, 0.63596123, 3.659458  ],[0.691268  , 0.6078301 , 3.6134305 ],[0.7816354 , 0.6850775 , 4.0952597 ],...,[0.7354155 , 0.61889124, 3.5470347 ],[0.73065996, 0.674034  , 3.8756945 ],[0.7783619 , 0.71084917, 3.7421072 ]],[[0.78505147, 0.64456797, 4.1005535 ],[0.8032606 , 0.6331405 , 4.0744505 ],[0.6837602 , 0.61223674, 3.4730036 ],...,[0.7947782 , 0.6737024 , 4.0648384 ],[0.75567025, 0.63698053, 3.9976614 ],[0.7539399 , 0.6266143 , 3.8750224 ]],[[0.7214086 , 0.606192  , 3.7481334 ],[0.72016746, 0.665495  , 3.9396014 ],[0.6971102 , 0.62767166, 3.675084  ],...,[0.75522906, 0.6409284 , 4.159316  ],[0.7227357 , 0.6478196 , 4.1206894 ],[0.7256212 , 0.6915238 , 3.8089597 ]]],shape=(4, 1000, 3), dtype=float32)


Attributes: (5)


created_at :  
2026-09-29T18:37:56.716020+00:00

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


object


1959-10-01 ... 2009-07-01


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


ime.date(1972, 4, 1), datetime.date(1972, 7, 1),datetime.date(1972, 10, 1), datetime.date(1973, 1, 1),datetime.date(1973, 4, 1), datetime.date(1973, 7, 1),datetime.date(1973, 10, 1), datetime.date(1974, 1, 1),datetime.date(1974, 4, 1), datetime.date(1974, 7, 1),datetime.date(1974, 10, 1), datetime.date(1975, 1, 1),datetime.date(1975, 4, 1), datetime.date(1975, 7, 1),datetime.date(1975, 10, 1), datetime.date(1976, 1, 1),datetime.date(1976, 4, 1), datetime.date(1976, 7, 1),datetime.date(1976, 10, 1), datetime.date(1977, 1, 1),datetime.date(1977, 4, 1), datetime.date(1977, 7, 1),datetime.date(1977, 10, 1), datetime.date(1978, 1, 1),datetime.date(1978, 4, 1), datetime.date(1978, 7, 1),datetime.date(1978, 10, 1), datetime.date(1979, 1, 1),datetime.date(1979, 4, 1), datetime.date(1979, 7, 1),datetime.date(1979, 10, 1), datetime.date(1980, 1, 1),datetime.date(1980, 4, 1), datetime.date(1980, 7, 1),datetime.date(1980, 10, 1), datetime.date(1981, 1, 1),datetime.date(1981, 4, 1), datetime.date(1981, 7, 1),datetime.date(1981, 10, 1), datetime.date(1982, 1, 1),datetime.date(1982, 4, 1), datetime.date(1982, 7, 1),datetime.date(1982, 10, 1), datetime.date(1983, 1, 1),datetime.date(1983, 4, 1), datetime.date(1983, 7, 1),datetime.date(1983, 10, 1), datetime.date(1984, 1, 1),datetime.date(1984, 4, 1), datetime.date(1984, 7, 1),datetime.date(1984, 10, 1), datetime.date(1985, 1, 1),datetime.date(1985, 4, 1), datetime.date(1985, 7, 1),datetime.date(1985, 10, 1), datetime.date(1986, 1, 1),datetime.date(1986, 4, 1), datetime.date(1986, 7, 1),datetime.date(1986, 10, 1), datetime.date(1987, 1, 1),datetime.date(1987, 4, 1), datetime.date(1987, 7, 1),datetime.date(1987, 10, 1), datetime.date(1988, 1, 1),datetime.date(1988, 4, 1), datetime.date(1988, 7, 1),datetime.date(1988, 10, 1), datetime.date(1989, 1, 1),datetime.date(1989, 4, 1), datetime.date(1989, 7, 1),datetime.date(1989, 10, 1), datetime.date(1990, 1, 1),datetime.date(1990, 4, 1), datetime.date(1990, 7, 1),datetime.date(1990, 10, 1), datetime.date(1991, 1, 1),datetime.date(1991, 4, 1), datetime.date(1991, 7, 1),datetime.date(1991, 10, 1), datetime.date(1992, 1, 1),datetime.date(1992, 4, 1), datetime.date(1992, 7, 1),datetime.date(1992, 10, 1), datetime.date(1993, 1, 1),datetime.date(1993, 4, 1), datetime.date(1993, 7, 1),datetime.date(1993, 10, 1), datetime.date(1994, 1, 1),datetime.date(1994, 4, 1), datetime.date(1994, 7, 1),datetime.date(1994, 10, 1), datetime.date(1995, 1, 1),datetime.date(1995, 4, 1), datetime.date(1995, 7, 1),datetime.date(1995, 10, 1), datetime.date(1996, 1, 1),datetime.date(1996, 4, 1), datetime.date(1996, 7, 1),datetime.date(1996, 10, 1), datetime.date(1997, 1, 1),datetime.date(1997, 4, 1), datetime.date(1997, 7, 1),datetime.date(1997, 10, 1), datetime.date(1998, 1, 1),datetime.date(1998, 4, 1), datetime.date(1998, 7, 1),datetime.date(1998, 10, 1), datetime.date(1999, 1, 1),datetime.date(1999, 4, 1), datetime.date(1999, 7, 1),datetime.date(1999, 10, 1), datetime.date(2000, 1, 1),datetime.date(2000, 4, 1), datetime.date(2000, 7, 1),datetime.date(2000, 10, 1), datetime.date(2001, 1, 1),datetime.date(2001, 4, 1), datetime.date(2001, 7, 1),datetime.date(2001, 10, 1), datetime.date(2002, 1, 1),datetime.date(2002, 4, 1), datetime.date(2002, 7, 1),datetime.date(2002, 10, 1), datetime.date(2003, 1, 1),datetime.date(2003, 4, 1), datetime.date(2003, 7, 1),datetime.date(2003, 10, 1), datetime.date(2004, 1, 1),datetime.date(2004, 4, 1), datetime.date(2004, 7, 1),datetime.date(2004, 10, 1), datetime.date(2005, 1, 1),datetime.date(2005, 4, 1), datetime.date(2005, 7, 1),datetime.date(2005, 10, 1), datetime.date(2006, 1, 1),datetime.date(2006, 4, 1), datetime.date(2006, 7, 1),datetime.date(2006, 10, 1), datetime.date(2007, 1, 1),datetime.date(2007, 4, 1), datetime.date(2007, 7, 1),datetime.date(2007, 10, 1), datetime.date(2008, 1, 1),datetime.date(2008, 4, 1), datetime.date(2008, 7, 1),datetime.date(2008, 10, 1), datetime.date(2009, 1, 1),datetime.date(2009, 4, 1), datetime.date(2009, 7, 1)], dtype=object)


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


2.369 1.81 1.288 ... 0.2418 -2.793


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 2.36938500e+00,  1.80970657e+00,  1.28773475e+00],[ 1.01300693e+00,  3.52852702e-01,  3.67349148e+00],[ 3.99645805e-01,  7.68919945e-01, -1.43399787e+00],...,[ 2.67722905e-02,  1.77623987e-01,  3.82750201e+00],[ 1.69704998e+00,  6.45348072e-01,  4.67950058e+00],[ 1.01084077e+00,  1.32256889e+00,  1.92360759e-01]],[[-4.63989317e-01, -4.71418083e-01, -1.71406531e+00],[ 1.18028641e+00,  1.19217098e+00,  4.28773594e+00],[ 9.25782263e-01,  1.81722569e+00,  3.02850342e+00],...,[-1.93392527e+00,  2.83901542e-01, -1.14555035e+01],[ 3.83404106e-01,  3.43319833e-01, -2.21220374e+00],[ 2.19941169e-01,  5.18263221e-01,  1.23630786e+00]],[[ 3.41543376e-01, -6.53452277e-02, -6.33448839e-01],[-1.14518702e-01,  3.74040097e-01, -4.12539959e+00],[ 6.74601197e-01,  9.60196018e-01, -2.46743417e+00],...,...[-1.47838324e-01,  1.03326523e+00, -6.34603691e+00],[ 1.79830194e-01,  6.39166176e-01, -5.54126978e+00],[-5.42129636e-01,  1.07204843e+00, -1.12391195e+01]],[[ 1.02627850e+00,  1.02928388e+00,  7.11708963e-01],[ 1.25904322e+00,  4.14302349e-02,  6.97592640e+00],[ 1.00803423e+00,  1.45398557e+00,  2.20830631e+00],...,[ 2.06223652e-02,  3.49572361e-01, -5.56141472e+00],[-4.28573579e-01, -2.91659504e-01, -1.07204580e+00],[ 3.33901465e-01,  7.15805352e-01,  4.44431305e-01]],[[ 2.86929059e+00,  2.67677927e+00,  3.51293278e+00],[-1.11281574e-01,  7.22499907e-01, -6.43599463e+00],[-2.63474911e-01,  6.28835201e-01, -4.44896030e+00],...,[ 1.32490528e+00,  1.18739438e+00, -8.52117538e-02],[ 1.65076923e+00,  1.96583605e+00,  2.72600627e+00],[ 5.10092974e-02,  2.41800845e-01, -2.79285026e+00]]]],shape=(4, 1000, 200, 3), dtype=float32)


Attributes: (5)


created_at :  
2026-09-29T18:37:57.733652+00:00

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


object


1959-10-01 ... 2009-07-01


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([datetime.date(1959, 10, 1), datetime.date(1960, 1, 1),datetime.date(1960, 4, 1), datetime.date(1960, 7, 1),datetime.date(1960, 10, 1), datetime.date(1961, 1, 1),datetime.date(1961, 4, 1), datetime.date(1961, 7, 1),datetime.date(1961, 10, 1), datetime.date(1962, 1, 1),datetime.date(1962, 4, 1), datetime.date(1962, 7, 1),datetime.date(1962, 10, 1), datetime.date(1963, 1, 1),datetime.date(1963, 4, 1), datetime.date(1963, 7, 1),datetime.date(1963, 10, 1), datetime.date(1964, 1, 1),datetime.date(1964, 4, 1), datetime.date(1964, 7, 1),datetime.date(1964, 10, 1), datetime.date(1965, 1, 1),datetime.date(1965, 4, 1), datetime.date(1965, 7, 1),datetime.date(1965, 10, 1), datetime.date(1966, 1, 1),datetime.date(1966, 4, 1), datetime.date(1966, 7, 1),datetime.date(1966, 10, 1), datetime.date(1967, 1, 1),datetime.date(1967, 4, 1), datetime.date(1967, 7, 1),datetime.date(1967, 10, 1), datetime.date(1968, 1, 1),datetime.date(1968, 4, 1), datetime.date(1968, 7, 1),datetime.date(1968, 10, 1), datetime.date(1969, 1, 1),datetime.date(1969, 4, 1), datetime.date(1969, 7, 1),datetime.date(1969, 10, 1), datetime.date(1970, 1, 1),datetime.date(1970, 4, 1), datetime.date(1970, 7, 1),datetime.date(1970, 10, 1), datetime.date(1971, 1, 1),datetime.date(1971, 4, 1), datetime.date(1971, 7, 1),datetime.date(1971, 10, 1), datetime.date(1972, 1, 1),datetime.date(1972, 4, 1), datetime.date(1972, 7, 1),datetime.date(1972, 10, 1), datetime.date(1973, 1, 1),datetime.date(1973, 4, 1), datetime.date(1973, 7, 1),datetime.date(1973, 10, 1), datetime.date(1974, 1, 1),datetime.date(1974, 4, 1), datetime.date(1974, 7, 1),datetime.date(1974, 10, 1), datetime.date(1975, 1, 1),datetime.date(1975, 4, 1), datetime.date(1975, 7, 1),datetime.date(1975, 10, 1), datetime.date(1976, 1, 1),datetime.date(1976, 4, 1), datetime.date(1976, 7, 1),datetime.date(1976, 10, 1), datetime.date(1977, 1, 1),datetime.date(1977, 4, 1), datetime.date(1977, 7, 1),datetime.date(1977, 10, 1), datetime.date(1978, 1, 1),datetime.date(1978, 4, 1), datetime.date(1978, 7, 1),datetime.date(1978, 10, 1), datetime.date(1979, 1, 1),datetime.date(1979, 4, 1), datetime.date(1979, 7, 1),datetime.date(1979, 10, 1), datetime.date(1980, 1, 1),datetime.date(1980, 4, 1), datetime.date(1980, 7, 1),datetime.date(1980, 10, 1), datetime.date(1981, 1, 1),datetime.date(1981, 4, 1), datetime.date(1981, 7, 1),datetime.date(1981, 10, 1), datetime.date(1982, 1, 1),datetime.date(1982, 4, 1), datetime.date(1982, 7, 1),datetime.date(1982, 10, 1), datetime.date(1983, 1, 1),datetime.date(1983, 4, 1), datetime.date(1983, 7, 1),datetime.date(1983, 10, 1), datetime.date(1984, 1, 1),datetime.date(1984, 4, 1), datetime.date(1984, 7, 1),datetime.date(1984, 10, 1), datetime.date(1985, 1, 1),datetime.date(1985, 4, 1), datetime.date(1985, 7, 1),datetime.date(1985, 10, 1), datetime.date(1986, 1, 1),datetime.date(1986, 4, 1), datetime.date(1986, 7, 1),datetime.date(1986, 10, 1), datetime.date(1987, 1, 1),datetime.date(1987, 4, 1), datetime.date(1987, 7, 1),datetime.date(1987, 10, 1), datetime.date(1988, 1, 1),datetime.date(1988, 4, 1), datetime.date(1988, 7, 1),datetime.date(1988, 10, 1), datetime.date(1989, 1, 1),datetime.date(1989, 4, 1), datetime.date(1989, 7, 1),datetime.date(1989, 10, 1), datetime.date(1990, 1, 1),datetime.date(1990, 4, 1), datetime.date(1990, 7, 1),datetime.date(1990, 10, 1), datetime.date(1991, 1, 1),datetime.date(1991, 4, 1), datetime.date(1991, 7, 1),datetime.date(1991, 10, 1), datetime.date(1992, 1, 1),datetime.date(1992, 4, 1), datetime.date(1992, 7, 1),datetime.date(1992, 10, 1), datetime.date(1993, 1, 1),datetime.date(1993, 4, 1), datetime.date(1993, 7, 1),datetime.date(1993, 10, 1), datetime.date(1994, 1, 1),datetime.date(1994, 4, 1), datetime.date(1994, 7, 1),datetime.date(1994, 10, 1), datetime.date(1995, 1, 1),datetime.date(1995, 4, 1), datetime.date(1995, 7, 1),datetime.date(1995, 10, 1), datetime.date(1996, 1, 1),datetime.date(1996, 4, 1), datetime.date(1996, 7, 1),datetime.date(1996, 10, 1), datetime.date(1997, 1, 1),datetime.date(1997, 4, 1), datetime.date(1997, 7, 1),datetime.date(1997, 10, 1), datetime.date(1998, 1, 1),datetime.date(1998, 4, 1), datetime.date(1998, 7, 1),datetime.date(1998, 10, 1), datetime.date(1999, 1, 1),datetime.date(1999, 4, 1), datetime.date(1999, 7, 1),datetime.date(1999, 10, 1), datetime.date(2000, 1, 1),datetime.date(2000, 4, 1), datetime.date(2000, 7, 1),datetime.date(2000, 10, 1), datetime.date(2001, 1, 1),datetime.date(2001, 4, 1), datetime.date(2001, 7, 1),datetime.date(2001, 10, 1), datetime.date(2002, 1, 1),datetime.date(2002, 4, 1), datetime.date(2002, 7, 1),datetime.date(2002, 10, 1), datetime.date(2003, 1, 1),datetime.date(2003, 4, 1), datetime.date(2003, 7, 1),datetime.date(2003, 10, 1), datetime.date(2004, 1, 1),datetime.date(2004, 4, 1), datetime.date(2004, 7, 1),datetime.date(2004, 10, 1), datetime.date(2005, 1, 1),datetime.date(2005, 4, 1), datetime.date(2005, 7, 1),datetime.date(2005, 10, 1), datetime.date(2006, 1, 1),datetime.date(2006, 4, 1), datetime.date(2006, 7, 1),datetime.date(2006, 10, 1), datetime.date(2007, 1, 1),datetime.date(2007, 4, 1), datetime.date(2007, 7, 1),datetime.date(2007, 10, 1), datetime.date(2008, 1, 1),datetime.date(2008, 4, 1), datetime.date(2008, 7, 1),datetime.date(2008, 10, 1), datetime.date(2009, 1, 1),datetime.date(2009, 4, 1), datetime.date(2009, 7, 1)], dtype=object)


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
2026-09-29T18:37:57.734550+00:00

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


object


1959-10-01 ... 2009-07-01


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([datetime.date(1959, 10, 1), datetime.date(1960, 1, 1),datetime.date(1960, 4, 1), datetime.date(1960, 7, 1),datetime.date(1960, 10, 1), datetime.date(1961, 1, 1),datetime.date(1961, 4, 1), datetime.date(1961, 7, 1),datetime.date(1961, 10, 1), datetime.date(1962, 1, 1),datetime.date(1962, 4, 1), datetime.date(1962, 7, 1),datetime.date(1962, 10, 1), datetime.date(1963, 1, 1),datetime.date(1963, 4, 1), datetime.date(1963, 7, 1),datetime.date(1963, 10, 1), datetime.date(1964, 1, 1),datetime.date(1964, 4, 1), datetime.date(1964, 7, 1),datetime.date(1964, 10, 1), datetime.date(1965, 1, 1),datetime.date(1965, 4, 1), datetime.date(1965, 7, 1),datetime.date(1965, 10, 1), datetime.date(1966, 1, 1),datetime.date(1966, 4, 1), datetime.date(1966, 7, 1),datetime.date(1966, 10, 1), datetime.date(1967, 1, 1),datetime.date(1967, 4, 1), datetime.date(1967, 7, 1),datetime.date(1967, 10, 1), datetime.date(1968, 1, 1),datetime.date(1968, 4, 1), datetime.date(1968, 7, 1),datetime.date(1968, 10, 1), datetime.date(1969, 1, 1),datetime.date(1969, 4, 1), datetime.date(1969, 7, 1),datetime.date(1969, 10, 1), datetime.date(1970, 1, 1),datetime.date(1970, 4, 1), datetime.date(1970, 7, 1),datetime.date(1970, 10, 1), datetime.date(1971, 1, 1),datetime.date(1971, 4, 1), datetime.date(1971, 7, 1),datetime.date(1971, 10, 1), datetime.date(1972, 1, 1),datetime.date(1972, 4, 1), datetime.date(1972, 7, 1),datetime.date(1972, 10, 1), datetime.date(1973, 1, 1),datetime.date(1973, 4, 1), datetime.date(1973, 7, 1),datetime.date(1973, 10, 1), datetime.date(1974, 1, 1),datetime.date(1974, 4, 1), datetime.date(1974, 7, 1),datetime.date(1974, 10, 1), datetime.date(1975, 1, 1),datetime.date(1975, 4, 1), datetime.date(1975, 7, 1),datetime.date(1975, 10, 1), datetime.date(1976, 1, 1),datetime.date(1976, 4, 1), datetime.date(1976, 7, 1),datetime.date(1976, 10, 1), datetime.date(1977, 1, 1),datetime.date(1977, 4, 1), datetime.date(1977, 7, 1),datetime.date(1977, 10, 1), datetime.date(1978, 1, 1),datetime.date(1978, 4, 1), datetime.date(1978, 7, 1),datetime.date(1978, 10, 1), datetime.date(1979, 1, 1),datetime.date(1979, 4, 1), datetime.date(1979, 7, 1),datetime.date(1979, 10, 1), datetime.date(1980, 1, 1),datetime.date(1980, 4, 1), datetime.date(1980, 7, 1),datetime.date(1980, 10, 1), datetime.date(1981, 1, 1),datetime.date(1981, 4, 1), datetime.date(1981, 7, 1),datetime.date(1981, 10, 1), datetime.date(1982, 1, 1),datetime.date(1982, 4, 1), datetime.date(1982, 7, 1),datetime.date(1982, 10, 1), datetime.date(1983, 1, 1),datetime.date(1983, 4, 1), datetime.date(1983, 7, 1),datetime.date(1983, 10, 1), datetime.date(1984, 1, 1),datetime.date(1984, 4, 1), datetime.date(1984, 7, 1),datetime.date(1984, 10, 1), datetime.date(1985, 1, 1),datetime.date(1985, 4, 1), datetime.date(1985, 7, 1),datetime.date(1985, 10, 1), datetime.date(1986, 1, 1),datetime.date(1986, 4, 1), datetime.date(1986, 7, 1),datetime.date(1986, 10, 1), datetime.date(1987, 1, 1),datetime.date(1987, 4, 1), datetime.date(1987, 7, 1),datetime.date(1987, 10, 1), datetime.date(1988, 1, 1),datetime.date(1988, 4, 1), datetime.date(1988, 7, 1),datetime.date(1988, 10, 1), datetime.date(1989, 1, 1),datetime.date(1989, 4, 1), datetime.date(1989, 7, 1),datetime.date(1989, 10, 1), datetime.date(1990, 1, 1),datetime.date(1990, 4, 1), datetime.date(1990, 7, 1),datetime.date(1990, 10, 1), datetime.date(1991, 1, 1),datetime.date(1991, 4, 1), datetime.date(1991, 7, 1),datetime.date(1991, 10, 1), datetime.date(1992, 1, 1),datetime.date(1992, 4, 1), datetime.date(1992, 7, 1),datetime.date(1992, 10, 1), datetime.date(1993, 1, 1),datetime.date(1993, 4, 1), datetime.date(1993, 7, 1),datetime.date(1993, 10, 1), datetime.date(1994, 1, 1),datetime.date(1994, 4, 1), datetime.date(1994, 7, 1),datetime.date(1994, 10, 1), datetime.date(1995, 1, 1),datetime.date(1995, 4, 1), datetime.date(1995, 7, 1),datetime.date(1995, 10, 1), datetime.date(1996, 1, 1),datetime.date(1996, 4, 1), datetime.date(1996, 7, 1),datetime.date(1996, 10, 1), datetime.date(1997, 1, 1),datetime.date(1997, 4, 1), datetime.date(1997, 7, 1),datetime.date(1997, 10, 1), datetime.date(1998, 1, 1),datetime.date(1998, 4, 1), datetime.date(1998, 7, 1),datetime.date(1998, 10, 1), datetime.date(1999, 1, 1),datetime.date(1999, 4, 1), datetime.date(1999, 7, 1),datetime.date(1999, 10, 1), datetime.date(2000, 1, 1),datetime.date(2000, 4, 1), datetime.date(2000, 7, 1),datetime.date(2000, 10, 1), datetime.date(2001, 1, 1),datetime.date(2001, 4, 1), datetime.date(2001, 7, 1),datetime.date(2001, 10, 1), datetime.date(2002, 1, 1),datetime.date(2002, 4, 1), datetime.date(2002, 7, 1),datetime.date(2002, 10, 1), datetime.date(2003, 1, 1),datetime.date(2003, 4, 1), datetime.date(2003, 7, 1),datetime.date(2003, 10, 1), datetime.date(2004, 1, 1),datetime.date(2004, 4, 1), datetime.date(2004, 7, 1),datetime.date(2004, 10, 1), datetime.date(2005, 1, 1),datetime.date(2005, 4, 1), datetime.date(2005, 7, 1),datetime.date(2005, 10, 1), datetime.date(2006, 1, 1),datetime.date(2006, 4, 1), datetime.date(2006, 7, 1),datetime.date(2006, 10, 1), datetime.date(2007, 1, 1),datetime.date(2007, 4, 1), datetime.date(2007, 7, 1),datetime.date(2007, 10, 1), datetime.date(2008, 1, 1),datetime.date(2008, 4, 1), datetime.date(2008, 7, 1),datetime.date(2008, 10, 1), datetime.date(2009, 1, 1),datetime.date(2009, 4, 1), datetime.date(2009, 7, 1)], dtype=object)


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
2026-09-29T18:37:57.735227+00:00

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


object


2009-10-01 ... 2017-01-01


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([datetime.date(2009, 10, 1), datetime.date(2010, 1, 1),datetime.date(2010, 4, 1), datetime.date(2010, 7, 1),datetime.date(2010, 10, 1), datetime.date(2011, 1, 1),datetime.date(2011, 4, 1), datetime.date(2011, 7, 1),datetime.date(2011, 10, 1), datetime.date(2012, 1, 1),datetime.date(2012, 4, 1), datetime.date(2012, 7, 1),datetime.date(2012, 10, 1), datetime.date(2013, 1, 1),datetime.date(2013, 4, 1), datetime.date(2013, 7, 1),datetime.date(2013, 10, 1), datetime.date(2014, 1, 1),datetime.date(2014, 4, 1), datetime.date(2014, 7, 1),datetime.date(2014, 10, 1), datetime.date(2015, 1, 1),datetime.date(2015, 4, 1), datetime.date(2015, 7, 1),datetime.date(2015, 10, 1), datetime.date(2016, 1, 1),datetime.date(2016, 4, 1), datetime.date(2016, 7, 1),datetime.date(2016, 10, 1), datetime.date(2017, 1, 1)], dtype=object)


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


1.207 1.29 2.245 ... 2.431 -2.456


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 1.20710158e+00,  1.28989005e+00,  2.24493885e+00],[ 9.75153387e-01,  6.62226081e-01,  5.61803961e+00],[ 9.91330624e-01,  8.00233483e-01,  4.96064520e+00],...,[-1.21346736e+00, -5.00150204e-01, -1.02502604e+01],[ 1.21014774e-01, -1.93878055e-01,  1.85976124e+00],[-6.07735932e-01, -1.77062094e-01, -1.05129838e+00]],[[ 8.87746811e-01,  1.03112698e+00,  1.46455312e+00],[ 6.41428649e-01,  6.17565870e-01,  5.97192955e+00],[-8.52721035e-01, -5.48458099e-03, -7.94079876e+00],...,[ 3.59264374e-01,  1.44553018e+00, -6.14232016e+00],[ 1.89767075e+00, -1.19157553e-01,  1.03197594e+01],[ 1.36333454e+00,  6.34127617e-01,  2.66870546e+00]],[[ 5.07922113e-01,  1.05372906e+00, -2.23390150e+00],[ 1.09555864e+00,  4.13703948e-01,  3.90078878e+00],[-6.63182557e-01, -1.69243753e-01, -3.40895557e+00],...,...[ 1.02872396e+00,  3.47514182e-01,  2.62850451e+00],[ 9.25298452e-01,  9.04427409e-01, -1.78939080e+00],[ 7.95014977e-01,  7.31956363e-01, -4.13092089e+00]],[[-7.15250552e-01, -1.35348380e+00, -3.61786723e+00],[-9.24997032e-01, -5.02070189e-01, -6.39409733e+00],[-1.39488959e+00,  8.48204792e-01, -1.61711464e+01],...,[ 7.48523533e-01,  1.05257857e+00,  7.41460383e-01],[-4.88512516e-02,  6.20472968e-01, -3.62961102e+00],[ 1.87596059e+00,  7.99615622e-01,  5.25599194e+00]],[[ 4.80662555e-01, -5.22970557e-01,  4.68374157e+00],[-1.37228155e+00, -7.65509844e-01, -5.56125307e+00],[-1.24161446e+00, -2.92782992e-01, -4.08072042e+00],...,[ 1.44711804e+00,  1.81191850e+00,  5.07093239e+00],[ 1.28685176e+00,  1.37934482e+00,  5.17915344e+00],[ 1.92112947e+00,  2.43080807e+00, -2.45609212e+00]]]],shape=(4, 1000, 30, 3), dtype=float32)


Attributes: (5)


created_at :  
2026-09-29T18:37:58.159090+00:00

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


object


2009-10-01 ... 2017-01-01


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([datetime.date(2009, 10, 1), datetime.date(2010, 1, 1),datetime.date(2010, 4, 1), datetime.date(2010, 7, 1),datetime.date(2010, 10, 1), datetime.date(2011, 1, 1),datetime.date(2011, 4, 1), datetime.date(2011, 7, 1),datetime.date(2011, 10, 1), datetime.date(2012, 1, 1),datetime.date(2012, 4, 1), datetime.date(2012, 7, 1),datetime.date(2012, 10, 1), datetime.date(2013, 1, 1),datetime.date(2013, 4, 1), datetime.date(2013, 7, 1),datetime.date(2013, 10, 1), datetime.date(2014, 1, 1),datetime.date(2014, 4, 1), datetime.date(2014, 7, 1),datetime.date(2014, 10, 1), datetime.date(2015, 1, 1),datetime.date(2015, 4, 1), datetime.date(2015, 7, 1),datetime.date(2015, 10, 1), datetime.date(2016, 1, 1),datetime.date(2016, 4, 1), datetime.date(2016, 7, 1),datetime.date(2016, 10, 1), datetime.date(2017, 1, 1)], dtype=object)


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
2026-09-29T18:37:58.159731+00:00

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
summary
```


|  | mean | sd | hdi94_lb | hdi94_ub | ess_bulk | ess_tail | r_hat | mcse_mean | mcse_sd |
|----|----|----|----|----|----|----|----|----|----|
| intercept\[realgdp\] | 0.242 | 0.101 | 0.053 | 0.44 | 3012 | 2959 | 1.00 | 0.0018 | 0.0013 |
| intercept\[realcons\] | 0.553 | 0.098 | 0.37 | 0.74 | 3611 | 3286 | 1.00 | 0.0016 | 0.0012 |
| intercept\[realinv\] | -1.74 | 0.49 | -2.6 | -0.81 | 3302 | 3030 | 1.00 | 0.0085 | 0.0059 |
| sigma\[realgdp\] | 0.747 | 0.036 | 0.68 | 0.82 | 2752 | 2705 | 1.00 | 0.00069 | 0.0005 |
| sigma\[realcons\] | 0.659 | 0.0342 | 0.6 | 0.73 | 3475 | 3028 | 1.00 | 0.00059 | 0.00041 |
| sigma\[realinv\] | 3.877 | 0.185 | 3.5 | 4.2 | 3409 | 2678 | 1.00 | 0.0032 | 0.0024 |
| phi\[1, realgdp, realgdp\] | -0.066 | 0.138 | -0.32 | 0.2 | 1957 | 2146 | 1.00 | 0.0031 | 0.0022 |
| phi\[1, realgdp, realcons\] | 0.447 | 0.112 | 0.24 | 0.66 | 1990 | 2557 | 1.00 | 0.0025 | 0.0017 |
| phi\[1, realgdp, realinv\] | 0.011 | 0.0221 | -0.03 | 0.052 | 1974 | 2592 | 1.00 | 0.0005 | 0.00035 |
| phi\[1, realcons, realgdp\] | -0.06 | 0.144 | -0.33 | 0.21 | 2021 | 2309 | 1.00 | 0.0032 | 0.0023 |
| phi\[1, realcons, realcons\] | 0.231 | 0.113 | 0.023 | 0.44 | 2287 | 2579 | 1.00 | 0.0024 | 0.0017 |
| phi\[1, realcons, realinv\] | 0.0212 | 0.0223 | -0.02 | 0.063 | 2190 | 2547 | 1.00 | 0.00048 | 0.00033 |
| phi\[1, realinv, realgdp\] | -0.53 | 0.59 | -1.6 | 0.58 | 2560 | 2717 | 1.00 | 0.012 | 0.0081 |
| phi\[1, realinv, realcons\] | 2.84 | 0.5 | 1.9 | 3.7 | 3127 | 3082 | 1.00 | 0.0089 | 0.0064 |
| phi\[1, realinv, realinv\] | 0.077 | 0.101 | -0.11 | 0.26 | 2656 | 2618 | 1.00 | 0.002 | 0.0014 |
| phi\[2, realgdp, realgdp\] | -0.008 | 0.143 | -0.27 | 0.25 | 1657 | 2354 | 1.00 | 0.0035 | 0.0024 |
| phi\[2, realgdp, realcons\] | 0.264 | 0.123 | 0.033 | 0.49 | 2147 | 2611 | 1.00 | 0.0026 | 0.0019 |
| phi\[2, realgdp, realinv\] | -0.001 | 0.0221 | -0.041 | 0.041 | 1801 | 2386 | 1.00 | 0.00052 | 0.00036 |
| phi\[2, realcons, realgdp\] | -0.115 | 0.147 | -0.39 | 0.16 | 1981 | 1863 | 1.00 | 0.0033 | 0.0023 |
| phi\[2, realcons, realcons\] | 0.223 | 0.123 | -0.011 | 0.46 | 2312 | 2561 | 1.00 | 0.0026 | 0.0018 |
| phi\[2, realcons, realinv\] | 0.0233 | 0.022 | -0.018 | 0.065 | 2027 | 2640 | 1.00 | 0.00049 | 0.00033 |
| phi\[2, realinv, realgdp\] | 0.23 | 0.62 | -0.9 | 1.4 | 2484 | 2449 | 1.00 | 0.012 | 0.0086 |
| phi\[2, realinv, realcons\] | 0.64 | 0.54 | -0.36 | 1.6 | 3022 | 2852 | 1.00 | 0.0098 | 0.0066 |
| phi\[2, realinv, realinv\] | -0.073 | 0.102 | -0.26 | 0.12 | 2819 | 2888 | 1.00 | 0.0019 | 0.0013 |


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


    stable draws: 1.000, median spectral radius: 0.599, max: 0.833


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
dates_num = mdates.date2num(dates)
future_dates_num = mdates.date2num(future_dates)


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
    pc.get_target("t", {"series": names[0]}).legend(
        handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9
    )
    fig = pc.viz["figure"].item()
    fig.supxlabel("date")
    fig.supylabel("growth rate (percent)")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)


train_pp = stack_draws("posterior_predictive", tree)
crps_train = eval_crps(train_pp, data)
print(f"in-sample CRPS: {float(crps_train):.4f}")
```


    in-sample CRPS: 0.9670


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
width_94_h = width_94[[h - 1 for h in horizons]]

pl.DataFrame({"h": horizons} | dict(zip(names, width_94_h.T, strict=True))).with_columns(
    pl.col(names).round(3)
)
```


shape: (5, 4)

| h   | realgdp | realcons | realinv |
|-----|---------|----------|---------|
| i64 | f64     | f64      | f64     |
| 1   | 2.855   | 2.493    | 14.853  |
| 5   | 3.234   | 2.787    | 16.574  |
| 10  | 3.266   | 2.69     | 17.109  |
| 20  | 3.173   | 2.789    | 15.907  |
| 30  | 3.122   | 2.749    | 16.015  |


``` python
phi_mean = np.asarray(posterior["phi"]).mean(axis=0)
c_mean = np.asarray(posterior["intercept"]).mean(axis=0)
unconditional_mean = np.linalg.solve(np.eye(k) - phi_mean.sum(axis=0), c_mean)

pl.DataFrame(
    {
        "series": names,
        "unconditional mean": unconditional_mean,
        "sample mean": np.asarray(data).mean(axis=0),
        "mean forecast at h=30": forecast_draws.mean(axis=0)[-1],
    }
).with_columns(pl.exclude("series").round(3))
```


shape: (3, 4)

| series     | unconditional mean | sample mean | mean forecast at h=30 |
|------------|--------------------|-------------|-----------------------|
| str        | f64                | f32         | f32                   |
| "realgdp"  | 0.788              | 0.772       | 0.785                 |
| "realcons" | 0.836              | 0.832       | 0.82                  |
| "realinv"  | 0.943              | 0.818       | 0.99                  |


# Impulse response functions

A forecast tells you where the system goes on average. An impulse response tells you how a shock to one series propagates to all series over time. For a stable VAR the moving-average (Wold) representation

 y_t = \mu + \sum\_{h=0}^{\infty} \Psi_h \\ \varepsilon\_{t-h} 

exists, and the coefficient matrices follow the recursion

 \Psi_0 = I, \qquad \Psi_h = \sum\_{j=1}^{\min(h, p)} \Phi_j \\ \Psi\_{h-j} \quad (h \geq 1). 

The entry \Psi_h\[i, j\] is the response of series i, h quarters after a unit shock to the reduced-form residual \varepsilon\_{t, j}, with the other residuals held at zero. `impulse_response(phi, horizon)` runs this recursion for all posterior draws at once (the draws pass through the leading batch axis; no `vmap` is needed) and returns an array of shape `(draws, horizon + 1, series, series)`, indexed as `[draw, h, response, shock]`.

The recursion exists for any coefficients, but the representation and the decay \Psi_h \to 0 need stability, which we checked above. If some draws were unstable we would mask them here; the mask below is the identity when all draws are stable.

Writing out the recursion for our VAR(2) (p = 2) makes it concrete. The sum only ever has one or two terms, because \min(h, p) \leq 2:

 \Psi_0 = I, \qquad \Psi_1 = \Phi_1 \\ \Psi_0 = \Phi_1, \qquad \Psi_2 = \Phi_1 \\ \Psi_1 + \Phi_2 \\ \Psi_0 = \Phi_1^2 + \Phi_2, \qquad \Psi_3 = \Phi_1 \\ \Psi_2 + \Phi_2 \\ \Psi_1. 

Each \Psi_h only ever combines the two lag matrices \Phi_1, \Phi_2 (sampled once per posterior draw) with the previously computed \Psi\_{h-1}, \Psi\_{h-2}: this is exactly what [impulse_response](../../reference/var.impulse_response.md#numpyro_forecast.var.impulse_response) scans over, and it is also what [companion_matrix](../../reference/var.companion_matrix.md#numpyro_forecast.var.companion_matrix) block-multiplies in one shot when we only need the stability check rather than every intermediate \Psi_h.


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

     [[-0.066  0.447  0.011]
      [-0.06   0.231  0.021]
      [-0.525  2.84   0.077]]

     [[-0.031  0.369  0.009]
      [-0.137  0.315  0.029]
      [ 0.049  1.281 -0.009]]]


\Psi_0 is the identity, as expected: at h = 0 every series responds only to its own shock, one for one. \Psi_1 and \Psi_2 show how the shock starts to spread to the other two series through the estimated \Phi_1, \Phi_2 coefficients; the full grid below plots this spread out to h = 10 with posterior uncertainty.


``` python
def plot_irf_grid(
    irf: Float[Array, " sample steps series series"],
    title: str,
    ylabel: str,
    overlay: Float[Array, " sample steps series series"] | None = None,
    overlay_label: str = "",
    legend_loc: str = "upper right",
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
    legend_loc
        Legend location, forwarded to `matplotlib.axes.Axes.legend`.
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
        handles=handles, loc=legend_loc, fontsize=8
    )
    fig = pc.viz["figure"].item()
    fig.supxlabel("quarters after the shock")
    fig.supylabel(ylabel)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)


plot_irf_grid(
    irf_draws,
    title="Impulse responses to a unit reduced-form shock",
    ylabel="response (percentage points)",
)
```


<figure class="figure">
<p><img src="var_files/figure-html/_src-var-cell-19-output-1.png" class="img-fluid figure-img" /></p>
</figure>


The `realinv response to realcons shock` panel dominates the grid: its y-axis runs past 3, while every other off-diagonal panel stays within about \pm 0.6. The printed h = 0, 1, 2 snapshot above puts a number on it: a one-unit consumption-growth shock moves investment growth by 2.84 percentage points at h = 1 and 1.28 at h = 2, six to ten times larger than any other off-diagonal entry at the same horizons, consistent with investment's well-known sensitivity to demand shocks. Every panel decays toward zero by h \approx 8-10, as stability requires; the own-shock (diagonal) panels start at exactly 1 by construction and decay the fastest, some dipping briefly negative (e.g. `realgdp response to realgdp shock` around h = 1-2) before settling.


## Orthogonalized and cumulative responses

A unit shock to one reduced-form residual with the others held at zero is not an experiment we can observe when \Sigma is not diagonal: the residuals move together. The standard fix is to rewrite the shocks as \varepsilon_t = L \\ u_t with u_t \sim \text{MultivariateNormal}(0, I) and L the Cholesky factor of \Sigma, and to report the responses to the *orthogonalized* shocks u_t:

 \Theta_h = \Psi_h \\ L. 

A unit shock to u\_{t, j} is a one-standard-deviation shock. Because L is lower triangular, the first series in the ordering responds only to its own shock in the impact quarter, the second series to the first two shocks, and so on. This is the recursive identification, and the ordering `realgdp, realcons, realinv` is part of the model: a different ordering gives different orthogonalized responses. [impulse_response](../../reference/var.impulse_response.md#numpyro_forecast.var.impulse_response) takes the factor through `scale_tril`, here built from the posterior draws of \sigma and L\_\Omega.

Our series are growth rates in percent, g_t = 100 \\ \Delta \log Y_t, so the running sum \sum\_{s=0}^{h} \Theta_s is the response of the *log level* in percent, approximately the percent change of the level. We request it with `cumulative=True` and extend the horizon to 20 quarters. For a stable VAR the cumulative response converges to the long-run effect (I - \sum_l \Phi_l)^{-1} L.


``` python
# Reassemble the Cholesky factor L = diag(sigma) @ L_Omega per draw, the same construction
# the model uses for the likelihood's scale_tril, so the orthogonalized shocks are one
# posterior-consistent standard deviation of the fitted shock covariance.
sigma_draws = jnp.asarray(posterior["sigma"])[stable]
l_omega_draws = jnp.asarray(posterior["l_omega"])[stable]
scale_tril_draws = sigma_draws[..., :, None] * l_omega_draws  # (draws, 3, 3)

# cumulative=True sums Theta_h = Psi_h @ L over h, turning the growth-rate response into a
# level response; horizon 20 is long enough for the stable draws to approach that long-run sum.
irf_level_draws = impulse_response(
    phi_draws[stable], 20, scale_tril=scale_tril_draws, cumulative=True
)

print(f"irf_level_draws: {irf_level_draws.shape}")
print("posterior mean cumulative response at h = 20 (rows: response, columns: shock):")
print(np.round(np.asarray(irf_level_draws.mean(axis=0)[-1]), 3))
```


    irf_level_draws: (4000, 21, 3, 3)
    posterior mean cumulative response at h = 20 (rows: response, columns: shock):
    [[1.261 0.613 0.143]
     [0.761 0.898 0.177]
     [5.186 1.384 2.687]]


``` python
plot_irf_grid(
    irf_level_draws,
    title="Cumulative responses to a one standard deviation orthogonalized shock",
    ylabel="level response (percent)",
    legend_loc="lower right",
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
sample_sd = y_pct.select(names).std().to_numpy()[0]
scale_ratio = sample_sd[:, None] / sample_sd[None, :]

pl.DataFrame({"equation": names} | dict(zip(names, scale_ratio.T, strict=True))).with_columns(
    pl.col(names).round(2)
)
```


shape: (3, 4)

| equation   | realgdp | realcons | realinv |
|------------|---------|----------|---------|
| str        | f64     | f64      | f64     |
| "realgdp"  | 1.0     | 1.27     | 0.19    |
| "realcons" | 0.79    | 1.0      | 0.15    |
| "realinv"  | 5.33    | 6.75     | 1.0     |


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

pl.DataFrame(
    {
        "lagged series": names,
        "unscaled prior sd": np.asarray(scale_unscaled[0, 2]),
        "scaled prior sd": np.asarray(scale_mn[0, 2]),
        "weak prior posterior mean": phi_weak_mean[0, 2],
        "weak prior posterior sd": phi_weak_sd[0, 2],
    }
).with_columns(pl.exclude("lagged series").round(3))
```


shape: (3, 5)

| lagged series | unscaled prior sd | scaled prior sd | weak prior posterior mean | weak prior posterior sd |
|----|----|----|----|----|
| str | f32 | f32 | f32 | f32 |
| "realgdp" | 0.25 | 1.331 | -0.525 | 0.594 |
| "realcons" | 0.25 | 1.687 | 2.84 | 0.498 |
| "realinv" | 0.5 | 0.5 | 0.077 | 0.101 |


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
summary_mn
```


    divergences: 0


|  | mean | sd | hdi94_lb | hdi94_ub | ess_bulk | ess_tail | r_hat | mcse_mean | mcse_sd |
|----|----|----|----|----|----|----|----|----|----|
| intercept\[realgdp\] | 0.251 | 0.092 | 0.078 | 0.42 | 2541 | 2992 | 1.00 | 0.0018 | 0.0013 |
| intercept\[realcons\] | 0.549 | 0.088 | 0.38 | 0.71 | 3380 | 3034 | 1.00 | 0.0015 | 0.0011 |
| intercept\[realinv\] | -1.69 | 0.46 | -2.5 | -0.82 | 3326 | 3247 | 1.00 | 0.0079 | 0.0056 |
| sigma\[realgdp\] | 0.745 | 0.0353 | 0.68 | 0.82 | 3293 | 2844 | 1.00 | 0.00062 | 0.00044 |
| sigma\[realcons\] | 0.657 | 0.0329 | 0.6 | 0.72 | 4296 | 3348 | 1.00 | 0.0005 | 0.00039 |
| sigma\[realinv\] | 3.862 | 0.184 | 3.5 | 4.2 | 3214 | 2933 | 1.00 | 0.0033 | 0.0023 |
| phi\[1, realgdp, realgdp\] | -0.057 | 0.119 | -0.28 | 0.17 | 1800 | 2413 | 1.00 | 0.0028 | 0.0021 |
| phi\[1, realgdp, realcons\] | 0.468 | 0.102 | 0.28 | 0.66 | 2310 | 2557 | 1.00 | 0.0021 | 0.0015 |
| phi\[1, realgdp, realinv\] | 0.0118 | 0.0184 | -0.024 | 0.045 | 1875 | 2325 | 1.00 | 0.00043 | 0.00031 |
| phi\[1, realcons, realgdp\] | 0.002 | 0.104 | -0.2 | 0.19 | 2691 | 2656 | 1.00 | 0.002 | 0.0015 |
| phi\[1, realcons, realcons\] | 0.194 | 0.094 | 0.015 | 0.37 | 3138 | 3054 | 1.00 | 0.0017 | 0.0012 |
| phi\[1, realcons, realinv\] | 0.0156 | 0.0163 | -0.016 | 0.047 | 2745 | 2832 | 1.00 | 0.00031 | 0.00023 |
| phi\[1, realinv, realgdp\] | -0.87 | 0.66 | -2.1 | 0.38 | 1963 | 1958 | 1.00 | 0.015 | 0.011 |
| phi\[1, realinv, realcons\] | 3.28 | 0.55 | 2.2 | 4.3 | 2478 | 2841 | 1.00 | 0.011 | 0.0081 |
| phi\[1, realinv, realinv\] | 0.119 | 0.102 | -0.081 | 0.31 | 1903 | 2238 | 1.00 | 0.0024 | 0.0017 |
| phi\[2, realgdp, realgdp\] | 0.059 | 0.085 | -0.1 | 0.22 | 2775 | 2637 | 1.00 | 0.0016 | 0.0011 |
| phi\[2, realgdp, realcons\] | 0.169 | 0.082 | 0.015 | 0.32 | 2511 | 2932 | 1.00 | 0.0016 | 0.0011 |
| phi\[2, realgdp, realinv\] | -0.0079 | 0.0135 | -0.033 | 0.018 | 2821 | 2960 | 1.00 | 0.00025 | 0.00018 |
| phi\[2, realcons, realgdp\] | -0.005 | 0.07 | -0.13 | 0.12 | 3821 | 3314 | 1.00 | 0.0011 | 0.00083 |
| phi\[2, realcons, realcons\] | 0.125 | 0.085 | -0.033 | 0.29 | 3695 | 3283 | 1.00 | 0.0014 | 0.00099 |
| phi\[2, realcons, realinv\] | 0.0085 | 0.0114 | -0.014 | 0.03 | 4157 | 3019 | 1.00 | 0.00018 | 0.00012 |
| phi\[2, realinv, realgdp\] | 0.19 | 0.44 | -0.62 | 1 | 2751 | 2824 | 1.00 | 0.0084 | 0.0059 |
| phi\[2, realinv, realcons\] | 0.44 | 0.43 | -0.34 | 1.2 | 3144 | 2410 | 1.00 | 0.0077 | 0.0054 |
| phi\[2, realinv, realinv\] | -0.061 | 0.078 | -0.21 | 0.09 | 2953 | 3000 | 1.00 | 0.0014 | 0.001 |


## Shrinkage of the coefficients

The table compares the posterior standard deviation of every coefficient under the two priors. The ratio column is below one where the Minnesota prior tightened the posterior. The effect is largest on the second lag, where the harmonic decay halves the prior standard deviation, and on the GDP and consumption equations. The investment equation at lag one is unchanged within Monte Carlo error: after the scale correction its prior is wide relative to what the data say, so the data decide.


``` python
lag_labels, equation_labels, lagged_labels = zip(
    *itertools.product(range(1, p + 1), names, names), strict=True
)

phi_sd = pl.DataFrame(
    {
        "lag": lag_labels,
        "equation": equation_labels,
        "lagged series": lagged_labels,
        "weak prior": phi_weak_sd.reshape(-1),
        "minnesota prior": np.asarray(posterior_mn["phi"]).std(axis=0).reshape(-1),
    }
).with_columns((pl.col("minnesota prior") / pl.col("weak prior")).alias("ratio"))

own_mean = phi_sd.filter(pl.col("equation") == pl.col("lagged series"))["ratio"].mean()
cross_mean = phi_sd.filter(pl.col("equation") != pl.col("lagged series"))["ratio"].mean()
print(f"mean ratio on own lags: {own_mean:.2f}, on cross lags: {cross_mean:.2f}")
phi_sd.with_columns(pl.exclude("lag", "equation", "lagged series").round(3))
```


    mean ratio on own lags: 0.79, on cross lags: 0.77


shape: (18, 6)

| lag | equation   | lagged series | weak prior | minnesota prior | ratio |
|-----|------------|---------------|------------|-----------------|-------|
| i64 | str        | str           | f32        | f32             | f32   |
| 1   | "realgdp"  | "realgdp"     | 0.138      | 0.119           | 0.866 |
| 1   | "realgdp"  | "realcons"    | 0.112      | 0.102           | 0.91  |
| 1   | "realgdp"  | "realinv"     | 0.022      | 0.018           | 0.832 |
| 1   | "realcons" | "realgdp"     | 0.144      | 0.104           | 0.721 |
| 1   | "realcons" | "realcons"    | 0.113      | 0.094           | 0.831 |
| …   | …          | …             | …          | …               | …     |
| 2   | "realcons" | "realcons"    | 0.123      | 0.085           | 0.69  |
| 2   | "realcons" | "realinv"     | 0.022      | 0.011           | 0.518 |
| 2   | "realinv"  | "realgdp"     | 0.616      | 0.438           | 0.712 |
| 2   | "realinv"  | "realcons"    | 0.538      | 0.431           | 0.802 |
| 2   | "realinv"  | "realinv"     | 0.102      | 0.078           | 0.765 |


## Forecast bands

Tighter coefficients mean less parameter uncertainty in the forecast. The table reports the mean width of the 94\\ HDI over the 30 forecast quarters, per series and per prior. The change is small: a few percent for GDP and consumption and none for investment. With 200 quarters for 18 coefficients, the forecast uncertainty comes from the shock covariance, not from the coefficients, and a prior of this tightness cannot move it much.


``` python
forecast_draws_mn = stack_draws("predictions", tree_mn)

pl.DataFrame(
    {
        "series": names,
        "weak prior": width_94.mean(axis=0),
        "minnesota prior": hdi_width(forecast_draws_mn, 0.94).mean(axis=0),
    }
).with_columns(pl.exclude("series").round(3))
```


shape: (3, 3)

| series     | weak prior | minnesota prior |
|------------|------------|-----------------|
| str        | f64        | f64             |
| "realgdp"  | 3.215      | 3.166           |
| "realcons" | 2.705      | 2.635           |
| "realinv"  | 16.509     | 16.579          |


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

[Source: Vector Autoregression (VAR) with `numpyro_forecast`](_src/var-preview.html#50138624)
