# Vector Autoregression (VAR)


Vector Autoregression (VAR) with `numpyro_forecast`

This notebook ports the blog post [**Bayesian VAR in NumPyro**](https://juanitorduz.github.io/var_numpyro/) to the [`numpyro_forecast`](https://github.com/juanitorduz/numpyro_forecast) package. A vector autoregression (VAR) models several time series jointly: each series is regressed on the past values of all series, and the shocks are correlated across series. We fit a VAR with two lags to the quarterly growth rates of US real GDP, consumption and investment, sample the posterior with NUTS, forecast 30 quarters ahead, and compute impulse response functions (IRFs), the standard tool to read a VAR.

The package provides the VAR pieces as reusable components. You do not write the lag recursion, the forecast loop or the IRF recursion yourself:

- [`var_step`](https://juanitorduz.github.io/numpyro_forecast/reference/var.var_step.html) turns sampled coefficients into a step for the [`ssoe`](https://juanitorduz.github.io/numpyro_forecast/reference/models.ssoe.html) building block. The block runs the in-sample recursion and the generative forecast.
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

The noise distribution is a `MultivariateNormal` over the series axis, so the future shocks are correlated across series exactly as the in-sample residuals are. `var_step(phi, intercept)` builds the step function from the sampled coefficients: the carry is the lag window with shape `(lags, series)` in natural time order (most recent row last), the mean is c + \sum_l \Phi_l y\_{t-l}, and the carry update drops the oldest row and appends the new one.


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
│           intercept      (chain, draw, series) float32 48kB 0.1826 0.4247 ... -1.274
│           l_omega        (chain, draw, l_omega_dim_0, l_omega_dim_1) float32 144kB ...
│           mu_t           (chain, draw, time, obs_dim) float32 10MB 1.075 ... -0.5934
│           phi            (chain, draw, lag, equation, lagged_series) float32 288kB ...
│           sigma          (chain, draw, series) float32 48kB 0.7399 0.6636 ... 3.953
│       Attributes:
│           created_at:                 2026-09-04T08:38:51.164265+00:00
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
│           obs      (chain, draw, time, obs_dim) float32 10MB 2.397 1.841 ... -2.806
│       Attributes:
│           created_at:                 2026-09-04T08:38:51.443220+00:00
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
│           created_at:                 2026-09-04T08:38:51.443862+00:00
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
│           created_at:                 2026-09-04T08:38:51.444395+00:00
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
│           obs      (chain, draw, time, obs_dim) float32 1MB 1.231 1.315 ... -2.136
│       Attributes:
│           created_at:                 2026-09-04T08:38:51.756720+00:00
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
            created_at:                 2026-09-04T08:38:51.757232+00:00
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


0.1826 0.4247 ... 0.5024 -1.274


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 0.18262418,  0.42467725, -1.3757132 ],[ 0.00995991,  0.47337285, -2.8731654 ],[ 0.36494485,  0.5364451 , -0.7612963 ],...,[ 0.26283634,  0.6368028 , -1.2074226 ],[ 0.17468914,  0.49328035, -2.1077285 ],[ 0.13499013,  0.4935203 , -1.8131164 ]],[[ 0.387278  ,  0.63216156, -1.1776836 ],[ 0.34054992,  0.5441619 , -0.9577487 ],[ 0.15526173,  0.5583605 , -2.132862  ],...,[ 0.14866434,  0.45134223, -2.133964  ],[ 0.25439677,  0.5560107 , -1.618279  ],[ 0.34654117,  0.6562711 , -1.7046548 ]],[[ 0.15503307,  0.561739  , -1.9879895 ],[ 0.31279776,  0.55984163, -1.4918717 ],[ 0.38439605,  0.68564695, -1.4616492 ],...,[-0.00988665,  0.49295676, -3.6588619 ],[ 0.03792755,  0.48151165, -2.41989   ],[ 0.42723405,  0.6597408 , -1.1022807 ]],[[ 0.20681006,  0.491232  , -1.6401768 ],[ 0.25413808,  0.5938929 , -2.0477955 ],[ 0.12585647,  0.5517798 , -2.1647048 ],...,[ 0.0847002 ,  0.36319292, -2.5300589 ],[ 0.4411324 ,  0.6868283 , -0.98628974],[ 0.27970895,  0.5024341 , -1.2744097 ]]],shape=(4, 1000, 3), dtype=float32)


l_omega


(chain, draw, l_omega_dim_0, l_omega_dim_1)


float32


1.0 0.0 0.0 ... -0.4702 0.5248


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 1.        ,  0.        ,  0.        ],[ 0.6138584 ,  0.7894162 ,  0.        ],[ 0.6829468 , -0.41667795,  0.59996927]],[[ 1.        ,  0.        ,  0.        ],[ 0.5683172 ,  0.8228096 ,  0.        ],[ 0.74024135, -0.38309976,  0.552519  ]],[[ 1.        ,  0.        ,  0.        ],[ 0.6064197 ,  0.79514474,  0.        ],[ 0.75550437, -0.43479455,  0.49006823]],...,[[ 1.        ,  0.        ,  0.        ],[ 0.5604336 ,  0.8281993 ,  0.        ],[ 0.7599967 , -0.37500888,  0.53082323]],[[ 1.        ,  0.        ,  0.        ],[ 0.6055008 ,  0.7958447 ,  0.        ],...[ 0.7224597 , -0.3626164 ,  0.58869463]],[[ 1.        ,  0.        ,  0.        ],[ 0.5924085 ,  0.8056378 ,  0.        ],[ 0.73225075, -0.39898995,  0.5519202 ]],...,[[ 1.        ,  0.        ,  0.        ],[ 0.5700733 ,  0.8215938 ,  0.        ],[ 0.680265  , -0.45527256,  0.574427  ]],[[ 1.        ,  0.        ,  0.        ],[ 0.5168506 ,  0.8560756 ,  0.        ],[ 0.7579728 , -0.43410873,  0.48685405]],[[ 1.        ,  0.        ,  0.        ],[ 0.5734395 ,  0.8192479 ,  0.        ],[ 0.7096048 , -0.47015053,  0.5248043 ]]]],shape=(4, 1000, 3, 3), dtype=float32)


mu_t


(chain, draw, time, obs_dim)


float32


1.075 0.9959 ... 0.2422 -0.5934


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 1.0751643 ,  0.995881  ,  1.2796315 ],[ 0.56156063,  0.70370865,  0.22177398],[ 0.45057875,  0.72062945,  0.13556886],...,[-0.24046992,  0.0612393 , -3.1102767 ],[ 0.59253764,  0.4359541 ,  0.84510577],[ 0.7644043 ,  0.69747823,  1.5910126 ]],[[ 0.81934077,  0.72139037,  0.7007842 ],[ 0.48236033,  0.68252045, -1.0817552 ],[ 0.6076685 ,  1.0580446 ,  0.3264134 ],...,[-0.7308303 , -0.02314478, -6.3484583 ],[-0.4686542 , -0.16294715, -4.343491  ],[-0.09091302,  0.19854322, -3.5265293 ]],[[ 1.1488541 ,  0.9242704 ,  2.2713404 ],[ 0.7181258 ,  0.815659  ,  0.5351925 ],[ 0.34526864,  0.7202967 , -0.7897938 ],...,...[-0.6715397 , -0.1727131 , -6.048209  ],[-0.24977353,  0.03521842, -2.9229085 ],[-0.19998565,  0.11963987, -2.4931464 ]],[[ 1.0331867 ,  1.0314984 ,  1.57637   ],[ 0.6170275 ,  0.8111854 , -0.21177542],[ 0.8296172 ,  0.95912766,  1.4560518 ],...,[-0.06510004,  0.28721485, -3.692123  ],[ 0.27333224,  0.16342485, -1.774644  ],[ 0.06369868, -0.07557565, -2.9052424 ]],[[ 1.0054642 ,  0.9810779 ,  1.6177324 ],[ 0.63194335,  0.5928664 ,  0.3050884 ],[ 0.82512116,  0.8242423 ,  1.5883914 ],...,[-0.27559236,  0.16505966, -3.9004455 ],[-0.10555318,  0.15015918, -2.424652  ],[ 0.42422345,  0.24220109, -0.59341776]]]],shape=(4, 1000, 200, 3), dtype=float32)


phi


(chain, draw, lag, equation, lagged_series)


float32


0.1481 0.3919 ... -0.4188 -0.2306


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[[ 1.48138195e-01,  3.91892105e-01, -3.62785012e-02],[ 1.26874357e-01,  2.10654721e-01, -1.52236158e-02],[ 4.54887390e-01,  2.24483585e+00, -1.11218452e-01]],[[ 9.77616161e-02,  1.73690140e-01, -3.33899260e-02],[ 1.17880382e-01,  1.01819150e-01, -2.39259526e-02],[ 1.79820955e-01,  3.37647617e-01, -1.73132822e-01]]],[[[ 1.36614824e-02,  4.01900053e-01,  1.50700705e-02],[ 4.10341918e-02,  2.04905167e-01,  2.75664870e-02],[-7.88825825e-02,  2.64867234e+00,  7.33065382e-02]],[[-3.73294135e-03,  3.44688177e-01, -1.90533767e-03],[ 8.74409825e-02,  4.42863442e-02, -5.84128173e-03],[-3.55236918e-01,  1.34304726e+00,  2.18720902e-02]]],[[[-3.32524538e-01,  5.54724872e-01,  2.15893853e-02],[-1.30414441e-01,  2.79323071e-01,  1.92454420e-02],...[ 1.90733954e-01,  1.05727708e+00, -5.76436855e-02]]],[[[-9.54011306e-02,  4.19936031e-01,  1.01843560e-02],[-4.06555235e-01,  4.81446862e-01,  6.07424416e-02],[-3.60088348e-01,  2.33947492e+00,  5.48449531e-02]],[[-3.15289855e-01,  4.11998808e-01,  4.67013195e-02],[-1.68386012e-01,  2.37609491e-01,  3.62860560e-02],[-1.85074055e+00,  1.99934947e+00,  2.55002826e-01]]],[[[ 1.99789032e-01,  1.93346709e-01, -5.50857233e-03],[ 8.34673047e-02,  8.26116949e-02, -4.42230783e-04],[ 2.26533547e-01,  2.18620157e+00,  5.98717704e-02]],[[ 3.20655614e-01,  2.37151906e-02, -4.07652408e-02],[ 6.42801300e-02,  1.14650473e-01,  7.98210874e-03],[ 1.43132532e+00, -4.18848872e-01, -2.30553299e-01]]]]],shape=(4, 1000, 2, 3, 3), dtype=float32)


sigma


(chain, draw, series)


float32


0.7399 0.6636 3.86 ... 0.6743 3.953


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.7399372 , 0.6636431 , 3.8603978 ],[0.77365357, 0.6616881 , 4.1457577 ],[0.7619016 , 0.6690334 , 3.7084103 ],...,[0.7923949 , 0.65217704, 4.042516  ],[0.69106686, 0.6599183 , 3.772103  ],[0.7315267 , 0.68191546, 3.8117237 ]],[[0.7694906 , 0.6708486 , 4.0801773 ],[0.72124636, 0.63928235, 4.02017   ],[0.77961993, 0.66288424, 3.9700599 ],...,[0.6961043 , 0.60509014, 3.7456017 ],[0.73252666, 0.6553079 , 3.8596523 ],[0.7212443 , 0.5878971 , 3.9361038 ]],[[0.75638425, 0.6800382 , 3.7402072 ],[0.7612609 , 0.64987725, 3.845205  ],[0.7033571 , 0.6093092 , 3.6634045 ],...,[0.8033195 , 0.668959  , 4.10603   ],[0.75730485, 0.6533364 , 3.9772904 ],[0.75211847, 0.62383014, 3.8470867 ]],[[0.7354738 , 0.6551204 , 3.7963169 ],[0.77205557, 0.6512871 , 3.9435248 ],[0.70560795, 0.6392633 , 3.6499646 ],...,[0.73183626, 0.6721806 , 4.0497727 ],[0.7197319 , 0.62728393, 3.8620677 ],[0.7030298 , 0.67425275, 3.95273   ]]],shape=(4, 1000, 3), dtype=float32)


Attributes: (5)


created_at :  
2026-09-04T08:38:51.164265+00:00

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


e(1974, 10, 1), datetime.date(1975, 1, 1),datetime.date(1975, 4, 1), datetime.date(1975, 7, 1),datetime.date(1975, 10, 1), datetime.date(1976, 1, 1),datetime.date(1976, 4, 1), datetime.date(1976, 7, 1),datetime.date(1976, 10, 1), datetime.date(1977, 1, 1),datetime.date(1977, 4, 1), datetime.date(1977, 7, 1),datetime.date(1977, 10, 1), datetime.date(1978, 1, 1),datetime.date(1978, 4, 1), datetime.date(1978, 7, 1),datetime.date(1978, 10, 1), datetime.date(1979, 1, 1),datetime.date(1979, 4, 1), datetime.date(1979, 7, 1),datetime.date(1979, 10, 1), datetime.date(1980, 1, 1),datetime.date(1980, 4, 1), datetime.date(1980, 7, 1),datetime.date(1980, 10, 1), datetime.date(1981, 1, 1),datetime.date(1981, 4, 1), datetime.date(1981, 7, 1),datetime.date(1981, 10, 1), datetime.date(1982, 1, 1),datetime.date(1982, 4, 1), datetime.date(1982, 7, 1),datetime.date(1982, 10, 1), datetime.date(1983, 1, 1),datetime.date(1983, 4, 1), datetime.date(1983, 7, 1),datetime.date(1983, 10, 1), datetime.date(1984, 1, 1),datetime.date(1984, 4, 1), datetime.date(1984, 7, 1),datetime.date(1984, 10, 1), datetime.date(1985, 1, 1),datetime.date(1985, 4, 1), datetime.date(1985, 7, 1),datetime.date(1985, 10, 1), datetime.date(1986, 1, 1),datetime.date(1986, 4, 1), datetime.date(1986, 7, 1),datetime.date(1986, 10, 1), datetime.date(1987, 1, 1),datetime.date(1987, 4, 1), datetime.date(1987, 7, 1),datetime.date(1987, 10, 1), datetime.date(1988, 1, 1),datetime.date(1988, 4, 1), datetime.date(1988, 7, 1),datetime.date(1988, 10, 1), datetime.date(1989, 1, 1),datetime.date(1989, 4, 1), datetime.date(1989, 7, 1),datetime.date(1989, 10, 1), datetime.date(1990, 1, 1),datetime.date(1990, 4, 1), datetime.date(1990, 7, 1),datetime.date(1990, 10, 1), datetime.date(1991, 1, 1),datetime.date(1991, 4, 1), datetime.date(1991, 7, 1),datetime.date(1991, 10, 1), datetime.date(1992, 1, 1),datetime.date(1992, 4, 1), datetime.date(1992, 7, 1),datetime.date(1992, 10, 1), datetime.date(1993, 1, 1),datetime.date(1993, 4, 1), datetime.date(1993, 7, 1),datetime.date(1993, 10, 1), datetime.date(1994, 1, 1),datetime.date(1994, 4, 1), datetime.date(1994, 7, 1),datetime.date(1994, 10, 1), datetime.date(1995, 1, 1),datetime.date(1995, 4, 1), datetime.date(1995, 7, 1),datetime.date(1995, 10, 1), datetime.date(1996, 1, 1),datetime.date(1996, 4, 1), datetime.date(1996, 7, 1),datetime.date(1996, 10, 1), datetime.date(1997, 1, 1),datetime.date(1997, 4, 1), datetime.date(1997, 7, 1),datetime.date(1997, 10, 1), datetime.date(1998, 1, 1),datetime.date(1998, 4, 1), datetime.date(1998, 7, 1),datetime.date(1998, 10, 1), datetime.date(1999, 1, 1),datetime.date(1999, 4, 1), datetime.date(1999, 7, 1),datetime.date(1999, 10, 1), datetime.date(2000, 1, 1),datetime.date(2000, 4, 1), datetime.date(2000, 7, 1),datetime.date(2000, 10, 1), datetime.date(2001, 1, 1),datetime.date(2001, 4, 1), datetime.date(2001, 7, 1),datetime.date(2001, 10, 1), datetime.date(2002, 1, 1),datetime.date(2002, 4, 1), datetime.date(2002, 7, 1),datetime.date(2002, 10, 1), datetime.date(2003, 1, 1),datetime.date(2003, 4, 1), datetime.date(2003, 7, 1),datetime.date(2003, 10, 1), datetime.date(2004, 1, 1),datetime.date(2004, 4, 1), datetime.date(2004, 7, 1),datetime.date(2004, 10, 1), datetime.date(2005, 1, 1),datetime.date(2005, 4, 1), datetime.date(2005, 7, 1),datetime.date(2005, 10, 1), datetime.date(2006, 1, 1),datetime.date(2006, 4, 1), datetime.date(2006, 7, 1),datetime.date(2006, 10, 1), datetime.date(2007, 1, 1),datetime.date(2007, 4, 1), datetime.date(2007, 7, 1),datetime.date(2007, 10, 1), datetime.date(2008, 1, 1),datetime.date(2008, 4, 1), datetime.date(2008, 7, 1),datetime.date(2008, 10, 1), datetime.date(2009, 1, 1),datetime.date(2009, 4, 1), datetime.date(2009, 7, 1)], dtype=object)


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


2.397 1.841 1.509 ... 0.1912 -2.806


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 2.39685869e+00,  1.84090698e+00,  1.50879526e+00],[ 1.00474524e+00,  3.58084679e-01,  3.63134432e+00],[ 3.90787870e-01,  7.47617841e-01, -1.57816327e+00],...,[ 1.26518160e-02,  1.76712930e-01,  3.86593437e+00],[ 1.68201852e+00,  6.77952230e-01,  4.70714951e+00],[ 1.02604866e+00,  1.36811638e+00,  2.19665885e-01]],[[-4.34013903e-01, -4.61416483e-01, -1.67746758e+00],[ 1.16093898e+00,  1.19790626e+00,  4.16909504e+00],[ 9.33605313e-01,  1.81474578e+00,  3.04200220e+00],...,[-1.95794547e+00,  2.61071831e-01, -1.14376488e+01],[ 3.53035957e-01,  3.33862275e-01, -2.20977378e+00],[ 1.76334113e-01,  5.05898476e-01,  1.03597307e+00]],[[ 3.14941764e-01, -6.93475604e-02, -6.22571230e-01],[-1.15938246e-01,  3.56995881e-01, -4.13632870e+00],[ 6.66287720e-01,  9.70663011e-01, -2.53870726e+00],...,...[-7.85625279e-01,  7.17735946e-01, -8.73674393e+00],[-3.05579066e-01,  2.19690353e-01, -5.61023712e+00],[-1.03030455e+00,  8.42356741e-01, -1.04273682e+01]],[[ 1.13297510e+00,  1.16376901e+00,  9.57052588e-01],[ 1.25400400e+00,  1.60922110e-01,  6.12456274e+00],[ 7.14956641e-01,  1.22657061e+00,  1.54611742e+00],...,[ 3.97115946e-03,  4.28220212e-01, -5.45101929e+00],[-2.11458504e-02,  5.01457676e-02, -6.76644802e-01],[ 6.92328691e-01,  8.06323707e-01, -5.01065493e-01]],[[ 2.60350943e+00,  2.61257887e+00,  3.84846926e+00],[ 6.75936937e-02,  7.36723423e-01, -6.39577675e+00],[ 1.91653252e-01,  7.21629381e-01, -3.08792830e+00],...,[ 1.43772638e+00,  1.20329142e+00,  7.97855377e-01],[ 1.28967571e+00,  1.81867361e+00,  2.63391066e+00],[ 8.78272951e-02,  1.91163883e-01, -2.80562782e+00]]]],shape=(4, 1000, 200, 3), dtype=float32)


Attributes: (5)


created_at :  
2026-09-04T08:38:51.443220+00:00

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
2026-09-04T08:38:51.443862+00:00

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
2026-09-04T08:38:51.444395+00:00

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


1.231 1.315 2.307 ... 2.243 -2.136


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 1.2311203e+00,  1.3154070e+00,  2.3068211e+00],[ 9.8046356e-01,  6.6539669e-01,  5.6390052e+00],[ 9.9563104e-01,  7.9932255e-01,  4.9751482e+00],...,[-1.1789633e+00, -4.8977244e-01, -1.0178500e+01],[ 1.6657013e-01, -1.5221035e-01,  2.1085837e+00],[-5.9128082e-01, -1.6188243e-01, -1.0127937e+00]],[[ 8.7287295e-01,  1.0225533e+00,  1.4369516e+00],[ 6.4491689e-01,  6.1911124e-01,  5.9293432e+00],[-8.4993684e-01, -8.1330538e-03, -7.9900713e+00],...,[ 4.0474135e-01,  1.4574713e+00, -6.0473919e+00],[ 1.9391749e+00, -8.7293804e-02,  1.0428045e+01],[ 1.3712672e+00,  6.5469855e-01,  2.6277082e+00]],[[ 5.1640552e-01,  1.0494291e+00, -2.2500315e+00],[ 1.0861408e+00,  4.1991833e-01,  3.8893633e+00],[-6.6325414e-01, -1.7456025e-01, -3.3793764e+00],...,...[ 7.0160317e-01,  2.3713350e-01,  2.0843704e+00],[ 3.5343063e-01,  7.3891044e-01, -3.4922953e+00],[ 3.7721527e-01,  5.7609951e-01, -5.4384098e+00]],[[-7.4494177e-01, -1.3857440e+00, -4.1040821e+00],[-9.7574532e-01, -5.1348805e-01, -6.5439034e+00],[-1.4341778e+00,  7.2461677e-01, -1.5955865e+01],...,[ 6.7337042e-01,  1.1177440e+00,  8.8544309e-01],[ 2.3839116e-02,  7.7607942e-01, -3.3321176e+00],[ 2.0783362e+00,  1.1009558e+00,  5.7282581e+00]],[[ 7.4984151e-01, -2.6356649e-01,  4.8683777e+00],[-8.5763830e-01, -6.8324542e-01, -4.2654014e+00],[-8.1557572e-01, -3.0947536e-02, -3.3929849e+00],...,[ 1.1299040e+00,  1.6817294e+00,  4.3380919e+00],[ 1.1613892e+00,  1.4201896e+00,  4.4219656e+00],[ 1.7786182e+00,  2.2427478e+00, -2.1358867e+00]]]],shape=(4, 1000, 30, 3), dtype=float32)


Attributes: (5)


created_at :  
2026-09-04T08:38:51.756720+00:00

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
2026-09-04T08:38:51.757232+00:00

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
| intercept\[realgdp\] | 0.241 | 0.102 | 0.047 | 0.44 | 2466 | 2555 | 1.00 | 0.0021 | 0.0014 |
| intercept\[realcons\] | 0.554 | 0.099 | 0.37 | 0.74 | 3181 | 3075 | 1.00 | 0.0018 | 0.0012 |
| intercept\[realinv\] | -1.74 | 0.49 | -2.6 | -0.83 | 3134 | 3121 | 1.00 | 0.0087 | 0.0062 |
| sigma\[realgdp\] | 0.747 | 0.036 | 0.68 | 0.82 | 2471 | 2351 | 1.00 | 0.00073 | 0.00051 |
| sigma\[realcons\] | 0.659 | 0.0335 | 0.6 | 0.73 | 4315 | 3253 | 1.00 | 0.00051 | 0.00038 |
| sigma\[realinv\] | 3.883 | 0.185 | 3.6 | 4.3 | 4092 | 3059 | 1.00 | 0.0029 | 0.0022 |
| phi\[1, realgdp, realgdp\] | -0.06 | 0.141 | -0.32 | 0.21 | 1751 | 2246 | 1.00 | 0.0034 | 0.0025 |
| phi\[1, realgdp, realcons\] | 0.443 | 0.114 | 0.23 | 0.66 | 2073 | 2159 | 1.00 | 0.0025 | 0.0017 |
| phi\[1, realgdp, realinv\] | 0.01 | 0.0227 | -0.032 | 0.053 | 1728 | 2293 | 1.00 | 0.00055 | 0.0004 |
| phi\[1, realcons, realgdp\] | -0.057 | 0.146 | -0.33 | 0.22 | 2171 | 1979 | 1.00 | 0.0031 | 0.0022 |
| phi\[1, realcons, realcons\] | 0.228 | 0.115 | 0.017 | 0.44 | 2373 | 2242 | 1.00 | 0.0024 | 0.0018 |
| phi\[1, realcons, realinv\] | 0.021 | 0.0225 | -0.022 | 0.064 | 2266 | 2193 | 1.00 | 0.00047 | 0.00033 |
| phi\[1, realinv, realgdp\] | -0.5 | 0.6 | -1.6 | 0.65 | 2403 | 2715 | 1.00 | 0.012 | 0.0089 |
| phi\[1, realinv, realcons\] | 2.83 | 0.5 | 1.9 | 3.7 | 2992 | 2925 | 1.00 | 0.0091 | 0.0065 |
| phi\[1, realinv, realinv\] | 0.072 | 0.104 | -0.12 | 0.26 | 2285 | 2675 | 1.00 | 0.0022 | 0.0015 |
| phi\[2, realgdp, realgdp\] | -0.007 | 0.144 | -0.28 | 0.26 | 1718 | 2099 | 1.00 | 0.0035 | 0.0024 |
| phi\[2, realgdp, realcons\] | 0.264 | 0.124 | 0.034 | 0.5 | 1995 | 2105 | 1.00 | 0.0028 | 0.002 |
| phi\[2, realgdp, realinv\] | -0.001 | 0.0223 | -0.042 | 0.041 | 1897 | 2479 | 1.00 | 0.00051 | 0.00035 |
| phi\[2, realcons, realgdp\] | -0.114 | 0.145 | -0.38 | 0.16 | 2061 | 2239 | 1.00 | 0.0032 | 0.0021 |
| phi\[2, realcons, realcons\] | 0.222 | 0.123 | -0.014 | 0.45 | 2301 | 2647 | 1.00 | 0.0026 | 0.0018 |
| phi\[2, realcons, realinv\] | 0.0231 | 0.0218 | -0.018 | 0.063 | 2183 | 2347 | 1.00 | 0.00047 | 0.00031 |
| phi\[2, realinv, realgdp\] | 0.22 | 0.62 | -0.95 | 1.4 | 1993 | 2446 | 1.00 | 0.014 | 0.0097 |
| phi\[2, realinv, realcons\] | 0.64 | 0.54 | -0.36 | 1.7 | 2469 | 2334 | 1.00 | 0.011 | 0.0077 |
| phi\[2, realinv, realinv\] | -0.073 | 0.103 | -0.26 | 0.12 | 2619 | 2591 | 1.00 | 0.002 | 0.0014 |


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


    stable draws: 1.000, median spectral radius: 0.597, max: 0.827


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
width_94_h = width_94[[h - 1 for h in horizons]]

pl.DataFrame({"h": horizons} | dict(zip(names, width_94_h.T, strict=True))).with_columns(
    pl.col(names).round(3)
)
```


shape: (5, 4)

| h   | realgdp | realcons | realinv |
|-----|---------|----------|---------|
| i64 | f64     | f64      | f64     |
| 1   | 2.897   | 2.499    | 14.846  |
| 5   | 3.219   | 2.735    | 16.676  |
| 10  | 3.249   | 2.697    | 16.686  |
| 20  | 3.14    | 2.717    | 15.905  |
| 30  | 3.094   | 2.735    | 15.976  |


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
| "realgdp"  | 0.789              | 0.772       | 0.79                  |
| "realcons" | 0.837              | 0.832       | 0.824                 |
| "realinv"  | 0.947              | 0.818       | 1.02                  |


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

     [[-0.06   0.443  0.01 ]
      [-0.057  0.228  0.021]
      [-0.498  2.827  0.072]]

     [[-0.028  0.367  0.008]
      [-0.134  0.314  0.029]
      [ 0.055  1.27  -0.009]]]


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


The `realinv response to realcons shock` panel dominates the grid: its y-axis runs past 3, while every other off-diagonal panel stays within about \pm 0.6. The printed h = 0, 1, 2 snapshot above puts a number on it: a one-unit consumption-growth shock moves investment growth by 2.827 percentage points at h = 1 and 1.27 at h = 2, six to ten times larger than any other off-diagonal entry at the same horizons, consistent with investment's well-known sensitivity to demand shocks. Every panel decays toward zero by h \approx 8-10, as stability requires; the own-shock (diagonal) panels start at exactly 1 by construction and decay the fastest, some dipping briefly negative (e.g. `realgdp response to realgdp shock` around h = 1-2) before settling.


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
    [[1.26  0.612 0.139]
     [0.762 0.897 0.174]
     [5.187 1.377 2.671]]


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
| "realgdp" | 0.25 | 1.331 | -0.498 | 0.605 |
| "realcons" | 0.25 | 1.687 | 2.827 | 0.498 |
| "realinv" | 0.5 | 0.5 | 0.072 | 0.104 |


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
| intercept\[realgdp\] | 0.25 | 0.092 | 0.079 | 0.43 | 2494 | 2731 | 1.00 | 0.0018 | 0.0013 |
| intercept\[realcons\] | 0.548 | 0.088 | 0.38 | 0.71 | 3226 | 3077 | 1.00 | 0.0015 | 0.0011 |
| intercept\[realinv\] | -1.69 | 0.46 | -2.6 | -0.81 | 3052 | 2944 | 1.00 | 0.0084 | 0.0061 |
| sigma\[realgdp\] | 0.746 | 0.0353 | 0.68 | 0.82 | 3335 | 2824 | 1.00 | 0.00062 | 0.00044 |
| sigma\[realcons\] | 0.658 | 0.0331 | 0.6 | 0.72 | 4268 | 3306 | 1.00 | 0.00051 | 0.00038 |
| sigma\[realinv\] | 3.867 | 0.183 | 3.5 | 4.2 | 3918 | 2927 | 1.00 | 0.0029 | 0.0021 |
| phi\[1, realgdp, realgdp\] | -0.056 | 0.119 | -0.28 | 0.17 | 2048 | 2568 | 1.00 | 0.0026 | 0.0019 |
| phi\[1, realgdp, realcons\] | 0.467 | 0.102 | 0.28 | 0.66 | 2304 | 2662 | 1.00 | 0.0021 | 0.0015 |
| phi\[1, realgdp, realinv\] | 0.0117 | 0.0183 | -0.023 | 0.045 | 2207 | 2590 | 1.00 | 0.00039 | 0.00028 |
| phi\[1, realcons, realgdp\] | 0.001 | 0.104 | -0.2 | 0.19 | 2800 | 2672 | 1.00 | 0.002 | 0.0015 |
| phi\[1, realcons, realcons\] | 0.195 | 0.094 | 0.017 | 0.37 | 3085 | 2960 | 1.00 | 0.0017 | 0.0012 |
| phi\[1, realcons, realinv\] | 0.0155 | 0.0162 | -0.015 | 0.046 | 2764 | 2955 | 1.00 | 0.00031 | 0.00023 |
| phi\[1, realinv, realgdp\] | -0.86 | 0.65 | -2.1 | 0.4 | 2067 | 2401 | 1.00 | 0.014 | 0.011 |
| phi\[1, realinv, realcons\] | 3.27 | 0.56 | 2.2 | 4.3 | 2496 | 2707 | 1.00 | 0.011 | 0.0078 |
| phi\[1, realinv, realinv\] | 0.119 | 0.101 | -0.077 | 0.31 | 2228 | 2297 | 1.00 | 0.0021 | 0.0016 |
| phi\[2, realgdp, realgdp\] | 0.06 | 0.084 | -0.096 | 0.22 | 2956 | 3155 | 1.00 | 0.0015 | 0.0011 |
| phi\[2, realgdp, realcons\] | 0.168 | 0.083 | 0.0083 | 0.32 | 2534 | 2936 | 1.00 | 0.0017 | 0.0011 |
| phi\[2, realgdp, realinv\] | -0.0078 | 0.0134 | -0.033 | 0.018 | 3293 | 2976 | 1.00 | 0.00023 | 0.00017 |
| phi\[2, realcons, realgdp\] | -0.004 | 0.07 | -0.13 | 0.13 | 3726 | 3191 | 1.00 | 0.0011 | 0.00083 |
| phi\[2, realcons, realcons\] | 0.126 | 0.085 | -0.028 | 0.29 | 3347 | 2830 | 1.00 | 0.0015 | 0.001 |
| phi\[2, realcons, realinv\] | 0.0083 | 0.0114 | -0.014 | 0.03 | 4401 | 2756 | 1.00 | 0.00017 | 0.00013 |
| phi\[2, realinv, realgdp\] | 0.19 | 0.44 | -0.64 | 1 | 2776 | 2965 | 1.00 | 0.0084 | 0.0059 |
| phi\[2, realinv, realcons\] | 0.44 | 0.44 | -0.36 | 1.3 | 3003 | 2792 | 1.00 | 0.0081 | 0.0057 |
| phi\[2, realinv, realinv\] | -0.06 | 0.079 | -0.21 | 0.088 | 3063 | 3013 | 1.00 | 0.0014 | 0.001 |


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


    mean ratio on own lags: 0.78, on cross lags: 0.76


shape: (18, 6)

| lag | equation   | lagged series | weak prior | minnesota prior | ratio |
|-----|------------|---------------|------------|-----------------|-------|
| i64 | str        | str           | f32        | f32             | f32   |
| 1   | "realgdp"  | "realgdp"     | 0.141      | 0.119           | 0.843 |
| 1   | "realgdp"  | "realcons"    | 0.114      | 0.102           | 0.894 |
| 1   | "realgdp"  | "realinv"     | 0.023      | 0.018           | 0.804 |
| 1   | "realcons" | "realgdp"     | 0.146      | 0.104           | 0.713 |
| 1   | "realcons" | "realcons"    | 0.115      | 0.094           | 0.818 |
| …   | …          | …             | …          | …               | …     |
| 2   | "realcons" | "realcons"    | 0.123      | 0.085           | 0.69  |
| 2   | "realcons" | "realinv"     | 0.022      | 0.011           | 0.525 |
| 2   | "realinv"  | "realgdp"     | 0.621      | 0.442           | 0.712 |
| 2   | "realinv"  | "realcons"    | 0.541      | 0.442           | 0.818 |
| 2   | "realinv"  | "realinv"     | 0.103      | 0.079           | 0.765 |


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
| "realgdp"  | 3.21       | 3.168           |
| "realcons" | 2.705      | 2.63            |
| "realinv"  | 16.489     | 16.536          |


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
