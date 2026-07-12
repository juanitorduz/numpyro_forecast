# Forecasting retail demand under stockouts


The [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) dataset ([Yang et al., 2025](https://arxiv.org/abs/2505.16319)) contains \\50{,}000\\ daily store-product demand series from fresh retail operations: \\90\\ days per series, with hourly sales, hourly stockout labels, and promotion, discount, and calendar covariates. The native stockout labels make it a great public benchmark for a classic operational problem: observed sales are a *censored* version of demand, because a store cannot sell what is not on the shelf.

We model this with a **multiplicative availability factor**: expected sales factor into a demand component and an availability component. A forecasting model describes what demand would be with the product fully in stock, and a multiplicative factor, a function of the day's recorded availability with parameters learned from the data, scales that demand down when it was not. In this example the two components are:

- **Forecasting model**: a state space model with a trend, weekly seasonality, and promotion effects.
- **Availability factor**: a saturating function of the day's stock availability.

The rationale is simple: when a product is out of stock, recorded sales say little about demand, and without an explicit correction the forecasting model would misread stockout days as low-demand days. The factorization also pays off at prediction time: because the demand component describes what would sell with the product fully in stock, setting availability to one over the forecast horizon turns the sales forecast into a *demand* forecast, which is the number a planner should order against.

There is a catch, though, and it is the heart of this example: **days whose stockout labels say the product was out of stock all day still record positive sales about \\15\\\\ of the time**. A pure multiplicative factor forces the mean to zero on those days and badly misfits them. The likely reason is that the stockout labels are reconstructed from imperfect inventory snapshots, so they carry noise, a common situation in practice. The model developed in this notebook absorbs the contradiction by learning a *floor* in the availability factor: even at zero recorded availability, a small share of demand can still be sold.

We proceed in four steps. First, an exploratory analysis of the full \\50{,}000\\-series dataset: we look closely at the stockout and availability labels, quantify the contradiction above, and trace it to label noise concentrated in hours that carry almost no demand, which motivates both a *sales-weighted* availability feature and the *learned floor* in the availability factor. Second, we fit a hierarchical state space model to the top \\1{,}000\\ series with SVI and a custom `optax` optimizer, wrapping the results in an ArviZ `DataTree`. Third, we evaluate the forecasts with CRPS and central-interval coverage on a simple train-test split against a seasonal-naive baseline. Fourth, we re-issue the forecast with availability pinned to one over the horizon: a counterfactual estimate of uncensored demand that is deliberately *not* meant to track the observed (censored) sales, and is exactly what a business should plan against, since nobody knows future availability at prediction time. We close by inspecting what the model learned: the fitted availability factor, the store hierarchy, and the promotion contributions.


# Prepare notebook


    In [1]:


``` python
from collections.abc import Callable
from typing import Any, cast

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import polars as pl
import preliz as pz
import xarray as xr
from huggingface_hub import hf_hub_download
from huggingface_hub import logging as hf_logging
from jax import random
from jaxtyping import Float, Int
from matplotlib import ticker as mtick
from numpyro import handlers
from numpyro.infer import Predictive
from numpyro.infer.reparam import LocScaleReparam
from sklearn.preprocessing import LabelEncoder

from numpyro_forecast import (
    Forecaster,
    ForecastingModel,
    eval_coverage,
    eval_crps,
    predictions_to_datatree,
    to_datatree,
)
from numpyro_forecast.features import periodic_repeat
from numpyro_forecast.functional import draw_posterior, fit_svi, forecast, predict_in_sample
from numpyro_forecast.metrics import crps_empirical
from numpyro_forecast.typing import Array

az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"

# Render polars tables without truncating string cells, and drop the shape and
# dtype headers, which are noise in a rendered document.
pl.Config.set_fmt_str_lengths(100)
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True)

# The Hub intermittently sends an unauthenticated-request warning header on
# responses, which huggingface_hub logs to stderr; the example needs no token,
# so keep the rendered document clean.
hf_logging.set_verbosity_error()

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


# Read data

We download the training split (a single parquet file, cached locally by `huggingface_hub`) and scan it lazily with polars, so the full-dataset aggregations below stream instead of materializing all \\4.5\\ million rows.


    In [2]:


``` python
parquet_path: str = hf_hub_download(
    repo_id="Dingdong-Inc/FreshRetailNet-50K",
    filename="data/train.parquet",
    repo_type="dataset",
)
data_lf: pl.LazyFrame = pl.scan_parquet(parquet_path)

data_lf.collect_schema()
```


    Schema([('city_id', Int64),
            ('store_id', Int64),
            ('management_group_id', Int64),
            ('first_category_id', Int64),
            ('second_category_id', Int64),
            ('third_category_id', Int64),
            ('product_id', Int64),
            ('dt', String),
            ('sale_amount', Float64),
            ('hours_sale', List(Float64)),
            ('stock_hour6_22_cnt', Int32),
            ('hours_stock_status', List(Int64)),
            ('discount', Float64),
            ('holiday_flag', Int32),
            ('activity_flag', Int32),
            ('precpt', Float64),
            ('avg_temperature', Float64),
            ('avg_humidity', Float64),
            ('avg_wind_level', Float64)])


    In [3]:


``` python
data_lf.select(
    pl.len().alias("rows"),
    pl.col("city_id").n_unique().alias("cities"),
    pl.col("store_id").n_unique().alias("stores"),
    pl.col("product_id").n_unique().alias("products"),
    pl.struct(["store_id", "product_id"]).n_unique().alias("series"),
    pl.col("dt").min().alias("start_date"),
    pl.col("dt").max().alias("end_date"),
).collect(engine="streaming")
```


| rows    | cities | stores | products | series | start_date   | end_date     |
|---------|--------|--------|----------|--------|--------------|--------------|
| 4500000 | 18     | 898    | 865      | 50000  | "2024-03-28" | "2024-06-25" |


Every one of the \\50{,}000\\ store-product series covers the same \\90\\ days. Three columns drive this notebook:

- `sale_amount` is the daily sales target, and `hours_sale` is its hourly decomposition (it sums to `sale_amount` up to float rounding).
- `hours_stock_status` is a \\24\\-vector of hourly stockout indicators (\\1\\ means out of stock in that hour).
- `stock_hour6_22_cnt` counts stockout hours within the \\6{:}00\\ to \\22{:}00\\ daytime window, so its maximum is \\16\\.

The next cell verifies these conventions directly instead of trusting the documentation.


    In [4]:


``` python
data_lf.select(
    (pl.col("hours_sale").list.sum() - pl.col("sale_amount")).abs().max().alias("max_abs_diff"),
    pl.col("hours_stock_status").list.len().max().alias("hours_per_day"),
    pl.col("stock_hour6_22_cnt").max().alias("max_daytime_stockout_hours"),
).collect(engine="streaming")
```


| max_abs_diff | hours_per_day | max_daytime_stockout_hours |
|--------------|---------------|----------------------------|
| 1.4211e-14   | 24            | 16                         |


# Exploratory data analysis

The queries in this section are built from small named polars expression functions, chained with `pipe` for the frame-level steps, so each query reads top to bottom. The expressions defined along the way (stockout hours, scaled sales, the sales-weighted availability) are reused all the way into the modeling panel.


## How prevalent are stockouts?


    In [5]:


``` python
def stockout_hours() -> pl.Expr:
    """Count the hours of the day flagged out of stock."""
    return pl.col("hours_stock_status").list.sum()


stockout_hours_df = (
    data_lf.group_by(stockout_hours().alias("stockout_hours"))
    .len()
    .sort("stockout_hours")
    .with_columns((pl.col("len") / pl.col("len").sum()).alias("share_of_days"))
    .collect(engine="streaming")
)

fig, ax = plt.subplots()
ax.bar(
    stockout_hours_df["stockout_hours"],
    stockout_hours_df["share_of_days"],
    color="C0",
    label="share of days",
)
ax.legend(loc="upper right")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
ax.set(
    xlabel="stockout hours per day",
    ylabel="share of days",
    title="Distribution of daily stockout hours",
);
```


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-6-output-1.png" class="figure-img" width="1211" height="711" /></p>
</figure>


The distribution is strongly bimodal: \\40\\\\ of the days have no stockout at all, most of the rest lose a handful of hours, and a visible spike of about \\3.8\\\\ of the days is flagged out of stock for all \\24\\ hours.


## The contradiction: sales on fully out-of-stock days

If the labels were exact, a day flagged out of stock for every hour could not sell anything. Let us check that under both stockout definitions (all \\24\\ hours flagged, and all \\16\\ daytime hours flagged).


    In [6]:


``` python
max_daytime_hours = 16


def oos_all_day() -> pl.Expr:
    """Day flagged out of stock for all 24 hours."""
    return stockout_hours() == 24


def oos_daytime() -> pl.Expr:
    """Day flagged out of stock for all 16 daytime (6:00 to 22:00) hours."""
    return pl.col("stock_hour6_22_cnt") == max_daytime_hours


def scaled_sales() -> pl.Expr:
    """Daily sales scaled by the series' own mean, so 1 is an average day."""
    return pl.col("sale_amount") / pl.col("sale_amount").mean().over("store_id", "product_id")


def share_positive_sales_when(flag: pl.Expr) -> pl.Expr:
    """Share of days with positive sales among the days where ``flag`` holds."""
    return (pl.col("sale_amount") > 0).filter(flag).mean()


data_lf.select(
    oos_all_day().mean().alias("share_days_oos_all_day"),
    oos_daytime().mean().alias("share_days_oos_daytime"),
    share_positive_sales_when(oos_all_day()).alias("p_sales_oos_all_day"),
    share_positive_sales_when(oos_daytime()).alias("p_sales_oos_daytime"),
    scaled_sales()
    .filter(oos_all_day() & (pl.col("sale_amount") > 0))
    .mean()
    .alias("relative_sales_when_positive"),
).collect(engine="streaming")
```


| share_days_oos_all_day | share_days_oos_daytime | p_sales_oos_all_day | p_sales_oos_daytime | relative_sales_when_positive |
|----|----|----|----|----|
| 0.037937 | 0.040398 | 0.147589 | 0.185631 | 0.248096 |


About \\15\\\\ of the all-day stockout days (and \\19\\\\ of the daytime ones) still record positive sales, and when they do the amount is not negligible: about a quarter of the series' average daily sales. Plausible mechanisms are stockout labels reconstructed from inventory snapshots, a sell-out followed by a restock within the same hour, or back-room stock that never registered on the shelf system. Whatever the cause, the labels are noisy, and a model that pins the mean at zero whenever recorded availability is zero is misspecified.


## When do sales and stockouts happen within the day?

The hourly decomposition tells us how much each flagged hour actually matters for daily demand.


    In [7]:


``` python
def total_sales_in_hour(hour: int) -> pl.Expr:
    """Total sales recorded in one hour of the day, over the full dataset."""
    return pl.col("hours_sale").list.get(hour).sum()


def stockout_rate_in_hour(hour: int) -> pl.Expr:
    """Share of days flagged out of stock in one hour of the day."""
    return pl.col("hours_stock_status").list.get(hour).mean()


hourly_row = (
    data_lf.select(
        *[total_sales_in_hour(h).alias(f"sales_{h}") for h in range(24)],
        *[stockout_rate_in_hour(h).alias(f"oos_{h}") for h in range(24)],
    )
    .collect(engine="streaming")
    .row(0)
)
sales_by_hour = np.asarray(hourly_row[:24], dtype=np.float64)
stockout_rate_by_hour = np.asarray(hourly_row[24:], dtype=np.float64)
hourly_weights = sales_by_hour / sales_by_hour.sum()

fig, ax = plt.subplots()
ax.bar(np.arange(24), hourly_weights, color="C0", alpha=0.8, label="share of daily sales")
ax_twin = ax.twinx()
ax_twin.plot(
    np.arange(24),
    stockout_rate_by_hour,
    color="C1",
    marker="o",
    markersize=8,
    linewidth=2,
    label="stockout rate",
)
ax_twin.grid(False)
ax_twin.set(ylabel="stockout rate", ylim=(0, 0.5))
handles, bar_labels = ax.get_legend_handles_labels()
handles_twin, labels_twin = ax_twin.get_legend_handles_labels()
ax.legend(handles + handles_twin, bar_labels + labels_twin, loc="upper left")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
ax_twin.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
ax.set(
    xlabel="hour of day",
    ylabel="share of daily sales",
    title="Hourly sales profile vs hourly stockout rate",
)
print(f"share of sales in the 6:00 to 22:00 window: {hourly_weights[6:22].sum():.1%}")
print(
    f"peak hourly stockout rate: {stockout_rate_by_hour.max():.1%} "
    f"(hour {int(stockout_rate_by_hour.argmax())})"
)
```


    share of sales in the 6:00 to 22:00 window: 95.2%
    peak hourly stockout rate: 41.9% (hour 22)


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-8-output-2.png" class="figure-img" width="1211" height="711" /></p>
</figure>


The two curves are almost mirror images: sales concentrate between \\7{:}00\\ and \\20{:}00\\ (\\95\\\\ of all sales fall in the \\6{:}00\\ to \\22{:}00\\ window), while the stockout rate peaks at \\42\\\\ late at night, exactly when nobody is buying. A raw \\24\\-hour stockout count therefore heavily over-penalizes availability. We can also measure the label noise directly: how much of the total sales volume is recorded in hours that are flagged out of stock?


    In [8]:


``` python
def sales_in_flagged_hours() -> pl.Expr:
    """Sales volume recorded in the hours whose own stockout flag is set."""
    return pl.sum_horizontal(
        [
            (pl.col("hours_sale").list.get(h) * pl.col("hours_stock_status").list.get(h))
            for h in range(24)
        ]
    )


data_lf.select(
    (sales_in_flagged_hours().sum() / pl.col("hours_sale").list.sum().sum()).alias(
        "share_of_sales_in_flagged_stockout_hours"
    )
).collect(engine="streaming")
```


| share_of_sales_in_flagged_stockout_hours |
|------------------------------------------|
| 0.025131                                 |


\\2.5\\\\ of *all* sales happen in hours the labels declare out of stock. This is direct, hour-level evidence that the stockout signal has noise that no availability feature can remove, and it is why the model below learns a floor instead of trusting availability zero to mean demand zero.


## A sales-weighted availability feature

Instead of counting stockout hours uniformly, we weight each hour by its share of global sales, so that losing a night hour costs almost nothing and losing the morning peak costs a lot:

\\a\_{t,s} = \sum\_{h=0}^{23} w_h \left(1 - \text{stockout}\_{t,s,h}\right), \qquad w_h = \frac{\text{total sales in hour } h}{\text{total sales}}.\\

One note on hygiene: the weights \\w_h\\ are a global hour-of-day profile computed over the full dataset, test window included, whereas a deployed system would compute them on history only. We keep the dataset-wide profile for simplicity; the effect of \\14\\ extra days on a fixed \\24\\-number profile is negligible, and the availability feature itself is treated as a known future input in the retrospective evaluation set up below anyway.


    In [9]:


``` python
def sales_weighted_availability(weights: Float[np.ndarray, " hours"]) -> pl.Expr:
    """In-stock share of the day, weighting each hour by its share of global sales."""
    return pl.sum_horizontal(
        [(1 - pl.col("hours_stock_status").list.get(h)) * float(weights[h]) for h in range(24)]
    ).alias("availability")


def zero_availability_summary(
    lf: pl.LazyFrame, zero_flag: pl.Expr, definition: str
) -> pl.LazyFrame:
    """Summarize the zero-availability days implied by one stockout definition."""
    return lf.select(
        pl.lit(definition).alias("definition"),
        zero_flag.mean().alias("share_days_zero_availability"),
        share_positive_sales_when(zero_flag).alias("p_positive_sales_given_zero"),
    )


availability_expr: pl.Expr = sales_weighted_availability(hourly_weights)

pl.concat(
    [
        data_lf.pipe(zero_availability_summary, oos_daytime(), "daytime hours all flagged"),
        data_lf.pipe(zero_availability_summary, oos_all_day(), "all 24 hours flagged"),
        data_lf.pipe(
            zero_availability_summary, availability_expr == 0, "sales-weighted availability = 0"
        ),
    ]
).collect(engine="streaming")
```


| definition | share_days_zero_availability | p_positive_sales_given_zero |
|----|----|----|
| "daytime hours all flagged" | 0.040398 | 0.185631 |
| "all 24 hours flagged" | 0.037937 | 0.147589 |
| "sales-weighted availability = 0" | 0.037937 | 0.147589 |


Two remarks on this table. First, because every hour carries a positive share of global sales, the weighted availability is exactly zero only when all \\24\\ hours are flagged: its zero set coincides with the all-day definition by construction, and it inherits that definition's lower \\15\\\\ contradiction rate. The daytime definition's hard zero is both noisier (\\19\\\\) and cruder, since it also zeroes out days that were merely stocked overnight; under the weighted feature those days keep a tiny positive \\a\_{t,s}\\ instead. Second, the feature's real contribution lies between the extremes: it grades partial days by how much *selling time* they lose, so a lost night hour costs almost nothing and a lost morning-peak hour costs a lot. The \\15\\\\ that remains at exact zero is irreducible label noise, and the model handles it with a learned floor rather than a data transformation.


## The empirical demand-availability curve

How do sales respond to partial availability? We scale each series by its own mean (so different volumes are comparable) and bin the scaled sales by weighted availability.


    In [10]:


``` python
def availability_bin() -> pl.Expr:
    """Decile bin (0 to 9) of the sales-weighted availability."""
    return (pl.col("availability") * 10).floor().clip(0, 9).alias("availability_bin")


def mean_scaled_sales_by_availability_bin(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Bin the days by availability and average the scaled sales within each bin."""
    return (
        lf.group_by(availability_bin())
        .agg(
            pl.col("availability").mean().alias("mean_availability"),
            pl.col("scaled_sales").mean().alias("mean_scaled_sales"),
            pl.len().alias("days"),
        )
        .sort("availability_bin")
    )


def mean_scaled_sales_at_zero_availability(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Average the scaled sales over the days with zero sales-weighted availability."""
    return lf.filter(pl.col("availability") == 0).select(
        pl.col("scaled_sales").mean().alias("mean_scaled_sales"),
        pl.len().alias("days"),
    )


availability_sales_lf = data_lf.with_columns(
    availability_expr, scaled_sales().alias("scaled_sales")
)
factor_curve_df = availability_sales_lf.pipe(mean_scaled_sales_by_availability_bin).collect(
    engine="streaming"
)
empirical_floor_df = availability_sales_lf.pipe(mean_scaled_sales_at_zero_availability).collect(
    engine="streaming"
)
empirical_floor = float(empirical_floor_df["mean_scaled_sales"][0])

fig, ax = plt.subplots()
ax.plot(
    factor_curve_df["mean_availability"],
    factor_curve_df["mean_scaled_sales"],
    "o-",
    color="C0",
    label="binned mean",
)
ax.plot(0.0, empirical_floor, "D", color="C3", markersize=8, label="mean at availability = 0")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="series average")
ax.legend(loc="lower right")
ax.set(
    xlabel="sales-weighted availability",
    ylabel="scaled sales (series mean = 1)",
    title="Scaled sales vs sales-weighted availability (full dataset)",
)
print(f"empirical floor at zero availability: {empirical_floor:.3f}")
```


    empirical floor at zero availability: 0.037


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-11-output-2.png" class="figure-img" width="1211" height="711" /></p>
</figure>


The curve is saturating, exactly the shape a multiplicative factor should have: steep gains at low availability, flattening out near full availability. Two details matter for the model. First, the value at zero availability is positive (about \\0.04\\), which is the empirical floor the factor must reproduce. Second, the bins just below full availability sit slightly *above* the fully-available bin. That is endogeneity, not magic: stockouts happen disproportionately on high-demand days (a sell-out is itself evidence of demand), so a naive read of this curve overstates what availability alone does. The model mitigates this by attributing day-to-day variation to the trend, weekly seasonality, and promotion covariates jointly with the factor.


## Promotion and calendar covariates

We encode the discount as `discount_magnitude = 1 - discount`, so zero means no discount and larger values mean deeper discounts (a positive coefficient then reads "more discount, more sales").


    In [11]:


``` python
def raw_discount_magnitude() -> pl.Expr:
    """Depth of the discount read literally from the raw column: 1 - discount."""
    return 1 - pl.col("discount")


def mean_scaled_sales_when(flag: str, active: bool) -> pl.Expr:
    """Mean scaled sales over the days where a 0/1 flag column is (in)active."""
    prefix = "scaled" if active else "scaled_no"
    name = flag.removesuffix("_flag")
    return scaled_sales().filter(pl.col(flag) == int(active)).mean().alias(f"{prefix}_{name}")


data_lf.select(
    raw_discount_magnitude().mean().alias("mean_discount_magnitude"),
    (raw_discount_magnitude() > 0).mean().alias("share_days_discounted"),
    (pl.col("discount") == 0).mean().alias("share_days_discount_zero"),
    pl.col("activity_flag").mean().alias("share_days_activity"),
    pl.col("holiday_flag").mean().alias("share_days_holiday"),
    mean_scaled_sales_when("activity_flag", active=True),
    mean_scaled_sales_when("activity_flag", active=False),
    mean_scaled_sales_when("holiday_flag", active=True),
    mean_scaled_sales_when("holiday_flag", active=False),
).unpivot().collect(engine="streaming")
```


| variable                   | value    |
|----------------------------|----------|
| "mean_discount_magnitude"  | 0.088859 |
| "share_days_discounted"    | 0.515299 |
| "share_days_discount_zero" | 0.003564 |
| "share_days_activity"      | 0.378421 |
| "share_days_holiday"       | 0.344444 |
| "scaled_activity"          | 1.113655 |
| "scaled_no_activity"       | 0.930806 |
| "scaled_holiday"           | 1.147654 |
| "scaled_no_holiday"        | 0.922419 |


Discounts are common (about half of all days, with a mean magnitude near \\9\\\\), and both promotion activity and holidays lift scaled sales by roughly \\20\\\\ on average. All three are worth including as regression covariates, with effects pooled hierarchically by store. One anomaly to keep in mind: a small share of days (\\0.4\\\\ dataset-wide) records `discount = 0`, which read literally would be a \\100\\\\ discount and is far more plausibly an unpriced placeholder; it looks negligible here, but we will meet it again in the modeling panel, where it turns out to be concentrated in exactly the series we model.


# Build the modeling panel

We model the top \\1{,}000\\ series by total sales. The last \\14\\ days are held out as a test set; the model trains on the first \\76\\ days and receives the *actual* covariates (availability, discount, promotion, holiday, launch indicator) over the forecast window, which is the standard retrospective evaluation setup.

Getting from the long dataframe to model-ready arrays is worth doing carefully, because every shape convention we set here is relied on by the model and by the ArviZ export. We proceed in five steps: pivot the long panel into dense `(time, series)` arrays, scale each series by its own training mean, build the integer store index that drives the hierarchical pooling, add the panel-wide launch indicator, and stack all exogenous inputs into a single tensor with named axes.

One data decision happens right in the panel build: the placeholder `discount = 0` days flagged in the EDA are encoded as *no discount* (and the handful of `discount > 1` artifacts are clipped). The feature itself stays in the model; the encoding fix is what makes its coefficient read as a genuine discount effect rather than a data-gap indicator, and the promotion plot further below shows why that matters for this panel.


    In [12]:


``` python
n_series_panel = 1_000
t_train = 76
horizon = 14


def top_series_by_total_sales(lf: pl.LazyFrame, n: int) -> pl.LazyFrame:
    """Rank the series by total sales over the window and keep the top ``n``."""
    return (
        lf.group_by("store_id", "product_id")
        .agg(pl.col("sale_amount").sum().alias("total_sales"))
        .sort("total_sales", descending=True)
        .head(n)
    )


def keep_top_series(lf: pl.LazyFrame, top_series: pl.DataFrame) -> pl.LazyFrame:
    """Keep only the rows belonging to the selected store-product series."""
    return lf.join(
        top_series.lazy().select("store_id", "product_id"), on=["store_id", "product_id"]
    )


def series_unique_id() -> pl.Expr:
    """Store-product identifier ``store::product`` naming each series."""
    return pl.concat_str(["store_id", "product_id"], separator="::").alias("unique_id")


def cleaned_discount_magnitude() -> pl.Expr:
    """Discount depth with the placeholder days encoded as no discount.

    ``discount == 0`` records are unpriced placeholders (not free giveaways), so
    they map to zero magnitude; values above 1 are clipped artifacts.
    """
    return (
        pl.when(pl.col("discount") > 0)
        .then(raw_discount_magnitude().clip(0.0, 1.0))
        .otherwise(0.0)
        .alias("discount_magnitude")
    )


top_series_df = data_lf.pipe(top_series_by_total_sales, n_series_panel).collect(engine="streaming")

panel_df = (
    data_lf.pipe(keep_top_series, top_series_df)
    .with_columns(
        pl.col("dt").cast(pl.Date),
        series_unique_id(),
        availability_expr,
        cleaned_discount_magnitude(),
    )
    .select(
        "dt",
        "unique_id",
        "store_id",
        "sale_amount",
        "availability",
        "discount_magnitude",
        pl.col("activity_flag").cast(pl.Float64),
        pl.col("holiday_flag").cast(pl.Float64),
    )
    .collect(engine="streaming")
    .sort("unique_id", "dt")
)

series_ids: list[str] = panel_df["unique_id"].unique().sort().to_list()
n_series = len(series_ids)
dates_series = panel_df["dt"].unique().sort()
dates = dates_series.to_numpy()

panel_df.head()
```


| dt | unique_id | store_id | sale_amount | availability | discount_magnitude | activity_flag | holiday_flag |
|----|----|----|----|----|----|----|----|
| 2024-03-28 | "0::117" | 0 | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2024-03-29 | "0::117" | 0 | 5.0 | 0.968539 | 0.0 | 0.0 | 0.0 |
| 2024-03-30 | "0::117" | 0 | 6.6 | 1.0 | 0.0 | 0.0 | 1.0 |
| 2024-03-31 | "0::117" | 0 | 8.0 | 1.0 | 0.0 | 0.0 | 1.0 |
| 2024-04-01 | "0::117" | 0 | 0.5 | 0.122896 | 0.0 | 0.0 | 0.0 |


## From long panel to a named dataset

The package convention places time at axis \\-2\\ and the observation (series) axis last, so the data panel is a dense `(time, n_series)` matrix. `make_pivot` builds one such matrix per column, always selecting the columns in `series_ids` order: that single sorted list defines the series axis *everywhere* (data, covariates, store index, ArviZ coordinates), so column \\s\\ refers to the same store-product pair in every array that follows. The function also validates the result, one row per date, one column per series, and no missing entries, so a silent join or pivot problem fails loudly here instead of corrupting the fit later.

The pivots themselves go straight into an `xarray.Dataset` with named `time` and `series` coordinates, and that dataset is the source of truth for everything downstream: selections read by label (`panel_ds["sale_amount"].sel(series="22::267")`) instead of positional index bookkeeping, reductions name their axis (`.mean("time")`), and the plain `jax.numpy` arrays the model consumes are extracted from it at the model boundary.


    In [13]:


``` python
def make_pivot(value: str) -> Float[np.ndarray, " duration n_series"]:
    """Build the dense (date x series) matrix of one panel column.

    Columns follow ``series_ids`` order so every pivot shares the same series axis.
    """
    pivot_df = panel_df.pivot(index="dt", on="unique_id", values=value).sort("dt")
    matrix = pivot_df.select(series_ids).to_numpy().astype(np.float64)
    if matrix.shape != (len(dates), n_series) or np.isnan(matrix).any():
        msg = f"Unexpected pivot for {value!r}: shape {matrix.shape}"
        raise ValueError(msg)
    return matrix


panel_ds: xr.Dataset = xr.Dataset(
    {
        name: (("time", "series"), make_pivot(name))
        for name in [
            "sale_amount",
            "availability",
            "discount_magnitude",
            "activity_flag",
            "holiday_flag",
        ]
    },
    coords={"time": dates, "series": series_ids},
)

panel_ds
```


![](data:image/svg+xml;base64,PHN2ZyBzdHlsZT0icG9zaXRpb246IGFic29sdXRlOyB3aWR0aDogMDsgaGVpZ2h0OiAwOyBvdmVyZmxvdzogaGlkZGVuIj4KPGRlZnM+CjxzeW1ib2wgaWQ9Imljb24tZGF0YWJhc2UiIHZpZXdib3g9IjAgMCAzMiAzMiI+CjxwYXRoIGQ9Ik0xNiAwYy04LjgzNyAwLTE2IDIuMjM5LTE2IDV2NGMwIDIuNzYxIDcuMTYzIDUgMTYgNXMxNi0yLjIzOSAxNi01di00YzAtMi43NjEtNy4xNjMtNS0xNi01eiIgLz4KPHBhdGggZD0iTTE2IDE3Yy04LjgzNyAwLTE2LTIuMjM5LTE2LTV2NmMwIDIuNzYxIDcuMTYzIDUgMTYgNXMxNi0yLjIzOSAxNi01di02YzAgMi43NjEtNy4xNjMgNS0xNiA1eiIgLz4KPHBhdGggZD0iTTE2IDI2Yy04LjgzNyAwLTE2LTIuMjM5LTE2LTV2NmMwIDIuNzYxIDcuMTYzIDUgMTYgNXMxNi0yLjIzOSAxNi01di02YzAgMi43NjEtNy4xNjMgNS0xNiA1eiIgLz4KPC9zeW1ib2w+CjxzeW1ib2wgaWQ9Imljb24tZmlsZS10ZXh0MiIgdmlld2JveD0iMCAwIDMyIDMyIj4KPHBhdGggZD0iTTI4LjY4MSA3LjE1OWMtMC42OTQtMC45NDctMS42NjItMi4wNTMtMi43MjQtMy4xMTZzLTIuMTY5LTIuMDMwLTMuMTE2LTIuNzI0Yy0xLjYxMi0xLjE4Mi0yLjM5My0xLjMxOS0yLjg0MS0xLjMxOWgtMTUuNWMtMS4zNzggMC0yLjUgMS4xMjEtMi41IDIuNXYyN2MwIDEuMzc4IDEuMTIyIDIuNSAyLjUgMi41aDIzYzEuMzc4IDAgMi41LTEuMTIyIDIuNS0yLjV2LTE5LjVjMC0wLjQ0OC0wLjEzNy0xLjIzLTEuMzE5LTIuODQxek0yNC41NDMgNS40NTdjMC45NTkgMC45NTkgMS43MTIgMS44MjUgMi4yNjggMi41NDNoLTQuODExdi00LjgxMWMwLjcxOCAwLjU1NiAxLjU4NCAxLjMwOSAyLjU0MyAyLjI2OHpNMjggMjkuNWMwIDAuMjcxLTAuMjI5IDAuNS0wLjUgMC41aC0yM2MtMC4yNzEgMC0wLjUtMC4yMjktMC41LTAuNXYtMjdjMC0wLjI3MSAwLjIyOS0wLjUgMC41LTAuNSAwIDAgMTUuNDk5LTAgMTUuNSAwdjdjMCAwLjU1MiAwLjQ0OCAxIDEgMWg3djE5LjV6IiAvPgo8cGF0aCBkPSJNMjMgMjZoLTE0Yy0wLjU1MiAwLTEtMC40NDgtMS0xczAuNDQ4LTEgMS0xaDE0YzAuNTUyIDAgMSAwLjQ0OCAxIDFzLTAuNDQ4IDEtMSAxeiIgLz4KPHBhdGggZD0iTTIzIDIyaC0xNGMtMC41NTIgMC0xLTAuNDQ4LTEtMXMwLjQ0OC0xIDEtMWgxNGMwLjU1MiAwIDEgMC40NDggMSAxcy0wLjQ0OCAxLTEgMXoiIC8+CjxwYXRoIGQ9Ik0yMyAxOGgtMTRjLTAuNTUyIDAtMS0wLjQ0OC0xLTFzMC40NDgtMSAxLTFoMTRjMC41NTIgMCAxIDAuNDQ4IDEgMXMtMC40NDggMS0xIDF6IiAvPgo8L3N5bWJvbD4KPC9kZWZzPgo8L3N2Zz4=)

``` xr-text-repr-fallback
<xarray.Dataset> Size: 4MB
Dimensions:             (time: 90, series: 1000)
Coordinates:
  * time                (time) datetime64[s] 720B 2024-03-28 ... 2024-06-25
  * series              (series) <U8 32kB '0::117' '0::691' ... '99::589' '9::4'
Data variables:
    sale_amount         (time, series) float64 720kB 0.3 3.5 5.1 ... 13.7 1.7
    availability        (time, series) float64 720kB 0.0 0.8421 ... 0.899 0.6361
    discount_magnitude  (time, series) float64 720kB 0.0 0.0 0.0 ... 0.0 0.059
    activity_flag       (time, series) float64 720kB 0.0 0.0 0.0 ... 0.0 0.0 0.0
    holiday_flag        (time, series) float64 720kB 0.0 0.0 0.0 ... 0.0 0.0 0.0
```


xarray.Dataset


Dimensions:


- time: 90
- series: 1000


Coordinates: (2)


time


(time)


datetime64\[s\]


2024-03-28 ... 2024-06-25


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['2024-03-28T00:00:00', '2024-03-29T00:00:00', '2024-03-30T00:00:00',
           '2024-03-31T00:00:00', '2024-04-01T00:00:00', '2024-04-02T00:00:00',
           '2024-04-03T00:00:00', '2024-04-04T00:00:00', '2024-04-05T00:00:00',
           '2024-04-06T00:00:00', '2024-04-07T00:00:00', '2024-04-08T00:00:00',
           '2024-04-09T00:00:00', '2024-04-10T00:00:00', '2024-04-11T00:00:00',
           '2024-04-12T00:00:00', '2024-04-13T00:00:00', '2024-04-14T00:00:00',
           '2024-04-15T00:00:00', '2024-04-16T00:00:00', '2024-04-17T00:00:00',
           '2024-04-18T00:00:00', '2024-04-19T00:00:00', '2024-04-20T00:00:00',
           '2024-04-21T00:00:00', '2024-04-22T00:00:00', '2024-04-23T00:00:00',
           '2024-04-24T00:00:00', '2024-04-25T00:00:00', '2024-04-26T00:00:00',
           '2024-04-27T00:00:00', '2024-04-28T00:00:00', '2024-04-29T00:00:00',
           '2024-04-30T00:00:00', '2024-05-01T00:00:00', '2024-05-02T00:00:00',
           '2024-05-03T00:00:00', '2024-05-04T00:00:00', '2024-05-05T00:00:00',
           '2024-05-06T00:00:00', '2024-05-07T00:00:00', '2024-05-08T00:00:00',
           '2024-05-09T00:00:00', '2024-05-10T00:00:00', '2024-05-11T00:00:00',
           '2024-05-12T00:00:00', '2024-05-13T00:00:00', '2024-05-14T00:00:00',
           '2024-05-15T00:00:00', '2024-05-16T00:00:00', '2024-05-17T00:00:00',
           '2024-05-18T00:00:00', '2024-05-19T00:00:00', '2024-05-20T00:00:00',
           '2024-05-21T00:00:00', '2024-05-22T00:00:00', '2024-05-23T00:00:00',
           '2024-05-24T00:00:00', '2024-05-25T00:00:00', '2024-05-26T00:00:00',
           '2024-05-27T00:00:00', '2024-05-28T00:00:00', '2024-05-29T00:00:00',
           '2024-05-30T00:00:00', '2024-05-31T00:00:00', '2024-06-01T00:00:00',
           '2024-06-02T00:00:00', '2024-06-03T00:00:00', '2024-06-04T00:00:00',
           '2024-06-05T00:00:00', '2024-06-06T00:00:00', '2024-06-07T00:00:00',
           '2024-06-08T00:00:00', '2024-06-09T00:00:00', '2024-06-10T00:00:00',
           '2024-06-11T00:00:00', '2024-06-12T00:00:00', '2024-06-13T00:00:00',
           '2024-06-14T00:00:00', '2024-06-15T00:00:00', '2024-06-16T00:00:00',
           '2024-06-17T00:00:00', '2024-06-18T00:00:00', '2024-06-19T00:00:00',
           '2024-06-20T00:00:00', '2024-06-21T00:00:00', '2024-06-22T00:00:00',
           '2024-06-23T00:00:00', '2024-06-24T00:00:00', '2024-06-25T00:00:00'],
          dtype='datetime64[s]')


series


(series)


\<U8


'0::117' '0::691' ... '9::4'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['0::117', '0::691', '0::70', ..., '98::267', '99::589', '9::4'],
          shape=(1000,), dtype='<U8')


Data variables: (5)


sale_amount


(time, series)


float64


0.3 3.5 5.1 3.9 ... 32.7 13.7 1.7


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[ 0.3,  3.5,  5.1, ...,  3. ,  5.8,  1.5],
           [ 5. ,  4.9,  2.7, ...,  3. ,  5.9,  2.4],
           [ 6.6,  6.6,  6.6, ...,  3. ,  9.2,  6.8],
           ...,
           [ 6.2,  6.1,  5.1, ..., 33.2, 15. ,  4.2],
           [ 4.8,  6.4,  5. , ..., 24.1, 11.2,  2. ],
           [ 5.1,  6.2,  5.4, ..., 32.7, 13.7,  1.7]], shape=(90, 1000))


availability


(time, series)


float64


0.0 0.8421 0.9677 ... 0.899 0.6361


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.        , 0.84214031, 0.96773072, ..., 0.30402569, 0.72536543,
            0.30627405],
           [0.96853881, 0.95190416, 0.96853881, ..., 0.18424695, 0.58969883,
            0.40329786],
           [1.        , 0.96853881, 1.        , ..., 0.0914353 , 0.56669943,
            0.47482856],
           ...,
           [0.8736015 , 0.68842766, 0.55782077, ..., 0.91093046, 0.85230857,
            1.        ],
           [0.98133034, 0.98357871, 0.92597036, ..., 0.79323237, 0.89900624,
            0.73382722],
           [1.        , 1.        , 0.97870707, ..., 0.97870707, 0.89900624,
            0.6360862 ]], shape=(90, 1000))


discount_magnitude


(time, series)


float64


0.0 0.0 0.0 0.0 ... 0.0 0.0 0.059


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.   , 0.   , 0.   , ..., 0.   , 0.   , 0.118],
           [0.   , 0.   , 0.   , ..., 0.   , 0.   , 0.118],
           [0.   , 0.003, 0.   , ..., 0.   , 0.   , 0.118],
           ...,
           [0.   , 0.174, 0.   , ..., 0.   , 0.   , 0.084],
           [0.   , 0.172, 0.   , ..., 0.   , 0.   , 0.064],
           [0.   , 0.232, 0.   , ..., 0.   , 0.   , 0.059]], shape=(90, 1000))


activity_flag


(time, series)


float64


0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0., 0., 0., ..., 0., 0., 1.],
           [0., 0., 0., ..., 0., 0., 1.],
           [0., 0., 0., ..., 0., 0., 1.],
           ...,
           [0., 1., 0., ..., 0., 0., 0.],
           [0., 1., 0., ..., 0., 0., 0.],
           [0., 1., 1., ..., 0., 0., 0.]], shape=(90, 1000))


holiday_flag


(time, series)


float64


0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [1., 1., 1., ..., 1., 1., 1.],
           ...,
           [1., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], shape=(90, 1000))


## Per-series scaling

The panel mixes products that sell a handful of units per day with products that sell dozens. A single set of priors cannot cover both on the raw scale: an initial-level prior like \\\text{Normal}(1, 0.5)\\ would be far too tight for one series and far too wide for another. Dividing each series by its own mean daily sales puts every series on a common unit scale, where \\1\\ means "an average day for this product", and one prior vocabulary works across the whole panel (\\\text{Normal}(1, 0.5)\\ is then exactly the prior the model places on the scaled initial level). Two details matter:

- The scale is computed on the **training window only**. Computing it on the full series would leak the held-out level into the model input; the effect is mild on this dataset but catastrophic whenever the test window carries a trend.
- The scale is also the inverse map for evaluation: the model's draws live on the scaled axis, and we multiply them by `scale` to score and plot in original sale units.


    In [14]:


``` python
scale: xr.DataArray = panel_ds["sale_amount"].isel(time=slice(None, t_train)).mean("time")
y_scaled: xr.DataArray = panel_ds["sale_amount"] / scale
y_train: Float[Array, " t_train n_series"] = jnp.asarray(
    y_scaled.isel(time=slice(None, t_train)).to_numpy(), dtype=jnp.float32
)
y_train_original: Float[Array, " t_train n_series"] = jnp.asarray(
    panel_ds["sale_amount"].isel(time=slice(None, t_train)).to_numpy(), dtype=jnp.float32
)
y_test_original: Float[Array, " horizon n_series"] = jnp.asarray(
    panel_ds["sale_amount"].isel(time=slice(t_train, None)).to_numpy(), dtype=jnp.float32
)
scale_jax: Float[Array, " n_series"] = jnp.asarray(scale.to_numpy(), dtype=jnp.float32)

print(
    f"sale units per scaled unit: min {float(scale.min()):.2f} | "
    f"median {float(scale.median()):.2f} | max {float(scale.max()):.2f}"
)
```


    sale units per scaled unit: min 3.43 | median 4.95 | max 18.10


## The store index `series_to_store`

The covariate effects \\\beta\_{c,s}\\ are pooled by store: series from the same store share a store-level location and scale. To express that inside the model we need a lookup from the series axis to the store axis, and that is exactly what `series_to_store` is: an integer vector with one entry per series, `series_to_store[s] = m(s)`, aligned with the same sorted `series_ids` order as every pivot (both come from sorting by `unique_id`). We build it with scikit-learn's `LabelEncoder`, which consumes the polars column directly: `fit_transform` maps each store id to its position among the sorted unique ids, and the fitted `classes_` are exactly those sorted ids, so the same encoder yields both the integer index the model gathers with and the `store` coordinate labels the ArviZ export uses below. Inside the model, the advanced indexing `b_loc_store[:, series_to_store]` gathers the `(n_cov, n_stores)` store-level parameters into an `(n_cov, n_series)` array of per-series prior locations, a vectorized dictionary lookup. The jaxtyping annotation records the contract in the code: one integer per series.

The printout below carries a caveat worth keeping in mind: the panel spreads its \\1{,}000\\ series over \\520\\ stores with a median of *one* series per store. For a singleton store the store-level location is informed by a single series, so the hierarchy there acts as regularization toward the global hyperpriors rather than as cross-series pooling; the genuine pooling happens in the multi-series stores (up to eight series here). We revisit this when inspecting the fitted hierarchy.


    In [15]:


``` python
series_store_df = panel_df.select("unique_id", "store_id").unique().sort("unique_id")
store_encoder = LabelEncoder()
series_to_store: Int[Array, " n_series"] = jnp.asarray(
    store_encoder.fit_transform(series_store_df["store_id"]), dtype=jnp.int32
)
store_ids: list[int] = store_encoder.classes_.tolist()
n_stores = len(store_ids)
# The store id of each series as a named lookup table, for label-based gathers on
# the store dimension of the posterior (used in the hierarchy plot below).
series_store_da = xr.DataArray(
    series_store_df["store_id"].to_numpy(), dims=["series"], coords={"series": series_ids}
)

series_per_store = np.bincount(np.asarray(series_to_store), minlength=n_stores)
print(f"panel: {n_series} series map to {n_stores} stores")
print(
    f"series per store: min {series_per_store.min()} | "
    f"median {np.median(series_per_store):.0f} | max {series_per_store.max()}"
)
print(f"train: {t_train} days | test: {horizon} days")
```


    panel: 1000 series map to 520 stores
    series per store: min 1 | median 1 | max 8
    train: 76 days | test: 14 days


## A launch indicator

The sales plots below share a striking pattern: most series jump to a new level in late April, when this panel's flagship product ramps up across the assortment. That jump is a one-off structural event, not demand dynamics, so we give the model an explicit launch indicator: without a dedicated regressor for the step, any feature that happens to flip around the launch can absorb it and come out with a nonsense coefficient, putting the promotion effects in a tug-of-war with the level. Alternatives that do not work here: relying on the cleaned discount encoding alone (it shrinks the launch-aligned placeholder step, but a shrunken step is still a step a coefficient can latch onto), and per-series change-point detection (a thousand extra change points for an event the data show is panel-wide and sharply dated).

We therefore fix one shared launch date: the day with the largest week-over-week jump falls on \\2024\\-\\04\\-\\27\\ for more series than any other date, and the panel-mean daily sales step up by more than \\50\\\\ on exactly that day (both facts are printed below). The indicator is \\0\\ before the launch date and \\1\\ from that day onward, so over the forecast window it is constantly \\1\\: a known future covariate.


    In [16]:


``` python
window = 7
# rolling labels each window by its last day, so the difference at day t compares
# the week ending at t with the week ending at t - 7; shifting the argmax back by
# window - 1 days labels each series' largest jump by the later week's first day.
rolling_mean = panel_ds["sale_amount"].isel(time=slice(None, t_train)).rolling(time=window).mean()
weekly_jump = rolling_mean - rolling_mean.shift(time=window)
jump_date = weekly_jump.idxmax("time") - np.timedelta64(window - 1, "D")

ramp_date = np.datetime64("2024-04-27")
ramp_x = float(mdates.date2num(ramp_date))
panel_ds["post_ramp"] = (
    (panel_ds["time"] >= ramp_date).astype(np.float64).broadcast_like(panel_ds["sale_amount"])
)

panel_mean_sales = panel_ds["sale_amount"].mean("series")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates, panel_mean_sales, color="black", linewidth=2, label="panel-mean daily sales")
ax.axvline(ramp_x, color="C1", linestyle="--", linewidth=2, label="launch date (2024-04-27)")
ax.legend(loc="upper left")
ax.set(
    xlabel="date", ylabel="mean sale amount", title="Panel-mean daily sales and the launch date"
)
modal_dates, modal_counts = np.unique(jump_date.to_numpy(), return_counts=True)
share_modal = float((jump_date == ramp_date).mean())
share_cluster = float(
    ((jump_date >= ramp_date) & (jump_date <= np.datetime64("2024-05-01"))).mean()
)
step_ratio = float(
    panel_mean_sales.sel(time=ramp_date)
    / panel_mean_sales.sel(time=ramp_date - np.timedelta64(1, "D"))
)
print(
    f"modal largest-weekly-jump date: {modal_dates[modal_counts.argmax()].astype('datetime64[D]')}"
)
print(f"share of series with their largest weekly jump on 2024-04-27: {share_modal:.2f}")
print(f"share with it between 2024-04-27 and 2024-05-01: {share_cluster:.2f}")
print(f"panel-mean sales step on 2024-04-27: {step_ratio:.2f}x day over day")
```


    modal largest-weekly-jump date: 2024-04-27
    share of series with their largest weekly jump on 2024-04-27: 0.12
    share with it between 2024-04-27 and 2024-05-01: 0.34
    panel-mean sales step on 2024-04-27: 1.56x day over day


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-17-output-2.png" class="figure-img" width="1211" height="511" /></p>
</figure>


## The model inputs tensor

The model consumes five exogenous inputs per series and day: availability, the three promotion features, and the launch indicator. Rather than flattening them into a wide 2-D matrix (packing and unpacking by hand is exactly the kind of index bookkeeping that fails silently), we stack the five panel variables into a single 3-D `DataArray` with `to_dataarray`, whose leading `input` axis carries the variable names as coordinate labels. The `jax.numpy` tensor the model consumes is extracted at the boundary and keeps the same layout, with the stack order also named in the jaxtyping hint, `availability_discount_activity_holiday_ramp`, so it stays readable in every signature that touches the tensor. This layout is fully compatible with the package's shape convention, which only requires time at axis \\-2\\ with batch axes to the left. The train-forecast split stays a pure time slice, and the forecast horizon is still derived from shapes alone: training sees `covariates[:, :t_train, :]`, forecasting the full tensor. The model unpacks the inputs by plain indexing instead of a reshape. [to_datatree](../../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) stores covariates in `constant_data` as `(time, covariate_dim)` by default, but accepts this tensor as-is through its `covariate_dims` argument, which we use in the export section below.


    In [17]:


``` python
covariate_names = ["discount_magnitude", "activity_flag", "holiday_flag", "post_ramp"]
n_covariates = len(covariate_names)
input_names = ["availability", *covariate_names]

covariates_da: xr.DataArray = panel_ds[input_names].to_dataarray(dim="input")
covariates: Float[Array, " availability_discount_activity_holiday_ramp duration n_series"] = (
    jnp.asarray(covariates_da.transpose("input", "time", "series").to_numpy(), dtype=jnp.float32)
)
covariates_train: Float[Array, " availability_discount_activity_holiday_ramp t_train n_series"] = (
    covariates[:, :t_train, :]
)

print(f"model inputs: {covariates.shape} | train: {covariates_train.shape}")
```


    model inputs: (5, 90, 1000) | train: (5, 76, 1000)


Before modeling, let us look at a few of the series we are about to fit: the three largest by volume and the three with the most zero-availability days.


    In [18]:


``` python
series_to_plot = 10
n_focus = 3


def top_series(ranking: xr.DataArray, n: int) -> list[str]:
    """Labels of the ``n`` series with the largest value of a per-series ranking."""
    return ranking["series"].to_numpy()[np.argsort(-ranking.to_numpy())[:n]].tolist()


total_sales_per_series = panel_ds["sale_amount"].sum("time")
zero_availability_days = (panel_ds["availability"] == 0.0).sum("time")
top_volume_labels = top_series(total_sales_per_series, series_to_plot)
top_stockout_labels = top_series(zero_availability_days, series_to_plot)
plot_labels = list(dict.fromkeys([*top_volume_labels, *top_stockout_labels]))
# The EDA facets and the forecast plot show the full ~20-series set; the prior
# predictive check below uses a 6-series focus set (3 largest by volume,
# 3 heaviest stockouts) to stay readable.
focus_labels = list(dict.fromkeys([*top_volume_labels[:n_focus], *top_stockout_labels[:n_focus]]))
split_x = float(mdates.date2num(dates[t_train]))
# plot_lm concatenates x with the float predictions internally, so the faceted
# plots below use matplotlib date numbers plus a date formatter on each axis.
dates_num = np.asarray(mdates.date2num(dates))

fig, axes = plt.subplots(
    nrows=len(plot_labels),
    figsize=(12, 2.2 * len(plot_labels)),
    sharex=True,
    layout="constrained",
)
for ax, label in zip(axes, plot_labels, strict=True):
    (sales_line,) = ax.plot(
        dates, panel_ds["sale_amount"].sel(series=label), color="black", linewidth=2, label="sales"
    )
    split_line = ax.axvline(
        split_x, color="C3", linestyle="--", linewidth=1, label="train-test split"
    )
    ramp_line = ax.axvline(ramp_x, color="C1", linestyle=":", linewidth=1.5, label="launch date")
    ax.set(title=label, ylabel="sale amount")
    ax_twin = ax.twinx()
    (availability_line,) = ax_twin.plot(
        dates,
        panel_ds["availability"].sel(series=label),
        color="red",
        linewidth=1.5,
        label="availability",
    )
    ax_twin.grid(False)
    ax_twin.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax_twin.set(ylabel="availability", ylim=(0, 1.05))
axes[0].legend(
    handles=[sales_line, split_line, ramp_line, availability_line], loc="upper left", fontsize=10
)
fig.supxlabel("date")
fig.suptitle("Sales and sales-weighted availability", fontsize=18, fontweight="bold");
```


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-19-output-1.png" class="figure-img" width="1211" height="4411" /></p>
</figure>


The stockout-heavy series make the modeling problem vivid: sales collapse toward zero when availability drops, but not exactly to zero, and they snap back as soon as stock returns. The dotted launch line confirms the panel-wide event: series after series either starts selling or doubles its level right at \\2024\\-\\04\\-\\27\\. It is also no coincidence that nearly every series shown is the same product (`267`) in a different store: the volume and stockout rankings alike are dominated by the flagship product whose launch shapes this panel.

The same view for the promotion covariates completes the picture: the discount magnitude on the right axis, with promotion-activity and holiday days shaded in the background.


    In [19]:


``` python
fig, axes = plt.subplots(
    nrows=len(plot_labels),
    figsize=(12, 2.2 * len(plot_labels)),
    sharex=True,
    layout="constrained",
)
discount_max = float(panel_ds["discount_magnitude"].max())
for ax, label in zip(axes, plot_labels, strict=True):
    (sales_line,) = ax.plot(
        dates, panel_ds["sale_amount"].sel(series=label), color="black", linewidth=2, label="sales"
    )
    split_line = ax.axvline(
        split_x, color="C3", linestyle="--", linewidth=1, label="train-test split"
    )
    activity_span = ax.fill_between(
        dates,
        0,
        1,
        where=(panel_ds["activity_flag"].sel(series=label) > 0.5).to_numpy(),
        transform=ax.get_xaxis_transform(),
        color="C0",
        alpha=0.25,
        linewidth=0,
        label="promotion activity day",
    )
    holiday_span = ax.fill_between(
        dates,
        0,
        1,
        where=(panel_ds["holiday_flag"].sel(series=label) > 0.5).to_numpy(),
        transform=ax.get_xaxis_transform(),
        color="C4",
        alpha=0.35,
        linewidth=0,
        label="holiday",
    )
    ax.set(title=label, ylabel="sale amount")
    ax_twin = ax.twinx()
    (discount_line,) = ax_twin.plot(
        dates,
        panel_ds["discount_magnitude"].sel(series=label),
        color="C1",
        alpha=0.8,
        linewidth=1,
        label="discount magnitude",
    )
    ax_twin.grid(False)
    ax_twin.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax_twin.set(ylabel="discount", ylim=(0, 1.1 * discount_max))
axes[0].legend(
    handles=[sales_line, split_line, discount_line, activity_span, holiday_span],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.45),
    ncol=5,
    fontsize=12,
)
fig.supxlabel("date")
fig.suptitle("Sales and promotion covariates", fontsize=18, fontweight="bold");
```


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-20-output-1.png" class="figure-img" width="1211" height="4411" /></p>
</figure>


Two patterns jump out. The holiday flag repeats weekly (weekends plus a solid block around the May Day week), and several series show local sales spikes on those shaded days; promotion activity is rarer and highly series-specific, absent for most of these series but covering most of the window for `409::309` in the last panel. The discount line also shows the placeholder cleanup at work: for many of the highest-volume series the raw `discount` column is zero on most days (the placeholder flagged in the EDA, concentrated in exactly this panel), so their cleaned magnitude sits at zero except on genuinely priced promotions, while series like `409::309` keep their real time-varying discounts. This heterogeneity in feature quality is one more argument for pooling the covariate effects by store rather than fitting one global discount effect: where the feature is quiet the coefficient is weakly identified and shrinks toward its store-level prior, and where the feature is informative it can act.


# Model specification

The model is a panel state space model on the scaled sales, with five components per series \\s\\:

- a random-walk local level for slow demand shifts,
- a damped AR(1) trend slope that carries the recent drift into the forecast window,
- a zero-sum weekly seasonal profile,
- promotion, calendar, and launch effects pooled hierarchically by store,
- a multiplicative availability factor with a learned floor, which also scales the observation noise.

\\ \begin{align\*} y\_{t,s} &\sim \text{Normal}\left(f\_{t,s} \\ \mu\_{t,s},\\ f\_{t,s} \left(\sigma_s + \lambda_s \\ \text{softplus}(\ell\_{t,s})\right) + \sigma_0\right) \\ \mu\_{t,s} &= \ell\_{t,s} + \gamma\_{d(t),s} + \sum\_{c=1}^{4} \beta\_{c,s} \\ x\_{c,t,s} \\ \ell\_{t,s} &= \ell\_{0,s} + \sum\_{u \le t} \left(\varepsilon\_{u,s} + \delta\_{u,s}\right), \qquad \varepsilon\_{u,s} \sim \text{Normal}(0, \tau_s) \\ \delta\_{u,s} &= \phi^{\text{trend}}\_s \\ \delta\_{u-1,s} + \eta\_{u,s}, \qquad \eta\_{u,s} \sim \text{Normal}\left(0, \tau^{\text{trend}}\_s\right), \quad \delta\_{0,s} = 0 \\ f\_{t,s} &= \phi_s + (1 - \phi_s) \\ \frac{1 - e^{-b_s a\_{t,s}}}{1 - e^{-b_s}} \\ \beta\_{c,s} &\sim \text{Normal}\left(\mu^{\text{store}}\_{c,\\m(s)},\\ \sigma^{\text{store}}\_{c,\\m(s)}\right) \end{align\*} \\

where \\d(t)\\ is the day of week, \\m(s)\\ the store of series \\s\\, \\a\_{t,s}\\ the sales-weighted availability, \\x\_{c,t,s}\\ the four regression features (discount magnitude, promotion activity, holiday, and the launch indicator), \\\lambda_s\\ the loading of the level-dependent noise component, and \\\sigma_0 = 0.02\\ a small constant basal noise; the last two are discussed with the noise scale below. The remaining priors, all on the scaled axis where \\1\\ is an average day for the series:

\\ \begin{align\*} \ell\_{0,s} &\sim \text{Normal}(1, 0.5), \quad \tau_s \sim \text{LogNormal}(-3, 1), \quad \phi^{\text{trend}}\_s \sim \text{Beta}(8, 2), \quad \tau^{\text{trend}}\_s \sim \text{LogNormal}(-4, 1) \\ \gamma\_{\cdot,s} &\sim \text{ZeroSumNormal}(\sigma\_\gamma, 7), \quad \sigma\_\gamma \sim \text{HalfNormal}(0.2) \\ \mu^{\text{store}}\_{c,m} &\sim \text{Normal}(0, 0.5), \quad \sigma^{\text{store}}\_{c,m} \sim \text{HalfNormal}(0.3) \\ \phi_s &\sim \text{Beta}(2, 18), \quad b_s \sim \text{LogNormal}(1, 0.5) \\ \sigma_s &\sim \text{HalfNormal}(0.5), \quad \lambda_s \sim \text{HalfNormal}(0.2) \end{align\*} \\

The random-walk drift \\\varepsilon\\ and the coefficients \\\beta\\ use `LocScaleReparam` with learned centeredness parameters (each a global \\\text{Uniform}(0, 1)\\ latent), as in the other hierarchical examples. The sections below motivate the trend, dynamics, and availability-factor priors in detail.


## The damped trend

We give each series a damped AR(1) slope on top of the random-walk level because a \\14\\-day forecast needs momentum: this panel keeps drifting upward through the test window, and only a trend state can carry that drift past the last observed day. In the state-space scan the forecast is seeded by the final in-sample slope, which then decays geometrically at rate \\\phi^{\text{trend}}\_s \< 1\\ while its uncertainty keeps growing, so the forecast inherits the current momentum without betting on it indefinitely. The interval-diagnostics section below quantifies what this buys on the holdout.

Alternatives that do not work here:

- **A pure random-walk level with no slope**, meaning the forecast freezes at the last fitted level. On a drifting panel like this one the interval misses then concentrate *above* the bands and grow with the horizon while the forecast fan barely widens, exactly the miscalibration signature the interval diagnostics below are built to detect.
- **An undamped slope** (an integrated random walk), meaning the last local trend is extrapolated as a straight line indefinitely. Fresh-retail momentum is short-lived (a promotion tail, a post-launch settling), so the straight line overshoots at precisely the long horizons where intervals are most fragile; the damping relaxes the trend toward zero within the horizon instead.


## The availability factor

The factor \\f\_{t,s}\\ deserves a close look, because its three ingredients each fix a concrete failure mode:

- **The floor \\\phi_s \in (0, 1)\\.** At zero recorded availability the factor equals \\\phi_s\\, not zero, so the \\15\\\\ of flagged stockout days with positive sales are explained by a small expected sale instead of blowing up the likelihood. The prior \\\phi_s \sim \text{Beta}(2, 18)\\ (mean \\0.1\\) is informed by the empirical floor of about \\0.04\\ measured above, while staying wide enough for series with sloppier labels.
- **The saturating link \\1 - e^{-b_s a}\\.** This is the classic random-encounter (reach) curve: if purchase attempts arrive through the day as a Poisson process with intensity \\b_s\\, and a share \\a\\ of the (sales-weighted) selling time is in stock, the probability that at least one attempt lands while the product is on the shelf is exactly \\1 - e^{-b_s a}\\. It matches the concave shape of the empirical curve, and compared to alternatives such as \\\tanh\\ it is cheaper (one exponential), has a smooth monotone gradient in \\b_s\\, and gives \\b_s\\ a direct interpretation as purchase-opportunity intensity.
- **The normalization by \\1 - e^{-b_s}\\.** Without it, the factor at full availability is \\1 - e^{-b_s} \< 1\\, and the model can trade the factor's overall scale against the level \\\ell\_{t,s}\\ (multiply one, divide the other), leaving both non-identified. Anchoring \\f\_{t,s} = 1\\ at \\a = 1\\ removes that degeneracy: the level is the demand at full availability, \\\phi_s\\ is exactly the share of demand still sold on a fully flagged-out day, and \\b_s\\ only controls the curvature. Numerically we compute the ratio with `expm1`, which avoids the catastrophic cancellation in \\1 - e^{-b_s}\\ for small \\b_s\\, where the naive expression degrades toward \\0/0\\.


    In [20]:


``` python
fig, ax = plt.subplots()
pz.Beta(2, 18).plot_pdf(color="C0", ax=ax)
ax.axvline(
    empirical_floor,
    color="C3",
    linestyle="--",
    label=f"empirical floor ({empirical_floor:.2f})",
)
ax.legend(loc="upper right")
ax.set(xlabel=r"floor $\phi_s$", ylabel="density", title="Floor prior");
```


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-21-output-1.png" class="figure-img" width="976" height="636" /></p>
</figure>


    In [21]:


``` python
a_grid = jnp.linspace(0, 1, 200)  # availability from 0 to 1
b_vals = [0.5, 1.5, 5.0, 12.0]  # some example b_s values

fig, ax = plt.subplots()
for b in b_vals:
    f = (1 - jnp.exp(-b * a_grid)) / (1 - jnp.exp(-b))
    ax.plot(a_grid, f, label=f"$b_s={b}$")
fig.text(0.7, 0.42, r"$\frac{1 - e^{-b_s a_{t,s}}}{1 - e^{-b_s}}$", fontsize=36)
ax.legend(loc="lower right")
ax.set(
    title="Saturating availability factor vs $a_{t,s}$",
    xlabel="Availability $a_{t,s}$",
    ylabel="Availability factor",
);
```


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-22-output-1.png" class="figure-img" width="1211" height="711" /></p>
</figure>


## Priors for the level and trend dynamics

Three priors govern how much the level is allowed to move, and they are worth choosing deliberately. All live on the scaled axis, where \\1\\ is an average day for the series:

- \\\tau_s \sim \text{LogNormal}(-3, 1)\\, the random-walk drift scale: median \\\approx 0.05\\, so a typical series may shift its level by around \\5\\\\ of an average day per step, with the long right tail leaving room for jumpier series.
- \\\phi^{\text{trend}}\_s \sim \text{Beta}(8, 2)\\, the trend damping: mean \\0.8\\, so a slope shock loses about half its size in three days (\\0.8^3 \approx 0.51\\) and the extrapolated trend flattens within the \\14\\-day horizon instead of running away.
- \\\tau^{\text{trend}}\_s \sim \text{LogNormal}(-4, 1)\\, the slope innovation scale: median \\\approx 0.018\\, deliberately well below the drift and observation scales, so the slope only accumulates persistent day-over-day signals and cannot chase daily noise.


    In [22]:


``` python
prior_panels = [
    (pz.LogNormal(-3, 1), "C0", "LogNormal(-3, 1)", "Drift scale prior", r"$\tau_s$", (0.0, 0.4)),
    (pz.Beta(8, 2), "C1", "Beta(8, 2)", "Trend damping prior", r"$\phi^{\mathrm{trend}}_s$", None),
    (
        pz.LogNormal(-4, 1),
        "C2",
        "LogNormal(-4, 1)",
        "Trend innovation prior",
        r"$\tau^{\mathrm{trend}}_s$",
        (0.0, 0.15),
    ),
]
fig, axes = plt.subplots(ncols=3, figsize=(15, 4.5), layout="constrained")
for ax, (prior, color, name, title, xlabel, xlim) in zip(axes, prior_panels, strict=True):
    prior.plot_pdf(color=color, legend=None, ax=ax)
    pdf_line = next(line for line in ax.lines if line.get_color() == color)
    pdf_line.set_label(name)
    ax.legend()
    ax.set(xlabel=xlabel, ylabel="density", title=title)
    if xlim is not None:
        ax.set_xlim(*xlim)
fig.suptitle("Priors for the level and trend dynamics", fontsize=18, fontweight="bold");
```


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-23-output-1.png" class="figure-img" width="1511" height="461" /></p>
</figure>


Finally, the noise scale is \\f\_{t,s} \left(\sigma_s + \lambda_s \\ \text{softplus}(\ell\_{t,s})\right) + \sigma_0\\. It has three parts: a per-series base scale \\\sigma_s\\, a level-dependent component \\\lambda_s \\ \text{softplus}(\ell\_{t,s})\\, sampled as `noise_loading` in the code (busier days are noisier in absolute terms, and its coverage payoff is quantified in the evaluation section), and the availability factor \\f\_{t,s}\\ shrinking the spread on stockout days, where sales are pinned near zero. The remaining piece is a small **constant** basal term \\\sigma_0 = 0.02\\ on the scaled axis, which keeps the scale bounded away from zero. Three design questions hide in this one constant:

- **Why not a learned basal term?** Many series sell exactly zero on their stockout days, where the mean is also pinned near zero. A Normal density at a perfectly fit point grows without bound as its scale shrinks, so the ELBO rewards collapsing the total noise scale at those observations; with a learned basal term the collapse runs away and the optimization hits `NaN` mid-run (the first non-finite ELBO appears around step \\6{,}000\\ on this panel). A constant cannot collapse.
- **Why not a tiny epsilon like \\10^{-6}\\?** The constant is not there to avoid division by zero; it must remove the *reward* for collapse. With \\\sigma_0 = 10^{-6}\\ the density at an exactly fit zero can still contribute \\\log\left(1 / (\sigma_0 \sqrt{2\pi})\right) \approx 12.9\\ per observation, and such a fit banks roughly a thousand nats of ELBO from these spikes while every predictive metric stays identical to the \\0.02\\ fit: the "improvement" is purely the degenerate optimum being exploited, and stability is then at the mercy of the learning-rate schedule (the learned-term variant diverged through exactly this mechanism).
- **Why \\0.02\\ specifically?** It sits at the data's resolution: one physical sale unit is between \\0.06\\ and \\0.29\\ on the per-series scaled axis, so a basal noise of \\0.02\\ is below measurement granularity and cannot distort any interval the data could support. Fits with \\\sigma_0 \in \\0.01, 0.02, 0.05\\\\ give the same CRPS and coverage to within noise.


    In [23]:


``` python
def availability_factor(
    availability: Float[np.ndarray | Array, " ..."],
    b_avail: Float[np.ndarray | Array, " ..."],
    floor: Float[np.ndarray | Array, " ..."],
) -> Float[np.ndarray | Array, " ..."]:
    """Floored, normalized saturating availability factor.

    The model-specification curve
    ``floor + (1 - floor) * expm1(-b_avail * availability) / expm1(-b_avail)``:
    ``floor`` at zero availability, exactly ``1`` at full availability, with the
    curvature set by ``b_avail``. Defined once and shared by the model and the
    prior/posterior diagnostic cells below, so the plotted curves can never
    drift from what the model computes. Inputs broadcast together per NumPy
    rules. NumPy inputs are accepted (``xarray.apply_ufunc`` passes them in
    the posterior diagnostic below) and computed with ``jax.numpy``.

    Parameters
    ----------
    availability
        Sales-weighted availability in ``[0, 1]``.
    b_avail
        Saturation rate (the purchase-opportunity intensity).
    floor
        The factor at zero availability, in ``(0, 1)``.

    Returns
    -------
    Array
        The multiplicative availability factor.
    """
    saturation = jnp.expm1(-b_avail * availability) / jnp.expm1(-b_avail)
    return floor + (1.0 - floor) * saturation


class FreshRetailModel(ForecastingModel):
    """Damped-trend hierarchical panel model with a floored availability factor.

    Parameters
    ----------
    series_to_store
        Integer index mapping each series to its store, shape ``(n_series,)``.
    n_stores
        Number of distinct stores in the panel.
    n_series
        Number of store-product series (the trailing observation axis).
    n_cov
        Number of regression features (excluding availability).
    """

    def __init__(
        self,
        series_to_store: Int[Array, " n_series"],
        n_stores: int,
        n_series: int,
        n_cov: int,
    ) -> None:
        super().__init__()
        self.series_to_store = series_to_store
        self.n_stores = n_stores
        self.n_series = n_series
        self.n_cov = n_cov

    def model(
        self,
        zero_data: Float[Array, " duration n_series"] | None,
        covariates: Float[Array, " availability_discount_activity_holiday_ramp duration n_series"],
    ) -> None:
        """Sample the joint model (the package calls this for training and forecasting)."""
        duration = covariates.shape[-2]
        # The inputs tensor keeps time at axis -2 (the package-wide convention) with
        # the stacked inputs as a leading batch axis, named explicitly in the
        # signature: availability first, then the regression features, already
        # shaped so that `features * b[:, None, :]` broadcasts against the
        # (n_cov, n_series) coefficients. The `assert isinstance` lines are
        # jaxtyping's runtime shape guards: they share the dimension memo with the
        # signature (so `duration` and `n_series` are already bound by
        # `covariates`); plain annotated assignments would NOT be checked at runtime.
        availability = covariates[0]
        assert isinstance(availability, Float[Array, " duration n_series"])  # ty: ignore[invalid-argument-type]
        features = covariates[1:]
        assert isinstance(features, Float[Array, " n_cov duration n_series"])  # ty: ignore[invalid-argument-type]

        # Global hyperpriors. The seasonal scale must be a scalar because
        # ZeroSumNormal cannot broadcast a per-series scale over the plate.
        centered_drift = numpyro.sample("centered_drift", dist.Uniform(0.0, 1.0))
        centered_b = numpyro.sample("centered_b", dist.Uniform(0.0, 1.0))
        seasonal_scale = numpyro.sample("seasonal_scale", dist.HalfNormal(0.2))

        with numpyro.plate("store", self.n_stores, dim=-1):
            with numpyro.plate("covariate", self.n_cov, dim=-2):
                b_loc_store = cast("Array", numpyro.sample("b_loc_store", dist.Normal(0.0, 0.5)))
                b_scale_store = cast(
                    "Array", numpyro.sample("b_scale_store", dist.HalfNormal(0.3))
                )

        with numpyro.plate("series", self.n_series, dim=-1):
            drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-3.0, 1.0))
            phi_trend = cast("Array", numpyro.sample("phi_trend", dist.Beta(8.0, 2.0)))
            tau_trend = cast("Array", numpyro.sample("tau_trend", dist.LogNormal(-4.0, 1.0)))
            init_level = cast("Array", numpyro.sample("init_level", dist.Normal(1.0, 0.5)))
            b_avail = cast("Array", numpyro.sample("b_avail", dist.LogNormal(1.0, 0.5)))
            floor = cast("Array", numpyro.sample("floor", dist.Beta(2.0, 18.0)))
            sigma = cast("Array", numpyro.sample("sigma", dist.HalfNormal(0.5)))
            noise_loading = cast("Array", numpyro.sample("noise_loading", dist.HalfNormal(0.2)))
            seasonal = cast(
                "Array",
                numpyro.sample("seasonal", dist.ZeroSumNormal(seasonal_scale, event_shape=(7,))),
            )
            with numpyro.plate("covariate", self.n_cov, dim=-2):
                with handlers.reparam(config={"b": LocScaleReparam(centered=centered_b)}):
                    b = cast(
                        "Array",
                        numpyro.sample(
                            "b",
                            dist.Normal(
                                b_loc_store[:, self.series_to_store],
                                b_scale_store[:, self.series_to_store],
                            ),
                        ),
                    )
            # time_series opens its own time plate at dim=-2, so the covariate
            # plate above must already be closed here.
            drift = self.time_series(
                "drift",
                lambda: dist.Normal(0.0, drift_scale),
                reparam=LocScaleReparam(centered=centered_drift),
            )

        # The damped AR(1) slope is a Markov latent: markov_time_series scans over
        # time (and must be called outside the series plate; the per-series
        # parameters enter through the closure), seeds the forecast scan with the
        # final in-sample slope, and returns the latent in the package layout
        # (duration, n_series).
        def slope_transition(
            carry: Array, _: Array | None
        ) -> tuple[dist.Distribution, Callable[[Array], Array]]:
            return dist.Normal(phi_trend * carry, tau_trend), lambda value: value

        slope = self.markov_time_series("slope", jnp.zeros(self.n_series), slope_transition)
        level = init_level + jnp.cumsum(drift, axis=-2) + jnp.cumsum(slope, axis=-2)
        seasonal_rep = periodic_repeat(seasonal.T, duration, axis=-2)
        covariates_contribution = (features * b[:, None, :]).sum(axis=0)
        factor = cast("Array", availability_factor(availability, b_avail, floor))
        mu = factor * (level + seasonal_rep + covariates_contribution)
        # The constant basal noise bounds the likelihood at exact-zero sales: a
        # learned basal term collapses there and NaNs the optimization, and a tiny
        # epsilon (e.g. 1e-6) leaves that degenerate optimum in play; 0.02 sits just
        # below one physical sale unit on the scaled axis (see the markdown above).
        sigma_t = factor * (sigma + noise_loading * jax.nn.softplus(level)) + 0.02
        self.predict_glm(lambda m: dist.Normal(m, sigma_t), mu)


model = FreshRetailModel(
    series_to_store=series_to_store,
    n_stores=n_stores,
    n_series=n_series,
    n_cov=n_covariates,
)
```


Let us visualize the model structure:


    In [24]:


``` python
numpyro.render_model(
    model,
    model_args=(covariates_train, y_train),
    render_distributions=True,
)
```


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-25-output-1.svg" class="img-fluid figure-img" /></p>
</figure>


# Prior predictive checks

First the factor itself: the priors on \\\phi_s\\ and \\b_s\\ should cover both gentle and sharp saturation, with the floor concentrated near the empirical value but not glued to it.

The plots in this and the following sections lean on the package helper [predictions_to_datatree](../../../reference/convert.predictions_to_datatree.md#numpyro_forecast.convert.predictions_to_datatree): it packs raw prediction-draw arrays (possibly rescaled, clipped, or subset) into the DataTree layout that `az.plot_lm` needs for per-series faceting, with the independent variable broadcast per series in `constant_data`. It complements rather than duplicates [to_datatree](../../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree), which is fit-centric (it draws its own predictive from a fit and stores covariates, not a faceting grid). On our side of that boundary, every predictive ensemble gets wrapped in a `DataArray` with named `time` and `series` coordinates (the small `draws_to_da` helper below), so subsetting for a plot is a label-based `.sel(series=...)` rather than a positional index expression.

Every banded plot shares two styling conventions, set once here. The `hdi_label` helper formats the legend entries from the probability itself (the `\%` escape is what mathtext requires), and each `az.plot_lm` call maps the band transparency explicitly onto the `prob` dimension via `aes={"alpha": ["prob"]}` with the `hdi_alphas` values below, so the narrower \\50\\\\ band sits more opaque on top of the lighter \\94\\\\ band in every figure.


    In [25]:


``` python
def hdi_label(prob: float, prefix: str = "") -> str:
    r"""Legend label for an HDI band, e.g. ``$94\%$ HDI``."""
    percent = f"{prob:.0%}".replace("%", r"\%")
    return f"{prefix}${percent}$ HDI"


hdi_probs = (0.5, 0.94)
hdi_alphas = [0.6, 0.3]
```


    In [26]:


``` python
availability_grid = np.linspace(0.0, 1.0, 101)
rng_key, key_floor, key_b = random.split(rng_key, 3)
floor_prior = dist.Beta(2.0, 18.0).sample(key_floor, (500,))
b_prior = dist.LogNormal(1.0, 0.5).sample(key_b, (500,))
grid_jax = jnp.asarray(availability_grid, dtype=jnp.float32)
factor_prior = availability_factor(grid_jax, b_prior[:, None], floor_prior[:, None])

pc = az.plot_lm(
    predictions_to_datatree(
        np.asarray(factor_prior)[:, :, None],
        availability_grid,
        ["prior factor"],
        group="prior_predictive",
    ),
    y="obs",
    x="t",
    plot_dim="time",
    group="prior_predictive",
    ci_kind="hdi",
    ci_prob=hdi_probs,
    smooth=True,
    visuals={
        "ci_band": {"color": "C0"},
        "observed_scatter": False,
        "pe_line": False,
        "xlabel": False,
        "ylabel": False,
    },
    aes={"alpha": ["prob"]},
    alpha=hdi_alphas,
    figure_kwargs={"figsize": (10, 6)},
)
ax = pc.get_target("t", {"series": "prior factor"})
ax.plot(0.0, empirical_floor, "D", color="C3", markersize=8, label="empirical floor")
bands = pc.viz["ci_band"]["t"].sel(series="prior factor")
for prob in (0.94, 0.5):
    bands.sel(prob=prob).item().set_label(hdi_label(prob))
ax.legend(loc="upper left")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
ax.set(
    xlabel="sales-weighted availability",
    ylabel="availability factor",
    title="Prior availability factor curves",
);
```


    /Users/juanitorduz/Documents/numpyro_forecast/.venv/lib/python3.14/site-packages/arviz_plots/plots/lm_plot.py:360: UserWarning: When multiple credible intervals are plotted, it is recommended to map 'alpha' aesthetic to 'prob' dimension to differentiate between intervals.
      warnings.warn(


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-27-output-2.png" class="figure-img" width="1011" height="611" /></p>
</figure>


Next the full prior predictive on the training window for our six focus series, with the scaled observations overlaid. We want wide but sane bands on the unit scale of the normalized data. The bands also dip below zero: a Normal likelihood on the scaled axis pays for its simplicity with prior (and posterior) mass on negative sales, a compromise we accept here and revisit in the next steps with a strictly positive observation model.


    In [27]:


``` python
def draws_to_da(
    draws: Float[np.ndarray | Array, " sample time n_series"],
    time_values: np.ndarray,
) -> xr.DataArray:
    """Wrap prediction draws in a DataArray with named time and series coordinates."""
    return xr.DataArray(
        np.asarray(draws),
        dims=["sample", "time", "series"],
        coords={"time": time_values, "series": series_ids},
    )


rng_key, key_prior = random.split(rng_key)
prior_obs = Predictive(model, num_samples=500, return_sites=["obs"])(key_prior, covariates_train)[
    "obs"
]
prior_obs_da = draws_to_da(prior_obs, dates[:t_train])

pc = az.plot_lm(
    predictions_to_datatree(
        prior_obs_da.sel(series=focus_labels).to_numpy(),
        dates_num[:t_train],
        focus_labels,
        group="prior_predictive",
    ),
    y="obs",
    x="t",
    plot_dim="time",
    group="prior_predictive",
    ci_kind="hdi",
    ci_prob=hdi_probs,
    smooth=False,
    col_wrap=2,
    visuals={
        "ci_band": {"color": "C0"},
        "observed_scatter": False,
        "pe_line": False,
        "xlabel": False,
        "ylabel": False,
    },
    aes={"alpha": ["prob"]},
    alpha=hdi_alphas,
    figure_kwargs={"figsize": (15, 9)},
)
truth_da = (
    y_scaled.isel(time=slice(None, t_train))
    .sel(series=focus_labels)
    .assign_coords(time=dates_num[:t_train])
    .rename("t")
)
x_da = xr.DataArray(dates_num[:t_train], dims=["time"], coords={"time": dates_num[:t_train]})
pc.map(
    az.visuals.line_xy,
    "truth",
    data=truth_da,
    x=x_da,
    ignore_aes=pc.aes_set,
    color="black",
    lw=1,
)
for label in focus_labels:
    ax = pc.get_target("t", {"series": label})
    ax.set_title(label, fontsize=11)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
ax0 = pc.get_target("t", {"series": focus_labels[-1]})
bands = pc.viz["ci_band"]["t"].sel(series=focus_labels[-1])
band_handles = []
for prob in (0.94, 0.5):
    band = bands.sel(prob=prob).item()
    band.set_label(hdi_label(prob))
    band_handles.append(band)
truth_line = pc.viz["truth"]["t"].sel(series=focus_labels[-1]).item()
truth_line.set_label("scaled training data")
ax0.legend(handles=[*band_handles, truth_line], loc="upper left", fontsize=9)
fig = pc.viz["figure"].item()
fig.supxlabel("date")
fig.supylabel("scaled sales")
fig.suptitle("Prior predictive check", fontsize=16, fontweight="bold", y=1.02);
```


    /Users/juanitorduz/Documents/numpyro_forecast/.venv/lib/python3.14/site-packages/arviz_plots/plots/lm_plot.py:360: UserWarning: When multiple credible intervals are plotted, it is recommended to map 'alpha' aesthetic to 'prob' dimension to differentiate between intervals.
      warnings.warn(


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-28-output-2.png" class="figure-img" width="1511" height="933" /></p>
</figure>


# Inference with SVI

We fit with [fit_svi](../../../reference/functional.svi.fit_svi.md#numpyro_forecast.functional.svi.fit_svi) and its default `AutoNormal` guide. Instead of a fixed learning rate we pass a custom `optax` optimizer, the one-cycle Adam schedule chained with `reduce_on_plateau`, which converges noticeably better on this panel (the same recipe as in the [inference methods comparison](inference_methods_comparison.md) example).

We set `progress_bar=False`, and not only because the scanned update loop compiles to a single `lax.scan` that finishes all \\60{,}000\\ steps in a few seconds on CPU. The step-by-step execution path behind the progress bar compiles to slightly different floating-point arithmetic, and on this panel that tiny perturbation is enough to steer the optimizer into a distinctly worse ELBO optimum (the evaluation section returns to this sensitivity). The scanned path is both the fast and the well-behaved one here.


    In [28]:


``` python
%%time

num_steps = 60_000

scheduler = optax.linear_onecycle_schedule(
    transition_steps=num_steps,
    peak_value=0.001,
    pct_start=0.3,
    pct_final=0.85,
    div_factor=2,
    final_div_factor=3,
)

optimizer = optax.chain(
    optax.adam(learning_rate=scheduler),
    optax.contrib.reduce_on_plateau(
        factor=0.8,
        patience=20,
        accumulation_size=100,
    ),
)

rng_key, key_fit = random.split(rng_key)
svi_fit = fit_svi(
    key_fit,
    model,
    y_train,
    covariates_train,
    optim=optimizer,
    num_steps=num_steps,
    progress_bar=False,
)
```


    CPU times: user 12 s, sys: 556 ms, total: 12.6 s
    Wall time: 6.45 s


    In [29]:


``` python
%%time

fig, ax = plt.subplots()
ax.plot(svi_fit.losses, color="C0", label="ELBO loss")
ax.legend(loc="upper right")
ax.set(yscale="log", xlabel="SVI step", ylabel="loss", title="SVI ELBO loss");
```


    CPU times: user 9min 31s, sys: 6min 51s, total: 16min 23s
    Wall time: 3min 17s


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-30-output-2.png" class="figure-img" width="1211" height="711" /></p>
</figure>


# Export to an ArviZ DataTree

A single [to_datatree](../../../reference/convert.to_datatree.md#numpyro_forecast.convert.to_datatree) call wraps everything: it draws the posterior from the guide, runs the in-sample posterior predictive, and, because the covariates extend \\14\\ days past the training data, also generates the forecast and stores it in the `predictions` group. We label every dimension so downstream selections read naturally; in particular, `covariate_dims` tells the export the covariates are an `(input, time, series)` tensor, so `constant_data` keeps the layout the model consumes instead of a flattened matrix, with the five inputs named on the `input` coordinate.


    In [30]:


``` python
first_weekday = int(dates_series.dt.weekday()[0])
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_labels = [day_names[(first_weekday - 1 + offset) % 7] for offset in range(7)]

rng_key, key_tree = random.split(rng_key)
tree = to_datatree(
    key_tree,
    svi_fit,
    model,
    y_train,
    covariates,
    num_predictive_samples=1_000,
    time_coord=list(dates),
    covariate_dims=["input", "time", "series"],
    coords={
        "series": series_ids,
        "obs_dim": series_ids,
        "store": store_ids,
        "covariate": covariate_names,
        "input": input_names,
        "day_of_week": dow_labels,
    },
    posterior_dims={
        "drift": ["time", "series"],
        "drift_decentered": ["time", "series"],
        "slope": ["time", "series"],
        "seasonal": ["series", "day_of_week"],
        "b": ["covariate", "series"],
        "b_decentered": ["covariate", "series"],
        "b_loc_store": ["covariate", "store"],
        "b_scale_store": ["covariate", "store"],
        "drift_scale": ["series"],
        "phi_trend": ["series"],
        "tau_trend": ["series"],
        "init_level": ["series"],
        "b_avail": ["series"],
        "floor": ["series"],
        "sigma": ["series"],
        "noise_loading": ["series"],
    },
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
│       Dimensions:           (chain: 1, draw: 1000, covariate: 4, series: 1000,
│                              store: 520, time: 76, day_of_week: 7)
│       Coordinates:
│         * chain             (chain) int64 8B 0
│         * draw              (draw) int64 8kB 0 1 2 3 4 5 6 ... 994 995 996 997 998 999
│         * covariate         (covariate) <U18 288B 'discount_magnitude' ... 'post_ramp'
│         * series            (series) <U8 32kB '0::117' '0::691' ... '99::589' '9::4'
│         * store             (store) int64 4kB 0 1 2 3 4 5 ... 891 892 893 894 896 897
│         * time              (time) datetime64[s] 608B 2024-03-28 ... 2024-06-11
│         * day_of_week       (day_of_week) <U3 84B 'Thu' 'Fri' 'Sat' ... 'Tue' 'Wed'
│       Data variables: (12/19)
│           b                 (chain, draw, covariate, series) float32 16MB 0.7327 .....
│           b_avail           (chain, draw, series) float32 4MB 0.9059 0.8302 ... 10.85
│           b_decentered      (chain, draw, covariate, series) float32 16MB 0.2937 .....
│           b_loc_store       (chain, draw, covariate, store) float32 8MB 0.6704 ... ...
│           b_scale_store     (chain, draw, covariate, store) float32 8MB 0.4211 ... ...
│           centered_b        (chain, draw) float32 4kB 0.2623 0.2655 ... 0.2611 0.2609
│           ...                ...
│           phi_trend         (chain, draw, series) float32 4MB 0.4503 0.352 ... 0.4761
│           seasonal          (chain, draw, series, day_of_week) float32 28MB 0.00956...
│           seasonal_scale    (chain, draw) float32 4kB 0.04323 0.04336 ... 0.04362
│           sigma             (chain, draw, series) float32 4MB 0.1728 0.1769 ... 0.3151
│           slope             (chain, draw, time, series) float32 304MB -0.03345 ... ...
│           tau_trend         (chain, draw, series) float32 4MB 0.02124 ... 0.04133
│       Attributes:
│           created_at:                 2026-07-12T08:01:34.009523+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
│           variational:                True
├── Group: /posterior_predictive
│       Dimensions:  (chain: 1, draw: 1000, time: 76, obs_dim: 1000)
│       Coordinates:
│         * chain    (chain) int64 8B 0
│         * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│         * time     (time) datetime64[s] 608B 2024-03-28 2024-03-29 ... 2024-06-11
│         * obs_dim  (obs_dim) <U8 32kB '0::117' '0::691' '0::70' ... '99::589' '9::4'
│       Data variables:
│           obs      (chain, draw, time, obs_dim) float32 304MB 0.09489 ... 0.6046
│       Attributes:
│           created_at:                 2026-07-12T08:01:35.587957+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
├── Group: /observed_data
│       Dimensions:  (time: 76, obs_dim: 1000)
│       Coordinates:
│         * time     (time) datetime64[s] 608B 2024-03-28 2024-03-29 ... 2024-06-11
│         * obs_dim  (obs_dim) <U8 32kB '0::117' '0::691' '0::70' ... '99::589' '9::4'
│       Data variables:
│           obs      (time, obs_dim) float32 304kB 0.06142 0.8523 1.193 ... 1.541 0.3627
│       Attributes:
│           created_at:                 2026-07-12T08:01:35.588481+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /constant_data
│       Dimensions:     (input: 5, time: 76, series: 1000)
│       Coordinates:
│         * input       (input) <U18 360B 'availability' ... 'post_ramp'
│         * time        (time) datetime64[s] 608B 2024-03-28 2024-03-29 ... 2024-06-11
│         * series      (series) <U8 32kB '0::117' '0::691' '0::70' ... '99::589' '9::4'
│       Data variables:
│           covariates  (input, time, series) float32 2MB 0.0 0.8421 0.9677 ... 1.0 1.0
│       Attributes:
│           created_at:                 2026-07-12T08:01:35.588963+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /predictions
│       Dimensions:  (chain: 1, draw: 1000, time: 14, obs_dim: 1000)
│       Coordinates:
│         * chain    (chain) int64 8B 0
│         * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│         * time     (time) datetime64[s] 112B 2024-06-12 2024-06-13 ... 2024-06-25
│         * obs_dim  (obs_dim) <U8 32kB '0::117' '0::691' '0::70' ... '99::589' '9::4'
│       Data variables:
│           obs      (chain, draw, time, obs_dim) float32 56MB 0.7749 1.083 ... 0.8734
│       Attributes:
│           created_at:                 2026-07-12T08:01:38.036898+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
└── Group: /predictions_constant_data
        Dimensions:     (input: 5, time: 14, series: 1000)
        Coordinates:
          * input       (input) <U18 360B 'availability' ... 'post_ramp'
          * time        (time) datetime64[s] 112B 2024-06-12 2024-06-13 ... 2024-06-25
          * series      (series) <U8 32kB '0::117' '0::691' '0::70' ... '99::589' '9::4'
        Data variables:
            covariates  (input, time, series) float32 280kB 0.7831 1.0 0.633 ... 1.0 1.0
        Attributes:
            created_at:                 2026-07-12T08:01:38.037527+00:00
            creation_library:           ArviZ
            creation_library_version:   1.2.0
            creation_library_language:  Python
            sample_dims:                []
```


xarray.DataTree


/posterior(32)

Dimensions:


- chain: 1
- draw: 1000
- covariate: 4
- series: 1000
- store: 520
- time: 76
- day_of_week: 7


Coordinates: (7)


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


covariate


(covariate)


\<U18


'discount_magnitude' ... 'post_r...


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['discount_magnitude', 'activity_flag', 'holiday_flag', 'post_ramp'],dtype='<U18')


series


(series)


\<U8


'0::117' '0::691' ... '9::4'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['0::117', '0::691', '0::70', ..., '98::267', '99::589', '9::4'],shape=(1000,), dtype='<U8')


store


(store)


int64


0 1 2 3 4 5 ... 892 893 894 896 897


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([  0,   1,   2, ..., 894, 896, 897], shape=(520,))


time


(time)


datetime64\[s\]


2024-03-28 ... 2024-06-11


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['2024-03-28T00:00:00', '2024-03-29T00:00:00', '2024-03-30T00:00:00','2024-03-31T00:00:00', '2024-04-01T00:00:00', '2024-04-02T00:00:00','2024-04-03T00:00:00', '2024-04-04T00:00:00', '2024-04-05T00:00:00','2024-04-06T00:00:00', '2024-04-07T00:00:00', '2024-04-08T00:00:00','2024-04-09T00:00:00', '2024-04-10T00:00:00', '2024-04-11T00:00:00','2024-04-12T00:00:00', '2024-04-13T00:00:00', '2024-04-14T00:00:00','2024-04-15T00:00:00', '2024-04-16T00:00:00', '2024-04-17T00:00:00','2024-04-18T00:00:00', '2024-04-19T00:00:00', '2024-04-20T00:00:00','2024-04-21T00:00:00', '2024-04-22T00:00:00', '2024-04-23T00:00:00','2024-04-24T00:00:00', '2024-04-25T00:00:00', '2024-04-26T00:00:00','2024-04-27T00:00:00', '2024-04-28T00:00:00', '2024-04-29T00:00:00','2024-04-30T00:00:00', '2024-05-01T00:00:00', '2024-05-02T00:00:00','2024-05-03T00:00:00', '2024-05-04T00:00:00', '2024-05-05T00:00:00','2024-05-06T00:00:00', '2024-05-07T00:00:00', '2024-05-08T00:00:00','2024-05-09T00:00:00', '2024-05-10T00:00:00', '2024-05-11T00:00:00','2024-05-12T00:00:00', '2024-05-13T00:00:00', '2024-05-14T00:00:00','2024-05-15T00:00:00', '2024-05-16T00:00:00', '2024-05-17T00:00:00','2024-05-18T00:00:00', '2024-05-19T00:00:00', '2024-05-20T00:00:00','2024-05-21T00:00:00', '2024-05-22T00:00:00', '2024-05-23T00:00:00','2024-05-24T00:00:00', '2024-05-25T00:00:00', '2024-05-26T00:00:00','2024-05-27T00:00:00', '2024-05-28T00:00:00', '2024-05-29T00:00:00','2024-05-30T00:00:00', '2024-05-31T00:00:00', '2024-06-01T00:00:00','2024-06-02T00:00:00', '2024-06-03T00:00:00', '2024-06-04T00:00:00','2024-06-05T00:00:00', '2024-06-06T00:00:00', '2024-06-07T00:00:00','2024-06-08T00:00:00', '2024-06-09T00:00:00', '2024-06-10T00:00:00','2024-06-11T00:00:00'], dtype='datetime64[s]')


day_of_week


(day_of_week)


\<U3


'Thu' 'Fri' 'Sat' ... 'Tue' 'Wed'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['Thu', 'Fri', 'Sat', 'Sun', 'Mon', 'Tue', 'Wed'], dtype='<U3')


Data variables: (19)


b


(chain, draw, covariate, series)


float32


0.7327 0.5352 ... -0.06066 -0.09926


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 0.7327015 ,  0.53517944,  0.58960736, ..., -0.4654757 ,0.45798087, -0.24595201],[ 0.21725501, -0.01030174,  0.00951051, ..., -0.8323779 ,-0.90406096,  0.09329031],[ 0.4049259 ,  0.4471735 ,  0.4085952 , ...,  0.3050568 ,0.32771415,  0.82860446],[-0.06217518, -0.1816787 , -0.20172696, ...,  0.82235193,-0.09970709, -0.08217292]],[[-0.11629076, -0.6525044 , -0.48076352, ...,  0.13252778,-0.64504236, -0.43378443],[-0.2418315 , -0.21309458, -0.63033175, ...,  0.05837838,-0.5871965 ,  0.26923087],[ 0.41184786,  0.48219496,  0.41113168, ...,  0.13552561,0.36682132,  0.8448245 ],[-0.1256606 , -0.1337784 , -0.1355328 , ...,  0.68275195,-0.01197849, -0.20233664]],[[ 0.56549406,  0.71846354,  1.0728538 , ..., -0.5624208 ,-0.8872545 ,  0.7249411 ],...-0.13660231, -0.28972152]],[[-0.32099184,  0.20441207, -1.7419033 , ...,  0.5771687 ,0.18680775, -0.01424128],[ 0.32904643,  0.36141235,  0.08917232, ..., -0.1455566 ,-0.90750074,  0.24665368],[ 0.36775553,  0.39903358,  0.3800826 , ...,  0.22436182,0.32496247,  0.9293406 ],[-0.18231729, -0.16127662, -0.14998025, ...,  0.76510495,-0.05744955, -0.16515183]],[[-0.09423722, -0.5923356 , -0.49523684, ..., -0.7479476 ,0.5707611 ,  0.2813222 ],[ 0.08421418,  0.50272197, -0.21997993, ..., -0.12335297,0.28789365, -0.13072014],[ 0.3391319 ,  0.34419033,  0.3264452 , ...,  0.30604604,0.34597763,  1.322784  ],[-0.16628475, -0.17143168, -0.16125275, ...,  0.742573  ,-0.06065665, -0.09926093]]]],shape=(1, 1000, 4, 1000), dtype=float32)


b_avail


(chain, draw, series)


float32


0.9059 0.8302 ... 0.7767 10.85


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 0.9058789 ,  0.83021224,  0.43924472, ...,  1.2430522 ,0.8291931 , 18.861685  ],[ 1.2929732 ,  0.5864446 ,  1.0245391 , ...,  0.9944278 ,1.2943091 ,  9.479123  ],[ 0.8335112 ,  0.75110525,  0.87743783, ...,  1.1896479 ,1.364799  , 12.192536  ],...,[ 0.8872638 ,  1.6542561 ,  0.5688779 , ...,  1.2341172 ,1.0077349 , 18.024633  ],[ 1.0031425 ,  1.1143782 ,  1.0505452 , ...,  1.1344355 ,1.5414153 ,  8.943121  ],[ 0.6858928 ,  1.1106776 ,  0.5392324 , ...,  1.3235066 ,0.7767347 , 10.849131  ]]], shape=(1, 1000, 1000), dtype=float32)


b_decentered


(chain, draw, covariate, series)


float32


0.2937 -0.08013 ... -0.04432 0.1851


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 2.93748766e-01, -8.01303908e-02,  2.28934214e-02, ...,-7.92792797e-01,  4.02963221e-01,  1.02302104e-01],[ 6.40261948e-01,  1.08475596e-01,  1.54775634e-01, ...,-2.50375926e-01, -1.55803108e+00,  4.66273695e-01],[ 4.07457948e-02,  3.85790199e-01,  7.07135648e-02, ...,2.40119845e-01,  1.16837338e-01,  1.59059525e-01],[ 6.93688512e-01, -1.71980277e-01, -3.17207426e-01, ...,3.53094488e-01, -2.15473443e-01, -2.91391462e-01]],[[ 2.48366445e-01, -6.70473278e-01, -3.76183212e-01, ...,-7.67537594e-01, -8.32740188e-01, -9.56300020e-01],[ 4.26516235e-01,  5.20034552e-01, -8.37776840e-01, ...,8.33343118e-02,  5.32736957e-01,  2.72843421e-01],[-1.27726734e-01,  9.06098366e-01, -1.38251752e-01, ...,-9.29040387e-02, -4.34324890e-03,  2.07521141e-01],[ 2.35558569e-01, -1.64984912e-01, -2.51549602e-01, ...,1.73847273e-01,  5.31734601e-02, -3.84902716e-01]],[[-3.59059006e-01, -7.66977519e-02,  5.77459395e-01, ...,-9.56864595e-01,  4.57581045e-04,  7.99997449e-01],...3.45631421e-01, -1.94450065e-01, -5.37208796e-01]],[[ 3.40368867e-01,  1.12568724e+00, -1.78345966e+00, ...,4.98287529e-01, -1.04069807e-01, -5.88475943e-01],[ 6.48384035e-01,  7.20498621e-01,  1.13919757e-01, ...,-7.95638710e-02, -8.51521492e-01, -1.68375373e-01],[-3.91097933e-01,  1.61869645e-01, -1.73166230e-01, ...,-4.15601581e-02,  9.30436552e-02,  3.45719576e-01],[-5.43934822e-01, -2.35571116e-01, -7.00159073e-02, ...,3.97237539e-01,  2.11958930e-01, -1.21341258e-01]],[[ 8.47829461e-01, -6.42952681e-01, -3.52341264e-01, ...,-5.91994166e-01, -3.57748955e-01,  1.99150056e-01],[-7.21614242e-01,  1.36813271e+00, -2.24055552e+00, ...,-1.15310812e+00,  7.41244316e-01,  2.44284794e-01],[ 2.62654543e-01,  4.13392276e-01, -1.15399219e-01, ...,1.92226648e-01, -4.25348952e-02,  8.18791866e-01],[ 1.64605215e-01,  3.60036902e-02,  2.90335804e-01, ...,4.31081831e-01, -4.43213284e-02,  1.85060605e-01]]]],shape=(1, 1000, 4, 1000), dtype=float32)


b_loc_store


(chain, draw, covariate, store)


float32


0.6704 0.3138 ... 0.6402 0.5791


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 0.6704295 ,  0.31378347, -0.29407752, ..., -0.273412  ,0.31995007,  0.26481318],[-0.06389178,  0.0674229 ,  0.13447534, ..., -0.16502619,-0.7032891 , -0.453522  ],[ 0.4132095 ,  0.43112522,  0.23895185, ...,  0.0961524 ,0.276298  ,  0.08649287],[-0.16387181, -0.12068717, -0.1540474 , ...,  0.24648267,0.67795366,  0.5277432 ]],[[-0.30912578,  0.21265057, -0.8063265 , ...,  0.2927245 ,0.26728678,  0.43142498],[-0.40601778, -0.00560521,  0.785333  , ..., -0.14988421,-0.5219016 ,  0.32269156],[ 0.42827606,  0.35327542,  0.25929588, ...,  0.17660142,0.34791428,  0.10271136],[-0.13114028, -0.14638393, -0.17883688, ...,  0.26493987,0.5927017 ,  0.5921562 ]],[[ 0.8873062 ,  0.5245105 , -1.0319564 , ..., -0.04479774,-0.10824993,  0.514835  ],...0.6331313 ,  0.47760954]],[[-0.6648563 ,  0.2347194 , -0.18899907, ..., -0.6213142 ,0.02422936, -0.5619592 ],[ 0.04309398,  0.04816489,  0.44132936, ..., -0.06960069,-0.08832747, -0.12124352],[ 0.39572227,  0.3741486 ,  0.35571873, ...,  0.12785394,0.25890088,  0.2345597 ],[-0.14783679, -0.13017501, -0.21209653, ...,  0.2814543 ,0.64379495,  0.5463222 ]],[[-0.41356906,  0.5106293 ,  0.4756703 , ...,  0.2904767 ,-0.14705971,  0.1513821 ],[ 0.24134147, -0.0044094 , -0.52075356, ...,  0.81762606,-0.33850962,  1.062957  ],[ 0.33323568,  0.39088345,  0.2545932 , ...,  0.09935924,0.3244992 ,  0.21726802],[-0.174697  , -0.10738784, -0.13876465, ...,  0.26119906,0.6401926 ,  0.5791291 ]]]],shape=(1, 1000, 4, 520), dtype=float32)


b_scale_store


(chain, draw, covariate, store)


float32


0.4211 0.1383 ... 0.05286 0.05782


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[0.42105007, 0.1383232 , 0.18002   , ..., 0.2038652 ,0.05850126, 0.5463838 ],[0.31641024, 0.02595679, 0.5805648 , ..., 0.21245587,0.16396692, 0.64148325],[0.05801855, 0.23624426, 0.07748052, ..., 0.07111645,0.04147619, 0.0851369 ],[0.06826492, 0.00454931, 0.10656209, ..., 0.0725597 ,0.03779022, 0.02867362]],[[0.48034543, 0.1528588 , 0.21853566, ..., 0.20255598,0.22881883, 0.5677708 ],[0.20059258, 0.07014091, 0.09907395, ..., 0.1572279 ,0.11306201, 0.38199198],[0.02575796, 0.23308755, 0.03157636, ..., 0.01804068,0.04207972, 0.08006065],[0.00495191, 0.01390419, 0.02763122, ..., 0.09513751,0.13219555, 0.09192435]],[[0.434429  , 0.09746231, 0.23265637, ..., 0.14950104,0.23741223, 0.09606674],...0.02629862, 0.06952746]],[[0.5804471 , 0.34387618, 0.31099313, ..., 0.20951234,0.458105  , 0.14025812],[0.338149  , 0.17689268, 0.42562014, ..., 0.22784273,0.3030713 , 0.53049713],[0.02049752, 0.2602991 , 0.03141032, ..., 0.00986498,0.04508592, 0.13850527],[0.02642063, 0.02895456, 0.02190295, ..., 0.03528734,0.26742893, 0.05654936]],[[0.22688743, 0.0677025 , 0.19491038, ..., 0.22725148,0.21071336, 0.42996585],[0.11351094, 0.10801568, 0.1750502 , ..., 0.28481746,0.16037919, 0.31000838],[0.01012316, 0.28337845, 0.04082821, ..., 0.09619088,0.1300085 , 0.7776074 ],[0.01284789, 0.03248451, 0.04190164, ..., 0.03807579,0.05285703, 0.05782085]]]],shape=(1, 1000, 4, 520), dtype=float32)


centered_b


(chain, draw)


float32


0.2623 0.2655 ... 0.2611 0.2609


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.26233548, 0.26549035, 0.26480514, 0.27075672, 0.2645799 ,0.25479868, 0.27248734, 0.26280552, 0.26650658, 0.26932094,0.26730692, 0.26439437, 0.26215747, 0.26810533, 0.26142687,0.2647225 , 0.26326337, 0.26595047, 0.26900443, 0.26258862,0.26737693, 0.26513466, 0.26389733, 0.26666996, 0.26159397,0.25834355, 0.2617045 , 0.26345554, 0.26948476, 0.26459652,0.26470765, 0.26578158, 0.26507854, 0.26038003, 0.26427913,0.26780644, 0.26221538, 0.26024058, 0.26643884, 0.26383096,0.2625572 , 0.25497437, 0.26644352, 0.26738983, 0.27038604,0.268384  , 0.26131558, 0.27050158, 0.2701728 , 0.26418036,0.26525983, 0.26156303, 0.2642218 , 0.2624453 , 0.2632251 ,0.2688676 , 0.26596558, 0.26000926, 0.26312163, 0.26687527,0.26388362, 0.26027644, 0.26883367, 0.26514766, 0.25908163,0.2669189 , 0.26677173, 0.26325932, 0.2623442 , 0.2601748 ,0.26557723, 0.2635987 , 0.26475415, 0.2682131 , 0.26537776,0.26832917, 0.2649922 , 0.26484057, 0.2659333 , 0.26063088,0.26039165, 0.26054186, 0.26932883, 0.2596511 , 0.2644878 ,0.26578188, 0.26611432, 0.26501253, 0.2569378 , 0.2669405 ,0.26172683, 0.26511312, 0.26498663, 0.27080092, 0.2653688 ,0.25942817, 0.26441082, 0.26284462, 0.27209255, 0.26539022,...0.26358685, 0.26320064, 0.2693098 , 0.25939497, 0.26086462,0.26869366, 0.26402253, 0.2662033 , 0.26436332, 0.26991302,0.25727707, 0.2706602 , 0.25924942, 0.26426548, 0.26351097,0.25967494, 0.26724434, 0.26044452, 0.2711084 , 0.26092464,0.26567104, 0.27035886, 0.2618136 , 0.262139  , 0.26658416,0.2649269 , 0.26400897, 0.2660365 , 0.2593376 , 0.2623409 ,0.2629965 , 0.26420066, 0.26925722, 0.2609797 , 0.2677085 ,0.27090248, 0.25864026, 0.26765764, 0.2633785 , 0.2655388 ,0.26392406, 0.26510254, 0.2620825 , 0.25972676, 0.26731643,0.27035835, 0.2697992 , 0.2637028 , 0.26549315, 0.26483706,0.26661128, 0.2652191 , 0.26892903, 0.2616572 , 0.27020648,0.26719013, 0.2677627 , 0.26460782, 0.26312158, 0.26523906,0.2647263 , 0.26443425, 0.26351357, 0.2676804 , 0.2658947 ,0.26398864, 0.26336983, 0.26456368, 0.26373973, 0.2634533 ,0.26370606, 0.26270792, 0.2611336 , 0.26478115, 0.26357445,0.2669539 , 0.25916097, 0.2592719 , 0.26496696, 0.26689798,0.26138535, 0.2642705 , 0.26630583, 0.2657603 , 0.26549116,0.25885347, 0.27451038, 0.26526484, 0.2676412 , 0.262451  ,0.26600066, 0.26307383, 0.2669383 , 0.26111427, 0.26093325]],dtype=float32)


centered_drift


(chain, draw)


float32


0.102 0.1015 ... 0.1005 0.1008


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.10200903, 0.10147061, 0.10184262, 0.10072391, 0.10047265,0.10119418, 0.10046414, 0.10142997, 0.10188889, 0.10091409,0.10174067, 0.10060004, 0.10182583, 0.10226828, 0.10108209,0.10147981, 0.10189167, 0.10222016, 0.10018819, 0.1015973 ,0.10145976, 0.10162587, 0.10127162, 0.10070696, 0.10078317,0.10150455, 0.10186842, 0.10118937, 0.10154779, 0.10139094,0.10080479, 0.10096683, 0.10102694, 0.10151596, 0.102057  ,0.10231588, 0.10150627, 0.10130467, 0.10131959, 0.10100754,0.10127474, 0.10152092, 0.10154755, 0.10110923, 0.10146827,0.10142217, 0.10087577, 0.10233668, 0.10238623, 0.10083882,0.10044427, 0.10169434, 0.10121613, 0.10175073, 0.10178118,0.10125247, 0.10251349, 0.10148335, 0.10149491, 0.10083044,0.1014469 , 0.10148546, 0.10162747, 0.10139471, 0.10057282,0.10118549, 0.10123903, 0.10165429, 0.10201883, 0.10184818,0.10119689, 0.10080298, 0.10165287, 0.10215403, 0.10084324,0.10167587, 0.10150154, 0.10115702, 0.10119887, 0.1013312 ,0.10072396, 0.10154606, 0.10126975, 0.10261185, 0.10172734,0.10134908, 0.10093088, 0.10075936, 0.10231053, 0.10154015,0.1017824 , 0.10201056, 0.10113538, 0.10165185, 0.10130633,0.10182794, 0.10047434, 0.10207901, 0.10155714, 0.10194006,...0.10173994, 0.10141982, 0.10151231, 0.10196619, 0.10154743,0.10202955, 0.1012869 , 0.10212576, 0.10186205, 0.10088629,0.10160609, 0.10222112, 0.10195988, 0.10296995, 0.10169876,0.10092045, 0.10138222, 0.10061155, 0.1011885 , 0.1010829 ,0.10109528, 0.10131344, 0.10082092, 0.10181584, 0.10227651,0.10054763, 0.10136829, 0.1015736 , 0.10100912, 0.10159831,0.10083707, 0.1019623 , 0.10115312, 0.10135324, 0.10089383,0.1019498 , 0.10190248, 0.10236104, 0.10117512, 0.10056707,0.10120717, 0.10063843, 0.10068265, 0.10071977, 0.10282162,0.1017727 , 0.10126705, 0.10158356, 0.10154911, 0.10188287,0.10162127, 0.1008053 , 0.10176098, 0.10067765, 0.10133233,0.10173839, 0.10220788, 0.10127144, 0.10139368, 0.10223482,0.1018373 , 0.10075901, 0.10177448, 0.10141557, 0.10133152,0.10150285, 0.10170973, 0.10095934, 0.10105994, 0.10144672,0.10084407, 0.1006033 , 0.10182112, 0.10206961, 0.10044647,0.10120528, 0.10175355, 0.10181423, 0.10229731, 0.10118724,0.1008775 , 0.10094085, 0.10145411, 0.10108715, 0.10132276,0.1016441 , 0.10146501, 0.10147661, 0.10165174, 0.10108821,0.1009978 , 0.10131455, 0.10088122, 0.1004556 , 0.10083622]],dtype=float32)


drift


(chain, draw, time, series)


float32


-0.002989 0.01071 ... 0.00557


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[-2.98885047e-03,  1.07113663e-02, -1.00465706e-02, ...,1.16400998e-02, -1.83755509e-03,  1.18578393e-02],[-5.35391737e-03,  2.56952690e-03,  2.13143639e-02, ...,-1.45548093e-03,  1.93246198e-03,  4.43763435e-02],[-5.19732106e-03,  5.33684483e-03, -5.38071363e-05, ...,9.20048822e-03, -1.11264619e-03,  2.65611894e-02],...,[-7.13699404e-03, -2.51443544e-03,  6.93696411e-03, ...,-1.80137083e-02, -1.81537587e-03,  2.23888606e-02],[ 5.45486691e-04, -8.45727231e-03, -2.89987493e-03, ...,-5.62306726e-03, -3.58525105e-03,  3.97494882e-02],[-5.33199869e-03,  4.86604264e-03, -1.55661386e-02, ...,7.25381915e-03, -3.95621452e-03,  4.88639511e-02]],[[-3.35268211e-03, -6.41041494e-04, -1.32722897e-03, ...,2.84257787e-03, -2.39744247e-03, -7.95080035e-04],[-7.98275787e-03, -6.39885082e-04, -9.78414249e-03, ...,-5.64473774e-03, -5.79305459e-03, -9.75012546e-04],[ 4.06606402e-03, -6.09052717e-04,  5.36730094e-03, ...,1.27980905e-03,  5.97285572e-03,  3.42376065e-03],...2.08267812e-02,  2.32245098e-03,  2.57462226e-02],[-2.67390232e-03, -1.07359339e-03,  1.18538961e-02, ...,-1.25873536e-02, -2.74756155e-03, -2.48940341e-04],[-3.07031139e-03, -6.89171627e-03,  1.09405280e-03, ...,-2.29067495e-03,  4.51709377e-03,  2.47531483e-04]],[[-3.97022162e-03,  5.69164113e-04,  2.02960558e-02, ...,-4.84618684e-03,  5.81624219e-03,  6.50619040e-04],[ 6.34197332e-03,  9.91552416e-03, -1.33768760e-03, ...,-5.55246184e-03, -9.63734929e-03, -1.89695042e-02],[-2.03325767e-02,  4.83612111e-03, -1.55277587e-02, ...,2.33469778e-04, -4.81401011e-03, -2.19891546e-03],...,[-1.05354504e-03, -8.98832269e-03,  1.68784102e-03, ...,7.59037444e-03,  7.87356310e-03, -4.70034871e-03],[-6.35251729e-03, -5.75470412e-03, -1.31127564e-02, ...,2.34250375e-03,  2.70218123e-03,  7.87812471e-03],[ 1.51919834e-02,  7.67166493e-03,  6.53121737e-04, ...,1.17621152e-02,  6.50672242e-03,  5.56968385e-03]]]],shape=(1, 1000, 76, 1000), dtype=float32)


drift_decentered


(chain, draw, time, series)


float32


-0.3274 0.8124 ... 0.6139 0.5427


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[-0.3274039 ,  0.8124239 , -0.5890946 , ...,  0.5498305 ,-0.28900805,  0.280123  ],[-0.58647746,  0.19489063,  1.2497972 , ..., -0.06875094,0.30393487,  1.0483221 ],[-0.56932366,  0.40478313, -0.00315506, ...,  0.43459326,-0.17499541,  0.62746674],...,[-0.78179884, -0.19071212,  0.4067585 , ..., -0.85089356,-0.28551975,  0.528902  ],[ 0.05975357, -0.64145786, -0.17003818, ..., -0.2656106 ,-0.5638832 ,  0.9390198 ],[-0.58407646,  0.36907423, -0.9127421 , ...,  0.3426406 ,-0.6222278 ,  1.1543348 ]],[[-0.24850877, -0.21949759, -0.14907843, ...,  0.23381662,-0.32858893, -0.05924674],[-0.59170103, -0.21910162, -1.098985  , ..., -0.46430868,-0.79398507, -0.07265472],[ 0.30138633, -0.20854439,  0.6028717 , ...,  0.10527086,0.81862825,  0.25512734],...0.29416224,  1.4914079 ],[-0.21773407, -0.14288896,  1.1872112 , ..., -0.73009396,-0.34800684, -0.01442043],[-0.25001338, -0.91724694,  0.10957339, ..., -0.13286415,0.5721363 ,  0.01433882]],[[-0.35150638,  0.0503104 ,  0.8073027 , ..., -0.43244722,0.5487591 ,  0.06339869],[ 0.56149113,  0.87646765, -0.05320831, ..., -0.49547133,-0.9092784 , -1.8484576 ],[-1.8001591 ,  0.42748156, -0.61763734, ...,  0.02083357,-0.45419908, -0.21427032],...,[-0.09327636, -0.7945091 ,  0.06713613, ...,  0.6773235 ,0.7428661 , -0.4580191 ],[-0.56242466, -0.5086783 , -0.5215774 , ...,  0.20903222,0.25494924,  0.76767313],[ 1.345033  ,  0.67812514,  0.02597879, ...,  1.0495869 ,0.6139055 ,  0.5427303 ]]]],shape=(1, 1000, 76, 1000), dtype=float32)


drift_scale


(chain, draw, series)


float32


0.005355 0.008063 ... 0.006141


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.00535464, 0.00806319, 0.01073926, ..., 0.01366267,0.00357928, 0.02955624],[0.00829617, 0.00151088, 0.00522363, ..., 0.00738853,0.00418577, 0.00824733],[0.01448783, 0.00995138, 0.00645197, ..., 0.00821516,0.00729921, 0.01327808],...,[0.00611331, 0.00402399, 0.00715354, ..., 0.00342364,0.00558669, 0.008528  ],[0.00751337, 0.00435139, 0.00596913, ..., 0.01095532,0.0045978 , 0.01097107],[0.00683161, 0.00684384, 0.0166336 , ..., 0.00677214,0.00636509, 0.00614071]]], shape=(1, 1000, 1000), dtype=float32)


floor


(chain, draw, series)


float32


0.04987 0.09404 ... 0.02281 0.0975


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.04987008, 0.09403825, 0.02455026, ..., 0.0471703 ,0.0264982 , 0.18505833],[0.03545643, 0.03019552, 0.04755045, ..., 0.07297904,0.05487986, 0.11232934],[0.06460002, 0.12286674, 0.01205032, ..., 0.05606999,0.03977155, 0.03879504],...,[0.08200859, 0.09337291, 0.01362799, ..., 0.0370608 ,0.04156823, 0.09276836],[0.04082322, 0.00847312, 0.01645038, ..., 0.04436991,0.09965207, 0.14226629],[0.01980396, 0.04671904, 0.01851355, ..., 0.0437461 ,0.02280968, 0.09749652]]], shape=(1, 1000, 1000), dtype=float32)


init_level


(chain, draw, series)


float32


1.032 1.005 1.051 ... 0.8267 0.5118


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[1.0315595 , 1.0054557 , 1.0512366 , ..., 0.6835014 ,0.8497987 , 0.40620345],[0.9901402 , 0.991026  , 1.0456327 , ..., 0.6507294 ,0.8222531 , 0.5003004 ],[1.0578299 , 1.0598993 , 0.98908925, ..., 0.6331284 ,0.8472848 , 0.6067999 ],...,[1.0135481 , 1.0334213 , 0.97165596, ..., 0.67463905,0.82708234, 0.5061352 ],[0.98273915, 1.0241371 , 0.9545655 , ..., 0.6881134 ,0.79352987, 0.4945127 ],[0.97720075, 0.98367965, 1.0147444 , ..., 0.61759174,0.8267136 , 0.5117955 ]]], shape=(1, 1000, 1000), dtype=float32)


noise_loading


(chain, draw, series)


float32


0.063 0.03913 ... 0.04601 0.1161


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.06300467, 0.03912514, 0.04147755, ..., 0.03326455,0.03198916, 0.18956716],[0.05048452, 0.03433239, 0.0786994 , ..., 0.04368647,0.06976815, 0.13635592],[0.04070584, 0.04562899, 0.03758578, ..., 0.04108103,0.03204321, 0.161661  ],...,[0.03843369, 0.05684227, 0.03812158, ..., 0.05204021,0.03857201, 0.08197345],[0.02596977, 0.0618213 , 0.04851258, ..., 0.0292407 ,0.03745   , 0.08897512],[0.03431905, 0.02765489, 0.0211112 , ..., 0.02465225,0.04600977, 0.11610676]]], shape=(1, 1000, 1000), dtype=float32)


phi_trend


(chain, draw, series)


float32


0.4503 0.352 ... 0.3154 0.4761


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.45032   , 0.35195625, 0.25650853, ..., 0.26384676,0.32993296, 0.37358183],[0.52015275, 0.30666614, 0.46984968, ..., 0.21059105,0.35017806, 0.35172603],[0.5205934 , 0.35759333, 0.41276425, ..., 0.23865275,0.39957234, 0.4769194 ],...,[0.5014282 , 0.42440665, 0.2874285 , ..., 0.3466804 ,0.4057308 , 0.5295952 ],[0.42061874, 0.24605444, 0.33961415, ..., 0.34252936,0.53967166, 0.56564397],[0.40888572, 0.23757204, 0.47105423, ..., 0.39403316,0.3154485 , 0.4761041 ]]], shape=(1, 1000, 1000), dtype=float32)


seasonal


(chain, draw, series, day_of_week)


float32


0.009561 -0.05798 ... -0.02125


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 0.00956087, -0.05797532,  0.01289185, ..., -0.01124135,-0.03108806,  0.06137533],[-0.03505449, -0.01905084,  0.11745142, ..., -0.01761758,-0.01758564, -0.00868104],[ 0.00233679,  0.02875559,  0.01504036, ...,  0.02948423,-0.0647293 , -0.05234095],...,[ 0.01067861,  0.00135523,  0.00159928, ..., -0.0748648 ,0.02878699, -0.01405538],[-0.03071514, -0.06157556,  0.06039578, ...,  0.03289732,-0.04549565, -0.01135569],[-0.04460856, -0.02146848,  0.00622398, ...,  0.0888956 ,-0.06263438,  0.02473539]],[[ 0.00296343, -0.02826936,  0.04682698, ...,  0.01206401,-0.02870766, -0.06217404],[-0.07371726, -0.00524618,  0.01244427, ...,  0.03855678,0.07853024, -0.08444241],[ 0.00361296, -0.00821699,  0.02820244, ...,  0.04848316,-0.01702142, -0.0147264 ],...0.03695434,  0.03298304],[-0.06623142, -0.01187318, -0.0132759 , ...,  0.05821058,0.02281078, -0.03975384],[ 0.0002784 , -0.00566969,  0.05026106, ...,  0.04504954,-0.0684071 ,  0.04364699]],[[-0.00299481, -0.02949455,  0.01396809, ..., -0.01490754,-0.02046689,  0.02984412],[-0.03801716,  0.02309235,  0.01445361, ...,  0.00361767,-0.00694137, -0.00613675],[-0.07295831,  0.04749529,  0.03288661, ..., -0.03070429,-0.04326733,  0.0262346 ],...,[ 0.00869107, -0.03245062, -0.02355226, ...,  0.01526209,0.05470578, -0.10482976],[-0.00756898, -0.09592773,  0.0639357 , ...,  0.06290121,-0.05788342, -0.01242638],[-0.02203647,  0.04108404,  0.01294765, ..., -0.0363759 ,0.00833636, -0.02125402]]]],shape=(1, 1000, 1000, 7), dtype=float32)


seasonal_scale


(chain, draw)


float32


0.04323 0.04336 ... 0.0439 0.04362


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.04322875, 0.04336356, 0.04304946, 0.04366585, 0.04287754,0.04339037, 0.04316967, 0.0421982 , 0.04382411, 0.04372697,0.04336275, 0.04341173, 0.04309391, 0.04325411, 0.04358628,0.04361146, 0.04314669, 0.04291718, 0.04249483, 0.04304069,0.04299839, 0.04350255, 0.04282907, 0.04341662, 0.04339822,0.04330965, 0.04359376, 0.04282482, 0.04383216, 0.04294991,0.04315653, 0.04327794, 0.04329769, 0.04324136, 0.04346867,0.04304013, 0.04307001, 0.04348217, 0.04337754, 0.04343932,0.04332095, 0.04354927, 0.04354881, 0.04301328, 0.04324969,0.04275395, 0.04373461, 0.04289543, 0.04312428, 0.04303712,0.04326146, 0.04331559, 0.042978  , 0.04321187, 0.04288776,0.04269573, 0.04374782, 0.04348957, 0.04282869, 0.04326428,0.0436417 , 0.04337151, 0.04354994, 0.04338058, 0.04302223,0.04335397, 0.04256213, 0.04267624, 0.04331248, 0.04309619,0.04338168, 0.04312943, 0.04310658, 0.04324235, 0.04417701,0.04273943, 0.04358589, 0.04403514, 0.04294303, 0.04275084,0.04314898, 0.0426501 , 0.0430967 , 0.0428238 , 0.04323477,0.04396377, 0.04270663, 0.04389881, 0.04294789, 0.04321359,0.04336986, 0.04347366, 0.04375276, 0.0431978 , 0.0435106 ,0.04400306, 0.04304588, 0.0430486 , 0.04350229, 0.0426559 ,...0.043154  , 0.04280531, 0.04402635, 0.0433495 , 0.04361435,0.04309192, 0.04341808, 0.04294398, 0.0427666 , 0.04328386,0.04324624, 0.04339274, 0.04257418, 0.04353574, 0.04297961,0.04350844, 0.04358373, 0.04310221, 0.04333892, 0.04301902,0.04353072, 0.04294636, 0.04278408, 0.04253123, 0.04320271,0.04314634, 0.04341194, 0.04304275, 0.04330185, 0.0431014 ,0.04294265, 0.04277737, 0.043323  , 0.04354027, 0.04334927,0.04365484, 0.04345117, 0.04291986, 0.0430706 , 0.04316252,0.04306724, 0.0430001 , 0.04347942, 0.04343528, 0.04324457,0.04297731, 0.04358321, 0.04374049, 0.04363283, 0.04309112,0.04374257, 0.04260633, 0.04320884, 0.04332821, 0.04310523,0.04370182, 0.0434755 , 0.04310061, 0.04291433, 0.04361084,0.04303801, 0.04347458, 0.04333552, 0.04258454, 0.04293197,0.04312298, 0.04381014, 0.04251461, 0.04314465, 0.04277023,0.04311198, 0.04333933, 0.04343827, 0.04284963, 0.04319448,0.04291925, 0.04360227, 0.0433718 , 0.04278955, 0.04401878,0.04350941, 0.04354069, 0.04338023, 0.04366054, 0.04298465,0.04321781, 0.0434421 , 0.04297736, 0.04319046, 0.04277753,0.04280677, 0.04310729, 0.04387914, 0.04390493, 0.0436245 ]],dtype=float32)


sigma


(chain, draw, series)


float32


0.1728 0.1769 ... 0.1854 0.3151


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.17282921, 0.17693496, 0.28679824, ..., 0.12429476,0.13206822, 0.30941185],[0.22862288, 0.18526997, 0.22020298, ..., 0.12397757,0.13065411, 0.42027494],[0.18432698, 0.18071245, 0.28276148, ..., 0.14982738,0.14725825, 0.3430943 ],...,[0.1847771 , 0.17919493, 0.21319161, ..., 0.13862173,0.12925315, 0.46628648],[0.1960159 , 0.17813471, 0.26120156, ..., 0.16013558,0.11549912, 0.36633798],[0.19741094, 0.24743408, 0.24587268, ..., 0.12912403,0.18536991, 0.31512374]]], shape=(1, 1000, 1000), dtype=float32)


slope


(chain, draw, time, series)


float32


-0.03345 0.02106 ... -0.05876


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[-3.34495157e-02,  2.10635327e-02,  2.58427057e-02, ...,-9.36982594e-03,  1.04181087e-02, -3.54602337e-02],[-5.11840824e-03,  1.16037577e-02,  1.41344229e-02, ...,1.26542489e-03, -1.10859042e-02, -2.02848185e-02],[ 1.67249294e-03, -6.20954204e-03, -5.88632515e-03, ...,-3.25256474e-02, -1.35209900e-03, -3.78991067e-02],...,[-1.85873471e-02,  1.57818794e-02, -2.66471431e-02, ...,5.11887204e-03, -5.13703562e-05, -6.07833862e-02],[-9.02725197e-03,  1.03124864e-02, -1.68117620e-02, ...,2.58901529e-02,  6.54008798e-03, -4.41747792e-02],[ 5.90714104e-02,  9.43768211e-03, -3.82812694e-02, ...,-1.72540303e-02, -2.94497926e-02,  2.30965726e-02]],[[-1.11487675e-02,  1.69407316e-02,  2.93317414e-03, ...,-3.44030820e-02,  2.59724818e-03,  1.13629168e-02],[-9.71307606e-03, -4.72927606e-03,  2.20265090e-02, ...,-1.17495926e-02, -3.56911682e-04, -5.26362378e-03],[ 3.73242535e-02, -6.88015949e-03, -2.04350092e-02, ...,-7.64415134e-03, -3.29811173e-03, -4.07273360e-02],...-2.84808148e-02, -7.98860379e-03,  7.75046460e-03],[-6.74106460e-03,  9.75709967e-03, -1.13702752e-02, ...,2.28801090e-03,  3.12302858e-02, -4.03537825e-02],[ 3.71551551e-02,  7.26937829e-03,  8.73142667e-03, ...,-1.04343640e-02,  1.69600882e-02,  1.44198323e-02]],[[ 4.17334847e-02, -1.95697881e-02,  1.19886072e-02, ...,-1.21980775e-02,  2.17427798e-02,  1.99548751e-02],[-2.83908844e-02, -6.12000003e-03,  6.16150210e-03, ...,-2.15857774e-02,  1.16697326e-02, -6.88494695e-03],[-5.43437898e-02, -8.11261311e-03,  9.16755013e-03, ...,2.92341877e-02,  4.57335934e-02,  7.46627748e-02],...,[-2.56485306e-02, -2.04821806e-02, -1.41616259e-02, ...,-1.39387678e-02,  1.83360768e-03, -5.03611416e-02],[ 7.18905684e-03,  1.83932073e-02,  1.04689961e-02, ...,-2.03526989e-02,  6.51011895e-03, -1.77273527e-02],[-4.02009971e-02,  2.46744454e-02, -3.22520896e-03, ...,-8.28396063e-04, -1.29121486e-02, -5.87593466e-02]]]],shape=(1, 1000, 76, 1000), dtype=float32)


tau_trend


(chain, draw, series)


float32


0.02124 0.02133 ... 0.02082 0.04133


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.02124109, 0.02132846, 0.01879675, ..., 0.0195722 ,0.02059693, 0.04338111],[0.02435392, 0.0172564 , 0.02241057, ..., 0.01742521,0.02357825, 0.0404405 ],[0.02656629, 0.01825685, 0.02437035, ..., 0.01702492,0.02257176, 0.04194565],...,[0.02482002, 0.01839594, 0.01606937, ..., 0.01835628,0.02490854, 0.0372172 ],[0.02741552, 0.01785041, 0.0199393 , ..., 0.01922768,0.02066401, 0.04292585],[0.02377922, 0.0172871 , 0.0204699 , ..., 0.01875438,0.02081667, 0.04133008]]], shape=(1, 1000, 1000), dtype=float32)


Attributes: (6)


created_at :  
2026-07-12T08:01:34.009523+00:00

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
- time: 76
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


datetime64\[s\]


2024-03-28 ... 2024-06-11


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['2024-03-28T00:00:00', '2024-03-29T00:00:00', '2024-03-30T00:00:00','2024-03-31T00:00:00', '2024-04-01T00:00:00', '2024-04-02T00:00:00','2024-04-03T00:00:00', '2024-04-04T00:00:00', '2024-04-05T00:00:00','2024-04-06T00:00:00', '2024-04-07T00:00:00', '2024-04-08T00:00:00','2024-04-09T00:00:00', '2024-04-10T00:00:00', '2024-04-11T00:00:00','2024-04-12T00:00:00', '2024-04-13T00:00:00', '2024-04-14T00:00:00','2024-04-15T00:00:00', '2024-04-16T00:00:00', '2024-04-17T00:00:00','2024-04-18T00:00:00', '2024-04-19T00:00:00', '2024-04-20T00:00:00','2024-04-21T00:00:00', '2024-04-22T00:00:00', '2024-04-23T00:00:00','2024-04-24T00:00:00', '2024-04-25T00:00:00', '2024-04-26T00:00:00','2024-04-27T00:00:00', '2024-04-28T00:00:00', '2024-04-29T00:00:00','2024-04-30T00:00:00', '2024-05-01T00:00:00', '2024-05-02T00:00:00','2024-05-03T00:00:00', '2024-05-04T00:00:00', '2024-05-05T00:00:00','2024-05-06T00:00:00', '2024-05-07T00:00:00', '2024-05-08T00:00:00','2024-05-09T00:00:00', '2024-05-10T00:00:00', '2024-05-11T00:00:00','2024-05-12T00:00:00', '2024-05-13T00:00:00', '2024-05-14T00:00:00','2024-05-15T00:00:00', '2024-05-16T00:00:00', '2024-05-17T00:00:00','2024-05-18T00:00:00', '2024-05-19T00:00:00', '2024-05-20T00:00:00','2024-05-21T00:00:00', '2024-05-22T00:00:00', '2024-05-23T00:00:00','2024-05-24T00:00:00', '2024-05-25T00:00:00', '2024-05-26T00:00:00','2024-05-27T00:00:00', '2024-05-28T00:00:00', '2024-05-29T00:00:00','2024-05-30T00:00:00', '2024-05-31T00:00:00', '2024-06-01T00:00:00','2024-06-02T00:00:00', '2024-06-03T00:00:00', '2024-06-04T00:00:00','2024-06-05T00:00:00', '2024-06-06T00:00:00', '2024-06-07T00:00:00','2024-06-08T00:00:00', '2024-06-09T00:00:00', '2024-06-10T00:00:00','2024-06-11T00:00:00'], dtype='datetime64[s]')


obs_dim


(obs_dim)


\<U8


'0::117' '0::691' ... '9::4'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['0::117', '0::691', '0::70', ..., '98::267', '99::589', '9::4'],shape=(1000,), dtype='<U8')


Data variables: (1)


obs


(chain, draw, time, obs_dim)


float32


0.09489 0.6722 ... 1.475 0.6046


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 0.09488568,  0.6722131 ,  1.1528677 , ...,  0.3608521 ,0.62936884,  0.07746825],[ 1.0615821 ,  0.71874034,  1.3380088 , ...,  0.21480674,0.54766077,  0.52777964],[ 1.4796339 ,  1.4130722 ,  1.5716678 , ...,  0.23654987,0.82639956,  1.0521206 ],...,[ 1.4127051 ,  0.72606766,  1.1114007 , ...,  1.6873982 ,1.4029967 ,  1.8957828 ],[ 0.9028806 ,  0.7697658 ,  1.5383823 , ...,  1.6141942 ,1.9217857 ,  1.8996933 ],[ 0.9632249 ,  0.05651683,  0.761746  , ...,  1.538854  ,0.90952134,  1.0909214 ]],[[ 0.05661617,  0.9236539 ,  1.0268744 , ...,  0.09166427,0.64716285,  1.336318  ],[ 1.1044478 ,  1.134439  ,  1.0784969 , ...,  0.18125595,0.58157164,  0.6595112 ],[ 1.8264773 ,  1.5527233 ,  1.2951499 , ...,  0.17970549,0.9794669 ,  2.6983528 ],...1.3063308 ,  1.2988968 ],[ 0.8649753 ,  1.0288852 ,  1.501949  , ...,  1.1776692 ,1.3268932 ,  0.8839393 ],[ 0.585702  ,  0.87490857,  1.5517995 , ...,  1.7336043 ,1.2808118 ,  1.1296356 ]],[[-0.00989706,  0.8360924 ,  0.31648713, ...,  0.47984475,0.8471403 ,  0.9642297 ],[ 0.7513049 ,  1.0518695 ,  1.2751569 , ...,  0.2435599 ,0.33145133,  1.2692978 ],[ 1.0334415 ,  0.90518457,  1.2741013 , ...,  0.20082684,1.1675223 ,  1.1041937 ],...,[ 1.4321228 ,  0.45690712,  1.8740747 , ...,  1.8223115 ,1.5109792 ,  2.5885959 ],[ 1.508797  ,  1.3880478 ,  1.7022482 , ...,  1.6485256 ,1.4627122 ,  1.3569854 ],[ 0.8199768 ,  0.81692797,  0.8191836 , ...,  1.7387298 ,1.4745936 ,  0.6046198 ]]]],shape=(1, 1000, 76, 1000), dtype=float32)


Attributes: (5)


created_at :  
2026-07-12T08:01:35.587957+00:00

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


- time: 76
- obs_dim: 1000


Coordinates: (2)


time


(time)


datetime64\[s\]


2024-03-28 ... 2024-06-11


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['2024-03-28T00:00:00', '2024-03-29T00:00:00', '2024-03-30T00:00:00','2024-03-31T00:00:00', '2024-04-01T00:00:00', '2024-04-02T00:00:00','2024-04-03T00:00:00', '2024-04-04T00:00:00', '2024-04-05T00:00:00','2024-04-06T00:00:00', '2024-04-07T00:00:00', '2024-04-08T00:00:00','2024-04-09T00:00:00', '2024-04-10T00:00:00', '2024-04-11T00:00:00','2024-04-12T00:00:00', '2024-04-13T00:00:00', '2024-04-14T00:00:00','2024-04-15T00:00:00', '2024-04-16T00:00:00', '2024-04-17T00:00:00','2024-04-18T00:00:00', '2024-04-19T00:00:00', '2024-04-20T00:00:00','2024-04-21T00:00:00', '2024-04-22T00:00:00', '2024-04-23T00:00:00','2024-04-24T00:00:00', '2024-04-25T00:00:00', '2024-04-26T00:00:00','2024-04-27T00:00:00', '2024-04-28T00:00:00', '2024-04-29T00:00:00','2024-04-30T00:00:00', '2024-05-01T00:00:00', '2024-05-02T00:00:00','2024-05-03T00:00:00', '2024-05-04T00:00:00', '2024-05-05T00:00:00','2024-05-06T00:00:00', '2024-05-07T00:00:00', '2024-05-08T00:00:00','2024-05-09T00:00:00', '2024-05-10T00:00:00', '2024-05-11T00:00:00','2024-05-12T00:00:00', '2024-05-13T00:00:00', '2024-05-14T00:00:00','2024-05-15T00:00:00', '2024-05-16T00:00:00', '2024-05-17T00:00:00','2024-05-18T00:00:00', '2024-05-19T00:00:00', '2024-05-20T00:00:00','2024-05-21T00:00:00', '2024-05-22T00:00:00', '2024-05-23T00:00:00','2024-05-24T00:00:00', '2024-05-25T00:00:00', '2024-05-26T00:00:00','2024-05-27T00:00:00', '2024-05-28T00:00:00', '2024-05-29T00:00:00','2024-05-30T00:00:00', '2024-05-31T00:00:00', '2024-06-01T00:00:00','2024-06-02T00:00:00', '2024-06-03T00:00:00', '2024-06-04T00:00:00','2024-06-05T00:00:00', '2024-06-06T00:00:00', '2024-06-07T00:00:00','2024-06-08T00:00:00', '2024-06-09T00:00:00', '2024-06-10T00:00:00','2024-06-11T00:00:00'], dtype='datetime64[s]')


obs_dim


(obs_dim)


\<U8


'0::117' '0::691' ... '9::4'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['0::117', '0::691', '0::70', ..., '98::267', '99::589', '9::4'],shape=(1000,), dtype='<U8')


Data variables: (1)


obs


(time, obs_dim)


float32


0.06142 0.8523 ... 1.541 0.3627


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.06142242, 0.8522909 , 1.1933497 , ..., 0.20759356, 0.6039184 ,0.2863602 ],[1.0237069 , 1.1932073 , 0.6317734 , ..., 0.20759356, 0.6143307 ,0.45817634],[1.3512931 , 1.6071771 , 1.544335  , ..., 0.20759356, 0.95793945,1.2981663 ],...,[0.83943963, 0.38961872, 1.0529556 , ..., 1.6607485 , 1.541033  ,0.83998996],[0.8803879 , 1.290612  , 1.614532  , ..., 1.7299463 , 1.5826826 ,0.897262  ],[0.57327586, 1.0958027 , 1.3103448 , ..., 1.4462351 , 1.541033  ,0.36272293]], shape=(76, 1000), dtype=float32)


Attributes: (5)


created_at :  
2026-07-12T08:01:35.588481+00:00

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


- input: 5
- time: 76
- series: 1000


Coordinates: (3)


input


(input)


\<U18


'availability' ... 'post_ramp'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['availability', 'discount_magnitude', 'activity_flag', 'holiday_flag','post_ramp'], dtype='<U18')


time


(time)


datetime64\[s\]


2024-03-28 ... 2024-06-11


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['2024-03-28T00:00:00', '2024-03-29T00:00:00', '2024-03-30T00:00:00','2024-03-31T00:00:00', '2024-04-01T00:00:00', '2024-04-02T00:00:00','2024-04-03T00:00:00', '2024-04-04T00:00:00', '2024-04-05T00:00:00','2024-04-06T00:00:00', '2024-04-07T00:00:00', '2024-04-08T00:00:00','2024-04-09T00:00:00', '2024-04-10T00:00:00', '2024-04-11T00:00:00','2024-04-12T00:00:00', '2024-04-13T00:00:00', '2024-04-14T00:00:00','2024-04-15T00:00:00', '2024-04-16T00:00:00', '2024-04-17T00:00:00','2024-04-18T00:00:00', '2024-04-19T00:00:00', '2024-04-20T00:00:00','2024-04-21T00:00:00', '2024-04-22T00:00:00', '2024-04-23T00:00:00','2024-04-24T00:00:00', '2024-04-25T00:00:00', '2024-04-26T00:00:00','2024-04-27T00:00:00', '2024-04-28T00:00:00', '2024-04-29T00:00:00','2024-04-30T00:00:00', '2024-05-01T00:00:00', '2024-05-02T00:00:00','2024-05-03T00:00:00', '2024-05-04T00:00:00', '2024-05-05T00:00:00','2024-05-06T00:00:00', '2024-05-07T00:00:00', '2024-05-08T00:00:00','2024-05-09T00:00:00', '2024-05-10T00:00:00', '2024-05-11T00:00:00','2024-05-12T00:00:00', '2024-05-13T00:00:00', '2024-05-14T00:00:00','2024-05-15T00:00:00', '2024-05-16T00:00:00', '2024-05-17T00:00:00','2024-05-18T00:00:00', '2024-05-19T00:00:00', '2024-05-20T00:00:00','2024-05-21T00:00:00', '2024-05-22T00:00:00', '2024-05-23T00:00:00','2024-05-24T00:00:00', '2024-05-25T00:00:00', '2024-05-26T00:00:00','2024-05-27T00:00:00', '2024-05-28T00:00:00', '2024-05-29T00:00:00','2024-05-30T00:00:00', '2024-05-31T00:00:00', '2024-06-01T00:00:00','2024-06-02T00:00:00', '2024-06-03T00:00:00', '2024-06-04T00:00:00','2024-06-05T00:00:00', '2024-06-06T00:00:00', '2024-06-07T00:00:00','2024-06-08T00:00:00', '2024-06-09T00:00:00', '2024-06-10T00:00:00','2024-06-11T00:00:00'], dtype='datetime64[s]')


series


(series)


\<U8


'0::117' '0::691' ... '9::4'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['0::117', '0::691', '0::70', ..., '98::267', '99::589', '9::4'],shape=(1000,), dtype='<U8')


Data variables: (1)


covariates


(input, time, series)


float32


0.0 0.8421 0.9677 ... 1.0 1.0 1.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.        , 0.8421403 , 0.9677307 , ..., 0.30402568,0.72536546, 0.30627406],[0.9685388 , 0.9519042 , 0.9685388 , ..., 0.18424696,0.58969885, 0.40329787],[1.        , 0.9685388 , 1.        , ..., 0.0914353 ,0.56669945, 0.47482857],...,[0.94906104, 0.40329787, 1.        , ..., 0.7932324 ,0.8911261 , 1.        ],[0.85718024, 0.7174059 , 0.94957167, ..., 0.7932324 ,0.954318  , 1.        ],[0.79585564, 0.970851  , 1.        , ..., 0.85493183,0.9865872 , 0.798104  ]],[[0.        , 0.        , 0.        , ..., 0.        ,0.        , 0.118     ],[0.        , 0.        , 0.        , ..., 0.        ,0.        , 0.118     ],[0.        , 0.003     , 0.        , ..., 0.        ,0.        , 0.118     ],...[1.        , 1.        , 1.        , ..., 1.        ,1.        , 1.        ],[1.        , 1.        , 1.        , ..., 1.        ,1.        , 1.        ],[0.        , 0.        , 0.        , ..., 0.        ,0.        , 0.        ]],[[0.        , 0.        , 0.        , ..., 0.        ,0.        , 0.        ],[0.        , 0.        , 0.        , ..., 0.        ,0.        , 0.        ],[0.        , 0.        , 0.        , ..., 0.        ,0.        , 0.        ],...,[1.        , 1.        , 1.        , ..., 1.        ,1.        , 1.        ],[1.        , 1.        , 1.        , ..., 1.        ,1.        , 1.        ],[1.        , 1.        , 1.        , ..., 1.        ,1.        , 1.        ]]], shape=(5, 76, 1000), dtype=float32)


Attributes: (5)


created_at :  
2026-07-12T08:01:35.588963+00:00

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
- time: 14
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


datetime64\[s\]


2024-06-12 ... 2024-06-25


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['2024-06-12T00:00:00', '2024-06-13T00:00:00', '2024-06-14T00:00:00','2024-06-15T00:00:00', '2024-06-16T00:00:00', '2024-06-17T00:00:00','2024-06-18T00:00:00', '2024-06-19T00:00:00', '2024-06-20T00:00:00','2024-06-21T00:00:00', '2024-06-22T00:00:00', '2024-06-23T00:00:00','2024-06-24T00:00:00', '2024-06-25T00:00:00'], dtype='datetime64[s]')


obs_dim


(obs_dim)


\<U8


'0::117' '0::691' ... '9::4'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['0::117', '0::691', '0::70', ..., '98::267', '99::589', '9::4'],shape=(1000,), dtype='<U8')


Data variables: (1)


obs


(chain, draw, time, obs_dim)


float32


0.7749 1.083 ... 0.9265 0.8734


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 0.7749015 ,  1.0831088 ,  0.87458855, ...,  1.4847032 ,1.2633886 ,  0.5862351 ],[ 0.63880116,  1.1125424 ,  1.0156932 , ...,  1.3194916 ,0.7461579 ,  0.45649508],[ 0.58533174,  0.545473  ,  1.4058175 , ...,  1.2805088 ,1.1481304 ,  1.8601927 ],...,[ 1.5106914 ,  1.1341801 ,  0.9137127 , ...,  1.6485139 ,1.5976317 ,  0.5005521 ],[ 0.79584336,  0.51304734,  0.8295214 , ...,  1.1197894 ,1.3308535 ,  0.0521664 ],[ 0.67939126,  0.3968082 ,  0.7444655 , ...,  1.4539548 ,1.3085897 , -0.2281333 ]],[[-0.02084762,  1.4115049 ,  1.2051075 , ...,  1.3065057 ,1.265929  , -0.08376957],[ 0.04979525,  0.6780671 ,  1.0424573 , ...,  1.7595876 ,1.529109  ,  0.9477426 ],[ 0.5930808 ,  0.8193363 ,  0.6022981 , ...,  1.5234557 ,0.8405203 ,  1.0871837 ],...1.6933969 ,  2.0979178 ],[ 0.4859651 ,  0.55426586,  1.57492   , ...,  1.1374562 ,1.0608088 ,  0.8450963 ],[ 0.8039319 ,  0.68722343,  1.0685695 , ...,  1.6560806 ,1.1583278 ,  0.13175844]],[[ 0.45728678,  0.8845994 ,  0.77249104, ...,  1.5282891 ,1.2765863 ,  0.3156627 ],[ 0.81055915,  0.63560724,  0.76543266, ...,  1.1669785 ,1.1610925 ,  0.0500759 ],[ 0.95232135,  1.174329  ,  1.4621426 , ...,  1.142947  ,0.67617637, -0.03002446],...,[ 1.3844055 ,  1.0044492 ,  0.99870586, ...,  1.743335  ,2.031648  ,  0.64790535],[ 0.8840452 ,  1.2393291 ,  0.6417512 , ...,  0.977619  ,1.2002982 ,  0.21185553],[ 0.967027  ,  0.90793353,  1.4985541 , ...,  1.49228   ,0.92647266,  0.87340575]]]],shape=(1, 1000, 14, 1000), dtype=float32)


Attributes: (5)


created_at :  
2026-07-12T08:01:38.036898+00:00

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


- input: 5
- time: 14
- series: 1000


Coordinates: (3)


input


(input)


\<U18


'availability' ... 'post_ramp'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['availability', 'discount_magnitude', 'activity_flag', 'holiday_flag','post_ramp'], dtype='<U18')


time


(time)


datetime64\[s\]


2024-06-12 ... 2024-06-25


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['2024-06-12T00:00:00', '2024-06-13T00:00:00', '2024-06-14T00:00:00','2024-06-15T00:00:00', '2024-06-16T00:00:00', '2024-06-17T00:00:00','2024-06-18T00:00:00', '2024-06-19T00:00:00', '2024-06-20T00:00:00','2024-06-21T00:00:00', '2024-06-22T00:00:00', '2024-06-23T00:00:00','2024-06-24T00:00:00', '2024-06-25T00:00:00'], dtype='datetime64[s]')


series


(series)


\<U8


'0::117' '0::691' ... '9::4'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['0::117', '0::691', '0::70', ..., '98::267', '99::589', '9::4'],shape=(1000,), dtype='<U8')


Data variables: (1)


covariates


(input, time, series)


float32


0.7831 1.0 0.633 ... 1.0 1.0 1.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.7830641 , 1.        , 0.6330469 , ..., 0.8421403 ,0.8601887 , 0.98133034],[0.9787071 , 1.        , 0.9787071 , ..., 0.7932324 ,0.92897886, 1.        ],[0.912419  , 1.        , 0.94239163, ..., 0.7125343 ,0.96694404, 0.8145253 ],...,[0.8736015 , 0.6884277 , 0.55782074, ..., 0.91093045,0.8523086 , 1.        ],[0.98133034, 0.9835787 , 0.9259704 , ..., 0.7932324 ,0.89900625, 0.73382723],[1.        , 1.        , 0.9787071 , ..., 0.9787071 ,0.89900625, 0.6360862 ]],[[0.        , 0.167     , 0.        , ..., 0.        ,0.        , 0.127     ],[0.        , 0.171     , 0.        , ..., 0.        ,0.        , 0.183     ],[0.        , 0.17      , 0.        , ..., 0.        ,0.        , 0.183     ],...[1.        , 1.        , 1.        , ..., 1.        ,1.        , 1.        ],[0.        , 0.        , 0.        , ..., 0.        ,0.        , 0.        ],[0.        , 0.        , 0.        , ..., 0.        ,0.        , 0.        ]],[[1.        , 1.        , 1.        , ..., 1.        ,1.        , 1.        ],[1.        , 1.        , 1.        , ..., 1.        ,1.        , 1.        ],[1.        , 1.        , 1.        , ..., 1.        ,1.        , 1.        ],...,[1.        , 1.        , 1.        , ..., 1.        ,1.        , 1.        ],[1.        , 1.        , 1.        , ..., 1.        ,1.        , 1.        ],[1.        , 1.        , 1.        , ..., 1.        ,1.        , 1.        ]]], shape=(5, 14, 1000), dtype=float32)


Attributes: (5)


created_at :  
2026-07-12T08:01:38.037527+00:00

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


# Forecast evaluation: CRPS and coverage

We score on the original sales scale (rescaling the draws by each series' training mean and clipping negatives at zero, since sales are non-negative). CRPS is a proper scoring rule for probabilistic forecasts that generalizes the mean absolute error; coverage checks calibration by asking how often the central \\94\\\\ and \\50\\\\ intervals contain the truth. One terminology note to keep the sections consistent: the coverage metrics score *central* (equal-tailed) intervals bounded by fixed quantiles, while the forecast figures further below draw *HDI* bands; for a near-symmetric predictive the two nearly coincide, but on zero-clipped stockout days, where the predictive piles mass at zero, they can differ. As a reference point we use a seasonal-naive ensemble: the weekday-aligned \\14\\-day windows from the training data, stacked as an empirical forecast distribution.

We score with \\1{,}000\\ predictive draws obtained through the functional API ([draw_posterior](../../../reference/functional.posterior.draw_posterior.md#numpyro_forecast.functional.posterior.draw_posterior), [predict_in_sample](../../../reference/functional.prediction.predict_in_sample.md#numpyro_forecast.functional.prediction.predict_in_sample), [forecast](../../../reference/functional.prediction.forecast.md#numpyro_forecast.functional.prediction.forecast)), the same draw count the DataTree export above uses. The count is set by the far tails: each \\3\\\\ tail of the central \\94\\\\ interval rests on about \\30\\ of the \\1{,}000\\ draws, which makes the tail quantiles the noisiest part of the whole evaluation. On this panel the estimate is nevertheless comfortable: rescoring with only the first \\500\\ draws moves both coverages by about \\0.002\\.


    In [31]:


``` python
def seasonal_naive_ensemble(
    y_history: Float[np.ndarray, " t_hist n_series"],
    horizon: int,
    period: int = 7,
) -> Float[np.ndarray, " n_windows horizon n_series"]:
    """Stack weekday-aligned historical windows as an empirical forecast ensemble.

    Window ``k`` starts ``k * period`` days before the end of the history and spans
    ``horizon`` days, so every window is weekday-aligned with the forecast window.
    Windows that would run past the end of the history are dropped (for
    ``horizon > period`` the most recent start is skipped).
    """
    t_hist = y_history.shape[0]
    windows = []
    k = 1
    while t_hist - k * period >= 0:
        start = t_hist - k * period
        if start + horizon <= t_hist:
            windows.append(y_history[start : start + horizon])
        k += 1
    return np.stack(windows)


def metrics_table(
    pred_train: Float[Array, " sample t_train n_series"],
    pred_test: Float[Array, " sample horizon n_series"],
    naive_test: Float[Array, " n_windows horizon n_series"],
    y_train_true: Float[Array, " t_train n_series"],
    y_test_true: Float[Array, " horizon n_series"],
) -> pl.DataFrame:
    """Build the CRPS and coverage summary for train, test, and the naive baseline.

    All inputs are on the original sales scale; coverage is not meaningful for the
    small naive ensemble, so its cells stay null.
    """
    return pl.DataFrame(
        {
            "split": ["model (train)", "model (test)", "seasonal naive (test)"],
            "crps": [
                float(eval_crps(pred_train, y_train_true)),
                float(eval_crps(pred_test, y_test_true)),
                float(eval_crps(naive_test, y_test_true)),
            ],
            "coverage_94": [
                float(eval_coverage(pred_train, y_train_true, alpha=0.94)),
                float(eval_coverage(pred_test, y_test_true, alpha=0.94)),
                None,
            ],
            "coverage_50": [
                float(eval_coverage(pred_train, y_train_true, alpha=0.5)),
                float(eval_coverage(pred_test, y_test_true, alpha=0.5)),
                None,
            ],
        }
    )


rng_key, key_score_post, key_score_in, key_score_fc = random.split(rng_key, 4)
posterior_draws = draw_posterior(key_score_post, svi_fit, 1_000)
pp_scaled = predict_in_sample(key_score_in, model, posterior_draws, covariates_train)
fc_scaled = forecast(key_score_fc, model, posterior_draws, y_train, covariates)
pred_train = jnp.clip(pp_scaled * scale_jax[None, None, :], min=0.0)
pred_test = jnp.clip(fc_scaled * scale_jax[None, None, :], min=0.0)
naive_test = jnp.asarray(
    seasonal_naive_ensemble(
        panel_ds["sale_amount"].isel(time=slice(None, t_train)).to_numpy(), horizon=horizon
    ),
    dtype=jnp.float32,
)
# Named views of the scored ensembles for the diagnostics and plots below; the
# metric helpers keep consuming the raw arrays.
pred_train_da = draws_to_da(pred_train, dates[:t_train])
pred_test_da = draws_to_da(pred_test, dates[t_train:])
y_test_da = panel_ds["sale_amount"].isel(time=slice(t_train, None))

results_df = metrics_table(pred_train, pred_test, naive_test, y_train_original, y_test_original)
crps_test_model = float(results_df["crps"][1])
crps_test_naive = float(results_df["crps"][2])
results_df
```


| split                   | crps     | coverage_94 | coverage_50 |
|-------------------------|----------|-------------|-------------|
| "model (train)"         | 0.855828 | 0.979947    | 0.687224    |
| "model (test)"          | 1.260417 | 0.9085      | 0.514429    |
| "seasonal naive (test)" | 2.387625 | null        | null        |


The model beats the seasonal-naive baseline on test CRPS by a wide margin. Calibration is more nuanced: on the holdout the \\50\\\\ interval covers \\51\\\\, essentially nominal, while the \\94\\\\ interval covers \\91\\\\, a modest under-coverage; in-sample, both intervals *over*-cover (\\0.98\\ and \\0.69\\). The per-day diagnostics below show that these aggregates hide structure worth dissecting. Before that, two of the modeling choices above earn their place directly in these numbers:

- **The damped trend is what keeps the coverage from decaying with the horizon.** Without it (a pure random-walk level), the median forecast percentile of the truth drifts from about \\0.45\\ on day \\1\\ to \\0.84\\ by day \\14\\ while the forecast fan barely widens: the frozen level cannot extrapolate the panel's upward drift, so the interval misses pile up above the bands. With the slope, test CRPS and both coverages improve together, most visibly on the late-horizon days (the level-dependent noise term \\\lambda_s \\ \text{softplus}(\ell\_{t,s})\\ plays the same role for the in-sample spread).
- **The cleaned discount encoding and the launch indicator remove a spurious optimum.** Without them, the placeholder discount days hand the optimizer a second ELBO optimum in which a launch-aligned discount step absorbs each series' launch jump with coefficients an order of magnitude too large, and which basin a run lands in depends on nothing more than the compilation path of the update loop (the progress-bar path lands badly; the scanned path lands well). With them, no execution path produces runaway coefficients, and the store-hierarchy plot further below hugs the identity line. The optimization as such remains sensitive to the update-loop compilation on a panel this large, which is why the fit above pins the well-behaved `lax.scan` path with `progress_bar=False`.

One artifact to rule out before reading the coverage numbers at face value is the point mass at zero: the draws are clipped at zero and sales are exactly zero on stockout days, so whenever the interval's lower edge touches zero a zero-sales day is covered "for free", which could flatter the coverage without the forecast earning it. The panel makes this easy to check:


    In [32]:


``` python
zero_mask = y_test_da == 0.0
quantiles = pred_test_da.quantile([0.03, 0.25, 0.75, 0.97], dim="sample")
inside50 = (y_test_da >= quantiles.sel(quantile=0.25)) & (
    y_test_da <= quantiles.sel(quantile=0.75)
)
inside94 = (y_test_da >= quantiles.sel(quantile=0.03)) & (
    y_test_da <= quantiles.sel(quantile=0.97)
)
print(f"test observations with zero sales: {float(zero_mask.mean()):.1%}")
print(
    f"50% coverage | zero-sales days: {float(inside50.where(zero_mask).mean()):.2f} | "
    f"positive days: {float(inside50.where(~zero_mask).mean()):.2f}"
)
print(
    f"94% coverage | zero-sales days: {float(inside94.where(zero_mask).mean()):.2f} | "
    f"positive days: {float(inside94.where(~zero_mask).mean()):.2f}"
)
```


    test observations with zero sales: 1.4%
    50% coverage | zero-sales days: 0.34 | positive days: 0.52
    94% coverage | zero-sales days: 0.82 | positive days: 0.91


The artifact is ruled out. Zero-sales days are rare in this test panel (\\1.4\\\\: these are the top sellers, and the test window sits after the launch with mostly high availability), and on them the intervals cover *less* than nominal, since the factor floor and the level often push the whole central band strictly above zero. The positive-sales days (\\0.52\\ and \\0.91\\) sit almost exactly at the panel-wide coverages, so the aggregate numbers reflect ordinary days, not zero-day bookkeeping. What the aggregates do hide is a drift over the horizon, which the per-day plots below make visible; the in-sample over-coverage already hints at one half of the story (daily sales fluctuations are heavier-tailed than a Normal, so the fitted noise scale widens the whole bell to accommodate the tail days, and in-sample the central band over-covers at \\0.69\\). The per-day breakdown shows where the CRPS margin comes from:


    In [33]:


``` python
def crps_by_day(
    pred: Float[Array, " sample horizon n_series"],
    y_true: Float[Array, " horizon n_series"],
) -> xr.DataArray:
    """Mean CRPS per forecast day, averaged over the series of the test window."""
    per_day_series = xr.DataArray(
        np.asarray(crps_empirical(pred, y_true)),
        dims=["time", "series"],
        coords={"time": dates[t_train:], "series": series_ids},
    )
    return per_day_series.mean("series")


crps_by_day_model = crps_by_day(pred_test, y_test_original)
crps_by_day_naive = crps_by_day(naive_test, y_test_original)

fig, ax = plt.subplots()
ax.plot(np.arange(1, horizon + 1), crps_by_day_model, marker="o", color="C0", label="model")
ax.plot(
    np.arange(1, horizon + 1), crps_by_day_naive, marker="s", color="C1", label="seasonal naive"
)
ax.legend(loc="upper left")
ax.set(
    xlabel="forecast day",
    ylabel="mean CRPS",
    title="Test CRPS by forecast horizon day",
    xticks=np.arange(1, horizon + 1),
);
```


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-34-output-1.png" class="figure-img" width="1211" height="711" /></p>
</figure>


The coverage diagnostic below resolves the calibration story day by day: observed central-interval coverage per forecast day against the nominal levels. Both intervals start the horizon *above* their nominal line, the in-sample over-coverage carrying over into the first few days, and then drift down through it as the horizon grows. The aggregate \\50\\\\ coverage lands on nominal only because these two regimes cancel in the average, a coincidence the next diagnostic unpacks.


    In [34]:


``` python
forecast_days = np.arange(1, horizon + 1)

fig, ax = plt.subplots()
for alpha, color, label in ((0.94, "C0", r"$94\%$ interval"), (0.5, "C1", r"$50\%$ interval")):
    coverage_by_day = np.array(
        [
            float(eval_coverage(pred_test[:, day, :], y_test_original[day], alpha=alpha))
            for day in range(horizon)
        ]
    )
    ax.plot(forecast_days, coverage_by_day, marker="o", color=color, label=f"observed, {label}")
    ax.axhline(alpha, color=color, linestyle=":", linewidth=1.5, label=f"nominal, {label}")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
ax.set(
    xlabel="forecast day",
    ylabel="observed coverage",
    title="Test interval coverage by forecast horizon day",
    xticks=forecast_days,
    ylim=(0.0, 1.05),
);
```


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-35-output-1.png" class="figure-img" width="1211" height="711" /></p>
</figure>


## Interval diagnostics

Two sharper views of the same calibration question. The top panel tracks the PIT, the fraction of forecast draws below the observed value (ties, which the zero clipping makes common, count half), by horizon day: a value of \\0.5\\ means the truth sits at the forecast median, and a calibrated forecast keeps the interquartile band centered on \\0.5\\. The bottom panel splits the \\94\\\\-interval misses by direction against the nominal \\3\\\\ per side; this is where a trend miss shows up most directly, since a level that cannot extrapolate drift produces an above-side excess that grows with the horizon.


    In [35]:


``` python
pit = (pred_test_da < y_test_da).mean("sample") + 0.5 * (pred_test_da == y_test_da).mean("sample")
above_share = (y_test_da > quantiles.sel(quantile=0.97)).mean("series")
below_share = (y_test_da < quantiles.sel(quantile=0.03)).mean("series")

fig, axes = plt.subplots(nrows=2, figsize=(12, 9), sharex=True, layout="constrained")
axes[0].plot(forecast_days, pit.median("series"), "o-", color="C0", label="median PIT")
axes[0].fill_between(
    forecast_days,
    pit.quantile(0.25, dim="series"),
    pit.quantile(0.75, dim="series"),
    color="C0",
    alpha=0.3,
    label="interquartile range",
)
axes[0].axhline(0.5, color="gray", linestyle=":", label="calibrated median")
axes[0].legend(loc="upper left")
axes[0].set(
    ylabel="PIT",
    title="Forecast PIT by horizon day",
    xticks=forecast_days,
    ylim=(0.0, 1.0),
)
axes[1].plot(forecast_days, above_share, "o-", color="C1", label="misses above the interval")
axes[1].plot(forecast_days, below_share, "s-", color="C0", label="misses below the interval")
axes[1].axhline(0.03, color="gray", linestyle=":", label=r"nominal ($3\%$ per side)")
axes[1].legend(loc="upper left")
axes[1].set(
    xlabel="forecast day",
    ylabel="share of series",
    title=r"$94\%$-interval miss direction",
    xticks=forecast_days,
)
fig.suptitle("Interval diagnostics on the test window", fontsize=18, fontweight="bold");
```


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-36-output-1.png" class="figure-img" width="1211" height="911" /></p>
</figure>


The two panels pin the story down. The median PIT starts just below \\0.5\\ and drifts upward through the horizon, and the miss directions are sharply asymmetric: below-side misses stay at or under the nominal \\3\\\\ on every day, while above-side misses cross nominal around day \\4\\ and reach the mid-teens in the second week. That asymmetry says the \\94\\\\ under-coverage is a *directional* miss, not a band that is uniformly too narrow: a merely narrow interval would leak on both sides (mean-field variational inference's tendency toward too-narrow posteriors can contribute to the level, but it cannot explain the one-sidedness). Without the damped trend these curves are far worse (median PIT \\0.84\\ and above-misses at \\0.24\\ by day \\14\\); with it much of the drift is gone, but the late days still run hot: the panel's momentum in the test window is at the upper end of what the damped slope extrapolates. The same drift explains the coverage cancellation noted above: early days over-cover with the heavy-tail-widened band, late days under-cover as the truth walks out the top, and the \\50\\\\ aggregate lands on nominal by coincidence rather than by calibration, which is exactly why the directional diagnostics are worth plotting next to the averages. A post-hoc interval calibration would target that residual drift directly; we leave it on the next-steps list rather than pursue it in this notebook.


## Scaling belongs inside the fold

One methodological remark before leaving the evaluation. The per-series scale was computed once, from the training window of our single split, and that is sound because there is only one split. The moment this evaluation graduates to rolling-origin backtesting, that global step becomes a leak: each fold has a different training window, and a scale computed outside the fold loop (worse, on the full series) feeds the fold information about levels it has not seen yet, exactly the leakage the scaling section warned about. The normalization is part of the model pipeline, and in a backtest the pipeline must run once per fold.

The package's [backtest](../../../reference/evaluate.backtest.md#numpyro_forecast.evaluate.backtest) helper leaves room for exactly this: its `forecaster_fn` is any callable `(rng_key, model, data, covariates, **options)` returning a fitted forecaster, and it slices the *raw* data per window before calling it. So the clean way to fold the scaling in is a [Forecaster](../../../reference/forecaster.Forecaster.md#numpyro_forecast.forecaster.Forecaster) subclass that derives the scale from whatever training window it is handed, fits on the scaled data, and returns forecasts on the original scale. We define it here but do not run it (the single split above is already scored); the next steps point to it for the backtesting extension.


    In [36]:


``` python
class ScaledForecaster(Forecaster):
    """SVI forecaster that owns the per-series mean scaling as a fit-time step.

    Computes the per-series scale from the training window it is handed, fits
    the model on the scaled data, and returns forecast draws rescaled to the
    original units (clipped at zero, since sales are non-negative). Because the
    scale is derived inside ``__init__`` from the training data alone, passing
    this class to ``backtest(..., forecaster_fn=ScaledForecaster)`` recomputes
    the normalization inside every fold, so no future level can leak into the
    model input.

    Parameters
    ----------
    rng_key
        PRNG key for inference.
    model
        The forecasting model to fit.
    data
        Raw (unscaled) training data with time at axis ``-2``.
    covariates
        Covariates with time at axis ``-2`` and the same duration as ``data``.
    **options
        Forwarded to :class:`numpyro_forecast.Forecaster` (``guide``, ``optim``,
        ``num_steps``, ...).
    """

    def __init__(
        self,
        rng_key: Array,
        model: ForecastingModel,
        data: Float[Array, " t_train n_series"],
        covariates: Float[Array, " n_inputs t_train n_series"],
        **options: Any,
    ) -> None:
        self.scale: Float[Array, " n_series"] = data.mean(axis=-2)
        super().__init__(rng_key, model, data / self.scale, covariates, **options)

    def __call__(
        self,
        rng_key: Array,
        data: Float[Array, " t_train n_series"],
        covariates: Float[Array, " n_inputs duration n_series"],
        num_samples: int,
        *,
        batch_size: int | None = None,
        parallel: bool = True,
    ) -> Float[Array, " sample horizon n_series"]:
        """Forecast on the original sales scale (draws rescaled and zero-clipped)."""
        pred = super().__call__(
            rng_key,
            data / self.scale,
            covariates,
            num_samples,
            batch_size=batch_size,
            parallel=parallel,
        )
        return jnp.clip(pred * self.scale, min=0.0)
```


# Forecast visualization

The in-sample posterior predictive (blue) and the \\14\\-day forecast (orange) for the same series we explored before modeling (the ten largest by volume and the ten with the most zero-availability days), with the \\50\\\\ and \\94\\\\ HDI bands, the observed sales in black, and the availability input in red on a secondary axis. Note how the bands collapse toward zero whenever availability drops, including in the forecast window: the factor propagates the known future availability into the predictive distribution. The next section removes exactly that ingredient to forecast demand instead of sales, and reuses the same panel layout, so the plotting code lives in a small helper that takes the test-window ensemble (and the forecast bands' color and legend label) as arguments.


    In [37]:


``` python
def plot_forecast_panel(
    pred_test_draws: xr.DataArray,
    availability: xr.DataArray,
    forecast_color: str,
    forecast_label: str,
    suptitle: str,
) -> None:
    """Facet the in-sample predictive and a test-window ensemble for the EDA series.

    Draws the in-sample posterior predictive bands (blue), the test-window
    forecast bands, the observed sales, the train-test split, and the
    availability on a twin axis, one row per series in ``plot_labels``. The
    in-sample ensemble is the same for both callers, so the helper reads
    ``pred_train_da`` (and the panel scaffolding) from the enclosing scope and
    parametrizes only what changes between the two figures.

    Parameters
    ----------
    pred_test_draws
        Test-window predictive draws on the original sales scale, dims
        ``(sample, time, series)``.
    availability
        Availability drawn on the twin axis, dims ``(time, series)`` over the
        full window. Pass the availability input the test-window predictions
        actually consumed, so the figure represents the features behind the
        forecast.
    forecast_color
        Matplotlib color for the test-window HDI bands.
    forecast_label
        Legend prefix for the test-window bands.
    suptitle
        Figure title.
    """
    pc = az.plot_lm(
        predictions_to_datatree(
            pred_train_da.sel(series=plot_labels).to_numpy(), dates_num[:t_train], plot_labels
        ),
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
        figure_kwargs={"figsize": (15, 2.5 * len(plot_labels))},
    )
    train_bands = pc.viz["ci_band"]["t"].sel(series=plot_labels[-1])
    az.plot_lm(
        predictions_to_datatree(
            pred_test_draws.sel(series=plot_labels).to_numpy(), dates_num[t_train:], plot_labels
        ),
        y="obs",
        x="t",
        plot_dim="time",
        plot_collection=pc,
        ci_kind="hdi",
        ci_prob=hdi_probs,
        smooth=False,
        visuals={
            "ci_band": {"color": forecast_color},
            "observed_scatter": False,
            "pe_line": False,
            "xlabel": False,
            "ylabel": False,
        },
    )

    truth_da = (
        panel_ds["sale_amount"].sel(series=plot_labels).assign_coords(time=dates_num).rename("t")
    )
    x_da = xr.DataArray(dates_num, dims=["time"], coords={"time": dates_num})
    pc.map(
        az.visuals.line_xy,
        "truth",
        data=truth_da,
        x=x_da,
        ignore_aes=pc.aes_set,
        color="black",
        lw=1.5,
    )

    for label in plot_labels:
        ax = pc.get_target("t", {"series": label})
        ax.set_title(label, fontsize=11)
        split_line = ax.axvline(split_x, color="C3", linestyle="--", linewidth=1)
        ax_twin = ax.twinx()
        (availability_line,) = ax_twin.plot(
            dates_num, availability.sel(series=label), color="red", linewidth=1.5
        )
        ax_twin.grid(False)
        ax_twin.set(ylim=(0, 1.05))
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    test_bands = pc.viz["ci_band"]["t"].sel(series=plot_labels[-1])
    band_handles = []
    for bands, prefix in ((train_bands, "in-sample "), (test_bands, f"{forecast_label} ")):
        for prob in (0.94, 0.5):
            band = bands.sel(prob=prob).item()
            band.set_label(hdi_label(prob, prefix=prefix))
            band_handles.append(band)
    truth_line = pc.viz["truth"]["t"].sel(series=plot_labels[-1]).item()
    truth_line.set_label("observed sales")
    split_line.set_label("train-test split")
    availability_line.set_label("availability")

    fig = pc.viz["figure"].item()
    ax_last = pc.get_target("t", {"series": plot_labels[-1]})
    ax_last.legend(
        handles=[*band_handles, truth_line, split_line, availability_line],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncols=4,
        fontsize=11,
    )
    fig.supxlabel("date")
    fig.supylabel("sale amount")
    fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.02)


plot_forecast_panel(
    pred_test_da,
    panel_ds["availability"],
    forecast_color="C1",
    forecast_label="forecast",
    suptitle=(
        f"FreshRetailNet forecasts (test CRPS {crps_test_model:.2f} "
        f"vs seasonal naive {crps_test_naive:.2f})"
    ),
)
```


    /Users/juanitorduz/Documents/numpyro_forecast/.venv/lib/python3.14/site-packages/arviz_plots/plots/lm_plot.py:360: UserWarning: When multiple credible intervals are plotted, it is recommended to map 'alpha' aesthetic to 'prob' dimension to differentiate between intervals.
      warnings.warn(


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-38-output-2.png" class="figure-img" width="1511" height="5115" /></p>
</figure>


# From sales to demand: forecasting at full availability

The forecast above answers the question the *evaluation* needed: what will sales be, given the availability the test window actually recorded. That is the right conditioning for retrospective scoring, but it is not a forecast a business can act on, for two reasons. First, nobody knows future availability at prediction time; the retrospective setup borrows it from the recorded data. Second, and more fundamentally, a planner who orders against a stockout-censored sales forecast bakes yesterday's stockouts into tomorrow's order quantities: the model predicts low sales where availability dipped, the buyer orders little, and the stockout repeats itself. What replenishment needs is the quantity the factorization was built to expose: the *demand*, what would sell with the product fully on the shelf.

Because expected sales factor into demand times availability, that counterfactual is one covariate edit away: pin the availability input to one over the forecast horizon and rerun the same [forecast](../../../reference/functional.prediction.forecast.md#numpyro_forecast.functional.prediction.forecast) call with the same posterior draws; the trend, seasonality, promotion, and launch inputs stay untouched. The factor's normalization makes the reading exact: \\f\_{t,s} = 1\\ at \\a\_{t,s} = 1\\, so the demand forecast coincides with the sales forecast on the days the shelf was in fact fully stocked, and rises above it exactly where availability dipped. We also reuse the *same* PRNG key as the sales forecast, so the two ensembles share their predictive noise draws (common random numbers) and their difference is purely the availability correction, not Monte Carlo noise.

One thing this forecast deliberately is *not*: a prediction of the observed test sales. Observed sales are censored by the very stockouts we are removing, so on stockout days the demand forecast *should* sit above the black line, and scoring it against observed sales (as the CRPS table did for the sales forecast) would penalize it for being right. The printouts quantify how much demand the sales forecast leaves on the table over the test window.


    In [38]:


``` python
covariates_demand_da: xr.DataArray = covariates_da.copy(deep=True)
covariates_demand_da.loc[{"input": "availability", "time": slice(dates[t_train], None)}] = 1.0
covariates_demand: Float[
    Array, " availability_discount_activity_holiday_ramp duration n_series"
] = jnp.asarray(
    covariates_demand_da.transpose("input", "time", "series").to_numpy(), dtype=jnp.float32
)

fc_demand_scaled = forecast(key_score_fc, model, posterior_draws, y_train, covariates_demand)
pred_test_demand = jnp.clip(fc_demand_scaled * scale_jax[None, None, :], min=0.0)
pred_test_demand_da = draws_to_da(pred_test_demand, dates[t_train:])

demand_total = pred_test_demand_da.mean("sample").sum("time")
sales_total = pred_test_da.mean("sample").sum("time")
series_uplift = demand_total / sales_total - 1
panel_uplift = float(demand_total.sum() / sales_total.sum() - 1)
print(f"expected demand above the sales forecast on the test window: {panel_uplift:+.1%}")
print(f"series with an uplift above 1%: {float((series_uplift > 0.01).mean()):.1%} of the panel")
print(
    f"largest per-series uplift: {float(series_uplift.max()):+.1%} "
    f"(series {series_uplift.idxmax().item()})"
)
```


    expected demand above the sales forecast on the test window: +8.8%
    series with an uplift above 1%: 84.3% of the panel
    largest per-series uplift: +168.7% (series 438::300)


The correction is meaningful in aggregate, roughly \\9\\\\ of the forecast test-window volume, and its anatomy follows the saturating factor: near full availability the factor is almost flat, so a day that loses a few sales-weighted hours contributes nothing visible, while a day that drops to low availability contributes a lot. Deep dips are scattered widely across the panel's two forecast weeks, so the uplift is broad (\\84\\\\ of the series gain more than \\1\\\\) but very uneven, running past \\+150\\\\ for the most stockout-prone series. The faceted view below shows this series by series, in the same layout as the forecast plot above but with the demand bands in green. One detail changes deliberately: the red availability line now shows the *input these predictions actually consumed*, the observed availability in-sample and a constant one over the forecast window, because a plot of a forecast should represent the features that produced it. To see where availability actually dipped in the test window, compare with the sales-forecast panel above; the single-series comparison further below makes that contrast explicit.


    In [39]:


``` python
plot_forecast_panel(
    pred_test_demand_da,
    covariates_demand_da.sel(input="availability"),
    forecast_color="C2",
    forecast_label="demand forecast",
    suptitle="FreshRetailNet demand forecasts at full availability (planning view)",
)
```


    /Users/juanitorduz/Documents/numpyro_forecast/.venv/lib/python3.14/site-packages/arviz_plots/plots/lm_plot.py:360: UserWarning: When multiple credible intervals are plotted, it is recommended to map 'alpha' aesthetic to 'prob' dimension to differentiate between intervals.
      warnings.warn(


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-40-output-2.png" class="figure-img" width="1511" height="5115" /></p>
</figure>


On fully stocked days the green bands reproduce the orange ones exactly (shared noise draws, factor pinned at one), so nothing is lost by planning on the demand forecast. Where availability dropped in the test window the demand bands detach upward from the observed sales, and that gap is the model's estimate of the unmet demand behind the stockout.


## Zooming in: the two forecasts on a decaying-availability series

The panel view compresses fourteen days into a thin strip, so let us zoom into the series where the counterfactual matters most in this test window: `22::267`, whose recorded availability drops sharply late in the forecast window, down to \\0.42\\ on the worst day. The two rows below show the test window only, on a shared sales axis: the top row is the sales forecast conditioned on the observed availability, the bottom row the demand forecast at availability one, and the red line in each row is the availability input that row's forecast consumed.


    In [40]:


``` python
comparison_id = "22::267"
comparison_labels = ["forecast with observed availability", "forecast with availability = 1"]
comparison_dim = xr.DataArray(comparison_labels, dims=["series"], name="series")
comparison_da = xr.concat(
    [
        pred_test_da.sel(series=comparison_id, drop=True),
        pred_test_demand_da.sel(series=comparison_id, drop=True),
    ],
    dim=comparison_dim,
)
availability_test_comp = (
    panel_ds["availability"].isel(time=slice(t_train, None)).sel(series=comparison_id, drop=True)
)
comparison_availability = xr.concat(
    [availability_test_comp, xr.ones_like(availability_test_comp)],
    dim=comparison_dim,
)

sales_mean_comp = pred_test_da.sel(series=comparison_id).mean("sample")
demand_mean_comp = pred_test_demand_da.sel(series=comparison_id).mean("sample")
uplift_comp = float(demand_mean_comp.sum() / sales_mean_comp.sum() - 1)
worst_day = (demand_mean_comp - sales_mean_comp).idxmax("time")
print(
    f"{comparison_id} | expected test-window sales {float(sales_mean_comp.sum()):.0f} units | "
    f"expected demand {float(demand_mean_comp.sum()):.0f} units ({uplift_comp:+.1%})"
)
print(
    f"largest daily gap on {worst_day.to_numpy().astype('datetime64[D]')} "
    f"(availability "
    f"{float(panel_ds['availability'].sel(series=comparison_id, time=worst_day)):.2f}): "
    f"expected sales {float(sales_mean_comp.sel(time=worst_day)):.1f} "
    f"vs demand {float(demand_mean_comp.sel(time=worst_day)):.1f}"
)

pc = az.plot_lm(
    predictions_to_datatree(
        comparison_da.transpose("sample", "time", "series").to_numpy(),
        dates_num[t_train:],
        comparison_labels,
    ),
    y="obs",
    x="t",
    plot_dim="time",
    ci_kind="hdi",
    ci_prob=hdi_probs,
    smooth=False,
    col_wrap=1,
    visuals={
        "ci_band": {"color": "C1"},
        "observed_scatter": False,
        "pe_line": False,
        "xlabel": False,
        "ylabel": False,
    },
    aes={"alpha": ["prob"]},
    alpha=hdi_alphas,
    figure_kwargs={"figsize": (12, 7)},
)
for prob in (0.94, 0.5):
    pc.viz["ci_band"]["t"].sel(series=comparison_labels[-1], prob=prob).item().set_color("C2")

truth_da = (
    y_test_da.sel(series=comparison_id, drop=True)
    .expand_dims(series=comparison_labels)
    .transpose("time", "series")
    .assign_coords(time=dates_num[t_train:])
    .rename("t")
)
x_da = xr.DataArray(dates_num[t_train:], dims=["time"], coords={"time": dates_num[t_train:]})
pc.map(
    az.visuals.line_xy,
    "truth",
    data=truth_da,
    x=x_da,
    ignore_aes=pc.aes_set,
    color="black",
    lw=1.5,
)

comparison_axes = [pc.get_target("t", {"series": label}) for label in comparison_labels]
shared_top = max(ax.get_ylim()[1] for ax in comparison_axes)
for ax, label in zip(comparison_axes, comparison_labels, strict=True):
    ax.set_title(label, fontsize=12)
    ax.set_ylim(0.0, shared_top)
    ax_twin = ax.twinx()
    (availability_line,) = ax_twin.plot(
        dates_num[t_train:],
        comparison_availability.sel(series=label),
        color="red",
        alpha=0.8,
        linewidth=1.5,
    )
    ax_twin.grid(False)
    ax_twin.set(ylim=(0, 1.05), ylabel="availability")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

legend_handles = []
for label, prefix in zip(comparison_labels, ("sales forecast ", "demand forecast "), strict=True):
    for prob in (0.94, 0.5):
        band = pc.viz["ci_band"]["t"].sel(series=label, prob=prob).item()
        band.set_label(hdi_label(prob, prefix=prefix))
        legend_handles.append(band)
truth_line = pc.viz["truth"]["t"].sel(series=comparison_labels[-1]).item()
truth_line.set_label("observed sales")
availability_line.set_label("availability input")
legend_handles += [truth_line, availability_line]
comparison_axes[-1].legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncols=3,
    fontsize=12,
)
fig = pc.viz["figure"].item()
fig.supxlabel("date")
fig.supylabel("sale amount")
fig.suptitle(
    f"Sales vs demand forecast for series {comparison_id} (test window)",
    fontsize=18,
    fontweight="bold",
    y=1.05,
);
```


    22::267 | expected test-window sales 388 units | expected demand 417 units (+7.6%)
    largest daily gap on 2024-06-22 (availability 0.42): expected sales 18.3 vs demand 34.6


    /Users/juanitorduz/Documents/numpyro_forecast/.venv/lib/python3.14/site-packages/arviz_plots/plots/lm_plot.py:360: UserWarning: When multiple credible intervals are plotted, it is recommended to map 'alpha' aesthetic to 'prob' dimension to differentiate between intervals.
      warnings.warn(


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-41-output-3.png" class="figure-img" width="1211" height="750" /></p>
</figure>


The comparison makes the counterfactual concrete, and the printout puts numbers on it. In the top row the orange bands are pulled down exactly where the availability input dips, most sharply on \\2024\\-\\06\\-\\22\\: the model expects the stockout to censor sales, and that censored view is precisely what makes the forecast scoreable against the observed black line. In the bottom row the green bands hold the underlying demand level through those same days, because the input that produced them says the shelf never empties; elsewhere the two rows nearly coincide, since availability sits close to one. On the worst day the expected demand (\\34.6\\ units) is nearly twice the expected sale (\\18.3\\ units), and over the full window the demand forecast carries \\7.6\\\\ more volume for this series. That gap is the demand a planner would silently forfeit by ordering to the censored forecast, and the stockout would then repeat itself by construction. This demand fan, not the sales forecast, is the input a replenishment decision should consume; the sales forecast's job was to be scoreable against what was actually observed.


# Inspecting the availability factor

The factor parameters are per series, so we can ask what the model actually learned about stockouts. First the floor \\\phi_s\\ and the saturation rate \\b_s\\ across the panel:


    In [41]:


``` python
floor_mean = tree["posterior"]["floor"].mean(dim=("chain", "draw"))
b_avail_mean = tree["posterior"]["b_avail"].mean(dim=("chain", "draw"))

train_availability = panel_ds["availability"].isel(time=slice(None, t_train))
train_scaled_sales = y_scaled.isel(time=slice(None, t_train))
panel_floor = float(train_scaled_sales.where(train_availability == 0.0).mean())

fig, axes = plt.subplots(ncols=2, figsize=(14, 5), layout="constrained")
axes[0].hist(floor_mean.to_numpy(), bins=40, color="C0", label="posterior mean per series")
axes[0].axvline(panel_floor, color="C3", linestyle="--", label="panel empirical floor")
axes[0].legend(loc="upper right")
axes[0].set(xlabel=r"floor $\phi_s$", ylabel="number of series", title="Availability floor")
axes[1].hist(b_avail_mean.to_numpy(), bins=40, color="C1", label="posterior mean per series")
axes[1].legend(loc="upper right")
axes[1].set(
    xlabel=r"saturation rate $b_s$",
    ylabel="number of series",
    title="Availability saturation rate",
)
fig.suptitle("Posterior availability-factor parameters", fontsize=16, fontweight="bold");
```


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-42-output-1.png" class="figure-img" width="1411" height="511" /></p>
</figure>


And the implied factor curve, averaged over series, against the panel's own empirical curve, with the per-series posterior-mean curves of the six focus series in gray for scale. To compare the two shapes on equal footing, the binned means are rescaled so that the top availability bin equals one: the factor is anchored at \\f(1) = 1\\, while raw scaled sales on fully available days average above one on this launch-driven panel (post-launch days have both high availability and a high level).


    In [42]:


``` python
posterior_factor = (
    tree["posterior"]
    .dataset[["floor", "b_avail"]]
    .stack(sample=("chain", "draw"))
    .isel(sample=slice(None, 250))
)
a_grid_da = xr.DataArray(availability_grid, dims=["a_grid"])
factor_draws = xr.apply_ufunc(
    availability_factor, a_grid_da, posterior_factor["b_avail"], posterior_factor["floor"]
)
factor_panel_curves = factor_draws.mean("series")

availability_bin_da = (
    (train_availability * 10).astype(np.int64).clip(0, 9).rename("availability_bin")
)
panel_bin_availability = train_availability.groupby(availability_bin_da).mean()
panel_bin_sales = train_scaled_sales.groupby(availability_bin_da).mean()
top_bin_sales = float(panel_bin_sales.isel(availability_bin=-1))

pc = az.plot_lm(
    predictions_to_datatree(
        factor_panel_curves.transpose("sample", "a_grid").to_numpy()[:, :, None],
        availability_grid,
        ["posterior factor"],
    ),
    y="obs",
    x="t",
    plot_dim="time",
    ci_kind="hdi",
    ci_prob=hdi_probs,
    smooth=False,
    visuals={
        "ci_band": {"color": "C0"},
        "observed_scatter": False,
        "pe_line": False,
        "xlabel": False,
        "ylabel": False,
    },
    aes={"alpha": ["prob"]},
    alpha=hdi_alphas,
    figure_kwargs={"figsize": (10, 6)},
)
ax = pc.get_target("t", {"series": "posterior factor"})
factor_series_means = factor_draws.mean("sample")
for i, label in enumerate(focus_labels):
    ax.plot(
        availability_grid,
        factor_series_means.sel(series=label),
        color="gray",
        alpha=0.5,
        linewidth=1,
        label="per-series posterior mean (focus series)" if i == 0 else None,
    )
ax.plot(
    panel_bin_availability,
    panel_bin_sales / top_bin_sales,
    "o",
    color="black",
    label="panel binned mean (top bin = 1)",
)
ax.plot(
    0.0,
    panel_floor / top_bin_sales,
    "D",
    color="C3",
    markersize=8,
    label="panel empirical floor (top bin = 1)",
)
bands = pc.viz["ci_band"]["t"].sel(series="posterior factor")
for prob in (0.94, 0.5):
    bands.sel(prob=prob).item().set_label(hdi_label(prob))
ax.legend(loc="lower right")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
ax.set(
    xlabel="sales-weighted availability",
    ylabel="availability factor",
    title="Posterior availability factor (panel mean) vs empirical curve",
);
```


    /Users/juanitorduz/Documents/numpyro_forecast/.venv/lib/python3.14/site-packages/arviz_plots/plots/lm_plot.py:360: UserWarning: When multiple credible intervals are plotted, it is recommended to map 'alpha' aesthetic to 'prob' dimension to differentiate between intervals.
      warnings.warn(


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-43-output-2.png" class="figure-img" width="1011" height="611" /></p>
</figure>


The posterior factor reproduces the saturating shape and the positive floor. The \\50\\\\ and \\94\\\\ HDI bands are so thin they read as a single line, and that is not a plotting artifact but a consequence of what is being plotted: the bands quantify the posterior uncertainty of the *panel-mean* curve, the average of a thousand per-series factor curves. The genuine heterogeneity across series (visible in the gray per-series posterior means, whose floors and curvatures differ substantially) is averaged away by construction, and what remains is the uncertainty about the average itself, which shrinks roughly like \\1/\sqrt{n\_{\text{series}}}\\ on top of per-series parameters that \\76\\ days of data already pin down well. A per-series version of this plot would show much wider bands; the panel mean is deliberately the sharpest view. The curve sits below the rescaled empirical points over most of the range, and that gap is the endogeneity correction at work: high-demand days both sell more and sell out more often, so part of the raw curve's height belongs to the trend, seasonality, and promotions, and the model attributes it there instead of to availability itself.


# Inspecting the store hierarchy

The covariate effects are pooled by store. Plotting each series' discount effect against its store-level location shows the partial pooling: series means line up along the identity line, shrunk toward their store's location, more strongly where the store scale \\\sigma^{\text{store}}\\ is small. Read the tightness with the store-size caveat from the panel build in mind: with a median of one series per store, many points sit near the line simply because the store location is informed by that single series, and the genuine cross-series pooling acts in the multi-series stores, where the vertical spread around the line is the shrinkage at work. The plot also shows why the cleaned discount encoding and the launch indicator matter: without them, a cluster of series escapes to coefficients an order of magnitude above their store locations (the spurious launch-step optimum described in the evaluation section); with them, the scatter hugs the identity line.


    In [43]:


``` python
b_series_mean = cast(
    "xr.DataArray",
    tree["posterior"]["b"].sel(covariate="discount_magnitude").mean(dim=("chain", "draw")),
)
b_store_loc_mean = cast(
    "xr.DataArray",
    tree["posterior"]["b_loc_store"]
    .sel(covariate="discount_magnitude")
    .mean(dim=("chain", "draw")),
)
# A vectorized label-based gather: for each series, look up its store's location
# on the posterior's store dimension via the series -> store id lookup table.
b_store_loc_per_series = b_store_loc_mean.sel(store=series_store_da)

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(
    b_store_loc_per_series.to_numpy(),
    b_series_mean.to_numpy(),
    color="C0",
    alpha=0.6,
    s=14,
    label="series",
)
ax.axline((0.0, 0.0), slope=1.0, color="black", linestyle="--", linewidth=1, label="identity")
ax.legend(loc="upper left")
ax.set(
    xlabel="store-level posterior mean",
    ylabel="series-level posterior mean",
    title="Discount effects shrink toward their store-level locations",
);
```


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-44-output-1.png" class="figure-img" width="811" height="711" /></p>
</figure>


# Promotion contributions

The reason to keep every promotion feature in the model is to read off what each one contributes to sales. On the scaled axis a contribution of \\0.1\\ means "one tenth of an average day's sales", so the units are directly comparable across series. For each series we take the posterior-mean coefficient times the feature's average value on its *active* days (days where the feature is positive) over the training window, and summarize that quantity across the panel with a forest plot: per feature, the open circle marks the cross-series median, the thick segment the \\50\\\\ HDI, and the thin line the \\94\\\\ HDI of the contribution across series. The plot stays entirely in named-tensor land: `az.plot_forest` consumes the `(covariate, series)` contributions array directly, treating the series axis as the sample dimension, and the `skipna` entries in its `stats` mapping drop the series where a feature has no active training day (whose active-day mean is NaN) instead of blanking that feature's row.


    In [44]:


``` python
b_post_mean = cast("xr.DataArray", tree["posterior"]["b"].mean(dim=("chain", "draw")))
features_train = cast(
    "xr.DataArray",
    tree["constant_data"]["covariates"]
    .sel(input=covariate_names)
    .isel(time=slice(None, t_train))
    .rename(input="covariate"),
)
active_mean_x = features_train.where(features_train > 0).mean("time")
contributions = b_post_mean * active_mean_x

pc = az.plot_forest(
    contributions.to_dataset(name="contribution"),  # ty: ignore[invalid-argument-type]
    sample_dims=["series"],
    ci_kind="hdi",
    ci_probs=hdi_probs,
    point_estimate="median",
    labels=["covariate"],
    stats={
        "trunk": {"skipna": True},
        "twig": {"skipna": True},
        "point_estimate": {"skipna": True},
    },
    figure_kwargs={"figsize": (10, 6)},
)
ax = pc.viz["plot"].sel(column="forest").item()
ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
ax.legend(
    handles=[
        mlines.Line2D([], [], color="C0", linewidth=3, label=hdi_label(0.5)),
        mlines.Line2D([], [], color="C0", linewidth=1, label=hdi_label(0.94)),
        mlines.Line2D(
            [], [], color="C0", marker="o", markerfacecolor="white", linewidth=0, label="median"
        ),
    ],
    loc="lower right",
)
ax.set(xlabel="contribution on active days (scaled units)")
fig = pc.viz["figure"].item()
fig.suptitle(
    "Posterior contributions as a fraction of an average day's sales",
    fontsize=14,
    fontweight="bold",
);
```


<figure class="figure">
<p><img src="fresh_retail_stockout_files/figure-html/cell-45-output-1.png" class="figure-img" width="1011" height="611" /></p>
</figure>


The holiday effect is the one large, consistent promotion signal: about a quarter of an average day's sales on holiday days, in line with the EDA lift. The cleaned discount effect is small and mostly non-negative (its active days are the genuinely priced promotions, which for this panel are sparse), and the activity effect is wide and centered near zero, informative only for the minority of series that actually run campaigns. The launch indicator deserves a careful read: its posterior contribution ends up small and slightly negative, because in-sample the random-walk level absorbs most of the launch step (under these priors a few large drift innovations are the cheaper explanation), leaving the indicator as a modest correction for series whose ramp missed the shared date. Its value is preventive rather than predictive: it takes the launch-shaped signal off the table for the *promotion* features, which is exactly the spurious optimum described in the evaluation section. Store-level pooling keeps the weakly informed series (the flagship product whose cleaned discount feature is almost always zero, for example) tied to their store's typical effect instead of letting them drift on noise.


# Next steps

- Calibrate the forecast intervals post hoc: estimate quantile-specific scaling on a held-out calibration window before the test period (a conformal-style correction), lifting the late-horizon upper tail flagged by the interval diagnostics and reining in the early-day over-coverage.
- Model the launch mechanism explicitly (an assortment-event effect shared across series within a store) instead of a fixed panel-wide indicator.
- Replace the mean-level factor with a censored likelihood: treat sales as latent demand right-censored by the available stock, which uses the same availability feature but models the mechanism instead of its average effect.
- Move to a strictly positive observation model (for example a negative binomial on rounded units), so the zero-sales days need no clipping and the heavier tail addresses the in-sample central-band over-coverage measured above.
- Add the weather covariates (precipitation, temperature) that this notebook left out.
- Evaluate with rolling-origin backtesting via the package's [backtest](../../../reference/evaluate.backtest.md#numpyro_forecast.evaluate.backtest) helper instead of a single split, passing the `ScaledForecaster` defined above as `forecaster_fn` so the per-series scaling is recomputed inside every fold.


# References

- Yang, T., et al. (2025). [*FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail*](https://arxiv.org/abs/2505.16319). Dataset on [Hugging Face](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K).
- Orduz, J. [*Hierarchical forecasting with NumPyro (part I)*](https://juanitorduz.github.io/numpyro_hierarchical_forecasting_1/).
- Pyro. [*Forecasting III: Hierarchical Models*](https://pyro.ai/examples/forecasting_iii.html).
- Related examples: [hierarchical forecasting I](hierarchical_forecasting_1.md), [inference methods comparison](inference_methods_comparison.md).
