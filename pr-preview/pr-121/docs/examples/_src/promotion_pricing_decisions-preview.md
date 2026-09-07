# From forecasts to promotion decisions


This notebook takes a probabilistic demand model all the way to a set of promotion decisions. The data are the dunnhumby [*Breakfast at the Frat*](https://www.dunnhumby.com/source-files/) scanner panel: weekly unit sales, shelf and base prices, and the promotion mechanics (in-store circular feature, in-store display, and shelf-tag-only price cuts) for products in four categories across 77 stores over 156 weeks, read here from the [figshare copy](https://doi.org/10.6084/m9.figshare.30121060) of the workbook. We work with six cereals of one substitution group and ask the question a category manager and a replenishment planner face every quarter: what should a promotion of Honey Nut Cheerios look like, who should fund it, and how much stock should each store order for it?

**Under the retailer's margin and a nominal funding share from the manufacturer, expected profit falls with discount depth for every promotion mechanics, so a point forecast and the posterior pick the same corner of the price grid. The posterior earns its keep elsewhere: the right expectation through partial pooling across stores, a break-even funding share with a credible interval, a downside and a tie-breaker where the objective is flat, and an order quantity from joint demand paths rather than from summed marginal quantiles.**

The algebra behind the first sentence is short. With gross margin g on the base price and a manufacturer that funds a share \alpha of the discount on every unit sold in a promotion week, a deeper cut raises expected profit only if the promotional elasticity under the chosen mechanics satisfies \varepsilon_m \< -(1 - \alpha) / g. We estimate \varepsilon_m with its posterior and read the decision off that inequality, with the cannibalization of the sibling products included. The funding and go/no-go questions belong to the category manager; the order quantity belongs to the replenishment planner once the promotion is committed. We proceed in five steps. First, we read and inspect the panel, count the weeks that identify a price effect separately from the promotion mechanics, and write down the causal assumption. Second, we specify a hierarchical negative binomial demand model with own and cross price elasticities as reusable components and fit it with NUTS. Third, we check the fit in sample and on a holdout quarter. Fourth, we forecast counterfactual promotions by editing the horizon covariates and reusing the posterior. Fifth, we turn the forecasts into the funding, mechanics, risk and order decisions.

The model uses the package's model building blocks:

- [`Horizon.from_data`](https://juanitorduz.github.io/numpyro_forecast/reference/models.Horizon.html) derives the train and forecast windows from the shapes of the covariates and the data.
- [`innovations`](https://juanitorduz.github.io/numpyro_forecast/reference/models.innovations.html) samples the weekly level innovations for every store-product series, with a separate site for the forecast horizon.
- [`predict`](https://juanitorduz.github.io/numpyro_forecast/reference/models.predict.html) attaches the negative binomial likelihood to the observed weeks and samples the horizon.

The prediction drivers and evaluation helpers do the rest: [`forecast`](https://juanitorduz.github.io/numpyro_forecast/reference/predictive.forecast.html) and [`predict_in_sample`](https://juanitorduz.github.io/numpyro_forecast/reference/predictive.predict_in_sample.html) draw the holdout and in-sample predictives, [`to_datatree`](https://juanitorduz.github.io/numpyro_forecast/reference/convert.to_datatree.html) and [`predictions_to_datatree`](https://juanitorduz.github.io/numpyro_forecast/reference/convert.predictions_to_datatree.html) export draws to ArviZ, [`fourier_features`](https://juanitorduz.github.io/numpyro_forecast/reference/features.fourier_features.html) builds the annual seasonality basis, and [`eval_crps`](https://juanitorduz.github.io/numpyro_forecast/reference/evaluate.eval_crps.html), [`eval_coverage`](https://juanitorduz.github.io/numpyro_forecast/reference/evaluate.eval_coverage.html) and [`make_mase`](https://juanitorduz.github.io/numpyro_forecast/reference/metrics.make_mase.html) score the forecasts.

Two things this notebook does not claim. Prices were never randomized, so every elasticity rests on a selection-on-observables assumption that we state explicitly and cannot test. And the holdout validates the forecasting engine under the realized promotion calendar, not the counterfactual calendars, which nobody observed.


# Prepare notebook


    In [1]:


``` python
import hashlib
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

# preliz warns at import time that PyMC is absent; the example does not need it.
warnings.filterwarnings("ignore", message="PyMC not installed", category=UserWarning)

import arviz as az
import graphviz as gr
import jax
import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
import preliz as pz
import xarray as xr
from jax import random
from jaxtyping import Float, Int
from matplotlib import ticker as mtick
from matplotlib.axes import Axes
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast import (
    Horizon,
    eval_coverage,
    eval_crps,
    eval_mae,
    forecast,
    innovations,
    predict,
    predict_in_sample,
    predictions_to_datatree,
    to_datatree,
)
from numpyro_forecast.features import fourier_features
from numpyro_forecast.metrics import crps_empirical, make_mase
from numpyro_forecast.typing import Array, ForecastModel

az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"
warnings.filterwarnings(
    "ignore", message="When multiple credible intervals are plotted", category=UserWarning
)

# Render polars tables without truncating string cells, and drop the shape and
# dtype headers, which are noise in a rendered document.
pl.Config.set_fmt_str_lengths(100)
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True)
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)

numpyro.set_host_device_count(n=4)

rng_key = random.PRNGKey(seed=42)

%load_ext autoreload
%autoreload 2
%load_ext jaxtyping
%jaxtyping.typechecker beartype.beartype
%config InlineBackend.figure_format = "retina"
```


# Read data

dunnhumby publishes the workbook on its source-files page behind a form. We download the copy that [Ghaedrahmati (2025)](https://doi.org/10.6084/m9.figshare.30121060) deposited on figshare, whose record declares a CC BY 4.0 license for the copy; dunnhumby's own terms govern the data. The download is cached under the home directory and verified against its SHA-256 digest, so the notebook reads the same bytes on every run. Every data sheet carries a title row above the header, which is why the reader skips one row.


    In [2]:


``` python
DATA_SHA256 = "61b1d77dd6d9298fed204cc231f2b853a4c7f79376cfc30231646e1e51d0daba"
cache_dir = Path.home() / ".cache" / "numpyro_forecast"
cache_dir.mkdir(parents=True, exist_ok=True)
workbook_path = cache_dir / "breakfast_at_the_frat.xlsx"
if not workbook_path.exists():
    urllib.request.urlretrieve("https://ndownloader.figshare.com/files/57937129", workbook_path)
digest = hashlib.sha256(workbook_path.read_bytes()).hexdigest()
if digest != DATA_SHA256:
    msg = f"unexpected workbook digest {digest}"
    raise ValueError(msg)

sheets = pl.read_excel(
    workbook_path,
    sheet_name=["dh Transaction Data", "dh Products Lookup", "dh Store Lookup"],
    engine="calamine",
    read_options={"header_row": 1},
)
transactions_raw = sheets["dh Transaction Data"]
products_df = sheets["dh Products Lookup"]
stores_raw = sheets["dh Store Lookup"]

print(
    f"transactions: {transactions_raw.height:,} rows | "
    f"{transactions_raw['STORE_NUM'].n_unique()} stores | "
    f"{transactions_raw['UPC'].n_unique()} UPCs | "
    f"{transactions_raw['WEEK_END_DATE'].n_unique()} weeks from "
    f"{transactions_raw['WEEK_END_DATE'].min()} to {transactions_raw['WEEK_END_DATE'].max()}"
)
print(f"products lookup: {products_df.height} rows | store lookup: {stores_raw.height} rows")
transactions_raw.head(3)
```


    transactions: 524,950 rows | 77 stores | 55 UPCs | 156 weeks from 2009-01-14 to 2012-01-04
    products lookup: 58 rows | store lookup: 79 rows


| WEEK_END_DATE | STORE_NUM | UPC | UNITS | VISITS | HHS | SPEND | PRICE | BASE_PRICE | FEATURE | DISPLAY | TPR_ONLY |
|----|----|----|----|----|----|----|----|----|----|----|----|
| 2009-01-14 | 367 | 1111009477 | 13 | 13 | 13 | 18.07 | 1.39 | 1.57 | 0 | 0 | 1 |
| 2009-01-14 | 367 | 1111009497 | 20 | 18 | 18 | 27.8 | 1.39 | 1.39 | 0 | 0 | 0 |
| 2009-01-14 | 367 | 1111009507 | 14 | 14 | 14 | 19.32 | 1.38 | 1.38 | 0 | 0 | 0 |


Three quirks of the workbook matter for what follows. A few rows have a missing price, a handful have non-positive units, and about one row in a hundred has a shelf price above the base price, which is a lagged base-price update rather than a mark-up. We treat a ratio above one as no discount and keep the asymmetry in mind: a ratio below one always reads as a temporary cut, so a permanent price drop recorded before its base-price update would look like a promotion. The store lookup also lists two store ids twice with different price-segment labels, which would duplicate every row of those stores on a join, so we keep the first row per store. The printed counts: 524{,}950 transaction rows over 77 stores, 55 products and 156 weeks; 185 rows without a base price and 23 without a shelf price; five rows with non-positive units; 1.2\\ of the rows with a shelf price above the base price; two duplicated store ids, and 77 stores after the dedupe, 43 mainstream, 15 upscale and 19 value.


    In [3]:


``` python
price_ratio_raw = transactions_raw["PRICE"] / transactions_raw["BASE_PRICE"]
quirks = pl.DataFrame(
    {
        "quirk": [
            "missing BASE_PRICE",
            "missing PRICE",
            "UNITS <= 0",
            "PRICE > BASE_PRICE",
        ],
        "rows": [
            transactions_raw["BASE_PRICE"].null_count(),
            transactions_raw["PRICE"].null_count(),
            int((transactions_raw["UNITS"] <= 0).sum()),
            int((price_ratio_raw > 1.0).sum()),
        ],
    }
).with_columns(share=(pl.col("rows") / transactions_raw.height))
quirks
```


| quirk                 | rows | share    |
|-----------------------|------|----------|
| "missing BASE_PRICE"  | 185  | 0.000352 |
| "missing PRICE"       | 23   | 0.000044 |
| "UNITS \<= 0"         | 5    | 0.00001  |
| "PRICE \> BASE_PRICE" | 6047 | 0.011519 |


    In [4]:


``` python
duplicated_stores = stores_raw.filter(pl.col("STORE_ID").is_duplicated())
stores_df = stores_raw.unique(subset="STORE_ID", keep="first", maintain_order=True)
print(f"duplicated store ids: {duplicated_stores['STORE_ID'].n_unique()} (rows kept: first)")
print(duplicated_stores.select("STORE_ID", "STORE_NAME", "SEG_VALUE_NAME"))
segment_counts = stores_df.group_by("SEG_VALUE_NAME").len().sort("SEG_VALUE_NAME")
print(f"stores after the dedupe: {stores_df.height}")
segment_counts
```


    duplicated store ids: 2 (rows kept: first)
    ┌──────────┬──────────────┬────────────────┐
    │ STORE_ID ┆ STORE_NAME   ┆ SEG_VALUE_NAME │
    ╞══════════╪══════════════╪════════════════╡
    │ 4503     ┆ ROCKWALL     ┆ MAINSTREAM     │
    │ 17627    ┆ FLOWER MOUND ┆ MAINSTREAM     │
    │ 17627    ┆ FLOWER MOUND ┆ UPSCALE        │
    │ 4503     ┆ ROCKWALL     ┆ UPSCALE        │
    └──────────┴──────────────┴────────────────┘
    stores after the dedupe: 77


| SEG_VALUE_NAME | len |
|----------------|-----|
| "MAINSTREAM"   | 43  |
| "UPSCALE"      | 15  |
| "VALUE"        | 19  |


# A first look at prices and promotions


## Why these six products

Missing store-product-weeks are absent rows, never recorded zeros, so a product that a store stops carrying simply disappears for a while. The table below counts, for every cold cereal, the stores that carry it and the stores that carry it in all weeks. The Post and Quaker products have delisting gaps in every store, so we keep the six products of the `ALL FAMILY CEREAL` sub-category from General Mills, Kellogg and the private label: one substitution group, complete in dozens of stores, with a private-label twin of the focal product. The six are carried in all 77 stores and complete in 62 to 75 of them.


    In [5]:


``` python
N_WEEKS = transactions_raw["WEEK_END_DATE"].n_unique()
cereal_all = transactions_raw.join(products_df, on="UPC").filter(
    pl.col("CATEGORY") == "COLD CEREAL"
)
weeks_per_series = cereal_all.group_by("UPC", "STORE_NUM").agg(
    pl.col("WEEK_END_DATE").n_unique().alias("weeks")
)
completeness = (
    weeks_per_series.group_by("UPC")
    .agg(
        stores_carrying=pl.len(),
        stores_complete=(pl.col("weeks") == N_WEEKS).sum(),
        min_weeks=pl.col("weeks").min(),
    )
    .join(products_df.select("UPC", "DESCRIPTION", "MANUFACTURER", "SUB_CATEGORY"), on="UPC")
    .sort("stores_complete", "UPC", descending=[True, False])
    .select(
        "UPC",
        "DESCRIPTION",
        "MANUFACTURER",
        "SUB_CATEGORY",
        "stores_carrying",
        "stores_complete",
        "min_weeks",
    )
)
completeness
```


| UPC | DESCRIPTION | MANUFACTURER | SUB_CATEGORY | stores_carrying | stores_complete | min_weeks |
|----|----|----|----|----|----|----|
| 1600027527 | "GM HONEY NUT CHEERIOS" | "GENERAL MI" | "ALL FAMILY CEREAL" | 77 | 75 | 131 |
| 1600027528 | "GM CHEERIOS" | "GENERAL MI" | "ALL FAMILY CEREAL" | 77 | 73 | 131 |
| 3800031838 | "KELL FROSTED FLAKES" | "KELLOGG" | "KIDS CEREAL" | 77 | 73 | 131 |
| 1111085345 | "PL RAISIN BRAN" | "PRIVATE LABEL" | "ADULT CEREAL" | 77 | 71 | 131 |
| 1600027564 | "GM CHEERIOS" | "GENERAL MI" | "ALL FAMILY CEREAL" | 77 | 71 | 131 |
| 3800031829 | "KELL BITE SIZE MINI WHEAT" | "KELLOGG" | "ALL FAMILY CEREAL" | 77 | 69 | 83 |
| 1111085350 | "PL BT SZ FRSTD SHRD WHT" | "PRIVATE LABEL" | "ALL FAMILY CEREAL" | 77 | 66 | 131 |
| 1111085319 | "PL HONEY NUT TOASTD OATS" | "PRIVATE LABEL" | "ALL FAMILY CEREAL" | 77 | 62 | 131 |
| 3800039118 | "KELL FROOT LOOPS" | "KELLOGG" | "KIDS CEREAL" | 77 | 58 | 120 |
| 3000006340 | "QKER LIFE ORIGINAL" | "QUAKER" | "ALL FAMILY CEREAL" | 77 | 0 | 41 |
| 3000006560 | "QKER CAP N CRUNCH BERRIES" | "QUAKER" | "KIDS CEREAL" | 77 | 0 | 102 |
| 3000006610 | "QKER CAP N CRUNCH" | "QUAKER" | "KIDS CEREAL" | 77 | 0 | 107 |
| 88491201426 | "POST HNY BN OTS HNY RSTD" | "POST FOODS" | "ADULT CEREAL" | 77 | 0 | 94 |
| 88491201427 | "POST FM SZ HNYBNCH OT ALM" | "POST FOODS" | "ADULT CEREAL" | 77 | 0 | 63 |
| 88491212971 | "POST FRUITY PEBBLES" | "POST FOODS" | "KIDS CEREAL" | 77 | 0 | 126 |


    In [6]:


``` python
PRODUCT_LABELS = {
    1600027527: "HNC",
    1600027564: "Cheerios 12oz",
    1600027528: "Cheerios 18oz",
    3800031829: "Mini Wheats",
    1111085319: "PL Honey Nut Oats",
    1111085350: "PL Frosted Wheat",
}
product_order: list[str] = list(PRODUCT_LABELS.values())
n_products = len(product_order)
FOCAL = "HNC"
FOCAL_INDEX = product_order.index(FOCAL)

cereal_df = (
    transactions_raw.filter(pl.col("UPC").is_in(list(PRODUCT_LABELS)))
    .join(products_df, on="UPC")
    .join(stores_df, left_on="STORE_NUM", right_on="STORE_ID")
    .drop_nulls(["PRICE", "BASE_PRICE"])
    .filter(pl.col("UNITS") > 0)
    .with_columns(
        product=pl.col("UPC").replace_strict(PRODUCT_LABELS, return_dtype=pl.String),
        discount=1.0 - pl.col("PRICE") / pl.col("BASE_PRICE"),
        x=(pl.col("PRICE") / pl.col("BASE_PRICE")).clip(upper_bound=1.0).log(),
        feature_display=pl.col("FEATURE") * pl.col("DISPLAY"),
        log_units=pl.col("UNITS").cast(pl.Float64).log(),
    )
    .with_columns(
        lam=-pl.col("x"),
        series=pl.concat_str([pl.col("STORE_NUM"), pl.col("product")], separator="::"),
        cut=(pl.col("discount") > 0.02).cast(pl.Int64),
        mechanics=pl.when((pl.col("FEATURE") == 1) & (pl.col("DISPLAY") == 1))
        .then(pl.lit("feature + display"))
        .when(pl.col("FEATURE") == 1)
        .then(pl.lit("feature"))
        .when(pl.col("DISPLAY") == 1)
        .then(pl.lit("display"))
        .when(pl.col("TPR_ONLY") == 1)
        .then(pl.lit("TPR-only"))
        .otherwise(pl.lit("none")),
    )
    .sort("STORE_NUM", "product", "WEEK_END_DATE")
)
n_before = transactions_raw.filter(pl.col("UPC").is_in(list(PRODUCT_LABELS))).height
tpr_definition = (cereal_df["TPR_ONLY"] == 1) == (
    (cereal_df["cut"] == 1) & (cereal_df["FEATURE"] == 0) & (cereal_df["DISPLAY"] == 0)
)
print(f"six-product rows: {cereal_df.height:,} of {n_before:,} before dropping missing prices")
print(f"TPR_ONLY == 1 exactly when discount > 2% and no mechanics: {bool(tpr_definition.all())}")
print(
    f"rows with PRICE > BASE_PRICE (read as no discount): {(cereal_df['discount'] < 0).to_numpy().mean():.1%}"
)
```


    six-product rows: 71,774 of 71,774 before dropping missing prices
    TPR_ONLY == 1 exactly when discount > 2% and no mechanics: True
    rows with PRICE > BASE_PRICE (read as no discount): 0.5%


## Promotions work through mechanics far more than through price

The flags mean the following. `FEATURE` marks a week in which the product appeared in the store circular, `DISPLAY` a week with an in-store display, and `TPR_ONLY` a temporary price reduction with a shelf tag and nothing else. Almost every feature week also carries a price cut; the mechanics table shows that the cut alone moves units far less than a feature or a display does, the empirical regularity that [Blattberg, Briesch and Fox (1995)](https://doi.org/10.1287/mksc.14.3.G122) list among the generalizations about how promotions work. It also shows the identification problem: because cuts and mechanics arrive together, a regression of units on price alone credits the mechanics' uplift to the price. The printed table over all 77 stores: store-weeks without mechanics average 31 units, shelf-tag cuts 40, displays 73, features 74, and feature with display 135, at mean depths of 18\\ to 26\\; 97\\ of the feature-with-display weeks carry a cut, the shelf-tag flag is exactly a cut of more than 2\\ without mechanics in every row, and 0.5\\ of the six-product rows have a shelf price above the base price.


    In [7]:


``` python
mechanics_order = ["none", "TPR-only", "display", "feature", "feature + display"]
mechanics_table = (
    cereal_df.group_by("mechanics")
    .agg(
        store_weeks=pl.len(),
        share_with_cut=pl.col("cut").mean(),
        mean_depth=pl.col("discount").clip(lower_bound=0.0).mean(),
        mean_units=pl.col("UNITS").mean(),
    )
    .with_columns(
        order=pl.col("mechanics").replace_strict({m: i for i, m in enumerate(mechanics_order)})
    )
    .sort("order")
    .drop("order")
)
mechanics_table
```


| mechanics           | store_weeks | share_with_cut | mean_depth | mean_units |
|---------------------|-------------|----------------|------------|------------|
| "none"              | 53777       | 0.0            | 0.0        | 30.776689  |
| "TPR-only"          | 10584       | 1.0            | 0.191906   | 39.888983  |
| "display"           | 1542        | 0.80415        | 0.188167   | 73.227626  |
| "feature"           | 2750        | 0.858909       | 0.176457   | 74.496727  |
| "feature + display" | 3121        | 0.9686         | 0.260721   | 134.880167 |


    In [8]:


``` python
def within_ols(
    frame: pl.DataFrame, columns: list[str], target: str = "log_units", by: str = "series"
) -> pl.DataFrame:
    """Within-group least squares (every column demeaned by ``by``) with classical standard errors.

    Parameters
    ----------
    frame
        Long table with one row per store-product-week.
    columns
        Regressor columns.
    target
        Response column.
    by
        Grouping column whose fixed effects are removed by demeaning.

    Returns
    -------
    pl.DataFrame
        One row per regressor with its coefficient and standard error.
    """
    demeaned = frame.select(
        [(pl.col(c) - pl.col(c).mean().over(by)).alias(c) for c in [target, *columns]]
    )
    design = demeaned.select(columns).to_numpy().astype(np.float64)
    response = demeaned[target].to_numpy().astype(np.float64)
    beta, *_ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ beta
    dof = response.size - design.shape[1] - frame[by].n_unique()
    covariance = (residual @ residual / dof) * np.linalg.pinv(design.T @ design)
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return pl.DataFrame({"term": columns, "coef": beta, "se": standard_error})


first_week = cereal_df["WEEK_END_DATE"].min()
cereal_df = cereal_df.with_columns(
    week_index=((pl.col("WEEK_END_DATE") - pl.lit(first_week)).dt.total_days() / 7.0)
).with_columns(
    sin1=(2.0 * np.pi * pl.col("week_index") / 52.18).sin(),
    cos1=(2.0 * np.pi * pl.col("week_index") / 52.18).cos(),
    sin2=(4.0 * np.pi * pl.col("week_index") / 52.18).sin(),
    cos2=(4.0 * np.pi * pl.col("week_index") / 52.18).cos(),
    trend=pl.col("week_index") / N_WEEKS,
    promo=((pl.col("FEATURE") == 1) | (pl.col("DISPLAY") == 1) | (pl.col("cut") == 1)).cast(
        pl.Float64
    ),
)
seasonal_terms = ["sin1", "cos1", "sin2", "cos2", "trend"]
```


## Do promotions borrow from the following weeks?

A promotion that loads the pantry depresses the weeks after it, the post-promotion dip of [van Heerde, Leeflang and Wittink (2000)](https://doi.org/10.1509/jmkr.37.3.383.18782). If the dip were large, the timing of promotion weeks would matter and the decision space would include the calendar. We check it with a within-series regression of log units on the price ratio, the mechanics, and indicators for the one and two weeks after any promotion, controlling for annual seasonality and a trend. The dip is +0.8\\ (standard error 0.5\\) in the first week after a promotion and -1.6\\ (standard error 0.4\\) in the second, against feature and display effects of +0.51 and +0.46 on the log scale: statistically visible, economically negligible. The calendar is therefore not a lever in this notebook.


    In [9]:


``` python
dip_df = cereal_df.with_columns(
    post1=pl.col("promo").shift(1).over("series").fill_null(0.0),
    post2=pl.col("promo").shift(2).over("series").fill_null(0.0),
)
dip_ols = within_ols(
    dip_df, ["x", "FEATURE", "DISPLAY", "feature_display", "post1", "post2", *seasonal_terms]
)
dip_ols.filter(
    pl.col("term").is_in(["x", "FEATURE", "DISPLAY", "feature_display", "post1", "post2"])
)
```


| term              | coef      | se       |
|-------------------|-----------|----------|
| "x"               | -0.891461 | 0.015639 |
| "FEATURE"         | 0.507507  | 0.008564 |
| "DISPLAY"         | 0.457711  | 0.011262 |
| "feature_display" | 0.075924  | 0.015563 |
| "post1"           | 0.008233  | 0.004645 |
| "post2"           | -0.016333 | 0.00434  |


## The realized calendar of the holdout quarter

The last 13 weeks of the panel (October 2011 to the first week of January 2012) are the holdout and, later, the horizon on which we evaluate counterfactual promotions. The table shows, for the focal product, the share of stores with a feature, a display or a shelf-tag-only cut in every horizon week, and the mean discount depth. The realized calendar ran a feature with display at about a 15\\ cut in the two Thanksgiving weeks and a deeper cut around Christmas. In horizon weeks 7 and 8 every store featured the product, 84\\ and 69\\ of the stores displayed it, and the mean depth was 14\\; weeks 11 to 13 carried a 31\\ cut with a feature in every store.


    In [10]:


``` python
all_weeks = cereal_df["WEEK_END_DATE"].unique().sort()
HORIZON = 13
t_train = N_WEEKS - HORIZON
holdout_weeks = all_weeks[t_train:]
realized_calendar = (
    cereal_df.filter(
        (pl.col("product") == FOCAL) & pl.col("WEEK_END_DATE").is_in(holdout_weeks.to_list())
    )
    .group_by("WEEK_END_DATE")
    .agg(
        feature=pl.col("FEATURE").mean(),
        display=pl.col("DISPLAY").mean(),
        tpr_only=pl.col("TPR_ONLY").mean(),
        depth=pl.col("discount").clip(lower_bound=0.0).mean(),
    )
    .sort("WEEK_END_DATE")
    .with_row_index("horizon_week", offset=1)
)
realized_calendar
```


| horizon_week | WEEK_END_DATE | feature  | display  | tpr_only | depth    |
|--------------|---------------|----------|----------|----------|----------|
| 1            | 2011-10-12    | 0.0      | 0.0      | 0.0      | 0.0      |
| 2            | 2011-10-19    | 0.0      | 0.0      | 0.142857 | 0.025053 |
| 3            | 2011-10-26    | 0.0      | 0.012987 | 0.0      | 0.0      |
| 4            | 2011-11-02    | 0.0      | 0.0      | 0.0      | 0.0      |
| 5            | 2011-11-09    | 0.0      | 0.0      | 0.077922 | 0.009592 |
| 6            | 2011-11-16    | 0.0      | 0.012987 | 0.298701 | 0.043214 |
| 7            | 2011-11-23    | 1.0      | 0.844156 | 0.0      | 0.142322 |
| 8            | 2011-11-30    | 1.0      | 0.688312 | 0.0      | 0.137347 |
| 9            | 2011-12-07    | 0.0      | 0.0      | 0.025974 | 0.003303 |
| 10           | 2011-12-14    | 0.0      | 0.0      | 0.038961 | 0.0074   |
| 11           | 2011-12-21    | 1.0      | 0.883117 | 0.0      | 0.308346 |
| 12           | 2011-12-28    | 1.0      | 0.844156 | 0.0      | 0.307668 |
| 13           | 2012-01-04    | 0.467532 | 0.727273 | 0.090909 | 0.308985 |


# Build the modeling panel

The panel keeps the stores in which all six series are complete over the 156 weeks and takes six stores per price segment (the segments are the retailer's `MAINSTREAM`, `UPSCALE` and `VALUE` labels), the largest by average weekly baskets. Six products in 18 stores give 108 series. The knob `n_stores_per_segment` scales the panel; the whole complete set costs about three times the fit. Every series is identified as `store::product`. The store index never enters the model: it drives the cross-price block, the sibling flags, the policy builder, the per-store base prices and the per-store decisions. All six series are complete in 52 stores: 32 mainstream, 11 upscale and 9 value.


    In [11]:


``` python
weeks_by_series = cereal_df.group_by("STORE_NUM", "product").agg(
    pl.col("WEEK_END_DATE").n_unique().alias("weeks")
)
complete_stores = (
    weeks_by_series.group_by("STORE_NUM")
    .agg(all_complete=((pl.col("weeks") == N_WEEKS).all()) & (pl.len() == n_products))
    .filter(pl.col("all_complete"))
    .join(
        stores_df.select("STORE_ID", "SEG_VALUE_NAME", "AVG_WEEKLY_BASKETS"),
        left_on="STORE_NUM",
        right_on="STORE_ID",
    )
    .sort(["SEG_VALUE_NAME", "AVG_WEEKLY_BASKETS", "STORE_NUM"], descending=[False, True, False])
)
print(f"stores with all six series complete: {complete_stores.height}")
print(complete_stores.group_by("SEG_VALUE_NAME").len().sort("SEG_VALUE_NAME"))

n_stores_per_segment = 6
selected = complete_stores.group_by("SEG_VALUE_NAME", maintain_order=True).head(
    n_stores_per_segment
)
selected_stores: list[int] = selected["STORE_NUM"].to_list()
n_stores = len(selected_stores)
store_segment: dict[int, str] = dict(
    zip(selected["STORE_NUM"].to_list(), selected["SEG_VALUE_NAME"].to_list(), strict=True)
)
selected.select("STORE_NUM", "SEG_VALUE_NAME", "AVG_WEEKLY_BASKETS")
```


    stores with all six series complete: 52
    ┌────────────────┬─────┐
    │ SEG_VALUE_NAME ┆ len │
    ╞════════════════╪═════╡
    │ MAINSTREAM     ┆ 32  │
    │ UPSCALE        ┆ 11  │
    │ VALUE          ┆ 9   │
    └────────────────┴─────┘


| STORE_NUM | SEG_VALUE_NAME | AVG_WEEKLY_BASKETS |
|-----------|----------------|--------------------|
| 25027     | "MAINSTREAM"   | 43892.923077       |
| 21237     | "MAINSTREAM"   | 38465.128205       |
| 25229     | "MAINSTREAM"   | 34977.435897       |
| 19265     | "MAINSTREAM"   | 31578.134615       |
| 9825      | "MAINSTREAM"   | 29915.903846       |
| 613       | "MAINSTREAM"   | 29386.416667       |
| 2277      | "UPSCALE"      | 54052.519231       |
| 24991     | "UPSCALE"      | 50618.99359        |
| 6179      | "UPSCALE"      | 35287.974359       |
| 2513      | "UPSCALE"      | 32422.99359        |
| 2281      | "UPSCALE"      | 32297.288462       |
| 11993     | "UPSCALE"      | 26100.711538       |
| 25021     | "VALUE"        | 34191.00641        |
| 4259      | "VALUE"        | 31177.333333       |
| 21479     | "VALUE"        | 29435.628205       |
| 23349     | "VALUE"        | 27822.608974       |
| 19523     | "VALUE"        | 24567.75           |
| 6431      | "VALUE"        | 24321.942308       |


    In [12]:


``` python
panel_df = cereal_df.filter(pl.col("STORE_NUM").is_in(selected_stores))
series_ids: list[str] = [
    f"{store}::{product}" for store in selected_stores for product in product_order
]
n_series = len(series_ids)
dates_series = panel_df["WEEK_END_DATE"].unique().sort()
dates = dates_series.to_numpy()
dates_num = np.asarray(mdates.date2num(dates))
split_x = float(dates_num[t_train])


def make_pivot(value: str) -> Float[np.ndarray, " duration n_series"]:
    """Build the dense (week x series) matrix of one panel column.

    Columns follow ``series_ids`` order (store-major, then the product order) so every
    pivot shares the same series axis; a missing cell is an error, not a zero.
    """
    pivot_df = panel_df.pivot(on="series", index="WEEK_END_DATE", values=value).sort(
        "WEEK_END_DATE"
    )
    matrix = pivot_df.select(series_ids).to_numpy().astype(np.float64)
    if matrix.shape != (len(dates), n_series) or np.isnan(matrix).any():
        msg = f"Unexpected pivot for {value!r}: shape {matrix.shape}"
        raise ValueError(msg)
    return matrix


panel_ds = xr.Dataset(
    {
        name: (("time", "series"), make_pivot(column))
        for name, column in {
            "units": "UNITS",
            "x": "x",
            "feature": "FEATURE",
            "display": "DISPLAY",
            "discount": "discount",
            "base_price": "BASE_PRICE",
        }.items()
    },
    coords={"time": dates, "series": series_ids},
)
series_to_product = jnp.asarray(
    [product_order.index(s.split("::")[1]) for s in series_ids], dtype=jnp.int32
)
series_to_store = jnp.asarray(
    [selected_stores.index(int(s.split("::")[0])) for s in series_ids], dtype=jnp.int32
)
series_to_product_np = np.asarray(series_to_product)
series_to_store_np = np.asarray(series_to_store)
print(
    f"panel: {n_series} series = {n_stores} stores x {n_products} "
    f"products | train {t_train} weeks | horizon {HORIZON} weeks"
)
panel_ds
```


    panel: 108 series = 18 stores x 6 products | train 143 weeks | horizon 13 weeks


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
<xarray.Dataset> Size: 820kB
Dimensions:     (time: 156, series: 108)
Coordinates:
  * time        (time) datetime64[s] 1kB 2009-01-14 2009-01-21 ... 2012-01-04
  * series      (series) <U24 10kB '25027::HNC' ... '6431::PL Frosted Wheat'
Data variables:
    units       (time, series) float64 135kB 70.0 181.0 69.0 ... 13.0 6.0 11.0
    x           (time, series) float64 135kB 0.0 -0.2239 0.0 0.0 ... 0.0 0.0 0.0
    feature     (time, series) float64 135kB 0.0 1.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
    display     (time, series) float64 135kB 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
    discount    (time, series) float64 135kB 0.0 0.2006 0.0 0.0 ... 0.0 0.0 0.0
    base_price  (time, series) float64 135kB 2.87 3.14 4.39 ... 3.36 1.56 2.2
```


xarray.Dataset


Dimensions:


- time: 156
- series: 108


Coordinates: (2)


time


(time)


datetime64\[s\]


2009-01-14 ... 2012-01-04


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['2009-01-14T00:00:00', '2009-01-21T00:00:00', '2009-01-28T00:00:00',
           '2009-02-04T00:00:00', '2009-02-11T00:00:00', '2009-02-18T00:00:00',
           '2009-02-25T00:00:00', '2009-03-04T00:00:00', '2009-03-11T00:00:00',
           '2009-03-18T00:00:00', '2009-03-25T00:00:00', '2009-04-01T00:00:00',
           '2009-04-08T00:00:00', '2009-04-15T00:00:00', '2009-04-22T00:00:00',
           '2009-04-29T00:00:00', '2009-05-06T00:00:00', '2009-05-13T00:00:00',
           '2009-05-20T00:00:00', '2009-05-27T00:00:00', '2009-06-03T00:00:00',
           '2009-06-10T00:00:00', '2009-06-17T00:00:00', '2009-06-24T00:00:00',
           '2009-07-01T00:00:00', '2009-07-08T00:00:00', '2009-07-15T00:00:00',
           '2009-07-22T00:00:00', '2009-07-29T00:00:00', '2009-08-05T00:00:00',
           '2009-08-12T00:00:00', '2009-08-19T00:00:00', '2009-08-26T00:00:00',
           '2009-09-02T00:00:00', '2009-09-09T00:00:00', '2009-09-16T00:00:00',
           '2009-09-23T00:00:00', '2009-09-30T00:00:00', '2009-10-07T00:00:00',
           '2009-10-14T00:00:00', '2009-10-21T00:00:00', '2009-10-28T00:00:00',
           '2009-11-04T00:00:00', '2009-11-11T00:00:00', '2009-11-18T00:00:00',
           '2009-11-25T00:00:00', '2009-12-02T00:00:00', '2009-12-09T00:00:00',
           '2009-12-16T00:00:00', '2009-12-23T00:00:00', '2009-12-30T00:00:00',
           '2010-01-06T00:00:00', '2010-01-13T00:00:00', '2010-01-20T00:00:00',
           '2010-01-27T00:00:00', '2010-02-03T00:00:00', '2010-02-10T00:00:00',
           '2010-02-17T00:00:00', '2010-02-24T00:00:00', '2010-03-03T00:00:00',
           '2010-03-10T00:00:00', '2010-03-17T00:00:00', '2010-03-24T00:00:00',
           '2010-03-31T00:00:00', '2010-04-07T00:00:00', '2010-04-14T00:00:00',
           '2010-04-21T00:00:00', '2010-04-28T00:00:00', '2010-05-05T00:00:00',
           '2010-05-12T00:00:00', '2010-05-19T00:00:00', '2010-05-26T00:00:00',
           '2010-06-02T00:00:00', '2010-06-09T00:00:00', '2010-06-16T00:00:00',
           '2010-06-23T00:00:00', '2010-06-30T00:00:00', '2010-07-07T00:00:00',
           '2010-07-14T00:00:00', '2010-07-21T00:00:00', '2010-07-28T00:00:00',
           '2010-08-04T00:00:00', '2010-08-11T00:00:00', '2010-08-18T00:00:00',
           '2010-08-25T00:00:00', '2010-09-01T00:00:00', '2010-09-08T00:00:00',
           '2010-09-15T00:00:00', '2010-09-22T00:00:00', '2010-09-29T00:00:00',
           '2010-10-06T00:00:00', '2010-10-13T00:00:00', '2010-10-20T00:00:00',
           '2010-10-27T00:00:00', '2010-11-03T00:00:00', '2010-11-10T00:00:00',
           '2010-11-17T00:00:00', '2010-11-24T00:00:00', '2010-12-01T00:00:00',
           '2010-12-08T00:00:00', '2010-12-15T00:00:00', '2010-12-22T00:00:00',
           '2010-12-29T00:00:00', '2011-01-05T00:00:00', '2011-01-12T00:00:00',
           '2011-01-19T00:00:00', '2011-01-26T00:00:00', '2011-02-02T00:00:00',
           '2011-02-09T00:00:00', '2011-02-16T00:00:00', '2011-02-23T00:00:00',
           '2011-03-02T00:00:00', '2011-03-09T00:00:00', '2011-03-16T00:00:00',
           '2011-03-23T00:00:00', '2011-03-30T00:00:00', '2011-04-06T00:00:00',
           '2011-04-13T00:00:00', '2011-04-20T00:00:00', '2011-04-27T00:00:00',
           '2011-05-04T00:00:00', '2011-05-11T00:00:00', '2011-05-18T00:00:00',
           '2011-05-25T00:00:00', '2011-06-01T00:00:00', '2011-06-08T00:00:00',
           '2011-06-15T00:00:00', '2011-06-22T00:00:00', '2011-06-29T00:00:00',
           '2011-07-06T00:00:00', '2011-07-13T00:00:00', '2011-07-20T00:00:00',
           '2011-07-27T00:00:00', '2011-08-03T00:00:00', '2011-08-10T00:00:00',
           '2011-08-17T00:00:00', '2011-08-24T00:00:00', '2011-08-31T00:00:00',
           '2011-09-07T00:00:00', '2011-09-14T00:00:00', '2011-09-21T00:00:00',
           '2011-09-28T00:00:00', '2011-10-05T00:00:00', '2011-10-12T00:00:00',
           '2011-10-19T00:00:00', '2011-10-26T00:00:00', '2011-11-02T00:00:00',
           '2011-11-09T00:00:00', '2011-11-16T00:00:00', '2011-11-23T00:00:00',
           '2011-11-30T00:00:00', '2011-12-07T00:00:00', '2011-12-14T00:00:00',
           '2011-12-21T00:00:00', '2011-12-28T00:00:00', '2012-01-04T00:00:00'],
          dtype='datetime64[s]')


series


(series)


\<U24


'25027::HNC' ... '6431::PL Frost...


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['25027::HNC', '25027::Cheerios 12oz', '25027::Cheerios 18oz',
           '25027::Mini Wheats', '25027::PL Honey Nut Oats',
           '25027::PL Frosted Wheat', '21237::HNC', '21237::Cheerios 12oz',
           '21237::Cheerios 18oz', '21237::Mini Wheats',
           '21237::PL Honey Nut Oats', '21237::PL Frosted Wheat', '25229::HNC',
           '25229::Cheerios 12oz', '25229::Cheerios 18oz', '25229::Mini Wheats',
           '25229::PL Honey Nut Oats', '25229::PL Frosted Wheat', '19265::HNC',
           '19265::Cheerios 12oz', '19265::Cheerios 18oz', '19265::Mini Wheats',
           '19265::PL Honey Nut Oats', '19265::PL Frosted Wheat', '9825::HNC',
           '9825::Cheerios 12oz', '9825::Cheerios 18oz', '9825::Mini Wheats',
           '9825::PL Honey Nut Oats', '9825::PL Frosted Wheat', '613::HNC',
           '613::Cheerios 12oz', '613::Cheerios 18oz', '613::Mini Wheats',
           '613::PL Honey Nut Oats', '613::PL Frosted Wheat', '2277::HNC',
           '2277::Cheerios 12oz', '2277::Cheerios 18oz', '2277::Mini Wheats',
           '2277::PL Honey Nut Oats', '2277::PL Frosted Wheat', '24991::HNC',
           '24991::Cheerios 12oz', '24991::Cheerios 18oz', '24991::Mini Wheats',
           '24991::PL Honey Nut Oats', '24991::PL Frosted Wheat', '6179::HNC',
           '6179::Cheerios 12oz', '6179::Cheerios 18oz', '6179::Mini Wheats',
           '6179::PL Honey Nut Oats', '6179::PL Frosted Wheat', '2513::HNC',
           '2513::Cheerios 12oz', '2513::Cheerios 18oz', '2513::Mini Wheats',
           '2513::PL Honey Nut Oats', '2513::PL Frosted Wheat', '2281::HNC',
           '2281::Cheerios 12oz', '2281::Cheerios 18oz', '2281::Mini Wheats',
           '2281::PL Honey Nut Oats', '2281::PL Frosted Wheat', '11993::HNC',
           '11993::Cheerios 12oz', '11993::Cheerios 18oz', '11993::Mini Wheats',
           '11993::PL Honey Nut Oats', '11993::PL Frosted Wheat', '25021::HNC',
           '25021::Cheerios 12oz', '25021::Cheerios 18oz', '25021::Mini Wheats',
           '25021::PL Honey Nut Oats', '25021::PL Frosted Wheat', '4259::HNC',
           '4259::Cheerios 12oz', '4259::Cheerios 18oz', '4259::Mini Wheats',
           '4259::PL Honey Nut Oats', '4259::PL Frosted Wheat', '21479::HNC',
           '21479::Cheerios 12oz', '21479::Cheerios 18oz', '21479::Mini Wheats',
           '21479::PL Honey Nut Oats', '21479::PL Frosted Wheat', '23349::HNC',
           '23349::Cheerios 12oz', '23349::Cheerios 18oz', '23349::Mini Wheats',
           '23349::PL Honey Nut Oats', '23349::PL Frosted Wheat', '19523::HNC',
           '19523::Cheerios 12oz', '19523::Cheerios 18oz', '19523::Mini Wheats',
           '19523::PL Honey Nut Oats', '19523::PL Frosted Wheat', '6431::HNC',
           '6431::Cheerios 12oz', '6431::Cheerios 18oz', '6431::Mini Wheats',
           '6431::PL Honey Nut Oats', '6431::PL Frosted Wheat'], dtype='<U24')


Data variables: (6)


units


(time, series)


float64


70.0 181.0 69.0 ... 13.0 6.0 11.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[ 70., 181.,  69., ...,  17.,   6.,  19.],
           [ 78.,  79.,  43., ...,  24.,  10.,  16.],
           [ 80.,  69.,  51., ...,  31.,  10.,  17.],
           ...,
           [756., 105.,  45., ...,  15.,   7.,  27.],
           [556.,  84.,  29., ...,  11.,   4.,   5.],
           [614.,  92.,  48., ...,  13.,   6.,  11.]], shape=(156, 108))


x


(time, series)


float64


0.0 -0.2239 0.0 0.0 ... 0.0 0.0 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[ 0.        , -0.22394005,  0.        , ...,  0.        ,
             0.        , -0.20731209],
           [ 0.        , -0.17344388,  0.        , ...,  0.        ,
             0.        , -0.16126815],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        , -0.19689533],
           ...,
           [-0.43354292,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [-0.43858072,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [-0.43858072,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ]], shape=(156, 108))


feature


(time, series)


float64


0.0 1.0 0.0 0.0 ... 0.0 0.0 0.0 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0., 1., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [1., 0., 0., ..., 0., 0., 0.],
           [1., 0., 0., ..., 0., 0., 0.],
           [1., 0., 0., ..., 0., 0., 0.]], shape=(156, 108))


display


(time, series)


float64


0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [1., 0., 0., ..., 0., 0., 0.],
           [1., 0., 0., ..., 0., 0., 0.],
           [1., 0., 0., ..., 0., 0., 0.]], shape=(156, 108))


discount


(time, series)


float64


0.0 0.2006 0.0 0.0 ... 0.0 0.0 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.        , 0.20063694, 0.        , ..., 0.        , 0.        ,
            0.18723404],
           [0.        , 0.15923567, 0.        , ..., 0.        , 0.        ,
            0.14893617],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.1787234 ],
           ...,
           [0.35179153, 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.35504886, 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.35504886, 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ]], shape=(156, 108))


base_price


(time, series)


float64


2.87 3.14 4.39 ... 3.36 1.56 2.2


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[2.87, 3.14, 4.39, ..., 2.88, 1.83, 2.35],
           [3.19, 3.14, 4.49, ..., 2.88, 1.85, 2.35],
           [3.19, 3.19, 4.54, ..., 2.88, 1.83, 2.35],
           ...,
           [3.07, 2.83, 4.79, ..., 3.36, 1.55, 2.18],
           [3.07, 2.92, 4.79, ..., 3.36, 1.61, 2.2 ],
           [3.07, 3.02, 4.79, ..., 3.36, 1.56, 2.2 ]], shape=(156, 108))


The inputs tensor keeps time at axis -2, the package-wide convention, with the stacked inputs as a leading axis: the own log price ratio x = \log(\text{PRICE} / \text{BASE\\PRICE}) \le 0, the feature and display flags, two sibling flags (any other product of the panel featured or displayed at the same store-week), and the cross-price block, the log price ratios of all six products at the series' own store. A promotion of the focal product therefore reaches its siblings through their covariates: their cross-price input and their sibling flags change. The counts are kept as `int32`, which the negative binomial likelihood requires; the scoring later casts them to floats. The tensor has shape (11, 156, 108). At the last training week the focal product's base price ranges from 2.61 to 3.07 across the 18 stores, with 11 distinct values, and the private-label twin's from 1.56 to 1.99; the currency numbers below use each store's own prices.


    In [13]:


``` python
x_np = panel_ds["x"].to_numpy()
feature_np = panel_ds["feature"].to_numpy()
display_np = panel_ds["display"].to_numpy()
prices_np = np.zeros((n_products, N_WEEKS, n_series))
sib_feature_np = np.zeros_like(feature_np)
sib_display_np = np.zeros_like(display_np)
for n in range(n_series):
    store_position = int(series_to_store_np[n])
    same_store = [m for m in range(n_series) if int(series_to_store_np[m]) == store_position]
    siblings = [m for m in same_store if m != n]
    for k in range(n_products):
        prices_np[k, :, n] = x_np[:, store_position * n_products + k]
    sib_feature_np[:, n] = feature_np[:, siblings].max(axis=1)
    sib_display_np[:, n] = display_np[:, siblings].max(axis=1)

input_names: list[str] = [
    "x",
    "feature",
    "display",
    "sib_feature",
    "sib_display",
    *[f"price {product}" for product in product_order],
]
n_inputs = len(input_names)
PRICE_BLOCK = 5  # first channel of the cross-price block
covariates = jnp.asarray(
    np.concatenate(
        [
            x_np[None],
            feature_np[None],
            display_np[None],
            sib_feature_np[None],
            sib_display_np[None],
            prices_np,
        ],
        axis=0,
    ),
    dtype=jnp.float32,
)
covariates_train = covariates[:, :t_train, :]
y = jnp.asarray(panel_ds["units"].to_numpy(), dtype=jnp.int32)
y_train = y[:t_train]
y_test = y[t_train:]
y_train_f = np.asarray(y_train, dtype=np.float32)
y_test_f = np.asarray(y_test, dtype=np.float32)
base_price = panel_ds["base_price"].isel(time=t_train - 1).to_numpy()
print(
    f"covariates {covariates.shape} {covariates.dtype} | y_train {y_train.shape} {y_train.dtype}"
)
base_price_table = (
    pl.DataFrame(
        {
            "product": [series_ids[n].split("::")[1] for n in range(n_series)],
            "base_price": base_price,
        }
    )
    .group_by("product", maintain_order=True)
    .agg(
        min=pl.col("base_price").min(),
        median=pl.col("base_price").median(),
        max=pl.col("base_price").max(),
        distinct=pl.col("base_price").n_unique(),
    )
)
print("base price per store at the last training week:")
base_price_table
```


    covariates (11, 156, 108) float32 | y_train (143, 108) int32
    base price per store at the last training week:


| product             | min  | median | max  | distinct |
|---------------------|------|--------|------|----------|
| "HNC"               | 2.61 | 3.02   | 3.07 | 11       |
| "Cheerios 12oz"     | 2.88 | 3.055  | 3.25 | 15       |
| "Cheerios 18oz"     | 4.35 | 4.79   | 4.79 | 3        |
| "Mini Wheats"       | 3.36 | 3.89   | 3.89 | 2        |
| "PL Honey Nut Oats" | 1.56 | 1.895  | 1.99 | 12       |
| "PL Frosted Wheat"  | 2.13 | 2.41   | 2.48 | 15       |


    In [14]:


``` python
focus_stores = [
    int(selected.filter(pl.col("SEG_VALUE_NAME") == segment)["STORE_NUM"][0])
    for segment in ["MAINSTREAM", "UPSCALE", "VALUE"]
]
focus_labels = [f"{store}::{FOCAL}" for store in focus_stores]

fig, axes = plt.subplots(
    nrows=len(focus_labels),
    figsize=(12, 2.6 * len(focus_labels)),
    sharex=True,
    layout="constrained",
)
for ax, label, store in zip(axes, focus_labels, focus_stores, strict=True):
    (units_line,) = ax.plot(
        dates, panel_ds["units"].sel(series=label), color="black", linewidth=1.5, label="units"
    )
    split_line = ax.axvline(
        split_x, color="C3", linestyle="--", linewidth=1, label="train-test split"
    )
    feature_span = ax.fill_between(
        dates,
        0,
        1,
        where=(panel_ds["feature"].sel(series=label) > 0.5).to_numpy().tolist(),
        transform=ax.get_xaxis_transform(),
        color="C0",
        alpha=0.3,
        linewidth=0,
        step="mid",
        label="feature week",
    )
    display_span = ax.fill_between(
        dates,
        0,
        1,
        where=(panel_ds["display"].sel(series=label) > 0.5).to_numpy().tolist(),
        transform=ax.get_xaxis_transform(),
        color="C4",
        alpha=0.25,
        linewidth=0,
        step="mid",
        label="display week",
    )
    ax.set(title=f"{label} ({store_segment[store]})", ylabel="units")
    ax_twin = ax.twinx()
    (discount_line,) = ax_twin.plot(
        dates,
        panel_ds["discount"].sel(series=label).clip(0.0, 1.0),
        color="C1",
        alpha=0.9,
        linewidth=1,
        label="discount depth",
    )
    ax_twin.grid(False)
    ax_twin.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax_twin.set(ylabel="discount", ylim=(0, 0.6))
axes[0].legend(
    handles=[units_line, discount_line, feature_span, display_span, split_line],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.5),
    ncol=5,
    fontsize=11,
)
fig.supxlabel("week")
fig.suptitle(
    "Honey Nut Cheerios units, discounts and mechanics in three focus stores",
    fontsize=16,
    fontweight="bold",
);
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-15-output-1.png" class="figure-img" width="1211" height="791" /></p>
</figure>


# Identification: the estimand, the promotion calendar and the naive elasticity

The quantity we want is the expected weekly units of the focal product under a discount d and a mechanics m, with the other covariates at their factual values. This is a promotional elasticity: the response to a temporary cut below the base price, not the response to a change in the base price itself, which the level of each series absorbs. The regular-price elasticity would need a different design (Bijmolt, van Heerde and Pieters (2005) document that promotional elasticities exceed regular-price ones), and this notebook cannot answer regular-price questions.

Prices were not randomized. The retailer and the manufacturers set the promotion calendar through trade deals, and the same deal sets the cut, the feature and the display together. The causal graph below draws the identifying assumption: conditional on the mechanics flags, the sibling flags, the competitor prices and the seasonal and level terms, the depth of the cut is as good as random with respect to the unobserved demand shocks. The right-hand cluster shows what would break it: a demand shock (a coupon drop, a competitor's promotion in another retailer) that moves both the deal calendar and the units. We cannot test the assumption with these data; the holdout below validates the forecasting engine under the realized calendar, not the counterfactual ones.


    In [15]:


``` python
dag = gr.Digraph()
dag.attr(rankdir="LR")
with dag.subgraph(name="cluster_assumption") as cluster:
    cluster.attr(label="Identifying assumption", color="gray")
    cluster.node("U", label="trade-deal calendar\n(unobserved)", color="lightgray", style="filled")
    cluster.node("d", label="discount depth\n(lever)", color="#2a2eec80", style="filled")
    cluster.node("F", label="feature\n(lever)", color="#2a2eec80", style="filled")
    cluster.node("D", label="display\n(lever)", color="#2a2eec80", style="filled")
    cluster.node(
        "Z",
        label="season, level, competitor prices,\nsibling flags\n(conditioned)",
        shape="box",
        color="#fa7c1780",
        style="filled",
    )
    cluster.node("Y", label="units\n(outcome)", color="#328c0680", style="filled")
    for lever in ("d", "F", "D"):
        cluster.edge("U", lever)
        cluster.edge(lever, "Y")
    cluster.edge("Z", "Y")
    cluster.edge("Z", "U")
with dag.subgraph(name="cluster_violation") as cluster:
    cluster.attr(label="What would break it", color="gray")
    cluster.node(
        "V", label="demand shock or coupons\n(unobserved)", color="lightgray", style="filled"
    )
    cluster.node(
        "U2", label="trade-deal calendar\n(unobserved)", color="lightgray", style="filled"
    )
    cluster.node("d2", label="discount depth\n(lever)", color="#2a2eec80", style="filled")
    cluster.node("Y2", label="units\n(outcome)", color="#328c0680", style="filled")
    cluster.edge("V", "U2")
    cluster.edge("U2", "d2")
    cluster.edge("d2", "Y2")
    cluster.edge("V", "Y2", color="red")
dag
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-16-output-1.svg" class="img-fluid figure-img" /></p>
</figure>


## The naive elasticity and what the flags absorb

Before the Bayesian model, three within-store least-squares regressions of log units on the log price ratio show the identification problem in numbers: alone, with the mechanics flags, the sibling flags and the other five prices, and with the extra depth slopes under feature and display. The regressions remove store fixed effects by demeaning and include annual seasonality and a trend. The pooled row stacks the six products. A fourth specification replaces the log-linear price term by depth bins, a check of the functional form the model assumes.

For the focal product the naive elasticity is -2.60 (standard error 0.06); with the flags it is -1.33, and with the depth slopes -1.08. Pooled over the six products the gap is -1.86 against -0.95: the flags absorb about half of the naive price effect, which is the trade-deal calendar at work. The focal product's full regression gives a feature multiplier of 1.99 and a display multiplier of 1.44, a feature-depth slope of +0.36 (standard error 0.14) and a display-depth slope of +0.13 (standard error 0.14). The least-squares cross terms range from -0.62 to +0.65; the largest is the response of the private-label twin to the focal product's price, +0.65. The binned specification rises with depth, from +0.08 for cuts up to 10\\ to +0.67 above 30\\, while the log-linear line predicts 0.08, 0.22, 0.38 and 0.63 at the bin centers against binned estimates of 0.08, 0.09, 0.26 and 0.67: on its own, a log-linear term overstates the response to moderate cuts. The model keeps the log-linear form, and its depth slopes under mechanics let the response of a featured cut differ from that of a shelf-tag cut, which is where the moderate cuts sit.


    In [16]:


``` python
time_index = np.repeat(np.arange(N_WEEKS), n_series)
week_fraction = time_index / 52.18
long_df = pl.DataFrame(
    {
        "series": np.tile(np.asarray(series_ids), N_WEEKS),
        "product": np.tile(np.asarray([s.split("::")[1] for s in series_ids]), N_WEEKS),
        "time": time_index,
        "log_units": np.log(panel_ds["units"].to_numpy().ravel()),
        "x": x_np.ravel(),
        "lam": -x_np.ravel(),
        "feature": feature_np.ravel(),
        "display": display_np.ravel(),
        "sib_feature": sib_feature_np.ravel(),
        "sib_display": sib_display_np.ravel(),
        "discount": panel_ds["discount"].to_numpy().ravel(),
        **{f"price {product}": prices_np[k].ravel() for k, product in enumerate(product_order)},
    }
).with_columns(
    feature_display=pl.col("feature") * pl.col("display"),
    feature_lam=pl.col("feature") * pl.col("lam"),
    display_lam=pl.col("display") * pl.col("lam"),
    sin1=np.sin(2.0 * np.pi * week_fraction),
    cos1=np.cos(2.0 * np.pi * week_fraction),
    sin2=np.sin(4.0 * np.pi * week_fraction),
    cos2=np.cos(4.0 * np.pi * week_fraction),
    trend=time_index / N_WEEKS,
)
train_long = long_df.filter(pl.col("time") < t_train)
mechanics_terms = ["feature", "display", "feature_display", "sib_feature", "sib_display"]
depth_terms = ["feature_lam", "display_lam"]


def own_elasticity_row(frame: pl.DataFrame, label: str) -> dict[str, float | str]:
    """Own price coefficient under the three specifications, for one product or the pooled panel."""
    naive = within_ols(frame, ["x", *seasonal_terms])
    flags = within_ols(frame, ["x", *mechanics_terms, *seasonal_terms])
    slopes = within_ols(frame, ["x", *mechanics_terms, *depth_terms, *seasonal_terms])
    return {
        "product": label,
        "naive": float(naive["coef"][0]),
        "naive_se": float(naive["se"][0]),
        "with_flags": float(flags["coef"][0]),
        "with_flags_se": float(flags["se"][0]),
        "with_depth_slopes": float(slopes["coef"][0]),
        "with_depth_slopes_se": float(slopes["se"][0]),
    }


own_elasticity_ols = pl.DataFrame(
    [own_elasticity_row(train_long.filter(pl.col("product") == p), p) for p in product_order]
    + [own_elasticity_row(train_long, "pooled")]
)
own_elasticity_ols
```


| product | naive | naive_se | with_flags | with_flags_se | with_depth_slopes | with_depth_slopes_se |
|----|----|----|----|----|----|----|
| "HNC" | -2.601798 | 0.05715 | -1.326977 | 0.07021 | -1.084921 | 0.099629 |
| "Cheerios 12oz" | -0.666798 | 0.051086 | -0.3183 | 0.038673 | -0.31104 | 0.039819 |
| "Cheerios 18oz" | -2.748362 | 0.046263 | -1.961065 | 0.104096 | -1.511146 | 0.160804 |
| "Mini Wheats" | -2.619433 | 0.077762 | -1.558958 | 0.10673 | -1.626816 | 0.128205 |
| "PL Honey Nut Oats" | -1.837123 | 0.160595 | -1.446412 | 0.173565 | -1.468914 | 0.181495 |
| "PL Frosted Wheat" | -1.129432 | 0.081269 | -0.8505 | 0.08354 | -0.886696 | 0.087231 |
| "pooled" | -1.86253 | 0.026677 | -0.948401 | 0.028264 | -0.674752 | 0.033019 |


    In [17]:


``` python
hnc_long = train_long.filter(pl.col("product") == FOCAL)
cross_terms = [f"price {product}" for product in product_order if product != FOCAL]
hnc_full_ols = within_ols(
    hnc_long, ["x", *mechanics_terms, *depth_terms, *cross_terms, *seasonal_terms]
).with_columns(
    multiplier=pl.when(pl.col("term").is_in(mechanics_terms))
    .then(pl.col("coef").exp())
    .otherwise(None)
)
hnc_full_ols.filter(~pl.col("term").is_in(seasonal_terms))
```


| term                      | coef      | se       | multiplier |
|---------------------------|-----------|----------|------------|
| "x"                       | -0.957322 | 0.101149 | null       |
| "feature"                 | 0.689083  | 0.052251 | 1.991889   |
| "display"                 | 0.362616  | 0.061912 | 1.437084   |
| "feature_display"         | -0.054911 | 0.071154 | 0.946569   |
| "sib_feature"             | -0.089341 | 0.018196 | 0.914534   |
| "sib_display"             | -0.030933 | 0.017986 | 0.969541   |
| "feature_lam"             | 0.36053   | 0.144082 | null       |
| "display_lam"             | 0.126645  | 0.143309 | null       |
| "price Cheerios 12oz"     | -0.232351 | 0.043367 | null       |
| "price Cheerios 18oz"     | -0.301489 | 0.052951 | null       |
| "price Mini Wheats"       | -0.128686 | 0.076184 | null       |
| "price PL Honey Nut Oats" | 0.11818   | 0.135306 | null       |
| "price PL Frosted Wheat"  | 0.167267  | 0.085503 | null       |


    In [18]:


``` python
cross_rows = []
for product in product_order:
    others = [f"price {other}" for other in product_order if other != product]
    fit = within_ols(
        train_long.filter(pl.col("product") == product),
        ["x", *mechanics_terms, *others, *seasonal_terms],
    )
    coefs = dict(zip(fit["term"].to_list(), fit["coef"].to_list(), strict=True))
    row: dict[str, float | str] = {"units of": product}
    for other in product_order:
        row[f"price of {other}"] = coefs["x"] if other == product else coefs[f"price {other}"]
    cross_rows.append(row)
cross_ols = pl.DataFrame(cross_rows)
cross_values = cross_ols.drop("units of").to_numpy()
off_diagonal = cross_values[~np.eye(n_products, dtype=bool)]
print(f"OLS cross terms: min {off_diagonal.min():.2f} | max {off_diagonal.max():.2f}")
cross_ols
```


    OLS cross terms: min -0.62 | max 0.65


| units of | price of HNC | price of Cheerios 12oz | price of Cheerios 18oz | price of Mini Wheats | price of PL Honey Nut Oats | price of PL Frosted Wheat |
|----|----|----|----|----|----|----|
| "HNC" | -1.145766 | -0.224495 | -0.316926 | -0.130353 | 0.162423 | 0.177489 |
| "Cheerios 12oz" | 0.07092 | -0.34379 | 0.277201 | -0.057408 | -0.62447 | 0.349092 |
| "Cheerios 18oz" | -0.047858 | 0.198809 | -1.988508 | 0.184071 | 0.119586 | 0.179472 |
| "Mini Wheats" | 0.024222 | -0.010586 | 0.010404 | -1.557781 | 0.385047 | 0.145947 |
| "PL Honey Nut Oats" | 0.654352 | 0.010516 | 0.069029 | -0.030701 | -1.495217 | 0.500309 |
| "PL Frosted Wheat" | 0.011771 | 0.020995 | 0.127626 | 0.499231 | 0.255877 | -0.815959 |


    In [19]:


``` python
bin_edges = [0.02, 0.10, 0.20, 0.30, 1.0]
bin_labels = ["cut (2%, 10%]", "cut (10%, 20%]", "cut (20%, 30%]", "cut > 30%"]
binned = hnc_long.with_columns(
    [
        ((pl.col("discount") > lo) & (pl.col("discount") <= hi)).cast(pl.Float64).alias(label)
        for label, lo, hi in zip(bin_labels, bin_edges[:-1], bin_edges[1:], strict=True)
    ]
)
binned_ols = within_ols(binned, [*bin_labels, *mechanics_terms, *seasonal_terms]).filter(
    pl.col("term").is_in(bin_labels)
)
hnc_slope = float(own_elasticity_ols.filter(pl.col("product") == FOCAL)["with_flags"][0])
bin_mids = np.array([0.06, 0.15, 0.25, 0.38])
binned_ols = binned_ols.with_columns(
    log_linear_at_bin_center=pl.Series(hnc_slope * np.log1p(-bin_mids)),
)
binned_ols
```


| term              | coef     | se       | log_linear_at_bin_center |
|-------------------|----------|----------|--------------------------|
| "cut (2%, 10%\]"  | 0.076028 | 0.076769 | 0.082107                 |
| "cut (10%, 20%\]" | 0.093826 | 0.027315 | 0.215659                 |
| "cut (20%, 30%\]" | 0.263718 | 0.048844 | 0.381747                 |
| "cut \> 30%"      | 0.668813 | 0.04247  | 0.634342                 |


## The weeks that identify the elasticity

The own elasticity of a product is identified by the weeks in which its price moved without a feature or a display, the `TPR_ONLY` weeks, plus the variation of the cut inside feature and display weeks. The first table counts them per product and per series; a product with a few dozen identifying store-weeks leans on the prior and on the partial pooling across stores. The second table gives the support of the focal product's discount depth under each mechanics over all 77 stores, which decides the grid of counterfactual policies: the shaded cells in the later figures are the depths with fewer than 20 observed store-weeks within \pm 2.5 points. The third block shows how the six log price ratios move together within a store-week, and how often a sibling is featured when the focal product is.

The focal product has 212 shelf-tag-only store-weeks in the training panel, between 6 and 22 per store; Cheerios 18 oz has 23, at most 4 per store, so its own elasticity leans on the depth variation inside its feature and display weeks and on the prior. Over all 77 stores the focal product's cut depth has a median of 14\\ under a shelf tag, 20\\ under a feature and 27\\ under feature with display; 17\\ of the display weeks and 18\\ of the feature weeks carry no cut. The thin cells are a display at 10\\, 15\\, 35\\ and 40\\, a feature at 5\\, and a shelf-tag cut at 5\\ or 40\\. The within-store correlations of the log price ratios stay below 0.30, and a sibling is featured in 27\\ of the store-weeks in which the focal product is featured, against 37\\ otherwise.


    In [20]:


``` python
train_panel_df = panel_df.filter(pl.col("WEEK_END_DATE").is_in(all_weeks[:t_train].to_list()))
identification = (
    train_panel_df.group_by("product", maintain_order=True)
    .agg(
        tpr_only=pl.col("TPR_ONLY").sum(),
        feature=pl.col("FEATURE").sum(),
        display=pl.col("DISPLAY").sum(),
    )
    .join(
        train_panel_df.group_by("series", "product")
        .agg(pl.col("TPR_ONLY").sum().alias("tpr_weeks"))
        .group_by("product")
        .agg(
            tpr_per_series_min=pl.col("tpr_weeks").min(),
            tpr_per_series_median=pl.col("tpr_weeks").median(),
            tpr_per_series_max=pl.col("tpr_weeks").max(),
        ),
        on="product",
    )
    .with_columns(
        order=pl.col("product").replace_strict({p: i for i, p in enumerate(product_order)})
    )
    .sort("order")
    .drop("order")
)
identification
```


| product | tpr_only | feature | display | tpr_per_series_min | tpr_per_series_median | tpr_per_series_max |
|----|----|----|----|----|----|----|
| "HNC" | 212 | 230 | 199 | 6 | 10.5 | 22 |
| "Cheerios 12oz" | 894 | 263 | 272 | 31 | 53.0 | 62 |
| "Cheerios 18oz" | 23 | 224 | 227 | 0 | 1.0 | 4 |
| "Mini Wheats" | 168 | 287 | 141 | 4 | 9.0 | 13 |
| "PL Honey Nut Oats" | 394 | 94 | 104 | 5 | 18.5 | 39 |
| "PL Frosted Wheat" | 604 | 137 | 103 | 22 | 35.5 | 44 |


    In [21]:


``` python
DEPTH_GRID = np.round(np.arange(0.0, 0.401, 0.05), 2)
MECHANICS: dict[str, tuple[float, float]] = {
    "TPR-only": (0.0, 0.0),
    "display": (0.0, 1.0),
    "feature": (1.0, 0.0),
    "feature + display": (1.0, 1.0),
}
hnc_all_stores = cereal_df.filter(pl.col("product") == FOCAL).with_columns(
    depth=pl.col("discount").clip(lower_bound=0.0)
)
support_rows = []
support_counts: dict[str, np.ndarray] = {}
for mechanics_name in MECHANICS:
    subset = hnc_all_stores.filter(pl.col("mechanics") == mechanics_name)
    depth = subset["depth"].to_numpy()
    counts = np.array([int(np.sum(np.abs(depth - d) <= 0.025)) for d in DEPTH_GRID])
    support_counts[mechanics_name] = counts
    support_rows.append(
        {
            "mechanics": mechanics_name,
            "store_weeks": depth.size,
            "share_without_cut": float(np.mean(depth <= 0.02)),
            "p05": float(np.quantile(depth, 0.05)),
            "p50": float(np.quantile(depth, 0.50)),
            "p95": float(np.quantile(depth, 0.95)),
            **{f"n@{d:.2f}": int(c) for d, c in zip(DEPTH_GRID, counts, strict=True)},
        }
    )
support_table = pl.DataFrame(support_rows)
support_table
```


| mechanics | store_weeks | share_without_cut | p05 | p50 | p95 | n@0.00 | n@0.05 | n@0.10 | n@0.15 | n@0.20 | n@0.25 | n@0.30 | n@0.35 | n@0.40 |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| "TPR-only" | 1088 | 0.0 | 0.102894 | 0.144591 | 0.336067 | 0 | 0 | 336 | 398 | 169 | 67 | 58 | 36 | 6 |
| "display" | 334 | 0.170659 | 0.0 | 0.28031 | 0.492163 | 57 | 24 | 9 | 9 | 30 | 34 | 20 | 14 | 3 |
| "feature" | 456 | 0.177632 | 0.0 | 0.198613 | 0.451561 | 83 | 18 | 39 | 67 | 34 | 37 | 22 | 88 | 28 |
| "feature + display" | 863 | 0.077636 | 0.0 | 0.266423 | 0.482759 | 68 | 37 | 65 | 95 | 81 | 100 | 54 | 156 | 38 |


    In [22]:


``` python
x_store_week = x_np[:t_train].reshape(t_train, n_stores, n_products).reshape(-1, n_products)
price_correlation = pl.DataFrame(
    {
        "x of": product_order,
        **{p: np.corrcoef(x_store_week.T)[:, k] for k, p in enumerate(product_order)},
    }
)
feature_store_week = feature_np[:t_train].reshape(t_train, n_stores, n_products)
hnc_featured = feature_store_week[:, :, FOCAL_INDEX] == 1
sibling_featured = np.delete(feature_store_week, FOCAL_INDEX, axis=2).max(axis=2) == 1
print(f"P(sibling featured | HNC featured): {sibling_featured[hnc_featured].mean():.2f}")
print(f"P(sibling featured | HNC not featured): {sibling_featured[~hnc_featured].mean():.2f}")
price_correlation
```


    P(sibling featured | HNC featured): 0.27
    P(sibling featured | HNC not featured): 0.37


| x of | HNC | Cheerios 12oz | Cheerios 18oz | Mini Wheats | PL Honey Nut Oats | PL Frosted Wheat |
|----|----|----|----|----|----|----|
| "HNC" | 1.0 | 0.297353 | 0.123075 | -0.078147 | 0.103915 | -0.050287 |
| "Cheerios 12oz" | 0.297353 | 1.0 | 0.278749 | -0.013253 | 0.093249 | -0.189566 |
| "Cheerios 18oz" | 0.123075 | 0.278749 | 1.0 | 0.001388 | -0.024091 | -0.102736 |
| "Mini Wheats" | -0.078147 | -0.013253 | 0.001388 | 1.0 | -0.016099 | -0.051092 |
| "PL Honey Nut Oats" | 0.103915 | 0.093249 | -0.024091 | -0.016099 | 1.0 | -0.103449 |
| "PL Frosted Wheat" | -0.050287 | -0.189566 | -0.102736 | -0.051092 | -0.103449 | 1.0 |


# Model specification

For series i let \text{prod}(i) be its product and s(i) its store; t indexes weeks and u \le t the past weeks; y\_{t,i} is the unit count and \mu\_{t,i} its conditional mean. The observed inputs at week t are the log price ratio x\_{t,i} = \log(\text{PRICE}\_{t,i} / \text{BASE\\PRICE}\_{t,i}) \le 0, the log depth \lambda\_{t,i} = -x\_{t,i}, the flags F\_{t,i} (feature) and D\_{t,i} (display), the sibling flags F^{\text{sib}}\_{t,i} and D^{\text{sib}}\_{t,i}, and the log price ratios x\_{k,t,s} of every product k at store s. The latent quantities are the initial level \ell\_{0,i} and the weekly level innovations \delta\_{u,i} (a random walk on the log scale); the annual Fourier basis f(t) with two harmonics and product coefficients \beta_p; the own promotional elasticity \varepsilon_i of each series, partially pooled around a product mean; the cross elasticities \gamma\_{k,p}, the response of product p's units to product k's price, zero on the diagonal; the mechanics effects b^{\text{feat}}\_p, b^{\text{disp}}\_p and their interaction b^{\text{fd}}\_p; the depth slopes under mechanics b^{\text{feat},\lambda}\_p and b^{\text{disp},\lambda}\_p; the sibling-mechanics effects b^{\text{sib,feat}}\_p and b^{\text{sib,disp}}\_p; and the negative binomial concentration \phi_p. With p = \text{prod}(i) and s = s(i):

 \log \mu\_{t,i} = \ell\_{0,i} + \sum\_{u \le t} \delta\_{u,i} + f(t)^\top \beta_p + \varepsilon_i\\ x\_{t,i} + \sum\_{k \ne p} \gamma\_{k,p}\\ x\_{k,t,s} + b^{\text{feat}}\_p F\_{t,i} + b^{\text{disp}}\_p D\_{t,i} + b^{\text{fd}}\_p F\_{t,i} D\_{t,i} + b^{\text{feat},\lambda}\_p F\_{t,i} \lambda\_{t,i} + b^{\text{disp},\lambda}\_p D\_{t,i} \lambda\_{t,i} + b^{\text{sib,feat}}\_p F^{\text{sib}}\_{t,i} + b^{\text{sib,disp}}\_p D^{\text{sib}}\_{t,i} 

 y\_{t,i} \sim \text{NegativeBinomial}(\mu\_{t,i}, \phi_p), 

in the mean and concentration parameterization. Because \lambda = -\log(1 - d) for a fractional discount d, the depth response under a mechanics m = (F_m, D_m) is (1 - d)^{\varepsilon_m} with the mechanics-specific elasticity \varepsilon_m = \varepsilon - b^{\text{feat},\lambda} F_m - b^{\text{disp},\lambda} D_m (a positive slope makes the response steeper), and the mechanics uplift at zero depth is e^{b_m} with b_m = b^{\text{feat}} F_m + b^{\text{disp}} D_m + b^{\text{fd}} F_m D_m. Every decision threshold below uses \varepsilon_m, never the bare \varepsilon. The sibling-mechanics effects are averages over "any sibling featured or displayed", so under a policy in which only the focal product is featured the mechanics part of the cannibalization is attenuated toward the average sibling. Remark: with a Poisson likelihood the same code runs with `dist.Poisson`; with continuous sales the link would be a Normal on the log scale.

We write each additive piece of \log \mu as a component that samples its own sites and returns its contribution, the pattern of the NumPyro [Hilbert space Gaussian process example](https://num.pyro.ai/en/stable/examples/hsgp.html): the model creates the plates once, samples the centering parameters, sums the pieces and calls [predict](../../../reference/models.predict.md#numpyro_forecast.models.predict). The store-level elasticities and the cross terms use NumPyro's [`LocScaleReparam`](https://num.pyro.ai/en/stable/reparam.html) with a sampled centering value: the reparameterization accepts a centering value in \[0, 1\], and we give it a \text{Uniform}(0, 1) prior. A reparameterization does not change the model, so this value has no likelihood and NUTS cannot learn it from the data; the diagnostics show that its posterior is its prior. We keep it on these two sites to make that point visible, because under variational inference the same device does select a parameterization ([Gorinova, Moore and Hoffman, 2020](https://arxiv.org/abs/1906.03028)). The level innovations are fully non-centered: a centering value shared by thousands of innovation sites cannot move at all under NUTS, and non-centering is the natural choice for small innovation scales. Only the 30 off-diagonal cross terms are sampled and scattered into the 6 \times 6 matrix. The Fourier basis is computed once outside the model and sliced inside it, because the cached helper must not run under a trace.


    In [23]:


``` python
@dataclass(frozen=True)
class CerealPriors:
    """Prior hyperparameters of the cereal demand model (all on the log scale)."""

    eps_loc: float
    eps_sd: float
    eps_store_sd: float
    drift_mu: float
    drift_sigma: float
    conc_mu: float
    conc_sigma: float
    mech_loc: float
    mech_sd: float
    aux_sd: float
    cross_sd: float
    seasonal_sd: float
    level_loc: float
    level_sd: float


def local_level(
    h: Horizon, series_plate: numpyro.plate, priors: CerealPriors
) -> Float[Array, " duration n_series"]:
    """Random-walk level per series: initial level plus the cumulative sum of the innovations."""
    with series_plate:
        level0 = cast(
            "Array", numpyro.sample("level0", dist.Normal(priors.level_loc, priors.level_sd))
        )
        drift_scale = cast(
            "Array",
            numpyro.sample("drift_scale", dist.LogNormal(priors.drift_mu, priors.drift_sigma)),
        )
        # innovations opens its own time plate at dim=-2 and registers the horizon
        # innovations as a separate site, so the forecast continues the walk. The walk
        # is fully non-centered: a sampled centering value shared by thousands of
        # innovation sites cannot move under NUTS, and non-centering is the natural
        # choice for small innovation scales.
        drift = innovations(
            h,
            "drift",
            lambda: dist.Normal(0.0, drift_scale),
            reparam=LocScaleReparam(centered=0.0),
        )
    return level0 + jnp.cumsum(drift, axis=-2)


def annual_seasonality(
    fourier: Float[Array, " duration n_fourier"],
    product_plate: numpyro.plate,
    fourier_plate: numpyro.plate,
    series_to_product: Int[Array, " n_series"],
    priors: CerealPriors,
) -> Float[Array, " duration n_series"]:
    """Annual Fourier seasonality with one coefficient vector per product."""
    with fourier_plate, product_plate:
        beta_s = cast("Array", numpyro.sample("beta_s", dist.Normal(0.0, priors.seasonal_sd)))
    return (fourier @ beta_s)[:, series_to_product]


def own_price_effect(
    x: Float[Array, " duration n_series"],
    product_plate: numpyro.plate,
    series_plate: numpyro.plate,
    series_to_product: Int[Array, " n_series"],
    centered_eps: Float[Array, ""],
    priors: CerealPriors,
) -> Float[Array, " duration n_series"]:
    """Own promotional elasticity per series, partially pooled around its product mean."""
    with product_plate:
        eps_prod = cast(
            "Array", numpyro.sample("eps_prod", dist.Normal(priors.eps_loc, priors.eps_sd))
        )
    eps_scale = cast("Array", numpyro.sample("eps_scale", dist.HalfNormal(priors.eps_store_sd)))
    with series_plate, handlers.reparam(config={"eps": LocScaleReparam(centered=centered_eps)}):
        eps = cast(
            "Array", numpyro.sample("eps", dist.Normal(eps_prod[series_to_product], eps_scale))
        )
    return eps * x


def cross_price_effect(
    prices: Float[Array, " n_products duration n_series"],
    pair_plate: numpyro.plate,
    series_to_product: Int[Array, " n_series"],
    pair_rows: Int[Array, " n_pairs"],
    pair_cols: Int[Array, " n_pairs"],
    centered_gamma: Float[Array, ""],
    priors: CerealPriors,
    n_products: int,
) -> Float[Array, " duration n_series"]:
    """Cross-price contribution from the 30 off-diagonal elasticities, shrunk toward zero."""
    cross_scale = cast("Array", numpyro.sample("cross_scale", dist.HalfNormal(priors.cross_sd)))
    with (
        pair_plate,
        handlers.reparam(config={"gamma_offdiag": LocScaleReparam(centered=centered_gamma)}),
    ):
        gamma_offdiag = cast(
            "Array", numpyro.sample("gamma_offdiag", dist.Normal(0.0, cross_scale))
        )
    gamma = cast(
        "Array",
        numpyro.deterministic(
            "gamma",
            jnp.zeros((n_products, n_products)).at[pair_rows, pair_cols].set(gamma_offdiag),
        ),
    )
    # cross[t, n] = sum_k gamma[k, p(n)] * prices[k, t, n]: for series n (product p(n) sold at
    # store s(n)), take the log price ratio of every product k at the same store and weight it
    # by the cross elasticity of p(n) with respect to k; the own cell k == p(n) is zero in
    # gamma and is handled by own_price_effect.
    return jnp.einsum("ktn,kn->tn", prices, gamma[:, series_to_product])


def mechanics_effect(
    feature: Float[Array, " duration n_series"],
    display: Float[Array, " duration n_series"],
    lam: Float[Array, " duration n_series"],
    sib_feature: Float[Array, " duration n_series"],
    sib_display: Float[Array, " duration n_series"],
    product_plate: numpyro.plate,
    series_to_product: Int[Array, " n_series"],
    priors: CerealPriors,
) -> Float[Array, " duration n_series"]:
    """Feature and display uplifts, their interaction, the depth slopes and the sibling effects."""
    with product_plate:
        b_feat = cast(
            "Array", numpyro.sample("b_feat", dist.Normal(priors.mech_loc, priors.mech_sd))
        )
        b_disp = cast(
            "Array", numpyro.sample("b_disp", dist.Normal(priors.mech_loc, priors.mech_sd))
        )
        b_fd = cast("Array", numpyro.sample("b_fd", dist.Normal(0.0, priors.aux_sd)))
        b_feat_depth = cast(
            "Array", numpyro.sample("b_feat_depth", dist.Normal(0.0, priors.aux_sd))
        )
        b_disp_depth = cast(
            "Array", numpyro.sample("b_disp_depth", dist.Normal(0.0, priors.aux_sd))
        )
        b_sib_feat = cast("Array", numpyro.sample("b_sib_feat", dist.Normal(0.0, priors.aux_sd)))
        b_sib_disp = cast("Array", numpyro.sample("b_sib_disp", dist.Normal(0.0, priors.aux_sd)))
    p = series_to_product
    return (
        b_feat[p] * feature
        + b_disp[p] * display
        + b_fd[p] * feature * display
        + b_feat_depth[p] * feature * lam
        + b_disp_depth[p] * display * lam
        + b_sib_feat[p] * sib_feature
        + b_sib_disp[p] * sib_display
    )


def dispersion(
    product_plate: numpyro.plate, series_to_product: Int[Array, " n_series"], priors: CerealPriors
) -> Float[Array, " n_series"]:
    """Negative binomial concentration per product, gathered to the series axis."""
    with product_plate:
        conc = cast(
            "Array", numpyro.sample("conc", dist.LogNormal(priors.conc_mu, priors.conc_sigma))
        )
    return conc[series_to_product]


def make_cereal_model(
    series_to_product: Int[Array, " n_series"],
    fourier_full: Float[Array, " duration_full n_fourier"],
    priors: CerealPriors,
    n_products: int,
    n_series: int,
) -> ForecastModel:
    """Build the cereal demand model as the plain ``(covariates, data=None)`` callable.

    Parameters
    ----------
    series_to_product
        Product index of every series, shape ``(n_series,)``.
    fourier_full
        Annual Fourier basis over the full duration, computed outside any trace.
    priors
        Prior hyperparameters.
    n_products
        Number of products (the cross matrix is ``n_products`` square).
    n_series
        Number of store-product series (the trailing observation axis).

    Returns
    -------
    ForecastModel
        The model function, callable as ``model(covariates)`` for prior sampling and
        ``model(covariates, data)`` for training and forecasting.
    """
    rows_np, cols_np = np.where(~np.eye(n_products, dtype=bool))
    pair_rows = jnp.asarray(rows_np, dtype=jnp.int32)
    pair_cols = jnp.asarray(cols_np, dtype=jnp.int32)
    n_pairs = int(rows_np.size)
    n_fourier = int(fourier_full.shape[-1])

    def cereal_model(
        covariates: Float[Array, " inputs duration n_series"],
        data: Int[Array, " t_obs n_series"] | None = None,
    ) -> None:
        """Sample the joint model (the drivers call this for training and forecasting)."""
        h = Horizon.from_data(covariates, data)
        duration = covariates.shape[-2]
        x = covariates[0]
        feature = covariates[1]
        display = covariates[2]
        sib_feature = covariates[3]
        sib_display = covariates[4]
        prices = covariates[PRICE_BLOCK:]
        assert isinstance(prices, Float[Array, " n_products duration n_series"])  # ty: ignore[invalid-argument-type]
        lam = -x

        product_plate = numpyro.plate("product", n_products, dim=-1)
        fourier_plate = numpyro.plate("fourier", n_fourier, dim=-2)
        pair_plate = numpyro.plate("pair", n_pairs, dim=-1)
        series_plate = numpyro.plate("series", n_series, dim=-1)
        centered_eps = cast("Array", numpyro.sample("centered_eps", dist.Uniform(0.0, 1.0)))
        centered_gamma = cast("Array", numpyro.sample("centered_gamma", dist.Uniform(0.0, 1.0)))

        eta = (
            local_level(h, series_plate, priors)
            + annual_seasonality(
                fourier_full[:duration], product_plate, fourier_plate, series_to_product, priors
            )
            + own_price_effect(
                x, product_plate, series_plate, series_to_product, centered_eps, priors
            )
            + cross_price_effect(
                prices,
                pair_plate,
                series_to_product,
                pair_rows,
                pair_cols,
                centered_gamma,
                priors,
                n_products,
            )
            + mechanics_effect(
                feature,
                display,
                lam,
                sib_feature,
                sib_display,
                product_plate,
                series_to_product,
                priors,
            )
        )
        conc_series = dispersion(product_plate, series_to_product, priors)
        if h.future > 0:
            # The conditional mean over the horizon, for the decision layer; registered
            # only in forecast mode so the training posterior carries no extra array.
            numpyro.deterministic("mu_future", jnp.exp(eta)[h.t_obs :])
        predict(h, lambda e: dist.NegativeBinomial2(jnp.exp(e), conc_series), eta)

    return cereal_model


fourier_full = jnp.asarray(fourier_features(N_WEEKS, 52.18, 2))
pair_rows_np, pair_cols_np = np.where(~np.eye(n_products, dtype=bool))
pair_labels = [
    f"{product_order[k]} -> {product_order[p]}"
    for k, p in zip(pair_rows_np, pair_cols_np, strict=True)
]
fourier_names = ["sin1", "sin2", "cos1", "cos2"]
```


# Priors and prior predictive checks

Every prior has a reason, and the table prints the 94\\ interval of each one with the quantity it implies, so the numbers quoted here come from the cells. The product elasticity prior \text{Normal}(-1.5, 1) is centered where the meta-analysis of [Bijmolt, van Heerde and Pieters (2005)](https://doi.org/10.1509/jmkr.42.2.141.62296) puts price elasticities (an average of about -2.6 across studies, with promotional elasticities above regular-price ones in magnitude), and it leaves the positive tail open so the data can reject the sign. The store deviations around the product mean have a \text{HalfNormal}(0.5) scale, deviations of up to about one unit. The weekly innovation scale of the level comes from `preliz.maxent`: we ask for a log-normal with 94\\ of its mass between weekly innovations of 1\\ and 8\\, because a wider prior lets the level absorb one-week promotion spikes (a check against the least-squares mechanics effects follows the fit). The concentration prior \text{LogNormal}(2, 1) implies, at 80 units a week, a coefficient of variation near the within-store spread of the focal product's weeks. The feature and display effects have a \text{Normal}(0.5, 0.5) prior, a median multiplier of 1.65 against the least-squares multipliers printed above, with negative values allowed. The interaction, the depth slopes and the sibling effects are centered at zero. The cross terms share a \text{HalfNormal}(0.5) scale that shrinks the 30 cells toward zero. The seasonal coefficients have a \text{Normal}(0, 0.2) prior, the initial level a \text{Normal}(3, 2) prior on the log scale, and the two centering parameters a \text{Uniform}(0, 1) prior, which is also their posterior. `preliz.maxent` returns \text{LogNormal}(-3.3, 0.488), a median weekly innovation of 3.7\\. The prior predictive bands cover the observed units of the focus series, and the prior implied multiplier of the focal product at a 35\\ cut under feature with display has a median of 4.4 with a 94\\ HDI from 0.1 to 29.3: wide, as a prior should be, and centered on a plausible value.


    In [24]:


``` python
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    drift_scale_prior = pz.maxent(pz.LogNormal(), lower=0.01, upper=0.08, mass=0.94, plot=False)
print(f"weekly drift scale prior: {drift_scale_prior} | median {drift_scale_prior.median():.3f}")

priors = CerealPriors(
    eps_loc=-1.5,
    eps_sd=1.0,
    eps_store_sd=0.5,
    drift_mu=float(drift_scale_prior.mu),
    drift_sigma=float(drift_scale_prior.sigma),
    conc_mu=2.0,
    conc_sigma=1.0,
    mech_loc=0.5,
    mech_sd=0.5,
    aux_sd=0.5,
    cross_sd=0.5,
    seasonal_sd=0.2,
    level_loc=3.0,
    level_sd=2.0,
)
prior_distributions = {
    "eps_prod": pz.Normal(priors.eps_loc, priors.eps_sd),
    "eps_scale": pz.HalfNormal(priors.eps_store_sd),
    "drift_scale": pz.LogNormal(priors.drift_mu, priors.drift_sigma),
    "conc": pz.LogNormal(priors.conc_mu, priors.conc_sigma),
    "b_feat, b_disp": pz.Normal(priors.mech_loc, priors.mech_sd),
    "b_fd, depth slopes, sibling effects": pz.Normal(0.0, priors.aux_sd),
    "cross_scale": pz.HalfNormal(priors.cross_sd),
    "beta_s": pz.Normal(0.0, priors.seasonal_sd),
    "level0": pz.Normal(priors.level_loc, priors.level_sd),
}
prior_rows = []
for site, distribution in prior_distributions.items():
    lower, upper = distribution.eti(mass=0.94)
    median = float(distribution.median())
    if site == "drift_scale":
        implied = f"level drift over 13 weeks about {np.sqrt(HORIZON) * median:.0%} (median)"
    elif site == "conc":
        implied = f"CV at 80 units about {np.sqrt(1 / 80 + 1 / median):.2f} (median)"
    elif site == "b_feat, b_disp":
        implied = (
            f"multiplier {np.exp(lower):.2f} to {np.exp(upper):.2f}, median {np.exp(median):.2f}"
        )
    elif site == "level0":
        implied = f"{np.exp(lower):.1f} to {np.exp(upper):.0f} units, median {np.exp(median):.0f}"
    else:
        implied = ""
    prior_rows.append(
        {
            "site": site,
            "prior": str(distribution),
            "median": median,
            "eti94_lower": float(lower),
            "eti94_upper": float(upper),
            "implied": implied,
        }
    )
prior_table = pl.DataFrame(prior_rows)
prior_table
```


    weekly drift scale prior: LogNormal(mu=-3.3, sigma=0.488) | median 0.037


| site | prior | median | eti94_lower | eti94_upper | implied |
|----|----|----|----|----|----|
| "eps_prod" | "Normal(mu=-1.5, sigma=1)" | -1.5 | -3.38 | 0.38 | "" |
| "eps_scale" | "HalfNormal(sigma=0.5)" | 0.337245 | 0.02 | 1.09 | "" |
| "drift_scale" | "LogNormal(mu=-3.3, sigma=0.488)" | 0.036894 | 0.01 | 0.09 | "level drift over 13 weeks about 13% (median)" |
| "conc" | "LogNormal(mu=2, sigma=1)" | 7.389056 | 1.13 | 48.46 | "CV at 80 units about 0.38 (median)" |
| "b_feat, b_disp" | "Normal(mu=0.5, sigma=0.5)" | 0.5 | -0.44 | 1.44 | "multiplier 0.64 to 4.22, median 1.65" |
| "b_fd, depth slopes, sibling effects" | "Normal(mu=0, sigma=0.5)" | 0.0 | -0.94 | 0.94 | "" |
| "cross_scale" | "HalfNormal(sigma=0.5)" | 0.337245 | 0.02 | 1.09 | "" |
| "beta_s" | "Normal(mu=0, sigma=0.2)" | 0.0 | -0.38 | 0.38 | "" |
| "level0" | "Normal(mu=3, sigma=2)" | 3.0 | -0.76 | 6.76 | "0.5 to 863 units, median 20" |


    In [25]:


``` python
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), layout="constrained")
panels = ["eps_prod", "eps_scale", "drift_scale", "conc", "b_feat, b_disp", "level0"]
for ax, site in zip(axes.ravel(), panels, strict=True):
    prior_distributions[site].plot_pdf(ax=ax, legend=None, color="C0")
    ax.set(title=f"{site}: {prior_distributions[site]}", xlabel="value", ylabel="density")
fig.suptitle("Prior distributions", fontsize=16, fontweight="bold");
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-26-output-1.png" class="figure-img" width="1544" height="811" /></p>
</figure>


    In [26]:


``` python
model = make_cereal_model(series_to_product, fourier_full, priors, n_products, n_series)
numpyro.render_model(model, model_args=(covariates_train, y_train), render_distributions=True)
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-27-output-1.svg" class="img-fluid figure-img" /></p>
</figure>


    In [27]:


``` python
def hdi_label(prob: float, prefix: str = "") -> str:
    r"""Legend label for an HDI band, e.g. ``$94\%$ HDI``."""
    percent = f"{prob:.0%}".replace("%", r"\%")
    return f"{prefix}${percent}$ HDI"


def hdi_bounds(draws: Float[np.ndarray, " sample"], prob: float) -> tuple[float, float]:
    """Highest density interval of one-dimensional draws (the shortest interval with mass ``prob``)."""
    ordered = np.sort(np.asarray(draws, dtype=np.float64))
    n_draws = ordered.size
    width = int(np.floor(prob * n_draws))
    widths = ordered[width:] - ordered[: n_draws - width]
    start = int(np.argmin(widths))
    return float(ordered[start]), float(ordered[start + width])


hdi_probs = (0.5, 0.94)
hdi_alphas = [0.6, 0.3]

rng_key, key_prior = random.split(rng_key)
prior_sites = ["obs", "eps_prod", "b_feat", "b_disp", "b_fd", "b_feat_depth", "b_disp_depth"]
prior_draws = Predictive(model, num_samples=500, return_sites=prior_sites)(
    key_prior, covariates_train
)
prior_obs = np.asarray(prior_draws["obs"], dtype=np.float32)

pc = az.plot_lm(
    predictions_to_datatree(
        prior_obs[:, :, [series_ids.index(label) for label in focus_labels]],
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
    figure_kwargs={"figsize": (12, 2.8 * len(focus_labels))},
)
truth_da = (
    panel_ds["units"]
    .isel(time=slice(None, t_train))
    .sel(series=focus_labels)
    .assign_coords(time=dates_num[:t_train])
    .rename("t")
)
x_da = xr.DataArray(dates_num[:t_train], dims=["time"], coords={"time": dates_num[:t_train]})
pc.map(
    az.visuals.line_xy, "truth", data=truth_da, x=x_da, ignore_aes=pc.aes_set, color="black", lw=1
)
for label in focus_labels:
    ax = pc.get_target("t", {"series": label})
    ax.set_title(label, fontsize=11)
    ax.set_yscale("log")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
bands = pc.viz["ci_band"]["t"].sel(series=focus_labels[-1])
band_handles = []
for prob in (0.94, 0.5):
    band = bands.sel(prob=prob).item()
    band.set_label(hdi_label(prob))
    band_handles.append(band)
truth_line = pc.viz["truth"]["t"].sel(series=focus_labels[-1]).item()
truth_line.set_label("observed units")
pc.get_target("t", {"series": focus_labels[-1]}).legend(
    handles=[*band_handles, truth_line], loc="upper left", fontsize=9
)
fig = pc.viz["figure"].item()
fig.supxlabel("week")
fig.supylabel("units (log scale)")
fig.suptitle(
    "Prior predictive check on the training window", fontsize=16, fontweight="bold", y=1.02
);
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-28-output-1.png" class="figure-img" width="1211" height="872" /></p>
</figure>


    In [28]:


``` python
depth_check = 0.35
lam_check = float(-np.log1p(-depth_check))
eps_prior = np.asarray(prior_draws["eps_prod"])[:, FOCAL_INDEX]
uplift_prior = np.exp(
    np.asarray(prior_draws["b_feat"])[:, FOCAL_INDEX]
    + np.asarray(prior_draws["b_disp"])[:, FOCAL_INDEX]
    + np.asarray(prior_draws["b_fd"])[:, FOCAL_INDEX]
)
eps_m_prior = (
    eps_prior
    - np.asarray(prior_draws["b_feat_depth"])[:, FOCAL_INDEX]
    - np.asarray(prior_draws["b_disp_depth"])[:, FOCAL_INDEX]
)
multiplier_prior = uplift_prior * (1.0 - depth_check) ** eps_m_prior
lower_m, upper_m = hdi_bounds(multiplier_prior, 0.94)
print(
    f"prior implied HNC multiplier at a {depth_check:.0%} cut with feature + display: "
    f"median {np.median(multiplier_prior):.1f}x, 94% HDI {lower_m:.1f}x to {upper_m:.1f}x"
)
```


    prior implied HNC multiplier at a 35% cut with feature + display: median 4.4x, 94% HDI 0.1x to 29.3x


# Inference with NUTS

The panel is small enough for full NUTS: four chains of 1{,}000 warmup and 1{,}000 draws each, run in parallel on four host devices. Two settings matter. The initialization is `init_to_median`, because NumPyro's default uniform initialization in the unconstrained space can put the cumulative level of a series far outside the range where the negative binomial mean is finite. And we ask NUTS for the number of leapfrog steps of every iteration, which gives the tree depth: a sampler stuck at the depth cap of 10 is the sign of a badly conditioned posterior. The fit takes just under ten minutes of wall time, with no divergences and every iteration at tree depth 8 or 9, so the depth cap is never reached.


    In [29]:


``` python
%%time


def fit_nuts(rng_key: Array, model: ForecastModel, data: Array, covariates: Array) -> MCMC:
    """Fit ``model`` with NUTS (4 chains, 1,000 warmup and 1,000 draws each, median init)."""
    mcmc = MCMC(
        NUTS(model, target_accept_prob=0.9, init_strategy=init_to_median()),
        num_warmup=1_000,
        num_samples=1_000,
        num_chains=4,
        progress_bar=False,
    )
    mcmc.run(rng_key, covariates, data, extra_fields=("diverging", "num_steps"))
    return mcmc


rng_key, key_fit = random.split(rng_key)
mcmc = fit_nuts(key_fit, model, y_train, covariates_train)
# The chains run asynchronously on the host devices; block on the draws so the wall time is real.
posterior = jax.block_until_ready(mcmc.get_samples())
n_draws = int(posterior["eps_prod"].shape[0])
```


    CPU times: user 1h 9min 13s, sys: 16min 15s, total: 1h 25min 29s
    Wall time: 9min 59s


    In [30]:


``` python
num_steps = np.asarray(mcmc.get_extra_fields()["num_steps"])
tree_depth = np.ceil(np.log2(num_steps + 1)).astype(int)
depth_values, depth_counts = np.unique(tree_depth, return_counts=True)
print(
    f"posterior draws: {n_draws} | divergences: {int(np.asarray(mcmc.get_extra_fields()['diverging']).sum())}"
)
print(f"share of iterations at the depth-10 cap: {np.mean(num_steps == 1023):.1%}")
pl.DataFrame({"tree_depth": depth_values, "share": depth_counts / depth_counts.sum()})
```


    posterior draws: 4000 | divergences: 0
    share of iterations at the depth-10 cap: 0.0%


| tree_depth | share   |
|------------|---------|
| 8          | 0.99825 |
| 9          | 0.00175 |


# Diagnostics

The posterior, the in-sample predictive and the holdout forecast go into one ArviZ tree with named coordinates. The convergence table lists the product-level parameters and the hyperparameters: the maximum \hat R is 1.00 and the smallest bulk effective sample size is 712, for the focal product's concentration. A second table lists the two centering parameters: their posterior is their prior, with means of 0.50 and 0.49 and 94\\ HDIs from about 0.02 to 0.98. The store-elasticity centering has a bulk effective sample size of 14 and an \hat R of 1.22, because NUTS moves it only along a ridge of the decentered coordinates; its trace shows every chain wandering across the whole unit interval. This does not affect the other sites, whose posteriors do not depend on the parameterization. The trace plots cover the same sites.


    In [31]:


``` python
coords = {
    "series": series_ids,
    "obs_dim": series_ids,
    "product": product_order,
    "competitor": product_order,
    "pair": pair_labels,
    "fourier": fourier_names,
    "input": input_names,
}
product_sites = [
    "eps_prod",
    "conc",
    "b_feat",
    "b_disp",
    "b_fd",
    "b_feat_depth",
    "b_disp_depth",
    "b_sib_feat",
    "b_sib_disp",
]
posterior_dims = {
    "level0": ["series"],
    "drift_scale": ["series"],
    "eps": ["series"],
    "eps_decentered": ["series"],
    "drift": ["time", "series"],
    "drift_decentered": ["time", "series"],
    **{site: ["product"] for site in product_sites},
    "beta_s": ["fourier", "product"],
    "gamma": ["competitor", "product"],
    "gamma_offdiag": ["pair"],
    "gamma_offdiag_decentered": ["pair"],
}
rng_key, key_tree = random.split(rng_key)
tree = to_datatree(
    key_tree,
    model,
    posterior,
    y_train,
    covariates,
    num_chains=4,
    time_coord=list(dates),
    covariate_dims=["input", "time", "series"],
    coords=coords,
    posterior_dims=posterior_dims,
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
│       Dimensions:                   (chain: 4, draw: 1000, product: 6, fourier: 4,
│                                      time: 143, series: 108, competitor: 6, pair: 30)
│       Coordinates:
│         * chain                     (chain) int64 32B 0 1 2 3
│         * draw                      (draw) int64 8kB 0 1 2 3 4 ... 995 996 997 998 999
│         * product                   (product) <U17 408B 'HNC' ... 'PL Frosted Wheat'
│         * fourier                   (fourier) <U4 64B 'sin1' 'sin2' 'cos1' 'cos2'
│         * time                      (time) datetime64[s] 1kB 2009-01-14 ... 2011-10-05
│         * series                    (series) <U24 10kB '25027::HNC' ... '6431::PL F...
│         * competitor                (competitor) <U17 408B 'HNC' ... 'PL Frosted Wh...
│         * pair                      (pair) <U37 4kB 'HNC -> Cheerios 12oz' ... 'PL ...
│       Data variables: (12/23)
│           b_disp                    (chain, draw, product) float32 96kB 0.383 ... 0...
│           b_disp_depth              (chain, draw, product) float32 96kB 0.02171 ......
│           b_fd                      (chain, draw, product) float32 96kB -0.02492 .....
│           b_feat                    (chain, draw, product) float32 96kB 0.7715 ... ...
│           b_feat_depth              (chain, draw, product) float32 96kB 0.06066 ......
│           b_sib_disp                (chain, draw, product) float32 96kB -0.03735 .....
│           ...                        ...
│           eps_prod                  (chain, draw, product) float32 96kB -0.9976 ......
│           eps_scale                 (chain, draw) float32 16kB 0.2884 ... 0.2401
│           gamma                     (chain, draw, competitor, product) float32 576kB ...
│           gamma_offdiag             (chain, draw, pair) float32 480kB 0.1502 ... 0....
│           gamma_offdiag_decentered  (chain, draw, pair) float32 480kB 0.2907 ... 0....
│           level0                    (chain, draw, series) float32 2MB 4.483 ... 2.286
│       Attributes:
│           created_at:                 2026-09-07T11:31:36.864779+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
├── Group: /posterior_predictive
│       Dimensions:  (chain: 4, draw: 1000, time: 143, obs_dim: 108)
│       Coordinates:
│         * chain    (chain) int64 32B 0 1 2 3
│         * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│         * time     (time) datetime64[s] 1kB 2009-01-14 2009-01-21 ... 2011-10-05
│         * obs_dim  (obs_dim) <U24 10kB '25027::HNC' ... '6431::PL Frosted Wheat'
│       Data variables:
│           obs      (chain, draw, time, obs_dim) int32 247MB 71 214 63 44 ... 19 7 26
│       Attributes:
│           created_at:                 2026-09-07T11:31:59.219193+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
├── Group: /observed_data
│       Dimensions:  (time: 143, obs_dim: 108)
│       Coordinates:
│         * time     (time) datetime64[s] 1kB 2009-01-14 2009-01-21 ... 2011-10-05
│         * obs_dim  (obs_dim) <U24 10kB '25027::HNC' ... '6431::PL Frosted Wheat'
│       Data variables:
│           obs      (time, obs_dim) int32 62kB 70 181 69 46 50 100 ... 19 20 18 14 18
│       Attributes:
│           created_at:                 2026-09-07T11:31:59.222122+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /constant_data
│       Dimensions:     (input: 11, time: 143, series: 108)
│       Coordinates:
│         * input       (input) <U23 1kB 'x' 'feature' ... 'price PL Frosted Wheat'
│         * time        (time) datetime64[s] 1kB 2009-01-14 2009-01-21 ... 2011-10-05
│         * series      (series) <U24 10kB '25027::HNC' ... '6431::PL Frosted Wheat'
│       Data variables:
│           covariates  (input, time, series) float32 680kB 0.0 -0.2239 0.0 ... 0.0 0.0
│       Attributes:
│           created_at:                 2026-09-07T11:31:59.223942+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                []
├── Group: /predictions
│       Dimensions:  (chain: 4, draw: 1000, time: 13, obs_dim: 108)
│       Coordinates:
│         * chain    (chain) int64 32B 0 1 2 3
│         * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 993 994 995 996 997 998 999
│         * time     (time) datetime64[s] 104B 2011-10-12 2011-10-19 ... 2012-01-04
│         * obs_dim  (obs_dim) <U24 10kB '25027::HNC' ... '6431::PL Frosted Wheat'
│       Data variables:
│           obs      (chain, draw, time, obs_dim) int32 22MB 173 125 62 58 ... 16 16 38
│       Attributes:
│           created_at:                 2026-09-07T11:32:01.499300+00:00
│           creation_library:           ArviZ
│           creation_library_version:   1.2.0
│           creation_library_language:  Python
│           sample_dims:                ['chain', 'draw']
└── Group: /predictions_constant_data
        Dimensions:     (input: 11, time: 13, series: 108)
        Coordinates:
          * input       (input) <U23 1kB 'x' 'feature' ... 'price PL Frosted Wheat'
          * time        (time) datetime64[s] 104B 2011-10-12 2011-10-19 ... 2012-01-04
          * series      (series) <U24 10kB '25027::HNC' ... '6431::PL Frosted Wheat'
        Data variables:
            covariates  (input, time, series) float32 62kB 0.0 0.0 0.0 ... 0.0 0.0 0.0
        Attributes:
            created_at:                 2026-09-07T11:32:01.499812+00:00
            creation_library:           ArviZ
            creation_library_version:   1.2.0
            creation_library_language:  Python
            sample_dims:                []
```


xarray.DataTree


/posterior(36)

Dimensions:


- chain: 4
- draw: 1000
- product: 6
- fourier: 4
- time: 143
- series: 108
- competitor: 6
- pair: 30


Coordinates: (8)


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


product


(product)


\<U17


'HNC' ... 'PL Frosted Wheat'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['HNC', 'Cheerios 12oz', 'Cheerios 18oz', 'Mini Wheats','PL Honey Nut Oats', 'PL Frosted Wheat'], dtype='<U17')


fourier


(fourier)


\<U4


'sin1' 'sin2' 'cos1' 'cos2'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['sin1', 'sin2', 'cos1', 'cos2'], dtype='<U4')


time


(time)


datetime64\[s\]


2009-01-14 ... 2011-10-05


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['2009-01-14T00:00:00', '2009-01-21T00:00:00', '2009-01-28T00:00:00','2009-02-04T00:00:00', '2009-02-11T00:00:00', '2009-02-18T00:00:00','2009-02-25T00:00:00', '2009-03-04T00:00:00', '2009-03-11T00:00:00','2009-03-18T00:00:00', '2009-03-25T00:00:00', '2009-04-01T00:00:00','2009-04-08T00:00:00', '2009-04-15T00:00:00', '2009-04-22T00:00:00','2009-04-29T00:00:00', '2009-05-06T00:00:00', '2009-05-13T00:00:00','2009-05-20T00:00:00', '2009-05-27T00:00:00', '2009-06-03T00:00:00','2009-06-10T00:00:00', '2009-06-17T00:00:00', '2009-06-24T00:00:00','2009-07-01T00:00:00', '2009-07-08T00:00:00', '2009-07-15T00:00:00','2009-07-22T00:00:00', '2009-07-29T00:00:00', '2009-08-05T00:00:00','2009-08-12T00:00:00', '2009-08-19T00:00:00', '2009-08-26T00:00:00','2009-09-02T00:00:00', '2009-09-09T00:00:00', '2009-09-16T00:00:00','2009-09-23T00:00:00', '2009-09-30T00:00:00', '2009-10-07T00:00:00','2009-10-14T00:00:00', '2009-10-21T00:00:00', '2009-10-28T00:00:00','2009-11-04T00:00:00', '2009-11-11T00:00:00', '2009-11-18T00:00:00','2009-11-25T00:00:00', '2009-12-02T00:00:00', '2009-12-09T00:00:00','2009-12-16T00:00:00', '2009-12-23T00:00:00', '2009-12-30T00:00:00','2010-01-06T00:00:00', '2010-01-13T00:00:00', '2010-01-20T00:00:00','2010-01-27T00:00:00', '2010-02-03T00:00:00', '2010-02-10T00:00:00','2010-02-17T00:00:00', '2010-02-24T00:00:00', '2010-03-03T00:00:00','2010-03-10T00:00:00', '2010-03-17T00:00:00', '2010-03-24T00:00:00','2010-03-31T00:00:00', '2010-04-07T00:00:00', '2010-04-14T00:00:00','2010-04-21T00:00:00', '2010-04-28T00:00:00', '2010-05-05T00:00:00','2010-05-12T00:00:00', '2010-05-19T00:00:00', '2010-05-26T00:00:00','2010-06-02T00:00:00', '2010-06-09T00:00:00', '2010-06-16T00:00:00','2010-06-23T00:00:00', '2010-06-30T00:00:00', '2010-07-07T00:00:00','2010-07-14T00:00:00', '2010-07-21T00:00:00', '2010-07-28T00:00:00','2010-08-04T00:00:00', '2010-08-11T00:00:00', '2010-08-18T00:00:00','2010-08-25T00:00:00', '2010-09-01T00:00:00', '2010-09-08T00:00:00','2010-09-15T00:00:00', '2010-09-22T00:00:00', '2010-09-29T00:00:00','2010-10-06T00:00:00', '2010-10-13T00:00:00', '2010-10-20T00:00:00','2010-10-27T00:00:00', '2010-11-03T00:00:00', '2010-11-10T00:00:00','2010-11-17T00:00:00', '2010-11-24T00:00:00', '2010-12-01T00:00:00','2010-12-08T00:00:00', '2010-12-15T00:00:00', '2010-12-22T00:00:00','2010-12-29T00:00:00', '2011-01-05T00:00:00', '2011-01-12T00:00:00','2011-01-19T00:00:00', '2011-01-26T00:00:00', '2011-02-02T00:00:00','2011-02-09T00:00:00', '2011-02-16T00:00:00', '2011-02-23T00:00:00','2011-03-02T00:00:00', '2011-03-09T00:00:00', '2011-03-16T00:00:00','2011-03-23T00:00:00', '2011-03-30T00:00:00', '2011-04-06T00:00:00','2011-04-13T00:00:00', '2011-04-20T00:00:00', '2011-04-27T00:00:00','2011-05-04T00:00:00', '2011-05-11T00:00:00', '2011-05-18T00:00:00','2011-05-25T00:00:00', '2011-06-01T00:00:00', '2011-06-08T00:00:00','2011-06-15T00:00:00', '2011-06-22T00:00:00', '2011-06-29T00:00:00','2011-07-06T00:00:00', '2011-07-13T00:00:00', '2011-07-20T00:00:00','2011-07-27T00:00:00', '2011-08-03T00:00:00', '2011-08-10T00:00:00','2011-08-17T00:00:00', '2011-08-24T00:00:00', '2011-08-31T00:00:00','2011-09-07T00:00:00', '2011-09-14T00:00:00', '2011-09-21T00:00:00','2011-09-28T00:00:00', '2011-10-05T00:00:00'], dtype='datetime64[s]')


series


(series)


\<U24


'25027::HNC' ... '6431::PL Frost...


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['25027::HNC', '25027::Cheerios 12oz', '25027::Cheerios 18oz','25027::Mini Wheats', '25027::PL Honey Nut Oats','25027::PL Frosted Wheat', '21237::HNC', '21237::Cheerios 12oz','21237::Cheerios 18oz', '21237::Mini Wheats','21237::PL Honey Nut Oats', '21237::PL Frosted Wheat', '25229::HNC','25229::Cheerios 12oz', '25229::Cheerios 18oz', '25229::Mini Wheats','25229::PL Honey Nut Oats', '25229::PL Frosted Wheat', '19265::HNC','19265::Cheerios 12oz', '19265::Cheerios 18oz', '19265::Mini Wheats','19265::PL Honey Nut Oats', '19265::PL Frosted Wheat', '9825::HNC','9825::Cheerios 12oz', '9825::Cheerios 18oz', '9825::Mini Wheats','9825::PL Honey Nut Oats', '9825::PL Frosted Wheat', '613::HNC','613::Cheerios 12oz', '613::Cheerios 18oz', '613::Mini Wheats','613::PL Honey Nut Oats', '613::PL Frosted Wheat', '2277::HNC','2277::Cheerios 12oz', '2277::Cheerios 18oz', '2277::Mini Wheats','2277::PL Honey Nut Oats', '2277::PL Frosted Wheat', '24991::HNC','24991::Cheerios 12oz', '24991::Cheerios 18oz', '24991::Mini Wheats','24991::PL Honey Nut Oats', '24991::PL Frosted Wheat', '6179::HNC','6179::Cheerios 12oz', '6179::Cheerios 18oz', '6179::Mini Wheats','6179::PL Honey Nut Oats', '6179::PL Frosted Wheat', '2513::HNC','2513::Cheerios 12oz', '2513::Cheerios 18oz', '2513::Mini Wheats','2513::PL Honey Nut Oats', '2513::PL Frosted Wheat', '2281::HNC','2281::Cheerios 12oz', '2281::Cheerios 18oz', '2281::Mini Wheats','2281::PL Honey Nut Oats', '2281::PL Frosted Wheat', '11993::HNC','11993::Cheerios 12oz', '11993::Cheerios 18oz', '11993::Mini Wheats','11993::PL Honey Nut Oats', '11993::PL Frosted Wheat', '25021::HNC','25021::Cheerios 12oz', '25021::Cheerios 18oz', '25021::Mini Wheats','25021::PL Honey Nut Oats', '25021::PL Frosted Wheat', '4259::HNC','4259::Cheerios 12oz', '4259::Cheerios 18oz', '4259::Mini Wheats','4259::PL Honey Nut Oats', '4259::PL Frosted Wheat', '21479::HNC','21479::Cheerios 12oz', '21479::Cheerios 18oz', '21479::Mini Wheats','21479::PL Honey Nut Oats', '21479::PL Frosted Wheat', '23349::HNC','23349::Cheerios 12oz', '23349::Cheerios 18oz', '23349::Mini Wheats','23349::PL Honey Nut Oats', '23349::PL Frosted Wheat', '19523::HNC','19523::Cheerios 12oz', '19523::Cheerios 18oz', '19523::Mini Wheats','19523::PL Honey Nut Oats', '19523::PL Frosted Wheat', '6431::HNC','6431::Cheerios 12oz', '6431::Cheerios 18oz', '6431::Mini Wheats','6431::PL Honey Nut Oats', '6431::PL Frosted Wheat'], dtype='<U24')


competitor


(competitor)


\<U17


'HNC' ... 'PL Frosted Wheat'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['HNC', 'Cheerios 12oz', 'Cheerios 18oz', 'Mini Wheats','PL Honey Nut Oats', 'PL Frosted Wheat'], dtype='<U17')


pair


(pair)


\<U37


'HNC -\> Cheerios 12oz' ... 'PL F...


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['HNC -> Cheerios 12oz', 'HNC -> Cheerios 18oz', 'HNC -> Mini Wheats','HNC -> PL Honey Nut Oats', 'HNC -> PL Frosted Wheat','Cheerios 12oz -> HNC', 'Cheerios 12oz -> Cheerios 18oz','Cheerios 12oz -> Mini Wheats', 'Cheerios 12oz -> PL Honey Nut Oats','Cheerios 12oz -> PL Frosted Wheat', 'Cheerios 18oz -> HNC','Cheerios 18oz -> Cheerios 12oz', 'Cheerios 18oz -> Mini Wheats','Cheerios 18oz -> PL Honey Nut Oats','Cheerios 18oz -> PL Frosted Wheat', 'Mini Wheats -> HNC','Mini Wheats -> Cheerios 12oz', 'Mini Wheats -> Cheerios 18oz','Mini Wheats -> PL Honey Nut Oats', 'Mini Wheats -> PL Frosted Wheat','PL Honey Nut Oats -> HNC', 'PL Honey Nut Oats -> Cheerios 12oz','PL Honey Nut Oats -> Cheerios 18oz','PL Honey Nut Oats -> Mini Wheats','PL Honey Nut Oats -> PL Frosted Wheat', 'PL Frosted Wheat -> HNC','PL Frosted Wheat -> Cheerios 12oz','PL Frosted Wheat -> Cheerios 18oz', 'PL Frosted Wheat -> Mini Wheats','PL Frosted Wheat -> PL Honey Nut Oats'], dtype='<U37')


Data variables: (23)


b_disp


(chain, draw, product)


float32


0.383 0.483 0.252 ... 0.3268 0.1394


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.38298243, 0.4829767 , 0.25195694, 0.19495496, 0.21985757,0.21247046],[0.38969648, 0.42635787, 0.23642576, 0.2530405 , 0.23554663,0.07472137],[0.33137813, 0.51766205, 0.41135725, 0.24465775, 0.18115598,0.17758115],...,[0.3024402 , 0.41174725, 0.28969902, 0.38079104, 0.35671553,0.16669056],[0.4595768 , 0.41116285, 0.24682534, 0.23526868, 0.27740678,0.17086111],[0.2842354 , 0.5483339 , 0.3211232 , 0.22225815, 0.24678655,0.18595064]],[[0.27418724, 0.47301286, 0.29563737, 0.2491765 , 0.24608494,0.0515247 ],[0.24657257, 0.38634846, 0.31943378, 0.24771823, 0.26439503,0.07126686],[0.37087107, 0.45767945, 0.26838222, 0.22400019, 0.20712937,0.3000833 ],...[0.37001538, 0.45552677, 0.25210384, 0.41263244, 0.36853346,0.19094774],[0.34083185, 0.4946139 , 0.3052854 , 0.3748869 , 0.27579576,0.1770411 ],[0.3532552 , 0.42496023, 0.1938304 , 0.21586144, 0.17711847,0.16649723]],[[0.350612  , 0.47253793, 0.28104964, 0.26572692, 0.29520008,0.24396257],[0.22799909, 0.4011553 , 0.33134753, 0.27435517, 0.30044666,0.17232932],[0.37366083, 0.5691593 , 0.2920778 , 0.19694088, 0.24047755,0.17892635],...,[0.35276026, 0.45340136, 0.32763103, 0.25714293, 0.23964863,0.19475812],[0.3253186 , 0.43777162, 0.20332508, 0.24539624, 0.31232533,0.16476648],[0.36262646, 0.4447004 , 0.19344468, 0.24744223, 0.32680047,0.13935429]]], shape=(4, 1000, 6), dtype=float32)


b_disp_depth


(chain, draw, product)


float32


0.02171 -0.2467 ... -1.458 0.4295


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 0.02171086, -0.24666356, -0.05261073,  0.6081825 ,-0.43014333,  0.11302017],[ 0.14191037, -0.01351458,  0.13450812,  0.54250634,-1.3641629 ,  0.89735156],[ 0.13049588, -0.0774826 , -0.2247631 ,  0.4960841 ,-0.5499046 ,  0.46958706],...,[ 0.13486165,  0.06513014, -0.10109747,  0.08519599,-1.3697128 ,  0.8847885 ],[-0.11286335,  0.12013753,  0.15385947,  0.41997728,-1.4895209 , -0.20010231],[ 0.23619488, -0.35557973, -0.2496633 ,  0.8219321 ,-0.53293616,  0.49237156]],[[ 0.27104276,  0.20089875, -0.18034391,  0.5429804 ,-1.3098059 ,  0.83914024],[ 0.24191684,  0.1418809 , -0.32106414,  0.7740956 ,-1.3768308 ,  0.84759784],[-0.13273527, -0.00680863, -0.09105856,  0.8290644 ,-1.1688266 ,  0.09711398],...[ 0.00813816,  0.07405788,  0.10511207,  0.27913877,-1.3573656 ,  0.22762963],[-0.00217401, -0.14307204,  0.06411114,  0.08178885,-0.93139315,  0.6158682 ],[ 0.29268515,  0.1690878 , -0.0558381 ,  0.40454453,-1.2756989 ,  0.48085856]],[[ 0.11790276, -0.20923676, -0.14322582,  0.48944888,-1.5593446 , -0.04387714],[ 0.155657  ,  0.35105568, -0.26207572,  0.72529984,-1.7055765 ,  0.729625  ],[ 0.11594864, -0.44848526,  0.03645325,  0.54957896,-1.1846163 ,  0.26694575],...,[-0.08266811,  0.02312678, -0.47793925,  0.56824833,-0.8861492 ,  0.37787664],[ 0.16492607, -0.00253594,  0.29326695,  0.56036425,-1.8532768 ,  0.5250837 ],[ 0.12782355,  0.06672426,  0.25000453,  0.4995648 ,-1.4581128 ,  0.42952755]]], shape=(4, 1000, 6), dtype=float32)


b_fd


(chain, draw, product)


float32


-0.02492 -0.05948 ... -0.03248


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[-2.49167643e-02, -5.94768487e-02, -2.63251096e-01,1.42583624e-02, -1.28579736e-01, -3.08918161e-03],[-1.08562119e-01,  1.30588179e-02, -3.00784886e-01,-1.25968233e-02,  8.15300941e-02,  2.04393324e-02],[-9.47594792e-02, -1.37059107e-01, -3.09432566e-01,2.38446835e-02,  7.84047134e-03,  3.34579498e-02],...,[ 1.31361699e-03, -6.21700734e-02, -2.23815277e-01,2.23979866e-03, -9.85560641e-02, -1.48071453e-01],[-1.11276075e-01, -5.61218651e-04, -3.84500623e-01,8.50159973e-02, -3.77868190e-02,  1.49555027e-01],[ 9.90642421e-03, -1.31363720e-01, -2.35382497e-01,-8.23498368e-02, -1.44546434e-01, -1.24840081e-01]],[[ 3.50680165e-02, -1.22373372e-01, -1.61810681e-01,6.74187765e-02,  6.95788395e-03,  7.22351074e-02],[-1.31931873e-02, -2.30129752e-02, -1.83011293e-01,-4.38039713e-02, -3.57217938e-02,  7.26742148e-02],[-5.32477498e-02, -1.63991019e-01, -2.52083212e-01,-2.60524871e-03,  1.39141858e-01, -1.38068244e-01],...-4.26204167e-02, -1.67562068e-01, -6.55921623e-02],[-1.68850385e-02, -3.45337689e-02, -4.00621861e-01,-4.29903753e-02, -6.94306940e-02, -1.74493659e-02],[-1.45229995e-01, -1.26608863e-01, -1.64014906e-01,7.62161836e-02,  4.88180481e-02, -3.69298793e-02]],[[-2.85983086e-02, -3.08024976e-02, -2.49581173e-01,4.63282056e-02, -7.56634846e-02, -6.10184856e-03],[ 2.68917978e-02, -9.58364904e-02, -1.63969740e-01,-9.86109376e-02, -3.07464413e-02, -7.00282156e-02],[-4.81008068e-02, -8.90254155e-02, -3.02285254e-01,8.28582719e-02, -1.55789286e-01, -2.72593461e-02],...,[ 3.51803899e-02, -4.23650071e-02, -1.15513653e-01,8.67611989e-02, -3.62493843e-02, -5.65460324e-02],[-8.16712752e-02, -7.96082094e-02, -3.53282601e-01,-1.43662374e-02, -8.22340176e-02,  5.09718508e-02],[-5.83254024e-02, -7.68978894e-02, -3.17516327e-01,-1.81979202e-02, -1.14107274e-01, -3.24790813e-02]]],shape=(4, 1000, 6), dtype=float32)


b_feat


(chain, draw, product)


float32


0.7715 0.7509 ... 0.1401 0.2483


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 0.77151024,  0.7509022 ,  0.15728539,  0.3362277 ,0.07471015,  0.2579289 ],[ 0.78164357,  0.754857  ,  0.12401855,  0.36826587,0.06216415,  0.17698784],[ 0.83897895,  0.8539299 ,  0.15420784,  0.33927792,0.11780034,  0.22315   ],...,[ 0.7343178 ,  0.83046347,  0.02597279,  0.2874872 ,0.14026517,  0.2832293 ],[ 0.7761257 ,  0.8139515 ,  0.1512219 ,  0.29136178,0.11544582,  0.13011494],[ 0.7859337 ,  0.76362383,  0.00914444,  0.3176979 ,0.11797298,  0.27949464]],[[ 0.7909072 ,  0.8372529 , -0.01507905,  0.36060756,0.09420171,  0.23523806],[ 0.82983655,  0.86552286,  0.05447362,  0.36709878,0.05545016,  0.2538846 ],[ 0.7429869 ,  0.8307701 ,  0.18399894,  0.27279025,0.04086264,  0.22415152],...[ 0.70427257,  0.7334037 ,  0.23364553,  0.2781603 ,0.11342175,  0.19690561],[ 0.7052783 ,  0.7122424 ,  0.22307962,  0.22207257,0.18137045,  0.20422886],[ 0.7835556 ,  0.85924846,  0.07710125,  0.33200446,0.1043661 ,  0.26509333]],[[ 0.7527424 ,  0.8243769 ,  0.02120349,  0.31392372,0.10984162,  0.27784434],[ 0.756482  ,  0.8045277 , -0.04036887,  0.3147325 ,0.12785387,  0.2595528 ],[ 0.7515834 ,  0.7308302 ,  0.2580024 ,  0.31539914,0.09020399,  0.26181483],...,[ 0.6815459 ,  0.7982662 , -0.04926921,  0.34409925,0.2458494 ,  0.26670304],[ 0.8144565 ,  0.8067404 ,  0.2511486 ,  0.26177946,0.05824902,  0.2381849 ],[ 0.7900295 ,  0.81267655,  0.21389012,  0.2520355 ,0.14006214,  0.24834344]]], shape=(4, 1000, 6), dtype=float32)


b_feat_depth


(chain, draw, product)


float32


0.06066 -0.1726 ... 0.02595 -0.4756


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 0.06065899, -0.17262895,  0.73160475, -0.5841254 ,0.35251313, -0.4355895 ],[ 0.12256724, -0.4974574 ,  0.9805494 , -0.7138139 ,0.21585448, -0.04879822],[-0.06339999, -0.44890952,  0.75285816, -0.4906728 ,-0.18201196, -0.32567614],...,[ 0.01111843, -0.49827614,  0.9423282 , -0.2772034 ,0.6964552 , -0.5053258 ],[ 0.05136193, -0.6902235 ,  0.9381291 , -0.46700728,0.11926115, -0.00526002],[-0.02869798, -0.08288907,  1.0539671 , -0.40544245,0.1191798 , -0.05704654]],[[-0.21413094, -0.6320684 ,  1.0235184 , -0.6039379 ,0.11553454, -0.13644038],[-0.22558397, -0.7365394 ,  0.896744  , -0.55509925,0.6843707 , -0.57459044],[ 0.22517808, -0.39425772,  0.6093044 , -0.4540528 ,0.36388075, -0.27318472],...[ 0.16320887, -0.40904495,  0.8606591 , -0.35270005,0.44078755,  0.05399841],[ 0.11930697, -0.18575972,  0.876694  , -0.06689822,0.05075729, -0.175943  ],[-0.03642005, -0.37495565,  0.71938276, -0.4610471 ,0.03974855, -0.32001895]],[[ 0.07017496, -0.7057908 ,  0.9293377 , -0.36878398,0.14343284, -0.36523613],[ 0.07206199, -0.52096766,  1.1488464 , -0.2801911 ,0.04227791, -0.38159102],[ 0.05785495, -0.25880778,  0.59430236, -0.5021971 ,0.29962078, -0.34087947],...,[ 0.13691866, -0.5528023 ,  0.9671679 , -0.6606184 ,-0.16109133, -0.2528019 ],[-0.0220079 , -0.35848263,  0.762049  , -0.3668743 ,0.5163839 , -0.4677353 ],[ 0.02993503, -0.55474484,  0.7659938 , -0.44340014,0.02594638, -0.47557232]]], shape=(4, 1000, 6), dtype=float32)


b_sib_disp


(chain, draw, product)


float32


-0.03735 0.003668 ... -0.001804


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[-3.73488143e-02,  3.66773619e-03, -1.70877539e-02,-4.04595211e-02, -6.25964925e-02,  1.42945023e-02],[-7.65243098e-02, -5.74957194e-05, -6.18927320e-03,-2.82030348e-02, -9.18584503e-03,  2.66909454e-04],[-4.22936268e-02,  3.53469662e-02, -9.84825287e-03,6.30521972e-04,  2.04069447e-03,  3.16241421e-02],...,[-4.93890271e-02,  1.96278598e-02, -8.51940364e-03,-4.43335511e-02, -8.07806943e-03,  1.42612820e-02],[-2.81274728e-02,  1.30232517e-03, -3.54565792e-02,-4.72815987e-03, -4.73042876e-02,  3.55226435e-02],[-6.55857846e-02,  1.99303613e-03, -6.54050289e-03,-4.65729572e-02,  5.07524889e-03,  1.09570250e-02]],[[-5.28331622e-02, -3.34573095e-03, -3.24216373e-02,-2.68947650e-02,  6.39934698e-03,  8.51596892e-03],[-4.85600457e-02, -1.52261544e-03, -1.11205885e-02,-1.47136329e-02, -1.74570195e-02,  5.91143384e-04],[-7.18945712e-02,  1.81283206e-02, -4.35492992e-02,-4.19730432e-02, -6.52476819e-03,  1.76690072e-02],...-2.52554566e-02, -2.39627976e-02, -7.51510845e-04],[-5.14623150e-02,  1.25691528e-02, -1.02650989e-02,1.14022708e-03,  1.19688897e-03,  1.94097050e-02],[-5.84240593e-02,  1.48572838e-02, -4.48407643e-02,-2.32615173e-02, -2.77671311e-02,  2.19745990e-02]],[[-4.29417379e-02,  1.53568191e-02, -3.57243456e-02,7.79336644e-03, -2.15622783e-02,  2.06761565e-02],[-3.84687148e-02, -4.63539083e-03, -2.18935031e-02,-4.22257408e-02, -8.46152380e-03, -2.46001855e-02],[-5.13929985e-02, -5.79202967e-03, -3.96857690e-03,-3.03344894e-02, -3.92452590e-02,  3.14914435e-02],...,[-3.05096004e-02,  1.27863733e-03, -3.05135380e-02,-1.13217216e-02, -4.33661491e-02,  4.78529967e-02],[-7.87387863e-02,  7.56129390e-03, -2.06848327e-02,-3.01340483e-02,  1.71305668e-02, -3.23034115e-02],[-7.11283684e-02, -2.43438780e-02, -1.18971337e-02,-2.79877577e-02, -1.81934163e-02, -1.80398964e-03]]],shape=(4, 1000, 6), dtype=float32)


b_sib_feat


(chain, draw, product)


float32


-0.07117 -0.006864 ... 0.04824


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[-0.07117211, -0.00686398,  0.01371778,  0.01464803,0.02579436,  0.02092483],[-0.05504441, -0.01607869,  0.04172726, -0.01022821,0.02935632,  0.04583144],[-0.09891065, -0.0270501 ,  0.02028327, -0.02139971,0.01192016,  0.02621184],...,[-0.05176069, -0.04077129,  0.04271805,  0.02613503,0.00765845,  0.00809709],[-0.09013702,  0.00589035,  0.04182616, -0.02070397,0.02122683,  0.00845687],[-0.0789685 , -0.03194879,  0.03249935,  0.02834415,0.02104983,  0.01661308]],[[-0.04925856, -0.00412997,  0.05140216, -0.00889898,0.01760018,  0.03041807],[-0.05861359, -0.01349701,  0.03997349,  0.0136345 ,0.05304636,  0.0197528 ],[-0.03397528, -0.0337327 ,  0.03852533,  0.01647417,0.01208601,  0.02062803],...[-0.07875009, -0.00835066,  0.02386351,  0.00341084,0.01413152,  0.03008768],[-0.09486795, -0.01685998,  0.03311084, -0.02258347,0.02061108,  0.00965796],[-0.05064462, -0.01258197,  0.0235302 ,  0.01559392,0.04246969,  0.04397152]],[[-0.09126062, -0.00488124,  0.03676983,  0.00538547,-0.00479939,  0.0321176 ],[-0.08290568, -0.02595359,  0.02519701,  0.02291361,0.03376579,  0.03520471],[-0.07499098, -0.0154623 ,  0.02843723,  0.00894644,0.0274682 ,  0.02042574],...,[-0.04964152, -0.01597984,  0.04019105,  0.01537459,0.04842249, -0.00323528],[-0.0854851 , -0.02676685,  0.02818193, -0.00336205,-0.0012458 ,  0.05485987],[-0.08716182, -0.01927463,  0.02964532,  0.01555437,0.00327003,  0.0482423 ]]], shape=(4, 1000, 6), dtype=float32)


beta_s


(chain, draw, fourier, product)


float32


0.009359 -0.04494 ... -0.02243


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 9.35867615e-03, -4.49374691e-02,  9.23872180e-03,8.21062028e-02, -2.80650239e-02,  1.96145643e-02],[-2.49550100e-02, -1.39321191e-02, -2.62786075e-03,1.85578521e-02,  2.12740172e-02, -2.72584409e-02],[ 2.78437305e-02, -4.47700620e-02, -3.34096663e-02,-5.02291881e-02, -3.77891846e-02, -1.09093543e-02],[ 6.91504404e-03, -4.42233263e-03,  7.17093656e-03,-1.84330475e-02,  2.23456044e-03, -1.62601061e-02]],[[ 3.74362455e-03, -1.11496560e-02,  5.44121815e-03,6.69705793e-02, -4.70413342e-02, -1.07889222e-02],[-1.30358906e-02, -2.09565740e-02,  6.62240805e-03,2.34396029e-02,  5.44934496e-02, -2.21936200e-02],[ 4.14542183e-02, -2.02326626e-02, -3.21949534e-02,-9.40863229e-03,  1.12010818e-02, -1.09647624e-02],[ 3.02566849e-02, -1.43334642e-02, -1.12188002e-03,8.86318833e-03,  1.06765183e-04, -2.30689775e-02]],[[ 8.21056869e-03, -1.75306257e-02,  1.24112917e-02,6.75027296e-02, -6.23366646e-02,  1.81540381e-02],...6.34551141e-03, -1.42529588e-02, -7.18004163e-03]],[[ 1.00837997e-03, -1.68896802e-02,  2.27773674e-02,6.72394559e-02, -3.46002541e-02,  5.09018870e-03],[-4.66468334e-02, -3.17932442e-02,  6.08773576e-03,3.55288424e-02,  4.98732738e-02, -2.12275647e-02],[ 4.59092148e-02, -4.82874662e-02, -2.09741984e-02,-3.56121138e-02, -1.15245895e-03, -5.17750066e-03],[ 2.87474282e-02, -2.20969412e-03, -2.85997218e-03,-8.69888905e-03,  1.05089406e-02, -3.25570256e-02]],[[-9.91299632e-04, -2.97691394e-02,  2.58840099e-02,7.56176040e-02, -2.43938919e-02, -2.41060788e-03],[-4.81521934e-02, -2.25062780e-02, -1.33840798e-03,3.41472775e-02,  5.07851578e-02, -1.77014675e-02],[ 3.90850939e-02, -4.37306948e-02, -2.74061859e-02,-4.44027483e-02,  1.01029770e-02, -4.52246284e-03],[ 3.16259898e-02, -1.66861881e-02, -8.68563890e-04,-9.66249779e-03,  1.28432531e-02, -2.24282909e-02]]]],shape=(4, 1000, 4, 6), dtype=float32)


centered_eps


(chain, draw)


float32


0.02888 0.0307 ... 0.4774 0.5124


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.02888367, 0.03070164, 0.06452599, ..., 0.13697244, 0.16247846,0.23154017],[0.56412685, 0.56503505, 0.7308739 , ..., 0.11416031, 0.13680224,0.25313652],[0.83001757, 0.86495715, 0.8904082 , ..., 0.36051005, 0.4290055 ,0.40498143],[0.69587195, 0.6416401 , 0.68883866, ..., 0.40148783, 0.47742948,0.51238364]], shape=(4, 1000), dtype=float32)


centered_gamma


(chain, draw)


float32


0.5093 0.5416 ... 0.6517 0.6237


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.5093397 , 0.5415604 , 0.5624977 , ..., 0.22575489, 0.19768107,0.24420728],[0.8413388 , 0.6805961 , 0.48968846, ..., 0.05149985, 0.46888974,0.3248403 ],[0.36537734, 0.22071093, 0.44387937, ..., 0.46858925, 0.5151969 ,0.5048461 ],[0.23249006, 0.11055353, 0.10855429, ..., 0.7663239 , 0.65169096,0.6237397 ]], shape=(4, 1000), dtype=float32)


conc


(chain, draw, product)


float32


17.97 18.73 23.66 ... 19.73 20.95


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[17.965424, 18.729488, 23.658913, 19.768488, 18.73075 ,20.895123],[17.672697, 20.085518, 20.35326 , 18.55363 , 20.189314,22.36894 ],[17.664677, 19.232147, 22.140574, 18.21326 , 19.078224,20.916271],...,[17.647346, 18.338957, 21.603035, 17.27484 , 18.24848 ,19.591578],[17.949137, 19.897236, 22.693865, 20.726875, 20.049026,21.823265],[17.499626, 18.49698 , 22.174574, 18.070429, 19.4189  ,20.704779]],[[18.759605, 19.430622, 22.3173  , 19.637033, 18.990732,20.12265 ],[17.46112 , 18.467789, 22.485527, 19.66679 , 18.21154 ,19.949396],[18.884777, 19.055332, 22.884167, 17.873913, 19.766361,21.091715],...[16.385273, 18.906168, 20.134777, 18.298283, 20.899801,21.278858],[16.60204 , 19.1571  , 24.34355 , 17.687021, 20.23447 ,20.787203],[15.937942, 18.718735, 20.619083, 19.198645, 18.029882,20.922804]],[[16.382893, 19.701757, 22.57385 , 19.923796, 21.190857,21.809153],[17.649055, 19.90485 , 21.626806, 20.883717, 19.59361 ,20.253416],[16.316292, 18.547174, 22.58792 , 19.077843, 21.499771,22.363049],...,[17.776222, 18.883175, 21.07484 , 19.72578 , 19.59072 ,20.461262],[16.280025, 20.860313, 23.430424, 17.834427, 20.505934,20.712397],[16.888786, 20.72535 , 23.44815 , 18.734182, 19.734154,20.950224]]], shape=(4, 1000, 6), dtype=float32)


cross_scale


(chain, draw)


float32


0.2603 0.2625 ... 0.3671 0.3279


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.2602671 , 0.26253003, 0.2826259 , ..., 0.23856963, 0.3137355 ,0.26728168],[0.25079316, 0.2423537 , 0.30215666, ..., 0.31687936, 0.23099373,0.28584552],[0.28009328, 0.33350062, 0.32045624, ..., 0.23982182, 0.23721443,0.30687505],[0.22180068, 0.23466906, 0.23373994, ..., 0.23847169, 0.36710754,0.32792547]], shape=(4, 1000), dtype=float32)


drift


(chain, draw, time, series)


float32


0.04233 0.03464 ... 0.02443 0.02234


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 4.23308201e-02,  3.46421897e-02,  5.99009395e-02, ...,2.50242203e-02,  3.12312320e-02, -2.58193701e-03],[ 1.83713518e-03,  1.73540432e-02,  2.89209057e-02, ...,-4.54287529e-02,  7.65932202e-02,  1.40768802e-02],[ 2.10909024e-02, -2.97092199e-02,  7.24490359e-02, ...,-4.54815775e-02,  5.12092151e-02,  7.04085305e-02],...,[-4.89786342e-02,  4.88663055e-02, -5.99504169e-03, ...,-3.14558633e-02, -3.03902067e-02,  3.11568379e-02],[ 6.86325133e-02,  5.23652397e-02, -1.84339043e-02, ...,2.49227472e-02, -7.33476952e-02, -2.17518806e-02],[ 1.04561850e-01,  2.86565367e-02,  2.59409193e-02, ...,1.84893683e-01,  2.51961388e-02, -1.03717536e-01]],[[ 2.10672040e-02, -5.88518940e-02, -7.36811310e-02, ...,-1.17346495e-01, -1.34410299e-02, -9.55890585e-03],[ 1.35685146e-01,  6.66183978e-03, -3.41917668e-03, ...,2.94162612e-02,  1.27585372e-02,  1.34806717e-02],[ 8.07183459e-02,  6.82417676e-02, -2.93523516e-03, ...,4.12944071e-02,  6.21420369e-02,  1.80448405e-02],...9.22648832e-02, -1.16275683e-01,  2.35891901e-02],[-7.90437460e-02,  2.03020852e-02,  1.15332948e-02, ...,-8.36295448e-03, -4.50189635e-02, -3.52576301e-02],[ 2.07996413e-01,  7.21600130e-02,  2.93028895e-02, ...,-6.75863326e-02,  4.33473103e-02, -6.18948136e-03]],[[ 8.20135325e-02, -5.40536977e-02,  2.79908646e-02, ...,-2.14538258e-02,  1.15056962e-01,  5.57962283e-02],[ 1.20426416e-01, -6.65546954e-02, -8.25612769e-02, ...,-5.76007105e-02,  1.09479792e-01, -2.74378434e-03],[-7.09819496e-02, -2.34499872e-02,  2.90649738e-02, ...,-8.61142352e-02, -1.04851983e-01, -1.05683627e-02],...,[-1.03536241e-01,  3.06676533e-02,  1.57680623e-02, ...,-9.04634967e-02, -1.52986243e-01,  1.33767352e-01],[-1.50585338e-01,  3.33971780e-04, -3.20737585e-02, ...,8.27348903e-02, -9.56630632e-02,  2.06493624e-02],[ 1.68546230e-01,  1.38569577e-02,  1.67066790e-02, ...,-1.30062997e-02,  2.44338289e-02,  2.23360639e-02]]]],shape=(4, 1000, 143, 108), dtype=float32)


drift_decentered


(chain, draw, time, series)


float32


0.5753 1.093 ... 0.2955 0.3894


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 5.75311661e-01,  1.09308410e+00,  1.85532391e+00, ...,3.98503125e-01,  3.23553920e-01, -6.03527986e-02],[ 2.49682199e-02,  5.47581673e-01,  8.95773053e-01, ...,-7.23439157e-01,  7.93501735e-01,  3.29047173e-01],[ 2.86643207e-01, -9.37431395e-01,  2.24397874e+00, ...,-7.24280357e-01,  5.30524790e-01,  1.64579988e+00],...,[-6.65661037e-01,  1.54190540e+00, -1.85685635e-01, ...,-5.00925124e-01, -3.14840943e-01,  7.28291333e-01],[ 9.32773888e-01,  1.65230918e+00, -5.70957065e-01, ...,3.96887213e-01, -7.59878278e-01, -5.08450389e-01],[ 1.42108393e+00,  9.04215455e-01,  8.03473353e-01, ...,2.94437599e+00,  2.61030674e-01, -2.42439818e+00]],[[ 2.63123870e-01, -1.60926020e+00, -2.79528332e+00, ...,-1.65462017e+00, -2.04199240e-01, -1.90848127e-01],[ 1.69467199e+00,  1.82162926e-01, -1.29715264e-01, ...,4.14777935e-01,  1.93830654e-01,  2.69148052e-01],[ 1.00815105e+00,  1.86601913e+00, -1.11355700e-01, ...,5.82263291e-01,  9.44076180e-01,  3.60273868e-01],...1.37263012e+00, -1.89099979e+00,  3.75603110e-01],[-1.17263556e+00,  4.16005641e-01,  2.77862519e-01, ...,-1.24416158e-01, -7.32146680e-01, -5.61395943e-01],[ 3.08568358e+00,  1.47861516e+00,  7.05971241e-01, ...,-1.00548577e+00,  7.04960406e-01, -9.85531285e-02]],[[ 1.14682293e+00, -1.41840613e+00,  7.47846305e-01, ...,-2.98500478e-01,  1.39164579e+00,  9.72843230e-01],[ 1.68396330e+00, -1.74644089e+00, -2.20583200e+00, ...,-8.01434696e-01,  1.32418835e+00, -4.78396490e-02],[-9.92564619e-01, -6.15343750e-01,  7.76543856e-01, ...,-1.19816113e+00, -1.26821375e+00, -1.84266210e-01],...,[-1.44778228e+00,  8.04740310e-01,  4.21283424e-01, ...,-1.25867522e+00, -1.85041094e+00,  2.33231997e+00],[-2.10568571e+00,  8.76364857e-03, -8.56931090e-01, ...,1.15114224e+00, -1.15707123e+00,  3.60034943e-01],[ 2.35683894e+00,  3.63616079e-01,  4.46360916e-01, ...,-1.80964783e-01,  2.95533925e-01,  3.89443666e-01]]]],shape=(4, 1000, 143, 108), dtype=float32)


drift_scale


(chain, draw, series)


float32


0.07358 0.03169 ... 0.08268 0.05735


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.07357894, 0.03169215, 0.03228597, ..., 0.06279554,0.09652559, 0.04278073],[0.08006573, 0.03657078, 0.02635909, ..., 0.0709205 ,0.06582312, 0.05008645],[0.07585378, 0.02272093, 0.03350165, ..., 0.08838325,0.06385601, 0.04518336],...,[0.14196417, 0.02505034, 0.02138299, ..., 0.08145213,0.03085872, 0.0405523 ],[0.11966176, 0.03733547, 0.01749855, ..., 0.06874073,0.08660883, 0.02979477],[0.11101419, 0.03405536, 0.03349711, ..., 0.07476811,0.07705759, 0.0468098 ]],[[0.10520072, 0.02856291, 0.04535935, ..., 0.04044906,0.07974016, 0.06604155],[0.10707523, 0.03314731, 0.03040374, ..., 0.03820201,0.0848567 , 0.05464223],[0.14795642, 0.02986687, 0.03307297, ..., 0.07521696,0.08316875, 0.02175614],...[0.05154614, 0.01851286, 0.02208153, ..., 0.06652679,0.04783574, 0.06449983],[0.05720695, 0.02785661, 0.0355291 , ..., 0.05210042,0.06281566, 0.0493722 ],[0.030666  , 0.03166175, 0.01177955, ..., 0.05208908,0.04762622, 0.05078421]],[[0.07354021, 0.01232524, 0.02732645, ..., 0.07312682,0.04312715, 0.06115137],[0.09873411, 0.03803125, 0.03334646, ..., 0.07504026,0.04223846, 0.06708872],[0.07632232, 0.00883653, 0.01311026, ..., 0.05920865,0.04907483, 0.02651352],...,[0.09803625, 0.04350502, 0.01711917, ..., 0.05707412,0.07558881, 0.0385587 ],[0.06740692, 0.04880243, 0.0415072 , ..., 0.06721759,0.061489  , 0.0628035 ],[0.07151368, 0.03810876, 0.03742863, ..., 0.071872  ,0.0826769 , 0.05735377]]], shape=(4, 1000, 108), dtype=float32)


eps


(chain, draw, series)


float32


-1.405 -0.3796 ... -1.978 -1.272


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[-1.4049416 , -0.37964645, -2.2913904 , ..., -2.2148032 ,-2.421891  , -0.8466574 ],[-1.1821475 , -0.29503325, -1.9637343 , ..., -1.9769824 ,-1.5741487 , -1.1008643 ],[-1.3739866 , -0.38304186, -2.3269444 , ..., -2.1848762 ,-1.813285  , -1.0250578 ],...,[-1.3640046 , -0.29936078, -2.3953483 , ..., -1.9630941 ,-1.5251436 , -1.2431334 ],[-1.516159  , -0.31841704, -2.0407517 , ..., -2.2985091 ,-2.2483804 , -1.6248462 ],[-1.6827458 , -0.26115894, -2.422551  , ..., -2.0040905 ,-1.3323169 , -0.92978   ]],[[-1.1611464 , -0.3791175 , -2.1494236 , ..., -1.5561024 ,-1.5897336 , -0.7731424 ],[-1.7381188 , -0.20743711, -2.3207428 , ..., -1.8893149 ,-2.0997324 , -0.989187  ],[-1.3759688 , -0.33766627, -2.380651  , ..., -2.569657  ,-1.49041   , -1.0751686 ],...[-1.2594848 , -0.26815635, -2.1697862 , ..., -1.8507779 ,-1.5716633 , -0.77042574],[-1.8333148 , -0.22683415, -2.1824946 , ..., -2.6616616 ,-1.4125621 , -0.7143606 ],[-1.738945  , -0.31292662, -2.5754805 , ..., -2.1770153 ,-2.0530407 , -1.2697638 ]],[[-1.373873  , -0.3634893 , -2.2380185 , ..., -1.5530525 ,-1.4853008 , -1.2143844 ],[-1.7442682 , -0.2953808 , -2.412394  , ..., -2.2254946 ,-1.9166561 , -0.5803008 ],[-1.4224894 , -0.26779488, -2.1963968 , ..., -1.643982  ,-1.6104468 , -1.1068248 ],...,[-1.6537578 , -0.15107751, -2.578432  , ..., -2.352786  ,-1.664039  , -0.91078913],[-1.549416  , -0.41854346, -2.0396683 , ..., -2.2260377 ,-1.9496051 , -1.0717552 ],[-1.6714766 , -0.46180838, -2.0257175 , ..., -2.3700757 ,-1.9782853 , -1.271801  ]]], shape=(4, 1000, 108), dtype=float32)


eps_decentered


(chain, draw, series)


float32


-1.391 -0.2642 ... -1.065 -0.8186


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[-1.3912467 , -0.2641939 , -0.8764941 , ..., -0.86829215,-1.4439851 ,  0.3235255 ],[-0.4021808 ,  0.21925141, -0.1944404 , ..., -0.07386154,1.3738246 , -0.13103776],[-0.8647006 , -0.16221118, -1.154807  , ..., -1.1805732 ,0.79035014, -0.16890834],...,[-1.143206  ,  0.251307  , -1.502132  , ..., -0.5031649 ,0.6723643 , -0.6429732 ],[-1.2836915 ,  0.33619204, -0.7034221 , ..., -1.2905364 ,-0.75893515, -0.9295405 ],[-2.0995293 ,  0.30320767, -1.5440134 , ..., -0.7256905 ,0.5332456 ,  0.0944624 ]],[[-0.8784882 , -0.30087975, -1.376104  , ..., -0.6097698 ,-0.18079865, -0.19911209],[-1.6541027 , -0.03630471, -1.699288  , ..., -1.0722656 ,-1.3942449 , -0.41440988],[-1.2345061 , -0.25517255, -1.9456989 , ..., -2.2998843 ,-0.79647124, -0.8290977 ],...[-1.1896205 ,  0.54529184, -1.5673927 , ..., -0.4495222 ,-0.23975782,  0.12695032],[-1.7606874 ,  0.36383453, -1.526013  , ..., -1.896032  ,-0.22358677,  0.22539881],[-1.8367835 ,  0.01239799, -1.8288301 , ..., -1.356319  ,-0.2776277 , -1.0029291 ]],[[-1.3038291 , -0.20829819, -1.7220752 , ..., -0.8193427 ,-0.7035358 , -1.1052376 ],[-1.6713371 , -0.01815467, -2.1183612 , ..., -1.8635818 ,-1.3063293 ,  0.01405503],[-1.2787815 , -0.0711171 , -1.6887211 , ..., -0.83033514,-0.7246409 , -0.83500046],...,[-1.4559921 ,  0.15626673, -1.7690024 , ..., -1.6181985 ,-0.20754698, -0.10016299],[-1.4537485 , -0.33439696, -1.6001084 , ..., -1.2950127 ,-0.9788497 , -0.5223194 ],[-1.9268507 , -0.32032415, -1.2167836 , ..., -1.3269489 ,-1.0651714 , -0.81859684]]], shape=(4, 1000, 108), dtype=float32)


eps_prod


(chain, draw, product)


float32


-0.9976 -0.3033 ... -1.944 -1.16


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[-0.99759495, -0.30327553, -2.0470095 , -1.9722288 ,-2.0074978 , -0.9516044 ],[-1.0520544 , -0.37620988, -1.9160364 , -1.9723015 ,-2.079803  , -1.0662903 ],[-1.1405339 , -0.34168   , -2.0247185 , -1.8721415 ,-2.0862393 , -0.99414337],...,[-0.9944126 , -0.41232473, -1.9416411 , -1.8725582 ,-1.86571   , -1.0604588 ],[-1.0760828 , -0.48298845, -1.8830036 , -1.9094554 ,-2.0813935 , -1.3425038 ],[-0.8637568 , -0.4353481 , -1.9516803 , -1.8787626 ,-1.7354783 , -1.0781372 ]],[[-0.96762943, -0.30376524, -2.0077798 , -1.7882569 ,-2.2098181 , -0.97842336],[-1.1869075 , -0.27379897, -2.002497  , -1.8854297 ,-1.9292663 , -1.1077256 ],[-1.0059323 , -0.32411432, -2.0479715 , -1.8876393 ,-1.9639063 , -1.0051783 ],...[-0.8586824 , -0.6179576 , -1.7419577 , -1.9694017 ,-1.749757  , -0.99094146],[-1.1755106 , -0.53821725, -1.784301  , -2.1524234 ,-1.6707789 , -1.0734935 ],[-1.0752106 , -0.3944632 , -2.1149523 , -1.8988329 ,-2.3776083 , -0.98335886]],[[-0.89083177, -0.4254946 , -2.009267  , -1.9160848 ,-1.9436946 , -0.853021  ],[-1.1316501 , -0.4855239 , -1.7794411 , -1.7421691 ,-1.831749  , -1.0090203 ],[-1.0524589 , -0.40811232, -1.9788245 , -2.0179584 ,-2.0866668 , -1.0169812 ],...,[-1.1793014 , -0.2810227 , -2.1388607 , -1.9492829 ,-1.9399234 , -1.0698388 ],[-1.0678056 , -0.3277729 , -1.6172136 , -2.069391  ,-1.9171942 , -1.0645838 ],[-0.95433426, -0.40573135, -1.9059011 , -2.2946637 ,-1.9437616 , -1.159952  ]]], shape=(4, 1000, 6), dtype=float32)


eps_scale


(chain, draw)


float32


0.2884 0.3403 ... 0.2759 0.2401


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.28843865, 0.34026563, 0.27127123, ..., 0.31304404, 0.33173704,0.3346136 ],[0.28861484, 0.26420143, 0.32849717, ..., 0.23867045, 0.29830432,0.28060374],[0.29185832, 0.32613087, 0.31430343, ..., 0.2923175 , 0.32198027,0.2848096 ],[0.318728  , 0.29814243, 0.27365413, ..., 0.29633573, 0.2758939 ,0.2401205 ]], shape=(4, 1000), dtype=float32)


gamma


(chain, draw, competitor, product)


float32


0.0 0.1502 0.01105 ... 0.3973 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 0.00000000e+00,  1.50192767e-01,  1.10477060e-02,-6.70225993e-02,  6.60089731e-01,  4.55766842e-02],[-2.60093510e-01,  0.00000000e+00,  1.84143797e-01,3.38866785e-02, -6.55938387e-02,  3.22967619e-02],[-2.60577619e-01,  2.08236992e-01,  0.00000000e+00,6.79840043e-04, -3.26817110e-02,  1.76854908e-01],[-4.88361977e-02, -2.76297599e-01,  1.90307811e-01,0.00000000e+00, -1.11485876e-01,  6.28313184e-01],[-3.57920788e-02, -8.44111681e-01, -1.67364135e-01,2.85924673e-01,  0.00000000e+00,  2.45847538e-01],[ 2.09031701e-01,  1.54354706e-01, -1.33196816e-01,4.09587435e-02,  4.65546668e-01,  0.00000000e+00]],[[ 0.00000000e+00,  8.42714831e-02,  6.33845851e-02,-7.70782605e-02,  6.67706490e-01,  5.19473664e-02],[-2.94068068e-01,  0.00000000e+00,  1.78094223e-01,7.64272884e-02, -1.23333678e-01, -5.62916137e-02],[-3.28668952e-01,  2.07576916e-01,  0.00000000e+00,9.71148349e-03,  1.09009095e-01,  1.61044315e-01],[-1.11741237e-01, -1.97758898e-01,  2.12050736e-01,...[-2.13084489e-01, -1.83955625e-01,  1.20433182e-01,0.00000000e+00, -4.12332565e-02,  5.64520180e-01],[-7.00712129e-02, -7.85785913e-01, -1.61436006e-01,-1.05401658e-01,  0.00000000e+00,  2.94438988e-01],[ 9.50777680e-02,  1.85212418e-01, -7.35321194e-02,8.80394801e-02,  3.98666382e-01,  0.00000000e+00]],[[ 0.00000000e+00,  5.63496277e-02,  5.85710490e-03,-4.14873697e-02,  6.89638138e-01,  6.50291443e-02],[-3.13166916e-01,  0.00000000e+00,  1.68508276e-01,6.64421404e-03,  3.45301479e-02,  2.16071624e-02],[-3.07494551e-01,  1.69959769e-01,  0.00000000e+00,7.06636906e-02, -1.58338416e-02,  1.09481923e-01],[-1.41212925e-01, -2.44214103e-01,  2.55869359e-01,0.00000000e+00, -6.40428066e-02,  5.56301534e-01],[-9.20274332e-02, -7.29154289e-01, -2.66108394e-01,2.32621834e-01,  0.00000000e+00,  4.20917332e-01],[ 2.04190761e-02,  2.29550794e-01, -1.57738719e-02,1.85866997e-01,  3.97300631e-01,  0.00000000e+00]]]],shape=(4, 1000, 6, 6), dtype=float32)


gamma_offdiag


(chain, draw, pair)


float32


0.1502 0.01105 ... 0.1859 0.3973


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 0.15019277,  0.01104771, -0.0670226 , ..., -0.13319682,0.04095874,  0.46554667],[ 0.08427148,  0.06338459, -0.07707826, ..., -0.07215574,0.11085454,  0.32143745],[ 0.08669145,  0.01409915,  0.001966  , ..., -0.01057153,-0.04086664,  0.30878702],...,[ 0.15812515,  0.03241547,  0.05155285, ..., -0.02104273,0.11327932,  0.33370712],[ 0.17070045,  0.04429929,  0.03702151, ..., -0.02164584,0.1219099 ,  0.4784594 ],[ 0.04247364, -0.02655079, -0.07058871, ...,  0.01713591,0.13689454,  0.34875363]],[[ 0.15859975,  0.0013834 , -0.0064155 , ...,  0.03388403,0.04881088,  0.33133772],[ 0.06212318, -0.0124311 , -0.08765399, ..., -0.02024514,0.05856027,  0.27450138],[ 0.12649612,  0.08685888,  0.00061303, ..., -0.08188414,0.01839575,  0.33119074],...[ 0.16599011,  0.1016091 ,  0.05285667, ..., -0.10713576,0.07934871,  0.37664595],[ 0.1007556 ,  0.0479848 ,  0.01323467, ...,  0.0664217 ,0.08736105,  0.32491004],[ 0.09925164,  0.03548478, -0.1230398 , ..., -0.13887385,0.03883326,  0.49609786]],[[ 0.09376027, -0.01912402, -0.04131041, ..., -0.13760942,0.12217288,  0.43775567],[ 0.08580323, -0.00376635, -0.0468081 , ..., -0.00883539,0.04927911,  0.2939978 ],[ 0.02023622,  0.03665521,  0.02539015, ...,  0.06021538,0.06618255,  0.46590966],...,[ 0.09122248,  0.03385347, -0.03195364, ..., -0.01956712,0.00157802,  0.3715568 ],[ 0.13745369,  0.05838204, -0.04117756, ..., -0.07353212,0.08803948,  0.39866638],[ 0.05634963,  0.0058571 , -0.04148737, ..., -0.01577387,0.185867  ,  0.39730063]]], shape=(4, 1000, 30), dtype=float32)


gamma_offdiag_decentered


(chain, draw, pair)


float32


0.2907 0.02138 ... 0.2827 0.6044


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 2.90723115e-01,  2.13846751e-02, -1.29733413e-01, ...,-2.57824630e-01,  7.92824700e-02,  9.01143134e-01],[ 1.55579373e-01,  1.17018625e-01, -1.42299458e-01, ...,-1.33211657e-01,  2.04656169e-01,  5.93427718e-01],[ 1.50685772e-01,  2.45069284e-02,  3.41727119e-03, ...,-1.83752794e-02, -7.10337833e-02,  5.36728978e-01],...,[ 4.79600221e-01,  9.83174816e-02,  1.56361952e-01, ...,-6.38234913e-02,  3.43580931e-01,  1.01214778e+00],[ 4.32664394e-01,  1.12282805e-01,  9.38362554e-02, ...,-5.48644401e-02,  3.08997869e-01,  1.21272290e+00],[ 1.15136221e-01, -7.19730556e-02, -1.91349700e-01, ...,4.64515015e-02,  3.71089488e-01,  9.45390582e-01]],[[ 1.97518513e-01,  1.72287831e-03, -7.98980147e-03, ...,4.21988219e-02,  6.07885681e-02,  4.12644625e-01],[ 9.76928771e-02, -1.95487347e-02, -1.37841791e-01, ...,-3.18368487e-02,  9.20899510e-02,  4.31671858e-01],[ 2.32981130e-01,  1.59977064e-01,  1.12909055e-03, ...,-1.50814578e-01,  3.38813663e-02,  6.09988570e-01],...-2.28806451e-01,  1.69462517e-01,  8.04390848e-01],[ 2.02396378e-01,  9.63911712e-02,  2.65856143e-02, ...,1.33426934e-01,  1.75489575e-01,  6.52674496e-01],[ 1.78143784e-01,  6.36905655e-02, -2.20840424e-01, ...,-2.49260485e-01,  6.97006583e-02,  8.90431106e-01]],[[ 2.97850817e-01, -6.07518107e-02, -1.31231919e-01, ...,-4.37147617e-01,  3.88109952e-01,  1.39063048e+00],[ 3.11493874e-01, -1.36730894e-02, -1.69928774e-01, ...,-3.20753828e-02,  1.78899348e-01,  1.06730843e+00],[ 7.39383325e-02,  1.33929446e-01,  9.27695930e-02, ...,2.20012695e-01,  2.41815358e-01,  1.70232332e+00],...,[ 1.27520502e-01,  4.73239869e-02, -4.46682014e-02, ...,-2.73530129e-02,  2.20593065e-03,  5.19401670e-01],[ 1.94869041e-01,  8.27686191e-02, -5.83777130e-02, ...,-1.04246989e-01,  1.24814168e-01,  5.65192044e-01],[ 8.57206807e-02,  8.90999660e-03, -6.31117821e-02, ...,-2.39956696e-02,  2.82746226e-01,  6.04385197e-01]]],shape=(4, 1000, 30), dtype=float32)


level0


(chain, draw, series)


float32


4.483 4.388 4.0 ... 2.081 2.286


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[4.4832015, 4.388071 , 4.0003195, ..., 3.1137707, 2.254405 ,2.685457 ],[4.1824675, 4.6319656, 4.067923 , ..., 3.1586072, 2.4224923,2.4800987],[4.1043644, 4.388304 , 4.025707 , ..., 3.0202045, 2.204742 ,2.7346554],...,[4.1731715, 4.4571595, 3.9427798, ..., 3.2558122, 2.5508509,2.4124944],[4.111037 , 4.667931 , 3.975012 , ..., 3.0569594, 2.6328366,2.6078188],[4.4598274, 4.583964 , 4.0915074, ..., 3.1219635, 2.2498338,2.6631649]],[[4.1125226, 4.544156 , 4.060948 , ..., 2.8797314, 2.5494509,2.9284194],[4.2941365, 4.429403 , 4.0524445, ..., 2.9568803, 2.7068913,2.7861745],[4.535729 , 4.596444 , 4.077172 , ..., 3.104344 , 2.6400008,2.679928 ],...[4.4741716, 4.4841413, 4.064919 , ..., 3.073399 , 2.5957172,2.5959775],[4.5702558, 4.7074428, 4.0437026, ..., 2.9387333, 2.640144 ,2.7655482],[4.4646482, 4.437217 , 3.8995621, ..., 3.1547484, 2.7149227,2.4918952]],[[4.7908883, 4.5234084, 4.0884986, ..., 3.2425003, 2.6540866,2.549779 ],[4.112007 , 4.7349505, 3.9415803, ..., 3.309825 , 2.6100154,2.512804 ],[4.6326766, 4.4950604, 4.007782 , ..., 3.0805697, 2.7091024,2.6924038],...,[4.6375303, 4.446762 , 4.0082536, ..., 2.676561 , 2.5524645,2.7273936],[4.2533197, 4.637438 , 3.9797206, ..., 3.4262552, 2.392842 ,2.504963 ],[4.2899346, 4.629561 , 3.9275076, ..., 3.1224067, 2.0809388,2.2857585]]], shape=(4, 1000, 108), dtype=float32)


Attributes: (5)


created_at :  
2026-09-07T11:31:36.864779+00:00

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
- time: 143
- obs_dim: 108


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


datetime64\[s\]


2009-01-14 ... 2011-10-05


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['2009-01-14T00:00:00', '2009-01-21T00:00:00', '2009-01-28T00:00:00','2009-02-04T00:00:00', '2009-02-11T00:00:00', '2009-02-18T00:00:00','2009-02-25T00:00:00', '2009-03-04T00:00:00', '2009-03-11T00:00:00','2009-03-18T00:00:00', '2009-03-25T00:00:00', '2009-04-01T00:00:00','2009-04-08T00:00:00', '2009-04-15T00:00:00', '2009-04-22T00:00:00','2009-04-29T00:00:00', '2009-05-06T00:00:00', '2009-05-13T00:00:00','2009-05-20T00:00:00', '2009-05-27T00:00:00', '2009-06-03T00:00:00','2009-06-10T00:00:00', '2009-06-17T00:00:00', '2009-06-24T00:00:00','2009-07-01T00:00:00', '2009-07-08T00:00:00', '2009-07-15T00:00:00','2009-07-22T00:00:00', '2009-07-29T00:00:00', '2009-08-05T00:00:00','2009-08-12T00:00:00', '2009-08-19T00:00:00', '2009-08-26T00:00:00','2009-09-02T00:00:00', '2009-09-09T00:00:00', '2009-09-16T00:00:00','2009-09-23T00:00:00', '2009-09-30T00:00:00', '2009-10-07T00:00:00','2009-10-14T00:00:00', '2009-10-21T00:00:00', '2009-10-28T00:00:00','2009-11-04T00:00:00', '2009-11-11T00:00:00', '2009-11-18T00:00:00','2009-11-25T00:00:00', '2009-12-02T00:00:00', '2009-12-09T00:00:00','2009-12-16T00:00:00', '2009-12-23T00:00:00', '2009-12-30T00:00:00','2010-01-06T00:00:00', '2010-01-13T00:00:00', '2010-01-20T00:00:00','2010-01-27T00:00:00', '2010-02-03T00:00:00', '2010-02-10T00:00:00','2010-02-17T00:00:00', '2010-02-24T00:00:00', '2010-03-03T00:00:00','2010-03-10T00:00:00', '2010-03-17T00:00:00', '2010-03-24T00:00:00','2010-03-31T00:00:00', '2010-04-07T00:00:00', '2010-04-14T00:00:00','2010-04-21T00:00:00', '2010-04-28T00:00:00', '2010-05-05T00:00:00','2010-05-12T00:00:00', '2010-05-19T00:00:00', '2010-05-26T00:00:00','2010-06-02T00:00:00', '2010-06-09T00:00:00', '2010-06-16T00:00:00','2010-06-23T00:00:00', '2010-06-30T00:00:00', '2010-07-07T00:00:00','2010-07-14T00:00:00', '2010-07-21T00:00:00', '2010-07-28T00:00:00','2010-08-04T00:00:00', '2010-08-11T00:00:00', '2010-08-18T00:00:00','2010-08-25T00:00:00', '2010-09-01T00:00:00', '2010-09-08T00:00:00','2010-09-15T00:00:00', '2010-09-22T00:00:00', '2010-09-29T00:00:00','2010-10-06T00:00:00', '2010-10-13T00:00:00', '2010-10-20T00:00:00','2010-10-27T00:00:00', '2010-11-03T00:00:00', '2010-11-10T00:00:00','2010-11-17T00:00:00', '2010-11-24T00:00:00', '2010-12-01T00:00:00','2010-12-08T00:00:00', '2010-12-15T00:00:00', '2010-12-22T00:00:00','2010-12-29T00:00:00', '2011-01-05T00:00:00', '2011-01-12T00:00:00','2011-01-19T00:00:00', '2011-01-26T00:00:00', '2011-02-02T00:00:00','2011-02-09T00:00:00', '2011-02-16T00:00:00', '2011-02-23T00:00:00','2011-03-02T00:00:00', '2011-03-09T00:00:00', '2011-03-16T00:00:00','2011-03-23T00:00:00', '2011-03-30T00:00:00', '2011-04-06T00:00:00','2011-04-13T00:00:00', '2011-04-20T00:00:00', '2011-04-27T00:00:00','2011-05-04T00:00:00', '2011-05-11T00:00:00', '2011-05-18T00:00:00','2011-05-25T00:00:00', '2011-06-01T00:00:00', '2011-06-08T00:00:00','2011-06-15T00:00:00', '2011-06-22T00:00:00', '2011-06-29T00:00:00','2011-07-06T00:00:00', '2011-07-13T00:00:00', '2011-07-20T00:00:00','2011-07-27T00:00:00', '2011-08-03T00:00:00', '2011-08-10T00:00:00','2011-08-17T00:00:00', '2011-08-24T00:00:00', '2011-08-31T00:00:00','2011-09-07T00:00:00', '2011-09-14T00:00:00', '2011-09-21T00:00:00','2011-09-28T00:00:00', '2011-10-05T00:00:00'], dtype='datetime64[s]')


obs_dim


(obs_dim)


\<U24


'25027::HNC' ... '6431::PL Frost...


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['25027::HNC', '25027::Cheerios 12oz', '25027::Cheerios 18oz','25027::Mini Wheats', '25027::PL Honey Nut Oats','25027::PL Frosted Wheat', '21237::HNC', '21237::Cheerios 12oz','21237::Cheerios 18oz', '21237::Mini Wheats','21237::PL Honey Nut Oats', '21237::PL Frosted Wheat', '25229::HNC','25229::Cheerios 12oz', '25229::Cheerios 18oz', '25229::Mini Wheats','25229::PL Honey Nut Oats', '25229::PL Frosted Wheat', '19265::HNC','19265::Cheerios 12oz', '19265::Cheerios 18oz', '19265::Mini Wheats','19265::PL Honey Nut Oats', '19265::PL Frosted Wheat', '9825::HNC','9825::Cheerios 12oz', '9825::Cheerios 18oz', '9825::Mini Wheats','9825::PL Honey Nut Oats', '9825::PL Frosted Wheat', '613::HNC','613::Cheerios 12oz', '613::Cheerios 18oz', '613::Mini Wheats','613::PL Honey Nut Oats', '613::PL Frosted Wheat', '2277::HNC','2277::Cheerios 12oz', '2277::Cheerios 18oz', '2277::Mini Wheats','2277::PL Honey Nut Oats', '2277::PL Frosted Wheat', '24991::HNC','24991::Cheerios 12oz', '24991::Cheerios 18oz', '24991::Mini Wheats','24991::PL Honey Nut Oats', '24991::PL Frosted Wheat', '6179::HNC','6179::Cheerios 12oz', '6179::Cheerios 18oz', '6179::Mini Wheats','6179::PL Honey Nut Oats', '6179::PL Frosted Wheat', '2513::HNC','2513::Cheerios 12oz', '2513::Cheerios 18oz', '2513::Mini Wheats','2513::PL Honey Nut Oats', '2513::PL Frosted Wheat', '2281::HNC','2281::Cheerios 12oz', '2281::Cheerios 18oz', '2281::Mini Wheats','2281::PL Honey Nut Oats', '2281::PL Frosted Wheat', '11993::HNC','11993::Cheerios 12oz', '11993::Cheerios 18oz', '11993::Mini Wheats','11993::PL Honey Nut Oats', '11993::PL Frosted Wheat', '25021::HNC','25021::Cheerios 12oz', '25021::Cheerios 18oz', '25021::Mini Wheats','25021::PL Honey Nut Oats', '25021::PL Frosted Wheat', '4259::HNC','4259::Cheerios 12oz', '4259::Cheerios 18oz', '4259::Mini Wheats','4259::PL Honey Nut Oats', '4259::PL Frosted Wheat', '21479::HNC','21479::Cheerios 12oz', '21479::Cheerios 18oz', '21479::Mini Wheats','21479::PL Honey Nut Oats', '21479::PL Frosted Wheat', '23349::HNC','23349::Cheerios 12oz', '23349::Cheerios 18oz', '23349::Mini Wheats','23349::PL Honey Nut Oats', '23349::PL Frosted Wheat', '19523::HNC','19523::Cheerios 12oz', '19523::Cheerios 18oz', '19523::Mini Wheats','19523::PL Honey Nut Oats', '19523::PL Frosted Wheat', '6431::HNC','6431::Cheerios 12oz', '6431::Cheerios 18oz', '6431::Mini Wheats','6431::PL Honey Nut Oats', '6431::PL Frosted Wheat'], dtype='<U24')


Data variables: (1)


obs


(chain, draw, time, obs_dim)


int32


71 214 63 44 60 ... 23 14 19 7 26


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 71, 214,  63, ...,  24,  12,  23],[112,  88,  54, ...,  26,  10,  17],[ 74,  77,  65, ...,  21,  12,  20],...,[102,  97,  62, ...,  25,  16,  30],[141,  89,  37, ...,   9,   9,  31],[157, 106,  62, ...,  32,  15,  18]],[[ 71, 184,  31, ...,  12,  11,  16],[104, 120,  87, ...,  21,   9,  12],[ 78, 124,  55, ...,  18,  13,  13],...,[131, 147,  39, ...,  27,  13,  11],[193, 103,  68, ...,  28,   8,  35],[ 78,  88,  59, ...,  10,  10,  25]],[[ 73, 152,  52, ...,  17,   6,  12],[ 48,  59,  43, ...,  17,  12,  12],[ 70, 100,  51, ...,  13,  10,  14],...,...[153, 101,  52, ...,  21,  14,  22],[251, 101,  55, ...,  22,   7,  15],[217, 149,  45, ...,  14,  10,  11]],[[ 73, 188,  49, ...,  31,  10,  21],[ 75,  38,  54, ...,  22,  19,   7],[ 69,  99,  62, ...,  22,  12,  12],...,[191, 122,  46, ...,  19,  13,  29],[196,  85,  69, ...,  21,   9,  16],[158, 117,  71, ...,  13,   9,  20]],[[ 74, 199,  67, ...,  24,  10,   7],[ 93,  75,  47, ...,  29,  13,  13],[ 50,  82,  41, ...,   8,  13,  23],...,[ 87,  77,  76, ...,  17,   7,  24],[121, 113,  68, ...,  13,   8,  20],[ 88, 100,  35, ...,  19,   7,  26]]]],shape=(4, 1000, 143, 108), dtype=int32)


Attributes: (5)


created_at :  
2026-09-07T11:31:59.219193+00:00

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


- time: 143
- obs_dim: 108


Coordinates: (2)


time


(time)


datetime64\[s\]


2009-01-14 ... 2011-10-05


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['2009-01-14T00:00:00', '2009-01-21T00:00:00', '2009-01-28T00:00:00','2009-02-04T00:00:00', '2009-02-11T00:00:00', '2009-02-18T00:00:00','2009-02-25T00:00:00', '2009-03-04T00:00:00', '2009-03-11T00:00:00','2009-03-18T00:00:00', '2009-03-25T00:00:00', '2009-04-01T00:00:00','2009-04-08T00:00:00', '2009-04-15T00:00:00', '2009-04-22T00:00:00','2009-04-29T00:00:00', '2009-05-06T00:00:00', '2009-05-13T00:00:00','2009-05-20T00:00:00', '2009-05-27T00:00:00', '2009-06-03T00:00:00','2009-06-10T00:00:00', '2009-06-17T00:00:00', '2009-06-24T00:00:00','2009-07-01T00:00:00', '2009-07-08T00:00:00', '2009-07-15T00:00:00','2009-07-22T00:00:00', '2009-07-29T00:00:00', '2009-08-05T00:00:00','2009-08-12T00:00:00', '2009-08-19T00:00:00', '2009-08-26T00:00:00','2009-09-02T00:00:00', '2009-09-09T00:00:00', '2009-09-16T00:00:00','2009-09-23T00:00:00', '2009-09-30T00:00:00', '2009-10-07T00:00:00','2009-10-14T00:00:00', '2009-10-21T00:00:00', '2009-10-28T00:00:00','2009-11-04T00:00:00', '2009-11-11T00:00:00', '2009-11-18T00:00:00','2009-11-25T00:00:00', '2009-12-02T00:00:00', '2009-12-09T00:00:00','2009-12-16T00:00:00', '2009-12-23T00:00:00', '2009-12-30T00:00:00','2010-01-06T00:00:00', '2010-01-13T00:00:00', '2010-01-20T00:00:00','2010-01-27T00:00:00', '2010-02-03T00:00:00', '2010-02-10T00:00:00','2010-02-17T00:00:00', '2010-02-24T00:00:00', '2010-03-03T00:00:00','2010-03-10T00:00:00', '2010-03-17T00:00:00', '2010-03-24T00:00:00','2010-03-31T00:00:00', '2010-04-07T00:00:00', '2010-04-14T00:00:00','2010-04-21T00:00:00', '2010-04-28T00:00:00', '2010-05-05T00:00:00','2010-05-12T00:00:00', '2010-05-19T00:00:00', '2010-05-26T00:00:00','2010-06-02T00:00:00', '2010-06-09T00:00:00', '2010-06-16T00:00:00','2010-06-23T00:00:00', '2010-06-30T00:00:00', '2010-07-07T00:00:00','2010-07-14T00:00:00', '2010-07-21T00:00:00', '2010-07-28T00:00:00','2010-08-04T00:00:00', '2010-08-11T00:00:00', '2010-08-18T00:00:00','2010-08-25T00:00:00', '2010-09-01T00:00:00', '2010-09-08T00:00:00','2010-09-15T00:00:00', '2010-09-22T00:00:00', '2010-09-29T00:00:00','2010-10-06T00:00:00', '2010-10-13T00:00:00', '2010-10-20T00:00:00','2010-10-27T00:00:00', '2010-11-03T00:00:00', '2010-11-10T00:00:00','2010-11-17T00:00:00', '2010-11-24T00:00:00', '2010-12-01T00:00:00','2010-12-08T00:00:00', '2010-12-15T00:00:00', '2010-12-22T00:00:00','2010-12-29T00:00:00', '2011-01-05T00:00:00', '2011-01-12T00:00:00','2011-01-19T00:00:00', '2011-01-26T00:00:00', '2011-02-02T00:00:00','2011-02-09T00:00:00', '2011-02-16T00:00:00', '2011-02-23T00:00:00','2011-03-02T00:00:00', '2011-03-09T00:00:00', '2011-03-16T00:00:00','2011-03-23T00:00:00', '2011-03-30T00:00:00', '2011-04-06T00:00:00','2011-04-13T00:00:00', '2011-04-20T00:00:00', '2011-04-27T00:00:00','2011-05-04T00:00:00', '2011-05-11T00:00:00', '2011-05-18T00:00:00','2011-05-25T00:00:00', '2011-06-01T00:00:00', '2011-06-08T00:00:00','2011-06-15T00:00:00', '2011-06-22T00:00:00', '2011-06-29T00:00:00','2011-07-06T00:00:00', '2011-07-13T00:00:00', '2011-07-20T00:00:00','2011-07-27T00:00:00', '2011-08-03T00:00:00', '2011-08-10T00:00:00','2011-08-17T00:00:00', '2011-08-24T00:00:00', '2011-08-31T00:00:00','2011-09-07T00:00:00', '2011-09-14T00:00:00', '2011-09-21T00:00:00','2011-09-28T00:00:00', '2011-10-05T00:00:00'], dtype='datetime64[s]')


obs_dim


(obs_dim)


\<U24


'25027::HNC' ... '6431::PL Frost...


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['25027::HNC', '25027::Cheerios 12oz', '25027::Cheerios 18oz','25027::Mini Wheats', '25027::PL Honey Nut Oats','25027::PL Frosted Wheat', '21237::HNC', '21237::Cheerios 12oz','21237::Cheerios 18oz', '21237::Mini Wheats','21237::PL Honey Nut Oats', '21237::PL Frosted Wheat', '25229::HNC','25229::Cheerios 12oz', '25229::Cheerios 18oz', '25229::Mini Wheats','25229::PL Honey Nut Oats', '25229::PL Frosted Wheat', '19265::HNC','19265::Cheerios 12oz', '19265::Cheerios 18oz', '19265::Mini Wheats','19265::PL Honey Nut Oats', '19265::PL Frosted Wheat', '9825::HNC','9825::Cheerios 12oz', '9825::Cheerios 18oz', '9825::Mini Wheats','9825::PL Honey Nut Oats', '9825::PL Frosted Wheat', '613::HNC','613::Cheerios 12oz', '613::Cheerios 18oz', '613::Mini Wheats','613::PL Honey Nut Oats', '613::PL Frosted Wheat', '2277::HNC','2277::Cheerios 12oz', '2277::Cheerios 18oz', '2277::Mini Wheats','2277::PL Honey Nut Oats', '2277::PL Frosted Wheat', '24991::HNC','24991::Cheerios 12oz', '24991::Cheerios 18oz', '24991::Mini Wheats','24991::PL Honey Nut Oats', '24991::PL Frosted Wheat', '6179::HNC','6179::Cheerios 12oz', '6179::Cheerios 18oz', '6179::Mini Wheats','6179::PL Honey Nut Oats', '6179::PL Frosted Wheat', '2513::HNC','2513::Cheerios 12oz', '2513::Cheerios 18oz', '2513::Mini Wheats','2513::PL Honey Nut Oats', '2513::PL Frosted Wheat', '2281::HNC','2281::Cheerios 12oz', '2281::Cheerios 18oz', '2281::Mini Wheats','2281::PL Honey Nut Oats', '2281::PL Frosted Wheat', '11993::HNC','11993::Cheerios 12oz', '11993::Cheerios 18oz', '11993::Mini Wheats','11993::PL Honey Nut Oats', '11993::PL Frosted Wheat', '25021::HNC','25021::Cheerios 12oz', '25021::Cheerios 18oz', '25021::Mini Wheats','25021::PL Honey Nut Oats', '25021::PL Frosted Wheat', '4259::HNC','4259::Cheerios 12oz', '4259::Cheerios 18oz', '4259::Mini Wheats','4259::PL Honey Nut Oats', '4259::PL Frosted Wheat', '21479::HNC','21479::Cheerios 12oz', '21479::Cheerios 18oz', '21479::Mini Wheats','21479::PL Honey Nut Oats', '21479::PL Frosted Wheat', '23349::HNC','23349::Cheerios 12oz', '23349::Cheerios 18oz', '23349::Mini Wheats','23349::PL Honey Nut Oats', '23349::PL Frosted Wheat', '19523::HNC','19523::Cheerios 12oz', '19523::Cheerios 18oz', '19523::Mini Wheats','19523::PL Honey Nut Oats', '19523::PL Frosted Wheat', '6431::HNC','6431::Cheerios 12oz', '6431::Cheerios 18oz', '6431::Mini Wheats','6431::PL Honey Nut Oats', '6431::PL Frosted Wheat'], dtype='<U24')


Data variables: (1)


obs


(time, obs_dim)


int32


70 181 69 46 50 ... 19 20 18 14 18


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[ 70, 181,  69, ...,  17,   6,  19],[ 78,  79,  43, ...,  24,  10,  16],[ 80,  69,  51, ...,  31,  10,  17],...,[156,  97,  51, ...,  21,  12,  26],[138,  67,  49, ...,  19,   8,  28],[170, 136,  57, ...,  18,  14,  18]], shape=(143, 108), dtype=int32)


Attributes: (5)


created_at :  
2026-09-07T11:31:59.222122+00:00

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


- input: 11
- time: 143
- series: 108


Coordinates: (3)


input


(input)


\<U23


'x' ... 'price PL Frosted Wheat'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['x', 'feature', 'display', 'sib_feature', 'sib_display', 'price HNC','price Cheerios 12oz', 'price Cheerios 18oz', 'price Mini Wheats','price PL Honey Nut Oats', 'price PL Frosted Wheat'], dtype='<U23')


time


(time)


datetime64\[s\]


2009-01-14 ... 2011-10-05


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['2009-01-14T00:00:00', '2009-01-21T00:00:00', '2009-01-28T00:00:00','2009-02-04T00:00:00', '2009-02-11T00:00:00', '2009-02-18T00:00:00','2009-02-25T00:00:00', '2009-03-04T00:00:00', '2009-03-11T00:00:00','2009-03-18T00:00:00', '2009-03-25T00:00:00', '2009-04-01T00:00:00','2009-04-08T00:00:00', '2009-04-15T00:00:00', '2009-04-22T00:00:00','2009-04-29T00:00:00', '2009-05-06T00:00:00', '2009-05-13T00:00:00','2009-05-20T00:00:00', '2009-05-27T00:00:00', '2009-06-03T00:00:00','2009-06-10T00:00:00', '2009-06-17T00:00:00', '2009-06-24T00:00:00','2009-07-01T00:00:00', '2009-07-08T00:00:00', '2009-07-15T00:00:00','2009-07-22T00:00:00', '2009-07-29T00:00:00', '2009-08-05T00:00:00','2009-08-12T00:00:00', '2009-08-19T00:00:00', '2009-08-26T00:00:00','2009-09-02T00:00:00', '2009-09-09T00:00:00', '2009-09-16T00:00:00','2009-09-23T00:00:00', '2009-09-30T00:00:00', '2009-10-07T00:00:00','2009-10-14T00:00:00', '2009-10-21T00:00:00', '2009-10-28T00:00:00','2009-11-04T00:00:00', '2009-11-11T00:00:00', '2009-11-18T00:00:00','2009-11-25T00:00:00', '2009-12-02T00:00:00', '2009-12-09T00:00:00','2009-12-16T00:00:00', '2009-12-23T00:00:00', '2009-12-30T00:00:00','2010-01-06T00:00:00', '2010-01-13T00:00:00', '2010-01-20T00:00:00','2010-01-27T00:00:00', '2010-02-03T00:00:00', '2010-02-10T00:00:00','2010-02-17T00:00:00', '2010-02-24T00:00:00', '2010-03-03T00:00:00','2010-03-10T00:00:00', '2010-03-17T00:00:00', '2010-03-24T00:00:00','2010-03-31T00:00:00', '2010-04-07T00:00:00', '2010-04-14T00:00:00','2010-04-21T00:00:00', '2010-04-28T00:00:00', '2010-05-05T00:00:00','2010-05-12T00:00:00', '2010-05-19T00:00:00', '2010-05-26T00:00:00','2010-06-02T00:00:00', '2010-06-09T00:00:00', '2010-06-16T00:00:00','2010-06-23T00:00:00', '2010-06-30T00:00:00', '2010-07-07T00:00:00','2010-07-14T00:00:00', '2010-07-21T00:00:00', '2010-07-28T00:00:00','2010-08-04T00:00:00', '2010-08-11T00:00:00', '2010-08-18T00:00:00','2010-08-25T00:00:00', '2010-09-01T00:00:00', '2010-09-08T00:00:00','2010-09-15T00:00:00', '2010-09-22T00:00:00', '2010-09-29T00:00:00','2010-10-06T00:00:00', '2010-10-13T00:00:00', '2010-10-20T00:00:00','2010-10-27T00:00:00', '2010-11-03T00:00:00', '2010-11-10T00:00:00','2010-11-17T00:00:00', '2010-11-24T00:00:00', '2010-12-01T00:00:00','2010-12-08T00:00:00', '2010-12-15T00:00:00', '2010-12-22T00:00:00','2010-12-29T00:00:00', '2011-01-05T00:00:00', '2011-01-12T00:00:00','2011-01-19T00:00:00', '2011-01-26T00:00:00', '2011-02-02T00:00:00','2011-02-09T00:00:00', '2011-02-16T00:00:00', '2011-02-23T00:00:00','2011-03-02T00:00:00', '2011-03-09T00:00:00', '2011-03-16T00:00:00','2011-03-23T00:00:00', '2011-03-30T00:00:00', '2011-04-06T00:00:00','2011-04-13T00:00:00', '2011-04-20T00:00:00', '2011-04-27T00:00:00','2011-05-04T00:00:00', '2011-05-11T00:00:00', '2011-05-18T00:00:00','2011-05-25T00:00:00', '2011-06-01T00:00:00', '2011-06-08T00:00:00','2011-06-15T00:00:00', '2011-06-22T00:00:00', '2011-06-29T00:00:00','2011-07-06T00:00:00', '2011-07-13T00:00:00', '2011-07-20T00:00:00','2011-07-27T00:00:00', '2011-08-03T00:00:00', '2011-08-10T00:00:00','2011-08-17T00:00:00', '2011-08-24T00:00:00', '2011-08-31T00:00:00','2011-09-07T00:00:00', '2011-09-14T00:00:00', '2011-09-21T00:00:00','2011-09-28T00:00:00', '2011-10-05T00:00:00'], dtype='datetime64[s]')


series


(series)


\<U24


'25027::HNC' ... '6431::PL Frost...


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['25027::HNC', '25027::Cheerios 12oz', '25027::Cheerios 18oz','25027::Mini Wheats', '25027::PL Honey Nut Oats','25027::PL Frosted Wheat', '21237::HNC', '21237::Cheerios 12oz','21237::Cheerios 18oz', '21237::Mini Wheats','21237::PL Honey Nut Oats', '21237::PL Frosted Wheat', '25229::HNC','25229::Cheerios 12oz', '25229::Cheerios 18oz', '25229::Mini Wheats','25229::PL Honey Nut Oats', '25229::PL Frosted Wheat', '19265::HNC','19265::Cheerios 12oz', '19265::Cheerios 18oz', '19265::Mini Wheats','19265::PL Honey Nut Oats', '19265::PL Frosted Wheat', '9825::HNC','9825::Cheerios 12oz', '9825::Cheerios 18oz', '9825::Mini Wheats','9825::PL Honey Nut Oats', '9825::PL Frosted Wheat', '613::HNC','613::Cheerios 12oz', '613::Cheerios 18oz', '613::Mini Wheats','613::PL Honey Nut Oats', '613::PL Frosted Wheat', '2277::HNC','2277::Cheerios 12oz', '2277::Cheerios 18oz', '2277::Mini Wheats','2277::PL Honey Nut Oats', '2277::PL Frosted Wheat', '24991::HNC','24991::Cheerios 12oz', '24991::Cheerios 18oz', '24991::Mini Wheats','24991::PL Honey Nut Oats', '24991::PL Frosted Wheat', '6179::HNC','6179::Cheerios 12oz', '6179::Cheerios 18oz', '6179::Mini Wheats','6179::PL Honey Nut Oats', '6179::PL Frosted Wheat', '2513::HNC','2513::Cheerios 12oz', '2513::Cheerios 18oz', '2513::Mini Wheats','2513::PL Honey Nut Oats', '2513::PL Frosted Wheat', '2281::HNC','2281::Cheerios 12oz', '2281::Cheerios 18oz', '2281::Mini Wheats','2281::PL Honey Nut Oats', '2281::PL Frosted Wheat', '11993::HNC','11993::Cheerios 12oz', '11993::Cheerios 18oz', '11993::Mini Wheats','11993::PL Honey Nut Oats', '11993::PL Frosted Wheat', '25021::HNC','25021::Cheerios 12oz', '25021::Cheerios 18oz', '25021::Mini Wheats','25021::PL Honey Nut Oats', '25021::PL Frosted Wheat', '4259::HNC','4259::Cheerios 12oz', '4259::Cheerios 18oz', '4259::Mini Wheats','4259::PL Honey Nut Oats', '4259::PL Frosted Wheat', '21479::HNC','21479::Cheerios 12oz', '21479::Cheerios 18oz', '21479::Mini Wheats','21479::PL Honey Nut Oats', '21479::PL Frosted Wheat', '23349::HNC','23349::Cheerios 12oz', '23349::Cheerios 18oz', '23349::Mini Wheats','23349::PL Honey Nut Oats', '23349::PL Frosted Wheat', '19523::HNC','19523::Cheerios 12oz', '19523::Cheerios 18oz', '19523::Mini Wheats','19523::PL Honey Nut Oats', '19523::PL Frosted Wheat', '6431::HNC','6431::Cheerios 12oz', '6431::Cheerios 18oz', '6431::Mini Wheats','6431::PL Honey Nut Oats', '6431::PL Frosted Wheat'], dtype='<U24')


Data variables: (1)


covariates


(input, time, series)


float32


0.0 -0.2239 0.0 0.0 ... 0.0 0.0 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 0.        , -0.22394004,  0.        , ...,  0.        ,0.        , -0.20731209],[ 0.        , -0.17344388,  0.        , ...,  0.        ,0.        , -0.16126814],[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        , -0.19689533],...,[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ]],[[ 0.        ,  1.        ,  0.        , ...,  0.        ,0.        ,  0.        ],[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],...[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ]],[[ 0.        ,  0.        ,  0.        , ..., -0.20731209,-0.20731209, -0.20731209],[ 0.        ,  0.        ,  0.        , ..., -0.16126814,-0.16126814, -0.16126814],[ 0.        ,  0.        ,  0.        , ..., -0.19689533,-0.19689533, -0.19689533],...,[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ]]], shape=(11, 143, 108), dtype=float32)


Attributes: (5)


created_at :  
2026-09-07T11:31:59.223942+00:00

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
- time: 13
- obs_dim: 108


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


datetime64\[s\]


2011-10-12 ... 2012-01-04


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['2011-10-12T00:00:00', '2011-10-19T00:00:00', '2011-10-26T00:00:00','2011-11-02T00:00:00', '2011-11-09T00:00:00', '2011-11-16T00:00:00','2011-11-23T00:00:00', '2011-11-30T00:00:00', '2011-12-07T00:00:00','2011-12-14T00:00:00', '2011-12-21T00:00:00', '2011-12-28T00:00:00','2012-01-04T00:00:00'], dtype='datetime64[s]')


obs_dim


(obs_dim)


\<U24


'25027::HNC' ... '6431::PL Frost...


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['25027::HNC', '25027::Cheerios 12oz', '25027::Cheerios 18oz','25027::Mini Wheats', '25027::PL Honey Nut Oats','25027::PL Frosted Wheat', '21237::HNC', '21237::Cheerios 12oz','21237::Cheerios 18oz', '21237::Mini Wheats','21237::PL Honey Nut Oats', '21237::PL Frosted Wheat', '25229::HNC','25229::Cheerios 12oz', '25229::Cheerios 18oz', '25229::Mini Wheats','25229::PL Honey Nut Oats', '25229::PL Frosted Wheat', '19265::HNC','19265::Cheerios 12oz', '19265::Cheerios 18oz', '19265::Mini Wheats','19265::PL Honey Nut Oats', '19265::PL Frosted Wheat', '9825::HNC','9825::Cheerios 12oz', '9825::Cheerios 18oz', '9825::Mini Wheats','9825::PL Honey Nut Oats', '9825::PL Frosted Wheat', '613::HNC','613::Cheerios 12oz', '613::Cheerios 18oz', '613::Mini Wheats','613::PL Honey Nut Oats', '613::PL Frosted Wheat', '2277::HNC','2277::Cheerios 12oz', '2277::Cheerios 18oz', '2277::Mini Wheats','2277::PL Honey Nut Oats', '2277::PL Frosted Wheat', '24991::HNC','24991::Cheerios 12oz', '24991::Cheerios 18oz', '24991::Mini Wheats','24991::PL Honey Nut Oats', '24991::PL Frosted Wheat', '6179::HNC','6179::Cheerios 12oz', '6179::Cheerios 18oz', '6179::Mini Wheats','6179::PL Honey Nut Oats', '6179::PL Frosted Wheat', '2513::HNC','2513::Cheerios 12oz', '2513::Cheerios 18oz', '2513::Mini Wheats','2513::PL Honey Nut Oats', '2513::PL Frosted Wheat', '2281::HNC','2281::Cheerios 12oz', '2281::Cheerios 18oz', '2281::Mini Wheats','2281::PL Honey Nut Oats', '2281::PL Frosted Wheat', '11993::HNC','11993::Cheerios 12oz', '11993::Cheerios 18oz', '11993::Mini Wheats','11993::PL Honey Nut Oats', '11993::PL Frosted Wheat', '25021::HNC','25021::Cheerios 12oz', '25021::Cheerios 18oz', '25021::Mini Wheats','25021::PL Honey Nut Oats', '25021::PL Frosted Wheat', '4259::HNC','4259::Cheerios 12oz', '4259::Cheerios 18oz', '4259::Mini Wheats','4259::PL Honey Nut Oats', '4259::PL Frosted Wheat', '21479::HNC','21479::Cheerios 12oz', '21479::Cheerios 18oz', '21479::Mini Wheats','21479::PL Honey Nut Oats', '21479::PL Frosted Wheat', '23349::HNC','23349::Cheerios 12oz', '23349::Cheerios 18oz', '23349::Mini Wheats','23349::PL Honey Nut Oats', '23349::PL Frosted Wheat', '19523::HNC','19523::Cheerios 12oz', '19523::Cheerios 18oz', '19523::Mini Wheats','19523::PL Honey Nut Oats', '19523::PL Frosted Wheat', '6431::HNC','6431::Cheerios 12oz', '6431::Cheerios 18oz', '6431::Mini Wheats','6431::PL Honey Nut Oats', '6431::PL Frosted Wheat'], dtype='<U24')


Data variables: (1)


obs


(chain, draw, time, obs_dim)


int32


173 125 62 58 56 ... 15 15 16 16 38


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 173,  125,   62, ...,   27,   13,   25],[ 207,   81,   70, ...,   24,   16,   29],[  99,  133,   50, ...,   17,   20,   33],...,[ 555,   92,   48, ...,   21,    4,   22],[ 499,   96,   53, ...,   15,    4,   30],[ 639,  113,   65, ...,   11,    5,   24]],[[ 149,   68,   35, ...,   22,    4,   24],[ 104,  117,   66, ...,   22,   19,   23],[ 107,   70,   71, ...,   14,   17,   18],...,[ 683,   41,   50, ...,   13,   10,   14],[ 535,   91,   65, ...,   27,    5,   18],[ 483,   80,   24, ...,   41,    4,   26]],[[ 254,   76,   71, ...,   20,    4,   19],[ 130,  107,   72, ...,   28,   12,   30],[ 186,   99,   45, ...,   32,    9,   32],...,...[1382,  108,   53, ...,   24,    5,   21],[1186,  183,   51, ...,   24,    9,   16],[1629,  148,   71, ...,   29,   13,   24]],[[ 248,  113,   43, ...,   16,   12,   24],[ 169,  172,   74, ...,   15,    6,   30],[ 245,  162,   47, ...,   18,   10,   30],...,[1762,   95,  103, ...,   11,    4,   33],[2277,  100,   50, ...,    9,    6,   28],[1330,  106,   56, ...,    8,    1,   30]],[[  83,   91,   36, ...,   24,   10,   22],[ 125,  115,   67, ...,   23,    6,   11],[ 109,  119,   62, ...,   16,   11,   23],...,[ 873,  113,   79, ...,   31,    5,   15],[1373,   99,   45, ...,   23,   13,   24],[ 765,  125,   65, ...,   16,   16,   38]]]],shape=(4, 1000, 13, 108), dtype=int32)


Attributes: (5)


created_at :  
2026-09-07T11:32:01.499300+00:00

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


- input: 11
- time: 13
- series: 108


Coordinates: (3)


input


(input)


\<U23


'x' ... 'price PL Frosted Wheat'


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['x', 'feature', 'display', 'sib_feature', 'sib_display', 'price HNC','price Cheerios 12oz', 'price Cheerios 18oz', 'price Mini Wheats','price PL Honey Nut Oats', 'price PL Frosted Wheat'], dtype='<U23')


time


(time)


datetime64\[s\]


2011-10-12 ... 2012-01-04


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array(['2011-10-12T00:00:00', '2011-10-19T00:00:00', '2011-10-26T00:00:00','2011-11-02T00:00:00', '2011-11-09T00:00:00', '2011-11-16T00:00:00','2011-11-23T00:00:00', '2011-11-30T00:00:00', '2011-12-07T00:00:00','2011-12-14T00:00:00', '2011-12-21T00:00:00', '2011-12-28T00:00:00','2012-01-04T00:00:00'], dtype='datetime64[s]')


series


(series)


\<U24


'25027::HNC' ... '6431::PL Frost...


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


2oz', '2513::Cheerios 18oz', '2513::Mini Wheats','2513::PL Honey Nut Oats', '2513::PL Frosted Wheat', '2281::HNC','2281::Cheerios 12oz', '2281::Cheerios 18oz', '2281::Mini Wheats','2281::PL Honey Nut Oats', '2281::PL Frosted Wheat', '11993::HNC','11993::Cheerios 12oz', '11993::Cheerios 18oz', '11993::Mini Wheats','11993::PL Honey Nut Oats', '11993::PL Frosted Wheat', '25021::HNC','25021::Cheerios 12oz', '25021::Cheerios 18oz', '25021::Mini Wheats','25021::PL Honey Nut Oats', '25021::PL Frosted Wheat', '4259::HNC','4259::Cheerios 12oz', '4259::Cheerios 18oz', '4259::Mini Wheats','4259::PL Honey Nut Oats', '4259::PL Frosted Wheat', '21479::HNC','21479::Cheerios 12oz', '21479::Cheerios 18oz', '21479::Mini Wheats','21479::PL Honey Nut Oats', '21479::PL Frosted Wheat', '23349::HNC','23349::Cheerios 12oz', '23349::Cheerios 18oz', '23349::Mini Wheats','23349::PL Honey Nut Oats', '23349::PL Frosted Wheat', '19523::HNC','19523::Cheerios 12oz', '19523::Cheerios 18oz', '19523::Mini Wheats','19523::PL Honey Nut Oats', '19523::PL Frosted Wheat', '6431::HNC','6431::Cheerios 12oz', '6431::Cheerios 18oz', '6431::Mini Wheats','6431::PL Honey Nut Oats', '6431::PL Frosted Wheat'], dtype='<U24')


Data variables: (1)


covariates


(input, time, series)


float32


0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],[ 0.        ,  0.        ,  0.        , ...,  0.        ,-0.01234584,  0.        ],[ 0.        ,  0.        ,  0.        , ...,  0.        ,-0.06331228,  0.        ],...,[-0.43354294,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],[-0.43858072,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],[-0.43858072,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ]],[[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],[ 0.        ,  0.        ,  0.        , ...,  0.        ,1.        ,  0.        ],[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],...[-0.18026182, -0.18026182, -0.18026182, ...,  0.        ,0.        ,  0.        ],[-0.15587038, -0.15587038, -0.15587038, ...,  0.        ,0.        ,  0.        ],[-0.17410795, -0.17410795, -0.17410795, ...,  0.        ,0.        ,  0.        ]],[[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],...,[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ],[ 0.        ,  0.        ,  0.        , ...,  0.        ,0.        ,  0.        ]]], shape=(11, 13, 108), dtype=float32)


Attributes: (5)


created_at :  
2026-09-07T11:32:01.499812+00:00

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


    In [32]:


``` python
hyper_vars = [*product_sites, "eps_scale", "cross_scale"]
summary = az.summary(tree, var_names=hyper_vars, ci_kind="hdi", ci_prob=0.94)
print(
    f"max r_hat: {summary['r_hat'].astype(float).max():.3f} | "
    f"min ess_bulk: {summary['ess_bulk'].astype(float).min():.0f}"
)
summary
```


    max r_hat: 1.000 | min ess_bulk: 712


|  | mean | sd | hdi94_lb | hdi94_ub | ess_bulk | ess_tail | r_hat | mcse_mean | mcse_sd |
|----|----|----|----|----|----|----|----|----|----|
| eps_prod\[HNC\] | -1.082 | 0.108 | -1.3 | -0.87 | 1976 | 2516 | 1.00 | 0.0024 | 0.0018 |
| eps_prod\[Cheerios 12oz\] | -0.377 | 0.081 | -0.53 | -0.22 | 1344 | 2024 | 1.00 | 0.0022 | 0.0016 |
| eps_prod\[Cheerios 18oz\] | -1.911 | 0.155 | -2.2 | -1.6 | 1582 | 2264 | 1.00 | 0.0039 | 0.0027 |
| eps_prod\[Mini Wheats\] | -1.95 | 0.127 | -2.2 | -1.7 | 2275 | 2228 | 1.00 | 0.0027 | 0.0019 |
| eps_prod\[PL Honey Nut Oats\] | -1.926 | 0.163 | -2.2 | -1.6 | 2961 | 2720 | 1.00 | 0.003 | 0.0022 |
| eps_prod\[PL Frosted Wheat\] | -1.034 | 0.104 | -1.2 | -0.83 | 2638 | 2705 | 1.00 | 0.002 | 0.0015 |
| conc\[HNC\] | 17.21 | 0.8 | 16 | 19 | 712 | 1186 | 1.00 | 0.03 | 0.022 |
| conc\[Cheerios 12oz\] | 19.2 | 0.85 | 18 | 21 | 2601 | 2538 | 1.00 | 0.017 | 0.012 |
| conc\[Cheerios 18oz\] | 21.95 | 1.04 | 20 | 24 | 3003 | 3025 | 1.00 | 0.019 | 0.013 |
| conc\[Mini Wheats\] | 18.85 | 0.86 | 17 | 21 | 3373 | 2770 | 1.00 | 0.015 | 0.011 |
| conc\[PL Honey Nut Oats\] | 19.55 | 1.18 | 17 | 22 | 1331 | 1813 | 1.00 | 0.033 | 0.024 |
| conc\[PL Frosted Wheat\] | 21.21 | 1 | 19 | 23 | 3084 | 3000 | 1.00 | 0.018 | 0.013 |
| b_feat\[HNC\] | 0.759 | 0.042 | 0.68 | 0.84 | 2514 | 2886 | 1.00 | 0.00083 | 0.0006 |
| b_feat\[Cheerios 12oz\] | 0.797 | 0.062 | 0.68 | 0.91 | 1466 | 2019 | 1.00 | 0.0016 | 0.0011 |
| b_feat\[Cheerios 18oz\] | 0.102 | 0.084 | -0.052 | 0.26 | 1949 | 2512 | 1.00 | 0.0019 | 0.0014 |
| b_feat\[Mini Wheats\] | 0.299 | 0.038 | 0.23 | 0.37 | 2369 | 2729 | 1.00 | 0.00078 | 0.00056 |
| b_feat\[PL Honey Nut Oats\] | 0.121 | 0.057 | 0.018 | 0.23 | 2569 | 2795 | 1.00 | 0.0011 | 0.00079 |
| b_feat\[PL Frosted Wheat\] | 0.237 | 0.042 | 0.16 | 0.32 | 2206 | 2722 | 1.00 | 0.00089 | 0.00065 |
| b_disp\[HNC\] | 0.344 | 0.052 | 0.24 | 0.44 | 1808 | 2173 | 1.00 | 0.0012 | 0.00084 |
| b_disp\[Cheerios 12oz\] | 0.468 | 0.053 | 0.37 | 0.57 | 1794 | 2407 | 1.00 | 0.0012 | 0.00087 |
| b_disp\[Cheerios 18oz\] | 0.266 | 0.051 | 0.17 | 0.36 | 2986 | 2851 | 1.00 | 0.00093 | 0.00064 |
| b_disp\[Mini Wheats\] | 0.283 | 0.064 | 0.16 | 0.4 | 1530 | 2326 | 1.00 | 0.0016 | 0.0011 |
| b_disp\[PL Honey Nut Oats\] | 0.269 | 0.053 | 0.17 | 0.37 | 2593 | 2787 | 1.00 | 0.001 | 0.00075 |
| b_disp\[PL Frosted Wheat\] | 0.164 | 0.057 | 0.058 | 0.27 | 2405 | 2790 | 1.00 | 0.0012 | 0.00081 |
| b_fd\[HNC\] | -0.027 | 0.058 | -0.13 | 0.087 | 1985 | 2360 | 1.00 | 0.0013 | 0.00093 |
| b_fd\[Cheerios 12oz\] | -0.064 | 0.055 | -0.17 | 0.042 | 2881 | 2860 | 1.00 | 0.001 | 0.00072 |
| b_fd\[Cheerios 18oz\] | -0.249 | 0.08 | -0.4 | -0.1 | 1520 | 2031 | 1.00 | 0.002 | 0.0014 |
| b_fd\[Mini Wheats\] | 0.007 | 0.06 | -0.11 | 0.12 | 2608 | 2586 | 1.00 | 0.0012 | 0.00079 |
| b_fd\[PL Honey Nut Oats\] | -0.062 | 0.076 | -0.2 | 0.083 | 3189 | 2970 | 1.00 | 0.0013 | 0.00097 |
| b_fd\[PL Frosted Wheat\] | -0.012 | 0.06 | -0.12 | 0.1 | 3873 | 2832 | 1.00 | 0.00097 | 0.00072 |
| b_feat_depth\[HNC\] | 0.039 | 0.109 | -0.16 | 0.24 | 1971 | 2565 | 1.00 | 0.0025 | 0.0017 |
| b_feat_depth\[Cheerios 12oz\] | -0.46 | 0.211 | -0.85 | -0.054 | 1529 | 2141 | 1.00 | 0.0054 | 0.0037 |
| b_feat_depth\[Cheerios 18oz\] | 0.879 | 0.171 | 0.56 | 1.2 | 1879 | 2503 | 1.00 | 0.0039 | 0.0028 |
| b_feat_depth\[Mini Wheats\] | -0.392 | 0.17 | -0.71 | -0.07 | 1805 | 2360 | 1.00 | 0.004 | 0.0029 |
| b_feat_depth\[PL Honey Nut Oats\] | 0.15 | 0.4 | -0.59 | 0.9 | 2891 | 3073 | 1.00 | 0.0074 | 0.0052 |
| b_feat_depth\[PL Frosted Wheat\] | -0.29 | 0.238 | -0.75 | 0.15 | 2078 | 2632 | 1.00 | 0.0052 | 0.0037 |
| b_disp_depth\[HNC\] | 0.039 | 0.118 | -0.18 | 0.26 | 2081 | 2422 | 1.00 | 0.0026 | 0.0018 |
| b_disp_depth\[Cheerios 12oz\] | -0.029 | 0.189 | -0.39 | 0.31 | 1544 | 2391 | 1.00 | 0.0048 | 0.0034 |
| b_disp_depth\[Cheerios 18oz\] | -0.043 | 0.162 | -0.35 | 0.26 | 1565 | 2110 | 1.00 | 0.0041 | 0.0028 |
| b_disp_depth\[Mini Wheats\] | 0.43 | 0.229 | -0.0057 | 0.84 | 1456 | 2548 | 1.00 | 0.006 | 0.0042 |
| b_disp_depth\[PL Honey Nut Oats\] | -1.15 | 0.356 | -1.8 | -0.49 | 2646 | 3071 | 1.00 | 0.0069 | 0.0047 |
| b_disp_depth\[PL Frosted Wheat\] | 0.46 | 0.328 | -0.17 | 1.1 | 2445 | 2711 | 1.00 | 0.0066 | 0.0047 |
| b_sib_feat\[HNC\] | -0.0713 | 0.0155 | -0.1 | -0.043 | 2861 | 3020 | 1.00 | 0.00029 | 0.0002 |
| b_sib_feat\[Cheerios 12oz\] | -0.0192 | 0.0146 | -0.047 | 0.0083 | 3335 | 3092 | 1.00 | 0.00025 | 0.00018 |
| b_sib_feat\[Cheerios 18oz\] | 0.0352 | 0.0148 | 0.0077 | 0.063 | 3457 | 2831 | 1.00 | 0.00025 | 0.00018 |
| b_sib_feat\[Mini Wheats\] | 0.0086 | 0.0171 | -0.024 | 0.04 | 2584 | 2430 | 1.00 | 0.00034 | 0.00023 |
| b_sib_feat\[PL Honey Nut Oats\] | 0.0239 | 0.0171 | -0.0084 | 0.056 | 3063 | 2868 | 1.00 | 0.00031 | 0.00022 |
| b_sib_feat\[PL Frosted Wheat\] | 0.0216 | 0.015 | -0.0069 | 0.051 | 3809 | 3222 | 1.00 | 0.00024 | 0.00017 |
| b_sib_disp\[HNC\] | -0.0531 | 0.0156 | -0.082 | -0.023 | 2834 | 3049 | 1.00 | 0.00029 | 0.00021 |
| b_sib_disp\[Cheerios 12oz\] | 0.007 | 0.0164 | -0.023 | 0.037 | 3627 | 3198 | 1.00 | 0.00027 | 0.0002 |
| b_sib_disp\[Cheerios 18oz\] | -0.0233 | 0.015 | -0.051 | 0.0051 | 3154 | 2871 | 1.00 | 0.00027 | 0.00019 |
| b_sib_disp\[Mini Wheats\] | -0.0234 | 0.0164 | -0.053 | 0.0073 | 3497 | 3156 | 1.00 | 0.00028 | 0.00019 |
| b_sib_disp\[PL Honey Nut Oats\] | -0.0189 | 0.0169 | -0.051 | 0.013 | 3422 | 3216 | 1.00 | 0.00029 | 0.0002 |
| b_sib_disp\[PL Frosted Wheat\] | 0.0158 | 0.0155 | -0.014 | 0.045 | 3851 | 3069 | 1.00 | 0.00025 | 0.00018 |
| eps_scale | 0.294 | 0.033 | 0.24 | 0.36 | 2226 | 2818 | 1.00 | 0.00069 | 0.00051 |
| cross_scale | 0.271 | 0.04 | 0.21 | 0.36 | 1310 | 2410 | 1.00 | 0.0011 | 0.00097 |


    In [33]:


``` python
centering_summary = az.summary(
    tree, var_names=["centered_eps", "centered_gamma"], ci_kind="hdi", ci_prob=0.94
)
centering_summary
```


|  | mean | sd | hdi94_lb | hdi94_ub | ess_bulk | ess_tail | r_hat | mcse_mean | mcse_sd |
|----|----|----|----|----|----|----|----|----|----|
| centered_eps | 0.5 | 0.3 | 0.03 | 0.98 | 14 | 53 | 1.22 | 0.082 | 0.035 |
| centered_gamma | 0.49 | 0.3 | 0.02 | 0.97 | 164 | 288 | 1.01 | 0.024 | 0.01 |


    In [34]:


``` python
az.summary(tree, var_names=["gamma_offdiag"], ci_kind="hdi", ci_prob=0.94)
```


|  | mean | sd | hdi94_lb | hdi94_ub | ess_bulk | ess_tail | r_hat | mcse_mean | mcse_sd |
|----|----|----|----|----|----|----|----|----|----|
| gamma_offdiag\[HNC -\> Cheerios 12oz\] | 0.11 | 0.048 | 0.018 | 0.2 | 3570 | 2998 | 1.00 | 0.00081 | 0.00059 |
| gamma_offdiag\[HNC -\> Cheerios 18oz\] | 0.017 | 0.047 | -0.07 | 0.1 | 2895 | 2791 | 1.00 | 0.00087 | 0.0006 |
| gamma_offdiag\[HNC -\> Mini Wheats\] | -0.025 | 0.049 | -0.12 | 0.064 | 2782 | 2679 | 1.00 | 0.00092 | 0.00066 |
| gamma_offdiag\[HNC -\> PL Honey Nut Oats\] | 0.692 | 0.052 | 0.59 | 0.79 | 4521 | 3670 | 1.00 | 0.00078 | 0.00056 |
| gamma_offdiag\[HNC -\> PL Frosted Wheat\] | 0.078 | 0.045 | -0.0056 | 0.16 | 3638 | 3235 | 1.00 | 0.00074 | 0.00051 |
| gamma_offdiag\[Cheerios 12oz -\> HNC\] | -0.25 | 0.038 | -0.32 | -0.18 | 4287 | 3635 | 1.00 | 0.00058 | 0.0004 |
| gamma_offdiag\[Cheerios 12oz -\> Cheerios 18oz\] | 0.151 | 0.0361 | 0.084 | 0.22 | 4887 | 3641 | 1.00 | 0.00052 | 0.00037 |
| gamma_offdiag\[Cheerios 12oz -\> Mini Wheats\] | 0.032 | 0.038 | -0.039 | 0.1 | 2631 | 2577 | 1.00 | 0.00073 | 0.00051 |
| gamma_offdiag\[Cheerios 12oz -\> PL Honey Nut Oats\] | -0.057 | 0.041 | -0.13 | 0.017 | 3246 | 2774 | 1.00 | 0.00071 | 0.00052 |
| gamma_offdiag\[Cheerios 12oz -\> PL Frosted Wheat\] | -0.007 | 0.036 | -0.075 | 0.06 | 2688 | 2319 | 1.00 | 0.0007 | 0.00049 |
| gamma_offdiag\[Cheerios 18oz -\> HNC\] | -0.308 | 0.0438 | -0.39 | -0.23 | 4380 | 3840 | 1.00 | 0.00066 | 0.00046 |
| gamma_offdiag\[Cheerios 18oz -\> Cheerios 12oz\] | 0.215 | 0.0455 | 0.13 | 0.3 | 4604 | 3723 | 1.00 | 0.00067 | 0.00046 |
| gamma_offdiag\[Cheerios 18oz -\> Mini Wheats\] | 0.01 | 0.047 | -0.078 | 0.1 | 2673 | 2758 | 1.00 | 0.00092 | 0.00066 |
| gamma_offdiag\[Cheerios 18oz -\> PL Honey Nut Oats\] | 0.038 | 0.047 | -0.049 | 0.13 | 2424 | 2837 | 1.00 | 0.00095 | 0.00067 |
| gamma_offdiag\[Cheerios 18oz -\> PL Frosted Wheat\] | 0.127 | 0.042 | 0.046 | 0.2 | 3631 | 3113 | 1.00 | 0.0007 | 0.0005 |
| gamma_offdiag\[Mini Wheats -\> HNC\] | -0.104 | 0.063 | -0.23 | 0.015 | 3355 | 3117 | 1.00 | 0.0011 | 0.00079 |
| gamma_offdiag\[Mini Wheats -\> Cheerios 12oz\] | -0.197 | 0.062 | -0.31 | -0.079 | 4308 | 3256 | 1.00 | 0.00095 | 0.00066 |
| gamma_offdiag\[Mini Wheats -\> Cheerios 18oz\] | 0.18 | 0.06 | 0.069 | 0.29 | 4307 | 3218 | 1.00 | 0.00091 | 0.00064 |
| gamma_offdiag\[Mini Wheats -\> PL Honey Nut Oats\] | -0.066 | 0.065 | -0.19 | 0.059 | 3633 | 3247 | 1.00 | 0.0011 | 0.00073 |
| gamma_offdiag\[Mini Wheats -\> PL Frosted Wheat\] | 0.534 | 0.06 | 0.42 | 0.65 | 4305 | 3850 | 1.00 | 0.00091 | 0.00065 |
| gamma_offdiag\[PL Honey Nut Oats -\> HNC\] | -0.074 | 0.115 | -0.28 | 0.14 | 3445 | 2777 | 1.00 | 0.002 | 0.0013 |
| gamma_offdiag\[PL Honey Nut Oats -\> Cheerios 12oz\] | -0.697 | 0.112 | -0.9 | -0.49 | 3958 | 3857 | 1.00 | 0.0018 | 0.0012 |
| gamma_offdiag\[PL Honey Nut Oats -\> Cheerios 18oz\] | -0.149 | 0.104 | -0.34 | 0.041 | 3848 | 3094 | 1.00 | 0.0017 | 0.0012 |
| gamma_offdiag\[PL Honey Nut Oats -\> Mini Wheats\] | 0.143 | 0.111 | -0.061 | 0.35 | 3853 | 3022 | 1.00 | 0.0018 | 0.0012 |
| gamma_offdiag\[PL Honey Nut Oats -\> PL Frosted Wheat\] | 0.281 | 0.103 | 0.09 | 0.47 | 4019 | 3288 | 1.00 | 0.0016 | 0.0011 |
| gamma_offdiag\[PL Frosted Wheat -\> HNC\] | 0.095 | 0.078 | -0.054 | 0.24 | 2621 | 2714 | 1.00 | 0.0015 | 0.0011 |
| gamma_offdiag\[PL Frosted Wheat -\> Cheerios 12oz\] | 0.238 | 0.074 | 0.099 | 0.37 | 4255 | 3422 | 1.00 | 0.0011 | 0.00082 |
| gamma_offdiag\[PL Frosted Wheat -\> Cheerios 18oz\] | -0.04 | 0.071 | -0.17 | 0.093 | 3263 | 2803 | 1.00 | 0.0012 | 0.00091 |
| gamma_offdiag\[PL Frosted Wheat -\> Mini Wheats\] | 0.085 | 0.074 | -0.053 | 0.23 | 2864 | 2388 | 1.00 | 0.0014 | 0.00098 |
| gamma_offdiag\[PL Frosted Wheat -\> PL Honey Nut Oats\] | 0.365 | 0.081 | 0.22 | 0.52 | 3714 | 3512 | 1.00 | 0.0013 | 0.00091 |


    In [35]:


``` python
pc_trace = az.plot_trace_dist(
    tree,
    var_names=[
        "eps_prod",
        "b_feat",
        "b_disp",
        "conc",
        "cross_scale",
        "centered_eps",
        "centered_gamma",
    ],
    compact=True,
    figure_kwargs={"figsize": (12, 16)},
)
pc_trace.viz["figure"].item().suptitle("Trace plots", fontsize=18, fontweight="bold", y=1.02);
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-36-output-1.png" class="figure-img" width="1211" height="1647" /></p>
</figure>


# What the model learned

The first table puts the posterior of each product's own elasticity next to the least-squares estimates with and without the depth slopes. A model whose feature and display effects fall far below the least-squares ones would be a model whose random-walk level absorbs the promotion spikes; the second table compares the mechanics multipliers. The forest plots show the own elasticities and the mechanics effects with their 50\\ and 94\\ HDIs, annotated with the identifying store-weeks; the heatmap shows the cross matrix, its cells annotated with the posterior probability of a positive cross elasticity (a positive value means that a cheaper competitor takes units away). The store-level plot of the focal product's elasticity shows the partial pooling at work: the within-store least-squares estimates scatter widely, the posterior medians shrink toward the product mean.

The focal product's elasticity has a posterior median of -1.08 (94\\ HDI -1.28 to -0.87), on top of the least-squares estimate with depth slopes. Cheerios 18 oz, with 23 identifying store-weeks, still gets a median of -1.91 (HDI -2.20 to -1.62), because its cut depth varies inside its feature and display weeks. The feature multiplier of the focal product is 2.13 (HDI 1.97 to 2.30) and the display multiplier 1.41 (HDI 1.28 to 1.55), above and near the pooled least-squares multipliers of 1.65 and 1.49, so the level is not absorbing the promotion spikes. The depth slopes under feature and display of the focal product are centered near zero, so its mechanics-specific elasticities differ little from the plain one. The cross matrix has one large cell: the focal product's price on the private-label twin's units, +0.69 (HDI +0.60 to +0.79, posterior probability of a positive value 1.00), while the twin's price barely moves the focal product (-0.07, HDI -0.29 to +0.13). The store-level elasticities of the focal product have a least-squares spread of 0.37 across stores and a posterior-median spread of 0.31, the shrinkage the planner section uses. The posterior-mean seasonal component peaks in horizon week 12, the week of December 28, and the level innovation scales range from 0.019 to 0.219 across series.


    In [36]:


``` python
eps_prod_draws = np.asarray(posterior["eps_prod"])


def interval_row(draws: Float[np.ndarray, " sample"]) -> dict[str, float]:
    r"""Posterior median with the $50\%$ and $94\%$ HDI bounds of one-dimensional draws."""
    lower_50, upper_50 = hdi_bounds(draws, 0.5)
    lower_94, upper_94 = hdi_bounds(draws, 0.94)
    return {
        "median": float(np.median(draws)),
        "hdi50_lower": lower_50,
        "hdi50_upper": upper_50,
        "hdi94_lower": lower_94,
        "hdi94_upper": upper_94,
    }


elasticity_table = pl.DataFrame(
    [
        {
            "product": product,
            "ols_with_flags": float(
                own_elasticity_ols.filter(pl.col("product") == product)["with_flags"][0]
            ),
            "ols_with_depth_slopes": float(
                own_elasticity_ols.filter(pl.col("product") == product)["with_depth_slopes"][0]
            ),
            **interval_row(eps_prod_draws[:, k]),
        }
        for k, product in enumerate(product_order)
    ]
)
elasticity_table
```


| product | ols_with_flags | ols_with_depth_slopes | median | hdi50_lower | hdi50_upper | hdi94_lower | hdi94_upper |
|----|----|----|----|----|----|----|----|
| "HNC" | -1.326977 | -1.084921 | -1.083406 | -1.151331 | -1.010191 | -1.275332 | -0.866474 |
| "Cheerios 12oz" | -0.3183 | -0.31104 | -0.377091 | -0.423967 | -0.317098 | -0.530284 | -0.229161 |
| "Cheerios 18oz" | -1.961065 | -1.511146 | -1.912121 | -2.020236 | -1.814652 | -2.201686 | -1.617019 |
| "Mini Wheats" | -1.558958 | -1.626816 | -1.950698 | -2.03799 | -1.870485 | -2.199207 | -1.714125 |
| "PL Honey Nut Oats" | -1.446412 | -1.468914 | -1.925505 | -2.033653 | -1.819127 | -2.221075 | -1.613081 |
| "PL Frosted Wheat" | -0.8505 | -0.886696 | -1.03509 | -1.097546 | -0.962007 | -1.238577 | -0.846595 |


    In [37]:


``` python
pooled_flags = within_ols(train_long, ["x", *mechanics_terms, *seasonal_terms])
ols_multipliers = dict(
    zip(pooled_flags["term"].to_list(), np.exp(pooled_flags["coef"].to_numpy()), strict=True)
)
multiplier_rows = []
for site, term in [
    ("b_feat", "feature"),
    ("b_disp", "display"),
    ("b_sib_feat", "sib_feature"),
    ("b_sib_disp", "sib_display"),
]:
    draws_site = np.exp(np.asarray(posterior[site]))
    for k, product in enumerate(product_order):
        multiplier_rows.append(
            {
                "effect": term,
                "product": product,
                "ols_pooled_multiplier": float(ols_multipliers[term]),
                **interval_row(draws_site[:, k]),
            }
        )
multiplier_table = pl.DataFrame(multiplier_rows)
multiplier_table.filter(pl.col("effect").is_in(["feature", "display"]))
```


| effect | product | ols_pooled_multiplier | median | hdi50_lower | hdi50_upper | hdi94_lower | hdi94_upper |
|----|----|----|----|----|----|----|----|
| "feature" | "HNC" | 1.654384 | 2.134438 | 2.075253 | 2.191578 | 1.9667 | 2.304996 |
| "feature" | "Cheerios 12oz" | 1.654384 | 2.223584 | 2.117718 | 2.301848 | 1.961222 | 2.473493 |
| "feature" | "Cheerios 18oz" | 1.654384 | 1.108033 | 1.045284 | 1.168874 | 0.942547 | 1.289435 |
| "feature" | "Mini Wheats" | 1.654384 | 1.349659 | 1.311065 | 1.379332 | 1.256019 | 1.451221 |
| "feature" | "PL Honey Nut Oats" | 1.654384 | 1.128953 | 1.083855 | 1.170183 | 1.015752 | 1.254522 |
| "feature" | "PL Frosted Wheat" | 1.654384 | 1.26855 | 1.234142 | 1.303695 | 1.17191 | 1.36787 |
| "display" | "HNC" | 1.494406 | 1.410854 | 1.360706 | 1.460109 | 1.277333 | 1.554527 |
| "display" | "Cheerios 12oz" | 1.494406 | 1.596619 | 1.525905 | 1.639914 | 1.452584 | 1.764689 |
| "display" | "Cheerios 18oz" | 1.494406 | 1.305058 | 1.26361 | 1.35197 | 1.187889 | 1.430054 |
| "display" | "Mini Wheats" | 1.494406 | 1.328147 | 1.256049 | 1.371086 | 1.17576 | 1.491455 |
| "display" | "PL Honey Nut Oats" | 1.494406 | 1.3072 | 1.257715 | 1.352842 | 1.181313 | 1.435106 |
| "display" | "PL Frosted Wheat" | 1.494406 | 1.176384 | 1.124028 | 1.213795 | 1.058303 | 1.309277 |


    In [38]:


``` python
def plot_intervals(
    ax: Axes,
    labels: list[str],
    draws: Float[np.ndarray, " sample k"],
    color: str = "C0",
    reference: Float[np.ndarray, " k"] | None = None,
    reference_label: str = "least squares",
) -> None:
    r"""Draw posterior medians with $50\%$ (thick) and $94\%$ (thin) HDI lines, one row per label."""
    positions = np.arange(len(labels))
    for position, column in zip(positions, range(draws.shape[1]), strict=True):
        lower_94, upper_94 = hdi_bounds(draws[:, column], 0.94)
        lower_50, upper_50 = hdi_bounds(draws[:, column], 0.5)
        ax.plot([lower_94, upper_94], [position, position], color=color, linewidth=1)
        ax.plot([lower_50, upper_50], [position, position], color=color, linewidth=4)
        ax.plot(
            np.median(draws[:, column]),
            position,
            "o",
            color="white",
            markeredgecolor=color,
            markersize=6,
        )
    if reference is not None:
        ax.plot(reference, positions, "x", color="C3", markersize=8, label=reference_label)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)


interval_handles = [
    mlines.Line2D([], [], color="C0", linewidth=4, label=hdi_label(0.5)),
    mlines.Line2D([], [], color="C0", linewidth=1, label=hdi_label(0.94)),
    mlines.Line2D(
        [], [], color="C0", marker="o", markerfacecolor="white", linewidth=0, label="median"
    ),
    mlines.Line2D([], [], color="C3", marker="x", linewidth=0, label="least squares (with flags)"),
]
identifying_weeks = dict(
    zip(identification["product"].to_list(), identification["tpr_only"].to_list(), strict=True)
)
fig, axes = plt.subplots(ncols=2, figsize=(15, 6), layout="constrained")
plot_intervals(
    axes[0],
    [f"{p} ({identifying_weeks[p]} TPR-only store-weeks)" for p in product_order],
    eps_prod_draws,
    reference=own_elasticity_ols.filter(pl.col("product") != "pooled")["with_flags"].to_numpy(),
)
axes[0].set(
    title="Own promotional elasticity by product",
    xlabel="elasticity (log units per log price ratio)",
)
axes[0].legend(handles=interval_handles, loc="lower left", fontsize=9)
mechanics_sites = [
    "b_feat",
    "b_disp",
    "b_fd",
    "b_feat_depth",
    "b_disp_depth",
    "b_sib_feat",
    "b_sib_disp",
]
hnc_mechanics_draws = np.stack(
    [np.asarray(posterior[site])[:, FOCAL_INDEX] for site in mechanics_sites], axis=1
)
hnc_ols_terms = dict(
    zip(hnc_full_ols["term"].to_list(), hnc_full_ols["coef"].to_list(), strict=True)
)
plot_intervals(
    axes[1],
    mechanics_sites,
    hnc_mechanics_draws,
    reference=np.array(
        [
            hnc_ols_terms[t]
            for t in [
                "feature",
                "display",
                "feature_display",
                "feature_lam",
                "display_lam",
                "sib_feature",
                "sib_display",
            ]
        ]
    ),
)
axes[1].set(title=f"Mechanics effects of {FOCAL}", xlabel="effect on log units")
axes[1].legend(handles=interval_handles, loc="lower right", fontsize=9)
fig.suptitle("Posterior elasticities and mechanics effects", fontsize=16, fontweight="bold");
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-39-output-1.png" class="figure-img" width="1511" height="611" /></p>
</figure>


    In [39]:


``` python
gamma_draws = np.asarray(posterior["gamma"])  # (sample, competitor, product)
gamma_mean = gamma_draws.mean(axis=0)
gamma_positive = (gamma_draws > 0).mean(axis=0)
fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
image = ax.imshow(gamma_mean.T, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
for p in range(n_products):
    for k in range(n_products):
        if k == p:
            ax.text(k, p, "own", ha="center", va="center", fontsize=9, color="gray")
        else:
            ax.text(
                k,
                p,
                f"{gamma_mean[k, p]:+.2f}\nP>0 {gamma_positive[k, p]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
ax.set_xticks(range(n_products), product_order, rotation=30, ha="right")
ax.set_yticks(range(n_products), product_order)
ax.set(xlabel="price of", ylabel="units of", title="Posterior mean cross-price elasticities")
fig.colorbar(image, ax=ax, label="cross elasticity (posterior mean)");
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-40-output-1.png" class="figure-img" width="909" height="711" /></p>
</figure>


    In [40]:


``` python
twin_index = product_order.index("PL Honey Nut Oats")
twin_cell = gamma_draws[:, FOCAL_INDEX, twin_index]
reverse_cell = gamma_draws[:, twin_index, FOCAL_INDEX]
for name, draws_cell in [
    ("HNC price on PL Honey Nut Oats units", twin_cell),
    ("PL Honey Nut Oats price on HNC units", reverse_cell),
]:
    lower_c, upper_c = hdi_bounds(draws_cell, 0.94)
    print(
        f"{name}: median {np.median(draws_cell):+.2f}, 94% HDI {lower_c:+.2f} "
        f"to {upper_c:+.2f}, P(> 0) {np.mean(draws_cell > 0):.2f}"
    )
```


    HNC price on PL Honey Nut Oats units: median +0.69, 94% HDI +0.60 to +0.79, P(> 0) 1.00
    PL Honey Nut Oats price on HNC units: median -0.07, 94% HDI -0.29 to +0.13, P(> 0) 0.26


    In [41]:


``` python
hnc_series_index = [n for n in range(n_series) if series_ids[n].endswith(f"::{FOCAL}")]
hnc_store_ids = [int(series_ids[n].split("::")[0]) for n in hnc_series_index]
eps_store_draws = np.asarray(posterior["eps"])[:, hnc_series_index]
store_ols = np.array(
    [
        float(
            within_ols(
                hnc_long.filter(pl.col("series") == f"{store}::{FOCAL}"),
                ["x", *mechanics_terms, *seasonal_terms],
            )["coef"][0]
        )
        for store in hnc_store_ids
    ]
)
store_tpr_weeks = np.array(
    [
        int(train_panel_df.filter(pl.col("series") == f"{store}::{FOCAL}")["TPR_ONLY"].sum())
        for store in hnc_store_ids
    ]
)
store_order = np.argsort(-store_tpr_weeks)
fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
plot_intervals(
    ax,
    [
        f"store {hnc_store_ids[j]} ({store_segment[hnc_store_ids[j]]}, {store_tpr_weeks[j]} TPR-only weeks)"
        for j in store_order
    ],
    eps_store_draws[:, store_order],
    reference=store_ols[store_order],
)
ax.axvline(
    float(np.median(eps_prod_draws[:, FOCAL_INDEX])),
    color="C0",
    linestyle="--",
    linewidth=1,
    label="product median",
)
ax.legend(
    handles=[
        *interval_handles,
        mlines.Line2D([], [], color="C0", linestyle="--", label="product-level median"),
    ],
    loc="lower right",
    fontsize=9,
)
ax.set(
    title=f"Store-level elasticity of {FOCAL}: partial pooling vs within-store least squares",
    xlabel="elasticity",
)
print(
    f"spread across stores: least squares sd {store_ols.std():.2f} | posterior medians sd "
    f"{np.median(eps_store_draws, axis=0).std():.2f}"
)
```


    spread across stores: least squares sd 0.37 | posterior medians sd 0.31


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-42-output-2.png" class="figure-img" width="1154" height="711" /></p>
</figure>


    In [42]:


``` python
beta_mean = np.asarray(posterior["beta_s"]).mean(axis=0)
seasonal_mean = np.asarray(fourier_full) @ beta_mean
hnc_seasonal_horizon = seasonal_mean[t_train:, FOCAL_INDEX]
peak_week = int(np.argmax(hnc_seasonal_horizon)) + 1
print(
    f"posterior-mean seasonal component of {FOCAL} peaks in horizon week {peak_week} ({holdout_weeks[peak_week - 1]})"
)
print(
    f"drift scale posterior medians across series: {np.median(np.asarray(posterior['drift_scale']), axis=0).min():.3f} "
    f"to {np.median(np.asarray(posterior['drift_scale']), axis=0).max():.3f}"
)
```


    posterior-mean seasonal component of HNC peaks in horizon week 12 (2011-12-28)
    drift scale posterior medians across series: 0.019 to 0.219


# In-sample fit and holdout forecast

We draw the in-sample posterior predictive and the holdout forecast with the realized covariates and score them with the continuous ranked probability score (CRPS), the mean absolute error, and the coverage of the central 50\\ and 94\\ intervals (central intervals, while the figures draw HDI bands). The seasonal naive comparator is the two-member ensemble of the units 52 and 104 weeks earlier, and the MASE scale is computed per product on its training block, because a pooled scale would be dominated by the high-volume products. Calibration is read the way [Gneiting and Katzfuss (2014)](https://doi.org/10.1146/annurev-statistics-062713-085831) frame it, sharpness subject to calibration, with the randomized probability integral transform (PIT) for counts of [Czado, Gneiting and Held (2009)](https://doi.org/10.1111/j.1541-0420.2009.01191.x): for a count y with predictive CDF G, u = G(y - 1) + v\\(G(y) - G(y - 1)) with v \sim \text{Uniform}(0, 1) is uniform for a calibrated forecast, U-shaped for an under-dispersed one and hump-shaped for an over-dispersed one. Two cautions: the holdout is the holiday quarter, with only two earlier Decembers in the training window; and the holdout cells of one series share a level path, so the effective sample size behind the histogram is well below the number of cells.

The holdout CRPS is 12.98 against 22.13 for the seasonal naive ensemble and the mean absolute error 17.66 against 26.57; the central 50\\ and 94\\ intervals cover 56\\ and 92\\ of the 1{,}404 holdout cells (in sample, 63\\ and 96\\). Per product the model's MASE is below the naive one and below one everywhere, with the focal product the hardest at 0.96 against 1.56 and a 94\\ coverage of 0.83 in its promotion-heavy quarter. The model beats the naive forecast in every horizon week except week 12, the week of the deepest realized cut. The PIT histogram slopes downward, with 16\\ of the cells in the lowest decile and 4\\ in the highest: the holdout forecasts run high on average, so the calibration is good but not perfect, and the effective sample size behind the histogram is far below 1{,}404.


    In [43]:


``` python
rng_key, key_in, key_fc = random.split(rng_key, 3)
pred_train = np.asarray(
    predict_in_sample(key_in, model, posterior, covariates_train, device="host"), dtype=np.float32
)
pred_test = np.asarray(
    forecast(key_fc, model, posterior, y_train, covariates, device="host"), dtype=np.float32
)
y_np = np.asarray(y, dtype=np.float32)
naive_test = np.stack(
    [y_np[t_train - 52 : t_train - 52 + HORIZON], y_np[t_train - 104 : t_train - 104 + HORIZON]]
)
print(
    f"in-sample draws {pred_train.shape} | holdout draws {pred_test.shape} | holdout cells {y_test_f.size:,}"
)


def score(
    pred: Float[np.ndarray, " sample time n_series"], truth: Float[np.ndarray, " time n_series"]
) -> dict[str, float]:
    """CRPS, MAE and central-interval coverage of an ensemble against the truth."""
    return {
        "crps": float(eval_crps(pred, truth)),
        "mae": float(eval_mae(pred, truth)),
        "coverage_50": float(eval_coverage(pred, truth, alpha=0.5)),
        "coverage_94": float(eval_coverage(pred, truth, alpha=0.94)),
    }


metrics_table = pl.DataFrame(
    [
        {"forecast": "model (train)", **score(pred_train, y_train_f)},
        {"forecast": "model (test)", **score(pred_test, y_test_f)},
        {"forecast": "seasonal naive (test)", **score(naive_test, y_test_f)},
    ]
)
metrics_table
```


    in-sample draws (4000, 143, 108) | holdout draws (4000, 13, 108) | holdout cells 1,404


| forecast                | crps      | mae       | coverage_50 | coverage_94 |
|-------------------------|-----------|-----------|-------------|-------------|
| "model (train)"         | 7.914122  | 10.870273 | 0.626198    | 0.959337    |
| "model (test)"          | 12.979872 | 17.657763 | 0.562678    | 0.915954    |
| "seasonal naive (test)" | 22.127493 | 26.566952 | 0.151709    | 0.29416     |


    In [44]:


``` python
product_rows = []
for k, product in enumerate(product_order):
    idx = np.where(series_to_product_np == k)[0]
    mase = make_mase(y_train_f[:, idx], seasonality=52)
    product_rows.append(
        {
            "product": product,
            "crps_model": float(eval_crps(pred_test[:, :, idx], y_test_f[:, idx])),
            "crps_naive": float(eval_crps(naive_test[:, :, idx], y_test_f[:, idx])),
            "mase_model": float(
                mase(jnp.asarray(pred_test[:, :, idx]), jnp.asarray(y_test_f[:, idx]))
            ),
            "mase_naive": float(
                mase(jnp.asarray(naive_test[:, :, idx]), jnp.asarray(y_test_f[:, idx]))
            ),
            "coverage_94": float(
                eval_coverage(pred_test[:, :, idx], y_test_f[:, idx], alpha=0.94)
            ),
        }
    )
per_product_table = pl.DataFrame(product_rows)
per_product_table
```


| product             | crps_model | crps_naive | mase_model | mase_naive | coverage_94 |
|---------------------|------------|------------|------------|------------|-------------|
| "HNC"               | 37.558746  | 69.972229  | 0.956127   | 1.555362   | 0.833333    |
| "Cheerios 12oz"     | 11.225743  | 21.534189  | 0.547111   | 0.976913   | 0.961538    |
| "Cheerios 18oz"     | 4.750279   | 6.376069   | 0.2552     | 0.293871   | 0.965812    |
| "Mini Wheats"       | 6.577312   | 10.685898  | 0.412927   | 0.560507   | 0.978633    |
| "PL Honey Nut Oats" | 7.769324   | 10.254274  | 0.810337   | 0.926579   | 0.884615    |
| "PL Frosted Wheat"  | 9.997833   | 13.942308  | 0.642997   | 0.811573   | 0.871795    |


    In [45]:


``` python
crps_week_model = np.array(
    [float(crps_empirical(pred_test[:, t, :], y_test_f[t, :]).mean()) for t in range(HORIZON)]
)
crps_week_naive = np.array(
    [float(crps_empirical(naive_test[:, t, :], y_test_f[t, :]).mean()) for t in range(HORIZON)]
)
cdf_at = (pred_test <= y_test_f[None]).mean(axis=0)
cdf_below = (pred_test <= y_test_f[None] - 1.0).mean(axis=0)
pit_rng = np.random.default_rng(seed=42)
pit = cdf_below + pit_rng.uniform(size=y_test_f.shape) * (cdf_at - cdf_below)
pit_counts, _ = np.histogram(pit, bins=10, range=(0.0, 1.0))
print(f"PIT bin shares: {np.round(pit_counts / pit_counts.sum(), 3)}")

fig, axes = plt.subplots(ncols=2, figsize=(14, 5), layout="constrained")
axes[0].plot(range(1, HORIZON + 1), crps_week_model, "o-", color="C0", label="model")
axes[0].plot(range(1, HORIZON + 1), crps_week_naive, "s-", color="C1", label="seasonal naive")
axes[0].legend(loc="upper left")
axes[0].set(
    xlabel="horizon week",
    ylabel="CRPS (units)",
    title="Holdout CRPS by horizon week",
    xticks=range(1, HORIZON + 1),
)
axes[1].bar(
    np.arange(0.05, 1.0, 0.1),
    pit_counts / pit_counts.sum(),
    width=0.1,
    color="C0",
    edgecolor="white",
    label="randomized PIT",
)
axes[1].axhline(0.1, color="gray", linestyle=":", label="uniform")
axes[1].legend(loc="upper right")
axes[1].set(
    xlabel="PIT value", ylabel="share of holdout cells", title="Randomized PIT histogram (holdout)"
)
fig.suptitle("Holdout accuracy and calibration", fontsize=16, fontweight="bold");
```


    PIT bin shares: [0.16  0.11  0.117 0.108 0.113 0.115 0.095 0.086 0.058 0.038]


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-46-output-2.png" class="figure-img" width="1411" height="511" /></p>
</figure>


    In [46]:


``` python
def plot_forecast_panel(
    pred_test_draws: Float[np.ndarray, " sample horizon n_series"],
    labels: list[str],
    forecast_color: str,
    forecast_label: str,
    suptitle: str,
) -> None:
    """Facet the in-sample predictive and the holdout forecast for the series in ``labels``."""
    idx = [series_ids.index(label) for label in labels]
    pc = az.plot_lm(
        predictions_to_datatree(pred_train[:, :, idx], dates_num[:t_train], labels),
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
        figure_kwargs={"figsize": (14, 2.6 * len(labels))},
    )
    train_bands = pc.viz["ci_band"]["t"].sel(series=labels[-1])
    az.plot_lm(
        predictions_to_datatree(pred_test_draws[:, :, idx], dates_num[t_train:], labels),
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
    truth_da = panel_ds["units"].sel(series=labels).assign_coords(time=dates_num).rename("t")
    x_da = xr.DataArray(dates_num, dims=["time"], coords={"time": dates_num})
    pc.map(
        az.visuals.line_xy,
        "truth",
        data=truth_da,
        x=x_da,
        ignore_aes=pc.aes_set,
        color="black",
        lw=1.2,
    )
    for label in labels:
        ax = pc.get_target("t", {"series": label})
        ax.set_title(label, fontsize=11)
        split_line = ax.axvline(split_x, color="C3", linestyle="--", linewidth=1)
        promo_span = ax.fill_between(
            dates_num,
            0,
            1,
            where=(
                (panel_ds["feature"].sel(series=label) + panel_ds["display"].sel(series=label))
                > 0.5
            )
            .to_numpy()
            .tolist(),
            transform=ax.get_xaxis_transform(),
            color="C4",
            alpha=0.2,
            linewidth=0,
            step="mid",
        )
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    test_bands = pc.viz["ci_band"]["t"].sel(series=labels[-1])
    band_handles = []
    for bands, prefix in ((train_bands, "in-sample "), (test_bands, f"{forecast_label} ")):
        for prob in (0.94, 0.5):
            band = bands.sel(prob=prob).item()
            band.set_label(hdi_label(prob, prefix=prefix))
            band_handles.append(band)
    truth_line = pc.viz["truth"]["t"].sel(series=labels[-1]).item()
    truth_line.set_label("observed units")
    split_line.set_label("train-test split")
    promo_span.set_label("feature or display week")
    fig = pc.viz["figure"].item()
    fig.legend(
        handles=[*band_handles, truth_line, split_line, promo_span],
        loc="outside lower center",
        ncols=4,
        fontsize=10,
    )
    pc.get_target("t", {"series": labels[-1]}).set_xlabel("week")
    fig.supylabel("units")
    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.02)


plot_labels = [
    f"{store}::{product}"
    for store in (focus_stores[0], focus_stores[2])
    for product in (FOCAL, "PL Honey Nut Oats")
]
plot_forecast_panel(
    pred_test,
    plot_labels,
    forecast_color="C1",
    forecast_label="holdout forecast",
    suptitle=f"In-sample fit and holdout forecast (test CRPS "
    f"{metrics_table['crps'][1]:.2f} vs naive {metrics_table['crps'][2]:.2f})",
)
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-47-output-1.png" class="figure-img" width="1411" height="1073" /></p>
</figure>


# Counterfactual promotions

A counterfactual promotion is a change of the horizon covariates, nothing else: the posterior draws stay fixed, the same PRNG key is reused, and the model is run again through NumPyro's `Predictive`. Nothing is refit. The library's [forecast](../../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast) returns the sampled units; the decision layer also needs the conditional mean \mu over the horizon and the future level innovations, so we wrap `Predictive` ourselves with the model as a static argument, the same pattern the library uses, and ask for three sites. The first thing to do with the wrapper is to check that it reproduces the [forecast](../../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast) draws of the previous section bit for bit under the same key: that validates the engine and the key discipline, not any causal claim.

A policy is a discount depth d and a mechanics m for the focal product in every panel store during one contiguous two-week event, horizon weeks 7 and 8, the realized Thanksgiving slot. Every other product sits at its base price with no promotion over the whole horizon, and the grid below is a response surface for the break-even and risk analyses, not a search space. The calendar is not a lever here for two reasons printed earlier: the post-promotion dip is economically negligible, and under a multiplicative model the timing question reduces to the seasonal peak, whose posterior-mean week was printed above. The grid runs from no cut to a 40\\ cut in steps of five points for each of the four mechanics; for the shelf-tag-only mechanics the zero-depth cell is the no-promotion baseline itself. Cells with fewer than 20 observed store-weeks within \pm 2.5 points of the depth are shaded in the figures as thin support. Under one key the future level innovations are bit-identical across policies and the conditional means agree wherever the covariates agree, while the sampled units are coupled but not identical (common random numbers), which the cells below print.

The engine check prints a maximum absolute difference of 0.0 between the wrapper's draws and the [forecast](../../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast) draws under the same key. Across two policies the future level innovations differ by 0.0, the conditional means differ by 0.0 outside the event weeks, and the focal product's event draws have a correlation of 0.99. The 36 policies take about 30 seconds on 4{,}000 draws each. The figures show the feature-with-display event lifting the focal product's zone-level units to more than three times the no-promotion level in the event weeks, and lowering the private-label twin's units in the same weeks.


    In [47]:


``` python
def _scenario_draws(
    rng_key: Array,
    model: ForecastModel,
    posterior: dict[str, Array],
    data: Array,
    covariates: Array,
) -> dict[str, Array]:
    """Return the sampled horizon units, conditional means and future innovations of one covariate tensor."""
    predictive = Predictive(
        model,
        posterior_samples=posterior,
        return_sites=["forecast", "mu_future", "drift_future"],
        parallel=True,
    )
    return predictive(rng_key, covariates, data)


scenario_draws = jax.jit(_scenario_draws, static_argnums=(1,))

engine = scenario_draws(key_fc, model, posterior, y_train, covariates)
engine_gap = float(np.abs(np.asarray(engine["forecast"], dtype=np.float32) - pred_test).max())
print(
    f"engine check: max |scenario draws - forecast() draws| under the same key = {engine_gap:.1f}"
)
```


    engine check: max |scenario draws - forecast() draws| under the same key = 0.0


    In [48]:


``` python
EVENT_OFFSETS = [6, 7]  # horizon weeks 7 and 8, one-based
event_rows = [t_train + offset for offset in EVENT_OFFSETS]
BASELINE = ("TPR-only", 0.0)


def policy_covariates(depth: float, feature_flag: float, display_flag: float) -> Array:
    """Horizon covariates of one policy: base prices and no promotion everywhere except the event.

    During the event weeks the focal product carries the cut and the mechanics, every series of
    the same store sees the focal product's price in its cross-price block, and the siblings carry
    the sibling flags. The training rows are left untouched.
    """
    tensor = np.asarray(covariates).copy()
    tensor[:, t_train:, :] = 0.0
    x_policy = float(np.log1p(-depth))
    for n in range(n_series):
        if int(series_to_product_np[n]) == FOCAL_INDEX:
            tensor[0, event_rows, n] = x_policy
            tensor[1, event_rows, n] = feature_flag
            tensor[2, event_rows, n] = display_flag
        else:
            tensor[3, event_rows, n] = feature_flag
            tensor[4, event_rows, n] = display_flag
        tensor[PRICE_BLOCK + FOCAL_INDEX, event_rows, n] = x_policy
    return jnp.asarray(tensor, dtype=jnp.float32)


policy_a = policy_covariates(0.15, *MECHANICS["feature + display"])
policy_b = policy_covariates(0.30, *MECHANICS["TPR-only"])
assert np.array_equal(np.asarray(policy_a)[:, :t_train], np.asarray(covariates)[:, :t_train])
first_store_series = list(range(n_products))
print(
    "event row of a feature + display policy at 15% off, first store (rows: inputs, columns: products):"
)
print(
    pl.DataFrame(
        {
            "input": input_names,
            **{
                product_order[j]: np.asarray(policy_a)[:, event_rows[0], j]
                for j in first_store_series
            },
        }
    )
)
```


    event row of a feature + display policy at 15% off, first store (rows: inputs, columns: products):
    ┌──────────────┬───────────┬──────────────┬──────────────┬─────────────┬─────────────┬─────────────┐
    │ input        ┆ HNC       ┆ Cheerios     ┆ Cheerios     ┆ Mini Wheats ┆ PL Honey    ┆ PL Frosted  │
    │              ┆           ┆ 12oz         ┆ 18oz         ┆             ┆ Nut Oats    ┆ Wheat       │
    ╞══════════════╪═══════════╪══════════════╪══════════════╪═════════════╪═════════════╪═════════════╡
    │ x            ┆ -0.162519 ┆ 0.0          ┆ 0.0          ┆ 0.0         ┆ 0.0         ┆ 0.0         │
    │ feature      ┆ 1.0       ┆ 0.0          ┆ 0.0          ┆ 0.0         ┆ 0.0         ┆ 0.0         │
    │ display      ┆ 1.0       ┆ 0.0          ┆ 0.0          ┆ 0.0         ┆ 0.0         ┆ 0.0         │
    │ sib_feature  ┆ 0.0       ┆ 1.0          ┆ 1.0          ┆ 1.0         ┆ 1.0         ┆ 1.0         │
    │ sib_display  ┆ 0.0       ┆ 1.0          ┆ 1.0          ┆ 1.0         ┆ 1.0         ┆ 1.0         │
    │ price HNC    ┆ -0.162519 ┆ -0.162519    ┆ -0.162519    ┆ -0.162519   ┆ -0.162519   ┆ -0.162519   │
    │ price        ┆ 0.0       ┆ 0.0          ┆ 0.0          ┆ 0.0         ┆ 0.0         ┆ 0.0         │
    │ Cheerios     ┆           ┆              ┆              ┆             ┆             ┆             │
    │ 12oz         ┆           ┆              ┆              ┆             ┆             ┆             │
    │ price        ┆ 0.0       ┆ 0.0          ┆ 0.0          ┆ 0.0         ┆ 0.0         ┆ 0.0         │
    │ Cheerios     ┆           ┆              ┆              ┆             ┆             ┆             │
    │ 18oz         ┆           ┆              ┆              ┆             ┆             ┆             │
    │ price Mini   ┆ 0.0       ┆ 0.0          ┆ 0.0          ┆ 0.0         ┆ 0.0         ┆ 0.0         │
    │ Wheats       ┆           ┆              ┆              ┆             ┆             ┆             │
    │ price PL     ┆ 0.0       ┆ 0.0          ┆ 0.0          ┆ 0.0         ┆ 0.0         ┆ 0.0         │
    │ Honey Nut    ┆           ┆              ┆              ┆             ┆             ┆             │
    │ Oats         ┆           ┆              ┆              ┆             ┆             ┆             │
    │ price PL     ┆ 0.0       ┆ 0.0          ┆ 0.0          ┆ 0.0         ┆ 0.0         ┆ 0.0         │
    │ Frosted      ┆           ┆              ┆              ┆             ┆             ┆             │
    │ Wheat        ┆           ┆              ┆              ┆             ┆             ┆             │
    └──────────────┴───────────┴──────────────┴──────────────┴─────────────┴─────────────┴─────────────┘


    In [49]:


``` python
rng_key, key_policy = random.split(rng_key)
out_a = scenario_draws(key_policy, model, posterior, y_train, policy_a)
out_b = scenario_draws(key_policy, model, posterior, y_train, policy_b)
non_event = [offset for offset in range(HORIZON) if offset not in EVENT_OFFSETS]
print(
    f"max |drift_future difference| across the two policies: "
    f"{float(np.abs(out_a['drift_future'] - out_b['drift_future']).max()):.2e}"
)
print(
    f"max |mu_future difference| outside the event weeks: "
    f"{float(np.abs(out_a['mu_future'][:, non_event] - out_b['mu_future'][:, non_event]).max()):.2e}"
)
event_a = np.asarray(out_a["forecast"])[:, EVENT_OFFSETS][:, :, hnc_series_index].ravel()
event_b = np.asarray(out_b["forecast"])[:, EVENT_OFFSETS][:, :, hnc_series_index].ravel()
print(
    f"correlation of the focal product's event draws across the two policies: {np.corrcoef(event_a, event_b)[0, 1]:.2f}"
)
```


    max |drift_future difference| across the two policies: 0.00e+00
    max |mu_future difference| outside the event weeks: 0.00e+00
    correlation of the focal product's event draws across the two policies: 0.99


    In [50]:


``` python
%%time

policies: list[tuple[str, float]] = [
    (mechanics_name, float(depth)) for mechanics_name in MECHANICS for depth in DEPTH_GRID
]
event_units_mu: dict[tuple[str, float], np.ndarray] = {}
event_units_paths: dict[tuple[str, float], np.ndarray] = {}
hnc_event_paths: dict[tuple[str, float], np.ndarray] = {}
hnc_event_mu: dict[tuple[str, float], np.ndarray] = {}
zone_paths: dict[tuple[str, float], np.ndarray] = {}
for policy in policies:
    mechanics_name, depth = policy
    out = scenario_draws(
        key_policy, model, posterior, y_train, policy_covariates(depth, *MECHANICS[mechanics_name])
    )
    mu_horizon = np.asarray(out["mu_future"], dtype=np.float32)
    y_horizon = np.asarray(out["forecast"], dtype=np.float32)
    event_units_mu[policy] = mu_horizon[:, EVENT_OFFSETS, :].sum(axis=1)
    event_units_paths[policy] = y_horizon[:, EVENT_OFFSETS, :].sum(axis=1)
    hnc_event_paths[policy] = y_horizon[:, EVENT_OFFSETS][:, :, hnc_series_index]
    hnc_event_mu[policy] = mu_horizon[:, EVENT_OFFSETS][:, :, hnc_series_index]
    zone_paths[policy] = np.stack(
        [y_horizon[:, :, series_to_product_np == k].sum(axis=2) for k in range(n_products)],
        axis=-1,
    )
print(f"{len(policies)} policies evaluated on {n_draws} posterior draws each")
```


    36 policies evaluated on 4000 posterior draws each
    CPU times: user 4min 37s, sys: 7.44 s, total: 4min 44s
    Wall time: 30.4 s


    In [51]:


``` python
def plot_zone_policies(
    policy_list: list[tuple[str, float]], products: list[str], figsize: tuple[float, float]
) -> None:
    """Zone-level weekly units over the horizon: the baseline vs each policy, one facet per product and policy."""
    labels = [
        f"{product}: {mechanics_name} at {depth:.0%}"
        for product in products
        for (mechanics_name, depth) in policy_list
    ]
    baseline_draws = np.stack(
        [
            zone_paths[BASELINE][:, :, product_order.index(product)]
            for product in products
            for _ in policy_list
        ],
        axis=-1,
    )
    policy_draws = np.stack(
        [
            zone_paths[policy][:, :, product_order.index(product)]
            for product in products
            for policy in policy_list
        ],
        axis=-1,
    )
    horizon_weeks = np.arange(1, HORIZON + 1, dtype=float)
    pc = az.plot_lm(
        predictions_to_datatree(baseline_draws, horizon_weeks, labels),
        y="obs",
        x="t",
        plot_dim="time",
        ci_kind="hdi",
        ci_prob=hdi_probs,
        smooth=False,
        col_wrap=len(policy_list),
        visuals={
            "ci_band": {"color": "C0"},
            "observed_scatter": False,
            "pe_line": False,
            "xlabel": False,
            "ylabel": False,
        },
        aes={"alpha": ["prob"]},
        alpha=hdi_alphas,
        figure_kwargs={"figsize": figsize},
    )
    baseline_bands = pc.viz["ci_band"]["t"].sel(series=labels[-1])
    az.plot_lm(
        predictions_to_datatree(policy_draws, horizon_weeks, labels),
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
    for label in labels:
        ax = pc.get_target("t", {"series": label})
        ax.set_title(label, fontsize=10)
        event_span = ax.axvspan(
            EVENT_OFFSETS[0] + 0.5, EVENT_OFFSETS[-1] + 1.5, color="gray", alpha=0.15
        )
        ax.set_xticks(range(1, HORIZON + 1, 2))
    policy_bands = pc.viz["ci_band"]["t"].sel(series=labels[-1])
    handles = []
    for bands, prefix in ((baseline_bands, "no promotion "), (policy_bands, "policy ")):
        for prob in (0.94, 0.5):
            band = bands.sel(prob=prob).item()
            band.set_label(hdi_label(prob, prefix=prefix))
            handles.append(band)
    event_span.set_label("event weeks")
    fig = pc.viz["figure"].item()
    fig.legend(handles=[*handles, event_span], loc="outside lower center", ncols=5, fontsize=10)
    for label in labels[-len(policy_list) :]:
        pc.get_target("t", {"series": label}).set_xlabel("horizon week")
    fig.supylabel(f"units per week, all {n_stores} stores")
    fig.suptitle(
        "Counterfactual promotion vs no promotion (posterior predictive)",
        fontsize=16,
        fontweight="bold",
        y=1.03,
    )
```


    In [52]:


``` python
plot_zone_policies(
    [("feature + display", 0.15)], [FOCAL, "PL Honey Nut Oats"], figsize=(14.0, 5.0)
)
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-53-output-1.png" class="figure-img" width="1411" height="528" /></p>
</figure>


    In [53]:


``` python
plot_zone_policies(
    [("TPR-only", 0.15), ("feature", 0.15), ("feature + display", 0.15)],
    [FOCAL, "PL Honey Nut Oats"],
    figsize=(16.0, 8.0),
)
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-54-output-1.png" class="figure-img" width="1611" height="837" /></p>
</figure>


# From units to profit

The economics are assumptions, stated once and printed. For product j at store s let p\_{j,s} be the base price at the last training week (base prices differ across stores, so every currency number uses the store's own), g_j the gross margin on the base price (0.28 for the national brands and 0.38 for the private label), and c\_{j,s} = (1 - g_j)\\ p\_{j,s} the unit cost. The manufacturer funds a share \alpha of the discount on every unit of the focal product \text{H} sold in a promotion week, so the promo-week unit margin is

 m\_{\text{H},s}(d) = p\_{\text{H},s}(1 - d) - c\_{\text{H},s} + \alpha\\ d\\ p\_{\text{H},s} = p\_{\text{H},s}\\\big(g\_{\text{H}} - (1 - \alpha)\\ d\big), 

while the siblings sit at their base price with margin m\_{k,s} = g_k\\ p\_{k,s}. A slot cost S_m per store-week (one number per mechanics, covering both slots under feature with display) is never assumed; it is solved for in the next section. Remarks: a lump-sum allowance instead of a per-unit one would enter as a constant, and a perishable leftover would replace the holding cost of the order section by a write-off.

The estimand of the decision layer is the expected event profit \text{E}\[\Pi(a)\] of a policy a = (d, m) under the posterior predictive, and the decision rules of the next sections compare policies on it, on its lower tail, and on the order quantity. Let r index a posterior draw of the parameters \theta and of the future level path, and let E = \\7, 8\\ be the event weeks. With \text{margin}\_{i,t}(a) = m\_{\text{H},s(i)}(d) for the focal series and m\_{k,s(i)} for the sibling series, the event profit of draw r is The no-promotion event profit over the 18 stores has a mean of 10{,}269 currency units, with a standard deviation of 225 from the parameters and the level path and of 316 once the demand noise is included.

 \Pi_r(a) = \sum\_{t \in E} \sum\_{i} \text{margin}\_{i,t}(a)\\ y\_{r,t,i}(a) - S(a), \qquad S(a) = 2 \times 18 \times S_m, 

with S = 0 in every break-even and risk computation; an assumed slot cost shifts every incremental-profit histogram by a constant, so \text{P}(\Delta\Pi \< S) can be read off the same figures. The units do not depend on \alpha, so every funding share is evaluated from the same draws: \Pi_r(a; \alpha) = A_r(a) + \alpha\\ B_r(a) with B_r(a) = d \sum\_{t \in E} \sum\_{i \in \text{H}} p\_{\text{H},s(i)}\\ y\_{r,t,i}(a) \> 0 whenever d \> 0. Profit is linear in units. So, given the parameters and the level path, the expected profit equals \Pi^\mu_r, the same contraction with the conditional means \mu in place of the sampled units. The draw average of \Pi^\mu is an unbiased estimate of \text{E}\[\Pi(a)\] with no observation-noise error; the \mu bands still carry the sampled level path. The sampled paths return in the risk and order sections, where the observation noise matters.

Single-product intuition, as a remark. With baseline event units N_0, the focal product's profit under mechanics m is \pi_m(d) = m(d)\\ N_0\\ (1 - d)^{\varepsilon_m}\\ e^{b_m} - S_m, and its derivative in d has the sign of (-g\varepsilon_m - (1 - \alpha)) + (1 - \alpha)(1 + \varepsilon_m)\\ d. For \varepsilon_m \< 0 a cut pays at the margin if and only if \varepsilon_m \< -\varepsilon^\star(\alpha) with \varepsilon^\star(\alpha) = (1 - \alpha) / g; the brand-only break-even share is \alpha^\star_m = 1 + g\\ \varepsilon_m, which reads as follows: at or below 0 the cut pays unfunded, at or above 1 no funding share makes it pay, and intervals are reported unclipped with this rule. Cannibalization raises the share to \alpha^\star\_{\text{cat},m} = \alpha^\star_m + \sum_s \sum_k m\_{k,s}\\ \tilde N\_{k,s,m}\\ \gamma\_{\text{H},k} \big/ \sum_s p\_{\text{H},s}\\ \tilde N\_{\text{H},s,m}, where \tilde N\_{j,s,m} is the expected event units of product j at store s at zero depth under mechanics m: the sibling margin dollars lost per unit of log price change, relative to the focal product's promo-week gross revenue. With \|\varepsilon_m\| \le 1 the optimum over a grid is a corner: the shallowest cell below the break-even share of the deepest cell, the deepest cell above the tangent share, and in the narrow band between them the comparison of the cells decides. With \|\varepsilon_m\| \> 1 an interior depth exists for a narrow band of funding shares. The notebook computes the shares numerically, per draw, from the stored conditional means: the marginal secant between the two shallowest cells of a mechanics (zero and five points for the flagged mechanics, no promotion and ten points for the shelf-tag-only one), and the event-level share of every grid cell against the no-promotion baseline, \alpha^{\text{ev}}\_r(a) = -A_r(a) / B_r(a) with A_r the incremental profit at \alpha = 0.


    In [54]:


``` python
is_private_label = np.array(
    [series_ids[n].split("::")[1].startswith("PL") for n in range(n_series)]
)
gross_margin = np.where(is_private_label, 0.38, 0.28)
unit_cost = (1.0 - gross_margin) * base_price
is_focal = series_to_product_np == FOCAL_INDEX
store_onehot = np.eye(n_stores)[series_to_store_np]  # (n_series, n_stores)
N_EVENT_WEEKS = len(EVENT_OFFSETS)
economics_table = pl.DataFrame(
    {
        "product": product_order,
        "gross_margin": [0.38 if p.startswith("PL") else 0.28 for p in product_order],
        "base_price_median": [
            float(np.median(base_price[series_to_product_np == k])) for k in range(n_products)
        ],
        "unit_cost_median": [
            float(np.median(unit_cost[series_to_product_np == k])) for k in range(n_products)
        ],
    }
)
economics_table
```


| product             | gross_margin | base_price_median | unit_cost_median |
|---------------------|--------------|-------------------|------------------|
| "HNC"               | 0.28         | 3.02              | 2.1744           |
| "Cheerios 12oz"     | 0.28         | 3.055             | 2.1996           |
| "Cheerios 18oz"     | 0.28         | 4.79              | 3.4488           |
| "Mini Wheats"       | 0.28         | 3.89              | 2.8008           |
| "PL Honey Nut Oats" | 0.38         | 1.895             | 1.1749           |
| "PL Frosted Wheat"  | 0.38         | 2.41              | 1.4942           |


    In [55]:


``` python
def profit_parts(
    units: Float[np.ndarray, " sample n_series"], depth: float, brand_only: bool = False
) -> tuple[Float[np.ndarray, " sample"], Float[np.ndarray, " sample"]]:
    """Event profit as ``A + alpha * B`` per draw: the part at zero funding and the funding coefficient."""
    margin_at_zero = np.where(
        is_focal,
        base_price * (gross_margin - depth),
        0.0 if brand_only else gross_margin * base_price,
    )
    part_a = units @ margin_at_zero
    part_b = depth * (units[:, is_focal] @ base_price[is_focal])
    return part_a, part_b


def profit_parts_by_store(
    units: Float[np.ndarray, " sample n_series"], depth: float, brand_only: bool = False
) -> tuple[Float[np.ndarray, " sample n_stores"], Float[np.ndarray, " sample n_stores"]]:
    """Decompose the event profit as ``A + alpha * B`` with one column per store."""
    margin_at_zero = np.where(
        is_focal,
        base_price * (gross_margin - depth),
        0.0 if brand_only else gross_margin * base_price,
    )
    part_a = (units * margin_at_zero) @ store_onehot
    part_b = depth * ((units * (is_focal * base_price)) @ store_onehot)
    return part_a, part_b


def event_profit(
    units: Float[np.ndarray, " sample n_series"],
    depth: float,
    alpha: float,
    brand_only: bool = False,
) -> Float[np.ndarray, " sample"]:
    """Event profit per draw at a funding share ``alpha`` and zero slot cost."""
    part_a, part_b = profit_parts(units, depth, brand_only=brand_only)
    return part_a + alpha * part_b


baseline_mu = event_profit(event_units_mu[BASELINE], 0.0, 0.5)
baseline_paths = event_profit(event_units_paths[BASELINE], 0.0, 0.5)
print(
    f"no-promotion event profit over {n_stores} stores: mean {baseline_mu.mean():,.0f} | "
    f"sd from parameters and level path {baseline_mu.std():,.0f} "
    f"| sd including demand noise {baseline_paths.std():,.0f}"
)
```


    no-promotion event profit over 18 stores: mean 10,269 | sd from parameters and level path 225 | sd including demand noise 316


# Break-even: who funds the discount and what is a slot worth


## The funding share

The figure shows the expected event profit of the category against the depth of the focal product's cut, one row per mechanics and one column per funding share: the nominal 0.5, the posterior median of the category break-even share of the feature-with-display mechanics, and that share plus 0.15. The bands are the 50\\ and 94\\ HDIs across draws of \Pi^\mu, so they carry the parameter and level-path uncertainty and not the demand noise; shaded depths have thin support. The printed tables give, per mechanics, the probability that no promotion or the shallowest cell is the best choice at the nominal share, the range of expected profit across the grid at the break-even share, the threshold table, and the posteriors of the break-even shares themselves: brand-only, category, and the event-level share of every grid cell. A negative event-level share means the event pays even if the retailer funds the whole cut; a share above one means no funding makes it pay.

The posterior median of the category break-even share of feature with display is 0.71, which becomes \tilde\alpha; the columns of the figure are 0.50, 0.71 and 0.86. At the nominal share every curve falls with depth and the probability that no promotion or the shallowest cell is optimal is 1.00 for all four mechanics; the expected profit lost between the shallowest and the deepest cell is 17\\ of the baseline event profit for a shelf-tag cut and 40\\ for feature with display. At \tilde\alpha the curves are flat: the range of expected profit across the grid is 1.4\\ of the baseline for feature with display and 3.3\\ for a shelf-tag cut, the deepest cell is optimal in 62\\ of the draws for feature with display, and the value of perfect information about the parameters and the level path is at most 1.1\\ of the baseline. The threshold table shows the bracket: at \alpha = 0.5 the threshold elasticity is -1.79 and the posterior probability of exceeding it is 0.00 under every mechanics; at \tilde\alpha the probability is 0.84 for feature with display from the brand-only tangent but 0.50 from the category secant, the difference being the cannibalization of the siblings; at \alpha = 0.9 every probability is 1.00. The brand-only break-even shares sit between 0.68 and 0.70 for the four mechanics with 94\\ HDIs about 0.1 wide, cannibalization adds 0.03 under feature with display and 0.09 under a shelf-tag cut, and the event-level shares say what a 15\\ cut needs: under feature with display the event pays unfunded with an event-level share of -0.32, under a feature alone -0.12, while the same cut under a shelf tag needs a share of 0.79 and a 30\\ cut under feature with display needs 0.30.


    In [56]:


``` python
G_FOCAL = 0.28
feature_depth_draws = np.asarray(posterior["b_feat_depth"])[:, FOCAL_INDEX]
display_depth_draws = np.asarray(posterior["b_disp_depth"])[:, FOCAL_INDEX]
eps_focal_draws = eps_prod_draws[:, FOCAL_INDEX]


def eps_m_draws(mechanics_name: str) -> Float[np.ndarray, " sample"]:
    """Zone-level mechanics-specific elasticity draws of the focal product."""
    feature_flag, display_flag = MECHANICS[mechanics_name]
    return (
        eps_focal_draws - feature_depth_draws * feature_flag - display_depth_draws * display_flag
    )


def marginal_secant(mechanics_name: str, brand_only: bool) -> Float[np.ndarray, " sample"]:
    """Per-draw funding share at which the two shallowest cells of a mechanics break even."""
    if mechanics_name == "TPR-only":
        shallow, deeper = BASELINE, ("TPR-only", 0.10)
    else:
        shallow, deeper = (mechanics_name, 0.0), (mechanics_name, 0.05)
    a_shallow, b_shallow = profit_parts(event_units_mu[shallow], shallow[1], brand_only=brand_only)
    a_deeper, b_deeper = profit_parts(event_units_mu[deeper], deeper[1], brand_only=brand_only)
    return (a_shallow - a_deeper) / (b_deeper - b_shallow)


break_even_rows = []
alpha_star_cat: dict[str, np.ndarray] = {}
for mechanics_name in MECHANICS:
    brand = marginal_secant(mechanics_name, brand_only=True)
    category = marginal_secant(mechanics_name, brand_only=False)
    tangent = 1.0 + G_FOCAL * eps_m_draws(mechanics_name)
    alpha_star_cat[mechanics_name] = category
    for label, draws_share in [
        ("brand-only secant", brand),
        ("category secant", category),
        ("brand-only tangent", tangent),
    ]:
        break_even_rows.append(
            {"mechanics": mechanics_name, "share": label, **interval_row(draws_share)}
        )
    break_even_rows.append(
        {
            "mechanics": mechanics_name,
            "share": "cannibalization (category - brand)",
            **interval_row(category - brand),
        }
    )
break_even_table = pl.DataFrame(break_even_rows)
alpha_tilde = float(np.clip(np.median(alpha_star_cat["feature + display"]), 0.0, 1.0))
print(
    f"posterior median category break-even share of feature + display: "
    f"{np.median(alpha_star_cat['feature + display']):.3f} (used as alpha tilde = {alpha_tilde:.3f})"
)
break_even_table
```


    posterior median category break-even share of feature + display: 0.710 (used as alpha tilde = 0.710)


| mechanics | share | median | hdi50_lower | hdi50_upper | hdi94_lower | hdi94_upper |
|----|----|----|----|----|----|----|
| "TPR-only" | "brand-only secant" | 0.697731 | 0.685975 | 0.716452 | 0.652923 | 0.740082 |
| "TPR-only" | "category secant" | 0.791211 | 0.770848 | 0.810963 | 0.736628 | 0.847541 |
| "TPR-only" | "brand-only tangent" | 0.696646 | 0.677627 | 0.717147 | 0.642907 | 0.757387 |
| "TPR-only" | "cannibalization (category - brand)" | 0.092992 | 0.081873 | 0.106172 | 0.060157 | 0.127248 |
| "display" | "brand-only secant" | 0.687705 | 0.670633 | 0.711524 | 0.632469 | 0.744919 |
| "display" | "category secant" | 0.755676 | 0.737886 | 0.781083 | 0.69544 | 0.814686 |
| "display" | "brand-only tangent" | 0.686522 | 0.660474 | 0.708161 | 0.623516 | 0.7574 |
| "display" | "cannibalization (category - brand)" | 0.068357 | 0.059796 | 0.077863 | 0.043927 | 0.095292 |
| "feature" | "brand-only secant" | 0.687662 | 0.666153 | 0.706195 | 0.628383 | 0.743316 |
| "feature" | "category secant" | 0.733413 | 0.710544 | 0.751272 | 0.674373 | 0.791926 |
| "feature" | "brand-only tangent" | 0.686786 | 0.664477 | 0.713164 | 0.610744 | 0.748408 |
| "feature" | "cannibalization (category - brand)" | 0.046264 | 0.039313 | 0.051463 | 0.030108 | 0.064212 |
| "feature + display" | "brand-only secant" | 0.676219 | 0.65641 | 0.694266 | 0.621679 | 0.726904 |
| "feature + display" | "category secant" | 0.709733 | 0.693351 | 0.731979 | 0.655648 | 0.761192 |
| "feature + display" | "brand-only tangent" | 0.6747 | 0.648875 | 0.693648 | 0.611627 | 0.739954 |
| "feature + display" | "cannibalization (category - brand)" | 0.033439 | 0.028684 | 0.037408 | 0.021263 | 0.045689 |


    In [57]:


``` python
alpha_grid = sorted({0.0, 0.25, 0.5, alpha_tilde, 0.75, 0.9, 1.0})
threshold_rows = []
for alpha_value in alpha_grid:
    eps_star = (1.0 - alpha_value) / G_FOCAL
    row: dict[str, float | str] = {"alpha": alpha_value, "eps_star": eps_star}
    for mechanics_name in MECHANICS:
        row[f"P(eps_m < -eps*) {mechanics_name}"] = float(
            np.mean(eps_m_draws(mechanics_name) < -eps_star)
        )
        row[f"P(alpha*_cat < alpha) {mechanics_name}"] = float(
            np.mean(alpha_star_cat[mechanics_name] < alpha_value)
        )
    threshold_rows.append(row)
threshold_table = pl.DataFrame(threshold_rows)
threshold_table
```


| alpha | eps_star | P(eps_m \< -eps\*) TPR-only | P(alpha\*\_cat \< alpha) TPR-only | P(eps_m \< -eps\*) display | P(alpha\*\_cat \< alpha) display | P(eps_m \< -eps\*) feature | P(alpha\*\_cat \< alpha) feature | P(eps_m \< -eps\*) feature + display | P(alpha\*\_cat \< alpha) feature + display |
|----|----|----|----|----|----|----|----|----|----|
| 0.0 | 3.571429 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.25 | 2.678571 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.5 | 1.785714 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.709733 | 1.036669 | 0.669 | 0.0045 | 0.7465 | 0.07675 | 0.741 | 0.20975 | 0.83525 | 0.5 |
| 0.75 | 0.892857 | 0.958 | 0.08825 | 0.9645 | 0.43225 | 0.96275 | 0.7035 | 0.988 | 0.916 |
| 0.9 | 0.357143 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |


    In [58]:


``` python
def profit_curve(
    mechanics_name: str, alpha_value: float, kind: str = "mu"
) -> Float[np.ndarray, " sample n_depth"]:
    """Event profit draws of one mechanics across the depth grid at a funding share."""
    source = event_units_mu if kind == "mu" else event_units_paths
    return np.stack(
        [
            event_profit(source[(mechanics_name, float(d))], float(d), alpha_value)
            for d in DEPTH_GRID
        ],
        axis=1,
    )


alpha_columns = [0.5, alpha_tilde, float(min(1.0, alpha_tilde + 0.15))]
curve_labels = [
    f"{mechanics_name} | alpha = {alpha_value:.2f}"
    for mechanics_name in MECHANICS
    for alpha_value in alpha_columns
]
curve_draws = np.stack(
    [
        profit_curve(mechanics_name, alpha_value)
        for mechanics_name in MECHANICS
        for alpha_value in alpha_columns
    ],
    axis=-1,
)
pc = az.plot_lm(
    predictions_to_datatree(curve_draws / 1_000.0, DEPTH_GRID.astype(float), curve_labels),
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
    figure_kwargs={"figsize": (15, 13)},
)
for label, mechanics_name in zip(
    curve_labels, [m for m in MECHANICS for _ in alpha_columns], strict=True
):
    ax = pc.get_target("t", {"series": label})
    ax.set_title(label, fontsize=10)
    ax.axhline(float(baseline_mu.mean()) / 1_000.0, color="C3", linestyle="--", linewidth=1)
    for d, count in zip(DEPTH_GRID, support_counts[mechanics_name], strict=True):
        if count < 20:
            ax.axvspan(float(d) - 0.025, float(d) + 0.025, color="gray", alpha=0.15, linewidth=0)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
bands = pc.viz["ci_band"]["t"].sel(series=curve_labels[-1])
curve_handles = [bands.sel(prob=prob).item() for prob in (0.94, 0.5)]
for handle, prob in zip(curve_handles, (0.94, 0.5), strict=True):
    handle.set_label(hdi_label(prob))
mean_line = pc.viz["pe_line"]["t"].sel(series=curve_labels[-1]).item()
mean_line.set_label("expected profit")
curve_handles += [
    mean_line,
    mlines.Line2D([], [], color="C3", linestyle="--", label="no promotion"),
    mlines.Line2D(
        [],
        [],
        color="gray",
        linewidth=8,
        alpha=0.3,
        label="thin support (fewer than 20 store-weeks)",
    ),
]
fig = pc.viz["figure"].item()
fig.legend(handles=curve_handles, loc="outside lower center", ncols=5, fontsize=10)
for label in curve_labels[-3:]:
    pc.get_target("t", {"series": label}).set_xlabel("discount depth of the focal product")
fig.supylabel("expected category event profit (thousand currency units)")
fig.suptitle(
    "Expected event profit vs depth by mechanics and funding share",
    fontsize=16,
    fontweight="bold",
    y=1.02,
);
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-59-output-1.png" class="figure-img" width="1511" height="1339" /></p>
</figure>


    In [59]:


``` python
optimal_rows = []
for mechanics_name in MECHANICS:
    for alpha_value in (0.5, alpha_tilde):
        curve = profit_curve(mechanics_name, alpha_value)
        choice_set = np.concatenate(
            [baseline_mu[:, None], curve], axis=1
        )  # no promotion plus the grid
        best = choice_set.argmax(axis=1)
        regret = choice_set.max(axis=1, keepdims=True) - choice_set
        expected_regret = regret.mean(axis=0)
        grid_range = curve.mean(axis=0).max() - curve.mean(axis=0).min()
        optimal_rows.append(
            {
                "mechanics": mechanics_name,
                "alpha": alpha_value,
                "P(no promotion or shallowest optimal)": float(np.isin(best, [0, 1]).mean()),
                "P(deepest optimal)": float(np.mean(best == choice_set.shape[1] - 1)),
                "expected profit range over the grid / baseline": float(
                    grid_range / baseline_mu.mean()
                ),
                "max expected regret / baseline": float(
                    expected_regret.max() / baseline_mu.mean()
                ),
                "min expected regret (value of perfect information) / baseline": float(
                    expected_regret.min() / baseline_mu.mean()
                ),
            }
        )
optimal_table = pl.DataFrame(optimal_rows)
optimal_table
```


| mechanics | alpha | P(no promotion or shallowest optimal) | P(deepest optimal) | expected profit range over the grid / baseline | max expected regret / baseline | min expected regret (value of perfect information) / baseline |
|----|----|----|----|----|----|----|
| "TPR-only" | 0.5 | 1.0 | 0.0 | 0.168422 | 0.168422 | 0.0 |
| "TPR-only" | 0.709733 | 0.9845 | 0.0155 | 0.03314 | 0.03323 | 0.00009 |
| "display" | 0.5 | 1.0 | 0.0 | 0.216939 | 0.216939 | 0.0 |
| "display" | 0.709733 | 0.83975 | 0.16025 | 0.022307 | 0.102488 | 0.00202 |
| "feature" | 0.5 | 1.0 | 0.0 | 0.307923 | 0.307923 | 0.0 |
| "feature" | 0.709733 | 0.65825 | 0.34175 | 0.013344 | 0.308157 | 0.00771 |
| "feature + display" | 0.5 | 1.0 | 0.0 | 0.398942 | 0.498955 | 0.0 |
| "feature + display" | 0.709733 | 0.3785 | 0.6215 | 0.013585 | 0.523756 | 0.011348 |


    In [60]:


``` python
event_rows_table = []
a_baseline, _ = profit_parts(event_units_mu[BASELINE], 0.0)
for mechanics_name in MECHANICS:
    for d in DEPTH_GRID:
        policy = (mechanics_name, float(d))
        a_policy, b_policy = profit_parts(event_units_mu[policy], float(d))
        incremental_at_zero = a_policy - a_baseline
        if d > 0:
            share = -incremental_at_zero / b_policy
            event_rows_table.append(
                {
                    "mechanics": mechanics_name,
                    "depth": float(d),
                    "P(pays unfunded)": float(np.mean(incremental_at_zero > 0)),
                    **interval_row(share),
                }
            )
        else:
            event_rows_table.append(
                {
                    "mechanics": mechanics_name,
                    "depth": float(d),
                    "P(pays unfunded)": float(np.mean(incremental_at_zero > 0)),
                    "median": float("nan"),
                    "hdi50_lower": float("nan"),
                    "hdi50_upper": float("nan"),
                    "hdi94_lower": float("nan"),
                    "hdi94_upper": float("nan"),
                }
            )
event_share_table = pl.DataFrame(event_rows_table)
event_share_table.filter(pl.col("depth").is_in([0.0, 0.15, 0.30]))
```


| mechanics | depth | P(pays unfunded) | median | hdi50_lower | hdi50_upper | hdi94_lower | hdi94_upper |
|----|----|----|----|----|----|----|----|
| "TPR-only" | 0.0 | 0.0 | NaN | NaN | NaN | NaN | NaN |
| "TPR-only" | 0.15 | 0.0 | 0.786378 | 0.766238 | 0.805272 | 0.733798 | 0.841932 |
| "TPR-only" | 0.3 | 0.0 | 0.771901 | 0.754395 | 0.790044 | 0.721781 | 0.820734 |
| "display" | 0.0 | 1.0 | NaN | NaN | NaN | NaN | NaN |
| "display" | 0.15 | 0.0 | 0.317723 | 0.277354 | 0.35212 | 0.214785 | 0.424787 |
| "display" | 0.3 | 0.0 | 0.567833 | 0.549616 | 0.581099 | 0.522274 | 0.611728 |
| "feature" | 0.0 | 1.0 | NaN | NaN | NaN | NaN | NaN |
| "feature" | 0.15 | 1.0 | -0.124948 | -0.145515 | -0.107673 | -0.175368 | -0.070063 |
| "feature" | 0.3 | 0.0 | 0.381597 | 0.373184 | 0.390163 | 0.358416 | 0.407376 |
| "feature + display" | 0.0 | 1.0 | NaN | NaN | NaN | NaN | NaN |
| "feature + display" | 0.15 | 1.0 | -0.319417 | -0.3343 | -0.306064 | -0.358003 | -0.278606 |
| "feature + display" | 0.3 | 0.0 | 0.297182 | 0.290768 | 0.302306 | 0.281677 | 0.313803 |


## The slot cost

A feature or a display is a slot with a cost the data do not contain, so we solve for it. The break-even slot cost of a mechanics at a depth is the incremental gross event profit it adds over a shelf-tag-only cut of the same depth, divided by the number of store-weeks it occupies; at zero depth the comparison is against no promotion at all (a pure mechanics uplift, which is not comparable to a shelf-tag cut at ten points). The forest plot shows these break-even slot costs with their HDIs at the nominal and at the break-even funding share. The table then assumes a cost per slot and store-week, the same for a feature and for a display (the pair costs twice as much), and prints the posterior probability that each mechanics is the best choice over the whole grid as that cost rises.

At a 15\\ cut and the break-even share, a display is worth 29 currency units per store-week over the same cut with a shelf tag alone (94\\ HDI 19 to 39), a feature 87 (74 to 101) and the pair 145 (127 to 169); at the nominal share the numbers are 25, 76 and 126, because the margin lost on the extra units counts against the slot. The ladder shows the switching points: with slots at 25 per store-week, feature with display is the best choice with probability 1.00 at both shares; at 50 it keeps a probability of 0.73 at the nominal share and 0.88 at the break-even share against a feature alone; at 75 the feature alone wins with probability 0.88 and 0.87; at 100 no promotion wins with probability 0.95 and 0.91.


    In [61]:


``` python
slot_rows = []
slot_draws: dict[tuple[str, float, float], np.ndarray] = {}
for mechanics_name in ["display", "feature", "feature + display"]:
    for d in (0.0, 0.15, 0.30):
        for alpha_value in (0.5, alpha_tilde):
            with_mechanics = event_profit(event_units_mu[(mechanics_name, d)], d, alpha_value)
            reference = (
                baseline_mu
                if d == 0.0
                else event_profit(event_units_mu[("TPR-only", d)], d, alpha_value)
            )
            slot_cost_draws = (with_mechanics - reference) / (N_EVENT_WEEKS * n_stores)
            slot_draws[(mechanics_name, d, alpha_value)] = slot_cost_draws
            slot_rows.append(
                {
                    "mechanics": mechanics_name,
                    "depth": d,
                    "alpha": alpha_value,
                    **interval_row(slot_cost_draws),
                }
            )
slot_table = pl.DataFrame(slot_rows)
slot_table
```


| mechanics | depth | alpha | median | hdi50_lower | hdi50_upper | hdi94_lower | hdi94_upper |
|----|----|----|----|----|----|----|----|
| "display" | 0.0 | 0.5 | 28.531165 | 24.78723 | 32.49596 | 18.104966 | 39.743591 |
| "display" | 0.0 | 0.709733 | 28.531165 | 24.78723 | 32.49596 | 18.104966 | 39.743591 |
| "display" | 0.15 | 0.5 | 25.266743 | 22.542541 | 28.61678 | 16.426147 | 33.433371 |
| "display" | 0.15 | 0.709733 | 29.363023 | 25.415184 | 32.329013 | 19.441142 | 38.823887 |
| "display" | 0.3 | 0.5 | 20.154919 | 17.476441 | 22.264246 | 13.725543 | 27.099451 |
| "display" | 0.3 | 0.709733 | 30.540859 | 26.334519 | 33.160972 | 21.113255 | 40.058977 |
| "feature" | 0.0 | 0.5 | 85.070997 | 79.702516 | 89.973642 | 71.137053 | 101.426744 |
| "feature" | 0.0 | 0.709733 | 85.070997 | 79.702516 | 89.973642 | 71.137053 | 101.426744 |
| "feature" | 0.15 | 0.5 | 75.676305 | 71.718136 | 79.93243 | 64.228481 | 87.925975 |
| "feature" | 0.15 | 0.709733 | 86.882961 | 81.75679 | 91.168002 | 74.186292 | 101.269469 |
| "feature" | 0.3 | 0.5 | 60.927176 | 57.139992 | 63.766736 | 51.529877 | 70.677255 |
| "feature" | 0.3 | 0.709733 | 89.197539 | 84.363312 | 93.978502 | 75.502399 | 103.124641 |
| "feature + display" | 0.0 | 0.5 | 141.543885 | 130.354744 | 146.554894 | 119.275345 | 164.970309 |
| "feature + display" | 0.0 | 0.709733 | 141.543885 | 130.354744 | 146.554894 | 119.275345 | 164.970309 |
| "feature + display" | 0.15 | 0.5 | 126.117806 | 118.588381 | 130.821924 | 109.87868 | 146.066976 |
| "feature + display" | 0.15 | 0.709733 | 145.271541 | 136.579965 | 150.604636 | 127.085162 | 168.720743 |
| "feature + display" | 0.3 | 0.5 | 101.830646 | 97.050444 | 107.035263 | 88.375678 | 116.700587 |
| "feature + display" | 0.3 | 0.709733 | 150.542689 | 142.473922 | 157.110973 | 130.622982 | 172.133874 |


    In [62]:


``` python
slot_labels = [
    f"{m} at {d:.0%} (alpha {a:.2f})"
    for m in ["display", "feature", "feature + display"]
    for d in (0.0, 0.15, 0.30)
    for a in (0.5, alpha_tilde)
]
slot_stack = np.stack(
    [
        slot_draws[(m, d, a)]
        for m in ["display", "feature", "feature + display"]
        for d in (0.0, 0.15, 0.30)
        for a in (0.5, alpha_tilde)
    ],
    axis=1,
)
fig, ax = plt.subplots(figsize=(11, 9), layout="constrained")
plot_intervals(ax, slot_labels, slot_stack)
ax.legend(handles=interval_handles[:3], loc="center right", fontsize=9)
ax.set(
    title="Break-even slot cost per store-week (incremental gross profit over a shelf-tag cut of the same depth)",
    xlabel="currency units per store-week",
);
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-63-output-1.png" class="figure-img" width="1315" height="911" /></p>
</figure>


    In [63]:


``` python
SLOT_LADDER = [0.0, 25.0, 50.0, 75.0, 100.0, 150.0]
slots_used = {"TPR-only": 0.0, "display": 1.0, "feature": 1.0, "feature + display": 2.0}
choice_rows = []
for alpha_value in (0.5, alpha_tilde):
    for slot_cost in SLOT_LADDER:
        stacks, names = [baseline_mu], ["no promotion"]
        for mechanics_name in MECHANICS:
            curve = (
                profit_curve(mechanics_name, alpha_value)
                - N_EVENT_WEEKS * n_stores * slots_used[mechanics_name] * slot_cost
            )
            stacks.append(curve)
            names.extend([mechanics_name] * len(DEPTH_GRID))
        choice_set = np.concatenate([stacks[0][:, None], *stacks[1:]], axis=1)
        best_names = np.asarray(names)[choice_set.argmax(axis=1)]
        row: dict[str, float | str] = {"alpha": alpha_value, "cost per slot": slot_cost}
        for name in ["no promotion", *MECHANICS]:
            row[f"P({name} optimal)"] = float(np.mean(best_names == name))
        choice_rows.append(row)
pl.DataFrame(choice_rows)
```


| alpha | cost per slot | P(no promotion optimal) | P(TPR-only optimal) | P(display optimal) | P(feature optimal) | P(feature + display optimal) |
|----|----|----|----|----|----|----|
| 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| 0.5 | 25.0 | 0.0 | 0.0 | 0.0 | 0.0005 | 0.9995 |
| 0.5 | 50.0 | 0.0 | 0.0 | 0.0 | 0.27025 | 0.72975 |
| 0.5 | 75.0 | 0.0815 | 0.0 | 0.0 | 0.87525 | 0.04325 |
| 0.5 | 100.0 | 0.95025 | 0.0 | 0.0 | 0.04975 | 0.0 |
| 0.5 | 150.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.709733 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| 0.709733 | 25.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| 0.709733 | 50.0 | 0.0 | 0.0 | 0.0 | 0.12125 | 0.87875 |
| 0.709733 | 75.0 | 0.039 | 0.0 | 0.0 | 0.87225 | 0.08875 |
| 0.709733 | 100.0 | 0.9135 | 0.01175 | 0.0 | 0.0745 | 0.00025 |
| 0.709733 | 150.0 | 0.9845 | 0.0155 | 0.0 | 0.0 | 0.0 |


## Store by store

The break-even shares are computable per store because the model has no cross-store terms, so the per-store decisions are separable. The figure shows the category break-even share of the feature-with-display mechanics per store, brand-only and with cannibalization, ordered by the number of identifying weeks of the store; the table prints, per store, the probability that a cut pays at the nominal and at the break-even funding share. Where the intervals are wide, the store's decision is genuinely uncertain, and it is the partial pooling of the store-level elasticities that keeps the intervals from being prior-dominated.

At the nominal share the probability that the cut pays is below 0.37 in every store and below 0.02 in fifteen of them; at the break-even share it ranges from 0.00 to 1.00, with the upscale stores 2513, 11993 and 2277 at 0.03 or less and the mainstream stores 19265 and 25027 above 0.99. The store-level break-even shares (brand-only medians from 0.53 to 0.87, category medians from 0.57 to 0.89) line up by segment more than by the number of identifying weeks, and their 94\\ HDIs are about twice as wide as the zone-level one.


    In [64]:


``` python
policy_shallow, policy_deeper = ("feature + display", 0.0), ("feature + display", 0.05)
a_shallow_s, b_shallow_s = profit_parts_by_store(event_units_mu[policy_shallow], 0.0)
a_deeper_s, b_deeper_s = profit_parts_by_store(event_units_mu[policy_deeper], 0.05)
store_share_cat = (a_shallow_s - a_deeper_s) / (b_deeper_s - b_shallow_s)
a_shallow_b, b_shallow_b = profit_parts_by_store(
    event_units_mu[policy_shallow], 0.0, brand_only=True
)
a_deeper_b, b_deeper_b = profit_parts_by_store(
    event_units_mu[policy_deeper], 0.05, brand_only=True
)
store_share_brand = (a_shallow_b - a_deeper_b) / (b_deeper_b - b_shallow_b)
eps_store_fd = eps_store_draws - feature_depth_draws[:, None] - display_depth_draws[:, None]
store_rows = []
for j, store in enumerate(hnc_store_ids):
    row = {
        "store": store,
        "segment": store_segment[store],
        "tpr_only_weeks": int(store_tpr_weeks[j]),
    }
    for alpha_value in (0.5, alpha_tilde):
        eps_star = (1.0 - alpha_value) / G_FOCAL
        row[f"P(cut pays | alpha={alpha_value:.2f})"] = float(
            np.mean(eps_store_fd[:, j] < -eps_star)
        )
    row["alpha*_category_median"] = float(np.median(store_share_cat[:, j]))
    row["alpha*_brand_median"] = float(np.median(store_share_brand[:, j]))
    store_rows.append(row)
store_table = pl.DataFrame(store_rows).sort("tpr_only_weeks", descending=True)
store_table
```


| store | segment | tpr_only_weeks | P(cut pays \| alpha=0.50) | P(cut pays \| alpha=0.71) | alpha\*\_category_median | alpha\*\_brand_median |
|----|----|----|----|----|----|----|
| 9825 | "MAINSTREAM" | 22 | 0.00425 | 0.9615 | 0.664335 | 0.630515 |
| 19265 | "MAINSTREAM" | 20 | 0.36025 | 1.0 | 0.571437 | 0.526222 |
| 2281 | "UPSCALE" | 19 | 0.0 | 0.606 | 0.751783 | 0.697792 |
| 25027 | "MAINSTREAM" | 17 | 0.1775 | 0.999 | 0.586807 | 0.554184 |
| 25021 | "VALUE" | 15 | 0.108 | 0.9885 | 0.64678 | 0.577455 |
| 2513 | "UPSCALE" | 14 | 0.0 | 0.0175 | 0.839358 | 0.814139 |
| 23349 | "VALUE" | 13 | 0.0065 | 0.92125 | 0.675191 | 0.640575 |
| 21479 | "VALUE" | 12 | 0.0005 | 0.7415 | 0.69259 | 0.677354 |
| 4259 | "VALUE" | 11 | 0.0075 | 0.87175 | 0.708484 | 0.646961 |
| 21237 | "MAINSTREAM" | 10 | 0.004 | 0.9525 | 0.675582 | 0.630669 |
| 6431 | "VALUE" | 10 | 0.0 | 0.277 | 0.760183 | 0.738507 |
| 11993 | "UPSCALE" | 9 | 0.0 | 0.0025 | 0.88808 | 0.867212 |
| 613 | "MAINSTREAM" | 7 | 0.0 | 0.54075 | 0.730697 | 0.705429 |
| 2277 | "UPSCALE" | 7 | 0.0 | 0.02725 | 0.832281 | 0.79901 |
| 24991 | "UPSCALE" | 7 | 0.00025 | 0.6725 | 0.720698 | 0.688759 |
| 19523 | "VALUE" | 7 | 0.012 | 0.905 | 0.658289 | 0.63379 |
| 25229 | "MAINSTREAM" | 6 | 0.0 | 0.6375 | 0.716104 | 0.692695 |
| 6179 | "UPSCALE" | 6 | 0.0 | 0.74525 | 0.718151 | 0.680101 |


    In [65]:


``` python
fig, axes = plt.subplots(ncols=2, figsize=(15, 7), sharey=True, layout="constrained")
store_labels_ordered = [
    f"store {hnc_store_ids[j]} ({store_segment[hnc_store_ids[j]]}, {store_tpr_weeks[j]} TPR-only weeks)"
    for j in store_order
]
plot_intervals(axes[0], store_labels_ordered, store_share_brand[:, store_order])
axes[0].axvline(1.0, color="gray", linestyle=":", linewidth=1)
axes[0].set(title="Brand-only share (feature + display)", xlabel="funding share", xlim=(-0.5, 1.5))
plot_intervals(axes[1], store_labels_ordered, store_share_cat[:, store_order], color="C2")
axes[1].axvline(1.0, color="gray", linestyle=":", linewidth=1)
axes[1].set(
    title="Category share (cannibalization included)", xlabel="funding share", xlim=(-0.5, 1.5)
)
axes[0].legend(handles=interval_handles[:3], loc="lower left", fontsize=9)
fig.suptitle("Break-even funding share per store", fontsize=16, fontweight="bold");
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-66-output-1.png" class="figure-img" width="1511" height="711" /></p>
</figure>


# Risk: the go/no-go

Expected profit is not the whole decision. The incremental profit of a policy against no promotion, \Delta\Pi_r(a) = \Pi_r(a) - \Pi_r(a_0), is computed here on the sampled event paths, so it carries the demand noise as well as the parameter and level-path uncertainty, draw by draw under common random numbers (the same posterior draw, the same level path, coupled demand draws). That coupling is an assumption the data cannot identify: the potential outcomes of the same week under two policies are never observed together. We report the conditional value at risk at the 10\\ level, \text{CVaR}\_{0.10}, the mean of the worst tenth of the draws ([Rockafellar and Uryasev, 2000](https://doi.org/10.21314/JOR.2000.038)), with two standard errors: an iid bootstrap over the draws and the between-chain error of the per-chain values, which respects the autocorrelation of the NUTS draws. A different-key baseline, in which the level path and the demand noise are redrawn while the parameters are shared, is printed as a sensitivity to the coupling, not as a bound. The go/no-go rule is \text{CVaR}\_{0.10}(\Delta\Pi) \ge 0: in the worst tenth of the worlds the promotion still does not lose money on average, before any slot cost. Where the expected profit is flat across depths, this is the tie-breaker. A remark: the CVaR of the level \Pi instead of the increment answers a different question, the total event risk.

At a zero slot cost every policy with a feature or a display has a positive \text{CVaR}\_{0.10} and a probability of loss of 0.00, so the go/no-go is a go at the break-even share for all of them; the shelf-tag cuts are the only policies with a negative downside. The three deepest feature-with-display cells have \text{CVaR}\_{0.10} values of 4{,}394, 4{,}403 and 4{,}402 with bootstrap standard errors of 12 to 13 and between-chain standard errors of 14 to 15, so the risk measure does not separate them either; the deepest cell has the highest expected increment, 5{,}264, and the earlier table gave it a 62\\ chance of being the best cell. The coupling matters for the downside number: the committed policy has a \text{CVaR}\_{0.10} of 4{,}365 under common random numbers and 3{,}960 with a redrawn baseline, a difference the data cannot arbitrate. The slot-cost table turns the histogram into a decision: with slots at 25 per store-week the committed event still has a probability of loss of 0.00 at both shares; at 50 the probability is 0.10 at the nominal share and 0.00 at the break-even share; at 75 it is 1.00 and 0.73.


    In [66]:


``` python
n_chains = 4


def incremental_paths(
    policy: tuple[str, float], alpha_value: float
) -> Float[np.ndarray, " sample"]:
    """Incremental event profit of a policy over no promotion, per draw, on the sampled paths."""
    return event_profit(event_units_paths[policy], policy[1], alpha_value) - event_profit(
        event_units_paths[BASELINE], 0.0, alpha_value
    )


def cvar(draws: Float[np.ndarray, " sample"], level: float = 0.10) -> float:
    """Mean of the lowest ``level`` share of the draws."""
    ordered = np.sort(draws)
    return float(ordered[: max(1, int(np.floor(level * ordered.size)))].mean())


def cvar_bootstrap_se(draws: Float[np.ndarray, " sample"], n_boot: int = 2_000) -> float:
    """Estimate the standard error of the CVaR with an iid bootstrap over the draws."""
    boot_rng = np.random.default_rng(seed=0)
    resampled = boot_rng.choice(draws, size=(n_boot, draws.size), replace=True)
    lowest = np.sort(resampled, axis=1)[:, : int(np.floor(0.10 * draws.size))]
    return float(lowest.mean(axis=1).std(ddof=1))


def cvar_chain_se(draws: Float[np.ndarray, " sample"]) -> float:
    """Estimate the standard error of the CVaR from the spread of the per-chain values."""
    if n_chains < 2:
        return float("nan")
    per_chain = np.array([cvar(chain) for chain in draws.reshape(n_chains, -1)])
    return float(per_chain.std(ddof=1) / np.sqrt(n_chains))


risk_rows = []
for policy in policies:
    if policy == BASELINE:
        continue
    delta = incremental_paths(policy, alpha_tilde)
    risk_rows.append(
        {
            "mechanics": policy[0],
            "depth": policy[1],
            "expected_increment": float(delta.mean()),
            "P(loss)": float(np.mean(delta < 0)),
            "cvar_10": cvar(delta),
        }
    )
risk_table = pl.DataFrame(risk_rows).sort("expected_increment", descending=True)
top_policies = [
    (row["mechanics"], row["depth"]) for row in risk_table.head(3).iter_rows(named=True)
]
for policy in top_policies:
    delta = incremental_paths(policy, alpha_tilde)
    print(
        f"{policy[0]} at {policy[1]:.0%}: CVaR10 {cvar(delta):,.0f} | bootstrap se {cvar_bootstrap_se(delta):,.0f} | "
        f"between-chain se {cvar_chain_se(delta):,.0f}"
    )
risk_table.head(10)
```


    feature + display at 40%: CVaR10 4,394 | bootstrap se 13 | between-chain se 14
    feature + display at 35%: CVaR10 4,403 | bootstrap se 12 | between-chain se 15
    feature + display at 30%: CVaR10 4,402 | bootstrap se 12 | between-chain se 15


| mechanics           | depth | expected_increment | P(loss) | cvar_10     |
|---------------------|-------|--------------------|---------|-------------|
| "feature + display" | 0.4   | 5264.137274        | 0.0     | 4393.760804 |
| "feature + display" | 0.35  | 5221.085987        | 0.0     | 4402.982189 |
| "feature + display" | 0.3   | 5186.875786        | 0.0     | 4401.653063 |
| "feature + display" | 0.25  | 5162.812994        | 0.0     | 4395.59269  |
| "feature + display" | 0.2   | 5145.460678        | 0.0     | 4381.875108 |
| "feature + display" | 0.15  | 5133.248211        | 0.0     | 4365.168784 |
| "feature + display" | 0.1   | 5127.001319        | 0.0     | 4343.225544 |
| "feature + display" | 0.0   | 5126.240712        | 0.0     | 4294.73945  |
| "feature + display" | 0.05  | 5125.685173        | 0.0     | 4319.798963 |
| "feature"           | 0.0   | 3088.023937        | 0.0     | 2556.063328 |


    In [67]:


``` python
committed = ("feature + display", 0.15)
rng_key, key_alt = random.split(rng_key)
out_alt = scenario_draws(
    key_alt, model, posterior, y_train, policy_covariates(0.0, *MECHANICS["TPR-only"])
)
baseline_alt_units = np.asarray(out_alt["forecast"], dtype=np.float32)[:, EVENT_OFFSETS, :].sum(
    axis=1
)
delta_common = incremental_paths(committed, alpha_tilde)
delta_alt = event_profit(event_units_paths[committed], committed[1], alpha_tilde) - event_profit(
    baseline_alt_units, 0.0, alpha_tilde
)
print(
    f"{committed[0]} at {committed[1]:.0%}, alpha {alpha_tilde:.2f}: "
    f"CVaR10 with common random numbers {cvar(delta_common):,.0f} | "
    f"with a redrawn baseline {cvar(delta_alt):,.0f} | P(loss) "
    f"{np.mean(delta_common < 0):.2f} vs {np.mean(delta_alt < 0):.2f}"
)
```


    feature + display at 15%, alpha 0.71: CVaR10 with common random numbers 4,365 | with a redrawn baseline 3,960 | P(loss) 0.00 vs 0.00


    In [68]:


``` python
go_rows = []
for slot_cost in SLOT_LADDER:
    total_slot_cost = N_EVENT_WEEKS * n_stores * slots_used[committed[0]] * slot_cost
    for alpha_value in (0.5, alpha_tilde):
        delta = incremental_paths(committed, alpha_value) - total_slot_cost
        go_rows.append(
            {
                "cost per slot": slot_cost,
                "alpha": alpha_value,
                "expected increment": float(delta.mean()),
                "P(loss)": float(np.mean(delta < 0)),
                "cvar_10": cvar(delta),
            }
        )
pl.DataFrame(go_rows)
```


| cost per slot | alpha    | expected increment | P(loss) | cvar_10      |
|---------------|----------|--------------------|---------|--------------|
| 0.0           | 0.5      | 4086.97247         | 0.0     | 3454.14883   |
| 0.0           | 0.709733 | 5133.248211        | 0.0     | 4365.168784  |
| 25.0          | 0.5      | 2286.97247         | 0.0     | 1654.14883   |
| 25.0          | 0.709733 | 3333.248211        | 0.0     | 2565.168784  |
| 50.0          | 0.5      | 486.97247          | 0.10075 | -145.85117   |
| 50.0          | 0.709733 | 1533.248211        | 0.0     | 765.168784   |
| 75.0          | 0.5      | -1313.02753        | 0.99825 | -1945.85117  |
| 75.0          | 0.709733 | -266.751789        | 0.73275 | -1034.831216 |
| 100.0         | 0.5      | -3113.02753        | 1.0     | -3745.85117  |
| 100.0         | 0.709733 | -2066.751789       | 0.99975 | -2834.831216 |
| 150.0         | 0.5      | -6713.02753        | 1.0     | -7345.85117  |
| 150.0         | 0.709733 | -5666.751789       | 1.0     | -6434.831216 |


    In [69]:


``` python
fig, axes = plt.subplots(ncols=2, figsize=(15, 6), layout="constrained")
for policy, color in zip(top_policies, ("C0", "C1", "C2"), strict=True):
    delta = incremental_paths(policy, alpha_tilde) / 1_000.0
    axes[0].hist(delta, bins=60, color=color, alpha=0.5, label=f"{policy[0]} at {policy[1]:.0%}")
    axes[0].axvline(cvar(delta), color=color, linestyle="--", linewidth=1)
axes[0].axvline(0.0, color="black", linewidth=1, label="no promotion")
axes[0].legend(loc="upper right", fontsize=9)
axes[0].set(
    xlabel="incremental event profit (thousand currency units)",
    ylabel="draws",
    title="Incremental profit of the top three policies (dashed: CVaR10)",
)
colors = dict(zip(MECHANICS, ("C0", "C1", "C2", "C3"), strict=True))
for row in risk_table.iter_rows(named=True):
    axes[1].scatter(
        row["expected_increment"] / 1_000.0,
        row["cvar_10"] / 1_000.0,
        color=colors[row["mechanics"]],
        s=30 + 200 * row["depth"],
        alpha=0.8,
    )
feasible = risk_table.filter(pl.col("cvar_10") >= 0)
best_unconstrained = risk_table.row(0, named=True)
best_feasible = (
    feasible.sort("expected_increment", descending=True).row(0, named=True)
    if feasible.height > 0
    else None
)
axes[1].axhline(0.0, color="black", linewidth=1)
axes[1].axhspan(
    0.0,
    max(0.1, float(risk_table["cvar_10"].to_numpy().max()) / 1_000.0 * 1.2),
    color="C2",
    alpha=0.08,
    label="CVaR10 >= 0",
)
axes[1].scatter(
    best_unconstrained["expected_increment"] / 1_000.0,
    best_unconstrained["cvar_10"] / 1_000.0,
    marker="*",
    s=300,
    color="black",
    label="highest expected increment",
)
if best_feasible is not None:
    axes[1].scatter(
        best_feasible["expected_increment"] / 1_000.0,
        best_feasible["cvar_10"] / 1_000.0,
        marker="D",
        s=120,
        facecolor="none",
        edgecolor="black",
        linewidth=2,
        label="best with CVaR10 >= 0",
    )
    print(
        f"best policy with CVaR10 >= 0: {best_feasible['mechanics']} at {best_feasible['depth']:.0%} "
        f"(expected increment {best_feasible['expected_increment']:,.0f})"
    )
print(
    f"highest expected increment: {best_unconstrained['mechanics']} at {best_unconstrained['depth']:.0%} "
    f"({best_unconstrained['expected_increment']:,.0f}, CVaR10 {best_unconstrained['cvar_10']:,.0f})"
)
axes[1].legend(
    handles=[
        *axes[1].get_legend_handles_labels()[0],
        *[
            mlines.Line2D([], [], color=c, marker="o", linewidth=0, label=m)
            for m, c in colors.items()
        ],
    ],
    loc="lower right",
    fontsize=9,
)
axes[1].set(
    xlabel="expected incremental profit (thousand)",
    ylabel="CVaR10 of the increment (thousand)",
    title="Expected increment vs downside per policy (dot size: depth)",
)
fig.suptitle(
    f"Risk of the promotion at a funding share of {alpha_tilde:.2f}",
    fontsize=16,
    fontweight="bold",
);
```


    best policy with CVaR10 >= 0: feature + display at 40% (expected increment 5,264)
    highest expected increment: feature + display at 40% (5,264, CVaR10 4,394)


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-70-output-2.png" class="figure-img" width="1511" height="611" /></p>
</figure>


# Point planner vs posterior planner

The expected-value planner of stochastic programming replaces the random units by their expectation and optimizes. Profit is linear in units, so that planner and the posterior planner rank every policy identically and the value of the stochastic solution for the pricing lever alone is zero, by construction. The interesting comparison is with planners that use point estimates of the parameters:

| planner | inputs | what it ignores | printed comparison |
|----|----|----|----|
| expected value | posterior expected units | the spread of the units | nothing to print: identical ranking |
| posterior-mean plug-in | parameters at their posterior means | Jensen's gap between \text{E}\[(1-d)^{\varepsilon}\] and (1-d)^{\text{E}\[\varepsilon\]} | the ratio of the two, per depth |
| store least-squares plug-in | within-store least-squares elasticity, pooled least-squares mechanics multipliers | shrinkage across stores, the selection effect of choosing on noisy estimates | store-by-store decisions and the postdecision disappointment |
| posterior | the partially pooled posterior | nothing the model does not | store-by-store decisions |

The Jensen gap is small whenever the elasticity posterior is tight, and the cell below prints it. The store least-squares planner is where the difference shows. It faces two store-level decisions about the committed event, feature with display at 15\\: whether to run the event at all against no promotion, and whether to add the cut against the same mechanics at base price. It decides each with the store's own within-store elasticity (the specification with flags, plus the focal product's least-squares depth slopes), the focal product's least-squares mechanics multipliers and the least-squares cross terms, applied to the same reference units and margins as the posterior planner. Both plans are evaluated under the posterior, stated as the evaluation measure: the posterior planner is optimal under it by construction, so the gap measures what shrinkage buys if the model is right, not an out-of-sample validation. The first decision is not a test of shrinkage, because the mechanics uplift is far larger than any elasticity error. The second one is: whether a cut pays depends on the store's elasticity against the threshold of the funding share, and the within-store estimates are noisy where the identifying weeks are few. The postdecision disappointment of [Smith and Winkler (2006)](https://doi.org/10.1287/mnsc.1050.0451), the planner's own predicted gain minus the posterior-evaluated gain over the stores it chose, is nonnegative in expectation when the planner chooses on noisy estimates, because choosing selects favorable noise; it can be negative when the point estimates are systematically pessimistic.

The Jensen ratio is at most 1.002 across the grid, so the posterior-mean plug-in and the posterior planner agree. For the decision to run the event, both planners choose all 18 stores at both shares; the least-squares planner's disappointment is negative (-722 and -701), because its own mechanics multipliers under-predict the uplift the posterior expects. For the decision to add the cut, they differ. At the nominal share the posterior planner adds the cut in no store and the least-squares planner in 3; that plan is worth -109 under the posterior, a disappointment of 221. At the break-even share the least-squares planner adds the cut in all 18 stores and the posterior planner in 9; the least-squares plan is worth 8 under the posterior against 162 for the posterior plan, and the disappointment is 862: the stores with the most extreme least-squares elasticities are the ones whose predicted gains evaporate under the posterior, the optimizer's curse in a table. The scatter shows it: at the break-even share every store sits on or below the identity line, and the store with the largest predicted gain keeps a small part of it.


    In [70]:


``` python
eps_fd_zone = eps_m_draws("feature + display")
jensen_rows = []
for d in DEPTH_GRID[1:]:
    expectation_of_power = float(np.mean((1.0 - d) ** eps_fd_zone))
    power_of_expectation = float((1.0 - d) ** np.mean(eps_fd_zone))
    jensen_rows.append(
        {
            "depth": float(d),
            "E[(1-d)^eps]": expectation_of_power,
            "(1-d)^E[eps]": power_of_expectation,
            "ratio": expectation_of_power / power_of_expectation,
        }
    )
pl.DataFrame(jensen_rows)
```


| depth | E\[(1-d)^eps\] | (1-d)^E\[eps\] | ratio    |
|-------|----------------|----------------|----------|
| 0.05  | 1.061324       | 1.061303       | 1.00002  |
| 0.1   | 1.130089       | 1.129995       | 1.000083 |
| 0.15  | 1.207693       | 1.207454       | 1.000198 |
| 0.2   | 1.295905       | 1.295421       | 1.000373 |
| 0.25  | 1.396987       | 1.39612        | 1.000621 |
| 0.3   | 1.513885       | 1.512442       | 1.000954 |
| 0.35  | 1.650501       | 1.648206       | 1.001392 |
| 0.4   | 1.812106       | 1.808564       | 1.001959 |


    In [71]:


``` python
ls_effects = dict(zip(hnc_full_ols["term"].to_list(), hnc_full_ols["coef"].to_list(), strict=True))
cross_on_focal = {
    row["units of"]: float(row[f"price of {FOCAL}"]) for row in cross_ols.iter_rows(named=True)
}
ls_eps_store = {
    store: float(store_ols[j] - ls_effects["feature_lam"] - ls_effects["display_lam"])
    for j, store in enumerate(hnc_store_ids)
}
committed_depth = committed[1]


def least_squares_units(
    reference_units: Float[np.ndarray, " n_series"], add_mechanics: bool
) -> Float[np.ndarray, " n_series"]:
    """Event units the store least-squares planner predicts for the committed policy from ``reference_units``."""
    units = np.empty(n_series)
    for n in range(n_series):
        store, product = series_ids[n].split("::")
        if product == FOCAL:
            uplift = (
                np.exp(
                    ls_effects["feature"] + ls_effects["display"] + ls_effects["feature_display"]
                )
                if add_mechanics
                else 1.0
            )
            units[n] = (
                reference_units[n] * (1.0 - committed_depth) ** ls_eps_store[int(store)] * uplift
            )
        else:
            uplift = (
                np.exp(ls_effects["sib_feature"] + ls_effects["sib_display"])
                if add_mechanics
                else 1.0
            )
            units[n] = (
                reference_units[n] * (1.0 - committed_depth) ** cross_on_focal[product] * uplift
            )
    return units


a_committed_s, b_committed_s = profit_parts_by_store(event_units_mu[committed], committed_depth)
planner_rows = []
planner_scatter: dict[float, tuple[np.ndarray, np.ndarray]] = {}
for decision_label, reference_policy, add_mechanics in [
    ("run the event vs no promotion", BASELINE, True),
    ("add the cut vs the mechanics at base price", (committed[0], 0.0), False),
]:
    reference_units_mean = event_units_mu[reference_policy].mean(axis=0)
    ls_units = least_squares_units(reference_units_mean, add_mechanics)
    a_reference_s, _ = profit_parts_by_store(event_units_mu[reference_policy], 0.0)
    a_ls, b_ls = profit_parts_by_store(ls_units[None, :], committed_depth)
    a_ls_reference, _ = profit_parts_by_store(reference_units_mean[None, :], 0.0)
    for alpha_value in (0.5, alpha_tilde):
        posterior_gain = (a_committed_s + alpha_value * b_committed_s - a_reference_s).mean(
            axis=0
        )  # (n_stores,)
        predicted_gain = (a_ls + alpha_value * b_ls - a_ls_reference)[0]
        choose_ls, choose_bayes = predicted_gain > 0, posterior_gain > 0
        planner_rows.append(
            {
                "decision": decision_label,
                "alpha": alpha_value,
                "stores chosen by least squares": int(choose_ls.sum()),
                "stores chosen by posterior": int(choose_bayes.sum()),
                "stores where they disagree": int((choose_ls != choose_bayes).sum()),
                "posterior value of least-squares plan": float((choose_ls * posterior_gain).sum()),
                "posterior value of posterior plan": float((choose_bayes * posterior_gain).sum()),
                "least-squares disappointment": float(
                    (choose_ls * (predicted_gain - posterior_gain)).sum()
                ),
            }
        )
        if not add_mechanics:
            planner_scatter[alpha_value] = (predicted_gain, posterior_gain)
pl.DataFrame(planner_rows)
```


| decision | alpha | stores chosen by least squares | stores chosen by posterior | stores where they disagree | posterior value of least-squares plan | posterior value of posterior plan | least-squares disappointment |
|----|----|----|----|----|----|----|----|
| "run the event vs no promotion" | 0.5 | 18 | 18 | 0 | 4086.068232 | 4086.068232 | -721.937819 |
| "run the event vs no promotion" | 0.709733 | 18 | 18 | 0 | 5131.847738 | 5131.847738 | -700.783752 |
| "add the cut vs the mechanics at base price" | 0.5 | 3 | 0 | 3 | -108.97745 | 0.0 | 221.142091 |
| "add the cut vs the mechanics at base price" | 0.709733 | 18 | 9 | 9 | 7.825564 | 162.024694 | 862.108039 |


    In [72]:


``` python
fig, axes = plt.subplots(ncols=2, figsize=(15, 6.5), layout="constrained")
for ax, alpha_value in zip(axes, (0.5, alpha_tilde), strict=True):
    predicted_gain, posterior_gain = planner_scatter[alpha_value]
    ax.scatter(predicted_gain, posterior_gain, color="C0", s=60, label="store")
    limit = max(np.abs(predicted_gain).max(), np.abs(posterior_gain).max()) * 1.1
    ax.plot([-limit, limit], [-limit, limit], color="gray", linestyle=":", label="identity")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set(
        xlabel="gain of the cut predicted by the store least-squares planner",
        ylabel="gain of the cut evaluated under the posterior",
        title=f"funding share {alpha_value:.2f}",
    )
axes[0].legend(loc="upper left")
fig.suptitle(
    f"Store least-squares planner vs posterior: add the {committed[1]:.0%} cut to {committed[0]}",
    fontsize=16,
    fontweight="bold",
);
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-73-output-1.png" class="figure-img" width="1511" height="661" /></p>
</figure>


# The promotion order

The funding and go/no-go questions belong to the category manager. The order belongs to the replenishment planner, for whom the promotion is committed: feature with display at a 15\\ cut, the grid cell nearest the realized Thanksgiving event, under the nominal funding share of 0.5. One order covers both event weeks with no mid-event replenishment, which is why the event demand W_r = \sum\_{t \in E} y\_{r,t,\text{H}} of a store sums two weeks of the same sampled path. The underage cost c_u is the promo-week unit margin m\_{\text{H},s}(0.15) at that share, because the allowance is earned only on units sold (the true underage cost is a little lower, since part of the stockout demand is recovered at the private-label margin, so the critical fractile below is an upper bound and the order leans high). The overage cost is a holding share \eta of the unit cost, c_o = \eta\\ c\_{\text{H},s}: cereal is shelf-stable, the leftover is carried and sold later at the regular margin, so only holding is lost, and a realistic \eta is small. The critical fractile is \kappa = c_u / (c_u + c_o) and the newsvendor profit of an order Q against a demand W is \text{prof}(Q; W) = c_u \min(W, Q) - c_o\\(Q - W)^+.

Three order rules compete. The paths rule takes Q^\star = \min\\Q : \text{P}(W \le Q) \ge \kappa\\ from the joint distribution of the event demand. The marginal rule sums the per-week quantiles, Q\_{\text{marg}} = \sum_t q\_\kappa(y_t), the shortcut of adding up per-week safety stocks. The mean rule orders the expected demand, Q\_{\text{mean}} = \text{E}\[W\], which is what a point forecast delivers. In the vocabulary of stochastic programming ([Birge and Louveaux, 2011](https://doi.org/10.1007/978-1-4614-0237-4), whose news vendor example of chapter 1 fixes it): the recourse value \text{RP} = \max_Q \text{E}\[\text{prof}(Q; W)\] is attained by Q^\star; the expected result of the mean-value solution is \text{EEV} = \text{E}\[\text{prof}(Q\_{\text{mean}}; W)\]; the value of the stochastic solution is \text{VSS} = \text{RP} - \text{EEV} \ge 0; the wait-and-see value is \text{WS} = \text{E}\[c_u W\] and \text{EVPI}\_W = \text{WS} - \text{RP} is the value of perfect information about the demand itself, a loose ceiling that includes the irreducible demand noise. A tighter ceiling is the value of perfect information about the parameters and the level path, \text{E}\_\theta\[\max_Q \text{E}\[\text{prof}(Q; W) \mid \theta\]\] - \text{RP}, which we compute with inner negative binomial draws per posterior draw; the inner maximum is optimistic by about one part in the number of inner draws, and RP is recomputed on the same inner draws so that the difference is not Monte Carlo noise. Because RP, VSS and both ceilings are maxima on the evaluation draws, we also print a split-half VSS, with the rule chosen on one half of the draws and evaluated on the other. The sweep over \eta then shows the value of the stochastic solution as a function of the cost asymmetry.

With a holding share of 0.1 the critical fractile is 0.74, the underage cost is 0.54 to 0.63 per unit across stores, and the two event weeks of a store have a correlation of 0.27 across draws (median store). The recourse value is 5{,}943; the mean order loses 110 against it, a value of the stochastic solution of 1.9\\ of RP (117 on the split-half check), and the marginal-quantile rule loses only 5: the per-week quantiles sum to an order above the joint quantile in all 18 stores, by 5 to 40 units, which barely matters at a fractile where over-ordering is cheap. The paths rule gives a fill rate of 0.94 and an expected leftover of 2{,}206 units over the 18 stores. Perfect information about the demand itself would be worth 14.7\\ of RP, perfect information about the parameters and the level path 5.6\\. The sweep confirms the shape: the value of the stochastic solution is smallest, 0.1\\ of RP, at a holding share of 0.2, where the critical fractile (0.59) sits nearest the probability that demand falls below its mean (0.55 in the median store), and it grows to 7.1\\ at a fractile of 0.93 and to 15.3\\ at 0.26; the marginal-quantile loss stays below 0.8\\ of RP everywhere.


    In [73]:


``` python
ALPHA_ORDER = 0.5
paths_committed = hnc_event_paths[committed]  # (draws, event weeks, stores)
event_demand = paths_committed.sum(axis=1)  # (draws, stores)
base_focal = base_price[hnc_series_index]
cost_focal = unit_cost[hnc_series_index]
c_u = base_focal * (G_FOCAL - (1.0 - ALPHA_ORDER) * committed_depth)
between_week_corr = np.array(
    [
        np.corrcoef(paths_committed[:, 0, j], paths_committed[:, 1, j])[0, 1]
        for j in range(n_stores)
    ]
)
print(
    f"underage cost per unit: {c_u.min():.3f} to {c_u.max():.3f} | between-week correlation "
    f"of event demand across draws, median store {np.median(between_week_corr):.2f}"
)


def integer_quantile(
    draws: Float[np.ndarray, " sample k"], kappa: float
) -> Float[np.ndarray, " k"]:
    """Smallest integer order with ``P(W <= Q) >= kappa`` under the empirical distribution of the draws."""
    return np.quantile(draws, kappa, axis=0, method="inverted_cdf")


def newsvendor_profit(
    order: np.ndarray, demand: np.ndarray, underage: np.ndarray, overage: np.ndarray
) -> np.ndarray:
    """Newsvendor profit of an order against demand draws (broadcast over leading axes)."""
    return underage * np.minimum(demand, order) - overage * np.maximum(order - demand, 0.0)


def order_rules(eta: float) -> dict[str, float | np.ndarray]:
    """Orders, expected profits, VSS and the demand-information ceiling at a holding share ``eta``."""
    c_o = eta * cost_focal
    kappa = float((c_u / (c_u + c_o))[0])
    q_paths = integer_quantile(event_demand, kappa)
    q_marginal = integer_quantile(paths_committed[:, 0, :], kappa) + integer_quantile(
        paths_committed[:, 1, :], kappa
    )
    q_mean = np.rint(event_demand.mean(axis=0))
    rp = float(newsvendor_profit(q_paths, event_demand, c_u, c_o).mean(axis=0).sum())
    eev = float(newsvendor_profit(q_mean, event_demand, c_u, c_o).mean(axis=0).sum())
    marginal_value = float(
        newsvendor_profit(q_marginal, event_demand, c_u, c_o).mean(axis=0).sum()
    )
    ws = float((c_u * event_demand).mean(axis=0).sum())
    half = event_demand.shape[0] // 2
    q_half = integer_quantile(event_demand[:half], kappa)
    q_mean_half = np.rint(event_demand[:half].mean(axis=0))
    vss_split = float(
        newsvendor_profit(q_half, event_demand[half:], c_u, c_o).mean(axis=0).sum()
        - newsvendor_profit(q_mean_half, event_demand[half:], c_u, c_o).mean(axis=0).sum()
    )
    return {
        "eta": eta,
        "kappa": kappa,
        "q_paths": q_paths,
        "q_marginal": q_marginal,
        "q_mean": q_mean,
        "RP": rp,
        "EEV": eev,
        "VSS": rp - eev,
        "VSS_split_half": vss_split,
        "marginal_rule_loss": rp - marginal_value,
        "EVPI_W": ws - rp,
        "fill_rate": float(
            np.minimum(event_demand, q_paths).mean(axis=0).sum() / event_demand.mean(axis=0).sum()
        ),
        "expected_leftover": float(np.maximum(q_paths - event_demand, 0.0).mean(axis=0).sum()),
    }


nominal_eta = 0.1
nominal = order_rules(nominal_eta)
store_orders = pl.DataFrame(
    {
        "store": hnc_store_ids,
        "expected_demand": event_demand.mean(axis=0),
        "Q_paths": nominal["q_paths"],
        "Q_marginal": nominal["q_marginal"],
        "Q_mean": nominal["q_mean"],
    }
).with_columns(marginal_minus_paths=pl.col("Q_marginal") - pl.col("Q_paths"))
print(
    f"eta {nominal_eta}: kappa {nominal['kappa']:.3f} | RP {nominal['RP']:,.0f} | EEV {nominal['EEV']:,.0f} | "
    f"VSS {nominal['VSS']:,.0f} ({nominal['VSS'] / nominal['RP']:.1%} "
    f"of RP, split-half {nominal['VSS_split_half']:,.0f}) | "
    f"marginal-rule loss {nominal['marginal_rule_loss']:,.0f} | EVPI_W "
    f"{nominal['EVPI_W']:,.0f} ({nominal['EVPI_W'] / nominal['RP']:.1%})"
)
print(
    f"fill rate {nominal['fill_rate']:.3f} | expected leftover {nominal['expected_leftover']:,.0f} units | "
    f"stores with Q_marginal >= Q_paths: "
    f"{int(np.sum(np.asarray(nominal['q_marginal']) >= np.asarray(nominal['q_paths'])))} of {n_stores}"
)
store_orders
```


    underage cost per unit: 0.535 to 0.629 | between-week correlation of event demand across draws, median store 0.27
    eta 0.1: kappa 0.740 | RP 5,943 | EEV 5,832 | VSS 110 (1.9% of RP, split-half 117) | marginal-rule loss 5 | EVPI_W 875 (14.7%)
    fill rate 0.941 | expected leftover 2,206 units | stores with Q_marginal >= Q_paths: 18 of 18


| store | expected_demand | Q_paths | Q_marginal | Q_mean | marginal_minus_paths |
|-------|-----------------|---------|------------|--------|----------------------|
| 25027 | 1314.596191     | 1519.0  | 1550.0     | 1315.0 | 31.0                 |
| 21237 | 700.506226      | 792.0   | 814.0      | 701.0  | 22.0                 |
| 25229 | 597.515991      | 670.0   | 692.0      | 598.0  | 22.0                 |
| 19265 | 601.269775      | 682.0   | 700.0      | 601.0  | 18.0                 |
| 9825  | 998.864014      | 1130.0  | 1161.0     | 999.0  | 31.0                 |
| 613   | 420.008759      | 473.0   | 486.0      | 420.0  | 13.0                 |
| 2277  | 1177.63855      | 1321.0  | 1361.0     | 1178.0 | 40.0                 |
| 24991 | 1130.042236     | 1270.0  | 1310.0     | 1130.0 | 40.0                 |
| 6179  | 554.828003      | 620.0   | 641.0      | 555.0  | 21.0                 |
| 2513  | 478.570496      | 539.0   | 554.0      | 479.0  | 15.0                 |
| 2281  | 736.377258      | 835.0   | 852.0      | 736.0  | 17.0                 |
| 11993 | 477.308258      | 538.0   | 554.0      | 477.0  | 16.0                 |
| 25021 | 297.720001      | 351.0   | 357.0      | 298.0  | 6.0                  |
| 4259  | 393.592987      | 477.0   | 482.0      | 394.0  | 5.0                  |
| 21479 | 377.081757      | 427.0   | 439.0      | 377.0  | 12.0                 |
| 23349 | 237.501495      | 276.0   | 283.0      | 238.0  | 7.0                  |
| 19523 | 424.298248      | 503.0   | 509.0      | 424.0  | 6.0                  |
| 6431  | 294.084991      | 332.0   | 343.0      | 294.0  | 11.0                 |


    In [74]:


``` python
conc_focal = np.asarray(posterior["conc"])[:, FOCAL_INDEX]
mu_committed = hnc_event_mu[committed]  # (draws, event weeks, stores)
inner_rng = np.random.default_rng(seed=7)
inner_chunks = []
for _ in range(3):
    inner = inner_rng.negative_binomial(
        conc_focal[:, None, None, None],
        (
            conc_focal[:, None, None, None]
            / (conc_focal[:, None, None, None] + mu_committed[:, None, :, :])
        ),
        size=(n_draws, 100, N_EVENT_WEEKS, n_stores),
    )
    inner_chunks.append(inner.sum(axis=2).astype(np.float32))
inner_demand = np.concatenate(inner_chunks, axis=1)  # (draws, inner, stores)
n_inner = inner_demand.shape[1]


def theta_information_value(eta: float) -> float:
    """Value of perfect information about the parameters and the level path (inner draws), as a share of RP."""
    c_o = eta * cost_focal
    kappa = float((c_u / (c_u + c_o))[0])
    q_theta = np.quantile(inner_demand, kappa, axis=1, method="inverted_cdf")  # (draws, stores)
    profit_theta = newsvendor_profit(q_theta[:, None, :], inner_demand, c_u, c_o).mean(
        axis=1
    )  # (draws, stores)
    q_paths = integer_quantile(event_demand, kappa)
    rp_inner = (
        newsvendor_profit(q_paths[None, :], inner_demand.reshape(-1, n_stores), c_u, c_o)
        .mean(axis=0)
        .sum()
    )
    return float((profit_theta.mean(axis=0).sum() - rp_inner) / rp_inner)


print(
    f"value of perfect information about parameters and level path at eta {nominal_eta}: "
    f"{theta_information_value(nominal_eta):.1%} of RP ({n_inner} inner draws)"
)
```


    value of perfect information about parameters and level path at eta 0.1: 5.6% of RP (300 inner draws)


    In [75]:


``` python
sweep_rows = []
for eta in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
    rules = order_rules(eta)
    sweep_rows.append(
        {
            "eta": eta,
            "kappa": rules["kappa"],
            "c_u / c_o": float((c_u / (eta * cost_focal))[0]),
            "VSS / RP": rules["VSS"] / rules["RP"],
            "VSS split-half / RP": rules["VSS_split_half"] / rules["RP"],
            "marginal-rule loss / RP": rules["marginal_rule_loss"] / rules["RP"],
            "EVPI_W / RP": rules["EVPI_W"] / rules["RP"],
            "EVPI_theta / RP": theta_information_value(eta),
        }
    )
sweep_table = pl.DataFrame(sweep_rows)
kappa_zero = float(np.median((event_demand <= event_demand.mean(axis=0)).mean(axis=0)))
print(
    f"P(W <= E[W]) median across stores: {kappa_zero:.3f} (the critical fractile at which the mean order is optimal)"
)
sweep_table
```


    P(W <= E[W]) median across stores: 0.546 (the critical fractile at which the mean order is optimal)


| eta | kappa | c_u / c_o | VSS / RP | VSS split-half / RP | marginal-rule loss / RP | EVPI_W / RP | EVPI_theta / RP |
|----|----|----|----|----|----|----|----|
| 0.02 | 0.934366 | 14.236111 | 0.070505 | 0.072269 | 0.001654 | 0.04964 | 0.022176 |
| 0.05 | 0.850622 | 5.694444 | 0.043379 | 0.04478 | 0.001637 | 0.094221 | 0.038452 |
| 0.1 | 0.740072 | 2.847222 | 0.018569 | 0.019769 | 0.000877 | 0.147288 | 0.056058 |
| 0.2 | 0.587393 | 1.423611 | 0.001038 | 0.001343 | 0.000033 | 0.221508 | 0.078473 |
| 0.4 | 0.415822 | 0.711806 | 0.021992 | 0.020296 | 0.001357 | 0.317079 | 0.104871 |
| 0.8 | 0.262484 | 0.355903 | 0.152743 | 0.146471 | 0.007328 | 0.431046 | 0.133662 |


    In [76]:


``` python
fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
ratio = sweep_table["c_u / c_o"].to_numpy()
ax.plot(
    ratio,
    sweep_table["VSS / RP"].to_numpy(),
    "o-",
    color="C0",
    label="value of the stochastic solution (VSS)",
)
ax.plot(
    ratio,
    sweep_table["VSS split-half / RP"].to_numpy(),
    "o--",
    color="C0",
    alpha=0.6,
    label="VSS, split-half",
)
ax.plot(
    ratio,
    sweep_table["marginal-rule loss / RP"].to_numpy(),
    "s-",
    color="C1",
    label="loss of the marginal-quantile rule",
)
ax.plot(
    ratio,
    sweep_table["EVPI_theta / RP"].to_numpy(),
    "^-",
    color="C2",
    label="perfect information about parameters and level",
)
ax.plot(
    ratio,
    sweep_table["EVPI_W / RP"].to_numpy(),
    "v-",
    color="C3",
    label="perfect information about demand",
)
for x_value, kappa in zip(ratio, sweep_table["kappa"].to_numpy(), strict=True):
    ax.annotate(
        f"$\\kappa$={kappa:.2f}",
        (x_value, 0.0),
        textcoords="offset points",
        xytext=(0, -18),
        ha="center",
        fontsize=8,
    )
ax.set_xscale("log")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
ax.legend(loc="upper left", fontsize=9)
ax.set(
    xlabel="underage cost / overage cost (log scale)",
    ylabel="share of the recourse value RP",
    title="What the joint predictive is worth for the promotion order",
);
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/cell-77-output-1.png" class="figure-img" width="1111" height="611" /></p>
</figure>


# What the posterior buys you

The four promises of the introduction, each backed by a table above.

- The right expectation. The store-level elasticities are partially pooled (a posterior-median spread of 0.31 across stores against 0.37 for within-store least squares). The store least-squares planner, which is not pooled, adds the cut in 3 stores at the nominal share where the posterior adds it in none, and in all 18 at the break-even share where the posterior adds it in 9; its predicted gains exceed the posterior-evaluated ones by 221 and 862, the postdecision disappointment of the planner table.
- A reservation share with an interval instead of an argmax. At the nominal share the posterior puts probability 1.00 on a falling profit curve for every mechanics, so the depth decision is a corner. What the posterior adds is the break-even funding share, 0.71 for feature with display including cannibalization, with a 94\\ HDI about 0.1 wide at zone level and about twice that per store, and the event-level share of every cell, negative for a 15\\ cut under a feature.
- A downside, and a tie-breaker where the objective is flat. Near the break-even share the expected profit moves by 1.4\\ of the baseline across the whole depth grid, every featured policy has a positive \text{CVaR}\_{0.10}, and the three deepest cells have downside values within their standard errors of each other; the risk table with slot costs is where the go turns into a no-go, between 50 and 75 per slot and store-week.
- An order from joint paths rather than from summed quantiles. The value of the stochastic solution is 1.9\\ of the recourse value at the nominal holding share, smallest (0.1\\) where the critical fractile meets the probability that demand falls below its mean, and up to 15\\ at low fractiles; the marginal-quantile rule over-orders in all 18 stores but loses below 1\\ everywhere, because the two event weeks are only weakly correlated (0.27) and over-ordering is cheap at high fractiles.


# Limitations

- No experiment. Every elasticity rests on the selection-on-observables assumption drawn in the causal graph; the holdout validates the forecasting engine under the realized calendar, not the counterfactual ones.
- The joint no-promotion baseline over a holiday quarter is never observed; its units rest on the log-additivity of the model.
- Common random numbers are a coupling assumption; the different-key baseline is a sensitivity, not a bound.
- Shelf capacity censors the sales in the strongest promotion weeks, which biases the feature-with-display uplift and the order quantities downward; the [censored demand example](censored_demand.md) shows the likelihood that would address it.
- The elasticity is promotional, not regular-price; base-price changes are absorbed by the level.
- The economics are assumptions: gross margins, a per-unit allowance, base prices frozen at the last training week (the brand-only break-even share is price-free; the category version depends on price ratios only), a holding-cost overage.
- The holdout is the holiday quarter with two earlier Decembers to learn from, and the holdout forecasts run high on average: the PIT histogram slopes downward, with 16\\ of the cells in the lowest decile.
- Post, Quaker, the products of other sub-categories and other retailers are omitted competitors; the manufacturer's side of the deal is outside the model (General Mills also owns two of the siblings, so part of the cannibalization is internal to it); the sibling-mechanics effects are averages over any featured sibling; part of the stockout demand of the focal product spills to the private-label twin.


# Next steps

- Make the calendar a lever: add a post-promotion term and let the seasonal profile choose the event weeks.
- Replace the average sibling-mechanics effects by per-pair terms, and give the mechanics effects a store level.
- Add the censored likelihood of the [censored demand example](censored_demand.md) for the weeks at shelf capacity.
- Run a rolling backtest with [backtest](../../../reference/evaluate.backtest.md#numpyro_forecast.evaluate.backtest) over several promotion quarters.
- Use `VISITS` and `HHS` to separate traffic from basket effects.
- Pool the orders at the distribution center and compare with the per-store orders.
- Promote the decision helpers (profit contraction, CVaR, newsvendor rules, VSS) into a package module, and let [forecast](../../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast) return extra sites such as the conditional mean.


# References

- dunnhumby. [*Source files: Breakfast at the Frat*](https://www.dunnhumby.com/source-files/).
- Ghaedrahmati, E. (2025). [*Breakfast at the Frat: A Time Series Analysis*](https://doi.org/10.6084/m9.figshare.30121060). figshare copy of the dunnhumby workbook.
- Blattberg, R. C., Briesch, R., Fox, E. J. (1995). [*How Promotions Work*](https://doi.org/10.1287/mksc.14.3.G122). Marketing Science, 14(3 supplement), G122-G132.
- van Heerde, H. J., Leeflang, P. S. H., Wittink, D. R. (2000). [*The Estimation of Pre- and Postpromotion Dips with Store-Level Scanner Data*](https://doi.org/10.1509/jmkr.37.3.383.18782). Journal of Marketing Research, 37(3), 383-395.
- Bijmolt, T. H. A., van Heerde, H. J., Pieters, R. G. M. (2005). [*New Empirical Generalizations on the Determinants of Price Elasticity*](https://doi.org/10.1509/jmkr.42.2.141.62296). Journal of Marketing Research, 42(2), 141-156.
- NumPyro documentation. [*Example: Hilbert space approximation for Gaussian processes*](https://num.pyro.ai/en/stable/examples/hsgp.html) and [*Reparameterization*](https://num.pyro.ai/en/stable/reparam.html).
- Gorinova, M. I., Moore, D., Hoffman, M. D. (2020). [*Automatic Reparameterisation of Probabilistic Programs*](https://arxiv.org/abs/1906.03028). Proceedings of the 37th International Conference on Machine Learning.
- Gneiting, T., Katzfuss, M. (2014). [*Probabilistic Forecasting*](https://doi.org/10.1146/annurev-statistics-062713-085831). Annual Review of Statistics and Its Application, 1, 125-151.
- Czado, C., Gneiting, T., Held, L. (2009). [*Predictive Model Assessment for Count Data*](https://doi.org/10.1111/j.1541-0420.2009.01191.x). Biometrics, 65(4), 1254-1261.
- Rockafellar, R. T., Uryasev, S. (2000). [*Optimization of conditional value-at-risk*](https://doi.org/10.21314/JOR.2000.038). Journal of Risk, 2(3), 21-41.
- Smith, J. E., Winkler, R. L. (2006). [*The Optimizer's Curse: Skepticism and Postdecision Surprise in Decision Analysis*](https://doi.org/10.1287/mnsc.1050.0451). Management Science, 52(3), 311-322.
- Birge, J. R., Louveaux, F. (2011). [*Introduction to Stochastic Programming*](https://doi.org/10.1007/978-1-4614-0237-4). Springer.
- Related examples: [fresh retail stockout](fresh_retail_stockout.md) (the model factory and the covariate-swap counterfactual), [availability TSB](availability_tsb.md) (scenario covariates), [censored demand](censored_demand.md) (the NUTS template and the censored likelihood).
