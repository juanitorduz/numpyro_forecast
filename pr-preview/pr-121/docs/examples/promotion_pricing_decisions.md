# From forecasts to promotion decisions


From forecasts to promotion decisions

This notebook takes a probabilistic demand model all the way to a set of promotion decisions. The data are the dunnhumby [*Breakfast at the Frat*](https://www.dunnhumby.com/source-files/) scanner panel: weekly unit sales, shelf and base prices, and the promotion mechanics (in-store circular feature, in-store display, and shelf-tag-only price cuts) for products in four categories across 77 stores over 156 weeks, read here from the [figshare copy](https://doi.org/10.6084/m9.figshare.30121060) of the workbook. We work with six cereals of one substitution group and ask the question a category manager and a replenishment planner face every quarter: what should a promotion of Honey Nut Cheerios look like, who should fund it, and how much stock should each store order for it?

**Under the retailer's margin and a nominal funding share from the manufacturer, expected profit falls with discount depth for every promotion mechanics, so a point forecast and the posterior pick the same corner of the price grid. The posterior earns its keep elsewhere: the right expectation through partial pooling across stores, a break-even funding share with a credible interval, a downside and a tie-breaker where the objective is flat, and an order quantity from joint demand paths rather than from summed marginal quantiles.**

The algebra behind the first sentence is short. With gross margin g on the base price and a manufacturer that funds a share \alpha of the discount on every unit sold in a promotion week, a deeper cut raises expected profit only if the promotional elasticity under the chosen mechanics satisfies \varepsilon_m \< -(1 - \alpha) / g. We estimate \varepsilon_m with its posterior and read the decision off that inequality, with the cannibalization of the sibling products included. The funding and go/no-go questions belong to the category manager; the order quantity belongs to the replenishment planner once the promotion is committed. We proceed in five steps. First, we read and inspect the panel, count the weeks that identify a price effect separately from the promotion mechanics, and write down the causal assumption. Second, we specify a hierarchical negative binomial demand model with own and cross price elasticities as reusable components, choose its parameterization with a short SVI pass, and fit it with NUTS. Third, we check the fit in sample and on a holdout quarter. Fourth, we forecast counterfactual promotions by editing the horizon covariates and reusing the posterior. Fifth, we turn the forecasts into the funding, mechanics, risk and order decisions.

The model uses the package's model building blocks:

- [`Horizon.from_data`](https://juanitorduz.github.io/numpyro_forecast/reference/models.Horizon.html) derives the train and forecast windows from the shapes of the covariates and the data.
- [`innovations`](https://juanitorduz.github.io/numpyro_forecast/reference/models.innovations.html) samples the weekly level innovations for every store-product series, with a separate site for the forecast horizon.
- [`predict`](https://juanitorduz.github.io/numpyro_forecast/reference/models.predict.html) attaches the negative binomial likelihood to the observed weeks and samples the horizon.

The prediction drivers and evaluation helpers do the rest: [`forecast`](https://juanitorduz.github.io/numpyro_forecast/reference/predictive.forecast.html) and [`predict_in_sample`](https://juanitorduz.github.io/numpyro_forecast/reference/predictive.predict_in_sample.html) draw the holdout and in-sample predictives, [`to_datatree`](https://juanitorduz.github.io/numpyro_forecast/reference/convert.to_datatree.html) and [`predictions_to_datatree`](https://juanitorduz.github.io/numpyro_forecast/reference/convert.predictions_to_datatree.html) export draws to ArviZ, [`fourier_features`](https://juanitorduz.github.io/numpyro_forecast/reference/features.fourier_features.html) builds the annual seasonality basis, and [`eval_crps`](https://juanitorduz.github.io/numpyro_forecast/reference/evaluate.eval_crps.html), [`eval_coverage`](https://juanitorduz.github.io/numpyro_forecast/reference/evaluate.eval_coverage.html) and [`make_mase`](https://juanitorduz.github.io/numpyro_forecast/reference/metrics.make_mase.html) score the forecasts.

Two things this notebook does not claim. Prices were never randomized, so every elasticity rests on a selection-on-observables assumption that we state explicitly and cannot test. And the holdout validates the forecasting engine under the realized promotion calendar, not the counterfactual calendars, which nobody observed.


# Prepare notebook


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
from jax.typing import ArrayLike
from jaxtyping import Float, Int
from matplotlib import ticker as mtick
from matplotlib.axes import Axes
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.reparam import LocScaleReparam
from numpyro.optim import Adam

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


``` python
N_WEEKS = transactions_raw["WEEK_END_DATE"].n_unique()
cereal_all = transactions_raw.join(products_df, on="UPC").filter(
    pl.col("CATEGORY") == "COLD CEREAL"
)


def weeks_per_series(df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Count the distinct weeks of every series keyed by ``keys``."""
    return df.group_by(keys).agg(weeks=pl.col("WEEK_END_DATE").n_unique())


completeness = (
    cereal_all.pipe(weeks_per_series, ["UPC", "STORE_NUM"])
    .group_by("UPC")
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


def keep_panel_products(df: pl.DataFrame, labels: dict[int, str]) -> pl.DataFrame:
    """Keep the labeled UPCs and add their short product label."""
    return df.filter(pl.col("UPC").is_in(list(labels))).with_columns(
        product=pl.col("UPC").replace_strict(labels, return_dtype=pl.String)
    )


def join_lookups(df: pl.DataFrame, products: pl.DataFrame, stores: pl.DataFrame) -> pl.DataFrame:
    """Join the product lookup on the UPC and the deduplicated store lookup on the store id."""
    return df.join(products, on="UPC").join(stores, left_on="STORE_NUM", right_on="STORE_ID")


def log_price_ratio() -> pl.Expr:
    """Take the log of shelf over base price, capped at one so a lagged base price is no cut."""
    return (pl.col("PRICE") / pl.col("BASE_PRICE")).clip(upper_bound=1.0).log()


def add_price_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add the discount, the log price ratio ``x`` and the log depth ``lam``."""
    return df.with_columns(
        discount=1.0 - pl.col("PRICE") / pl.col("BASE_PRICE"), x=log_price_ratio()
    ).with_columns(lam=-pl.col("x"))


def mechanics_label() -> pl.Expr:
    """Label a store-week by its mechanics from the feature, display and shelf-tag flags."""
    feature = pl.col("FEATURE") == 1
    display = pl.col("DISPLAY") == 1
    return (
        pl.when(feature & display)
        .then(pl.lit("feature + display"))
        .when(feature)
        .then(pl.lit("feature"))
        .when(display)
        .then(pl.lit("display"))
        .when(pl.col("TPR_ONLY") == 1)
        .then(pl.lit("TPR-only"))
        .otherwise(pl.lit("none"))
    )


def add_promotion_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add the series id, cut flag, mechanics label, feature-display interaction and log units."""
    return df.with_columns(
        series=pl.concat_str([pl.col("STORE_NUM"), pl.col("product")], separator="::"),
        cut=(pl.col("discount") > 0.02).cast(pl.Int64),
        mechanics=mechanics_label(),
        feature_display=pl.col("FEATURE") * pl.col("DISPLAY"),
        log_units=pl.col("UNITS").cast(pl.Float64).log(),
    )


cereal_df = (
    transactions_raw.pipe(keep_panel_products, PRODUCT_LABELS)
    .pipe(join_lookups, products_df, stores_df)
    .drop_nulls(["PRICE", "BASE_PRICE"])
    .filter(pl.col("UNITS") > 0)
    .pipe(add_price_columns)
    .pipe(add_promotion_columns)
    .sort("STORE_NUM", "product", "WEEK_END_DATE")
)
n_before = transactions_raw.pipe(keep_panel_products, PRODUCT_LABELS).height
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


``` python
def cut_depth() -> pl.Expr:
    """Clip the discount at zero, so a shelf price above the base price reads as no cut."""
    return pl.col("discount").clip(lower_bound=0.0)


def sort_by_order(df: pl.DataFrame, column: str, order: list[str]) -> pl.DataFrame:
    """Sort the rows by the position of ``column`` in ``order``."""
    position = {value: i for i, value in enumerate(order)}
    return (
        df.with_columns(order=pl.col(column).replace_strict(position)).sort("order").drop("order")
    )


mechanics_order = ["none", "TPR-only", "display", "feature", "feature + display"]
mechanics_table = (
    cereal_df.group_by("mechanics")
    .agg(
        store_weeks=pl.len(),
        share_with_cut=pl.col("cut").mean(),
        mean_depth=cut_depth().mean(),
        mean_units=pl.col("UNITS").mean(),
    )
    .pipe(sort_by_order, "mechanics", mechanics_order)
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


def annual_fourier(week: str, period: float = 52.18) -> list[pl.Expr]:
    """Build the annual Fourier terms sin1, cos1, sin2, cos2 of a week-index column."""
    terms = []
    for harmonic in (1, 2):
        angle = 2.0 * harmonic * np.pi * pl.col(week) / period
        terms += [angle.sin().alias(f"sin{harmonic}"), angle.cos().alias(f"cos{harmonic}")]
    return terms


def any_promotion() -> pl.Expr:
    """Flag a store-week with a feature, a display or a cut, as a float."""
    promoted = (pl.col("FEATURE") == 1) | (pl.col("DISPLAY") == 1) | (pl.col("cut") == 1)
    return promoted.cast(pl.Float64)


first_week = cereal_df["WEEK_END_DATE"].min()
cereal_df = cereal_df.with_columns(
    week_index=((pl.col("WEEK_END_DATE") - pl.lit(first_week)).dt.total_days() / 7.0)
).with_columns(
    *annual_fourier("week_index"), trend=pl.col("week_index") / N_WEEKS, promo=any_promotion()
)
seasonal_terms = ["sin1", "cos1", "sin2", "cos2", "trend"]
```


## Do promotions borrow from the following weeks?

A promotion that loads the pantry depresses the weeks after it, the post-promotion dip of [van Heerde, Leeflang and Wittink (2000)](https://doi.org/10.1509/jmkr.37.3.383.18782). If the dip were large, the timing of promotion weeks would matter and the decision space would include the calendar. We check it with a within-series regression of log units on the price ratio, the mechanics, and indicators for the one and two weeks after any promotion, controlling for annual seasonality and a trend. The dip is +0.8\\ (standard error 0.5\\) in the first week after a promotion and -1.6\\ (standard error 0.4\\) in the second, against feature and display effects of +0.51 and +0.46 on the log scale: statistically visible, economically negligible. The calendar is therefore not a lever in this notebook.


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
        depth=cut_depth().mean(),
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


``` python
def complete_stores_by_segment(
    df: pl.DataFrame, stores: pl.DataFrame, n_weeks: int, n_products: int
) -> pl.DataFrame:
    """Keep the stores whose series all span every week, sorted by segment and basket size."""
    return (
        df.pipe(weeks_per_series, ["STORE_NUM", "product"])
        .group_by("STORE_NUM")
        .agg(all_complete=((pl.col("weeks") == n_weeks).all()) & (pl.len() == n_products))
        .filter(pl.col("all_complete"))
        .join(
            stores.select("STORE_ID", "SEG_VALUE_NAME", "AVG_WEEKLY_BASKETS"),
            left_on="STORE_NUM",
            right_on="STORE_ID",
        )
        .sort(
            ["SEG_VALUE_NAME", "AVG_WEEKLY_BASKETS", "STORE_NUM"], descending=[False, True, False]
        )
    )


complete_stores = cereal_df.pipe(complete_stores_by_segment, stores_df, N_WEEKS, n_products)
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
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-15-output-1.png" class="figure-img" width="1211" height="791" /></p>
</figure>


# Identification: the estimand, the promotion calendar and the naive elasticity

The quantity we want is the expected weekly units of the focal product under a discount d and a mechanics m, with the other covariates at their factual values. This is a promotional elasticity: the response to a temporary cut below the base price, not the response to a change in the base price itself, which the level of each series absorbs. The regular-price elasticity would need a different design (Bijmolt, van Heerde and Pieters (2005) document that promotional elasticities exceed regular-price ones), and this notebook cannot answer regular-price questions.

Prices were not randomized. The retailer and the manufacturers set the promotion calendar through trade deals, and the same deal sets the cut, the feature and the display together. The causal graph below draws the identifying assumption: conditional on the mechanics flags, the sibling flags, the competitor prices and the seasonal and level terms, the depth of the cut is as good as random with respect to the unobserved demand shocks. The right-hand cluster shows what would break it: a demand shock (a coupon drop, a competitor's promotion in another retailer) that moves both the deal calendar and the units. We cannot test the assumption with these data; the holdout below validates the forecasting engine under the realized calendar, not the counterfactual ones.


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
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-16-output-1.svg" class="img-fluid figure-img" /></p>
</figure>


## The naive elasticity and what the flags absorb

Before the Bayesian model, three within-store least-squares regressions of log units on the log price ratio show the identification problem in numbers: alone, with the mechanics flags, the sibling flags and the other five prices, and with the extra depth slopes under feature and display. The regressions remove store fixed effects by demeaning and include annual seasonality and a trend. The pooled row stacks the six products. A fourth specification replaces the log-linear price term by depth bins, a check of the functional form the model assumes.

For the focal product the naive elasticity is -2.60 (standard error 0.06); with the flags it is -1.33, and with the depth slopes -1.08. Pooled over the six products the gap is -1.86 against -0.95: the flags absorb about half of the naive price effect, which is the trade-deal calendar at work. The focal product's full regression gives a feature multiplier of 1.99 and a display multiplier of 1.44, a feature-depth slope of +0.36 (standard error 0.14) and a display-depth slope of +0.13 (standard error 0.14). The least-squares cross terms range from -0.62 to +0.65; the largest is the response of the private-label twin to the focal product's price, +0.65. The binned specification rises with depth, from +0.08 for cuts up to 10\\ to +0.67 above 30\\, while the log-linear line predicts 0.08, 0.22, 0.38 and 0.63 at the bin centers against binned estimates of 0.08, 0.09, 0.26 and 0.67: on its own, a log-linear term overstates the response to moderate cuts. The model keeps the log-linear form, and its depth slopes under mechanics let the response of a featured cut differ from that of a shelf-tag cut, which is where the moderate cuts sit.


``` python
time_index = np.repeat(np.arange(N_WEEKS), n_series)
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
    *annual_fourier("time"),
    feature_display=pl.col("feature") * pl.col("display"),
    feature_lam=pl.col("feature") * pl.col("lam"),
    display_lam=pl.col("display") * pl.col("lam"),
    trend=pl.col("time") / N_WEEKS,
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


``` python
def depth_bins(labels: list[str], edges: list[float]) -> list[pl.Expr]:
    """Build one 0/1 column per discount bin (lo, hi], in the order of ``labels``."""
    return [
        ((pl.col("discount") > lo) & (pl.col("discount") <= hi)).cast(pl.Float64).alias(label)
        for label, lo, hi in zip(labels, edges[:-1], edges[1:], strict=True)
    ]


bin_edges = [0.02, 0.10, 0.20, 0.30, 1.0]
bin_labels = ["cut (2%, 10%]", "cut (10%, 20%]", "cut (20%, 30%]", "cut > 30%"]
binned = hnc_long.with_columns(depth_bins(bin_labels, bin_edges))
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


``` python
def tpr_weeks_spread(df: pl.DataFrame) -> pl.DataFrame:
    """Summarize the shelf-tag-only weeks per series as a min, median and max per product."""
    return (
        df.group_by("series", "product")
        .agg(tpr_weeks=pl.col("TPR_ONLY").sum())
        .group_by("product")
        .agg(
            tpr_per_series_min=pl.col("tpr_weeks").min(),
            tpr_per_series_median=pl.col("tpr_weeks").median(),
            tpr_per_series_max=pl.col("tpr_weeks").max(),
        )
    )


train_panel_df = panel_df.filter(pl.col("WEEK_END_DATE").is_in(all_weeks[:t_train].to_list()))
identification = (
    train_panel_df.group_by("product", maintain_order=True)
    .agg(
        tpr_only=pl.col("TPR_ONLY").sum(),
        feature=pl.col("FEATURE").sum(),
        display=pl.col("DISPLAY").sum(),
    )
    .join(train_panel_df.pipe(tpr_weeks_spread), on="product")
    .pipe(sort_by_order, "product", product_order)
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


``` python
DEPTH_GRID = np.round(np.arange(0.0, 0.401, 0.05), 2)
MECHANICS: dict[str, tuple[float, float]] = {
    "TPR-only": (0.0, 0.0),
    "display": (0.0, 1.0),
    "feature": (1.0, 0.0),
    "feature + display": (1.0, 1.0),
}
hnc_all_stores = cereal_df.filter(pl.col("product") == FOCAL).with_columns(depth=cut_depth())
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

We write each additive piece of \log \mu as a component that samples its own sites and returns its contribution, the pattern of the NumPyro [Hilbert space Gaussian process example](https://num.pyro.ai/en/stable/examples/hsgp.html): the model creates the plates once, samples the centering values, sums the pieces and calls [predict](../../reference/models.predict.md#numpyro_forecast.models.predict). The level innovations, the store-level elasticities and the cross terms use NumPyro's [`LocScaleReparam`](https://num.pyro.ai/en/stable/reparam.html) with a sampled centering value in \[0, 1\], the idiom of the [hierarchical forecasting example](hierarchical_forecasting_1.md): 0 is the non-centered parameterization, 1 the centered one. A reparameterization does not change the posterior, so under NUTS this value has no likelihood and cannot be learned: its posterior would be its prior and its chains would wander over the unit interval. Under variational inference the ELBO does depend on it, so the inference section learns it with a short SVI pass and hands it to NUTS as a constant, the recipe of [Gorinova, Moore and Hoffman (2020)](https://arxiv.org/abs/1906.03028). Only the 30 off-diagonal cross terms are sampled and scattered into the 6 \times 6 matrix. The Fourier basis is computed once outside the model and sliced inside it, because the cached helper must not run under a trace.


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


CENTERING_SITES = ["centered_drift", "centered_eps", "centered_gamma"]


def local_level(
    h: Horizon,
    series_plate: numpyro.plate,
    centered_drift: Float[Array, ""],
    priors: CerealPriors,
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
        # innovations as a separate site, so the forecast continues the walk. The
        # centering value is a sampled site that SVI learns and NUTS receives as a constant.
        drift = innovations(
            h,
            "drift",
            lambda: dist.Normal(0.0, drift_scale),
            reparam=LocScaleReparam(centered=centered_drift),
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
        # One centering value per reparameterized site, in [0, 1]: 0 is non-centered, 1 centered.
        centered_drift = cast("Array", numpyro.sample("centered_drift", dist.Uniform(0.0, 1.0)))
        centered_eps = cast("Array", numpyro.sample("centered_eps", dist.Uniform(0.0, 1.0)))
        centered_gamma = cast("Array", numpyro.sample("centered_gamma", dist.Uniform(0.0, 1.0)))

        eta = (
            local_level(h, series_plate, centered_drift, priors)
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

Every prior has a reason, and the table prints the 94\\ interval of each one with the quantity it implies, so the numbers quoted here come from the cells. The product elasticity prior \text{Normal}(-1.5, 1) is centered where the meta-analysis of [Bijmolt, van Heerde and Pieters (2005)](https://doi.org/10.1509/jmkr.42.2.141.62296) puts price elasticities (an average of about -2.6 across studies, with promotional elasticities above regular-price ones in magnitude), and it leaves the positive tail open so the data can reject the sign. The store deviations around the product mean have a \text{HalfNormal}(0.5) scale, deviations of up to about one unit. The weekly innovation scale of the level comes from `preliz.maxent`: we ask for a log-normal with 94\\ of its mass between weekly innovations of 1\\ and 8\\, because a wider prior lets the level absorb one-week promotion spikes (a check against the least-squares mechanics effects follows the fit). The concentration prior \text{LogNormal}(2, 1) implies, at 80 units a week, a coefficient of variation near the within-store spread of the focal product's weeks. The feature and display effects have a \text{Normal}(0.5, 0.5) prior, a median multiplier of 1.65 against the least-squares multipliers printed above, with negative values allowed. The interaction, the depth slopes and the sibling effects are centered at zero. The cross terms share a \text{HalfNormal}(0.5) scale that shrinks the 30 cells toward zero. The seasonal coefficients have a \text{Normal}(0, 0.2) prior, the initial level a \text{Normal}(3, 2) prior on the log scale, and the three centering values a \text{Uniform}(0, 1) prior, which the SVI pass of the inference section turns into a choice of parameterization. `preliz.maxent` returns \text{LogNormal}(-3.3, 0.488), a median weekly innovation of 3.7\\. The prior predictive bands cover the observed units of the focus series, and the prior implied multiplier of the focal product at a 35\\ cut under feature with display has a median of 4.5 with a 94\\ HDI from 0.3 to 27.5: wide, as a prior should be, and centered on a plausible value.


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


``` python
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), layout="constrained")
panels = ["eps_prod", "eps_scale", "drift_scale", "conc", "b_feat, b_disp", "level0"]
for ax, site in zip(axes.ravel(), panels, strict=True):
    prior_distributions[site].plot_pdf(ax=ax, legend=None, color="C0")
    ax.set(title=f"{site}: {prior_distributions[site]}", xlabel="value", ylabel="density")
fig.suptitle("Prior distributions", fontsize=16, fontweight="bold");
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-26-output-1.png" class="figure-img" width="1544" height="811" /></p>
</figure>


``` python
model = make_cereal_model(series_to_product, fourier_full, priors, n_products, n_series)
numpyro.render_model(model, model_args=(covariates_train, y_train), render_distributions=True)
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-27-output-1.svg" class="img-fluid figure-img" /></p>
</figure>


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
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-28-output-1.png" class="figure-img" width="1211" height="872" /></p>
</figure>


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


    prior implied HNC multiplier at a 35% cut with feature + display: median 4.5x, 94% HDI 0.3x to 27.5x


# Inference


## The parameterization, learned with SVI

The three centering values decide how the sampler sees the hierarchy. At 0 a site is drawn as a standard normal and shifted and scaled afterwards; at 1 it is drawn on its own scale. A reparameterization leaves the posterior unchanged, so NUTS cannot choose between them. A mean-field variational approximation can: the ELBO of an `AutoNormal` guide depends on the parameterization, because a diagonal Gaussian fits one geometry better than the other. So we run a short SVI pass first, read the fitted centering values from the guide, and hand them to NUTS through `handlers.condition`, which fixes the three sites at those values. The guide's draws are used for nothing else. The ELBO curve is the convergence check of the pass. How to read the learned values: near 0 the non-centered form fits a mean-field Gaussian best, near 1 the centered form, and small innovation scales push the drift value toward 0. The pass takes 13 seconds of wall time for 5{,}000 Adam steps, and the negative ELBO is flat over the second half of the run: the mean of the last 500 steps is 63{,}502 against 63{,}543 for steps 3{,}000 to 3{,}500. The learned values are 0.34 for the level innovations, 0.68 for the store elasticities and 0.40 for the cross terms, with 90\\ intervals of the guide from 0.33 to 0.34, from 0.66 to 0.70 and from 0.36 to 0.45: the innovations and the cross terms lean toward the non-centered form, the store elasticities toward the centered one.


``` python
%%time

guide = AutoNormal(model, init_loc_fn=init_to_median)
svi = SVI(model, guide, Adam(step_size=0.01), Trace_ELBO())
rng_key, key_svi = random.split(rng_key)
svi_result = svi.run(key_svi, 5_000, covariates_train, y_train, progress_bar=False)
svi_losses = np.asarray(jax.block_until_ready(svi_result.losses))
```


    CPU times: user 26.1 s, sys: 15.6 s, total: 41.7 s
    Wall time: 12.6 s


``` python
fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
ax.plot(svi_losses, color="C0", label="negative ELBO")
ax.set_yscale("log")
ax.legend(loc="upper right")
ax.set(title="SVI pass that learns the centering values", xlabel="step", ylabel="negative ELBO");
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-31-output-1.png" class="figure-img" width="1011" height="411" /></p>
</figure>


``` python
centering_median = guide.median(svi_result.params)
centering_quantiles = guide.quantiles(svi_result.params, [0.05, 0.95])
centering_table = pl.DataFrame(
    {
        "site": CENTERING_SITES,
        "learned centering": [round(float(centering_median[name]), 3) for name in CENTERING_SITES],
        "q05": [round(float(centering_quantiles[name][0]), 3) for name in CENTERING_SITES],
        "q95": [round(float(centering_quantiles[name][1]), 3) for name in CENTERING_SITES],
    }
)
learned_centering: dict[str, ArrayLike] = {
    name: jnp.asarray(centering_median[name], dtype=jnp.float32) for name in CENTERING_SITES
}
# NUTS samples the model with the three centering sites fixed at the learned values.
nuts_model = handlers.condition(model, data=learned_centering)
print(
    f"final loss {svi_losses[-1]:,.0f} | mean of the last 500 steps {svi_losses[-500:].mean():,.0f} "
    f"| mean of steps 3,000 to 3,500 {svi_losses[3_000:3_500].mean():,.0f}"
)
centering_table
```


    final loss 63,524 | mean of the last 500 steps 63,502 | mean of steps 3,000 to 3,500 63,543


| site             | learned centering | q05   | q95   |
|------------------|-------------------|-------|-------|
| "centered_drift" | 0.336             | 0.334 | 0.338 |
| "centered_eps"   | 0.677             | 0.655 | 0.698 |
| "centered_gamma" | 0.404             | 0.363 | 0.446 |


## Sampling with NUTS

The panel is small enough for full NUTS: four chains of 1{,}000 warmup and 1{,}000 draws each, run in parallel on four host devices, on the model conditioned on the learned centering values. Two settings matter. The initialization is `init_to_median`, because NumPyro's default uniform initialization in the unconstrained space can put the cumulative level of a series far outside the range where the negative binomial mean is finite. And we ask NUTS for the number of leapfrog steps of every iteration, which gives the tree depth: a sampler stuck at the depth cap of 10 is the sign of a badly conditioned posterior. The fit takes 8 minutes and 20 seconds of wall time, with no divergences and every iteration at tree depth 8, so the depth cap is never reached.


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
mcmc = fit_nuts(key_fit, nuts_model, y_train, covariates_train)
# The chains run asynchronously on the host devices; block on the draws so the wall time is real.
posterior = jax.block_until_ready(mcmc.get_samples())
assert not any(name in posterior for name in CENTERING_SITES), "the centering sites must be fixed"
n_draws = int(posterior["eps_prod"].shape[0])
```


    CPU times: user 56min 16s, sys: 16min 26s, total: 1h 12min 42s
    Wall time: 8min 20s


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


| tree_depth | share |
|------------|-------|
| 8          | 1.0   |


# Diagnostics

The posterior, the in-sample predictive and the holdout forecast go into one ArviZ tree with named coordinates. The convergence table lists the product-level parameters and the hyperparameters: the maximum \hat R is 1.01, on the mechanics effects of Cheerios 18 oz, the product with the fewest identifying weeks, and the smallest bulk effective sample size is 608, for the private-label twin's concentration. The centering values are constants of the NUTS run, so they do not appear in the table. The trace plots show the six hyperparameters with overlapping chains and no drift.


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
    nuts_model,
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
│       Data variables: (12/21)
│           b_disp                    (chain, draw, product) float32 96kB 0.3411 ... ...
│           b_disp_depth              (chain, draw, product) float32 96kB -0.02473 .....
│           b_fd                      (chain, draw, product) float32 96kB 0.01275 ......
│           b_feat                    (chain, draw, product) float32 96kB 0.7142 ... ...
│           b_feat_depth              (chain, draw, product) float32 96kB 0.154 ... -...
│           b_sib_disp                (chain, draw, product) float32 96kB -0.03467 .....
│           ...                        ...
│           eps_prod                  (chain, draw, product) float32 96kB -1.244 ... ...
│           eps_scale                 (chain, draw) float32 16kB 0.4144 ... 0.3397
│           gamma                     (chain, draw, competitor, product) float32 576kB ...
│           gamma_offdiag             (chain, draw, pair) float32 480kB 0.06844 ... 0...
│           gamma_offdiag_decentered  (chain, draw, pair) float32 480kB 0.1501 ... 0....
│           level0                    (chain, draw, series) float32 2MB 4.558 ... 2.814
│       Attributes:
│           created_at:                 2026-09-07T13:25:52.911307+00:00
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
│           obs      (chain, draw, time, obs_dim) int32 247MB 89 146 49 37 ... 16 5 32
│       Attributes:
│           created_at:                 2026-09-07T13:26:10.486057+00:00
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
│           created_at:                 2026-09-07T13:26:10.488747+00:00
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
│           created_at:                 2026-09-07T13:26:10.489640+00:00
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
│           obs      (chain, draw, time, obs_dim) int32 22MB 146 62 78 66 ... 13 11 4 29
│       Attributes:
│           created_at:                 2026-09-07T13:26:13.021705+00:00
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
            created_at:                 2026-09-07T13:26:13.022100+00:00
            creation_library:           ArviZ
            creation_library_version:   1.2.0
            creation_library_language:  Python
            sample_dims:                []
```


xarray.DataTree


/posterior(34)

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


Data variables: (21)


b_disp


(chain, draw, product)


float32


0.3411 0.5068 ... 0.2935 0.1676


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.3411352 , 0.5068287 , 0.26756483, 0.30726317, 0.25791156,0.17568488],[0.27132073, 0.5408569 , 0.27307275, 0.35972947, 0.24789219,0.09644578],[0.4299899 , 0.3597503 , 0.2189067 , 0.33005872, 0.25699517,0.06072594],...,[0.36074868, 0.53177154, 0.35483536, 0.3124817 , 0.27103487,0.20327726],[0.29194203, 0.49571717, 0.2754948 , 0.21490645, 0.26411578,0.1403054 ],[0.32401988, 0.54139143, 0.288775  , 0.21073158, 0.19398428,0.11857867]],[[0.34225792, 0.4819428 , 0.3401656 , 0.336099  , 0.20475289,0.09481442],[0.3367862 , 0.44354752, 0.32658905, 0.34098685, 0.23963393,0.08108804],[0.25491786, 0.45615697, 0.18806954, 0.3999946 , 0.29657543,0.22548595],...[0.34606415, 0.45932186, 0.31098098, 0.26425976, 0.28689378,0.14328109],[0.34485942, 0.50350773, 0.18576081, 0.2618001 , 0.27056763,0.10313197],[0.31627995, 0.36853576, 0.24030758, 0.37151256, 0.22102904,0.1542033 ]],[[0.2986673 , 0.4856584 , 0.29151616, 0.29874068, 0.32895342,0.25645983],[0.43458238, 0.42139536, 0.27596673, 0.24343832, 0.27283934,0.33960953],[0.39638948, 0.41722116, 0.22187741, 0.20776276, 0.29437792,0.3198834 ],...,[0.34143737, 0.41071635, 0.3184121 , 0.19042395, 0.28656077,0.19219232],[0.3559721 , 0.50467277, 0.24221294, 0.20326084, 0.21054354,0.21218823],[0.3545762 , 0.4250428 , 0.18365699, 0.2882868 , 0.2934577 ,0.16758107]]], shape=(4, 1000, 6), dtype=float32)


b_disp_depth


(chain, draw, product)


float32


-0.02473 -0.08202 ... 0.3909


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[-2.47260127e-02, -8.20153952e-02,  8.08971450e-02,4.96429592e-01, -1.19730306e+00,  1.98233098e-01],[ 1.18166223e-01,  1.00209042e-01,  8.60818848e-03,1.43732458e-01, -1.04807246e+00,  7.10331619e-01],[-2.17786074e-01,  2.13050842e-01,  2.42387310e-01,4.15994436e-01, -9.91402388e-01,  8.42787564e-01],...,[ 4.98363897e-02, -2.38941416e-01, -2.27763727e-01,-9.45098232e-03, -1.36329424e+00,  2.86598295e-01],[ 1.29933462e-01, -2.21322775e-01,  1.04475617e-02,6.88737571e-01, -8.19899440e-01,  6.94725454e-01],[ 3.07411030e-02, -1.85393468e-01, -7.02881962e-02,4.78072673e-01, -4.05719370e-01,  1.03251100e+00]],[[ 2.90080197e-02,  3.81028309e-04, -5.13531119e-02,5.24218231e-02, -1.23409021e+00,  7.78822482e-01],[ 2.49665398e-02, -1.88049637e-02, -2.26533800e-01,1.40603110e-01, -1.11710346e+00,  1.03380334e+00],[ 1.83192700e-01, -5.57038188e-02,  1.83884889e-01,1.56057939e-01, -1.07680070e+00,  1.30184203e-01],...4.96335179e-01, -1.63412738e+00,  4.78026539e-01],[ 2.05380186e-01, -1.95509374e-01,  1.74901426e-01,3.51793110e-01, -1.19717550e+00,  5.92851222e-01],[ 7.96948373e-03,  1.89819306e-01,  2.75723636e-01,1.59914996e-02, -1.04415345e+00,  5.80163479e-01]],[[ 7.96676576e-02, -8.92967880e-02, -3.15996893e-02,6.33095980e-01, -1.29899085e+00,  8.68171602e-02],[-6.95844588e-04,  7.77782723e-02,  3.84219550e-02,3.71250778e-01, -1.41650581e+00, -4.79879439e-01],[-5.93456514e-02,  1.13686430e-03,  1.62921190e-01,2.44513139e-01, -1.23543775e+00, -4.98855770e-01],...,[ 1.46734966e-02,  2.65899897e-01,  1.21653460e-01,8.79516125e-01, -8.79213452e-01,  3.16201687e-01],[ 5.17848320e-02, -4.09600791e-03,  1.71520755e-01,2.51993001e-01, -3.58248532e-01,  4.97152776e-01],[ 3.09351355e-01,  1.39900476e-01,  2.69062817e-01,5.03684580e-01, -9.53291833e-01,  3.90903980e-01]]],shape=(4, 1000, 6), dtype=float32)


b_fd


(chain, draw, product)


float32


0.01275 -0.0017 ... -0.1563 0.02909


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 0.01274526, -0.0017004 , -0.28647426, -0.14213748,-0.15150006, -0.00492395],[ 0.04002931, -0.19262786, -0.26146433,  0.04033975,-0.07240778,  0.02088093],[-0.03993816,  0.0383773 , -0.35845065, -0.00919099,-0.06890627,  0.09863719],...,[-0.09187922, -0.06873903, -0.2547005 ,  0.06339928,-0.15899025, -0.04056524],[ 0.05960783, -0.05145077, -0.31168896,  0.03423182,-0.12677273, -0.06390913],[ 0.03915649, -0.07005024, -0.29986438,  0.10984772,-0.01666681, -0.0100336 ]],[[ 0.00975369, -0.05370746, -0.3433557 ,  0.1110425 ,-0.00274591,  0.01028036],[ 0.01763022, -0.12980267, -0.24164987, -0.00480648,0.00250976,  0.02872233],[-0.00726193, -0.01916461, -0.327773  ,  0.01519999,-0.05319539,  0.00321992],...[-0.03956757, -0.22554985, -0.3036859 ,  0.00702061,0.08548178, -0.05071095],[-0.0638219 , -0.0131775 , -0.26752833,  0.06410608,-0.0647255 , -0.00823194],[ 0.0211888 , -0.06528416, -0.3983914 ,  0.03686601,0.04908308,  0.0462355 ]],[[-0.02021872, -0.06904162, -0.27097163, -0.11517487,-0.08520319, -0.10417534],[-0.02742095,  0.01650363, -0.28370744,  0.08527432,-0.03634344, -0.02588805],[-0.02958309,  0.01928876, -0.3066186 ,  0.0626737 ,-0.12659977,  0.03944535],...,[ 0.01413176, -0.09630561, -0.37563264, -0.06074851,-0.08401797, -0.04085656],[-0.02214429, -0.18425816, -0.34370732,  0.14820711,-0.01076629, -0.04269591],[-0.1529317 , -0.00595084, -0.38471988, -0.00670147,-0.15633702,  0.02909048]]], shape=(4, 1000, 6), dtype=float32)


b_feat


(chain, draw, product)


float32


0.7142 0.7178 ... 0.2067 0.2618


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 0.7142295 ,  0.7178441 ,  0.1325612 ,  0.32715878,0.06351098,  0.2044406 ],[ 0.74270594,  0.84775317,  0.03775882,  0.32169667,0.19188376,  0.2774    ],[ 0.6983323 ,  0.7993515 ,  0.24914663,  0.3346042 ,0.07645477,  0.21882108],...,[ 0.80377245,  0.7393316 ,  0.03853203,  0.26597053,0.06564465,  0.26391742],[ 0.7574953 ,  0.768253  ,  0.21378194,  0.34682903,0.21982849,  0.22196212],[ 0.78057   ,  0.7456948 ,  0.18101563,  0.27710116,0.23099448,  0.23498589]],[[ 0.7628648 ,  0.83247393,  0.13931352,  0.25722423,0.15893339,  0.23798658],[ 0.76462364,  0.94666845,  0.15197696,  0.24866197,0.12739651,  0.23830368],[ 0.82328904,  0.74625176,  0.22868463,  0.25602475,0.17159374,  0.2139992 ],...[ 0.8302902 ,  0.95964885,  0.13930674,  0.33587348,0.17647073,  0.29849803],[ 0.71726394,  0.7625792 ,  0.1000412 ,  0.3092264 ,0.07667316,  0.33023524],[ 0.74679375,  0.8692044 ,  0.23308276,  0.3007568 ,0.06784268,  0.24395213]],[[ 0.7482898 ,  0.7631741 ,  0.08452191,  0.24842711,0.10268176,  0.18978529],[ 0.735698  ,  0.7545218 ,  0.16120338,  0.2653053 ,0.11087242,  0.19901787],[ 0.7201336 ,  0.7585442 ,  0.19430172,  0.2673309 ,0.11616717,  0.22825652],...,[ 0.710958  ,  0.8891974 ,  0.06596008,  0.3312012 ,0.1338831 ,  0.2923628 ],[ 0.78025544,  0.9409306 ,  0.21240065,  0.34344548,0.09174219,  0.24490386],[ 0.8347106 ,  0.7740806 ,  0.24151279,  0.35893217,0.206719  ,  0.2618365 ]]], shape=(4, 1000, 6), dtype=float32)


b_feat_depth


(chain, draw, product)


float32


0.154 -0.3659 ... 0.4895 -0.431


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 0.15404028, -0.3658827 ,  0.8745704 , -0.34598324,0.6177338 ,  0.05177722],[-0.04682773, -0.555783  ,  1.0433495 , -0.25572935,-0.31498605, -0.66888404],[ 0.18010566, -0.6864979 ,  0.7546561 , -0.3410012 ,0.39524582, -0.23162295],...,[-0.03323258, -0.19595858,  1.0373652 , -0.22406189,0.9923637 , -0.5010775 ],[-0.09893042, -0.21279864,  0.8392605 , -0.62026465,-0.9832057 , -0.12049648],[-0.04531492, -0.3042134 ,  0.8151083 , -0.4473787 ,-0.72610074, -0.16868694]],[[ 0.021463  , -0.73046523,  0.8011511 , -0.45155692,-0.10765523, -0.5650607 ],[-0.10192786, -0.7472592 ,  0.7990083 , -0.4381345 ,0.08221451, -0.5034325 ],[-0.1111329 , -0.33519533,  0.59488326, -0.04628859,0.10835966, -0.2779928 ],...[-0.04364342, -0.83829254,  0.89142406, -0.30341974,-0.35952824, -0.31759873],[ 0.0692354 , -0.38496807,  0.98534304, -0.589579  ,0.17555286, -0.53826296],[ 0.01516938, -0.62359315,  0.8543634 , -0.5360967 ,0.46393272, -0.5422506 ]],[[ 0.1371795 , -0.13131703,  1.0244374 , -0.31138316,0.4744774 ,  0.10538208],[-0.02837937, -0.5925111 ,  0.9371999 , -0.25846407,0.25319475, -0.21934046],[ 0.01687419, -0.47378224,  0.8896432 , -0.28952336,0.20371853, -0.23858313],...,[ 0.18410593, -0.6760288 ,  1.2493131 , -0.40823534,0.47497544, -0.21168494],[ 0.0671873 , -0.7097479 ,  0.81281424, -0.74061596,0.19271253, -0.21767764],[ 0.03414429, -0.539763  ,  0.8501167 , -0.690275  ,0.48947722, -0.4310118 ]]], shape=(4, 1000, 6), dtype=float32)


b_sib_disp


(chain, draw, product)


float32


-0.03467 0.01273 ... 0.01377


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[-3.46668102e-02,  1.27312206e-02, -3.55297141e-02,-3.09116505e-02, -1.55059332e-02, -1.65128172e-03],[-4.64338101e-02, -1.18704839e-02, -2.28593848e-03,-1.59920901e-02, -1.04738250e-02,  7.91825447e-03],[-3.91167775e-02,  2.12208480e-02, -9.29427613e-03,-1.92711800e-02, -1.11234952e-02,  4.24002204e-03],...,[-6.11290038e-02,  3.06531023e-02, -2.17640121e-02,-1.51690003e-02, -8.93592648e-03,  3.19547690e-02],[-3.75618897e-02,  1.94372365e-03,  5.91495726e-03,-5.92621677e-02, -7.42386677e-04,  2.99875299e-03],[-6.07039444e-02, -1.31668802e-02, -1.85689144e-02,-2.74593364e-02, -7.17223389e-03,  2.21320949e-02]],[[-4.81678061e-02,  2.16699820e-02, -2.17496548e-02,-2.92411838e-02, -1.47154070e-02,  1.90218340e-03],[-4.95015793e-02,  1.40712624e-02, -1.80398040e-02,-4.88983952e-02,  1.49934627e-02,  4.06953786e-03],[-6.37443513e-02,  1.91995110e-02, -2.97932010e-02,-5.96989458e-03, -2.08303705e-02,  1.25378538e-02],...-4.08895984e-02, -2.01638527e-02, -1.55191785e-02],[-6.52227178e-02,  4.59631206e-03, -2.54470706e-02,-4.66491142e-03, -2.66246628e-02,  4.48906459e-02],[-1.36082927e-02, -9.06589627e-03, -3.40501452e-03,-2.34250743e-02, -3.19737382e-02,  1.30593022e-02]],[[-8.50433558e-02,  1.62217077e-02, -2.54124720e-02,-2.82433107e-02,  1.01953233e-02,  1.58327147e-02],[-5.26080243e-02,  7.00703124e-03, -4.44471799e-02,-4.40905057e-03, -1.45996753e-02,  2.64176372e-02],[-4.65217941e-02, -4.13830811e-03, -2.62135752e-02,-4.25304752e-03, -8.95082019e-03,  2.19557043e-02],...,[-6.84417859e-02,  1.93598475e-02, -8.30611680e-03,-3.89953284e-03, -2.54416112e-02,  2.38650870e-02],[-5.14778942e-02, -4.10782450e-05, -2.10235659e-02,-3.59865651e-02, -2.15521939e-02,  4.79476620e-03],[-4.43728194e-02,  4.47416585e-03, -1.28566008e-02,-5.88479266e-02, -7.55965803e-03,  1.37692261e-02]]],shape=(4, 1000, 6), dtype=float32)


b_sib_feat


(chain, draw, product)


float32


-0.08828 -0.03673 ... 0.01918


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[-8.82833302e-02, -3.67299952e-02,  4.77612652e-02,1.20295743e-02,  2.74751354e-02,  4.34634499e-02],[-1.07428610e-01, -9.68286954e-03,  1.70161594e-02,2.10723447e-05,  2.34154686e-02,  2.52073966e-02],[-4.54978868e-02, -7.24219531e-03,  2.63501387e-02,3.42866853e-02,  1.77906770e-02,  3.95345055e-02],...,[-8.66414085e-02, -6.11840114e-02,  1.30629931e-02,-7.56906625e-03,  3.71690956e-03,  1.32388743e-02],[-6.29727989e-02, -1.79772731e-02,  3.74123305e-02,3.59960534e-02,  1.50648030e-02,  2.98455618e-02],[-6.16047718e-02,  1.32770918e-03,  2.03719456e-02,2.59711500e-02,  1.08735282e-02,  3.16266976e-02]],[[-7.40574151e-02, -1.53654227e-02,  3.04483194e-02,1.24364020e-02,  1.58385765e-02,  2.94967070e-02],[-7.48957917e-02, -1.51604451e-02,  3.75934131e-02,8.38670135e-03,  1.82031039e-02,  4.37210761e-02],[-6.32294863e-02, -3.49684991e-02,  4.42458242e-02,1.25036864e-02,  7.50083884e-04,  1.41549315e-02],...1.47496341e-02,  4.29379269e-02,  4.05721404e-02],[-8.06918442e-02, -1.29278097e-02,  4.69494052e-02,-1.98138747e-02,  4.43795174e-02,  6.70162612e-04],[-8.78945440e-02, -2.97230761e-02,  4.00390178e-02,1.88051108e-02,  5.56533560e-02,  4.97460999e-02]],[[-4.80730608e-02, -1.24277845e-02,  2.76308432e-02,1.49669703e-02,  7.81867001e-03,  7.95433577e-03],[-8.04358423e-02,  1.01634057e-03,  3.60564664e-02,-3.64752370e-03,  6.27621682e-03,  3.10632363e-02],[-7.34941885e-02, -6.92208670e-03,  3.38037089e-02,-2.06522434e-03,  1.17824990e-02,  2.85723228e-02],...,[-8.13424438e-02, -3.20115946e-02,  4.14486751e-02,-1.07951984e-02,  1.63114406e-02,  6.07351540e-03],[-6.84393346e-02, -1.54505260e-02,  2.59021353e-02,2.56915223e-02,  3.93401831e-02,  2.85412781e-02],[-6.84564561e-02, -3.30647193e-02,  1.84655245e-02,3.42037380e-02,  1.78188756e-02,  1.91809498e-02]]],shape=(4, 1000, 6), dtype=float32)


beta_s


(chain, draw, fourier, product)


float32


-0.01898 -0.008101 ... -0.008993


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[-1.89751573e-02, -8.10128637e-03,  1.23703461e-02,6.42269403e-02, -5.36769070e-02,  2.11442523e-02],[-1.63886491e-02, -2.29061004e-02, -1.12382919e-02,1.90362614e-02,  4.37856428e-02, -6.83775684e-03],[ 6.91387877e-02, -2.79170424e-02, -3.36971544e-02,-3.69107723e-02,  1.92409568e-02, -3.26253213e-02],[ 1.36721544e-02, -7.20381225e-03, -9.05072375e-04,-1.59349944e-02, -8.22010171e-03, -3.03410459e-02]],[[ 3.74952890e-03, -3.02696694e-02,  7.06466322e-04,5.66255599e-02, -5.49304225e-02, -3.07847210e-03],[-3.19811106e-02, -9.26999561e-03,  5.70899015e-03,3.44795622e-02,  3.24603617e-02, -2.97957081e-02],[ 6.37276545e-02, -2.95596011e-02, -1.17439507e-02,-1.71779911e-03,  1.45183608e-03, -5.15645929e-03],[ 3.38994823e-02, -2.76596635e-03, -1.45690246e-02,-6.92906044e-03,  5.17250691e-03, -1.20657394e-02]],[[ 7.74748798e-04, -1.88851282e-02,  6.72089122e-03,5.46522066e-02, -2.70547923e-02,  2.68900744e-03],...3.05352197e-03, -8.05666111e-03, -8.15693289e-03]],[[ 1.93183613e-03, -5.22026327e-03,  1.92981232e-02,6.93748295e-02, -3.35017182e-02,  2.31632758e-02],[-1.64415371e-02, -1.97403934e-02,  1.20867630e-02,3.80349234e-02,  2.58849785e-02, -2.89611053e-02],[ 3.63536403e-02, -3.51245292e-02, -2.20126808e-02,-3.60169671e-02, -2.86772866e-02, -1.18393712e-02],[ 1.70432590e-02, -8.70568678e-03,  1.15365312e-02,-2.63060513e-03,  1.29058734e-02, -3.49766277e-02]],[[ 2.82300898e-04, -2.81609036e-02,  1.43775195e-02,8.49796683e-02, -5.16219176e-02,  1.88226085e-02],[-5.54608591e-02, -2.53200904e-02, -1.03527692e-03,2.04598047e-02,  3.91123556e-02, -2.35113055e-02],[ 4.33750562e-02, -3.39794643e-02, -3.31626311e-02,-1.88193340e-02,  7.53627566e-04, -1.67302880e-02],[ 3.44927162e-02, -5.22125652e-03,  6.04374288e-03,-1.60251521e-02, -1.17673939e-02, -8.99320375e-03]]]],shape=(4, 1000, 4, 6), dtype=float32)


conc


(chain, draw, product)


float32


17.08 19.4 21.94 ... 19.71 21.7


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[17.083784, 19.400288, 21.942505, 20.368444, 20.659822,20.963634],[16.625673, 18.736233, 21.33818 , 17.685297, 20.388742,21.615307],[16.918211, 19.340427, 20.559969, 19.408968, 20.127314,20.380316],...,[17.159922, 21.323784, 21.759024, 18.610725, 19.971378,21.434008],[17.398289, 17.938095, 22.638113, 19.046652, 19.21274 ,21.004164],[17.228828, 19.680214, 23.372454, 17.577227, 19.825731,19.666237]],[[18.099615, 19.415354, 22.113749, 20.316912, 18.414597,22.254284],[17.891436, 18.668941, 23.063452, 19.494654, 18.777071,20.449903],[17.38446 , 20.472837, 23.572739, 19.175703, 20.870352,21.54769 ],...[16.675364, 19.716019, 20.900787, 19.31005 , 20.050991,21.762993],[16.622095, 19.356009, 23.444885, 18.567778, 18.917831,21.128029],[16.948187, 18.726011, 22.674768, 19.380035, 18.098969,20.700844]],[[17.371115, 18.930737, 20.774845, 19.852777, 18.822256,20.365152],[16.671385, 18.151089, 21.87238 , 18.288033, 19.76614 ,19.373707],[17.272367, 19.548615, 21.657969, 17.84552 , 19.852484,19.781208],...,[17.580265, 19.669561, 23.227388, 17.743225, 19.623344,20.395544],[18.673786, 19.311462, 21.075947, 19.34932 , 18.1544  ,21.74994 ],[17.973408, 19.475073, 21.957216, 18.615854, 19.70789 ,21.697542]]], shape=(4, 1000, 6), dtype=float32)


cross_scale


(chain, draw)


float32


0.2681 0.2501 ... 0.2396 0.2558


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.26809466, 0.25013673, 0.33999228, ..., 0.3138818 , 0.23483881,0.23935318],[0.25443467, 0.25548786, 0.23831415, ..., 0.29781464, 0.28700107,0.23230289],[0.37217152, 0.42292118, 0.27313098, ..., 0.33782855, 0.22903179,0.3127264 ],[0.21018253, 0.21888326, 0.2367111 , ..., 0.24165215, 0.23956074,0.25580087]], shape=(4, 1000), dtype=float32)


drift


(chain, draw, time, series)


float32


-0.07196 0.04289 ... 0.0424


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[-7.19551817e-02,  4.28864695e-02, -1.97732765e-02, ...,5.40078711e-03, -4.27770466e-02,  9.04403813e-03],[-8.83597806e-02, -5.40869050e-02, -1.56207364e-02, ...,3.39879160e-04,  1.36711895e-02, -2.59768851e-02],[-8.46159533e-02,  6.37038937e-03,  9.14770830e-03, ...,-5.15279965e-03, -6.39558733e-02, -3.58320959e-02],...,[-2.23544799e-02, -1.66788325e-02,  4.03650664e-02, ...,1.16819432e-02, -3.10755218e-03, -1.21170599e-02],[-2.39281245e-02, -1.63392983e-02,  4.91296351e-02, ...,-7.95503855e-02,  3.44748399e-03,  2.30159033e-02],[-2.15590727e-02, -3.70007381e-02,  8.57319683e-03, ...,-4.39267680e-02,  5.23185655e-02, -5.02265394e-02]],[[ 9.29319859e-02, -3.95848639e-02,  3.85534437e-03, ...,1.24872126e-01, -5.95259182e-02, -5.28425165e-03],[ 1.53844967e-01,  7.05481693e-02,  5.61998459e-03, ...,1.82020977e-01, -4.37562875e-02,  1.49164405e-02],[ 6.36466071e-02, -4.55095582e-02,  7.15944823e-03, ...,1.66640002e-02,  5.58542013e-02,  2.44535897e-02],...5.65495482e-03, -1.69760622e-02,  1.45909078e-02],[-5.54033667e-02,  4.21899697e-03, -1.51791016e-03, ...,5.53659424e-02, -2.27772072e-02,  2.25413367e-02],[ 1.47955775e-01,  2.69195036e-04,  2.59598941e-02, ...,-3.44998203e-02,  1.03235818e-01, -9.19267419e-04]],[[ 4.89865132e-02, -7.62991905e-02, -1.83109916e-03, ...,-9.91598666e-02,  7.93900415e-02, -1.84414759e-02],[ 4.13492918e-02,  1.80839433e-03, -1.79524068e-02, ...,-3.19876708e-03,  5.17535955e-02, -6.69762790e-02],[-7.60218827e-03,  7.84076080e-02, -1.26080099e-03, ...,8.96354672e-03,  5.05037270e-02,  1.28878895e-02],...,[-2.60239858e-02, -2.93169264e-02,  2.37174216e-03, ...,1.16422102e-02, -5.88899776e-02, -2.49604173e-02],[-2.11611204e-02, -2.50161346e-02,  9.63837956e-05, ...,7.61196166e-02, -1.14162557e-01,  8.71168170e-03],[-9.02970508e-02,  1.08167548e-02,  1.38798337e-02, ...,-7.70234242e-02, -5.01322672e-02,  4.23990004e-02]]]],shape=(4, 1000, 143, 108), dtype=float32)


drift_decentered


(chain, draw, time, series)


float32


-0.3865 0.4074 ... -0.3497 0.4103


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[-3.86523247e-01,  4.07423764e-01, -2.26665750e-01, ...,4.12804149e-02, -2.59361327e-01,  8.93184617e-02],[-4.74644214e-01, -5.13828516e-01, -1.79064199e-01, ...,2.59783491e-03,  8.28897282e-02, -2.56546408e-01],[-4.54533428e-01,  6.05190434e-02,  1.04862340e-01, ...,-3.93849462e-02, -3.87770563e-01, -3.53875965e-01],...,[-1.20082058e-01, -1.58449799e-01,  4.62714285e-01, ...,8.92898515e-02, -1.88413858e-02, -1.19667470e-01],[-1.28535241e-01, -1.55224204e-01,  5.63184619e-01, ...,-6.08035982e-01,  2.09024251e-02,  2.27303892e-01],[-1.15809351e-01, -3.51509005e-01,  9.82765779e-02, ...,-3.35750192e-01,  3.17212462e-01, -4.96034771e-01]],[[ 5.12091875e-01, -2.86316723e-01,  6.41446337e-02, ...,6.01734936e-01, -4.01094496e-01, -6.35839552e-02],[ 8.47746432e-01,  5.10273874e-01,  9.35044512e-02, ...,8.77124310e-01, -2.94836372e-01,  1.79485455e-01],[ 3.50717902e-01, -3.29169959e-01,  1.19117811e-01, ...,8.03006366e-02,  3.76353920e-01,  2.94243366e-01],...3.14317048e-02, -9.44036841e-02,  1.60652131e-01],[-2.64632881e-01,  6.43069521e-02, -2.32363045e-02, ...,3.07738245e-01, -1.26663789e-01,  2.48189762e-01],[ 7.06707299e-01,  4.10313439e-03,  3.97396386e-01, ...,-1.91758931e-01,  5.74093223e-01, -1.01215271e-02]],[[ 3.31017613e-01, -6.71705902e-01, -2.67212559e-02, ...,-6.08051896e-01,  5.53823531e-01, -1.78443581e-01],[ 2.79410452e-01,  1.59203410e-02, -2.61979729e-01, ...,-1.96149554e-02,  3.61032158e-01, -6.48076534e-01],[-5.13704307e-02,  6.90267503e-01, -1.83988865e-02, ...,5.49647957e-02,  3.52313101e-01,  1.24705918e-01],...,[-1.75852433e-01, -2.58093864e-01,  3.46108675e-02, ...,7.13904575e-02, -4.10815418e-01, -2.41522238e-01],[-1.42992496e-01, -2.20231503e-01,  1.40653003e-03, ...,4.66768235e-01, -7.96395957e-01,  8.42960626e-02],[-6.10166192e-01,  9.52261463e-02,  2.02548608e-01, ...,-4.72310424e-01, -3.49721789e-01,  4.10261631e-01]]]],shape=(4, 1000, 143, 108), dtype=float32)


drift_scale


(chain, draw, series)


float32


0.07947 0.03367 ... 0.05361 0.03275


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[0.07947393, 0.03367009, 0.02537219, ..., 0.04671952,0.06622498, 0.03175841],[0.07648059, 0.05076961, 0.01447614, ..., 0.09360155,0.05648893, 0.02358522],[0.07418273, 0.03375674, 0.01814971, ..., 0.0769455 ,0.07397935, 0.05831948],...,[0.05863094, 0.04810175, 0.01572927, ..., 0.07863794,0.0737507 , 0.02097575],[0.07103465, 0.0399282 , 0.01902252, ..., 0.06901143,0.06923698, 0.03416828],[0.08537655, 0.04308476, 0.03285278, ..., 0.06215073,0.0531256 , 0.03828393]],[[0.08407392, 0.04574266, 0.01235822, ..., 0.06386501,0.04443037, 0.03639057],[0.11676988, 0.04591077, 0.01202211, ..., 0.06672288,0.05858812, 0.0369881 ],[0.07181643, 0.03427061, 0.01433314, ..., 0.0989903 ,0.058421  , 0.03861966],...[0.07814018, 0.02731213, 0.02109564, ..., 0.09005226,0.03952967, 0.03597017],[0.08118264, 0.03535899, 0.02364253, ..., 0.07933161,0.05143693, 0.05503741],[0.0764184 , 0.05028925, 0.0229952 , ..., 0.06635354,0.04818732, 0.04127147]],[[0.10804875, 0.03722514, 0.02457714, ..., 0.07947727,0.05895413, 0.03407892],[0.10735393, 0.03228206, 0.0275521 , ..., 0.04191061,0.03605965, 0.04766052],[0.12440065, 0.02692812, 0.02895836, ..., 0.04537579,0.03782902, 0.04706014],...,[0.08956318, 0.02139478, 0.01161584, ..., 0.05598541,0.06301938, 0.02210684],[0.09485392, 0.01651829, 0.01641138, ..., 0.07549069,0.07543488, 0.02696019],[0.05624768, 0.03776183, 0.01763762, ..., 0.06510665,0.05361313, 0.03275105]]], shape=(4, 1000, 108), dtype=float32)


eps


(chain, draw, series)


float32


-1.757 -0.2659 ... -1.484 -0.4967


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[-1.7574551 , -0.26586622, -2.3790913 , ..., -2.2230186 ,-1.555318  , -1.1351467 ],[-1.7737577 , -0.37167072, -2.060865  , ..., -1.8517216 ,-1.5146569 , -0.89029163],[-1.2226504 , -0.3279953 , -2.2168922 , ..., -2.0752668 ,-1.4594266 , -0.93723154],...,[-1.407234  , -0.51680684, -2.218767  , ..., -2.3284824 ,-1.6658865 , -1.0797267 ],[-1.6019514 , -0.08927833, -2.3405771 , ..., -2.049778  ,-1.4959338 , -1.029592  ],[-1.7953731 , -0.3123821 , -2.2145975 , ..., -2.0383718 ,-1.5675948 , -1.0991083 ]],[[-1.5825857 , -0.2707955 , -2.2689662 , ..., -2.3769138 ,-1.7285005 , -1.171268  ],[-1.6585839 , -0.3478758 , -2.3385904 , ..., -2.0259993 ,-1.7710772 , -0.9752885 ],[-1.3857265 , -0.2928013 , -2.323079  , ..., -1.9024646 ,-1.8705627 , -1.191082  ],...[-1.6187662 , -0.38120034, -2.3084795 , ..., -2.4464602 ,-1.995206  , -0.9038553 ],[-1.6978562 , -0.2457155 , -2.1474204 , ..., -1.8701096 ,-1.7431036 , -1.448111  ],[-1.5706779 , -0.33598036, -2.0451624 , ..., -2.222236  ,-1.9313843 , -1.4910717 ]],[[-1.5926874 , -0.12620102, -2.2525299 , ..., -1.771388  ,-1.5632579 , -1.1311535 ],[-1.2022935 , -0.43950015, -2.1160548 , ..., -2.4032245 ,-2.2040553 , -1.2698522 ],[-1.4161158 , -0.49517536, -2.0781386 , ..., -2.0936491 ,-2.085445  , -1.277437  ],...,[-1.4058136 , -0.2529309 , -2.0200903 , ..., -1.6172214 ,-1.2466631 , -0.9376833 ],[-1.5998083 , -0.35111824, -2.0526571 , ..., -2.6769173 ,-1.7458446 , -1.3456414 ],[-1.5756744 , -0.22493581, -2.1237109 , ..., -2.1536975 ,-1.4839698 , -0.49673223]]], shape=(4, 1000, 108), dtype=float32)


eps_decentered


(chain, draw, series)


float32


-1.524 -0.172 ... -0.827 -0.03799


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[-1.5242959e+00, -1.7197469e-01, -1.8447280e+00, ...,-1.6190526e+00, -7.8631783e-01, -1.0462530e+00],[-1.6998432e+00, -3.8176787e-01, -1.5961661e+00, ...,-1.4355493e+00, -7.3643827e-01, -2.8426638e-01],[-9.4633800e-01, -1.0769859e-01, -1.7822710e+00, ...,-1.6002245e+00, -7.8539538e-01, -7.1139932e-01],...,[-1.1222870e+00, -3.6810651e-01, -1.7640939e+00, ...,-1.9695174e+00, -9.7487229e-01, -6.8649364e-01],[-1.5076725e+00,  1.4593700e-01, -2.0057218e+00, ...,-1.2749840e+00, -8.1062186e-01, -8.2120597e-01],[-1.8164896e+00, -1.0832498e-01, -1.6972001e+00, ...,-1.3073245e+00, -1.0829519e+00, -9.1838920e-01]],[[-1.4715356e+00, -1.4317043e-01, -1.8661435e+00, ...,-1.8406587e+00, -9.5936871e-01, -7.6627904e-01],[-1.4453396e+00, -2.6422369e-01, -1.9421064e+00, ...,-1.1939510e+00, -1.0921359e+00, -5.0465149e-01],[-1.1854131e+00, -2.4017608e-01, -1.8965119e+00, ...,-1.3491530e+00, -1.2620330e+00, -7.7135122e-01],...-2.3022745e+00, -1.2256120e+00, -5.7758921e-01],[-1.5741149e+00, -1.4063889e-01, -1.6282167e+00, ...,-9.4437706e-01, -1.0970470e+00, -1.1678312e+00],[-1.3986567e+00, -3.1508440e-01, -1.7489070e+00, ...,-1.5777705e+00, -1.0301838e+00, -1.4354826e+00]],[[-1.5109743e+00,  1.4501992e-01, -1.8325897e+00, ...,-1.1224158e+00, -8.7315321e-01, -8.7489122e-01],[-9.1978848e-01, -2.4661826e-01, -1.6803361e+00, ...,-1.9372334e+00, -1.7009953e+00, -1.0531312e+00],[-1.1426988e+00, -3.1849450e-01, -1.6451927e+00, ...,-1.5154139e+00, -1.5078157e+00, -1.1070865e+00],...,[-1.1602697e+00, -9.7432017e-02, -1.6715791e+00, ...,-9.7458839e-01, -4.7027332e-01, -5.8800036e-01],[-1.5937884e+00, -1.6559015e-01, -1.5530914e+00, ...,-2.2007625e+00, -1.0766943e+00, -1.2366890e+00],[-1.7056992e+00, -6.9119863e-02, -1.7243369e+00, ...,-1.5557936e+00, -8.2698923e-01, -3.7988629e-02]]],shape=(4, 1000, 108), dtype=float32)


eps_prod


(chain, draw, product)


float32


-1.244 -0.2781 ... -1.724 -0.8995


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[-1.2444247 , -0.2780997 , -2.0198395 , -2.0477405 ,-1.9636643 , -0.7092058 ],[-1.1825689 , -0.21749671, -1.8225722 , -1.6359928 ,-1.8566284 , -1.2672657 ],[-1.0592561 , -0.48526296, -1.8292971 , -1.8062432 ,-1.7379472 , -0.8312729 ],...,[-1.209594  , -0.49550766, -1.9134383 , -1.8704655 ,-1.8436655 , -1.1313436 ],[-1.0886241 , -0.3394983 , -1.8295575 , -2.1861193 ,-1.7407727 , -0.87886906],[-1.0689977 , -0.43695655, -1.975143  , -2.12582   ,-1.541142  , -0.8883719 ]],[[-1.1095632 , -0.31820315, -1.878632  , -2.1030247 ,-1.9773383 , -1.202598  ],[-1.2547345 , -0.31188622, -1.8890718 , -2.2479928 ,-1.9045486 , -1.1698827 ],[-1.0808105 , -0.241064  , -1.9237353 , -1.8267657 ,-1.875605  , -1.2340297 ],...[-1.1486895 , -0.5281105 , -1.8100989 , -1.6970552 ,-2.139999  , -0.94351107],[-1.0857829 , -0.28270617, -1.8957633 , -2.3331609 ,-1.8623883 , -1.1777918 ],[-1.1759522 , -0.23283857, -1.6162169 , -2.1356368 ,-2.257398  , -0.989629  ]],[[-1.0484306 , -0.415897  , -1.8654288 , -1.8676608 ,-1.7957863 , -0.9938128 ],[-1.0454886 , -0.5147495 , -1.7491024 , -1.9437795 ,-1.8946592 , -0.98343265],[-1.1447682 , -0.51974165, -1.7266548 , -1.9488388 ,-1.9436475 , -0.92026126],...,[-1.1280639 , -0.35109797, -1.6152225 , -1.7769868 ,-1.7437648 , -0.9998127 ],[-0.99151206, -0.43692353, -1.8577461 , -2.2139742 ,-1.87087   , -0.9576742 ],[-0.7125635 , -0.33719346, -1.7363437 , -2.0213313 ,-1.7236077 , -0.8994529 ]]], shape=(4, 1000, 6), dtype=float32)


eps_scale


(chain, draw)


float32


0.4144 0.2729 ... 0.2755 0.3397


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[0.41437835, 0.27293608, 0.3501091 , ..., 0.2650199 , 0.28418836,0.2824325 ],[0.27190718, 0.29977757, 0.29204726, ..., 0.28854996, 0.298483  ,0.34686798],[0.26897046, 0.29498625, 0.33925897, ..., 0.26317847, 0.37658224,0.26989314],[0.30204636, 0.39250103, 0.3900153 , ..., 0.3317781 , 0.2754712 ,0.33974943]], shape=(4, 1000), dtype=float32)


gamma


(chain, draw, competitor, product)


float32


0.0 0.06844 ... 0.3755 0.0


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 0.00000000e+00,  6.84406236e-02, -3.83301498e-03,-7.34236464e-02,  7.01235652e-01,  7.28043169e-02],[-2.47216359e-01,  0.00000000e+00,  1.67413846e-01,2.77808327e-02, -9.84547511e-02, -3.30462307e-02],[-3.60903829e-01,  2.06770524e-01,  0.00000000e+00,2.19514337e-03,  1.20481879e-01,  2.07659304e-01],[-9.98440906e-02, -1.35262132e-01,  1.57556295e-01,0.00000000e+00, -3.83474045e-02,  5.53708673e-01],[-3.11648756e-01, -8.90759945e-01, -1.08279914e-01,1.75174162e-01,  0.00000000e+00,  3.15057844e-01],[ 5.35296537e-02,  1.81869298e-01,  3.61497477e-02,1.11985482e-01,  3.71995747e-01,  0.00000000e+00]],[[ 0.00000000e+00,  1.27270117e-01,  6.91112280e-02,-4.45767073e-05,  6.74514413e-01,  1.01047866e-01],[-2.48341411e-01,  0.00000000e+00,  1.42449245e-01,6.06725961e-02, -4.54611517e-02,  2.10030843e-02],[-3.16931367e-01,  1.48702830e-01,  0.00000000e+00,2.64007542e-02, -1.41965663e-02,  7.49055669e-02],[-2.39267722e-01, -2.79874742e-01,  2.06545144e-01,...[-1.40202612e-01, -1.81343451e-01,  1.13699161e-01,0.00000000e+00, -9.42735374e-02,  4.76956397e-01],[-1.62938982e-01, -6.12873018e-01, -2.25727350e-01,2.78737191e-02,  0.00000000e+00,  2.28364274e-01],[ 1.29435569e-01,  2.98816532e-01, -1.23395786e-01,-7.70602375e-04,  3.45580578e-01,  0.00000000e+00]],[[ 0.00000000e+00,  5.82125299e-02, -3.13877910e-02,2.94336267e-02,  6.55845344e-01,  1.09586000e-01],[-2.03666613e-01,  0.00000000e+00,  1.48052871e-01,1.83295943e-02, -9.38624665e-02,  2.00661626e-02],[-2.72403151e-01,  2.17534781e-01,  0.00000000e+00,-4.14808244e-02,  6.68770149e-02,  1.07287012e-01],[-3.10615916e-03, -3.02412868e-01,  2.04387218e-01,0.00000000e+00,  2.09000651e-02,  5.05874515e-01],[-1.56528160e-01, -7.06096053e-01, -6.45255595e-02,9.93607566e-02,  0.00000000e+00,  3.58125269e-01],[ 3.83024663e-02,  1.90592185e-01, -1.02474630e-01,1.36397947e-02,  3.75471175e-01,  0.00000000e+00]]]],shape=(4, 1000, 6, 6), dtype=float32)


gamma_offdiag


(chain, draw, pair)


float32


0.06844 -0.003833 ... 0.3755


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 6.84406236e-02, -3.83301498e-03, -7.34236464e-02, ...,3.61497477e-02,  1.11985482e-01,  3.71995747e-01],[ 1.27270117e-01,  6.91112280e-02, -4.45767073e-05, ...,1.56332441e-02,  1.26138791e-01,  1.86109006e-01],[ 1.11147232e-01,  1.91575736e-02, -2.38141362e-02, ...,-1.62991136e-01, -2.82191280e-02,  4.30944711e-01],...,[ 9.61578637e-02,  9.18599889e-02, -4.77402024e-02, ...,-1.37739731e-02,  1.11273639e-01,  5.11659503e-01],[ 5.86128235e-02,  1.36223370e-02, -3.28929015e-02, ...,-9.75034833e-02,  8.06352943e-02,  4.20176089e-01],[ 9.41556916e-02,  8.48203525e-03, -5.43725751e-02, ...,-8.30868632e-02, -1.40438307e-04,  4.99972492e-01]],[[ 1.45220682e-01,  1.22094601e-01, -2.80457772e-02, ...,-1.06350712e-01, -4.77754399e-02,  3.08835775e-01],[ 9.08976048e-02,  9.65235606e-02, -2.92696599e-02, ...,-9.64321494e-02,  1.19578582e-03,  3.80213588e-01],[ 6.31864890e-02, -1.35294646e-02, -1.19119093e-01, ...,-5.88041879e-02, -2.27790177e-02,  4.52128023e-01],...-2.11325176e-02,  2.50147339e-02,  3.91708285e-01],[ 7.52881989e-02,  4.53650467e-02, -8.84738043e-02, ...,2.11198274e-02,  1.42986029e-01,  4.19159740e-01],[ 1.08811475e-01,  6.49477318e-02, -4.18032557e-02, ...,-1.33453920e-01,  5.07726707e-02,  3.22853804e-01]],[[ 8.62468332e-02, -2.01245546e-02, -4.53520454e-02, ...,-7.38964006e-02,  1.21533990e-01,  4.13152575e-01],[ 1.41782314e-01, -3.31161208e-02, -5.45493476e-02, ...,-5.58307432e-02,  1.63301125e-01,  5.19802630e-01],[ 9.89510641e-02, -4.70811538e-02, -5.90440109e-02, ...,-2.79981997e-02,  2.19022527e-01,  5.14212489e-01],...,[ 7.82869011e-02,  4.27889600e-02,  6.21696096e-03, ...,-6.66545108e-02, -5.92694841e-02,  2.73565292e-01],[ 1.00691520e-01,  2.48451419e-02, -1.90553647e-02, ...,-1.23395786e-01, -7.70602375e-04,  3.45580578e-01],[ 5.82125299e-02, -3.13877910e-02,  2.94336267e-02, ...,-1.02474630e-01,  1.36397947e-02,  3.75471175e-01]]],shape=(4, 1000, 30), dtype=float32)


gamma_offdiag_decentered


(chain, draw, pair)


float32


0.1501 -0.008404 ... 0.03075 0.8466


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[ 1.50050268e-01, -8.40356015e-03, -1.60975114e-01, ...,7.92552531e-02,  2.45518669e-01,  8.15569103e-01],[ 2.90806979e-01,  1.57916307e-01, -1.01855934e-04, ...,3.57213169e-02,  2.88221925e-01,  4.25251395e-01],[ 2.11491406e-01,  3.64531092e-02, -4.53136377e-02, ...,-3.10140193e-01, -5.36954738e-02,  8.20003390e-01],...,[ 1.91899061e-01,  1.83321938e-01, -9.52735394e-02, ...,-2.74882615e-02,  2.22065106e-01,  1.02110195e+00],[ 1.39063999e-01,  3.23201753e-02, -7.80412555e-02, ...,-2.31335461e-01,  1.91314220e-01,  9.96904135e-01],[ 2.20870376e-01,  1.98971536e-02, -1.27547160e-01, ...,-1.94905132e-01, -3.29440110e-04,  1.17283523e+00]],[[ 3.28469306e-01,  2.76161283e-01, -6.34357110e-02, ...,-2.40550756e-01, -1.08061507e-01,  6.98544264e-01],[ 2.05092132e-01,  2.17785954e-01, -6.60410896e-02, ...,-2.17579708e-01,  2.69804965e-03,  8.57875347e-01],[ 1.48607865e-01, -3.18198539e-02, -2.80155361e-01, ...,-1.38301164e-01, -5.35738133e-02,  1.06335676e+00],...-4.03644219e-02,  4.77796942e-02,  7.48187184e-01],[ 1.81314871e-01,  1.09251618e-01, -2.13069454e-01, ...,5.08624017e-02,  3.44350010e-01,  1.00945294e+00],[ 2.17629522e-01,  1.29899383e-01, -8.36090371e-02, ...,-2.66915917e-01,  1.01548411e-01,  6.45727098e-01]],[[ 2.18620986e-01, -5.10123074e-02, -1.14959687e-01, ...,-1.87314749e-01,  3.08067918e-01,  1.04727125e+00],[ 3.50805253e-01, -8.19376409e-02, -1.34968862e-01, ...,-1.38139352e-01,  4.04048204e-01,  1.28612304e+00],[ 2.33660877e-01, -1.11176409e-01, -1.39425233e-01, ...,-6.61143363e-02,  5.17194986e-01,  1.21425009e+00],...,[ 1.82601541e-01,  9.98037979e-02,  1.45008499e-02, ...,-1.55469388e-01, -1.38244063e-01,  6.38081789e-01],[ 2.36080080e-01,  5.82516082e-02, -4.46769670e-02, ...,-2.89312214e-01, -1.80674472e-03,  8.10243905e-01],[ 1.31248981e-01, -7.07685351e-02,  6.63625747e-02, ...,-2.31044590e-01,  3.07529867e-02,  8.46556723e-01]]],shape=(4, 1000, 30), dtype=float32)


level0


(chain, draw, series)


float32


4.558 4.627 3.976 ... 2.51 2.814


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[4.557632 , 4.6268854, 3.9762526, ..., 3.0580869, 2.5662746,2.664699 ],[4.3326454, 4.59555  , 3.9723191, ..., 2.7941167, 2.5302222,2.5141437],[4.4335012, 4.516539 , 3.9867048, ..., 2.9042056, 2.3078353,2.7411034],...,[4.474144 , 4.482262 , 3.932602 , ..., 2.9600303, 2.4870906,2.5111141],[4.1281333, 4.394543 , 4.098597 , ..., 3.2179694, 2.9433804,2.6597621],[4.252075 , 4.368609 , 4.034776 , ..., 3.2503633, 2.6589904,2.5514817]],[[4.5569644, 4.6149364, 3.9700313, ..., 2.8905692, 2.4885247,2.5916178],[4.423984 , 4.633698 , 3.992783 , ..., 2.9291155, 2.607357 ,2.6614127],[4.560329 , 4.689282 , 3.8699157, ..., 3.0117905, 2.9664845,2.618389 ],...[4.522133 , 4.483061 , 3.9849608, ..., 2.934512 , 2.5027869,2.6146984],[4.567824 , 4.4755588, 3.992938 , ..., 3.325741 , 2.7956607,2.5650952],[4.533059 , 4.3147497, 4.135053 , ..., 3.123872 , 2.7428849,2.5764222]],[[4.5581293, 4.43532  , 3.9866765, ..., 3.0908732, 2.4777157,2.7309952],[4.6999955, 4.6136065, 3.880272 , ..., 3.2144296, 2.6367948,2.6286154],[4.6963468, 4.6019936, 3.90636  , ..., 3.2592568, 2.686092 ,2.7105742],...,[4.4558706, 4.3601403, 3.9554663, ..., 3.1827376, 2.8592887,2.6845148],[4.718162 , 4.630901 , 3.9859667, ..., 3.156888 , 2.7488842,2.6210694],[4.254391 , 4.426921 , 3.9078739, ..., 3.362545 , 2.5098522,2.8139935]]], shape=(4, 1000, 108), dtype=float32)


Attributes: (5)


created_at :  
2026-09-07T13:25:52.911307+00:00

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


89 146 49 37 67 ... 15 12 16 5 32


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 89, 146,  49, ...,  21,  19,  16],[ 57, 111,  58, ...,  29,   8,  18],[ 83,  86,  37, ...,  14,  12,  12],...,[ 97, 117,  61, ...,  14,  10,  19],[105, 137,  35, ...,  24,  28,  16],[108, 131, 106, ...,  24,  15,   7]],[[ 59, 215,  74, ...,  18,   6,  20],[147, 123,  58, ...,  22,  13,  17],[ 68, 126,  28, ...,  20,  13,  14],...,[116, 102, 102, ...,  24,  13,  40],[129,  76,  71, ...,  29,  12,  31],[175,  66,  53, ...,  17,   7,  30]],[[ 57, 187,  60, ...,  14,  10,  21],[139,  92,  43, ...,  26,   9,  11],[ 51, 122,  59, ...,  25,   7,  18],...,...[170, 112,  60, ...,  13,   4,  34],[167,  56,  72, ...,  21,  12,  21],[183,  72,  79, ...,  21,   5,  23]],[[ 89, 237,  70, ...,  25,  16,  17],[ 85, 100,  64, ...,  22,   4,  29],[120,  71,  63, ...,  13,  11,  18],...,[131, 141,  94, ...,  22,  11,  21],[154, 125,  54, ...,  10,   7,  26],[119,  77,  61, ...,   9,  14,  19]],[[111, 177,  59, ...,  19,   6,  16],[ 61,  79,  43, ...,  20,   9,  16],[ 54,  70,  67, ...,  31,  19,  12],...,[164,  93,  53, ...,  11,  10,  23],[128, 150,  83, ...,  16,   7,  38],[163, 130,  64, ...,  16,   5,  32]]]],shape=(4, 1000, 143, 108), dtype=int32)


Attributes: (5)


created_at :  
2026-09-07T13:26:10.486057+00:00

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
2026-09-07T13:26:10.488747+00:00

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
2026-09-07T13:26:10.489640+00:00

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


146 62 78 66 65 ... 12 13 11 4 29


<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWZpbGUtdGV4dDIiPjx1c2UgaHJlZj0iI2ljb24tZmlsZS10ZXh0MiIgLz48L3N2Zz4=" class="icon xr-icon-file-text2" />

<img src="data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iaWNvbiB4ci1pY29uLWRhdGFiYXNlIj48dXNlIGhyZWY9IiNpY29uLWRhdGFiYXNlIiAvPjwvc3ZnPg==" class="icon xr-icon-database" />


    array([[[[ 146,   62,   78, ...,   27,   20,   23],[ 125,   60,   59, ...,   18,    8,   25],[ 120,   96,  107, ...,   15,   25,   22],...,[1258,  173,   67, ...,   14,   17,   35],[1368,  138,   67, ...,   19,   10,   24],[1384,   77,   54, ...,   33,    8,   20]],[[ 194,   40,   81, ...,   15,    5,   48],[ 221,  131,   65, ...,   24,   14,   23],[ 210,   70,   57, ...,   17,    7,   36],...,[1177,   83,   42, ...,   29,    5,   29],[ 870,   91,   69, ...,   18,   14,   43],[1224,   77,   57, ...,   23,   18,   19]],[[ 173,  118,   50, ...,   22,   18,   19],[ 154,   74,   23, ...,   13,   24,   27],[  89,  134,   67, ...,   12,   16,   26],...,...[1058,   66,   64, ...,   21,   10,   40],[ 903,   67,   85, ...,   20,    6,   20],[1035,  111,   56, ...,   15,    5,   40]],[[ 165,  132,   53, ...,   12,   12,   24],[ 122,  101,   50, ...,    9,   22,   21],[ 132,   72,   74, ...,   14,   14,   18],...,[ 902,  101,   49, ...,   22,   14,   17],[1072,   83,   58, ...,   21,   19,   26],[ 788,  115,   45, ...,   16,   15,   11]],[[ 103,  151,   37, ...,   17,    9,   27],[ 133,  111,   53, ...,   26,   17,   31],[ 184,   98,   56, ...,   17,    8,   19],...,[ 909,   67,   74, ...,   12,    3,   29],[1360,   66,   58, ...,   19,    9,   21],[ 932,  129,   63, ...,   11,    4,   29]]]],shape=(4, 1000, 13, 108), dtype=int32)


Attributes: (5)


created_at :  
2026-09-07T13:26:13.021705+00:00

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


    array(['25027::HNC', '25027::Cheerios 12oz', '25027::Cheerios 18oz','25027::Mini Wheats', '25027::PL Honey Nut Oats','25027::PL Frosted Wheat', '21237::HNC', '21237::Cheerios 12oz','21237::Cheerios 18oz', '21237::Mini Wheats','21237::PL Honey Nut Oats', '21237::PL Frosted Wheat', '25229::HNC','25229::Cheerios 12oz', '25229::Cheerios 18oz', '25229::Mini Wheats','25229::PL Honey Nut Oats', '25229::PL Frosted Wheat', '19265::HNC','19265::Cheerios 12oz', '19265::Cheerios 18oz', '19265::Mini Wheats','19265::PL Honey Nut Oats', '19265::PL Frosted Wheat', '9825::HNC','9825::Cheerios 12oz', '9825::Cheerios 18oz', '9825::Mini Wheats','9825::PL Honey Nut Oats', '9825::PL Frosted Wheat', '613::HNC','613::Cheerios 12oz', '613::Cheerios 18oz', '613::Mini Wheats','613::PL Honey Nut Oats', '613::PL Frosted Wheat', '2277::HNC','2277::Cheerios 12oz', '2277::Cheerios 18oz', '2277::Mini Wheats','2277::PL Honey Nut Oats', '2277::PL Frosted Wheat', '24991::HNC','24991::Cheerios 12oz', '24991::Cheerios 18oz', '24991::Mini Wheats','24991::PL Honey Nut Oats', '24991::PL Frosted Wheat', '6179::HNC','6179::Cheerios 12oz', '6179::Cheerios 18oz', '6179::Mini Wheats','6179::PL Honey Nut Oats', '6179::PL Frosted Wheat', '2513::HNC','2513::Cheerios 12oz', '2513::Cheerios 18oz', '2513::Mini Wheats','2513::PL Honey Nut Oats', '2513::PL Frosted Wheat', '2281::HNC','2281::Cheerios 12oz', '2281::Cheerios 18oz', '2281::Mini Wheats','2281::PL Honey Nut Oats', '2281::PL Frosted Wheat', '11993::HNC','11993::Cheerios 12oz', '11993::Cheerios 18oz', '11993::Mini Wheats','11993::PL Honey Nut Oats', '11993::PL Frosted Wheat', '25021::HNC','25021::Cheerios 12oz', '25021::Cheerios 18oz', '25021::Mini Wheats','25021::PL Honey Nut Oats', '25021::PL Frosted Wheat', '4259::HNC','4259::Cheerios 12oz', '4259::Cheerios 18oz', '4259::Mini Wheats','4259::PL Honey Nut Oats', '4259::PL Frosted Wheat', '21479::HNC','21479::Cheerios 12oz', '21479::Cheerios 18oz', '21479::Mini Wheats','21479::PL Honey Nut Oats', '21479::PL Frosted Wheat', '23349::HNC','23349::Cheerios 12oz', '23349::Cheerios 18oz', '23349::Mini Wheats','23349::PL Honey Nut Oats', '23349::PL Frosted Wheat', '19523::HNC','19523::Cheerios 12oz', '19523::Cheerios 18oz', '19523::Mini Wheats','19523::PL Honey Nut Oats', '19523::PL Frosted Wheat', '6431::HNC','6431::Cheerios 12oz', '6431::Cheerios 18oz', '6431::Mini Wheats','6431::PL Honey Nut Oats', '6431::PL Frosted Wheat'], dtype='<U24')


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
2026-09-07T13:26:13.022100+00:00

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


``` python
hyper_vars = [*product_sites, "eps_scale", "cross_scale"]
summary = az.summary(tree, var_names=hyper_vars, ci_kind="hdi", ci_prob=0.94)
print(
    f"max r_hat: {summary['r_hat'].astype(float).max():.3f} | "
    f"min ess_bulk: {summary['ess_bulk'].astype(float).min():.0f}"
)
summary
```


    max r_hat: 1.010 | min ess_bulk: 608


|  | mean | sd | hdi94_lb | hdi94_ub | ess_bulk | ess_tail | r_hat | mcse_mean | mcse_sd |
|----|----|----|----|----|----|----|----|----|----|
| eps_prod\[HNC\] | -1.083 | 0.11 | -1.3 | -0.88 | 2491 | 2882 | 1.00 | 0.0022 | 0.0016 |
| eps_prod\[Cheerios 12oz\] | -0.381 | 0.081 | -0.53 | -0.23 | 2609 | 2527 | 1.00 | 0.0016 | 0.0011 |
| eps_prod\[Cheerios 18oz\] | -1.912 | 0.151 | -2.2 | -1.6 | 1492 | 2075 | 1.00 | 0.0039 | 0.0028 |
| eps_prod\[Mini Wheats\] | -1.946 | 0.126 | -2.2 | -1.7 | 3161 | 2903 | 1.00 | 0.0022 | 0.0016 |
| eps_prod\[PL Honey Nut Oats\] | -1.927 | 0.157 | -2.2 | -1.6 | 2908 | 3202 | 1.00 | 0.0029 | 0.0021 |
| eps_prod\[PL Frosted Wheat\] | -1.034 | 0.106 | -1.2 | -0.84 | 4621 | 3130 | 1.00 | 0.0016 | 0.0011 |
| conc\[HNC\] | 17.19 | 0.79 | 16 | 19 | 693 | 1865 | 1.00 | 0.03 | 0.021 |
| conc\[Cheerios 12oz\] | 19.18 | 0.83 | 18 | 21 | 2424 | 2505 | 1.00 | 0.017 | 0.012 |
| conc\[Cheerios 18oz\] | 21.95 | 1.04 | 20 | 24 | 4087 | 2338 | 1.00 | 0.016 | 0.012 |
| conc\[Mini Wheats\] | 18.85 | 0.87 | 17 | 21 | 3125 | 2535 | 1.00 | 0.016 | 0.011 |
| conc\[PL Honey Nut Oats\] | 19.6 | 1.23 | 17 | 22 | 608 | 1352 | 1.00 | 0.05 | 0.038 |
| conc\[PL Frosted Wheat\] | 21.13 | 0.99 | 19 | 23 | 2822 | 2625 | 1.00 | 0.019 | 0.013 |
| b_feat\[HNC\] | 0.757 | 0.042 | 0.68 | 0.84 | 2525 | 2938 | 1.00 | 0.00084 | 0.0006 |
| b_feat\[Cheerios 12oz\] | 0.799 | 0.062 | 0.69 | 0.92 | 1815 | 1948 | 1.00 | 0.0014 | 0.001 |
| b_feat\[Cheerios 18oz\] | 0.099 | 0.086 | -0.057 | 0.26 | 2090 | 2695 | 1.01 | 0.0019 | 0.0014 |
| b_feat\[Mini Wheats\] | 0.299 | 0.038 | 0.23 | 0.37 | 2610 | 2704 | 1.00 | 0.00074 | 0.00051 |
| b_feat\[PL Honey Nut Oats\] | 0.12 | 0.055 | 0.016 | 0.23 | 3296 | 2993 | 1.00 | 0.00096 | 0.00067 |
| b_feat\[PL Frosted Wheat\] | 0.238 | 0.042 | 0.16 | 0.32 | 3031 | 2801 | 1.00 | 0.00077 | 0.00055 |
| b_disp\[HNC\] | 0.342 | 0.053 | 0.24 | 0.44 | 2156 | 2567 | 1.00 | 0.0012 | 0.00083 |
| b_disp\[Cheerios 12oz\] | 0.469 | 0.054 | 0.37 | 0.57 | 2089 | 2506 | 1.00 | 0.0012 | 0.00082 |
| b_disp\[Cheerios 18oz\] | 0.265 | 0.05 | 0.17 | 0.36 | 3475 | 2987 | 1.00 | 0.00085 | 0.00061 |
| b_disp\[Mini Wheats\] | 0.282 | 0.065 | 0.16 | 0.41 | 2374 | 2508 | 1.00 | 0.0013 | 0.00095 |
| b_disp\[PL Honey Nut Oats\] | 0.268 | 0.054 | 0.16 | 0.37 | 3124 | 2957 | 1.00 | 0.00097 | 0.0007 |
| b_disp\[PL Frosted Wheat\] | 0.164 | 0.057 | 0.057 | 0.27 | 2889 | 2822 | 1.00 | 0.0011 | 0.00075 |
| b_fd\[HNC\] | -0.024 | 0.058 | -0.13 | 0.084 | 1729 | 2268 | 1.00 | 0.0014 | 0.00098 |
| b_fd\[Cheerios 12oz\] | -0.065 | 0.055 | -0.17 | 0.04 | 3422 | 3022 | 1.00 | 0.00095 | 0.00069 |
| b_fd\[Cheerios 18oz\] | -0.244 | 0.079 | -0.39 | -0.099 | 1296 | 1696 | 1.01 | 0.0022 | 0.0015 |
| b_fd\[Mini Wheats\] | 0.004 | 0.061 | -0.11 | 0.12 | 3125 | 2540 | 1.00 | 0.0011 | 0.00076 |
| b_fd\[PL Honey Nut Oats\] | -0.062 | 0.077 | -0.21 | 0.081 | 4237 | 3232 | 1.00 | 0.0012 | 0.00086 |
| b_fd\[PL Frosted Wheat\] | -0.011 | 0.062 | -0.13 | 0.1 | 4518 | 3155 | 1.00 | 0.00092 | 0.00067 |
| b_feat_depth\[HNC\] | 0.038 | 0.11 | -0.17 | 0.24 | 2119 | 2805 | 1.00 | 0.0024 | 0.0017 |
| b_feat_depth\[Cheerios 12oz\] | -0.462 | 0.211 | -0.87 | -0.065 | 1915 | 1926 | 1.00 | 0.0048 | 0.0034 |
| b_feat_depth\[Cheerios 18oz\] | 0.88 | 0.173 | 0.56 | 1.2 | 1824 | 2157 | 1.00 | 0.0041 | 0.003 |
| b_feat_depth\[Mini Wheats\] | -0.384 | 0.174 | -0.71 | -0.061 | 2016 | 2430 | 1.00 | 0.0039 | 0.0028 |
| b_feat_depth\[PL Honey Nut Oats\] | 0.15 | 0.398 | -0.62 | 0.88 | 3812 | 3309 | 1.00 | 0.0064 | 0.0046 |
| b_feat_depth\[PL Frosted Wheat\] | -0.301 | 0.238 | -0.75 | 0.15 | 3096 | 2849 | 1.00 | 0.0043 | 0.003 |
| b_disp_depth\[HNC\] | 0.037 | 0.118 | -0.19 | 0.26 | 2232 | 2689 | 1.00 | 0.0025 | 0.0017 |
| b_disp_depth\[Cheerios 12oz\] | -0.028 | 0.185 | -0.37 | 0.33 | 1904 | 2325 | 1.00 | 0.0043 | 0.003 |
| b_disp_depth\[Cheerios 18oz\] | -0.045 | 0.164 | -0.36 | 0.25 | 1278 | 1900 | 1.01 | 0.0046 | 0.0033 |
| b_disp_depth\[Mini Wheats\] | 0.438 | 0.222 | 0.022 | 0.86 | 2542 | 2826 | 1.00 | 0.0044 | 0.0031 |
| b_disp_depth\[PL Honey Nut Oats\] | -1.15 | 0.356 | -1.8 | -0.5 | 2599 | 2643 | 1.00 | 0.007 | 0.0049 |
| b_disp_depth\[PL Frosted Wheat\] | 0.45 | 0.324 | -0.16 | 1.1 | 2610 | 2905 | 1.00 | 0.0063 | 0.0045 |
| b_sib_feat\[HNC\] | -0.0706 | 0.0155 | -0.1 | -0.042 | 3517 | 3269 | 1.00 | 0.00026 | 0.00018 |
| b_sib_feat\[Cheerios 12oz\] | -0.0197 | 0.0146 | -0.048 | 0.0079 | 3658 | 3170 | 1.00 | 0.00024 | 0.00018 |
| b_sib_feat\[Cheerios 18oz\] | 0.035 | 0.0146 | 0.007 | 0.063 | 4401 | 3154 | 1.00 | 0.00022 | 0.00015 |
| b_sib_feat\[Mini Wheats\] | 0.0079 | 0.0165 | -0.024 | 0.039 | 3941 | 2985 | 1.00 | 0.00026 | 0.00018 |
| b_sib_feat\[PL Honey Nut Oats\] | 0.0245 | 0.0173 | -0.0084 | 0.057 | 3545 | 2901 | 1.00 | 0.00029 | 0.00021 |
| b_sib_feat\[PL Frosted Wheat\] | 0.0217 | 0.0151 | -0.0071 | 0.05 | 3907 | 3007 | 1.00 | 0.00024 | 0.00017 |
| b_sib_disp\[HNC\] | -0.0536 | 0.0156 | -0.083 | -0.024 | 4313 | 2949 | 1.00 | 0.00024 | 0.00017 |
| b_sib_disp\[Cheerios 12oz\] | 0.0077 | 0.0157 | -0.022 | 0.037 | 3946 | 3020 | 1.00 | 0.00025 | 0.00018 |
| b_sib_disp\[Cheerios 18oz\] | -0.0234 | 0.0147 | -0.051 | 0.0051 | 4000 | 3000 | 1.00 | 0.00023 | 0.00017 |
| b_sib_disp\[Mini Wheats\] | -0.0236 | 0.0161 | -0.054 | 0.0073 | 4948 | 3121 | 1.00 | 0.00023 | 0.00016 |
| b_sib_disp\[PL Honey Nut Oats\] | -0.0189 | 0.0168 | -0.051 | 0.013 | 4047 | 3222 | 1.00 | 0.00026 | 0.00019 |
| b_sib_disp\[PL Frosted Wheat\] | 0.016 | 0.0151 | -0.013 | 0.044 | 4020 | 2976 | 1.00 | 0.00024 | 0.00017 |
| eps_scale | 0.294 | 0.0328 | 0.24 | 0.36 | 2914 | 2914 | 1.00 | 0.00061 | 0.00043 |
| cross_scale | 0.273 | 0.04 | 0.21 | 0.36 | 1328 | 1942 | 1.00 | 0.0011 | 0.00088 |


``` python
az.summary(tree, var_names=["gamma_offdiag"], ci_kind="hdi", ci_prob=0.94)
```


|  | mean | sd | hdi94_lb | hdi94_ub | ess_bulk | ess_tail | r_hat | mcse_mean | mcse_sd |
|----|----|----|----|----|----|----|----|----|----|
| gamma_offdiag\[HNC -\> Cheerios 12oz\] | 0.11 | 0.048 | 0.019 | 0.2 | 3393 | 2868 | 1.00 | 0.00083 | 0.00057 |
| gamma_offdiag\[HNC -\> Cheerios 18oz\] | 0.017 | 0.047 | -0.073 | 0.1 | 3511 | 2855 | 1.00 | 0.00079 | 0.00057 |
| gamma_offdiag\[HNC -\> Mini Wheats\] | -0.025 | 0.048 | -0.12 | 0.063 | 4237 | 3108 | 1.00 | 0.00074 | 0.00053 |
| gamma_offdiag\[HNC -\> PL Honey Nut Oats\] | 0.695 | 0.053 | 0.59 | 0.79 | 5078 | 3882 | 1.00 | 0.00075 | 0.00054 |
| gamma_offdiag\[HNC -\> PL Frosted Wheat\] | 0.079 | 0.0453 | -0.0059 | 0.16 | 4401 | 3288 | 1.00 | 0.00068 | 0.00045 |
| gamma_offdiag\[Cheerios 12oz -\> HNC\] | -0.25 | 0.0393 | -0.32 | -0.18 | 4343 | 3033 | 1.00 | 0.0006 | 0.00042 |
| gamma_offdiag\[Cheerios 12oz -\> Cheerios 18oz\] | 0.15 | 0.0353 | 0.084 | 0.22 | 4075 | 2895 | 1.00 | 0.00055 | 0.0004 |
| gamma_offdiag\[Cheerios 12oz -\> Mini Wheats\] | 0.032 | 0.0379 | -0.04 | 0.1 | 4121 | 3014 | 1.00 | 0.00059 | 0.00044 |
| gamma_offdiag\[Cheerios 12oz -\> PL Honey Nut Oats\] | -0.056 | 0.0411 | -0.14 | 0.02 | 3980 | 3025 | 1.00 | 0.00065 | 0.00046 |
| gamma_offdiag\[Cheerios 12oz -\> PL Frosted Wheat\] | -0.008 | 0.0348 | -0.074 | 0.058 | 4059 | 3129 | 1.00 | 0.00055 | 0.00039 |
| gamma_offdiag\[Cheerios 18oz -\> HNC\] | -0.309 | 0.0454 | -0.39 | -0.23 | 5040 | 2907 | 1.00 | 0.00064 | 0.00046 |
| gamma_offdiag\[Cheerios 18oz -\> Cheerios 12oz\] | 0.217 | 0.0453 | 0.13 | 0.3 | 4624 | 3392 | 1.00 | 0.00067 | 0.00048 |
| gamma_offdiag\[Cheerios 18oz -\> Mini Wheats\] | 0.008 | 0.046 | -0.078 | 0.097 | 3938 | 3032 | 1.00 | 0.00074 | 0.00052 |
| gamma_offdiag\[Cheerios 18oz -\> PL Honey Nut Oats\] | 0.039 | 0.047 | -0.046 | 0.13 | 3403 | 2590 | 1.00 | 0.00081 | 0.00057 |
| gamma_offdiag\[Cheerios 18oz -\> PL Frosted Wheat\] | 0.127 | 0.0427 | 0.046 | 0.21 | 3854 | 2985 | 1.00 | 0.00069 | 0.00049 |
| gamma_offdiag\[Mini Wheats -\> HNC\] | -0.104 | 0.063 | -0.23 | 0.015 | 4504 | 3023 | 1.00 | 0.00094 | 0.00065 |
| gamma_offdiag\[Mini Wheats -\> Cheerios 12oz\] | -0.198 | 0.06 | -0.31 | -0.085 | 4504 | 3368 | 1.00 | 0.0009 | 0.00063 |
| gamma_offdiag\[Mini Wheats -\> Cheerios 18oz\] | 0.179 | 0.062 | 0.063 | 0.29 | 4029 | 2812 | 1.00 | 0.00097 | 0.0007 |
| gamma_offdiag\[Mini Wheats -\> PL Honey Nut Oats\] | -0.064 | 0.066 | -0.19 | 0.058 | 4012 | 2689 | 1.00 | 0.001 | 0.00072 |
| gamma_offdiag\[Mini Wheats -\> PL Frosted Wheat\] | 0.535 | 0.06 | 0.42 | 0.65 | 4882 | 3361 | 1.00 | 0.00086 | 0.0006 |
| gamma_offdiag\[PL Honey Nut Oats -\> HNC\] | -0.075 | 0.111 | -0.28 | 0.13 | 4084 | 3211 | 1.00 | 0.0017 | 0.0012 |
| gamma_offdiag\[PL Honey Nut Oats -\> Cheerios 12oz\] | -0.697 | 0.114 | -0.91 | -0.49 | 4184 | 3204 | 1.00 | 0.0018 | 0.0012 |
| gamma_offdiag\[PL Honey Nut Oats -\> Cheerios 18oz\] | -0.146 | 0.105 | -0.34 | 0.05 | 4293 | 3058 | 1.00 | 0.0016 | 0.0011 |
| gamma_offdiag\[PL Honey Nut Oats -\> Mini Wheats\] | 0.145 | 0.111 | -0.065 | 0.35 | 4105 | 3211 | 1.00 | 0.0017 | 0.0012 |
| gamma_offdiag\[PL Honey Nut Oats -\> PL Frosted Wheat\] | 0.279 | 0.104 | 0.079 | 0.48 | 4372 | 3101 | 1.00 | 0.0016 | 0.0011 |
| gamma_offdiag\[PL Frosted Wheat -\> HNC\] | 0.093 | 0.076 | -0.052 | 0.24 | 3040 | 2676 | 1.00 | 0.0014 | 0.00096 |
| gamma_offdiag\[PL Frosted Wheat -\> Cheerios 12oz\] | 0.236 | 0.073 | 0.099 | 0.37 | 4161 | 3129 | 1.00 | 0.0011 | 0.0008 |
| gamma_offdiag\[PL Frosted Wheat -\> Cheerios 18oz\] | -0.043 | 0.072 | -0.18 | 0.09 | 4096 | 3145 | 1.00 | 0.0011 | 0.00081 |
| gamma_offdiag\[PL Frosted Wheat -\> Mini Wheats\] | 0.087 | 0.075 | -0.051 | 0.23 | 4081 | 3163 | 1.00 | 0.0012 | 0.0008 |
| gamma_offdiag\[PL Frosted Wheat -\> PL Honey Nut Oats\] | 0.366 | 0.079 | 0.22 | 0.52 | 3550 | 3271 | 1.00 | 0.0013 | 0.00092 |


``` python
pc_trace = az.plot_trace_dist(
    tree,
    var_names=["eps_prod", "eps_scale", "b_feat", "b_disp", "conc", "cross_scale"],
    compact=True,
    figure_kwargs={"figsize": (12, 14)},
)
pc_trace.viz["figure"].item().suptitle("Trace plots", fontsize=18, fontweight="bold", y=1.02);
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-38-output-1.png" class="figure-img" width="1211" height="1443" /></p>
</figure>


# What the model learned

The first table puts the posterior of each product's own elasticity next to the least-squares estimates with and without the depth slopes. A model whose feature and display effects fall far below the least-squares ones would be a model whose random-walk level absorbs the promotion spikes; the second table compares the mechanics multipliers. The forest plots show the own elasticities and the mechanics effects with their 50\\ and 94\\ HDIs, annotated with the identifying store-weeks; the heatmap shows the cross matrix, its cells annotated with the posterior probability of a positive cross elasticity (a positive value means that a cheaper competitor takes units away). The store-level plot of the focal product's elasticity shows the partial pooling at work: the within-store least-squares estimates scatter widely, the posterior medians shrink toward the product mean.

The focal product's elasticity has a posterior median of -1.08 (94\\ HDI -1.29 to -0.88), on top of the least-squares estimate with depth slopes. Cheerios 18 oz, with 23 identifying store-weeks, still gets a median of -1.91 (HDI -2.21 to -1.64), because its cut depth varies inside its feature and display weeks. The feature multiplier of the focal product is 2.13 (HDI 1.96 to 2.29) and the display multiplier 1.41 (HDI 1.27 to 1.55), above and near the pooled least-squares multipliers of 1.65 and 1.49, so the level is not absorbing the promotion spikes. The depth slopes under feature and display of the focal product are centered near zero, so its mechanics-specific elasticities differ little from the plain one. The cross matrix has one large cell: the focal product's price on the private-label twin's units, +0.69 (HDI +0.59 to +0.79, posterior probability of a positive value 1.00), while the twin's price barely moves the focal product (-0.08, HDI -0.28 to +0.13). The store-level elasticities of the focal product have a least-squares spread of 0.37 across stores and a posterior-median spread of 0.31, the shrinkage the planner section uses. The posterior-mean seasonal component peaks in horizon week 12, the week of December 28, and the level innovation scales range from 0.019 to 0.218 across series.


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
| "HNC" | -1.326977 | -1.084921 | -1.082029 | -1.139235 | -0.994873 | -1.289832 | -0.875182 |
| "Cheerios 12oz" | -0.3183 | -0.31104 | -0.382625 | -0.429511 | -0.323211 | -0.537446 | -0.232483 |
| "Cheerios 18oz" | -1.961065 | -1.511146 | -1.905879 | -1.998456 | -1.794018 | -2.209666 | -1.642067 |
| "Mini Wheats" | -1.558958 | -1.626816 | -1.94702 | -2.027605 | -1.857185 | -2.186641 | -1.711004 |
| "PL Honey Nut Oats" | -1.446412 | -1.468914 | -1.927437 | -2.046492 | -1.83645 | -2.222483 | -1.633524 |
| "PL Frosted Wheat" | -0.8505 | -0.886696 | -1.032909 | -1.094682 | -0.956179 | -1.218427 | -0.823438 |


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
| "feature" | "HNC" | 1.654384 | 2.131905 | 2.06832 | 2.187409 | 1.956841 | 2.291173 |
| "feature" | "Cheerios 12oz" | 1.654384 | 2.219052 | 2.126567 | 2.310077 | 1.980169 | 2.486434 |
| "feature" | "Cheerios 18oz" | 1.654384 | 1.102757 | 1.036413 | 1.163704 | 0.94592 | 1.291129 |
| "feature" | "Mini Wheats" | 1.654384 | 1.348858 | 1.318365 | 1.387212 | 1.257218 | 1.445063 |
| "feature" | "PL Honey Nut Oats" | 1.654384 | 1.127978 | 1.079181 | 1.16256 | 1.009625 | 1.245206 |
| "feature" | "PL Frosted Wheat" | 1.654384 | 1.269117 | 1.227314 | 1.298753 | 1.173447 | 1.37554 |
| "display" | "HNC" | 1.494406 | 1.408505 | 1.356387 | 1.455645 | 1.266144 | 1.546968 |
| "display" | "Cheerios 12oz" | 1.494406 | 1.599538 | 1.540608 | 1.657762 | 1.447606 | 1.768491 |
| "display" | "Cheerios 18oz" | 1.494406 | 1.303566 | 1.257622 | 1.343683 | 1.18035 | 1.426616 |
| "display" | "Mini Wheats" | 1.494406 | 1.325217 | 1.25908 | 1.37297 | 1.16014 | 1.487416 |
| "display" | "PL Honey Nut Oats" | 1.494406 | 1.306934 | 1.260018 | 1.353727 | 1.175031 | 1.442661 |
| "display" | "PL Frosted Wheat" | 1.494406 | 1.177952 | 1.133142 | 1.222175 | 1.054195 | 1.302381 |


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
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-41-output-1.png" class="figure-img" width="1511" height="611" /></p>
</figure>


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
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-42-output-1.png" class="figure-img" width="909" height="711" /></p>
</figure>


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


    HNC price on PL Honey Nut Oats units: median +0.69, 94% HDI +0.59 to +0.79, P(> 0) 1.00
    PL Honey Nut Oats price on HNC units: median -0.08, 94% HDI -0.28 to +0.13, P(> 0) 0.25


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
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-44-output-2.png" class="figure-img" width="1150" height="711" /></p>
</figure>


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
    drift scale posterior medians across series: 0.019 to 0.218


# In-sample fit and holdout forecast

We draw the in-sample posterior predictive and the holdout forecast with the realized covariates and score them with the continuous ranked probability score (CRPS), the mean absolute error, and the coverage of the central 50\\ and 94\\ intervals (central intervals, while the figures draw HDI bands). The seasonal naive comparator is the two-member ensemble of the units 52 and 104 weeks earlier, and the MASE scale is computed per product on its training block, because a pooled scale would be dominated by the high-volume products. Calibration is read the way [Gneiting and Katzfuss (2014)](https://doi.org/10.1146/annurev-statistics-062713-085831) frame it, sharpness subject to calibration, with the randomized probability integral transform (PIT) for counts of [Czado, Gneiting and Held (2009)](https://doi.org/10.1111/j.1541-0420.2009.01191.x): for a count y with predictive CDF G, u = G(y - 1) + v\\(G(y) - G(y - 1)) with v \sim \text{Uniform}(0, 1) is uniform for a calibrated forecast, U-shaped for an under-dispersed one and hump-shaped for an over-dispersed one. Two cautions: the holdout is the holiday quarter, with only two earlier Decembers in the training window; and the holdout cells of one series share a level path, so the effective sample size behind the histogram is well below the number of cells.

The holdout CRPS is 12.98 against 22.13 for the seasonal naive ensemble and the mean absolute error 17.65 against 26.57; the central 50\\ and 94\\ intervals cover 56\\ and 92\\ of the 1{,}404 holdout cells (in sample, 62\\ and 96\\). Per product the model's MASE is below the naive one and below one everywhere, with the focal product the hardest at 0.95 against 1.56 and a 94\\ coverage of 0.84 in its promotion-heavy quarter. The model beats the naive forecast in every horizon week except week 12, the week of the deepest realized cut. The PIT histogram slopes downward, with 16\\ of the cells in the lowest decile and 4\\ in the highest: the holdout forecasts run high on average, so the calibration is good but not perfect, and the effective sample size behind the histogram is far below 1{,}404.


``` python
rng_key, key_in, key_fc = random.split(rng_key, 3)
pred_train = np.asarray(
    predict_in_sample(key_in, nuts_model, posterior, covariates_train, device="host"),
    dtype=np.float32,
)
pred_test = np.asarray(
    forecast(key_fc, nuts_model, posterior, y_train, covariates, device="host"), dtype=np.float32
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
| "model (train)"         | 7.917095  | 10.87364  | 0.62432     | 0.959596    |
| "model (test)"          | 12.975068 | 17.647436 | 0.561254    | 0.919516    |
| "seasonal naive (test)" | 22.127493 | 26.566952 | 0.151709    | 0.29416     |


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
| "HNC"               | 37.492382  | 69.972229  | 0.952045   | 1.555362   | 0.837607    |
| "Cheerios 12oz"     | 11.245326  | 21.534189  | 0.550353   | 0.976913   | 0.965812    |
| "Cheerios 18oz"     | 4.746489   | 6.376069   | 0.255705   | 0.293871   | 0.965812    |
| "Mini Wheats"       | 6.565374   | 10.685898  | 0.412274   | 0.560507   | 0.978633    |
| "PL Honey Nut Oats" | 7.784356   | 10.254274  | 0.809331   | 0.926579   | 0.893162    |
| "PL Frosted Wheat"  | 10.016487  | 13.942308  | 0.647122   | 0.811573   | 0.876068    |


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


    PIT bin shares: [0.159 0.108 0.122 0.107 0.11  0.115 0.099 0.084 0.061 0.036]


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-48-output-2.png" class="figure-img" width="1411" height="511" /></p>
</figure>


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
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-49-output-1.png" class="figure-img" width="1411" height="1073" /></p>
</figure>


# Counterfactual promotions

A counterfactual promotion is a change of the horizon covariates, nothing else: the posterior draws stay fixed, the same PRNG key is reused, and the model is run again through NumPyro's `Predictive`. Nothing is refit. The library's [forecast](../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast) returns the sampled units; the decision layer also needs the conditional mean \mu over the horizon and the future level innovations, so we wrap `Predictive` ourselves with the model as a static argument, the same pattern the library uses, and ask for three sites. The first thing to do with the wrapper is to check that it reproduces the [forecast](../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast) draws of the previous section bit for bit under the same key: that validates the engine and the key discipline, not any causal claim.

A policy is a discount depth d and a mechanics m for the focal product in every panel store during one contiguous two-week event, horizon weeks 7 and 8, the realized Thanksgiving slot. Every other product sits at its base price with no promotion over the whole horizon, and the grid below is a response surface for the break-even and risk analyses, not a search space. The calendar is not a lever here for two reasons printed earlier: the post-promotion dip is economically negligible, and under a multiplicative model the timing question reduces to the seasonal peak, whose posterior-mean week was printed above. The grid runs from no cut to a 40\\ cut in steps of five points for each of the four mechanics; for the shelf-tag-only mechanics the zero-depth cell is the no-promotion baseline itself. Cells with fewer than 20 observed store-weeks within \pm 2.5 points of the depth are shaded in the figures as thin support. Under one key the future level innovations are bit-identical across policies and the conditional means agree wherever the covariates agree, while the sampled units are coupled but not identical (common random numbers), which the cells below print.

The engine check prints a maximum absolute difference of 0.0 between the wrapper's draws and the [forecast](../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast) draws under the same key. Across two policies the future level innovations differ by 0.0, the conditional means differ by 0.0 outside the event weeks, and the focal product's event draws have a correlation of 0.99. The 36 policies take under 30 seconds on 4{,}000 draws each. The figures show the feature-with-display event lifting the focal product's zone-level units to more than three times the no-promotion level in the event weeks, and lowering the private-label twin's units in the same weeks.


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

engine = scenario_draws(key_fc, nuts_model, posterior, y_train, covariates)
engine_gap = float(np.abs(np.asarray(engine["forecast"], dtype=np.float32) - pred_test).max())
print(
    f"engine check: max |scenario draws - forecast() draws| under the same key = {engine_gap:.1f}"
)
```


    engine check: max |scenario draws - forecast() draws| under the same key = 0.0


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


``` python
rng_key, key_policy = random.split(rng_key)
out_a = scenario_draws(key_policy, nuts_model, posterior, y_train, policy_a)
out_b = scenario_draws(key_policy, nuts_model, posterior, y_train, policy_b)
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
        key_policy,
        nuts_model,
        posterior,
        y_train,
        policy_covariates(depth, *MECHANICS[mechanics_name]),
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
    CPU times: user 4min 40s, sys: 5.78 s, total: 4min 46s
    Wall time: 26.9 s


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


``` python
plot_zone_policies(
    [("feature + display", 0.15)], [FOCAL, "PL Honey Nut Oats"], figsize=(14.0, 5.0)
)
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-55-output-1.png" class="figure-img" width="1411" height="528" /></p>
</figure>


``` python
plot_zone_policies(
    [("TPR-only", 0.15), ("feature", 0.15), ("feature + display", 0.15)],
    [FOCAL, "PL Honey Nut Oats"],
    figsize=(16.0, 8.0),
)
```


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-56-output-1.png" class="figure-img" width="1611" height="837" /></p>
</figure>


# From units to profit

The model forecasts units. The decisions need money. This section states the economics once, prints them, and turns the forecast units of every draw into an event profit. The break-even, risk and order sections then work on that profit.


## The margin of a promoted unit

Take product j at store s. Let p\_{j,s} be its base price at the last training week. Base prices differ across stores, so every currency number below uses the store's own price. Let g_j be the gross margin on the base price: 0.28 for the national brands and 0.38 for the private label. The unit cost is the part of the base price that is not margin:

 c\_{j,s} = (1 - g_j)\\ p\_{j,s}. 

Now cut the price of the focal product \text{H} by a fraction d of its base price. The manufacturer refunds a share \alpha of the discount on every unit sold in a promotion week. We call this refund the allowance. Three flows set the margin of one promoted unit. The retailer sells the unit at the cut price. The retailer pays the unit cost. The retailer receives the allowance. The sum of the three flows is the promo-week unit margin:

\begin{align\*} m\_{\text{H},s}(d) &= p\_{\text{H},s}\\(1 - d) - c\_{\text{H},s} + \alpha\\ d\\ p\_{\text{H},s} \\ &= p\_{\text{H},s}\\ \big( g\_{\text{H}} - (1 - \alpha)\\ d \big). \end{align\*}

The second line is the rule to remember. Each point of depth costs the retailer (1 - \alpha) points of margin rate, and the manufacturer pays the rest. At the median focal base price of 3.02 the margin is 0.85 per unit at full price. A 15\\ cut at the nominal share \alpha = 0.5 brings it to 0.62. An unfunded 30\\ cut brings it to -0.06: the cut is deeper than the margin rate, so every unit sells below cost. The two cells below print the economics table and these worked margins.

The siblings k \ne \text{H} stay at their base price during the event, so their margin is the base-price margin:

 m\_{k,s} = g_k\\ p\_{k,s}. 

A feature or a display also uses a slot. Its cost S_m per store-week is one number per mechanics, and under feature with display it covers both slots. We never assume a value for S_m. The next section solves for it.


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


``` python
G_FOCAL = float(gross_margin[is_focal][0])
p_focal_median = float(np.median(base_price[is_focal]))
margin_depths = [0.0, 0.15, 0.30]
margin_shares = [0.0, 0.5, 1.0]
unit_margin_table = pl.DataFrame(
    {
        "depth": margin_depths,
        **{
            f"unit margin at alpha = {alpha_value:.1f}": [
                round(p_focal_median * (G_FOCAL - (1.0 - alpha_value) * d), 2)
                for d in margin_depths
            ]
            for alpha_value in margin_shares
        },
    }
)
print(
    f"median focal base price {p_focal_median:.2f} | gross margin {G_FOCAL:.2f} | threshold "
    "elasticity (1 - alpha) / g: "
    + " | ".join(f"alpha {a:.1f}: {(1.0 - a) / G_FOCAL:.2f}" for a in margin_shares)
)
unit_margin_table
```


    median focal base price 3.02 | gross margin 0.28 | threshold elasticity (1 - alpha) / g: alpha 0.0: 3.57 | alpha 0.5: 1.79 | alpha 1.0: 0.00


| depth | unit margin at alpha = 0.0 | unit margin at alpha = 0.5 | unit margin at alpha = 1.0 |
|----|----|----|----|
| 0.0 | 0.85 | 0.85 | 0.85 |
| 0.15 | 0.39 | 0.62 | 0.85 |
| 0.3 | -0.06 | 0.39 | 0.85 |


## The event profit and the funding share

A policy a = (d, m) fixes a depth d and a mechanics m for the two event weeks. The quantity we want is the expected event profit of a policy under the posterior predictive, \text{E}\[\Pi(a)\]. This is the estimand of the decision layer. The rules of the next sections compare policies on it, on its lower tail, and on the order quantity.

We estimate it from posterior draws. Let r = 1, \dots, R index one draw of the parameters and of the future level path; we write \theta_r for both together. Let E = \\7, 8\\ be the set of event weeks. Let y\_{r,t,i}(a) be the sampled unit count of series i in week t under policy a in draw r; it includes the demand noise. Write i \in \text{H} for the 18 focal series, one per store, and i \notin \text{H} for the sibling series. Let \text{margin}\_{i,t}(a) be the unit margin of series i under the policy: m\_{\text{H},s(i)}(d) for a focal series and m\_{\text{prod}(i),s(i)} for a sibling series. The event profit of draw r adds the margin dollars of every series over the event weeks and subtracts the slot cost:

 \Pi_r(a) = \sum\_{t \in E} \sum\_{i} \text{margin}\_{i,t}(a)\\ y\_{r,t,i}(a) - S(a), \qquad S(a) = 2 \times 18 \times S_m. 

The slot is paid in each of the two event weeks in each of the 18 stores. We set S = 0 in every break-even and risk computation. Let a_0 be the no-promotion baseline and \Delta\Pi_r(a) = \Pi_r(a) - \Pi_r(a_0) the incremental profit of a policy. An assumed slot cost shifts every histogram of \Delta\Pi by a constant, so \text{P}(\Delta\Pi \< S) can be read off the same figures.

The units do not depend on \alpha, because the model has no funding term. Only the margin does, and it is linear in \alpha. So the event profit of a draw is a straight line in the funding share:

 \Pi_r(a) = A_r(a) + \alpha\\ B_r(a), 

with

\begin{align\*} A_r(a) &= \sum\_{t \in E} \sum\_{i \in \text{H}} p\_{\text{H},s(i)}\\(g\_{\text{H}} - d)\\ y\_{r,t,i}(a) + \sum\_{t \in E} \sum\_{i \notin \text{H}} m\_{\text{prod}(i),s(i)}\\ y\_{r,t,i}(a) - S(a), \\ B_r(a) &= d \sum\_{t \in E} \sum\_{i \in \text{H}} p\_{\text{H},s(i)}\\ y\_{r,t,i}(a). \end{align\*}

A_r(a) is the event profit when the retailer funds the whole cut. B_r(a) is the discount dollars given away on the focal units sold: depth times base price times units. The allowance refunds the share \alpha of these dollars. B_r(a) is positive whenever d \> 0, so a higher funding share always raises profit. One set of draws therefore serves every funding share, and `profit_parts` in the next cell returns exactly these two parts.

Profit is linear in units. So, given the parameters and the level path of draw r, the expected profit is the same sum with the conditional means \mu\_{r,t,i}(a) in place of the sampled counts:

 \Pi^\mu_r(a) = \text{E}\[\Pi(a) \mid \theta_r\] = \sum\_{t \in E} \sum\_{i} \text{margin}\_{i,t}(a)\\ \mu\_{r,t,i}(a) - S(a). 

The average of \Pi^\mu_r(a) over the draws estimates the estimand without any observation-noise error:

 \text{E}\[\Pi(a)\] = \text{E}\big\[\text{E}\[\Pi(a) \mid \theta\]\big\] \approx \frac{1}{R} \sum\_{r=1}^{R} \Pi^\mu_r(a). 

The bands of \Pi^\mu in the next section still carry the sampled level path. The sampled counts return in the risk and order sections, where the demand noise matters. The next cell builds the two parts and prints the no-promotion event profit. Over the 18 stores its mean is 10{,}269 currency units. The parameters and the level path give it a standard deviation of 221; with the demand noise it is 312.


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


    no-promotion event profit over 18 stores: mean 10,269 | sd from parameters and level path 221 | sd including demand noise 312


## A single-product rule of thumb

The numerical shares below come from the full model. A one-product version shows the mechanics of the answer. Let N_0 be the expected event units of the focal product in one store at base price and without mechanics. From the model section, a mechanics m multiplies the units by the uplift e^{b_m} and a cut of depth d by (1 - d)^{\varepsilon_m}, with \varepsilon_m \< 0 the mechanics-specific elasticity. Write m\_{\text{H}}(d) for the promo-week unit margin of the store. The profit of the focal product alone is margin per unit times units, minus the slot:

 \pi_m(d) = m\_{\text{H}}(d)\\ N_0\\ (1 - d)^{\varepsilon_m}\\ e^{b_m} - S_m. 

Its derivative in d has the sign of

 \big(-g\_{\text{H}}\\ \varepsilon_m - (1 - \alpha)\big) + (1 - \alpha)(1 + \varepsilon_m)\\ d. 

The first bracket is the trade-off at the first point of depth: the point raises units by -\varepsilon_m percent, each worth g\_{\text{H}} of margin rate, and it costs (1 - \alpha) points of margin rate. So a first point of depth pays if and only if \varepsilon_m \< -\varepsilon^\star(\alpha), with the threshold elasticity

 \varepsilon^\star(\alpha) = \frac{1 - \alpha}{g\_{\text{H}}}. 

The cell above prints it: 3.57 when the retailer funds the whole cut, 1.79 at the nominal share, 0 when the manufacturer funds everything. Solve the same condition for \alpha instead, and you get the brand-only break-even share of a mechanics:

 \alpha^\star_m = 1 + g\_{\text{H}}\\ \varepsilon_m. 

It reads as follows. At or below 0 the cut pays even if the retailer funds it alone. At or above 1 no funding share makes it pay. The tables report its intervals unclipped with this rule. Cannibalization raises the share, because the cut moves sibling units through the cross elasticities \gamma\_{\text{H},k}. Let \tilde N\_{j,s,m} be the expected event units of product j at store s at zero depth under mechanics m. The category break-even share is

 \alpha^\star\_{\text{cat},m} = \alpha^\star_m + \frac{\sum_s \sum\_{k \ne \text{H}} m\_{k,s}\\ \tilde N\_{k,s,m}\\ \gamma\_{\text{H},k}}{\sum_s p\_{\text{H},s}\\ \tilde N\_{\text{H},s,m}}. 

The numerator is the sibling margin lost per unit of log price change; the denominator is the focal product's promo-week gross revenue at zero depth. The ratio is the extra share the manufacturer must fund to cover what the siblings lose. With \|\varepsilon_m\| \le 1 the second bracket of the derivative grows with d, so the best cell of a depth grid is a corner:

- Below the event-level share of the deepest cell, the shallowest cell is best.
- Above the tangent share \alpha^\star_m, the deepest cell is best.
- In the narrow band between them, compare the two corners directly.

With \|\varepsilon_m\| \> 1 an interior depth exists for a narrow band of funding shares. The notebook computes two numerical shares per draw from the stored conditional means, so that the cross terms and the sibling flags enter through the model. The secant share sets the profits of the two shallowest cells of a mechanics equal, a\_{\text{lo}} and a\_{\text{hi}} (zero and five points for a flagged mechanics, no promotion and ten points for a shelf tag alone):

 \alpha^{\text{sec}}\_{r,m} = \frac{A_r(a\_{\text{lo}}) - A_r(a\_{\text{hi}})}{B_r(a\_{\text{hi}}) - B_r(a\_{\text{lo}})}. 

Its brand-only version drops the sibling terms from A_r. The event-level share of a grid cell a is the funding share at which the cell breaks even against no promotion:

 \alpha^{\text{ev}}\_r(a) = -\frac{A_r(a) - A_r(a_0)}{B_r(a)}. 

A negative value means the cell pays unfunded; a value above one means no funding share makes it pay. Three variants, as remarks:

- A lump-sum allowance L instead of a per-unit one adds L to A_r and removes \alpha B_r, so the break-even question becomes a question about L.
- A perishable product replaces the holding cost of the order section by a write-off of the leftover.
- The \Pi^\mu argument only uses linearity in units, so it holds under a Poisson likelihood or any likelihood whose conditional mean the model registers.


# Break-even: who funds the discount and what is a slot worth


## The funding share

The figure shows the expected event profit of the category against the depth of the focal product's cut, one row per mechanics and one column per funding share: the nominal 0.5, the posterior median of the category break-even share of the feature-with-display mechanics, and that share plus 0.15. The bands are the 50\\ and 94\\ HDIs across draws of \Pi^\mu, so they carry the parameter and level-path uncertainty and not the demand noise; shaded depths have thin support. The printed tables give, per mechanics, the probability that no promotion or the shallowest cell is the best choice at the nominal share, the range of expected profit across the grid at the break-even share, the threshold table, and the posteriors of the break-even shares themselves: brand-only, category, and the event-level share of every grid cell. A negative event-level share means the event pays even if the retailer funds the whole cut; a share above one means no funding makes it pay.

The posterior median of the category break-even share of feature with display is 0.71, which becomes \tilde\alpha; the columns of the figure are 0.50, 0.71 and 0.86. The tables bracket the decision:

- At the nominal share every curve falls with depth. The probability that no promotion or the shallowest cell is optimal is 1.00 for all four mechanics, and the expected profit lost between the shallowest and the deepest cell is 17\\ of the baseline event profit for a shelf-tag cut and 40\\ for feature with display.
- At \tilde\alpha the curves are flat. The range of expected profit across the grid is 1.5\\ of the baseline for feature with display and 3.2\\ for a shelf-tag cut, the deepest cell is optimal in 63\\ of the draws for feature with display, and the value of perfect information about the parameters and the level path is at most 1.1\\ of the baseline.
- The threshold table: at \alpha = 0.5 the threshold elasticity is -1.79 and the posterior probability of exceeding it is 0.00 under every mechanics; at \tilde\alpha the probability is 0.85 for feature with display from the brand-only tangent but 0.50 from the category secant, the difference being the cannibalization of the siblings; at \alpha = 0.9 every probability is 1.00.
- The break-even shares: the brand-only shares sit between 0.68 and 0.70 for the four mechanics with 94\\ HDIs about 0.1 wide, and cannibalization adds 0.03 under feature with display and 0.09 under a shelf-tag cut.
- What a 15\\ cut needs: under feature with display the event pays unfunded, with an event-level share of -0.32, under a feature alone -0.12; the same cut under a shelf tag needs a share of 0.79, and a 30\\ cut under feature with display needs 0.30.


``` python
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


    posterior median category break-even share of feature + display: 0.711 (used as alpha tilde = 0.711)


| mechanics | share | median | hdi50_lower | hdi50_upper | hdi94_lower | hdi94_upper |
|----|----|----|----|----|----|----|
| "TPR-only" | "brand-only secant" | 0.697013 | 0.683315 | 0.714202 | 0.654446 | 0.741138 |
| "TPR-only" | "category secant" | 0.790207 | 0.773714 | 0.812475 | 0.73587 | 0.847788 |
| "TPR-only" | "brand-only tangent" | 0.697032 | 0.681014 | 0.721436 | 0.638847 | 0.754949 |
| "TPR-only" | "cannibalization (category - brand)" | 0.092827 | 0.080796 | 0.104642 | 0.061416 | 0.126411 |
| "display" | "brand-only secant" | 0.687541 | 0.669759 | 0.710609 | 0.631388 | 0.745311 |
| "display" | "category secant" | 0.756083 | 0.734172 | 0.776233 | 0.696231 | 0.814705 |
| "display" | "brand-only tangent" | 0.686353 | 0.662039 | 0.711423 | 0.622384 | 0.757348 |
| "display" | "cannibalization (category - brand)" | 0.068226 | 0.060932 | 0.079043 | 0.043099 | 0.092802 |
| "feature" | "brand-only secant" | 0.686663 | 0.6644 | 0.704854 | 0.630727 | 0.745228 |
| "feature" | "category secant" | 0.73353 | 0.71566 | 0.756331 | 0.678535 | 0.793314 |
| "feature" | "brand-only tangent" | 0.6859 | 0.66266 | 0.710148 | 0.613656 | 0.751455 |
| "feature" | "cannibalization (category - brand)" | 0.046377 | 0.040102 | 0.052069 | 0.029975 | 0.063318 |
| "feature + display" | "brand-only secant" | 0.677435 | 0.658818 | 0.697398 | 0.6253 | 0.729504 |
| "feature + display" | "category secant" | 0.710882 | 0.694166 | 0.732156 | 0.657808 | 0.762894 |
| "feature + display" | "brand-only tangent" | 0.676653 | 0.649661 | 0.695725 | 0.6141 | 0.739532 |
| "feature + display" | "cannibalization (category - brand)" | 0.033392 | 0.029088 | 0.037671 | 0.021523 | 0.045531 |


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
| 0.710882 | 1.032565 | 0.676 | 0.00375 | 0.74675 | 0.07225 | 0.7575 | 0.22725 | 0.8485 | 0.5 |
| 0.75 | 0.892857 | 0.956 | 0.0865 | 0.96175 | 0.4245 | 0.96025 | 0.708 | 0.9855 | 0.919 |
| 0.9 | 0.357143 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |


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
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-62-output-1.png" class="figure-img" width="1511" height="1339" /></p>
</figure>


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
| "TPR-only" | 0.5 | 1.0 | 0.0 | 0.168363 | 0.168363 | 0.0 |
| "TPR-only" | 0.710882 | 0.98775 | 0.01225 | 0.032303 | 0.032384 | 0.000082 |
| "display" | 0.5 | 1.0 | 0.0 | 0.216845 | 0.216845 | 0.0 |
| "display" | 0.710882 | 0.834 | 0.166 | 0.021611 | 0.102136 | 0.002087 |
| "feature" | 0.5 | 1.0 | 0.0 | 0.307424 | 0.307424 | 0.0 |
| "feature" | 0.710882 | 0.64275 | 0.35725 | 0.011714 | 0.307613 | 0.008284 |
| "feature + display" | 0.5 | 1.0 | 0.0 | 0.399228 | 0.498545 | 0.0 |
| "feature + display" | 0.710882 | 0.3675 | 0.6325 | 0.014817 | 0.523878 | 0.010572 |


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
| "TPR-only" | 0.15 | 0.0 | 0.7855 | 0.766753 | 0.804573 | 0.733981 | 0.843155 |
| "TPR-only" | 0.3 | 0.0 | 0.771285 | 0.754647 | 0.7894 | 0.72233 | 0.821388 |
| "display" | 0.0 | 1.0 | NaN | NaN | NaN | NaN | NaN |
| "display" | 0.15 | 0.0 | 0.320007 | 0.275784 | 0.352053 | 0.219066 | 0.432905 |
| "display" | 0.3 | 0.0 | 0.568122 | 0.548148 | 0.579725 | 0.526168 | 0.615842 |
| "feature" | 0.0 | 1.0 | NaN | NaN | NaN | NaN | NaN |
| "feature" | 0.15 | 1.0 | -0.122924 | -0.143968 | -0.106459 | -0.174297 | -0.068881 |
| "feature" | 0.3 | 0.0 | 0.382358 | 0.374322 | 0.39086 | 0.358368 | 0.406404 |
| "feature + display" | 0.0 | 1.0 | NaN | NaN | NaN | NaN | NaN |
| "feature + display" | 0.15 | 1.0 | -0.318425 | -0.332935 | -0.304871 | -0.359708 | -0.280608 |
| "feature + display" | 0.3 | 0.0 | 0.297224 | 0.292543 | 0.303767 | 0.2819 | 0.313026 |


## The slot cost

A feature or a display is a slot with a cost the data do not contain, so we solve for it. The break-even slot cost of a mechanics at a depth is the incremental gross event profit it adds over a shelf-tag-only cut of the same depth, divided by the number of store-weeks it occupies; at zero depth the comparison is against no promotion at all (a pure mechanics uplift, which is not comparable to a shelf-tag cut at ten points). The forest plot shows these break-even slot costs with their HDIs at the nominal and at the break-even funding share. The table then assumes a cost per slot and store-week, the same for a feature and for a display (the pair costs twice as much), and prints the posterior probability that each mechanics is the best choice over the whole grid as that cost rises.

The slot values and the ladder, from the cells below:

- At a 15\\ cut and the break-even share, a display is worth 29 currency units per store-week over the same cut with a shelf tag alone (94\\ HDI 20 to 39), a feature 87 (74 to 102) and the pair 145 (126 to 167).
- At the nominal share the same values are 25, 75 and 126, because the margin lost on the extra units counts against the slot.
- With slots at 25 per store-week, feature with display is the best choice with probability 1.00 at both shares.
- At 50 it keeps a probability of 0.74 at the nominal share and 0.89 at the break-even share, against a feature alone.
- At 75 the feature alone wins with probability 0.86 and 0.87; at 100 no promotion wins with probability 0.96 and 0.92.


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
| "display" | 0.0 | 0.5 | 28.2748 | 24.367533 | 32.177065 | 17.83479 | 40.102751 |
| "display" | 0.0 | 0.710882 | 28.2748 | 24.367533 | 32.177065 | 17.83479 | 40.102751 |
| "display" | 0.15 | 0.5 | 25.132089 | 21.187563 | 27.276858 | 16.993795 | 34.191711 |
| "display" | 0.15 | 0.710882 | 29.183999 | 24.676837 | 31.629881 | 19.796266 | 39.334415 |
| "display" | 0.3 | 0.5 | 20.091773 | 17.735554 | 22.596463 | 13.714185 | 27.389253 |
| "display" | 0.3 | 0.710882 | 30.523789 | 27.812051 | 34.736939 | 21.073995 | 40.435422 |
| "feature" | 0.0 | 0.5 | 85.030022 | 78.347939 | 89.288312 | 70.367315 | 101.062421 |
| "feature" | 0.0 | 0.710882 | 85.030022 | 78.347939 | 89.288312 | 70.367315 | 101.062421 |
| "feature" | 0.15 | 0.5 | 75.414691 | 70.808471 | 79.123069 | 64.392674 | 88.398613 |
| "feature" | 0.15 | 0.710882 | 86.618457 | 81.552727 | 91.111667 | 74.062825 | 101.682816 |
| "feature" | 0.3 | 0.5 | 60.621426 | 56.60796 | 63.287476 | 51.460489 | 70.921084 |
| "feature" | 0.3 | 0.710882 | 88.846512 | 83.472607 | 93.054187 | 75.196861 | 103.713568 |
| "feature + display" | 0.0 | 0.5 | 141.378065 | 132.964419 | 149.12988 | 119.695285 | 165.338717 |
| "feature + display" | 0.0 | 0.710882 | 141.378065 | 132.964419 | 149.12988 | 119.695285 | 165.338717 |
| "feature + display" | 0.15 | 0.5 | 126.033009 | 117.991695 | 130.364298 | 109.014724 | 144.459345 |
| "feature + display" | 0.15 | 0.710882 | 145.278746 | 136.043769 | 150.225112 | 125.802931 | 166.59897 |
| "feature + display" | 0.3 | 0.5 | 101.512895 | 95.373338 | 104.894454 | 88.432417 | 116.227676 |
| "feature + display" | 0.3 | 0.710882 | 150.382917 | 141.814336 | 155.716232 | 131.34207 | 171.900978 |


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
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-66-output-1.png" class="figure-img" width="1315" height="911" /></p>
</figure>


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
| 0.5 | 25.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| 0.5 | 50.0 | 0.0 | 0.0 | 0.0 | 0.265 | 0.735 |
| 0.5 | 75.0 | 0.097 | 0.0 | 0.0 | 0.85725 | 0.04575 |
| 0.5 | 100.0 | 0.9555 | 0.0 | 0.0 | 0.0445 | 0.0 |
| 0.5 | 150.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.710882 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| 0.710882 | 25.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| 0.710882 | 50.0 | 0.0 | 0.0 | 0.0 | 0.1055 | 0.8945 |
| 0.710882 | 75.0 | 0.0455 | 0.00025 | 0.0 | 0.8715 | 0.08275 |
| 0.710882 | 100.0 | 0.9185 | 0.011 | 0.0 | 0.07025 | 0.00025 |
| 0.710882 | 150.0 | 0.98775 | 0.01225 | 0.0 | 0.0 | 0.0 |


## Store by store

The break-even shares are computable per store because the model has no cross-store terms, so the per-store decisions are separable. The figure shows the category break-even share of the feature-with-display mechanics per store, brand-only and with cannibalization, ordered by the number of identifying weeks of the store; the table prints, per store, the probability that a cut pays at the nominal and at the break-even funding share. Where the intervals are wide, the store's decision is genuinely uncertain, and it is the partial pooling of the store-level elasticities that keeps the intervals from being prior-dominated.

The store table and the figure show the spread:

- At the nominal share the probability that the cut pays is below 0.35 in every store and below 0.02 in fifteen of them.
- At the break-even share it ranges from 0.00 to 1.00: the upscale stores 2513, 11993 and 2277 sit at 0.03 or less, the mainstream stores 19265 and 25027 above 0.99.
- The store-level break-even shares (brand-only medians from 0.53 to 0.87, category medians from 0.57 to 0.89) line up by segment more than by the number of identifying weeks, and their 94\\ HDIs are about twice as wide as the zone-level one.


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
| 9825 | "MAINSTREAM" | 22 | 0.0025 | 0.9575 | 0.666967 | 0.632392 |
| 19265 | "MAINSTREAM" | 20 | 0.34775 | 1.0 | 0.573566 | 0.528595 |
| 2281 | "UPSCALE" | 19 | 0.0 | 0.61425 | 0.751862 | 0.697736 |
| 25027 | "MAINSTREAM" | 17 | 0.1705 | 0.99975 | 0.586479 | 0.553858 |
| 25021 | "VALUE" | 15 | 0.10075 | 0.99125 | 0.648283 | 0.577343 |
| 2513 | "UPSCALE" | 14 | 0.0 | 0.0175 | 0.839245 | 0.814681 |
| 23349 | "VALUE" | 13 | 0.0045 | 0.92275 | 0.675589 | 0.640883 |
| 21479 | "VALUE" | 12 | 0.00025 | 0.7605 | 0.693553 | 0.6784 |
| 4259 | "VALUE" | 11 | 0.0105 | 0.86725 | 0.708092 | 0.644411 |
| 21237 | "MAINSTREAM" | 10 | 0.004 | 0.954 | 0.676426 | 0.632472 |
| 6431 | "VALUE" | 10 | 0.0 | 0.27875 | 0.760733 | 0.738274 |
| 11993 | "UPSCALE" | 9 | 0.0 | 0.0015 | 0.88791 | 0.866336 |
| 613 | "MAINSTREAM" | 7 | 0.0 | 0.5665 | 0.727709 | 0.703582 |
| 2277 | "UPSCALE" | 7 | 0.0 | 0.0265 | 0.831554 | 0.798355 |
| 24991 | "UPSCALE" | 7 | 0.00075 | 0.68225 | 0.720257 | 0.68798 |
| 19523 | "VALUE" | 7 | 0.0165 | 0.89925 | 0.658284 | 0.633994 |
| 25229 | "MAINSTREAM" | 6 | 0.0 | 0.65275 | 0.715752 | 0.692349 |
| 6179 | "UPSCALE" | 6 | 0.00025 | 0.74125 | 0.720045 | 0.681805 |


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
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-69-output-1.png" class="figure-img" width="1511" height="711" /></p>
</figure>


# Risk: the go/no-go

Expected profit is not the whole decision. The incremental profit of a policy against no promotion, \Delta\Pi_r(a) = \Pi_r(a) - \Pi_r(a_0), is computed here on the sampled event paths, so it carries the demand noise as well as the parameter and level-path uncertainty, draw by draw under common random numbers (the same posterior draw, the same level path, coupled demand draws). That coupling is an assumption the data cannot identify: the potential outcomes of the same week under two policies are never observed together. We report the conditional value at risk at the 10\\ level, \text{CVaR}\_{0.10}, the mean of the worst tenth of the draws ([Rockafellar and Uryasev, 2000](https://doi.org/10.21314/JOR.2000.038)), with two standard errors: an iid bootstrap over the draws and the between-chain error of the per-chain values, which respects the autocorrelation of the NUTS draws. A different-key baseline, in which the level path and the demand noise are redrawn while the parameters are shared, is printed as a sensitivity to the coupling, not as a bound. The go/no-go rule is \text{CVaR}\_{0.10}(\Delta\Pi) \ge 0: in the worst tenth of the worlds the promotion still does not lose money on average, before any slot cost. Where the expected profit is flat across depths, this is the tie-breaker. A remark: the CVaR of the level \Pi instead of the increment answers a different question, the total event risk.

The risk table, the sensitivity and the go/no-go ladder say the following:

- At a zero slot cost every policy with a feature or a display has a positive \text{CVaR}\_{0.10} and a probability of loss of 0.00, so the go/no-go is a go at the break-even share for all of them; the shelf-tag cuts are the only policies with a negative downside.
- The three deepest feature-with-display cells have \text{CVaR}\_{0.10} values of 4{,}392, 4{,}403 and 4{,}400, with bootstrap standard errors of 12 to 13 and between-chain standard errors of 6 to 8, so the risk measure does not separate them either; the deepest cell has the highest expected increment, 5{,}263, and the earlier table gave it a 63\\ chance of being the best cell.
- The coupling matters for the downside number: the committed policy has a \text{CVaR}\_{0.10} of 4{,}339 under common random numbers and 3{,}952 with a redrawn baseline, a difference the data cannot arbitrate.
- The slot-cost ladder turns the histogram into a decision: with slots at 25 per store-week the committed event still has a probability of loss of 0.00 at both shares; at 50 the probability is 0.11 at the nominal share and 0.00 at the break-even share; at 75 it is 1.00 and 0.73.


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


    feature + display at 40%: CVaR10 4,400 | bootstrap se 13 | between-chain se 8
    feature + display at 35%: CVaR10 4,403 | bootstrap se 12 | between-chain se 6
    feature + display at 30%: CVaR10 4,392 | bootstrap se 12 | between-chain se 6


| mechanics           | depth | expected_increment | P(loss) | cvar_10     |
|---------------------|-------|--------------------|---------|-------------|
| "feature + display" | 0.4   | 5263.055091        | 0.0     | 4400.175597 |
| "feature + display" | 0.35  | 5217.502863        | 0.0     | 4402.791111 |
| "feature + display" | 0.3   | 5182.392866        | 0.0     | 4391.721151 |
| "feature + display" | 0.25  | 5155.341689        | 0.0     | 4378.410548 |
| "feature + display" | 0.2   | 5135.639616        | 0.0     | 4359.728478 |
| "feature + display" | 0.15  | 5122.948226        | 0.0     | 4338.612252 |
| "feature + display" | 0.1   | 5115.035875        | 0.0     | 4315.730318 |
| "feature + display" | 0.0   | 5113.553921        | 0.0     | 4273.552736 |
| "feature + display" | 0.05  | 5112.227595        | 0.0     | 4295.985766 |
| "feature"           | 0.0   | 3068.963931        | 0.0     | 2522.549936 |


``` python
committed = ("feature + display", 0.15)
rng_key, key_alt = random.split(rng_key)
out_alt = scenario_draws(
    key_alt, nuts_model, posterior, y_train, policy_covariates(0.0, *MECHANICS["TPR-only"])
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


    feature + display at 15%, alpha 0.71: CVaR10 with common random numbers 4,339 | with a redrawn baseline 3,952 | P(loss) 0.00 vs 0.00


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
| 0.0           | 0.5      | 4073.783472        | 0.0     | 3426.271506  |
| 0.0           | 0.710882 | 5122.948226        | 0.0     | 4338.612252  |
| 25.0          | 0.5      | 2273.783472        | 0.0     | 1626.271506  |
| 25.0          | 0.710882 | 3322.948226        | 0.0     | 2538.612252  |
| 50.0          | 0.5      | 473.783472         | 0.107   | -173.728494  |
| 50.0          | 0.710882 | 1522.948226        | 0.0     | 738.612252   |
| 75.0          | 0.5      | -1326.216528       | 0.9975  | -1973.728494 |
| 75.0          | 0.710882 | -277.051774        | 0.7315  | -1061.387748 |
| 100.0         | 0.5      | -3126.216528       | 1.0     | -3773.728494 |
| 100.0         | 0.710882 | -2077.051774       | 0.9995  | -2861.387748 |
| 150.0         | 0.5      | -6726.216528       | 1.0     | -7373.728494 |
| 150.0         | 0.710882 | -5677.051774       | 1.0     | -6461.387748 |


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


    best policy with CVaR10 >= 0: feature + display at 40% (expected increment 5,263)
    highest expected increment: feature + display at 40% (5,263, CVaR10 4,400)


<figure class="figure">
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-73-output-2.png" class="figure-img" width="1511" height="611" /></p>
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

The planner table and the scatter show where shrinkage matters:

- The Jensen ratio is at most 1.002 across the grid, so the posterior-mean plug-in and the posterior planner agree.
- For the decision to run the event, both planners choose all 18 stores at both shares; the least-squares planner's disappointment is negative (-718 and -696), because its own mechanics multipliers under-predict the uplift the posterior expects.
- For the decision to add the cut at the nominal share, the posterior planner adds it in no store and the least-squares planner in 3; that plan is worth -110 under the posterior, a disappointment of 222.
- At the break-even share the least-squares planner adds the cut in all 18 stores and the posterior planner in 9; the least-squares plan is worth 11 under the posterior against 162 for the posterior plan, and the disappointment is 865: the stores with the most extreme least-squares elasticities are the ones whose predicted gains evaporate under the posterior, the optimizer's curse in a table.
- The scatter shows it: at the break-even share nearly every store sits on or below the identity line, and the store with the largest predicted gain keeps a small part of it.


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
| 0.05  | 1.061188       | 1.061167       | 1.000019 |
| 0.1   | 1.129789       | 1.129698       | 1.000081 |
| 0.15  | 1.207196       | 1.206964       | 1.000192 |
| 0.2   | 1.295168       | 1.2947         | 1.000362 |
| 0.25  | 1.395957       | 1.395118       | 1.000601 |
| 0.3   | 1.512493       | 1.511096       | 1.000924 |
| 0.35  | 1.648655       | 1.646435       | 1.001348 |
| 0.4   | 1.809685       | 1.806259       | 1.001896 |


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
| "run the event vs no promotion" | 0.5 | 18 | 18 | 0 | 4079.610051 | 4079.610051 | -718.140271 |
| "run the event vs no promotion" | 0.710882 | 18 | 18 | 0 | 5130.077241 | 5130.077241 | -696.438115 |
| "add the cut vs the mechanics at base price" | 0.5 | 3 | 0 | 3 | -109.679484 | 0.0 | 221.736542 |
| "add the cut vs the mechanics at base price" | 0.710882 | 18 | 9 | 9 | 10.697087 | 162.14862 | 864.696882 |


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
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-76-output-1.png" class="figure-img" width="1511" height="661" /></p>
</figure>


# The promotion order

The funding and go/no-go questions belong to the category manager. The order belongs to the replenishment planner, who takes the promotion as committed: feature with display at a 15\\ cut, the grid cell nearest the realized Thanksgiving event, under the nominal funding share of 0.5.


## Event demand, costs and the critical fractile

Every quantity here is per store; the printed totals sum the 18 stores. Let y\_{r,t,\text{H}} be the sampled units of the focal product in one store in event week t under the committed policy, in draw r. One order covers both event weeks with no mid-event replenishment, so the demand the order faces is the two-week sum of the same sampled path, which keeps the correlation between the weeks:

 W_r = \sum\_{t \in E} y\_{r,t,\text{H}}. 

A unit short loses the margin it would have earned. The underage cost is the promo-week unit margin at the committed depth and the nominal share, because the allowance is earned only on units sold:

 c_u = m\_{\text{H},s}(0.15) = p\_{\text{H},s}\\ \big(g\_{\text{H}} - 0.5 \times 0.15\big). 

The true underage cost is a little lower, since part of the stockout demand is recovered at the private-label margin, so the fractile below is an upper bound and the order leans high. A unit left over is carried and sold later at the regular margin, because cereal is shelf-stable, so only the holding cost is lost. The overage cost is a holding share \eta of the unit cost, and a realistic \eta is small:

 c_o = \eta\\ c\_{\text{H},s}. 

The critical fractile is the service level that balances the two costs. Both costs scale with the base price, so it is the same in every store:

 \kappa = \frac{c_u}{c_u + c_o}. 

The newsvendor profit of an order Q against a demand W earns c_u on every unit sold and pays c_o on every unit left over, with (x)^+ = \max(x, 0):

 \text{prof}(Q; W) = c_u\\ \min(W, Q) - c_o\\ (Q - W)^+. 


## Three order rules

- The paths rule takes the \kappa-quantile of the joint event demand, the newsvendor optimum: Q^\star = \min\\Q : \text{P}(W \le Q) \ge \kappa\\.
- The marginal rule adds up per-week safety stocks, with q\_\kappa the per-week \kappa-quantile, and ignores the correlation between the weeks: Q\_{\text{marg}} = \sum\_{t \in E} q\_\kappa(y\_{t,\text{H}}).
- The mean rule orders the expected demand, which is what a point forecast delivers: Q\_{\text{mean}} = \text{E}\[W\].


## What the joint predictive is worth

The vocabulary follows [Birge and Louveaux (2011)](https://doi.org/10.1007/978-1-4614-0237-4), whose news vendor example of chapter 1 fixes it. Here \theta stands for the parameters and the level path of one draw, and every expectation is over the posterior predictive of W.

\begin{align\*} \text{RP} &= \max_Q \text{E}\[\text{prof}(Q; W)\] && \text{recourse value, attained by } Q^\star \\ \text{EEV} &= \text{E}\[\text{prof}(Q\_{\text{mean}}; W)\] && \text{expected result of the mean order} \\ \text{VSS} &= \text{RP} - \text{EEV} \ge 0 && \text{value of the stochastic solution} \\ \text{WS} &= \text{E}\[c_u\\ W\] && \text{wait-and-see: order after seeing the demand} \\ \text{EVPI}\_W &= \text{WS} - \text{RP} && \text{value of perfect information about the demand} \\ \text{EVPI}\_\theta &= \text{E}\_\theta\big\[\max_Q \text{E}\[\text{prof}(Q; W) \mid \theta\]\big\] - \text{RP} && \text{value of perfect information about } \theta \end{align\*}

\text{EVPI}\_W is a loose ceiling: it includes the irreducible demand noise, which no model removes. \text{EVPI}\_\theta is the tighter one, the value of knowing the parameters and the level path, computed with inner negative binomial draws per posterior draw. Two caveats:

- The inner maximum is optimistic by about one part in the number of inner draws, so RP is recomputed on the same inner draws and the difference is not Monte Carlo noise.
- RP, VSS and both ceilings are maxima on the evaluation draws, so we also print a split-half VSS: the rule is chosen on one half of the draws and evaluated on the other.

The sweep over \eta then shows the value of the stochastic solution as a function of the cost asymmetry c_u / c_o. The results, from the three cells below:

- Costs and dependence: with a holding share of 0.1 the critical fractile is 0.74, the underage cost is 0.54 to 0.63 per unit across stores, and the two event weeks of a store have a correlation of 0.27 across draws in the median store.
- Orders: the recourse value is 5{,}936; the mean order loses 112 against it, a value of the stochastic solution of 1.9\\ of RP (116 on the split-half check); the marginal rule loses only 5.
- Why the marginal rule loses little: its per-week quantiles add up to an order above the joint quantile in 17 of the 18 stores, by 5 to 41 units (the remaining store orders 2 units less), and over-ordering is cheap at a fractile of 0.74.
- Service of the paths rule: a fill rate of 0.94 and an expected leftover of 2{,}206 units over the 18 stores.
- Information ceilings: perfect information about the demand itself would be worth 14.5\\ of RP; about the parameters and the level path 5.5\\, from 300 inner draws.
- The sweep: the value of the stochastic solution is smallest, 0.1\\ of RP, at a holding share of 0.2, where the critical fractile of 0.59 sits nearest the probability that demand falls below its mean, 0.54 in the median store; it grows to 7.0\\ at a fractile of 0.93 and to 15.0\\ at 0.26, and the marginal-rule loss stays below 0.8\\ of RP everywhere.


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
    eta 0.1: kappa 0.740 | RP 5,936 | EEV 5,824 | VSS 112 (1.9% of RP, split-half 116) | marginal-rule loss 5 | EVPI_W 863 (14.5%)
    fill rate 0.943 | expected leftover 2,206 units | stores with Q_marginal >= Q_paths: 17 of 18


| store | expected_demand | Q_paths | Q_marginal | Q_mean | marginal_minus_paths |
|-------|-----------------|---------|------------|--------|----------------------|
| 25027 | 1310.271973     | 1521.0  | 1544.0     | 1310.0 | 23.0                 |
| 21237 | 698.522522      | 790.0   | 810.0      | 699.0  | 20.0                 |
| 25229 | 595.529724      | 667.0   | 690.0      | 596.0  | 23.0                 |
| 19265 | 599.775024      | 679.0   | 697.0      | 600.0  | 18.0                 |
| 9825  | 992.892517      | 1133.0  | 1159.0     | 993.0  | 26.0                 |
| 613   | 419.285004      | 473.0   | 488.0      | 419.0  | 15.0                 |
| 2277  | 1174.550537     | 1316.0  | 1357.0     | 1175.0 | 41.0                 |
| 24991 | 1123.129761     | 1263.0  | 1295.0     | 1123.0 | 32.0                 |
| 6179  | 555.458008      | 626.0   | 642.0      | 555.0  | 16.0                 |
| 2513  | 478.426239      | 539.0   | 558.0      | 478.0  | 19.0                 |
| 2281  | 736.149231      | 830.0   | 854.0      | 736.0  | 24.0                 |
| 11993 | 477.855988      | 539.0   | 555.0      | 478.0  | 16.0                 |
| 25021 | 294.112762      | 352.0   | 357.0      | 294.0  | 5.0                  |
| 4259  | 388.494263      | 477.0   | 475.0      | 388.0  | -2.0                 |
| 21479 | 374.196503      | 423.0   | 435.0      | 374.0  | 12.0                 |
| 23349 | 238.769257      | 278.0   | 283.0      | 239.0  | 5.0                  |
| 19523 | 430.647003      | 508.0   | 515.0      | 431.0  | 7.0                  |
| 6431  | 293.052002      | 331.0   | 342.0      | 293.0  | 11.0                 |


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


    value of perfect information about parameters and level path at eta 0.1: 5.5% of RP (300 inner draws)


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


    P(W <= E[W]) median across stores: 0.544 (the critical fractile at which the mean order is optimal)


| eta | kappa | c_u / c_o | VSS / RP | VSS split-half / RP | marginal-rule loss / RP | EVPI_W / RP | EVPI_theta / RP |
|----|----|----|----|----|----|----|----|
| 0.02 | 0.934366 | 14.236111 | 0.070443 | 0.071692 | 0.001718 | 0.048695 | 0.021857 |
| 0.05 | 0.850622 | 5.694444 | 0.043824 | 0.044779 | 0.001691 | 0.09251 | 0.037948 |
| 0.1 | 0.740072 | 2.847222 | 0.01887 | 0.019497 | 0.00083 | 0.145417 | 0.055345 |
| 0.2 | 0.587393 | 1.423611 | 0.00114 | 0.001289 | 0.000024 | 0.219178 | 0.077295 |
| 0.4 | 0.415822 | 0.711806 | 0.021346 | 0.020233 | 0.001265 | 0.314106 | 0.102992 |
| 0.8 | 0.262484 | 0.355903 | 0.150088 | 0.146713 | 0.007486 | 0.426959 | 0.131114 |


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
<p><img src="promotion_pricing_decisions_files/figure-html/_src-promotion_pricing_decisions-cell-80-output-1.png" class="figure-img" width="1111" height="611" /></p>
</figure>


# What the posterior buys you

The four promises of the introduction, each backed by a table above.

- The right expectation. The store-level elasticities are partially pooled (a posterior-median spread of 0.31 across stores against 0.37 for within-store least squares). The store least-squares planner, which is not pooled, adds the cut in 3 stores at the nominal share where the posterior adds it in none, and in all 18 at the break-even share where the posterior adds it in 9; its predicted gains exceed the posterior-evaluated ones by 222 and 865, the postdecision disappointment of the planner table.
- A reservation share with an interval instead of an argmax. At the nominal share the posterior puts probability 1.00 on a falling profit curve for every mechanics, so the depth decision is a corner. What the posterior adds is the break-even funding share, 0.71 for feature with display including cannibalization, with a 94\\ HDI about 0.1 wide at zone level and about twice that per store, and the event-level share of every cell, negative for a 15\\ cut under a feature.
- A downside, and a tie-breaker where the objective is flat. Near the break-even share the expected profit moves by 1.5\\ of the baseline across the whole depth grid, every featured policy has a positive \text{CVaR}\_{0.10}, and the three deepest cells have downside values within their standard errors of each other; the risk table with slot costs is where the go turns into a no-go, between 50 and 75 per slot and store-week.
- An order from joint paths rather than from summed quantiles. The value of the stochastic solution is 1.9\\ of the recourse value at the nominal holding share, smallest (0.1\\) where the critical fractile meets the probability that demand falls below its mean, and up to 15\\ at low fractiles; the marginal-quantile rule over-orders in 17 of the 18 stores but loses below 1\\ everywhere, because the two event weeks are only weakly correlated (0.27) and over-ordering is cheap at high fractiles.


# Limitations

- No experiment. Every elasticity rests on the selection-on-observables assumption drawn in the causal graph; the holdout validates the forecasting engine under the realized calendar, not the counterfactual ones.
- The joint no-promotion baseline over a holiday quarter is never observed; its units rest on the log-additivity of the model.
- Common random numbers are a coupling assumption; the different-key baseline is a sensitivity, not a bound.
- The centering values are chosen by a mean-field guide, a heuristic for the sampler's geometry, not part of the posterior; another guide could pick other values without changing any posterior quantity.
- Shelf capacity censors the sales in the strongest promotion weeks, which biases the feature-with-display uplift and the order quantities downward; the [censored demand example](censored_demand.md) shows the likelihood that would address it.
- The elasticity is promotional, not regular-price; base-price changes are absorbed by the level.
- The economics are assumptions: gross margins, a per-unit allowance, base prices frozen at the last training week (the brand-only break-even share is price-free; the category version depends on price ratios only), a holding-cost overage.
- The holdout is the holiday quarter with two earlier Decembers to learn from, and the holdout forecasts run high on average: the PIT histogram slopes downward, with 16\\ of the cells in the lowest decile.
- Post, Quaker, the products of other sub-categories and other retailers are omitted competitors; the manufacturer's side of the deal is outside the model (General Mills also owns two of the siblings, so part of the cannibalization is internal to it); the sibling-mechanics effects are averages over any featured sibling; part of the stockout demand of the focal product spills to the private-label twin.


# Next steps

- Make the calendar a lever: add a post-promotion term and let the seasonal profile choose the event weeks.
- Replace the average sibling-mechanics effects by per-pair terms, and give the mechanics effects a store level.
- Add the censored likelihood of the [censored demand example](censored_demand.md) for the weeks at shelf capacity.
- Run a rolling backtest with [backtest](../../reference/evaluate.backtest.md#numpyro_forecast.evaluate.backtest) over several promotion quarters.
- Use `VISITS` and `HHS` to separate traffic from basket effects.
- Pool the orders at the distribution center and compare with the per-store orders.
- Promote the decision helpers (profit contraction, CVaR, newsvendor rules, VSS) into a package module, and let [forecast](../../reference/predictive.forecast.md#numpyro_forecast.predictive.forecast) return extra sites such as the conditional mean.


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
- Related examples: [hierarchical forecasting 1](hierarchical_forecasting_1.md) (the sampled centering idiom), [fresh retail stockout](fresh_retail_stockout.md) (the model factory and the covariate-swap counterfactual), [availability TSB](availability_tsb.md) (scenario covariates), [censored demand](censored_demand.md) (the NUTS template and the censored likelihood).

[Source: From forecasts to promotion decisions](_src/promotion_pricing_decisions-preview.html#75c3769a)
