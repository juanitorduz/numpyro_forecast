# ---
# jupyter:
#   description: A hierarchical negative binomial state space model for all 30,490 M5 series, trained with minibatch SVI, built up component by component against the Pyro starter kit baselines and scored at the twelve hierarchy levels with a weighted scaled CRPS.
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: description
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # M5 forecasting II: a hierarchical count state space model with `numpyro_forecast`
#
# INTRO_PLACEHOLDER

# %% [markdown]
# ## Prepare notebook

# %%
import warnings
from functools import partial
from time import perf_counter
from typing import NamedTuple

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import polars as pl
import scipy.sparse as sp
import xarray as xr
from jax import random
from numpyro.infer import SVI, Predictive, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.svi import SVIRunResult
from numpyro.optim import optax_to_numpyro

from numpyro_forecast import (
    Horizon,
    backtest,
    draw_posterior,
    evaluate_forecast,
    forecast,
    predict,
    predict_in_sample,
    predictions_to_datatree,
    results_to_dataframe,
)
from numpyro_forecast.datasets import load_m5
from numpyro_forecast.features import fourier_features
from numpyro_forecast.metrics import crps_empirical
from numpyro_forecast.typing import Array, ForecastModel

az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"
numpyro.set_host_device_count(n=4)
rng_key = random.PRNGKey(seed=42)

pl.Config.set_fmt_str_lengths(100)
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True)
pl.Config.set_tbl_rows(20)

# `predict` observes a (time, series) array under the series plate; the time axis is not a
# plate, so numpyro cannot check it and warns at every trace.
warnings.filterwarnings("ignore", message="Missing a plate statement for batch dimension -2")
# The second `plot_lm` call of an overlay reuses the alpha of the first one.
warnings.filterwarnings("ignore", message="When multiple credible intervals are plotted")

# %load_ext autoreload
# %autoreload 2
# %load_ext jaxtyping
# %jaxtyping.typechecker beartype.beartype
# %config InlineBackend.figure_format = "retina"

# %% [markdown]
# ## Read data
#
# DATA_PLACEHOLDER

# %%
m5 = load_m5()

KEYS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
N_DAYS_TRAIN = 1_941
HORIZON = 28
N_DAYS = N_DAYS_TRAIN + HORIZON
sales = m5.sales
price = m5.price
keys_df = m5.keys
n_series = sales.shape[1]


def is_christmas() -> pl.Expr:
    """Flag December 25, the one day a year every Walmart store is closed."""
    return pl.col("date").dt.month().eq(pl.lit(12)).and_(pl.col("date").dt.day().eq(pl.lit(25)))


def day_of_week_index() -> pl.Expr:
    """Index the weekday from 0 (Monday) to 6 (Sunday)."""
    return pl.col("date").dt.weekday().sub(pl.lit(1))


def event_index(column: str, names: list[str]) -> pl.Expr:
    """Index an event column by its position in ``names``, 0 when there is no event."""
    return pl.col(column).replace_strict(names, list(range(1, len(names) + 1)), default=0)


event_names = sorted(
    set(m5.calendar["event_name_1"].drop_nulls().to_list())
    | set(m5.calendar["event_name_2"].drop_nulls().to_list())
)
calendar_df = (
    m5.calendar.lazy()
    .with_row_index("t")
    .with_columns(
        christmas=is_christmas().cast(pl.Float32),
        dow=day_of_week_index(),
        event_1=event_index("event_name_1", event_names),
        event_2=event_index("event_name_2", event_names),
    )
    .select("t", "date", "dow", "christmas", "event_1", "event_2", "snap_CA", "snap_TX", "snap_WI")
    .collect(engine="streaming")
)
print(f"{n_series} series over {N_DAYS} days, {len(event_names)} event names")
calendar_df.filter(pl.col("event_2").gt(0))

# %% [markdown]
# HIERARCHY_PLACEHOLDER

# %%
LEVELS: dict[str, list[str]] = {
    "Level1": [],
    "Level2": ["state_id"],
    "Level3": ["store_id"],
    "Level4": ["cat_id"],
    "Level5": ["dept_id"],
    "Level6": ["state_id", "cat_id"],
    "Level7": ["state_id", "dept_id"],
    "Level8": ["store_id", "cat_id"],
    "Level9": ["store_id", "dept_id"],
    "Level10": ["item_id"],
    "Level11": ["state_id", "item_id"],
    "Level12": ["item_id", "store_id"],
}


def level_label(columns: list[str]) -> pl.Expr:
    """Label of the aggregate a series belongs to at one level, ``Agg_Level_1/Agg_Level_2``."""
    parts = [pl.col(column) for column in columns] or [pl.lit("Total")]
    if len(parts) == 1:
        parts.append(pl.lit("X"))
    return pl.concat_str(parts, separator="/")


def group_id(label: str) -> pl.Expr:
    """Dense 0-based id of the aggregate named by ``label``, in sorted label order."""
    return pl.col(label).rank("dense").cast(pl.Int64).sub(pl.lit(1))


def add_level_ids(keys: pl.DataFrame) -> pl.DataFrame:
    """Add one label and one group id column per M5 level."""
    return keys.with_columns(
        **{f"{level}_label": level_label(columns) for level, columns in LEVELS.items()}
    ).with_columns(**{level: group_id(f"{level}_label") for level in LEVELS})


def aggregation_matrix(
    hierarchy: pl.DataFrame,
) -> tuple[sp.csr_matrix, list[str], dict[str, slice]]:
    """Sparse ``(n_series, n_aggregates)`` sum matrix over the 12 levels, with labels and slices."""
    rows, cols, labels, slices = [], [], [], {}
    offset = 0
    for level in LEVELS:
        rows.append(np.arange(hierarchy.height))
        cols.append(offset + hierarchy[level].to_numpy())
        level_labels = (
            hierarchy.select(f"{level}_label", level)
            .unique()
            .sort(level)[f"{level}_label"]
            .to_list()
        )
        labels += [f"{level}/{label}" for label in level_labels]
        slices[level] = slice(offset, offset + len(level_labels))
        offset += len(level_labels)
    values = np.ones(hierarchy.height * len(LEVELS), dtype=np.float32)
    matrix = sp.csr_matrix(
        (values, (np.concatenate(rows), np.concatenate(cols))), shape=(hierarchy.height, offset)
    )
    return matrix, labels, slices


hierarchy_df = keys_df.pipe(add_level_ids)
agg_matrix, agg_labels, level_slices = aggregation_matrix(hierarchy_df)
sales_agg = sales @ agg_matrix
store_ids = [
    label.removeprefix("Level3/").removesuffix("/X")
    for label in agg_labels[level_slices["Level3"]]
]
dept_ids = [
    label.removeprefix("Level5/").removesuffix("/X")
    for label in agg_labels[level_slices["Level5"]]
]
group_ids = [label.removeprefix("Level9/") for label in agg_labels[level_slices["Level9"]]]
state_index = hierarchy_df["Level2"].to_numpy()
store_index = jnp.asarray(hierarchy_df["Level3"].to_numpy(), dtype=jnp.int32)
dept_index = jnp.asarray(hierarchy_df["Level5"].to_numpy(), dtype=jnp.int32)
group_index = jnp.asarray(hierarchy_df["Level9"].to_numpy(), dtype=jnp.int32)
N_STORES, N_DEPTS, N_GROUPS = len(store_ids), len(dept_ids), len(group_ids)
print(
    f"aggregation matrix {agg_matrix.shape}: {N_STORES} stores, {N_DEPTS} departments, {N_GROUPS} store-departments"
)

# %% [markdown]
# SCORING_PLACEHOLDER

# %%
christmas = calendar_df["christmas"].to_numpy()[:N_DAYS]
saled = (~np.isnan(price)).astype(np.float32) * (1.0 - christmas[:, None])
price_filled = np.nan_to_num(price, nan=0.0)


def m5_weights(t1: int) -> np.ndarray:
    """Dollar-sales share of every aggregate over the 28 days before ``t1``, normalized per level."""
    dollars = (sales[t1 - 28 : t1] * price_filled[t1 - 28 : t1]).sum(0) @ agg_matrix
    weights = np.empty_like(dollars)
    for level_slice in level_slices.values():
        weights[level_slice] = dollars[level_slice] / dollars[level_slice].sum()
    return weights


def m5_scales(y: np.ndarray) -> np.ndarray:
    """Mean absolute lag-1 difference of every column of ``y`` over its active period.

    The active period starts at the first nonzero value; the jump from zero on that day is
    excluded, the sum of absolute differences is clamped at one and divided by the number of
    active days minus one (the kit's ``get_metric_scale("pl", ...)``).
    """
    duration, n_columns = y.shape
    active = np.maximum((np.cumsum(y, axis=0) != 0).sum(0), 2)
    start_value = y[duration - active, np.arange(n_columns)]
    lag1_norm = np.abs(np.diff(y, axis=0, prepend=0.0)).sum(0) - np.abs(start_value)
    return np.maximum(lag1_norm, 1.0) / (active - 1)


def aggregate_transform(
    pred: np.ndarray | Array, truth: np.ndarray | Array
) -> tuple[np.ndarray, np.ndarray]:
    """Sum bottom-level draws and truth to the 42,840 aggregates of the 12 M5 levels."""
    n = pred.shape[-1]
    pred_levels = np.asarray(pred, dtype=np.float32).reshape(-1, n) @ agg_matrix
    truth_levels = np.asarray(truth, dtype=np.float32) @ agg_matrix
    return pred_levels.reshape(*pred.shape[:-1], -1), truth_levels


def ws_crps_level(
    pred: Array, truth: Array, *, level_slice: slice, weights: np.ndarray, scales: np.ndarray
) -> Array:
    """Weighted scaled CRPS of one level: weight times mean CRPS over the horizon, over scale."""
    crps = crps_empirical(pred[..., level_slice], truth[..., level_slice]).mean(axis=0)
    return jnp.sum(jnp.asarray(weights) * crps / jnp.asarray(scales))


def m5_metrics(t0: int, t1: int, t2: int) -> dict:
    """Build the 12 weighted scaled CRPS metrics of a window whose training data ends at ``t1``."""
    weights = m5_weights(t1)
    scales = m5_scales(sales_agg[:t1])
    return {
        f"ws_crps_{level}": partial(
            ws_crps_level,
            level_slice=level_slice,
            weights=weights[level_slice],
            scales=scales[level_slice],
        )
        for level, level_slice in level_slices.items()
    }


M5_QUANTILES = np.array(
    [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995], dtype=np.float32
)


def ws_pinball(
    pred_levels: np.ndarray, truth_levels: np.ndarray, weights: np.ndarray, scales: np.ndarray
) -> float:
    """M5 WSPL: pinball loss at the nine competition quantiles, weighted, scaled, mean over levels."""
    quantiles = np.quantile(pred_levels, M5_QUANTILES, axis=0)
    error = quantiles - truth_levels[None]
    u = M5_QUANTILES[:, None, None]
    per_series = (np.where(error <= 0, -u, 1 - u) * error).mean(axis=(0, 1))
    level_scores = [
        np.sum(weights[level_slice] * per_series[level_slice] / scales[level_slice])
        for level_slice in level_slices.values()
    ]
    return float(np.mean(level_scores))


LEVEL_COLUMNS = [f"metric_ws_crps_{level}" for level in LEVELS]
TOP_LEVELS = LEVEL_COLUMNS[:9]
ITEM_LEVELS = LEVEL_COLUMNS[9:]


def summarize_backtest(results: list, name: str) -> pl.DataFrame:
    """One row per window with the mean WS-CRPS over all levels, levels 1 to 9 and levels 10 to 12."""
    return (
        pl.from_pandas(results_to_dataframe(results))
        .with_columns(
            model=pl.lit(name),
            ws_crps=pl.mean_horizontal(LEVEL_COLUMNS),
            ws_crps_1_9=pl.mean_horizontal(TOP_LEVELS),
            ws_crps_10_12=pl.mean_horizontal(ITEM_LEVELS),
        )
        .select(
            "model",
            "t1",
            "t2",
            "walltime",
            "ws_crps",
            "ws_crps_1_9",
            "ws_crps_10_12",
            *LEVEL_COLUMNS,
        )
    )


# %% [markdown]
# FEATURES_PLACEHOLDER

# %%
REFERENCE_DAYS = N_DAYS_TRAIN - HORIZON - 2 * 35  # 1,843: the first backtest origin
BLOCK = 28
N_BLOCKS = 13  # one year of 28-day blocks before the origin

listed_days = saled[:REFERENCE_DAYS].sum(0)
reference_price = np.nansum(price[:REFERENCE_DAYS], axis=0) / np.maximum(listed_days, 1.0)
log_price_ratio = np.nan_to_num(
    np.log(price / np.where(listed_days > 0, reference_price, 1.0)), nan=0.0
).astype(np.float32)
covariates_np = np.empty((3, N_DAYS, n_series), dtype=np.float32)
covariates_np[0] = np.arange(N_DAYS, dtype=np.float32)[:, None]
covariates_np[1] = saled
covariates_np[2] = log_price_ratio
covariates = jnp.asarray(covariates_np)
del covariates_np
y_counts = jnp.asarray(np.rint(sales * saled).astype(np.int32))  # nothing sells on a closed day

dow_table = jnp.asarray(calendar_df["dow"].to_numpy()[:N_DAYS], dtype=jnp.int32)
snap_table = jnp.asarray(
    calendar_df.select("snap_CA", "snap_TX", "snap_WI").to_numpy()[:N_DAYS], dtype=jnp.float32
)
event_tables = (
    jnp.asarray(calendar_df["event_1"].to_numpy()[:N_DAYS], dtype=jnp.int32),
    jnp.asarray(calendar_df["event_2"].to_numpy()[:N_DAYS], dtype=jnp.int32),
)
N_EVENTS = len(event_names) + 1
N_HARMONICS = 4
fourier_table = jnp.asarray(fourier_features(N_DAYS, 365.25, N_HARMONICS), dtype=jnp.float32)
state_of_store = jnp.asarray(
    [state_index[np.argmax(np.asarray(store_index) == s)] for s in range(N_STORES)],
    dtype=jnp.int32,
)
print(
    f"covariates {covariates.shape}, {covariates.nbytes / 1e9:.2f} GB; {int((sales * (1 - saled) > 0).sum())} nonzero sales on unlisted or closed days set to 0"
)

# %% [markdown]
# MODEL_PLACEHOLDER


# %%
class ModelConfig(NamedTuple):
    """Feature flags of the model ladder."""

    dispersion: bool = False
    recency: bool = False
    walks: bool = False
    shocks: bool = False
    seasonal: bool = False
    price: bool = False


SUBSAMPLE_SIZE = 1_500
HISTORY_DAYS = 730  # the item model trains on the last two years before the origin
MAX_LOG_MEAN = float(np.log(2_000.0))  # no series sells more than 763 units on a day
WALK_PRIOR = 0.005  # daily random walk scale on the log scale: smooth level changes only


def block_index(h: Horizon) -> Array:
    """Index of the 28-day block of every day, counted so that the horizon is block ``N_BLOCKS``.

    The last training block ends at the forecast origin; days more than ``N_BLOCKS`` blocks
    before the origin share block 0.
    """
    t = jnp.arange(h.duration)
    return jnp.clip((t - h.t_obs) // BLOCK + N_BLOCKS, 0, N_BLOCKS)


def block_walk(name: str, scale: Array, h: Horizon, size: int | None) -> Array:
    """Random walk over 28-day blocks that is zero on the last training block.

    Earlier blocks are minus the innovations that follow them, so the intercepts carry the
    level at the origin; when forecasting, one more innovation moves the horizon block. The
    site is sampled under a ``block`` plate at ``dim=-2`` and inherits the enclosing plate
    (``size`` series, or the subsampled series plate when ``size`` is ``None``).
    """
    with numpyro.plate(f"{name}_blocks", N_BLOCKS - 1, dim=-2):
        drift = numpyro.sample(name, dist.Normal(0.0, scale))
    walk = jnp.concatenate(
        [-jnp.flip(jnp.cumsum(jnp.flip(drift, 0), 0), 0), jnp.zeros_like(drift[:1])]
    )
    if h.future > 0:
        with numpyro.plate(f"{name}_future_blocks", 1, dim=-2):
            future = numpyro.sample(f"{name}_future", dist.Normal(0.0, scale))
        walk = jnp.concatenate([walk, future], axis=0)
    return walk


def weekly_shocks(name: str, scale: Array, h: Horizon, size: int) -> Array:
    """Daily shocks that sum to zero within every week, one week per row.

    A zero-sum week cannot absorb a level change, so the shocks carry only the common
    day-to-day variation of the series that share them; the horizon gets fresh weeks.
    """
    n_weeks = -(-h.t_obs // 7)
    with numpyro.plate(f"{name}_weeks", n_weeks, dim=-2):
        shock = numpyro.sample(name, dist.ZeroSumNormal(1.0, event_shape=(7,)))
    if h.future > 0:
        with numpyro.plate(f"{name}_future_weeks", (h.future + 6) // 7, dim=-2):
            future = numpyro.sample(f"{name}_future", dist.ZeroSumNormal(1.0, event_shape=(7,)))
        shock = jnp.concatenate([shock, future], axis=0)
    days = jnp.transpose(scale[:, None] * shock, (0, 2, 1)).reshape(-1, size)
    return days[n_weeks * 7 - h.t_obs :][: h.duration]  # weeks end at the origin


def make_model(
    config: ModelConfig,
    level_at_origin: Array,
    *,
    store_index: Array = store_index,
    group_index: Array = group_index,
    dept_index: Array = dept_index,
    forecast_only: bool = False,
) -> ForecastModel:
    """Build the hierarchical negative binomial model with the components in ``config``.

    ``level_at_origin`` is the log of every series' mean sales over the last 28 listed
    training days: the prior center of the item intercepts, which the end-anchored walks
    make the level at the forecast origin. With ``forecast_only=True`` the item-level terms
    are evaluated over the horizon rows only and exposed as the deterministic sites
    ``mean_future`` and ``concentration`` instead of the likelihood, which is how the
    notebook draws the 30,490-series forecast.
    """
    n_series = store_index.shape[0]

    def model(covariates: Array, data: Array | None = None) -> None:
        h = Horizon.from_data(covariates, data)
        t_index = covariates[0, :, 0].astype(jnp.int32)
        rows = slice(h.t_obs, None) if forecast_only else slice(None)
        dow = dow_table[t_index][rows]
        with numpyro.plate("dept", N_DEPTS):
            sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(0.5))
            if config.dispersion:
                tau_phi = numpyro.sample("tau_phi", dist.HalfNormal(1.0))
            if config.recency:
                sigma_recency = numpyro.sample("sigma_recency", dist.HalfNormal(0.3))
            if config.seasonal:
                sigma_item_dow = numpyro.sample("sigma_item_dow", dist.HalfNormal(0.2))
                w_fourier = numpyro.sample(
                    "w_fourier", dist.Normal(0.0, 0.3).expand([2 * N_HARMONICS]).to_event(1)
                )
                sigma_event = numpyro.sample("sigma_event", dist.HalfNormal(0.3))
                gamma_event = numpyro.sample(
                    "gamma_event",
                    dist.Normal(0.0, sigma_event[:, None]).expand([N_DEPTS, N_EVENTS]).to_event(1),
                )
            if config.price:
                rho_dept = numpyro.sample("rho_dept", dist.Normal(-1.0, 1.0))
        block = block_index(h)
        with numpyro.plate("store", N_STORES):
            store_level = jnp.zeros((h.duration, N_STORES))
            if config.walks:
                store_walk_scale = numpyro.sample("store_walk_scale", dist.HalfNormal(0.1))
                store_level = (
                    store_level + block_walk("store_walk", store_walk_scale, h, N_STORES)[block]
                )
            if config.shocks:
                store_shock_scale = numpyro.sample("store_shock_scale", dist.HalfNormal(0.05))
                store_level = store_level + weekly_shocks(
                    "store_shock", store_shock_scale, h, N_STORES
                )
        with numpyro.plate("group", N_GROUPS):
            if config.dispersion:
                log_phi_group = numpyro.sample("log_phi_group", dist.Normal(0.0, 1.0))
            seasonal_group = numpyro.sample(
                "seasonal_group", dist.ZeroSumNormal(0.3, event_shape=(7,))
            )
            snap_group = numpyro.sample("snap_group", dist.Normal(0.0, 0.3))
            group_level = jnp.zeros((h.duration, N_GROUPS))
            if config.walks:
                group_walk_scale = numpyro.sample("group_walk_scale", dist.HalfNormal(0.1))
                group_level = (
                    group_level + block_walk("group_walk", group_walk_scale, h, N_GROUPS)[block]
                )
        with numpyro.plate("series", n_series):
            batch = numpyro.subsample(covariates, event_dim=0)
            y = None if data is None else numpyro.subsample(data, event_dim=0)
            store = numpyro.subsample(store_index, event_dim=0)
            group = numpyro.subsample(group_index, event_dim=0)
            dept = numpyro.subsample(dept_index, event_dim=0)
            offset = numpyro.subsample(level_at_origin, event_dim=0)
            alpha = offset + numpyro.sample("alpha", dist.Normal(0.0, sigma_alpha[dept]))
            if config.dispersion:
                log_phi = numpyro.sample(
                    "log_phi", dist.Normal(log_phi_group[group], tau_phi[dept])
                )
            eta = alpha + seasonal_group[group][:, dow].T
            eta = eta + snap_group[group] * snap_table[t_index][rows][:, state_of_store[store]]
            if config.walks or config.shocks:
                eta = eta + store_level[rows][:, store] + group_level[rows][:, group]
            if config.seasonal:
                item_dow = sigma_item_dow[dept][:, None] * numpyro.sample(
                    "item_dow", dist.ZeroSumNormal(1.0, event_shape=(7,))
                )
                eta = eta + item_dow[:, dow].T + fourier_table[t_index][rows] @ w_fourier[dept].T
                for event_table in event_tables:
                    eta = eta + gamma_event[dept][:, event_table[t_index][rows]].T
            if config.price:
                eta = eta + rho_dept[dept] * batch[2][rows]
            if config.recency:
                eta = eta + block_walk("item_walk", sigma_recency[dept], h, None)[block[rows]]
            mean = batch[1][rows] * jnp.exp(jnp.clip(eta, -12.0, MAX_LOG_MEAN)) + 1e-6
            concentration = (
                jnp.exp(log_phi) if config.dispersion else jnp.full_like(alpha, jnp.inf)
            )
            if forecast_only:
                numpyro.deterministic("mean_future", mean)
                numpyro.deterministic("concentration", concentration)
            elif config.dispersion:
                predict(
                    Horizon.from_data(batch, y),
                    lambda m: dist.NegativeBinomial2(m, concentration),
                    mean,
                )
            else:
                predict(Horizon.from_data(batch, y), lambda m: dist.Poisson(m), mean)

    return model


def create_series_plates(covariates: Array, data: Array | None = None) -> numpyro.plate:
    """Subsample the series plate in the guide; the model replays the same indices."""
    return numpyro.plate("series", n_series, subsample_size=SUBSAMPLE_SIZE)


LADDER = {
    "V1 Poisson": ModelConfig(),
    "V2 + item dispersion (NB)": ModelConfig(dispersion=True),
    "V3 + item block walk": ModelConfig(dispersion=True, recency=True),
    "V4 + store and group walks, weekly shocks": ModelConfig(
        dispersion=True, recency=True, walks=True, shocks=True
    ),
    "V5 + yearly seasonality, events, item weekdays": ModelConfig(
        dispersion=True, recency=True, seasonal=True
    ),
    "V6 + price": ModelConfig(dispersion=True, recency=True, seasonal=True, price=True),
}

# %% [markdown]
# FIT_PLACEHOLDER

# %%
NUM_STEPS = 3_000
PEAK_LR = 0.03


def recent_level(train_counts: Array, train_saled: Array, days: int = BLOCK) -> Array:
    """Log of the mean sales of every series over its listed days of the last ``days`` days.

    Series with no listed day in that window fall back to the last 13 blocks (a year); series
    with no listed day at all get the floor.
    """
    listed_recent = train_saled[-days:].sum(0)
    mean_recent = train_counts[-days:].sum(0) / jnp.maximum(listed_recent, 1.0)
    listed_year = train_saled[-N_BLOCKS * BLOCK :].sum(0)
    mean_year = train_counts[-N_BLOCKS * BLOCK :].sum(0) / jnp.maximum(listed_year, 1.0)
    mean = jnp.where(listed_recent > 0, mean_recent, jnp.where(listed_year > 0, mean_year, 0.0))
    return jnp.log(mean + 0.05)


def fit_model(
    rng_key: Array,
    model: ForecastModel,
    train_covariates: Array,
    train_counts: Array,
    *,
    num_steps: int = NUM_STEPS,
) -> tuple[AutoNormal, SVIRunResult, float]:
    """Fit a model variant with subsampled SVI; return the guide, the result and the wall time."""
    guide = AutoNormal(model, init_loc_fn=init_to_median, create_plates=create_series_plates)
    schedule = optax.linear_onecycle_schedule(
        transition_steps=num_steps,
        peak_value=PEAK_LR,
        pct_start=0.2,
        pct_final=0.8,
        div_factor=10,
        final_div_factor=10,
    )
    optimizer = optax_to_numpyro(
        optax.chain(optax.clip_by_global_norm(10.0), optax.adam(schedule))
    )
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    start = perf_counter()
    result = svi.run(rng_key, num_steps, train_covariates, train_counts, progress_bar=False)
    jax.block_until_ready(result.losses)
    return guide, result, perf_counter() - start


def forecast_bottom_level(
    rng_key: Array,
    config: ModelConfig,
    guide: AutoNormal,
    params: dict,
    train_counts: Array,
    full_covariates: Array,
    num_samples: int,
    *,
    chunk: int = 50,
) -> np.ndarray:
    """Draw ``num_samples`` bottom-level forecasts of every series, ``chunk`` posterior draws at a time.

    Each chunk draws the posterior, runs the forecast-only model for the horizon mean and
    dispersion of every series, and samples the negative binomial counts with NumPy.
    """
    model_fc = make_model(
        config,
        recent_level(train_counts, full_covariates[1][: train_counts.shape[0]]),
        forecast_only=True,
    )
    n_days = full_covariates.shape[1] - train_counts.shape[0]
    draws = np.empty((num_samples, n_days, train_counts.shape[1]), dtype=np.float32)
    generator = np.random.default_rng(int(rng_key[1]))
    for start in range(0, num_samples, chunk):
        key_post, key_pred = random.split(random.fold_in(rng_key, start))
        posterior = draw_posterior(key_post, guide, params, min(chunk, num_samples - start))
        pred = Predictive(model_fc, posterior, return_sites=["mean_future", "concentration"])(
            key_pred, full_covariates, train_counts
        )
        mean = np.asarray(pred["mean_future"], dtype=np.float64)
        concentration = np.asarray(pred["concentration"], dtype=np.float64)[:, None, :]
        if np.isinf(concentration).all():
            draws[start : start + mean.shape[0]] = generator.poisson(mean)
        else:
            draws[start : start + mean.shape[0]] = generator.negative_binomial(
                concentration, concentration / (concentration + mean)
            )
    return draws


def make_forecast_fn(config: ModelConfig):
    """Build the `backtest` closure of a model variant: build the target, fit, forecast."""

    def forecast_fn(
        rng_key: Array,
        model: ForecastModel,
        train_data: Array,
        train_covariates: Array,
        full_covariates: Array,
        num_samples: int,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        key_fit, key_fc = random.split(rng_key)
        t1 = train_data.shape[0]
        train_covariates = train_covariates[:, -HISTORY_DAYS:]
        train_counts = jnp.rint(train_data[-HISTORY_DAYS:] * train_covariates[1]).astype(jnp.int32)
        model = make_model(config, recent_level(train_counts, train_covariates[1]))
        guide, result, _ = fit_model(key_fit, model, train_covariates, train_counts)
        return forecast_bottom_level(
            key_fc,
            config,
            guide,
            result.params,
            train_counts,
            full_covariates[:, t1 - HISTORY_DAYS :],
            num_samples,
        )

    return forecast_fn


# %% [markdown]
# BASELINE_PLACEHOLDER

# %%
day_of_month = calendar_df["date"].dt.day().to_numpy()[:N_DAYS]
covariates_top = jnp.asarray(
    np.column_stack(
        [
            np.arange(N_DAYS, dtype=np.float32) / 365.0,
            calendar_df["dow"].to_numpy()[:N_DAYS].astype(np.float32),
            np.stack([(day_of_month == day).astype(np.float32) for day in range(1, 32)], axis=-1),
        ]
    )
)


def top_down_model(covariates: Array, data: Array | None = None) -> None:
    """Kit model 1: trend, weekday and day-of-month effects on the log of the total sales."""
    h = Horizon.from_data(covariates, data)
    time = covariates[:, 0]
    dow = covariates[:, 1].astype(jnp.int32)
    feature = covariates[:, 2:]
    bias = numpyro.sample("bias", dist.Normal(0.0, 10.0))
    trend = numpyro.sample("trend", dist.LogNormal(-2.0, 1.0))
    weight = numpyro.sample(
        "weight", dist.Normal(0.0, 1.0).expand([feature.shape[-1]]).to_event(1)
    )
    with numpyro.plate("day_of_week", 7):
        seasonal = numpyro.sample("seasonal", dist.Normal(0.0, 5.0))
    prediction = bias + trend * time + seasonal[dow] + feature @ weight
    dof = numpyro.sample("dof", dist.Uniform(1.0, 10.0))
    noise_scale = numpyro.sample("noise_scale", dist.LogNormal(-2.0, 1.0))
    predict(h, dist.StudentT(dof, 0.0, noise_scale), prediction[:, None])


def forecast_fn_top_down(
    rng_key: Array,
    model: ForecastModel,
    train_data: Array,
    train_covariates: Array,
    full_covariates: Array,
    num_samples: int,
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    """Fit the kit's top-down model and split its forecast to the items with Poisson noise."""
    key_fit, key_post, key_fc = random.split(rng_key, 3)
    y = jnp.log(train_data.sum(-1, keepdims=True))
    guide = AutoNormal(model)
    schedule = optax.exponential_decay(0.1, transition_steps=1_001, decay_rate=0.1)
    optimizer = optax_to_numpyro(
        optax.chain(optax.clip_by_global_norm(10.0), optax.adam(schedule))
    )
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    result = svi.run(key_fit, 1_001, train_covariates, y, progress_bar=False)
    posterior = draw_posterior(key_post, guide, result.params, num_samples)
    total = np.exp(np.asarray(forecast(key_fc, model, posterior, y, full_covariates)))
    last_28 = np.asarray(train_data[-28:]).sum(0)
    shares = (last_28 / last_28.sum()).astype(np.float32)
    generator = np.random.default_rng(int(train_data.shape[0]))
    draws = np.empty((num_samples, total.shape[1], shares.size), dtype=np.float32)
    for start in range(0, num_samples, 50):
        draws[start : start + 50] = generator.poisson(total[start : start + 50] * shares)
    return draws


# %% [markdown]
# LADDER_PLACEHOLDER

# %%
SELECTION_ORIGIN = REFERENCE_DAYS + 35  # 1,878: the window the ladder is selected on
HELD_OUT_ORIGIN = SELECTION_ORIGIN + 35  # 1,913: the kit's last backtest window, held out
NUM_SAMPLES_WINDOW = 250
sales_train = jnp.asarray(sales[:N_DAYS_TRAIN])


def run_window(
    rng_key: Array,
    name: str,
    model: ForecastModel,
    covariates_model: Array,
    forecast_fn,
    origin: int,
) -> pl.DataFrame:
    """Backtest one model on the single window that starts its forecast at ``origin``."""
    results = backtest(
        rng_key,
        lambda: model,
        sales_train[: origin + HORIZON],
        covariates_model[..., : origin + HORIZON, :],
        forecast_fn=forecast_fn,
        test_window=HORIZON,
        stride=35,
        min_train_window=origin,
        num_samples=NUM_SAMPLES_WINDOW,
        transform=aggregate_transform,
        per_window_metrics=m5_metrics,
    )
    return summarize_backtest(results, name)


rng_key, key_top = random.split(rng_key)
ladder_rows = [
    run_window(
        key_top,
        "V0 top-down (kit)",
        top_down_model,
        covariates_top,
        forecast_fn_top_down,
        SELECTION_ORIGIN,
    )
]
for name, config in LADDER.items():
    rng_key, key_run = random.split(rng_key)
    start = perf_counter()
    ladder_rows.append(
        run_window(
            key_run, name, top_down_model, covariates, make_forecast_fn(config), SELECTION_ORIGIN
        )
    )
    print(f"{name}: {perf_counter() - start:.0f} s")
ladder_df = pl.concat(ladder_rows)
ladder_df.select("model", "walltime", "ws_crps", "ws_crps_1_9", "ws_crps_10_12").with_columns(
    pl.col(pl.Float64).round(3)
).with_columns(pl.col("walltime").round(0))

# %% [markdown]
# SEED_PLACEHOLDER

# %%
rng_key, key_seed = random.split(rng_key)
seed_row = run_window(
    key_seed,
    "V3 + item block walk (second seed)",
    top_down_model,
    covariates,
    make_forecast_fn(LADDER["V3 + item block walk"]),
    SELECTION_ORIGIN,
)
seed_noise = float(
    abs(
        seed_row["ws_crps"][0]
        - ladder_df.filter(pl.col("model").eq(pl.lit("V3 + item block walk")))["ws_crps"][0]
    )
)
print(
    f"WS-CRPS of V3 with a second SVI seed: {seed_row['ws_crps'][0]:.4f} (seed noise {seed_noise:.4f})"
)

# %%
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(ladder_df.height)
width = 0.27
for k, (column, label) in enumerate(
    [
        ("ws_crps", "all levels"),
        ("ws_crps_1_9", "levels 1 to 9"),
        ("ws_crps_10_12", "levels 10 to 12"),
    ]
):
    ax.bar(x + (k - 1) * width, ladder_df[column].to_numpy(), width, label=label)
ax.set_xticks(x, ladder_df["model"].to_list(), rotation=30, ha="right")
ax.axhline(float(ladder_df["ws_crps"][0]), color="gray", ls="--", lw=1)
ax.set(
    title=f"Model ladder on the selection window (origin day {SELECTION_ORIGIN:,})",
    ylabel="WS-CRPS",
)
ax.legend();
# %% [markdown]
# FINAL_CHOICE_PLACEHOLDER

# %%
ladder_only = ladder_df.filter(pl.col("model").ne(pl.lit("V0 top-down (kit)")))
best_ws_crps = float(ladder_only["ws_crps"].min())
within_noise = ladder_only.filter(pl.col("ws_crps").le(pl.lit(best_ws_crps + seed_noise)))
FINAL_NAME = within_noise["model"][0]  # the simplest rung within seed noise of the best one
FINAL = LADDER[FINAL_NAME]
print(
    f"best rung: {ladder_only.sort('ws_crps')['model'][0]} ({best_ws_crps:.4f}); kept: {FINAL_NAME}"
)
FINAL

# %% [markdown]
# FINAL_FIT_PLACEHOLDER

# %%
covariates_final = covariates[:, N_DAYS_TRAIN - HISTORY_DAYS :]
counts_final = jnp.rint(sales_train[-HISTORY_DAYS:] * covariates_final[1, :HISTORY_DAYS]).astype(
    jnp.int32
)
level_final = recent_level(counts_final, covariates_final[1, :HISTORY_DAYS])
final_model = make_model(FINAL, level_final)
rng_key, key_fit = random.split(rng_key)
guide_final, svi_final, time_final = fit_model(
    key_fit, final_model, covariates_final[:, :HISTORY_DAYS], counts_final
)
print(
    f"final model: {NUM_STEPS} steps in {time_final:.0f} s, final loss {float(svi_final.losses[-1]):.4g}"
)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(np.asarray(svi_final.losses[100:]), color="C0")
ax.set(title="Final model: ELBO loss (from step 100)", xlabel="SVI step", ylabel="loss");
# %% [markdown]
# FINAL_POSTERIOR_PLACEHOLDER

# %%
rng_key, key_post = random.split(rng_key)
posterior_final = draw_posterior(key_post, guide_final, svi_final.params, 500)
DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
posterior_shared = {
    name: np.asarray(value)[None]
    for name, value in posterior_final.items()
    if name
    in ("sigma_alpha", "tau_phi", "sigma_recency", "log_phi_group", "seasonal_group", "snap_group")
}
tree_final = az.from_dict(
    {"posterior": posterior_shared},
    coords={"dept": dept_ids, "group": group_ids, "day_of_week": DOW_LABELS},
    dims={
        "sigma_alpha": ["dept"],
        "tau_phi": ["dept"],
        "sigma_recency": ["dept"],
        "log_phi_group": ["group"],
        "seasonal_group": ["group", "day_of_week"],
        "snap_group": ["group"],
    },
)
az.summary(tree_final, var_names=["sigma_alpha", "tau_phi", "sigma_recency"])

# %%
ca1_groups = [label for label in group_ids if label.startswith("CA_1/")]
pc = az.plot_forest(
    tree_final,
    var_names=["seasonal_group"],
    coords={"group": ca1_groups},
    combined=True,
    labels=["group", "day_of_week"],
    figure_kwargs={"figsize": (8, 12)},
)
pc.viz["figure"].item().suptitle(
    "Weekday effects of the departments of store CA_1 (log scale)", fontsize=14
);
# %%
pc = az.plot_forest(
    tree_final,
    var_names=["snap_group"],
    combined=True,
    labels=["group"],
    figure_kwargs={"figsize": (8, 14)},
)
pc.viz["figure"].item().suptitle("SNAP effect by store-department (log scale)", fontsize=14);
# %% [markdown]
# FINAL_ITEMS_PLACEHOLDER

# %%
HDI_PROBS = (0.94, 0.5)
HDI_ALPHAS = (0.3, 0.6)
dates = calendar_df["date"].to_numpy()[:N_DAYS]
date_num = np.asarray(mdates.date2num(dates.astype("datetime64[s]").astype(object)))
split_date = date_num[N_DAYS_TRAIN]
plot_start = N_DAYS_TRAIN - 16 * 7
last_year_sales = sales[N_DAYS_TRAIN - 365 : N_DAYS_TRAIN].sum(0)
focus_df = (
    hierarchy_df.with_row_index("n")
    .with_columns(last_year=pl.Series(last_year_sales))
    .filter(pl.col("store_id").eq(pl.lit("CA_1")))
    .sort("last_year", descending=True)
    .group_by("dept_id", maintain_order=True)
    .head(1)
    .sort("dept_id")
)
focus_index = focus_df["n"].to_numpy()
focus_labels = focus_df["id"].to_list()
ITEM_SITES = {"alpha": -1, "log_phi": -1, "item_walk": -1, "item_dow": -2}


def subset_posterior(posterior: dict, index: np.ndarray) -> dict:
    """Keep the draws of the series in ``index`` for the item-level sites, everything else as is."""
    out = {}
    for name, value in posterior.items():
        if name in ITEM_SITES:
            axis = ITEM_SITES[name] % value.ndim
            out[name] = jnp.take(value, jnp.asarray(index), axis=axis)
        else:
            out[name] = value
    return out


def hdi_label(prob: float) -> str:
    r"""Legend label of an HDI band, for example ``$94\%$ HDI``."""
    return rf"${prob:.0%}$ HDI".replace("%", r"\%")


def plot_series_panel(
    train_draws: np.ndarray | Array | None,
    test_draws: np.ndarray | Array,
    truth: np.ndarray,
    labels: list[str],
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    ylabel: str,
    suptitle: str,
    col_wrap: int = 2,
    figsize: tuple[float, float] = (15.0, 10.0),
    group: str = "posterior_predictive",
) -> None:
    """Facet the predictive bands of a few series over a window, with the observed series."""
    visuals = {
        "ci_band": {"color": "C0"},
        "observed_scatter": False,
        "pe_line": False,
        "xlabel": False,
        "ylabel": False,
    }
    first_draws, first_x = (test_draws, x_test) if train_draws is None else (train_draws, x_train)
    pc = az.plot_lm(
        predictions_to_datatree(
            np.asarray(first_draws, dtype=np.float32), first_x, labels, group=group
        ),
        y="obs",
        x="t",
        plot_dim="time",
        group=group,
        ci_kind="hdi",
        ci_prob=HDI_PROBS,
        smooth=False,
        col_wrap=col_wrap,
        visuals=visuals if train_draws is not None else {**visuals, "ci_band": {"color": "C1"}},
        aes={"alpha": ["prob"]},
        alpha=HDI_ALPHAS,
        figure_kwargs={"figsize": figsize},
    )
    if train_draws is not None:
        az.plot_lm(
            predictions_to_datatree(
                np.asarray(test_draws, dtype=np.float32), x_test, labels, group=group
            ),
            y="obs",
            x="t",
            plot_dim="time",
            group=group,
            plot_collection=pc,
            ci_kind="hdi",
            ci_prob=HDI_PROBS,
            smooth=False,
            visuals={**visuals, "ci_band": {"color": "C1"}},
        )
    x_all = np.concatenate([x_train, x_test]) if train_draws is not None else x_test
    truth_da = xr.DataArray(
        np.asarray(truth)[-len(x_all) :],
        dims=["time", "series"],
        coords={"time": x_all, "series": labels},
    ).rename("t")
    x_da = xr.DataArray(x_all, dims=["time"], coords={"time": x_all})
    pc.map(
        az.visuals.line_xy,
        "truth",
        data=truth_da,
        x=x_da,
        ignore_aes=pc.aes_set,
        color="black",
        lw=1,
    )
    for label in labels:
        ax = pc.get_target("t", {"series": label})
        ax.set_title(label, fontsize=11)
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax0 = pc.get_target("t", {"series": labels[0]})
    handles = []
    for prob in HDI_PROBS:
        band = pc.viz["ci_band"]["t"].sel(series=labels[0], prob=prob).item()
        band.set_label(hdi_label(prob))
        handles.append(band)
    truth_line = pc.viz["truth"]["t"].sel(series=labels[0]).item()
    truth_line.set_label("observed")
    ax0.legend(handles=[*handles, truth_line], loc="upper left", fontsize=9)
    fig = pc.viz["figure"].item()
    fig.supylabel(ylabel)
    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.02)


focus_model = make_model(
    FINAL,
    level_final[focus_index],
    store_index=store_index[focus_index],
    group_index=group_index[focus_index],
    dept_index=dept_index[focus_index],
)
posterior_focus = subset_posterior(posterior_final, focus_index)
covariates_focus = covariates_final[:, :, focus_index]
rng_key, key_prior, key_fc = random.split(rng_key, 3)
prior_focus = Predictive(focus_model, num_samples=500, return_sites=["obs"])(
    key_prior, covariates_focus[:, :HISTORY_DAYS]
)["obs"]
fc_focus = forecast(
    key_fc, focus_model, posterior_focus, counts_final[:, focus_index], covariates_focus
)
plot_series_panel(
    None,
    prior_focus[:, -16 * 7 :],
    sales[plot_start:N_DAYS_TRAIN, focus_index],
    focus_labels,
    date_num[plot_start:N_DAYS_TRAIN],
    date_num[plot_start:N_DAYS_TRAIN],
    ylabel="units sold",
    suptitle="Prior predictive check of the focus items (last 16 training weeks)",
    group="prior_predictive",
)

# %%
rng_key, key_pp = random.split(rng_key)
pp_focus = predict_in_sample(
    key_pp, focus_model, posterior_focus, covariates_focus[:, :HISTORY_DAYS]
)
plot_series_panel(
    np.asarray(pp_focus)[:, -16 * 7 :],
    fc_focus,
    sales[plot_start:N_DAYS, focus_index],
    focus_labels,
    date_num[plot_start:N_DAYS_TRAIN],
    date_num[N_DAYS_TRAIN:N_DAYS],
    ylabel="units sold",
    suptitle="Final model: in-sample predictive and forecast of the focus items",
)
del prior_focus, pp_focus

# %% [markdown]
# HOLDOUT_PLACEHOLDER

# %%
rng_key, key_fc = random.split(rng_key)
start = perf_counter()
bottom_final = forecast_bottom_level(
    key_fc, FINAL, guide_final, svi_final.params, counts_final, covariates_final, 500
)
print(f"bottom-level forecast {bottom_final.shape} in {perf_counter() - start:.0f} s")
truth_holdout = sales[N_DAYS_TRAIN:N_DAYS]
holdout_metrics = m5_metrics(0, N_DAYS_TRAIN, N_DAYS)
weights_holdout = m5_weights(N_DAYS_TRAIN)
scales_holdout = m5_scales(sales_agg[:N_DAYS_TRAIN])
pred_levels, truth_levels = aggregate_transform(bottom_final, truth_holdout)
scores_final = evaluate_forecast(pred_levels, truth_levels, metrics=holdout_metrics)
wspl_final = ws_pinball(pred_levels, truth_levels, weights_holdout, scales_holdout)
level1_final = pred_levels[..., 0]
level3_final = pred_levels[..., level_slices["Level3"]]
del bottom_final, pred_levels

rng_key, key_top = random.split(rng_key)
bottom_top = forecast_fn_top_down(
    key_top, top_down_model, sales_train, covariates_top[:N_DAYS_TRAIN], covariates_top, 500
)
pred_levels, _ = aggregate_transform(bottom_top, truth_holdout)
scores_top = evaluate_forecast(pred_levels, truth_levels, metrics=holdout_metrics)
wspl_top = ws_pinball(pred_levels, truth_levels, weights_holdout, scales_holdout)
level1_top = pred_levels[..., 0]
del bottom_top, pred_levels

holdout_table = pl.DataFrame(
    {
        "level": [*LEVELS, "mean", "WSPL"],
        "top-down (kit)": [
            *[scores_top[f"ws_crps_{level}"] for level in LEVELS],
            np.mean(list(scores_top.values())),
            wspl_top,
        ],
        "final": [
            *[scores_final[f"ws_crps_{level}"] for level in LEVELS],
            np.mean(list(scores_final.values())),
            wspl_final,
        ],
    }
).with_columns(pl.col(pl.Float64).round(3))
holdout_table

# %% [markdown]
# HELDOUT_WINDOW_PLACEHOLDER

# %%
rng_key, key_a, key_b = random.split(rng_key, 3)
heldout_df = pl.concat(
    [
        run_window(
            key_a,
            "V0 top-down (kit)",
            top_down_model,
            covariates_top,
            forecast_fn_top_down,
            HELD_OUT_ORIGIN,
        ),
        run_window(
            key_b, FINAL_NAME, top_down_model, covariates, make_forecast_fn(FINAL), HELD_OUT_ORIGIN
        ),
    ]
)
heldout_df.select(
    "model", "t1", "walltime", "ws_crps", "ws_crps_1_9", "ws_crps_10_12"
).with_columns(pl.col(pl.Float64).round(3)).with_columns(pl.col("walltime").round(0))

# %% [markdown]
# COMPARISON_PLACEHOLDER

# %% tags=["thumbnail"]
fig, axes = plt.subplots(
    nrows=2, ncols=1, figsize=(12, 9), sharex=True, sharey=True, layout="constrained"
)
x_hist = date_num[plot_start:N_DAYS_TRAIN]
x_test = date_num[N_DAYS_TRAIN:N_DAYS]
y_total = sales_agg[:, 0]
for ax, (name, draws, ws) in zip(
    axes,
    [("top-down (kit)", level1_top, scores_top), ("final model", level1_final, scores_final)],
    strict=True,
):
    for prob, alpha in zip(HDI_PROBS, HDI_ALPHAS, strict=True):
        hdi = az.hdi(draws.T, prob=prob)
        ax.fill_between(
            x_test, hdi[:, 0], hdi[:, 1], color="C1", alpha=alpha, label=hdi_label(prob)
        )
    ax.plot(x_hist, y_total[plot_start:N_DAYS_TRAIN], color="black", lw=1, label="observed")
    ax.plot(x_test, y_total[N_DAYS_TRAIN:N_DAYS], color="black", lw=1)
    ax.axvline(split_date, color="gray", ls="--")
    ax.set(title=f"{name}: WS-CRPS {np.mean(list(ws.values())):.3f}", ylabel="units sold")
axes[0].legend(loc="upper left")
axes[-1].xaxis_date()
fig.suptitle(
    "Total daily sales: forecasts of the evaluation window", fontsize=16, fontweight="bold"
);
# %%
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(LEVELS))
width = 0.4
ax.bar(
    x - width / 2,
    holdout_table["top-down (kit)"][: len(LEVELS)].to_numpy(),
    width,
    label="top-down (kit)",
)
ax.bar(x + width / 2, holdout_table["final"][: len(LEVELS)].to_numpy(), width, label="final model")
ax.set_xticks(x, list(LEVELS), rotation=45)
ax.set(title="Evaluation window: WS-CRPS by level", ylabel="WS-CRPS")
ax.legend();
# %%
plot_series_panel(
    None,
    level3_final,
    sales_agg[N_DAYS_TRAIN:N_DAYS, level_slices["Level3"]],
    store_ids,
    date_num[plot_start:N_DAYS_TRAIN],
    date_num[N_DAYS_TRAIN:N_DAYS],
    ylabel="units sold",
    suptitle="Final model: store-level forecasts of the evaluation window",
    figsize=(15.0, 14.0),
)

# %% [markdown]
# TIMING_PLACEHOLDER

# %%
fit_times = pl.DataFrame(
    {
        "model": ["top-down (kit)", "bottom-up (kit)", "middle-out (kit)", "final model"],
        "fit_seconds": [3.0, 57.0, 5.0, time_final],
        "series": [1, n_series, N_GROUPS, n_series],
    }
).with_columns(pl.col("fit_seconds").round(0))
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(fit_times["model"], fit_times["fit_seconds"], color=["C0", "C1", "C2", "C3"])
ax.set(title="SVI fitting time of the final fits", ylabel="seconds")
fit_times

# %% [markdown]
# DISCUSSION_PLACEHOLDER
