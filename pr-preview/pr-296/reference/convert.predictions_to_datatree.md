## convert.predictions_to_datatree()


Pack prediction draws into a DataTree laid out for per-series `plot_lm` faceting.


Usage

``` python
convert.predictions_to_datatree(
    predictions, x, series, *, group=Group.POSTERIOR_PREDICTIVE, observed=None
)
```


The array-level counterpart of [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree): instead of a fit, it takes prediction draws from **any** predictive group (prior predictive, posterior predictive, or forecasts), possibly already transformed (rescaled to original units, clipped at zero, subset to a few series). The draws get a single pseudo-chain, and `constant_data` carries the independent variable `"t"` broadcast to `(time, series)` so that `arviz.plot_lm(tree, y="obs", x="t", plot_dim="time", ...)` facets one panel per series; band artists are then reachable via `pc.viz["ci_band"]["t"]` and axes via `pc.get_target("t", {"series": label})`.

`plot_lm` requires an `observed_data` group even when the observation scatter is disabled, so when `observed` is `None` a zeros placeholder is stored; it is never drawn under `visuals={"observed_scatter": False}`.


## Parameters


`predictions: Real[ArrayLike, ``" sample time series"]`  
Prediction draws with the sample axis first, shape `(sample, time, series)`. Integer or floating; the integer draws of a count model are stored as they are, keeping their dtype (`plot_lm` draws them unchanged).

`x: Num[ArrayLike, ``" time"]`  
Independent-variable values, shape `(time,)`. Must be numeric: `plot_lm` cannot draw `datetime64` values (it concatenates `x` with the predictions internally), so pass `matplotlib.dates.date2num()` floats and re-format the tick labels with `matplotlib.dates.ConciseDateFormatter`.

`series: Collection[Any]`  
One label per series, defining the `series` coordinate. Any one-dimensional collection is accepted (a list, a NumPy array, a pandas index) and stored as `np.asarray(series)`.

`group: str = Group.POSTERIOR_PREDICTIVE`  
Predictive group to store the draws under, `"posterior_predictive"` by default (also `"prior_predictive"` or `"predictions"`). `"observed_data"` and `"constant_data"` are reserved: the function always writes those two groups.

`observed: Real[ArrayLike, ``" time series"] | None`` = None`  
Optional observations, shape `(time, series)`, stored in `observed_data`; when `None` a zeros placeholder is stored instead. Integer or floating, stored with its own dtype, so the integer observations of a count model need no cast.


## Returns


`xarray.DataTree`  
A tree with the `group`, `observed_data`, and `constant_data` groups; `obs` has dims `(chain, draw, time, series)` and `t` has dims `(time, series)`.


## Raises


`ValueError`  
If `group` is `"observed_data"` or `"constant_data"` (reserved names), if `series` is not a one-dimensional collection of labels (a set, a dict, a scalar, a 2-D array, ragged input) or is a `pd.MultiIndex`, or if `series` does not have one label per series in `predictions`.
