## convert.predictions_to_datatree()


Pack prediction draws into a DataTree laid out for per-series `plot_lm` faceting.


Usage

``` python
convert.predictions_to_datatree(
    predictions, x, series, *, group="posterior_predictive", observed=None
)
```


The array-level counterpart of [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree): instead of a fit, it takes prediction draws from **any** predictive group (prior predictive, posterior predictive, or forecasts), possibly already transformed (rescaled to original units, clipped at zero, subset to a few series). The draws get a single pseudo-chain, and `constant_data` carries the independent variable `"t"` broadcast to `(time, series)` so that `arviz.plot_lm(tree, y="obs", x="t", plot_dim="time", ...)` facets one panel per series; band artists are then reachable via `pc.viz["ci_band"]["t"]` and axes via `pc.get_target("t", {"series": label})`.

`plot_lm` requires an `observed_data` group even when the observation scatter is disabled, so when `observed` is `None` a zeros placeholder is stored; it is never drawn under `visuals={"observed_scatter": False}`.


## Parameters


`predictions: Float[np.ndarray | Array, ``" sample time series"]`  
Prediction draws with the sample axis first, shape `(sample, time, series)`.

`x: Num[np.ndarray | Array, ``" time"]`  
Independent-variable values, shape `(time,)`. Must be numeric: `plot_lm` cannot draw `datetime64` values (it concatenates `x` with the float predictions internally), so pass `matplotlib.dates.date2num()` floats and re-format the tick labels with `matplotlib.dates.ConciseDateFormatter`.

`series: Sequence[Any]`  
One label per series, defining the `series` coordinate.

`group: str = ``"posterior_predictive"`  
Predictive group to store the draws under (e.g. `"prior_predictive"`, `"posterior_predictive"`, `"predictions"`).

`observed: Float[np.ndarray | Array, ``" time series"] | None`` = None`  
Optional observations, shape `(time, series)`, stored in `observed_data`; when `None` a zeros placeholder is stored instead.


## Returns


`xarray.DataTree`  
A tree with the `group`, `observed_data`, and `constant_data` groups; `obs` has dims `(chain, draw, time, series)` and `t` has dims `(time, series)`.


## Raises


`ValueError`  
If `series` does not have one label per series in `predictions`.
