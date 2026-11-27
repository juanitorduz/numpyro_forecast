## metrics.make_mase()


Build a Mean Absolute Scaled Error metric scaled by `train_data`.


Usage

``` python
metrics.make_mase(
    train_data,
    *,
    seasonality=1,
)
```


MASE divides the forecast MAE (using the sample median as point estimate) by the in-sample MAE of the seasonal-naive forecast on `train_data`, `mean(|y_t - y_{t-seasonality}|)`. The scale is computed once at factory time; the returned metric has the standard scalar-array signature (see [Metric](typing.Metric.md#numpyro_forecast.typing.Metric)). `train_data` and the returned metric's `pred`/`truth` are all moved to device memory first (`float_view()`), so any of them may be host-committed, e.g. draws sampled with `device="host"`. Integer counts are cast to the default float dtype in the same step, so the seasonal-naive differences of `train_data` are computed in floating point: in the integer dtype a `uint8` difference wraps and the [abs](typing.Array.md#numpyro_forecast.typing.Array.__abs__) of the `int16` minimum stays negative.


## Parameters


`train_data: Real[ArrayLike, ``" *batch time obs_dim"]`  
Training data with time at axis `-2`; leading batch axes are allowed and the seasonal-naive scale is averaged over all axes. Integer counts are accepted.

`seasonality: int = ``1`  
Seasonal period (`>= 1`); `1` is the random-walk naive baseline.


## Returns


`Metric`  
A `(pred, truth)` callable computing MASE as a scalar array. It raises `ValueError` if `truth.shape` is not `pred.shape[1:]` (see [Metric](typing.Metric.md#numpyro_forecast.typing.Metric)).


## Raises


`ValueError`  
If `seasonality < 1`, `train_data` is not longer than `seasonality` along the time axis, or the seasonal-naive scale is zero (a constant series, for which MASE is undefined).
