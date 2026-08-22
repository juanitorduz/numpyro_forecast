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


MASE divides the forecast MAE (using the sample median as point estimate) by the in-sample MAE of the seasonal-naive forecast on `train_data`, `mean(|y_t - y_{t-seasonality}|)`. The scale is computed once at factory time; the returned metric has the standard scalar-array signature (see `~numpyro_forecast.typing.Metric`). `train_data` and the returned metric's `pred`/`truth` are all moved to device memory first (`~numpyro_forecast.functional._offload._device_view()`), so any of them may be host-committed, e.g. draws sampled with `device="host"`.


## Parameters


`train_data: Float[ArrayLike, ``"*batch time obs_dim"]`  
Training data with time at axis `-2`; leading batch axes are allowed and the seasonal-naive scale is averaged over all axes.

`seasonality: int = ``1`  
Seasonal period (`>= 1`); `1` is the random-walk naive baseline.


## Returns


`Metric`  
A `(pred, truth)` callable computing MASE as a scalar array.


## Raises


`ValueError`  
If `seasonality < 1`, `train_data` is not longer than `seasonality` along the time axis, or the seasonal-naive scale is zero (a constant series, for which MASE is undefined).
