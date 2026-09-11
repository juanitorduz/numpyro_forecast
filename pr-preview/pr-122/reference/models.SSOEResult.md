## models.SSOEResult


The means and sampled future values produced by [ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe).


Usage

``` python
models.SSOEResult(
    mu,
    mu_future,
    y_future,
)
```


## Attributes


`mu: Float[Array, ``" *batch t_obs obs"]`  
In-sample one-step-ahead means, shape `(*batch, t_obs, obs)`: the predictor the caller writes its likelihood against.

`mu_future: Float[Array, ``" *batch future obs"]`  
Forecast-horizon one-step-ahead means, shape `(*batch, future, obs)` (a size-0 time axis while training).

`y_future: Float[Array, ``" *batch future obs"]`  
Sampled future values `mu_future + eps`, shape `(*batch, future, obs)` (a size-0 time axis while training); the caller registers them as the `"forecast"` deterministic when `h.future > 0`.
