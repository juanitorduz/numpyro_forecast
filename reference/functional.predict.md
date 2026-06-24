## functional.predict()


Register the observation/forecast sites for the model.


Usage

``` python
functional.predict(
    h,
    noise_dist,
    prediction,
)
```


`noise_dist` is a zero-centered observation noise distribution and `prediction` the deterministic mean over the full horizon. While training the residual is observed; while forecasting the in-sample prefix is observed and the forecast suffix is sampled and exposed as the `"forecast"` deterministic site.


## Parameters


`h: Horizon`  
The horizon for the current model call (see [Horizon](functional.Horizon.md#numpyro_forecast.functional.Horizon)).

`noise_dist: dist.Distribution`  
Zero-centered observation noise (e.g. `Normal(0, sigma)`).

`prediction: Array`  
Deterministic mean with time at axis `-2`, shape `(*batch, duration, obs)`.


## Raises


`RuntimeError`  
If forecasting (`future > 0`) but no observed data is available.
