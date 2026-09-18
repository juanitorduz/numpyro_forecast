## models.predict()


Register the observation and forecast sites for the model.


Usage

``` python
models.predict(
    h,
    obs_dist,
    prediction,
)
```


`prediction` is the deterministic predictor over the full horizon (time at axis `-2`, shape `(*batch, duration, obs)`). `obs_dist` takes one of two forms. A `numpyro.distributions.Distribution` is zero-centered observation noise (e.g. `dist.StudentT(nu, 0.0, sigma)`) shifted onto the predictor with [shift_loc()](surgery.shift_loc.md#numpyro_forecast.surgery.shift_loc), which also owns the multivariate-normal layout check. A callable is a link mapping the predictor to the observation distribution directly (the GLM form, e.g. `lambda eta: dist.Poisson(jnp.exp(eta))`). Either way, while training the observation site `"obs"` is observed; while forecasting the in-sample prefix is observed and the forecast suffix is sampled at `"obs_future"` and exposed as the `"forecast"` deterministic site that [forecast()](predictive.forecast.md#numpyro_forecast.predictive.forecast) reads. The observation distribution must support time-axis surgery ([slice_time()](surgery.slice_time.md#numpyro_forecast.surgery.slice_time) / [prefix_condition()](surgery.prefix_condition.md#numpyro_forecast.surgery.prefix_condition)), i.e. an elementwise family or a registered one.


## Parameters


`h: Horizon`  
The horizon for the current model call (see [Horizon](models.Horizon.md#numpyro_forecast.models.Horizon)).

`obs_dist: dist.Distribution | Callable[[Array], dist.Distribution]`  
Zero-centered noise distribution (shifted onto `prediction`) or a link callable from the full-horizon predictor to the observation distribution.

`prediction: Array`  
The deterministic predictor over the full horizon, time at axis `-2`.


## Raises


`TypeError`  
If `obs_dist` is a callable that does not return a distribution (for example a link that returns the predictor itself).

`RuntimeError`  
If forecasting (`future > 0`) but no observed data is available.

`ValueError`  
If the observation distribution has discrete support but `h.data` is not integer-dtyped (the usual mistake for count models).
