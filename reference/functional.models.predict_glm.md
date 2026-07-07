## functional.models.predict_glm()


Register GLM-style observation/forecast sites from a latent predictor.


Usage

``` python
functional.models.predict_glm(
    h,
    obs_dist_fn,
    latent,
)
```


The generalized-linear counterpart of [predict()](functional.models.predict.md#numpyro_forecast.functional.models.predict): instead of a zero-centered noise distribution shifted by a mean, the caller supplies a link `obs_dist_fn` that maps the full-horizon `latent` predictor to the observation distribution directly (e.g. `lambda eta: Poisson(jnp.exp(eta))`). The prefix/suffix mirroring is identical to [predict()](functional.models.predict.md#numpyro_forecast.functional.models.predict): while training the observation is observed; while forecasting the in-sample prefix is observed and the forecast suffix is sampled and exposed as the `"forecast"` deterministic site. The observation distribution must support time-axis surgery (`~numpyro_forecast.surgery.slice_time()` / `~numpyro_forecast.surgery.prefix_condition()`), i.e. an elementwise family.


## Parameters


`h: Horizon`  
The horizon for the current model call (see [Horizon](functional.models.Horizon.md#numpyro_forecast.functional.models.Horizon)).

`obs_dist_fn: Callable[[Array], dist.Distribution]`  
Link mapping the full-horizon `latent` to the observation distribution (time at axis `-2`, shape `(*batch, duration, obs)`).

`latent: Array`  
The deterministic latent predictor over the full horizon, time at axis `-2`.


## Raises


`RuntimeError`  
If forecasting (`future > 0`) but no observed data is available.

`ValueError`  
If the observation distribution has discrete support but `h.data` is not integer-dtyped (the usual mistake for count models).
