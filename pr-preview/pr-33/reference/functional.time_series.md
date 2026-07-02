## functional.time_series()


Sample a time-varying latent over the full horizon.


Usage

``` python
functional.time_series(
    h,
    name,
    dist_fn,
    *,
    reparam=None,
)
```


The in-sample portion is sampled under `plate("time", t)` with the fixed site `name`; when forecasting, the horizon portion is sampled under a separate site `f"{name}_future"` and concatenated. The separate site keeps the guide shape fixed and lets `Predictive` draw the forecast suffix from the prior.


## Parameters


`h: Horizon`  
The horizon for the current model call (see [Horizon](functional.Horizon.md#numpyro_forecast.functional.Horizon)).

`name: str`  
Base sample-site name for the in-sample latent.

`dist_fn: Callable[[], dist.Distribution]`  
Zero-argument callable returning the per-step prior distribution.

`reparam: Reparam | None = None`  
Optional reparameterization (e.g. `LocScaleReparam`) applied to both the in-sample and forecast sites.


## Returns


`Array`  
The latent over the full horizon with time at axis `-2`.
