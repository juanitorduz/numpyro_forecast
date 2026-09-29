## models.innovations()


Sample conditionally iid per-step innovations over the full horizon.


Usage

``` python
models.innovations(
    h,
    name,
    prior,
    *,
    reparam=None,
)
```


The in-sample portion is sampled under `plate("time", t)` with the fixed site `name`; when forecasting, the horizon portion is sampled under a separate site `f"{name}_future"` and concatenated. The separate site keeps the guide shape fixed and lets `Predictive` draw the forecast suffix from the prior. Build the series arithmetically from the result (a random walk is `jnp.cumsum(drift, axis=-2)`); a latent whose per-step distribution depends on the previous state is [markov_series()](models.markov_series.md#numpyro_forecast.models.markov_series), and a deterministic error-feedback recursion driven by the observed series is [ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe).


## Parameters


`h: Horizon`  
The horizon for the current model call (see [Horizon](models.Horizon.md#numpyro_forecast.models.Horizon)).

`name: str`  
Base sample-site name for the in-sample latent.

`prior: dist.Distribution`  
The per-step prior distribution, shared by the in-sample and forecast sites (each time plate expands a copy; the instance is never mutated). Its batch shape is the per-step shape, for example `()` for a scalar latent or `(n_series,)` under an enclosing series plate; the time axis comes from the plate.

`reparam: Reparam | None = None`  
Optional reparameterization (e.g. `LocScaleReparam`) applied to both the in-sample and forecast sites.


## Returns


`Array`  
The latent over the full horizon with time at axis `-2`.
