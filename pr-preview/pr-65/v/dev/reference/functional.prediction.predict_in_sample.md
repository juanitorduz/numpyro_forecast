## functional.prediction.predict_in_sample()


Sample the in-sample posterior predictive of the `obs` site.


Usage

``` python
functional.prediction.predict_in_sample(
    rng_key,
    model,
    posterior,
    covariates,
    *,
    batch_size=None,
    parallel=True,
    device=None
)
```


Runs `Predictive` with the in-sample `covariates` and the supplied posterior latent draws. Unlike [forecast()](functional.prediction.forecast.md#numpyro_forecast.functional.prediction.forecast) there is no forecast horizon: `covariates` span only the observed window, so the model's `obs` site is sampled at every step. The number of predictive samples equals the leading (sample) axis of `posterior` (see `~numpyro_forecast.functional.posterior.draw_posterior()`).


## Parameters


`rng_key: Array`  
PRNG key.

`model: ForecastModel`  
The forecasting model callable (the same one that produced `posterior`).

`posterior: dict[str, Array]`  
Posterior samples of the latent sites, sample axis leading.

`covariates: Array`  
Covariates with time at axis `-2` spanning the observed window. Its time length must match the data the `posterior` was fit on, since the in-sample latent sites are sized to that window.

`batch_size: int | None = None`  
Optional chunk size for sampling (caps peak memory).

`parallel: bool = ``True`  
Whether `Predictive` vectorizes over the sample axis with `vmap` (`True`, faster, higher peak memory) or maps it serially with `lax.map` (`False`). See [forecast()](functional.prediction.forecast.md#numpyro_forecast.functional.prediction.forecast) for how this interacts with `batch_size`.

`device: jax.Device | str | None = None`  
Optional device (or platform name like `"cpu"`) where each chunk of draws is placed as soon as it is drawn and where the stitched result lives. With `batch_size` set on an accelerator, `"cpu"` bounds accelerator memory by a single chunk instead of the full `(sample, time, obs)` array; the draw values are unchanged, only the placement of the result. The bound requires `batch_size` strictly below the sample count: at or above it, the single-shot path runs and the full array is materialized on the default device before the one transfer. `None` keeps everything on the default device.


## Returns


`Num[Array, ``" sample *batch time obs"]`  
In-sample posterior-predictive draws of the `obs` site.
