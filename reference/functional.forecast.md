## functional.forecast()


Sample forecasts for the steps in `[t, duration)` from a posterior.


Usage

``` python
functional.forecast(
    rng_key,
    model,
    posterior,
    data,
    covariates,
    *,
    batch_size=None,
    parallel=True
)
```


Runs `Predictive` with full-horizon `covariates` and the in-sample `data`: the in-sample latent sites are drawn from `posterior` while the `_future` suffix is drawn from the prior, and the `"forecast"` site is returned. The number of forecast samples equals the leading (sample) axis of `posterior` (see [draw_posterior()](functional.draw_posterior.md#numpyro_forecast.functional.draw_posterior)).


## Parameters


`rng_key: Array`  
PRNG key.

`model: ForecastModel`  
The forecasting model callable (the same one that produced `posterior`).

`posterior: dict[str, Array]`  
Posterior samples of the latent sites, sample axis leading.

`data: Array`  
Observed data with time at axis `-2` and length `t`.

`covariates: Array`  
Covariates with time at axis `-2` and length `duration > t`.

`batch_size: int | None = None`  
Optional chunk size for sampling (caps peak memory).

`parallel: bool = ``True`  
Whether `Predictive` vectorizes over the sample axis with `vmap` (`True`, faster, higher peak memory) or maps it serially with `lax.map` (`False`). With `parallel=True` the samples in each `batch_size` chunk are vectorized while the chunks are looped over, so `batch_size` remains the peak-memory governor. The two settings produce the same draws up to floating-point reduction order.


## Returns


`Num[Array, ``" sample *batch future obs"]`  
Forecast samples over the `future = duration - t` horizon (floating point for continuous observations, integer for discrete/count models built with [predict_glm()](functional.predict_glm.md#numpyro_forecast.functional.predict_glm)).


## Raises


`ValueError`  
If `covariates` does not extend beyond `data` along the time axis.


## Notes

Chunking is a memory knob, not a reproducibility knob: reproducibility is per `(rng_key, batch_size)`. Each chunk is padded to a whole multiple of `batch_size` so the underlying `_predict` compiles exactly once for a fixed shape (the pad draws are discarded), but changing `batch_size` changes the PRNG stream layout and therefore the exact draws.
