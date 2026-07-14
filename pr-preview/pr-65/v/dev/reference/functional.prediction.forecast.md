## functional.prediction.forecast()


Sample forecasts for the steps in `[t, duration)` from a posterior.


Usage

``` python
functional.prediction.forecast(
    rng_key,
    model,
    posterior,
    data,
    covariates,
    *,
    batch_size=None,
    parallel=True,
    device=None
)
```


Runs `Predictive` with full-horizon `covariates` and the in-sample `data`: the in-sample latent sites are drawn from `posterior` while the `_future` suffix is drawn from the prior, and the `"forecast"` site is returned. The number of forecast samples equals the leading (sample) axis of `posterior` (see `~numpyro_forecast.functional.posterior.draw_posterior()`).


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

`device: jax.Device | str | None = None`  
Optional device (or platform name like `"cpu"`) where each chunk of draws is placed as soon as it is drawn and where the stitched result lives. With `batch_size` set on an accelerator, `"cpu"` bounds accelerator memory by a single chunk instead of the full `(sample, future, obs)` array; the draw values are unchanged, only the placement of the result. `None` keeps everything on the default device.


## Returns


`Num[Array, ``" sample *batch future obs"]`  
Forecast samples over the `future = duration - t` horizon (floating point for continuous observations, integer for discrete/count models built with `~numpyro_forecast.functional.models.predict_glm()`).


## Raises


`ValueError`  
If `covariates` does not extend beyond `data` along the time axis.


## Notes

Chunking is a memory knob, not a reproducibility knob: reproducibility is per `(rng_key, batch_size)`. Every chunk shares the exact `batch_size` shape (the final chunk wraps around to re-used draws that are discarded), so the underlying `_predict` compiles exactly once for a fixed shape, but changing `batch_size` changes the PRNG stream layout and therefore the exact draws. `device` never changes the draws, only where they live.
