## predictive.forecast()


Sample forecasts for the steps in `[t, duration)` from a posterior.


Usage

``` python
predictive.forecast(
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


Runs `Predictive` with full-horizon `covariates` and the in-sample `data`: the in-sample latent sites are drawn from `posterior` while the `_future` suffix is drawn from the prior, and the `"forecast"` site is returned. The number of forecast samples equals the leading (sample) axis of `posterior` (see [draw_posterior()](predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior)).


## Parameters


`rng_key: Array`  
PRNG key.

`model: ForecastModel`  
The forecasting model callable (the same one that produced `posterior`).

`posterior: Mapping[str, ArrayLike]`  
Posterior samples of the latent sites, sample axis leading. The output of a `device="host"` stage (CPU-committed jax leaves or NumPy leaves) is accepted directly.

`data: Array`  
Observed data with time at axis `-2` and length `t`.

`covariates: Array`  
Covariates with time at axis `-2` and length `duration > t`.

`batch_size: int | None = None`  
Optional chunk size for sampling (caps peak memory).

`parallel: bool = ``True`  
Whether `Predictive` vectorizes over the sample axis with `vmap` (`True`, faster, higher peak memory) or maps it serially with `lax.map` (`False`). With `parallel=True` the samples in each `batch_size` chunk are vectorized while the chunks are looped over, so `batch_size` remains the peak-memory governor. The two settings produce the same draws up to floating-point reduction order.

`device: jax.Device | str | None = None`  
Where each chunk of draws is placed as soon as it is drawn and where the stitched result lives; the same placement contract as the `device` argument of [draw_posterior()](predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior) (`"host"` for pageable host memory as a CPU-committed `jax.Array`, or a NumPy array when no CPU backend is initialized; `"numpy"`; `"pinned_host"`; a `jax.Device` or platform name; `None` for the default device), including its mixing rules for host-committed results. With `batch_size` set on an accelerator, any host target bounds accelerator memory by a single chunk instead of the full `(sample, future, obs)` array; the draw values are unchanged, only where the result lives. The bound requires `batch_size` strictly below the sample count: at or above it, the single-shot path runs and the full array is materialized on the default device before the one transfer. The result feeds straight into [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) and the `batch_size`-chunked evaluation metrics in `evaluate`, which accept host-resident draws.


## Returns


`Num[Array, ``" sample *batch future obs"]`  
Forecast samples over the `future = duration - t` horizon (floating point for continuous observations, integer for discrete/count models built with [predict()](models.predict.md#numpyro_forecast.models.predict); with `device="host"` committed to the CPU device, or a NumPy array when no CPU backend is initialized).


## Raises


`ValueError`  
If `covariates` does not extend beyond `data` along the time axis.

`HostMemoryKindError`  
If `device="pinned_host"` is requested on a device that exposes no host memory kind (see `_host_memory_kind()`).

`DevicePlatformError`  
If `device` names a platform whose backend is not initialized (see `_resolve_device()`).


## Warns


`UserWarning`  
If `device="cpu"` is requested and the JAX CPU backend is not initialized, so the draws take the NumPy path of `"host"` instead.


## Notes

Chunking is a memory knob, not a reproducibility knob: reproducibility is per `(rng_key, batch_size)`. Every chunk shares the exact `batch_size` shape (the final chunk wraps around to re-used draws that are discarded), so the underlying `_predict` compiles exactly once for a fixed shape, but changing `batch_size` changes the PRNG stream layout and therefore the exact draws. `device` never changes the draws, only where they live.
