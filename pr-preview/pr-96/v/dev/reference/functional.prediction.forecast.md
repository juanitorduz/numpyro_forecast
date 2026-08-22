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

`posterior: Mapping[str, ArrayLike]`  
Posterior samples of the latent sites, sample axis leading. NumPy leaves are accepted directly (e.g. host-offloaded draws).

`data: Array`  
Observed data with time at axis `-2` and length `t`.

`covariates: Array`  
Covariates with time at axis `-2` and length `duration > t`.

`batch_size: int | None = None`  
Optional chunk size for sampling (caps peak memory).

`parallel: bool = ``True`  
Whether `Predictive` vectorizes over the sample axis with `vmap` (`True`, faster, higher peak memory) or maps it serially with `lax.map` (`False`). With `parallel=True` the samples in each `batch_size` chunk are vectorized while the chunks are looped over, so `batch_size` remains the peak-memory governor. The two settings produce the same draws up to floating-point reduction order.

`device: jax.Device | str | None = None`  
Where each chunk of draws is placed as soon as it is drawn and where the stitched result lives. `"host"` commits every chunk to host memory and returns a `jax.Array` whose sharding carries a host memory kind (`"pinned_host"` where the backend offers it), so nothing of the result occupies accelerator memory; call `numpy.asarray()` on it for a plain NumPy copy. It needs no CPU backend, so it works even when `numpyro.set_platform("cuda")` (or `jax_platforms`) leaves only an accelerator backend initialized, which makes it the recommended choice on GPU. A `jax.Device` or platform name like `"cpu"` commits the draws to that device instead (`"cpu"` falls back to `"host"` with a `UserWarning` when the CPU backend is not initialized). With `batch_size` set on an accelerator, either bounds accelerator memory by a single chunk instead of the full `(sample, future, obs)` array; the draw values are unchanged, only where the result lives. The bound requires `batch_size` strictly below the sample count: at or above it, the single-shot path runs and the full array is materialized on the default device before the one transfer. `None` keeps everything on the default device. Arithmetic that mixes a host-committed result with a device-resident array raises in JAX rather than running on the accelerator, so feed it straight into `~numpyro_forecast.convert.to_datatree()` or the `batch_size`-chunked evaluation metrics in `~numpyro_forecast.evaluate` (both already accept it), or convert explicitly first with `np.asarray(x)` (stays on host) or `jax.device_put(x, device)` (moves it to an accelerator).


## Returns


`Num[Array, ``" sample *batch future obs"]`  
Forecast samples over the `future = duration - t` horizon (floating point for continuous observations, integer for discrete/count models built with `~numpyro_forecast.functional.models.predict_glm()`; committed to host memory when `device` resolves to `"host"`).


## Raises


`ValueError`  
If `covariates` does not extend beyond `data` along the time axis.

`RuntimeError`  
If `device` resolves to `"host"` and the array's device exposes no host memory kind (see `~numpyro_forecast.functional._offload._host_memory_kind()`).


## Notes

Chunking is a memory knob, not a reproducibility knob: reproducibility is per `(rng_key, batch_size)`. Every chunk shares the exact `batch_size` shape (the final chunk wraps around to re-used draws that are discarded), so the underlying `_predict` compiles exactly once for a fixed shape, but changing `batch_size` changes the PRNG stream layout and therefore the exact draws. `device` never changes the draws, only where they live.
