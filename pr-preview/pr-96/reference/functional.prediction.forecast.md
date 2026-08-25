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
Where each chunk of draws is placed as soon as it is drawn and where the stitched result lives. `"host"` keeps the result in pageable host memory, so nothing of it occupies accelerator memory: with the JAX CPU backend initialized it commits every chunk to `jax.devices("cpu")[0]` and returns a committed `jax.Array` (`np.asarray` on it is a zero-copy view); without it (for example after `numpyro.set_platform("cuda")`, or a `JAX_PLATFORMS` preset) it copies each chunk with `jax.device_get()` and returns a NumPy array, since a CUDA client offers no pageable `jax.Array` container. It therefore needs no CPU backend and never pins memory. `"numpy"` forces the NumPy path; `"pinned_host"` commits to the accelerator's pinned host memory kind instead, a pool capped by `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` (64 GB by default on CUDA), so prefer `"host"` for large panels. A `jax.Device` or platform name like `"cpu"` commits the draws to that device (`"cpu"` warns and takes the NumPy path when the CPU backend is missing). With `batch_size` set on an accelerator, any of these bounds accelerator memory by a single chunk instead of the full `(sample, future, obs)` array; the draw values are unchanged, only where the result lives. The bound requires `batch_size` strictly below the sample count: at or above it, the single-shot path runs and the full array is materialized on the default device before the one transfer. `None` keeps everything on the default device. A host-committed jax result is not a drop-in replacement for a device array in your own `jnp` code: mixed with an uncommitted array an op runs on the CPU and returns a CPU-committed array, mixed with an accelerator-committed array it raises, and a pinned array raises on any mix. Feed it straight into `~numpyro_forecast.convert.to_datatree()` or the `batch_size`-chunked evaluation metrics in `~numpyro_forecast.evaluate` (both already accept it), or convert explicitly first with `np.asarray(x)` (stays on host) or `jax.device_put(x, device)` (moves it to an accelerator).


## Returns


`Num[Array, ``" sample *batch future obs"]`  
Forecast samples over the `future = duration - t` horizon (floating point for continuous observations, integer for discrete/count models built with `~numpyro_forecast.functional.models.predict_glm()`; with `device="host"` committed to the CPU device, or a NumPy array when no CPU backend is initialized).


## Raises


`ValueError`  
If `covariates` does not extend beyond `data` along the time axis.

`RuntimeError`  
If `device="pinned_host"` is requested on a device that exposes no host memory kind (see `~numpyro_forecast.functional._offload._host_memory_kind()`).


## Warns


`UserWarning`  
If `device="cpu"` is requested and the JAX CPU backend is not initialized, so the draws take the NumPy path of `"host"` instead.


## Notes

Chunking is a memory knob, not a reproducibility knob: reproducibility is per `(rng_key, batch_size)`. Every chunk shares the exact `batch_size` shape (the final chunk wraps around to re-used draws that are discarded), so the underlying `_predict` compiles exactly once for a fixed shape, but changing `batch_size` changes the PRNG stream layout and therefore the exact draws. `device` never changes the draws, only where they live.
