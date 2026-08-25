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

`posterior: Mapping[str, ArrayLike]`  
Posterior samples of the latent sites, sample axis leading. The output of a `device="host"` stage (CPU-committed jax leaves or NumPy leaves) is accepted directly.

`covariates: Array`  
Covariates with time at axis `-2` spanning the observed window. Its time length must match the data the `posterior` was fit on, since the in-sample latent sites are sized to that window.

`batch_size: int | None = None`  
Optional chunk size for sampling (caps peak memory).

`parallel: bool = ``True`  
Whether `Predictive` vectorizes over the sample axis with `vmap` (`True`, faster, higher peak memory) or maps it serially with `lax.map` (`False`). See [forecast()](functional.prediction.forecast.md#numpyro_forecast.functional.prediction.forecast) for how this interacts with `batch_size`.

`device: jax.Device | str | None = None`  
Where each chunk of draws is placed as soon as it is drawn and where the stitched result lives. `"host"` keeps the result in pageable host memory, so nothing of it occupies accelerator memory: with the JAX CPU backend initialized it commits every chunk to `jax.devices("cpu")[0]` and returns a committed `jax.Array` (`np.asarray` on it is a zero-copy view); without it (for example after `numpyro.set_platform("cuda")`, or a `JAX_PLATFORMS` preset) it copies each chunk with `jax.device_get()` and returns a NumPy array, since a CUDA client offers no pageable `jax.Array` container. It therefore needs no CPU backend and never pins memory. `"numpy"` forces the NumPy path; `"pinned_host"` commits to the accelerator's pinned host memory kind instead, a pool capped by `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` (64 GB by default on CUDA), so prefer `"host"` for large panels. A `jax.Device` or platform name like `"cpu"` commits the draws to that device (`"cpu"` warns and takes the NumPy path when the CPU backend is missing). With `batch_size` set on an accelerator, any of these bounds accelerator memory by a single chunk instead of the full `(sample, time, obs)` array; the draw values are unchanged, only where the result lives. The bound requires `batch_size` strictly below the sample count: at or above it, the single-shot path runs and the full array is materialized on the default device before the one transfer. `None` keeps everything on the default device. A host-committed jax result is not a drop-in replacement for a device array in your own `jnp` code: mixed with an uncommitted array an op runs on the CPU and returns a CPU-committed array, mixed with an accelerator-committed array it raises, and a pinned array raises on any mix. Feed it straight into `~numpyro_forecast.convert.to_datatree()` or the `batch_size`-chunked evaluation metrics in `~numpyro_forecast.evaluate` (both already accept it), or convert explicitly first with `np.asarray(x)` (stays on host) or `jax.device_put(x, device)` (moves it to an accelerator).


## Returns


`Num[Array, ``" sample *batch time obs"]`  
In-sample posterior-predictive draws of the `obs` site (with `device="host"` committed to the CPU device, or a NumPy array when no CPU backend is initialized).


## Raises


`RuntimeError`  
If `device="pinned_host"` is requested on a device that exposes no host memory kind (see `~numpyro_forecast.functional._offload._host_memory_kind()`).


## Warns


`UserWarning`  
If `device="cpu"` is requested and the JAX CPU backend is not initialized, so the draws take the NumPy path of `"host"` instead.
