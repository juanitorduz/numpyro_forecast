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
Posterior samples of the latent sites, sample axis leading. NumPy leaves are accepted directly (e.g. host-offloaded draws).

`covariates: Array`  
Covariates with time at axis `-2` spanning the observed window. Its time length must match the data the `posterior` was fit on, since the in-sample latent sites are sized to that window.

`batch_size: int | None = None`  
Optional chunk size for sampling (caps peak memory).

`parallel: bool = ``True`  
Whether `Predictive` vectorizes over the sample axis with `vmap` (`True`, faster, higher peak memory) or maps it serially with `lax.map` (`False`). See [forecast()](functional.prediction.forecast.md#numpyro_forecast.functional.prediction.forecast) for how this interacts with `batch_size`.

`device: jax.Device | str | None = None`  
Where each chunk of draws is placed as soon as it is drawn and where the stitched result lives. `"host"` commits every chunk to host memory and returns a `jax.Array` whose sharding carries a host memory kind (`"pinned_host"` where the backend offers it), so nothing of the result occupies accelerator memory; call `numpy.asarray()` on it for a plain NumPy copy. It needs no CPU backend, so it works even when `numpyro.set_platform("cuda")` (or `jax_platforms`) leaves only an accelerator backend initialized, which makes it the recommended choice on GPU. A `jax.Device` or platform name like `"cpu"` commits the draws to that device instead (`"cpu"` falls back to `"host"` with a `UserWarning` when the CPU backend is not initialized). With `batch_size` set on an accelerator, either bounds accelerator memory by a single chunk instead of the full `(sample, time, obs)` array; the draw values are unchanged, only where the result lives. The bound requires `batch_size` strictly below the sample count: at or above it, the single-shot path runs and the full array is materialized on the default device before the one transfer. `None` keeps everything on the default device. Arithmetic that mixes a host-committed result with a device-resident array raises in JAX rather than running on the accelerator, so feed it straight into `~numpyro_forecast.convert.to_datatree()` or the `batch_size`-chunked evaluation metrics in `~numpyro_forecast.evaluate` (both already accept it), or convert explicitly first with `np.asarray(x)` (stays on host) or `jax.device_put(x, device)` (moves it to an accelerator).


## Returns


`Num[Array, ``" sample *batch time obs"]`  
In-sample posterior-predictive draws of the `obs` site (committed to host memory when `device` resolves to `"host"`).


## Raises


`RuntimeError`  
If `device` resolves to `"host"` and the array's device exposes no host memory kind (see `~numpyro_forecast.functional._offload._host_memory_kind()`).
