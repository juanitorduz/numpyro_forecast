## predictive.predict_in_sample()


Sample the in-sample posterior predictive of the `obs` site.


Usage

``` python
predictive.predict_in_sample(
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


Runs `Predictive` with the in-sample `covariates` and the supplied posterior latent draws. Unlike [forecast()](predictive.forecast.md#numpyro_forecast.predictive.forecast) there is no forecast horizon: `covariates` span only the observed window, so the model's `obs` site is sampled at every step. The number of predictive samples equals the leading (sample) axis of `posterior` (see [draw_posterior()](predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior)).


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
Whether `Predictive` vectorizes over the sample axis with `vmap` (`True`, faster, higher peak memory) or maps it serially with `lax.map` (`False`). See [forecast()](predictive.forecast.md#numpyro_forecast.predictive.forecast) for how this interacts with `batch_size`.

`device: jax.Device | str | None = None`  
Where each chunk of draws is placed as soon as it is drawn and where the stitched result lives; the same placement contract as the `device` argument of [draw_posterior()](predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior) (`"host"` for pageable host memory as a CPU-committed `jax.Array`, or a NumPy array when no CPU backend is initialized; `"numpy"`; `"pinned_host"`; a `jax.Device` or platform name; `None` for the default device), including its mixing rules for host-committed results. With `batch_size` set on an accelerator, any host target bounds accelerator memory by a single chunk instead of the full `(sample, time, obs)` array; the draw values are unchanged, only where the result lives. The bound requires `batch_size` strictly below the sample count: at or above it, the single-shot path runs and the full array is materialized on the default device before the one transfer. The result feeds straight into [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) and the `batch_size`-chunked evaluation metrics in `evaluate`, which accept host-resident draws.


## Returns


`Num[Array, ``" sample *batch time obs"]`  
In-sample posterior-predictive draws of the `obs` site (with `device="host"` committed to the CPU device, or a NumPy array when no CPU backend is initialized).


## Raises


`HostMemoryKindError`  
If `device="pinned_host"` is requested on a device that exposes no host memory kind (see `_host_memory_kind()`).

`DevicePlatformError`  
If `device` names a platform whose backend is not initialized (see `_resolve_device()`).


## Warns


`UserWarning`  
If `device="cpu"` is requested and the JAX CPU backend is not initialized, so the draws take the NumPy path of `"host"` instead.
