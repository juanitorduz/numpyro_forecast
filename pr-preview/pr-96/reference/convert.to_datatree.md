## convert.to_datatree()


Convert an already-drawn posterior into an ArviZ-schema `xarray.DataTree`.


Usage

``` python
convert.to_datatree(
    rng_key,
    model,
    posterior,
    data,
    covariates,
    *,
    num_chains=1,
    predictive_batch_size=None,
    predictive_device="host",
    coords=None,
    time_coord=None,
    posterior_dims=None,
    covariate_dims=None
)
```


Posterior-first: callers draw their own posterior (`mcmc.get_samples()` for MCMC, `~numpyro_forecast.functional.posterior.draw_posterior()` for a variational fit) and pass it in; [to_datatree](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) never draws a posterior of its own. `rng_key` is consumed only by the in-sample posterior-predictive draws and, when a forecast horizon is present, the forecast draws.


## Parameters


`rng_key: Array`  
PRNG key for the in-sample predictive draws and, when a horizon is present, the forecast draws.

`model: ForecastModel`  
The forecasting model that produced `posterior`.

`posterior: Mapping[str, ArrayLike]`  
Posterior samples of the latent sites, with a single flattened sample axis leading (NumPyro's `mcmc.get_samples()` order, or the output of `~numpyro_forecast.functional.posterior.draw_posterior()`). NumPy leaves are accepted directly (e.g. host-offloaded draws).

`data: Array`  
In-sample data with time at axis `-2`.

`covariates: Array`  
Covariates with time at axis `-2`. When `covariates` extends beyond `data` along the time axis (the package-wide shape convention for a forecast horizon), the trailing rows are treated as future covariates: the returned tree additionally carries `predictions` (forecast `obs` draws from `~numpyro_forecast.functional.prediction.forecast()`) and `predictions_constant_data` groups.

`num_chains: int = ``1`  
Number of chains to split `posterior`'s flattened sample axis into (and, identically, the in-sample/forecast predictive draws, which are drawn with the same sample count). Defaults to `1` (a single pseudo-chain, correct for a posterior with no chain structure, e.g. SVI or Pathfinder draws). For an MCMC posterior, pass the `num_chains` the sampler was run with; see `_reshape_chains()` for the reshape contract and its divisibility requirement.

`predictive_batch_size: int | None = None`  
Optional chunk size that bounds how many draws touch the accelerator at once, across both the in-sample and forecast predictive sampling. When set, sampling runs in chunks of this many draws, each chunk moved to `predictive_device` before the next is drawn. The per-chunk accelerator footprint is a handful of `(batch_size, time, series)` buffers, so it scales linearly with this value times the panel width: on wide panels lower it until a chunk fits. The batch size must be strictly below the draw count for that bound to hold: at or above it, sampling falls back to the single-shot path and the full array is materialized on the default device before the single transfer. Chunking changes the PRNG stream layout of the predictive draws, so results are reproducible per `(rng_key, predictive_batch_size)`. `None` (default) samples everything in one shot (the results are still moved to `predictive_device`).

`predictive_device: jax.Device | str | None = ``"host"`  
Where the predictive draws are moved as they are sampled, forwarded to the `device` argument of `~numpyro_forecast.functional.prediction.predict_in_sample()` and `~numpyro_forecast.functional.prediction.forecast()`. The default `"host"` commits every chunk to host memory, returning `jax.Array` values whose sharding carries a host memory kind (`"pinned_host"` where the backend offers it, the form the tree is built from anyway); it is what bounds accelerator memory when `predictive_batch_size` is set, and it needs no CPU backend, so it works even when `numpyro.set_platform("cuda")` (or `jax_platforms`) leaves only an accelerator backend initialized. A `jax.Device` or platform name like `"cpu"` commits the draws to that device instead; pass `None` to keep the draws on the default device (chunked compute without per-chunk host transfers, for when the draws fit on the accelerator and transfers would dominate runtime). Arithmetic that mixes a host-committed result with a device-resident array raises in JAX rather than running on the accelerator; convert such a result explicitly first with `np.asarray(x)` (stays on host) or `jax.device_put(x, device)` (moves it to an accelerator) before doing your own array math on it.

`coords: Mapping[str, Sequence[Any]] | None = None`  
Optional extra coordinates; these take precedence over the generated `time` coordinate. They also propagate to the forecast groups, where the generated forecast `time` takes precedence instead (a user `time` entry covers the in-sample window; use `time_coord` for explicit forecast time values).

`time_coord: Sequence[Any] | None = None`  
Optional explicit time coordinate values. Without a forecast horizon it covers the in-sample window (defaults to `range(n_time)`); with a horizon it must cover the full `covariates` length and is split into the in-sample and forecast time coordinates (the default is the integer continuation).

`posterior_dims: Mapping[str, Sequence[str]] | None = None`  
Optional mapping from a posterior site name to its non-sample dimension names, e.g. `{"drift": ["time"]}`. Sites listed here share the tree-wide `time` coordinate; unlisted sites keep ArviZ's auto-named dims. This is an explicit opt-in on purpose: inferring time-indexed sites from trace shapes is fragile (a coincidental `n_params == n_time` would misattribute the axis).

`covariate_dims: Sequence[str] | None = None`  
Optional dimension names for the stored covariates, one per axis; defaults to the 2-D `("time", "covariate_dim")` layout. Use this when `covariates` carries extra batch axes, e.g. a panel tensor shaped `(channel, time, series)` with `covariate_dims=["channel", "time", "series"]`. The time axis is always `-2` (the package-wide convention), so its entry should be named `"time"` to share the tree-wide time coordinate.


## Returns


`xarray.DataTree`  
A tree with `posterior` (`(chain, draw, ...)`, split per `num_chains`), `posterior_predictive` (in-sample `obs`), `observed_data`, and `constant_data` groups. When `covariates` extends beyond `data`, also `predictions` and `predictions_constant_data` groups (sharing the same `num_chains` split).


## Raises


`ValueError`  
If `covariates` is shorter than `data` along the time axis, if `time_coord` is given but its length does not match the in-sample window plus the forecast horizon, or if `posterior`'s sample count is not evenly divisible by `num_chains`.

`CovariateDimsError`  
If `covariate_dims` does not name every `covariates` axis.

`RuntimeError`  
If `predictive_device` resolves to `"host"` and the array's device exposes no host memory kind (see `~numpyro_forecast.functional._offload._host_memory_kind()`).


## Notes

[to_datatree](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) no longer accepts a fit object or draws a posterior itself (no `num_predictive_samples`, no internal `~numpyro_forecast.functional.posterior.draw_posterior()` call): callers draw the posterior first and pass it in. The `variational`/`is_mcmc` attrs previously stamped on the `posterior` group are gone too, since a fit type is no longer knowable from a plain posterior dict; use `num_chains` (`1` vs. `> 1`) to tell the two apart if needed. When a forecast horizon is present, `rng_key` is split internally into a predictive subkey and a forecast subkey, so passing the same key twice never correlates the two sample sets. When there is no horizon, `rng_key` is used unsplit for the in-sample predictive draw. `predictive_batch_size` is the built-in route to memory-bounded predictive sampling; for fully manual control over the forecast draws, build the in-sample tree with matching-length covariates and attach the horizon with [add_forecast_groups()](convert.add_forecast_groups.md#numpyro_forecast.convert.add_forecast_groups).
