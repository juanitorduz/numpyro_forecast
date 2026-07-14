## convert.to_datatree()


Convert a fit into an ArviZ-schema `xarray.DataTree`.


Usage

``` python
convert.to_datatree(
    rng_key,
    fit,
    model,
    data,
    covariates,
    *,
    num_predictive_samples=None,
    predictive_batch_size=None,
    predictive_device="host",
    coords=None,
    time_coord=None,
    posterior_dims=None,
    covariate_dims=None
)
```


PRNG: `rng_key` is consumed by the in-sample posterior-predictive draws (for a variational fit, also the posterior draws) and, when a forecast horizon is present, the forecast draws.


## Parameters


`rng_key: Array`  
PRNG key for the predictive (and variational posterior) draws.

`fit: object`  
A fit from `~numpyro_forecast.functional.mcmc.fit_mcmc()`, `~numpyro_forecast.functional.svi.fit_svi()`, or `~numpyro_forecast.contrib.blackjax.fit_pathfinder()`.

`model: ForecastModel`  
The forecasting model that produced `fit`.

`data: Array`  
In-sample data with time at axis `-2`.

`covariates: Array`  
Covariates with time at axis `-2`. When `covariates` extends beyond `data` along the time axis (the package-wide shape convention for a forecast horizon), the trailing rows are treated as future covariates: the returned tree additionally carries `predictions` (forecast `obs` draws from `~numpyro_forecast.functional.prediction.forecast()`) and `predictions_constant_data` groups.

`num_predictive_samples: int | None = None`  
Number of posterior draws for a variational fit (ignored for `~numpyro_forecast.functional.mcmc.MCMCFit`, which uses its own draws). The same draws drive the in-sample predictive and the forecast. Defaults to `1_000`.

`predictive_batch_size: int | None = None`  
Optional chunk size for the predictive sampling. When set, the in-sample posterior predictive and the forecast draws are sampled in chunks of this many posterior draws, and each chunk is moved to `predictive_device` before the next is drawn, so accelerator memory is bounded by one chunk instead of the full `(samples, time, obs)` array. The batch size must be strictly below the draw count for that bound to hold: at or above it, sampling falls back to the single-shot path and the full array is materialized on the default device before the single transfer. Chunking changes the PRNG stream layout, so draws are reproducible per `(rng_key, predictive_batch_size)`. `None` (default) samples everything in one shot (the result is still moved to `predictive_device`).

`predictive_device: jax.Device | str | None = ``"host"`  
Where the predictive draws are moved as they are sampled, forwarded to the `device` argument of `~numpyro_forecast.functional.prediction.predict_in_sample()` and `~numpyro_forecast.functional.prediction.forecast()`. The default `"host"` copies every chunk to host memory as a NumPy array (where the tree is built anyway); it is what bounds accelerator memory when `predictive_batch_size` is set, and it needs no CPU backend, so it works even when `numpyro.set_platform("cuda")` (or `jax_platforms`) leaves only an accelerator backend initialized. A `jax.Device` or platform name like `"cpu"` commits the draws to that device instead; pass `None` to keep the draws on the default device (chunked compute without per-chunk host transfers, for when the draws fit on the accelerator and transfers would dominate runtime).

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
A tree with `posterior` (`(chain, draw, ...)`; a single pseudo-chain plus `variational: True` attrs for SVI/Pathfinder), `posterior_predictive` (in-sample `obs`), `observed_data`, and `constant_data` groups. When `covariates` extends beyond `data`, also `predictions` and `predictions_constant_data` groups (the forecast keeps an MCMC fit's real chain structure).


## Raises


`ValueError`  
If `covariates` is shorter than `data` along the time axis, or if `time_coord` is given but its length does not match the in-sample window plus the forecast horizon.

`CovariateDimsError`  
If `covariate_dims` does not name every `covariates` axis.


## Notes

`rng_key` is split internally: one subkey drives the posterior draws (for variational fits), one the in-sample predictive, and, when a horizon is present, a third the forecast. The split is a deterministic derivation applied for every fit type, so passing the same key twice never correlates the sample sets. `predictive_batch_size` is the built-in route to memory-bounded predictive sampling; for fully manual control over the forecast draws, build the in-sample tree with matching-length covariates and attach the horizon with [add_forecast_groups()](convert.add_forecast_groups.md#numpyro_forecast.convert.add_forecast_groups).
