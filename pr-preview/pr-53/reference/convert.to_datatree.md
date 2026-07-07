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
    coords=None,
    time_coord=None,
    posterior_dims=None
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

`coords: Mapping[str, Sequence[Any]] | None = None`  
Optional extra coordinates; these take precedence over the generated `time` coordinate.

`time_coord: Sequence[Any] | None = None`  
Optional explicit time coordinate values. Without a forecast horizon it covers the in-sample window (defaults to `range(n_time)`); with a horizon it must cover the full `covariates` length and is split into the in-sample and forecast time coordinates (the default is the integer continuation).

`posterior_dims: Mapping[str, Sequence[str]] | None = None`  
Optional mapping from a posterior site name to its non-sample dimension names, e.g. `{"drift": ["time"]}`. Sites listed here share the tree-wide `time` coordinate; unlisted sites keep ArviZ's auto-named dims. This is an explicit opt-in on purpose: inferring time-indexed sites from trace shapes is fragile (a coincidental `n_params == n_time` would misattribute the axis).


## Returns


`xarray.DataTree`  
A tree with `posterior` (`(chain, draw, ...)`; a single pseudo-chain plus `variational: True` attrs for SVI/Pathfinder), `posterior_predictive` (in-sample `obs`), `observed_data`, and `constant_data` groups. When `covariates` extends beyond `data`, also `predictions` and `predictions_constant_data` groups (the forecast keeps an MCMC fit's real chain structure).


## Raises


`ValueError`  
If `time_coord` is given but its length does not match the in-sample window plus the forecast horizon.


## Notes

`rng_key` is split internally: one subkey drives the posterior draws (for variational fits), one the in-sample predictive, and, when a horizon is present, a third the forecast. The split is a deterministic derivation applied for every fit type, so passing the same key twice never correlates the sample sets. For step-by-step control over the forecast draws (e.g. a custom `batch_size`), build the in-sample tree with matching-length covariates and attach the horizon with [add_forecast_groups()](convert.add_forecast_groups.md#numpyro_forecast.convert.add_forecast_groups).
