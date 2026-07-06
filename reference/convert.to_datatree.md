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


PRNG: `rng_key` is consumed by the in-sample posterior-predictive draws (and, for a variational fit, the posterior draws).


## Parameters


`rng_key: Array`  
PRNG key for the predictive (and variational posterior) draws.

`fit: object`  
A fit from `~numpyro_forecast.functional.fit_mcmc()`, `~numpyro_forecast.functional.fit_svi()`, or `~numpyro_forecast.contrib.blackjax.fit_pathfinder()`.

`model: ForecastModel`  
The forecasting model that produced `fit`.

`data: Array`  
In-sample data with time at axis `-2`.

`covariates: Array`  
In-sample covariates with time at axis `-2`.

`num_predictive_samples: int | None = None`  
Number of posterior draws for a variational fit (ignored for `~numpyro_forecast.functional.MCMCFit`, which uses its own draws). Defaults to `1_000`.

`coords: Mapping[str, Sequence[Any]] | None = None`  
Optional extra coordinates; these take precedence over the generated `time` coordinate.

`time_coord: Sequence[Any] | None = None`  
Optional explicit in-sample time coordinate values; defaults to `range(n_time)`.

`posterior_dims: Mapping[str, Sequence[str]] | None = None`  
Optional mapping from a posterior site name to its non-sample dimension names, e.g. `{"drift": ["time"]}`. Sites listed here share the tree-wide `time` coordinate; unlisted sites keep ArviZ's auto-named dims. This is an explicit opt-in on purpose: inferring time-indexed sites from trace shapes is fragile (a coincidental `n_params == n_time` would misattribute the axis).


## Returns


`xarray.DataTree`  
A tree with `posterior` (`(chain, draw, ...)`; a single pseudo-chain plus `variational: True` attrs for SVI/Pathfinder), `posterior_predictive` (in-sample `obs`), `observed_data`, and `constant_data` groups.


## Notes

`rng_key` is split internally: one subkey drives the posterior draws (for variational fits) and the other the in-sample predictive. The split is a deterministic derivation applied for every fit type, so passing the same key twice never correlates the two sample sets.
