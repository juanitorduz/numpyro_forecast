## contrib.blackjax.MultiPathfinderFit


The result of fitting a forecasting model with multi-path BlackJAX Pathfinder.


Usage

``` python
contrib.blackjax.MultiPathfinderFit(
    state, model, covariates, data, elbos, log_weights, pareto_k
)
```


A plain-data (picklable) container, mirroring [PathfinderFit](contrib.blackjax.PathfinderFit.md#numpyro_forecast.contrib.blackjax.PathfinderFit): it holds the raw blackjax `MultipathfinderState` (the pooled per-path approximations), the model and its data/covariates, the per-path ELBOs, and the Pareto-smoothed importance sampling (PSIS) weights/diagnostic over the pooled draws. Draws are produced lazily by [multipathfinder_samples()](contrib.blackjax.multipathfinder_samples.md#numpyro_forecast.contrib.blackjax.multipathfinder_samples).


## Attributes


`state: Any`  
The blackjax `MultipathfinderState` (per-path `PathfinderState` objects, the pooled samples, and their log target/approximation densities).

`model: ForecastModel`  
The forecasting model that was fit.

`covariates: Array`  
In-sample covariates used at fit time (time at axis `-2`).

`data: Array`  
In-sample data used at fit time (time at axis `-2`).

`elbos: tuple[float, …]`  
The evidence lower bound of each path's fitted approximation, in path order; converted eagerly with `float()` so pickling this fit never carries JAX tracers.

`log_weights: Array`  
Normalized PSIS log importance weights over the flattened pool of `num_paths * num_elbo_samples` draws (path-major, matching `state.samples`); a fit-time diagnostic, valid only for that exact stored pool. [multipathfinder_samples()](contrib.blackjax.multipathfinder_samples.md#numpyro_forecast.contrib.blackjax.multipathfinder_samples) draws fresh samples and recomputes its own weights, so it never consumes these.

`pareto_k: float`  
The Pareto shape-parameter diagnostic for the fit-time PSIS weights: below `0.5` is reliable, `0.5` to `0.7` is borderline, and above `0.7` indicates the importance weights are unreliable. [multipathfinder_samples()](contrib.blackjax.multipathfinder_samples.md#numpyro_forecast.contrib.blackjax.multipathfinder_samples) reads it to decide whether `resample="auto"` uses PSIS or ELBO-weighted path sampling.
