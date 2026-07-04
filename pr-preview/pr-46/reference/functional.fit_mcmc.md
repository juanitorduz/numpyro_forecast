## functional.fit_mcmc()


Fit a forecasting model with NUTS (Hamiltonian Monte Carlo).


Usage

``` python
functional.fit_mcmc(
    rng_key,
    model,
    data,
    covariates,
    *,
    num_warmup=1000,
    num_samples=1000,
    num_chains=1,
    progress_bar=False
)
```


## Parameters


`rng_key: Array`  
PRNG key for inference.

`model: ForecastModel`  
The forecasting model callable (OOP instance or functional model).

`data: Array`  
In-sample data with time at axis `-2`.

`covariates: Array`  
Covariates with time at axis `-2` and the same duration as `data`.

`num_warmup: int = ``1000`  
Number of warmup steps.

`num_samples: int = ``1000`  
Number of posterior samples.

`num_chains: int = ``1`  
Number of MCMC chains.

`progress_bar: bool = ``False`  
Whether to display the MCMC progress bar.


## Returns


`MCMCFit`  
The posterior samples.


## Raises


`ValueError`  
If `data` and `covariates` have different durations.
