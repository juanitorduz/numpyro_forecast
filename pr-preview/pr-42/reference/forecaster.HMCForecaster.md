## forecaster.HMCForecaster


Fit a forecasting model with NUTS (Hamiltonian Monte Carlo).


Usage

``` python
forecaster.HMCForecaster()
```


## Parameters


`rng_key: Array`  
PRNG key for inference.

`model: ForecastModel`  
The forecasting model to fit (OOP instance or functional model).

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
