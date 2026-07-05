## forecaster.PathfinderForecaster


Fit a forecasting model with BlackJAX Pathfinder variational inference.


Usage

``` python
forecaster.PathfinderForecaster()
```


A thin shim over [numpyro_forecast.contrib.blackjax.fit_pathfinder()](contrib.blackjax.fit_pathfinder.md#numpyro_forecast.contrib.blackjax.fit_pathfinder). BlackJAX is an optional dependency (`pip install numpyro_forecast[blackjax]`) imported lazily here, so constructing this class is the opt-in that pulls it in; importing `numpyro_forecast` never does (invariant I8). The constructor mirrors [Forecaster](forecaster.Forecaster.md#numpyro_forecast.forecaster.Forecaster) without `guide`/`optim` (Pathfinder has neither) and adds `num_elbo_samples`/`ftol`.


## Parameters


`rng_key: Array`  
PRNG key for inference.

`model: ForecastModel`  
The forecasting model to fit (OOP instance or functional model).

`data: Array`  
In-sample data with time at axis `-2`.

`covariates: Array`  
Covariates with time at axis `-2` and the same duration as `data`.

`num_elbo_samples: int = ``200`  
Number of Monte Carlo samples used to estimate the ELBO along the L-BFGS path.

`ftol: float = ``1e-05`  
L-BFGS relative function-value tolerance (convergence criterion).
