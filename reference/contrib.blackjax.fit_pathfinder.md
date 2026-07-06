## contrib.blackjax.fit_pathfinder()


Fit a forecasting model with BlackJAX Pathfinder variational inference.


Usage

``` python
contrib.blackjax.fit_pathfinder(
    rng_key, model, data, covariates, *, num_elbo_samples=200, ftol=1e-05
)
```


PRNG: `rng_key` is split into a model-initialization stream and a Pathfinder-approximation stream.


## Parameters


`rng_key: Array`  
PRNG key for initialization and the Pathfinder run.

`model: ForecastModel`  
The forecasting model callable (OOP instance or functional model).

`data: Array`  
In-sample data with time at axis `-2`.

`covariates: Array`  
Covariates with time at axis `-2` and the same duration as `data`.

`num_elbo_samples: int = ``200`  
Number of Monte Carlo samples used to estimate the ELBO along the L-BFGS optimization path.

`ftol: float = ``1e-05`  
L-BFGS relative function-value tolerance (convergence criterion).


## Returns


`PathfinderFit`  
The fitted variational approximation.
