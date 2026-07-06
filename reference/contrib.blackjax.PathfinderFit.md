## contrib.blackjax.PathfinderFit


The result of fitting a forecasting model with BlackJAX Pathfinder.


Usage

``` python
contrib.blackjax.PathfinderFit(
    state,
    model,
    covariates,
    data,
    elbo,
)
```


A plain-data (picklable) container: it holds the raw blackjax `PathfinderState`, the model and its data/covariates (needed to rebuild the unconstrained-to-constrained transform when drawing), and the fitted ELBO. Draws are produced lazily by `~numpyro_forecast.functional.draw_posterior()`.


## Parameter Attributes


`state: Any`  

`model: ForecastModel`  

`covariates: Array`  

`data: Array`  

`elbo: float`  


## Attributes


`state: Any`  
The blackjax `PathfinderState` (the variational approximation).

`model: ForecastModel`  
The forecasting model that was fit.

`covariates: Array`  
In-sample covariates used at fit time (time at axis `-2`).

`data: Array`  
In-sample data used at fit time (time at axis `-2`).

`elbo: float`  
The evidence lower bound of the fitted approximation.
