## functional.svi.SVIFit


The result of fitting a forecasting model with SVI.


Usage

``` python
functional.svi.SVIFit(
    guide,
    params,
    losses,
    data=None,
    covariates=None,
)
```


## Parameter Attributes


`guide: AutoGuide | Callable[…, None]`  

`params: dict[str, Array]`  

`losses: Array`  

`data: Array | None = None`  

`covariates: Array | None = None`  


## Attributes


`guide: AutoGuide | Callable[…, None]`  
The fitted variational guide.

`params: dict[str, Array]`  
The learned variational parameters.

`losses: Array`  
The ELBO loss per SVI step (shape `(num_steps,)`).

`data: Array | None`  
The in-sample data the model was fit on (needed to draw from hand-written guides and by `~numpyro_forecast.convert.to_datatree()`). `None` for fits constructed without it.

`covariates: Array | None`  
The in-sample covariates the model was fit on. `None` for fits constructed without it.
