## functional.SVIFit


The result of fitting a forecasting model with SVI.


Usage

``` python
functional.SVIFit(
    guide,
    params,
    losses,
)
```


## Parameter Attributes


`guide: AutoGuide`  

`params: dict[str, Array]`  

`losses: Array`  


## Attributes


`guide: AutoGuide`  
The fitted variational guide.

`params: dict[str, Array]`  
The learned variational parameters.

`losses: Array`  
The ELBO loss per SVI step (shape `(num_steps,)`).
