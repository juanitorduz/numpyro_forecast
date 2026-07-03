## evaluate.eval_rmse()


Root mean squared error using the forecast sample mean as point estimate.


Usage

``` python
evaluate.eval_rmse(
    pred,
    truth,
)
```


## Parameters


`pred: Array`  
Forecast samples with the sample axis first.

`truth: Array`  
Ground-truth values (matching `pred` without the sample axis).


## Returns


`float`  
The root mean squared error.
