## evaluate.eval_mae()


Mean absolute error using the forecast sample median as point estimate.


Usage

``` python
evaluate.eval_mae(
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
The mean absolute error.
