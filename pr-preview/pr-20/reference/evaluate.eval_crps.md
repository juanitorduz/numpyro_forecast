## evaluate.eval_crps()


Empirical CRPS averaged over all data elements.


Usage

``` python
evaluate.eval_crps(
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
The mean empirical CRPS.
