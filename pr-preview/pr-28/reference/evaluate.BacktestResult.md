## evaluate.BacktestResult


Per-window result of a [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) run.


Usage

``` python
evaluate.BacktestResult()
```


## Parameter Attributes


`t0: int`  

`t1: int`  

`t2: int`  

`num_samples: int`  

`train_walltime: float`  

`test_walltime: float`  

`metrics: dict[str, float]`  

`params: dict[str, float] = dict()`    

`train_metrics: dict[str, float] = dict()`    

`prediction: Array | None = None`  


## Attributes


`t0, t1, t2`  
Train-begin, train/test split, and test-end time indices.

`num_samples: int`  
Number of forecast samples drawn.

`train_walltime, test_walltime`  
Wall-clock seconds for fitting and forecasting.

`metrics: dict[str, float]`  
Mapping of metric name to value for the window.

`params: dict[str, float]`  
Mapping of scalar parameter name to value (when available).

`train_metrics: dict[str, float]`  
Mapping of metric name to in-sample value for the window. Empty unless [backtest](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) was called with `eval_train=True`.

`prediction: Array | None`  
Out-of-sample forecast samples for the window (sample axis leading), or `None` unless [backtest](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) was called with `keep_predictions=True`.


## Methods

| Name | Description |
|----|----|
| [to_dict()](#to_dict) | Return a flat dictionary view (Pyro-style access). |

------------------------------------------------------------------------


#### to_dict()


Return a flat dictionary view (Pyro-style access).


Usage

``` python
to_dict()
```


##### Returns


`dict[str, Any]`  
All fields as a plain dictionary.
