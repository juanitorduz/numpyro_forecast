## evaluate.VectorizedBacktestResult


Result of a [backtest_vectorized()](evaluate.backtest_vectorized.md#numpyro_forecast.evaluate.backtest_vectorized) run (all windows at once).


Usage

``` python
evaluate.VectorizedBacktestResult()
```


Unlike [BacktestResult](evaluate.BacktestResult.md#numpyro_forecast.evaluate.BacktestResult) this holds every window's values stacked along a leading window axis, because the windows are fitted, drawn, and scored in single vmapped passes rather than one call each. There are no per-window walltimes (a single fused computation covers all windows).


## Parameter Attributes


`t0: Array`  

`t1: Array`  

`t2: Array`  

`num_samples: int`  

`losses: Array`  

`metrics: dict[str, Array]`  

`predictions: Array | None = None`  


## Attributes


`t0, t1, t2`  
Train-begin, train/test split, and test-end time indices, each an integer array of shape `(num_windows,)`.

`num_samples: int`  
Number of forecast samples drawn per window.

`losses: Array`  
SVI loss history with shape `(num_windows, num_steps)`.

`metrics: dict[str, Array]`  
Mapping of metric name to a `(num_windows,)` array of per-window values.

`predictions: Array | None`  
Stacked out-of-sample forecast samples with shape `(num_windows, num_samples, *batch, test_window, obs)`, or `None` unless `keep_predictions=True`.


## Methods

| Name | Description |
|----|----|
| [to_dict()](#to_dict) | Return a flat dictionary view. |

------------------------------------------------------------------------


#### to_dict()


Return a flat dictionary view.


Usage

``` python
to_dict()
```


##### Returns


`dict[str, Any]`  
All fields as a plain dictionary.
