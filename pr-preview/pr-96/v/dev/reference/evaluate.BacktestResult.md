## evaluate.BacktestResult


Per-window result of a [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) run.


Usage

``` python
evaluate.BacktestResult(
    t0,
    t1,
    t2,
    num_samples,
    walltime,
    metrics,
    train_metrics=dict(),
    prediction=None
)
```


## Attributes


`t0, t1, t2`  
Train-begin, train/test split, and test-end time indices.

`num_samples: int`  
Number of forecast samples drawn.

`walltime: float`  
Wall-clock seconds for the window's timed `forecast_fn` call.

`metrics: dict[str, float]`  
Mapping of metric name to value for the window.

`train_metrics: dict[str, float]`  
Mapping of metric name to in-sample value for the window. Empty unless [backtest](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) was called with `eval_train=True`.

`prediction: Array | np.ndarray | None`  
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
