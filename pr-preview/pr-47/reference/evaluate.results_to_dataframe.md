## evaluate.results_to_dataframe()


Flatten backtest results into a tidy one-row-per-window `DataFrame`.


Usage

``` python
evaluate.results_to_dataframe(results)
```


Columns are prefix-namespaced so metric, in-sample-metric, and parameter names never collide: window metrics become `metric_<name>`, in-sample metrics `train_metric_<name>`, and scalar parameters `param_<name>`, alongside the window indices `t0`/`t1`/`t2`, `num_samples`, and `train_walltime`/`test_walltime`. Forecast samples are excluded. Windows may carry different metric sets (e.g. via `backtest(per_window_metrics=...)`); the union of columns is used and missing entries are left as `NaN`.


## Parameters


`results: Sequence[BacktestResult]`  
A sequence of [BacktestResult](evaluate.BacktestResult.md#numpyro_forecast.evaluate.BacktestResult) from [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest).


## Returns


`pandas.DataFrame`  
One row per window with the namespaced columns described above.


## Raises


`NotImplementedError`  
If a vectorized backtest result is passed; use the loop [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) output for now (vectorized support lands with `backtest_vectorized`).

`ImportError`  
If `pandas` is not installed (`pip install numpyro_forecast[dataframes]`).
