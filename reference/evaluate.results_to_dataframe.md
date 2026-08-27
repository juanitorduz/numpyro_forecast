## evaluate.results_to_dataframe()


Flatten backtest results into a tidy one-row-per-window `DataFrame`.


Usage

``` python
evaluate.results_to_dataframe(results)
```


Columns are prefix-namespaced so metric and in-sample-metric names never collide: window metrics become `metric_<name>` and in-sample metrics `train_metric_<name>`, alongside the window indices `t0`/`t1`/`t2`, `num_samples`, and `walltime`. Forecast samples are excluded. Windows may carry different metric sets (e.g. via `backtest(per_window_metrics=...)`); the union of columns is used and missing entries are left as `NaN`.

A [VectorizedBacktestResult](evaluate.VectorizedBacktestResult.md#numpyro_forecast.evaluate.VectorizedBacktestResult) from [backtest_vectorized()](evaluate.backtest_vectorized.md#numpyro_forecast.evaluate.backtest_vectorized) is also accepted; it produces the same `metric_<name>` columns for the same metric set, but has no `train_metric_*` or `walltime` columns (a vectorized run has no per-window walltimes), which are simply absent.


## Parameters


`results: Sequence[BacktestResult] | VectorizedBacktestResult`  
A sequence of [BacktestResult](evaluate.BacktestResult.md#numpyro_forecast.evaluate.BacktestResult) from [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest), or a single [VectorizedBacktestResult](evaluate.VectorizedBacktestResult.md#numpyro_forecast.evaluate.VectorizedBacktestResult) from [backtest_vectorized()](evaluate.backtest_vectorized.md#numpyro_forecast.evaluate.backtest_vectorized).


## Returns


`pandas.DataFrame`  
One row per window with the namespaced columns described above.


## Raises


`ImportError`  
If `pandas` is not installed (`pip install numpyro_forecast[dataframes]`).
