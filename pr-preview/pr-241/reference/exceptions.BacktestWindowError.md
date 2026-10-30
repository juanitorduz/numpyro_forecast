## exceptions.BacktestWindowError


A backtest window configuration is invalid.


Usage

``` python
exceptions.BacktestWindowError(message=None)
```


Raised by [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) when `window_type` and `train_window` disagree (`"rolling"` without a `train_window`, or `"expanding"` with one), and by [backtest_vectorized()](evaluate.backtest_vectorized.md#numpyro_forecast.evaluate.backtest_vectorized) when `train_window`, `test_window`, or `stride` is below 1, or when the series has no room for a single window.
