## exceptions.BacktestWindowError


A backtest window configuration is invalid.


Usage

``` python
exceptions.BacktestWindowError(message=None)
```


Raised by `~numpyro_forecast.evaluate.backtest_vectorized()` when `train_window`, `test_window`, or `stride` is below 1, or when the series has no room for a single window.
