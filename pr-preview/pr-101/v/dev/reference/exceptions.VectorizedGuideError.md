## exceptions.VectorizedGuideError


The vectorized backtest requires an `AutoGuide` instance.


Usage

``` python
exceptions.VectorizedGuideError(message=None)
```


Raised by [backtest_vectorized()](evaluate.backtest_vectorized.md#numpyro_forecast.evaluate.backtest_vectorized) when `guide` is hand-written: those are not vmappable, use [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) instead.
