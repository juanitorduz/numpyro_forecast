## exceptions.VectorizedGuideError


The vectorized backtest requires an `AutoGuide` instance.


Usage

``` python
exceptions.VectorizedGuideError(message=None)
```


Raised by `~numpyro_forecast.evaluate.backtest_vectorized()` when `guide` is hand-written: those are not vmappable, use `~numpyro_forecast.evaluate.backtest()` instead.
