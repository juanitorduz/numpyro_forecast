## exceptions.VectorizedGuideError


The vectorized backtest requires an `AutoGuide`.


Usage

``` python
exceptions.VectorizedGuideError(message=None)
```


Raised by `~numpyro_forecast.evaluate.backtest_vectorized()` when the resolved guide is hand-written: those are not vmappable, use `~numpyro_forecast.evaluate.backtest()` instead.
