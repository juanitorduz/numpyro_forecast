## typing.Guide


A NumPyro guide for a [ForecastModel](typing.ForecastModel.md#numpyro_forecast.typing.ForecastModel): a callable with the model's signature.


Usage

``` python
typing.Guide()
```


A guide is called with the same `(covariates, data=None)` arguments as the model it approximates, so this Protocol has the same shape as [ForecastModel](typing.ForecastModel.md#numpyro_forecast.typing.ForecastModel). Both an autoguide instance (`numpyro.infer.autoguide.AutoGuide` subclasses are callables) and a hand-written guide function satisfy it structurally; nothing needs to subclass it. [backtest_vectorized()](evaluate.backtest_vectorized.md#numpyro_forecast.evaluate.backtest_vectorized) samples an autoguide through its `sample_posterior` and any other guide through `numpyro.infer.Predictive(guide, params=...)`.

The same runtime caveat as [ForecastModel](typing.ForecastModel.md#numpyro_forecast.typing.ForecastModel) applies: the beartype hook's `isinstance` check on a `runtime_checkable` Protocol reduces to `callable(guide)` (Python never inspects signatures at runtime), while `ty` checks the signature structurally at the call site.


## Methods

| Name | Description |
|----|----|
| [__call__()](#__call__) | Run the guide against `covariates` and optional `data`. |

------------------------------------------------------------------------


#### \_\_call\_\_()


Run the guide against `covariates` and optional `data`.


Usage

``` python
__call__(covariates, data=None)
```
