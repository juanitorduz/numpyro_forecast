## typing.ForecastModel


A NumPyro forecasting model: a callable `(covariates, data=None) -> None`.


Usage

``` python
typing.ForecastModel()
```


Any plain function with this signature satisfies this Protocol structurally (for example, one that derives its `~numpyro_forecast.models.Horizon` from the shapes via [Horizon.from_data](models.Horizon.md#numpyro_forecast.models.Horizon.from_data) and calls the model building blocks), so nothing needs to subclass it. The parameters are positional-only so a user model's own parameter names (`cov`, `y`, …) stay free instead of being forced to match `covariates`/`data`.

`ty` checks call sites against this signature structurally (duck typing), which is the main payoff of the Protocol over a bare `Callable` alias. At runtime, the beartype import hook's `isinstance` check on a `runtime_checkable` Protocol only verifies that the named methods exist (Python runtime protocols never inspect signatures), so it reduces to `callable(model)`: a model missing the `data=None` default still passes this check and only fails loudly at the first driver call that invokes it with `data=None`.


## Methods

| Name | Description |
|----|----|
| [__call__()](#__call__) | Run the forecasting model against `covariates` and optional `data`. |

------------------------------------------------------------------------


#### \_\_call\_\_()


Run the forecasting model against `covariates` and optional `data`.


Usage

``` python
__call__(covariates, data=None)
```
