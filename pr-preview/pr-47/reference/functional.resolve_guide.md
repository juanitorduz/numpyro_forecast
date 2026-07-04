## functional.resolve_guide()


Normalize a guide specification against [model](forecaster.ForecastingModel.md#numpyro_forecast.forecaster.ForecastingModel.model).


Usage

``` python
functional.resolve_guide(
    guide,
    model,
)
```


Resolution: `None` -\> `AutoNormal(model)`; an `AutoGuide` instance -\> returned unchanged; an `AutoGuide` subclass or a `functools.partial` of one -\> called with [model](forecaster.ForecastingModel.md#numpyro_forecast.forecaster.ForecastingModel.model); any other callable -\> a hand-written guide, after `_probe_handwritten_guide()`. Anything else -\> `TypeError`.


## Parameters


`guide: GuideLike`  
The guide specification (see `~numpyro_forecast.typing.GuideLike`).

`model: ForecastModel`  
The model the guide is built against.


## Returns


`AutoGuide | Callable[…, None]`  
The resolved guide.


## Raises


`TypeError`  
If `guide` is neither an `AutoGuide` (instance/subclass/partial) nor a callable, or if it has the mistyped-factory shape.
