## exceptions.OptimizerResolutionError


An optimizer specification could not be resolved.


Usage

``` python
exceptions.OptimizerResolutionError(message=None)
```


Raised by `~numpyro_forecast.functional.resolve_optimizer()` for boolean inputs of any form (`bool` is an `int` subclass, so a bool would silently mean `Adam(1.0)`; the default message) and for any other unrecognized type.
