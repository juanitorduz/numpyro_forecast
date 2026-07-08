## exceptions.GuideResolutionError


A guide specification could not be resolved.


Usage

``` python
exceptions.GuideResolutionError(message=None)
```


Raised by `~numpyro_forecast.functional.svi.resolve_guide()` for a callable shaped like a guide *factory* (the default message) or for an unsupported type.
