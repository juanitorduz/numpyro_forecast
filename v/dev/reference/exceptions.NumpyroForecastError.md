## exceptions.NumpyroForecastError


Base class for all deliberate `numpyro_forecast` errors.


Usage

``` python
exceptions.NumpyroForecastError()
```


Subclasses may set `default_message`; instantiating without arguments then carries that message, and a positional message overrides it.


## Parameters


`message: str | None = None`  
Optional explicit message; defaults to the class `default_message`.
