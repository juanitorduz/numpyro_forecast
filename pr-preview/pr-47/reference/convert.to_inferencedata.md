## convert.to_inferencedata()


Legacy shim converting via [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) to an `InferenceData`.


Usage

``` python
convert.to_inferencedata(
    *args,
    **kwargs,
)
```


Deprecated in favor of [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree); emits a `FutureWarning`. Requires classic ArviZ exposing `InferenceData.from_datatree`; raises an actionable `ImportError` otherwise (the package pins ArviZ \>= 1.2, so this path is only for environments that additionally install classic ArviZ).


## Parameters


`*args: Any`  
Positional arguments forwarded to [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree).

`**kwargs: Any`  
Keyword arguments forwarded to [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree).


## Returns


`arviz.InferenceData`  
The converted legacy container.


## Raises


`ImportError`  
If the installed ArviZ does not expose `InferenceData.from_datatree`.
