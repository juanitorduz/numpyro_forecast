## optional.require()


Import an optional dependency, or raise a targeted `ImportError`.


Usage

``` python
optional.require(
    module,
    *,
    extra,
)
```


Optional features (dataframes, blackjax, optax) live behind `pyproject` extras and are never imported at package import time. This helper imports the backing module lazily at first use and, when it is missing, raises an `ImportError` naming the exact `pip install` invocation that provides it.


## Parameters


`module: str`  
The importable module name (e.g. `"pandas"`).

`extra: str`  
The `numpyro_forecast` extra that installs it (e.g. `"dataframes"`).


## Returns


`ModuleType`  
The imported module.


## Raises


`ImportError`  
If `module` is not importable, with an actionable install hint.
