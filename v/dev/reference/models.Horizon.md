## models.Horizon


The train/forecast split for a single model call.


Usage

``` python
models.Horizon()
```


An immutable value derived once per model call from the covariate and data shapes by [from_data()](models.Horizon.md#numpyro_forecast.models.Horizon.from_data); every building block ([innovations()](models.innovations.md#numpyro_forecast.models.innovations), [markov_series()](models.markov_series.md#numpyro_forecast.models.markov_series), [ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe), [predict()](models.predict.md#numpyro_forecast.models.predict)) takes it as its first argument.

A JAX pytree (an `equinox.Module`): `data` is the only leaf, while `t_obs`, `future` and `duration` are static metadata. A jitted function that takes a [Horizon](models.Horizon.md#numpyro_forecast.models.Horizon) can therefore use the three integers as shapes and recompiles once per horizon length.


## Attributes


`data: Array | None`  
Observed in-sample data with time at axis `-2` (`None` during pure prior sampling).

`t_obs: int`  
Number of observed (in-sample) time steps `t`.

`future: int`  
Number of forecast time steps `f` (`0` while training).

`duration: int`  
Total horizon length `t + future` (in time steps).


## Attributes

| Name | Description |
|----|----|
| [zero_data](#zero_data) | Zeros shaped like `data` extended to the full horizon. |

------------------------------------------------------------------------


#### zero_data


Zeros shaped like `data` extended to the full horizon.


`zero_data: Array | None`


Mirrors Pyro's [zero_data](models.Horizon.md#numpyro_forecast.models.Horizon.zero_data) (and [numpyro_forecast.arrays.zero_data_like()](arrays.zero_data_like.md#numpyro_forecast.arrays.zero_data_like)): it exposes the shape/dtype of the data over the forecast horizon without leaking observed values. `None` when there is no data.


## Methods

| Name | Description |
|----|----|
| [__check_init__()](#__check_init__) | Validate that the horizon fields are internally consistent. |
| [from_data()](#from_data) | Derive the horizon from the covariate and data shapes. |

------------------------------------------------------------------------


#### \_\_check_init\_\_()


Validate that the horizon fields are internally consistent.


Usage

``` python
__check_init__()
```


------------------------------------------------------------------------


#### from_data()


Derive the horizon from the covariate and data shapes.


Usage

``` python
from_data(covariates, data)
```


The first line of every model: `h = Horizon.from_data(covariates, data)`.


##### Parameters


`covariates: Array`  
Covariates with time at axis `-2` spanning the full horizon.

`data: Array | None`  
Observed data with time at axis `-2` (`None` for prior sampling).


##### Returns


`Horizon`  
The horizon with `duration = covariates.shape[-2]`, `t_obs = data.shape[-2]` (or `duration` when `data` is `None`), and `future = duration - t_obs`.


##### Raises


`ValueError`  
If `data` is longer than `covariates` along the time axis.
