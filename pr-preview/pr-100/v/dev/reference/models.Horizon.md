## models.Horizon


The train/forecast split for a single model call.


Usage

``` python
models.Horizon(
    data,
    t_obs,
    future,
    duration,
)
```


An immutable value derived once per model call from the covariate and data shapes by [from_data()](models.Horizon.md#numpyro_forecast.models.Horizon.from_data); every building block ([innovations()](models.innovations.md#numpyro_forecast.models.innovations), [markov_series()](models.markov_series.md#numpyro_forecast.models.markov_series), [predict()](models.predict.md#numpyro_forecast.models.predict)) takes it as its first argument.


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
| [__post_init__()](#__post_init__) | Validate that the horizon fields are internally consistent. |
| [from_data()](#from_data) | Derive the horizon from the covariate and data shapes. |

------------------------------------------------------------------------


#### \_\_post_init\_\_()


Validate that the horizon fields are internally consistent.


Usage

``` python
__post_init__()
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
