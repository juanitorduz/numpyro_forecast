## functional.models.Horizon


The train/forecast split for a single model call.


Usage

``` python
functional.models.Horizon(
    data,
    t_obs,
    future,
    duration,
)
```


Replaces the mutable `self._*` state of the OOP base class with an immutable value derived from the covariate and data shapes via [from_data()](functional.models.Horizon.md#numpyro_forecast.functional.models.Horizon.from_data). The functional primitives ([time_series()](functional.models.time_series.md#numpyro_forecast.functional.models.time_series), [predict()](functional.models.predict.md#numpyro_forecast.functional.models.predict)) take it as their first argument.


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


Mirrors Pyro's [zero_data](functional.models.Horizon.md#numpyro_forecast.functional.models.Horizon.zero_data) (and [numpyro_forecast.arrays.zero_data_like()](arrays.zero_data_like.md#numpyro_forecast.arrays.zero_data_like)): it exposes the shape/dtype of the data over the forecast horizon without leaking observed values. `None` when there is no data.


## Methods

| Name | Description |
|----|----|
| [__post_init__()](#__post_init__) | Validate that the horizon fields are internally consistent. |
| [from_data()](#from_data) | Derive the horizon from covariate and data shapes. |

------------------------------------------------------------------------


#### \_\_post_init\_\_()


Validate that the horizon fields are internally consistent.


Usage

``` python
__post_init__()
```


------------------------------------------------------------------------


#### from_data()


Derive the horizon from covariate and data shapes.


Usage

``` python
from_data(covariates, data)
```


##### Parameters


`covariates: Array`  
Covariates with time at axis `-2` spanning the full horizon.

`data: Array | None`  
Observed data with time at axis `-2` (`None` for prior sampling).


##### Returns


`Horizon`  
The horizon with `duration = covariates.shape[-2]`, `t_obs = data.shape[-2]` (or [duration](forecaster.ForecastingModel.md#numpyro_forecast.forecaster.ForecastingModel.duration) when `data` is `None`), and `future = duration - t_obs`.


##### Raises


`ValueError`  
If `data` is longer than `covariates` along the time axis.
