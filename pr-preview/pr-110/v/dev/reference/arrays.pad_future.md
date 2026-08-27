## arrays.pad_future()


Append `future` rows filled with `value` along the time axis.


Usage

``` python
arrays.pad_future(
    x,
    future,
    *,
    value=0.0,
)
```


The frozen-gate recipe of [numpyro_forecast.models.ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe): an update gate observed in-sample (`(*batch, t_obs, obs)`) becomes a full-horizon `xs` leaf whose forecast rows are `value` (`0.0` freezes the carry, `1.0` keeps an availability mask open). Zero `future` returns `x` unchanged in shape.


## Parameters


`x: Array`  
In-sample array with time at axis `-2`, shape `(*batch, t_obs, obs)`.

`future: int`  
Number of forecast rows to append.

`value: float = ``0.0`  
Fill value of the appended rows, cast to `x.dtype`.


## Returns


`Array`  
`x` followed by `future` constant rows, shape `(*batch, t_obs + future, obs)`.
