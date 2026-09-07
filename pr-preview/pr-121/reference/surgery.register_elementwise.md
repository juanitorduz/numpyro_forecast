## surgery.register_elementwise()


Declare a distribution family elementwise (usable as a decorator).


Usage

``` python
surgery.register_elementwise(cls)
```


An elementwise family is independent across every batch cell with an empty event shape, so [slice_time()](surgery.slice_time.md#numpyro_forecast.surgery.slice_time) may slice its broadcast parameters along the time axis and [prefix_condition()](surgery.prefix_condition.md#numpyro_forecast.surgery.prefix_condition) may reduce to the horizon marginal. Membership is by exact type; register each concrete subclass you rely on.


## Parameters


`cls: type[dist.Distribution]`  
The distribution class to register.


## Returns


`type[dist.Distribution]`  
`cls` unchanged, so this can decorate a class definition.
