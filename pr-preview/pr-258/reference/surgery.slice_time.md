## surgery.slice_time()


Slice an elementwise distribution along the time axis `-2`.


Usage

``` python
surgery.slice_time(
    noise_dist,
    index,
)
```


The default implementation handles registered elementwise families (empty `event_shape`, `batch_shape` ending in `(time, obs)`; e.g. `Normal`, `StudentT`, `Poisson`) by slicing each broadcast parameter. Correlated families register a dedicated dispatch instead: a `MultivariateNormal` over time slices its time axis and keeps its layout, so the event `(time, 1)` returned by [shift_loc()](surgery.shift_loc.md#numpyro_forecast.surgery.shift_loc) stays `(steps, 1)` and a bare `(time,)` event stays `(steps,)`.


## Parameters


`noise_dist: dist.Distribution`  
The distribution to slice.

`index: slice`  
A `slice` applied to the time axis `-2` of the batch shape.


## Returns


`dist.Distribution`  
The same distribution family restricted to the selected time steps.


## Raises


`NotImplementedError`  
If `noise_dist` is not a registered elementwise family (and has no dedicated dispatch).
