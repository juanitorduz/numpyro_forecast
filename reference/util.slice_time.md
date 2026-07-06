## util.slice_time()


Slice an elementwise distribution along the time axis `-2`.


Usage

``` python
util.slice_time(
    noise_dist,
    index,
)
```


The default implementation handles registered elementwise families (empty `event_shape`, `batch_shape` ending in `(time, obs)`; e.g. `Normal`, `StudentT`, `Poisson`) by slicing each broadcast parameter. Correlated families register a dedicated dispatch instead.


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
