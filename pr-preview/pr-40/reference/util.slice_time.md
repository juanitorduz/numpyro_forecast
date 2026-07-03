## util.slice_time()


Slice an elementwise distribution along the time axis `-2`.


Usage

``` python
util.slice_time(
    noise_dist,
    index,
)
```


The default implementation handles distributions with empty `event_shape` whose `batch_shape` ends with `(time, obs)` (e.g. `Normal`, `StudentT`) by slicing each broadcast parameter.


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
If the distribution has a non-empty event shape.
