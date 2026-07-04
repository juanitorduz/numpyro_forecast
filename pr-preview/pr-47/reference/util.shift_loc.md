## util.shift_loc()


Re-center a zero-centered noise distribution at `loc`.


Usage

``` python
util.shift_loc(
    noise_dist,
    loc,
)
```


This converts Pyro's `obs = data - prediction` idiom into an additive shift of the observation distribution's location.


## Parameters


`noise_dist: dist.Distribution`  
A zero-centered location-family distribution.

`loc: Array`  
The deterministic mean to add to the distribution's location.


## Returns


`dist.Distribution`  
A distribution centered at `loc`.


## Raises


`NotImplementedError`  
If `noise_dist` is of an unsupported type.
