## util.prefix_condition()


Condition a `(t+f)`-length distribution on a `t`-length data prefix.


Usage

``` python
util.prefix_condition(
    noise_dist,
    data,
)
```


For independent-over-time noise (the elementwise default) the conditional reduces to the forecast-horizon marginal, i.e. a time slice `[t:]`. Only registered elementwise families take this path; correlated families (e.g. `MultivariateNormal`) need a dedicated dispatch implementing a genuine Gaussian conditional, which is checked here rather than silently reduced to a slice (the R1 fix).


## Parameters


`noise_dist: dist.Distribution`  
The observation distribution over the full horizon `(*batch, t+f, obs)`.

`data: Array`  
The observed prefix with shape `(*batch, t, obs)`.


## Returns


`dist.Distribution`  
The forecast-horizon distribution over `(*batch, f, obs)`.


## Raises


`NotImplementedError`  
If `noise_dist` is not a registered elementwise family (and has no dedicated dispatch).
