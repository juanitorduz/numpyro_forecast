## util.prefix_condition()


Condition a `(t+f)`-length distribution on a `t`-length data prefix.


Usage

``` python
util.prefix_condition(
    noise_dist,
    data,
)
```


For independent-over-time noise (the default) the conditional reduces to the forecast-horizon marginal, i.e. a time slice `[t:]`. Only independent families are supported today; correlated families (e.g. `MultivariateNormal`) would need a registered dispatch implementing a genuine Gaussian conditional, which is not yet provided.


## Parameters


`noise_dist: dist.Distribution`  
The observation distribution over the full horizon `(*batch, t+f, obs)`.

`data: Array`  
The observed prefix with shape `(*batch, t, obs)`.


## Returns


`dist.Distribution`  
The forecast-horizon distribution over `(*batch, f, obs)`.
