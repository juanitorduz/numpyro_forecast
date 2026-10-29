## surgery.shift_loc()


Shift the location parameter of a noise distribution by `loc`.


Usage

``` python
surgery.shift_loc(
    noise_dist,
    loc,
)
```


This converts Pyro's `obs = data - prediction` idiom into an additive shift of the observation distribution's location: the result is `noise_dist` with `loc` added to its location parameter and every other parameter unchanged. For noise with location `0` the result has its location at `loc`, and that location is not the mean for every family:

- `Normal`, `Laplace` and `MultivariateNormal`: `loc` is the mean (and also the median and the mode).
- `StudentT`: `loc` is the mean when `df > 1` (and always the median and the mode); for `df <= 1` there is no mean.
- `Cauchy`: `loc` is the mode (and the median); there is no mean.
- `Gumbel`: `loc` is the mode; the mean is `loc + np.euler_gamma * scale` (about `loc + 0.5772 * scale`).
- `AsymmetricLaplace`: `loc` is the mode; the mean is `loc + scale * (1 / asymmetry - asymmetry)`, which is `loc` only for `asymmetry = 1` (the symmetric case).

An `Independent` wrapper follows its base distribution. When the predictor must be the mean of a `Gumbel` or `AsymmetricLaplace` observation, center the noise yourself, for example `dist.Gumbel(-np.euler_gamma * scale, scale)`.


## Parameters


`noise_dist: dist.Distribution`  
A location-family noise distribution, usually with location `0`.

`loc: Array`  
The deterministic predictor to add to the distribution's location.


## Returns


`dist.Distribution`  
The same family with its location shifted by `loc` (the mean or the mode, depending on the family, see above).


## Raises


`NotImplementedError`  
If `noise_dist` is of an unsupported type.
