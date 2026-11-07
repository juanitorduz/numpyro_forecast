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

An `Independent` wrapper follows its base distribution. When the predictor must be the mean of a `Gumbel` or `AsymmetricLaplace` observation, center the noise yourself, for example `dist.Gumbel(-np.euler_gamma * scale, scale)` or `dist.AsymmetricLaplace(scale * (asymmetry - 1 / asymmetry), scale, asymmetry)`.

A `MultivariateNormal` over time is noise correlated across the time steps of one series (`obs == 1`): a `(*batch, time)` location such as `jnp.zeros(h.duration)` with a `(*batch, time, time)` covariance. Shifted by a predictor with the observation axis, `(*batch, time, 1)` as [predict()](models.predict.md#numpyro_forecast.models.predict) supplies it, the result is the same Gaussian with event shape `(time, 1)`: `log_prob` of `(*batch, time, 1)` data is the joint density over time, one value per series, and draws keep the observation axis. The series of a panel go in the batch, `(series, time, 1)`, and a size-1 time axis, `(*batch, 1, 1)`, shifts every time step by the same value. A `(*batch, time)` predictor, with no more axes than the noise location, keeps the `(time,)` event. A link callable for [predict()](models.predict.md#numpyro_forecast.models.predict) gets the `(time, 1)` event with `lambda mu: shift_loc(dist.MultivariateNormal(jnp.zeros(h.duration), cov), mu)`.


## Parameters


`noise_dist: dist.Distribution`  
A location-family noise distribution, usually with location `0`.

`loc: Array`  
The deterministic predictor to add to the distribution's location.


## Returns


`dist.Distribution`  
The same family with its location shifted by `loc` (the mean or the mode, depending on the family, see above). For a `MultivariateNormal` and a `(*batch, time, 1)` predictor, a distribution with event shape `(time, 1)`: its [mean](typing.Array.md#numpyro_forecast.typing.Array.mean) and `variance` have the observation axis, and its `base_dist` is the shifted `MultivariateNormal` (`loc`, `covariance_matrix`, `entropy()`).


## Raises


`NotImplementedError`  
If `noise_dist` is of an unsupported type.

`MVNLayoutError`  
If `noise_dist` is a `MultivariateNormal` and its location or `loc` matches neither layout above, for example a `(time, obs)` predictor with `obs > 1`.
