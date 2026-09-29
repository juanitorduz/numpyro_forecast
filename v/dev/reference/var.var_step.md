## var.var_step()


Build the [ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe) mean and update of a VAR.


Usage

``` python
var.var_step(
    phi,
    intercept=None,
)
```


The carry is the lag window `(*batch, lags, obs)`. `mean` emits [var_mean()](var.var_mean.md#numpyro_forecast.var.var_mean) of the window as the one-step-ahead mean and, given the row's value, `update` drops the oldest row and appends the new one. Both ignore their exogenous input `x_t`; add regressors by wrapping `mean` (a VARX):

``` python
mean, update = var_step(phi, intercept)


def mean_x(carry, x_t):
    return mean(carry, x_t) + beta @ x_t
```

The pair knows nothing about priors: `phi` and `intercept` are whatever the model sampled (a weakly informative `Normal`, the moments of [minnesota_prior()](priors.minnesota_prior.md#numpyro_forecast.priors.minnesota_prior), a hierarchical prior, …), so changing the prior never touches the recursion.


## Parameters


`phi: Float[Array, ``" *#batch lags obs obs"]`  
Coefficient tensor `(*batch, lags, obs, obs)`; see the module docstring.

`intercept: Float[Array, ``" *#batch obs"] | None`` = None`  
Optional intercept of shape `(*batch, obs)`.


## Returns


`tuple[`\
`    SSOEMean[Float[Array, `<span class="st">`"*batch lags obs"]],`\
`    SSOEUpdate[Float[Array, ``"*batch lags obs"``]],`\
`]`</span>  
A `(mean, update)` pair for [ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe): `mean(carry, x_t)` is [var_mean()](var.var_mean.md#numpyro_forecast.var.var_mean) of the window and `update(carry, y_t, eps_t, x_t)` drops the oldest row and appends `y_t`.


## Raises


`ValueError`  
At step time, if the carry does not hold exactly `phi.shape[-3]` rows (the usual cause is an `init_carry` with the wrong number of lags).


## Examples

An observed VAR(p) conditioned on its first `p` rows. `y_init` is the seed window, a constant closed over by the model; the likelihood rows travel through `covariates` (padded with [pad_future()](arrays.pad_future.md#numpyro_forecast.arrays.pad_future) to fix the forecast horizon) and through `data`:

``` python
def var_model(covariates, data=None):
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :]
    intercept = jnp.asarray(
        numpyro.sample("intercept", dist.Normal(0.0, 1.0).expand([k]).to_event(1))
    )
    sigma = jnp.asarray(numpyro.sample("sigma", dist.HalfNormal(1.0).expand([k]).to_event(1)))
    l_omega = jnp.asarray(numpyro.sample("l_omega", dist.LKJCholesky(k, concentration=1.0)))
    phi = jnp.asarray(
        numpyro.sample("phi", dist.Normal(0.0, 1.0).expand([p, k, k]).to_event(3))
    )
    scale_tril = sigma[..., :, None] * l_omega
    noise = dist.MultivariateNormal(jnp.zeros(k), scale_tril=scale_tril)
    mean, update = var_step(phi, intercept)
    r = ssoe(h, "eps", y, y_init, mean, update, noise)
    numpyro.sample("obs", dist.MultivariateNormal(r.mu, scale_tril=scale_tril), obs=h.data)
    if h.future > 0:
        numpyro.deterministic("forecast", r.y_future)
```

Closing over `y_init` is right for a single fit but wrong under [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest), which slices `covariates` per window: for a backtest, ship the seed rows inside `covariates` and slice the carry from them in the model.
