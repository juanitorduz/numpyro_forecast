## var.var_step()


Build the [ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe) step of a VAR from its coefficients.


Usage

``` python
var.var_step(
    phi,
    intercept=None,
)
```


The carry is the lag window `(*batch, lags, obs)`. Each step emits [var_mean()](var.var_mean.md#numpyro_forecast.var.var_mean) of the window as the one-step-ahead mean and, given the row's value, drops the oldest row and appends the new one. The step ignores its exogenous input `x_t`; add regressors by wrapping it (a VARX):

``` python
base = var_step(phi, intercept)


def step(carry, x_t):
    mu, carry_fn = base(carry, x_t)
    return mu + beta @ x_t, carry_fn
```

The step knows nothing about priors: `phi` and `intercept` are whatever the model sampled (a weakly informative `Normal`, the moments of [minnesota_prior()](priors.minnesota_prior.md#numpyro_forecast.priors.minnesota_prior), a hierarchical prior, …), so changing the prior never touches the recursion.


## Parameters


`phi: Float[Array, ``" *#batch lags obs obs"]`  
Coefficient tensor `(*batch, lags, obs, obs)`; see the module docstring.

`intercept: Float[Array, ``" *#batch obs"] | None`` = None`  
Optional intercept of shape `(*batch, obs)`.


## Returns


`SSOEStep[Float[Array, ``"*batch lags obs"]]`  
A `(carry, x_t) -> (mu_t, carry_fn)` callable for [ssoe()](models.ssoe.md#numpyro_forecast.models.ssoe).


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
    r = ssoe(h, "eps", y, y_init, var_step(phi, intercept), noise)
    numpyro.sample("obs", dist.MultivariateNormal(r.mu, scale_tril=scale_tril), obs=h.data)
    if h.future > 0:
        numpyro.deterministic("forecast", r.y_future)
```

Closing over `y_init` is right for a single fit but wrong under [backtest()](evaluate.backtest.md#numpyro_forecast.evaluate.backtest), which slices `covariates` per window: for a backtest, ship the seed rows inside `covariates` and slice the carry from them in the model.
