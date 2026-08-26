## typing.InSampleFn


A closure that fits a model on a training window and scores its in-sample fit.


Usage

``` python
typing.InSampleFn()
```


Called by `~numpyro_forecast.evaluate.backtest()` (only when `eval_train=True`) positionally. Returns in-sample posterior-predictive samples with the sample axis first, shape `(num_samples, *batch, t1 - t0, obs)`. The same `batch_size`/host-offload notes as [ForecastFn](typing.ForecastFn.md#numpyro_forecast.typing.ForecastFn) apply.


## Methods

| Name | Description |
|----|----|
| [__call__()](#__call__) | Fit on the training window and return in-sample predictive draws. |

------------------------------------------------------------------------


#### \_\_call\_\_()


Fit on the training window and return in-sample predictive draws.


Usage

``` python
__call__(
    rng_key,
    model,
    train_data,
    train_covariates,
    num_samples,
    /,
    *,
    batch_size=None
)
```
