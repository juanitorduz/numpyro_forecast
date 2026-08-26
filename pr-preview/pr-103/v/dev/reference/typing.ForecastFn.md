## typing.ForecastFn


A closure that fits a model on a training window and forecasts its test horizon.


Usage

``` python
typing.ForecastFn()
```


Called by `~numpyro_forecast.evaluate.backtest()` positionally, with `full_covariates` spanning the *full* window (train followed by test, i.e. `covariates[..., t0:t2, :]`) and `batch_size` forwarded unchanged from [backtest](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) so a chunked closure can bound its own device memory. Returns forecast samples with the sample axis first, shape `(num_samples, *batch, t2 - t1, obs)`. The draws may stay in host memory (e.g. via `device="host"`): a jax Array committed to the CPU backend device or, without a CPU backend, a NumPy array. Every metric in `~numpyro_forecast.evaluate.DEFAULT_METRICS` accepts such a `pred` or `truth` (or both), in any mix and regardless of `batch_size`, moving a host-resident operand to device memory first where needed; draws already on-device avoid that hop for the metrics scored every window. The parameters are positional-only so a closure keeps its own parameter names. At runtime the beartype hook only checks that the value is callable (Python protocols never inspect signatures); `ty` checks the signature structurally at the [backtest](evaluate.backtest.md#numpyro_forecast.evaluate.backtest) call site.


## Methods

| Name | Description |
|----|----|
| [__call__()](#__call__) | Fit on the training window and return forecast draws for the test window. |

------------------------------------------------------------------------


#### \_\_call\_\_()


Fit on the training window and return forecast draws for the test window.


Usage

``` python
__call__(
    rng_key,
    model,
    train_data,
    train_covariates,
    full_covariates,
    num_samples,
    /,
    *,
    batch_size=None
)
```
