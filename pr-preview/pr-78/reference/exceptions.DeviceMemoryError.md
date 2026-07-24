## exceptions.DeviceMemoryError


The accelerator ran out of memory during posterior or predictive sampling.


Usage

``` python
exceptions.DeviceMemoryError(message=None)
```


Raised by `~numpyro_forecast.functional.posterior.draw_posterior()` and the predictive drivers (`~numpyro_forecast.functional.prediction.forecast()`, `~numpyro_forecast.functional.prediction.predict_in_sample()`, and everything built on them, e.g. `~numpyro_forecast.convert.to_datatree()`) when XLA reports `RESOURCE_EXHAUSTED`. The message embeds the device's memory budget and the lever: the per-chunk footprint scales linearly with `batch_size` times the panel width, so lower (or set) `batch_size`, free large device arrays still referenced elsewhere, and keep results off the accelerator with `device="host"`. The original XLA error is chained as `__cause__`.
