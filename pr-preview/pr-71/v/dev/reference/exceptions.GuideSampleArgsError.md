## exceptions.GuideSampleArgsError


Drawing from a hand-written guide needs the in-sample arguments.


Usage

``` python
exceptions.GuideSampleArgsError(message=None)
```


Raised by `~numpyro_forecast.functional.posterior.draw_posterior()` when an `~numpyro_forecast.functional.svi.SVIFit` holding a hand-written guide was constructed without its in-sample covariates/data.
