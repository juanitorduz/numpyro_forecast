## exceptions.KernelConfigError


A `contrib.blackjax` kernel is run unbound or misconfigured.


Usage

``` python
exceptions.KernelConfigError(message=None)
```


Raised by `~numpyro_forecast.contrib.blackjax._BlackjaxKernel.init()` when the kernel was constructed with no model bound (e.g. [BlackjaxNUTSKernel()](contrib.blackjax.BlackjaxNUTSKernel.md#numpyro_forecast.contrib.blackjax.BlackjaxNUTSKernel) instead of `BlackjaxNUTSKernel(model)`); the fix is to pass the model as the kernel's first argument at construction time, before handing the kernel to `~numpyro.infer.MCMC`. BlackJAX kernels also require `chain_method="sequential"` and `num_warmup=0`; see the "Run configuration" section of `~numpyro_forecast.contrib.blackjax.BlackjaxNUTSKernel` and its sibling kernels for why, and how misconfiguring either surfaces.
