## contrib.blackjax.BlackjaxNUTSKernel


BlackJAX NUTS with Stan-style window adaptation.


Usage

``` python
contrib.blackjax.BlackjaxNUTSKernel(
    model=None, *, num_adaptation_steps=500, target_acceptance_rate=0.8
)
```


Adaptation (step size and inverse mass matrix) runs once in `~_BlackjaxKernel.init()` via `blackjax.window_adaptation`; each MCMC step is then a single tuned NUTS step.


## Run Configuration

Pass this kernel to `~numpyro.infer.MCMC` with **`chain_method="sequential"` only**: the instance holds its step/postprocess functions as plain attributes (`self._step_fn`, `self._postprocess_fn`), and `vmap`/`pmap` chain parallelism (`"vectorized"`/`"parallel"`) traces this instance, capturing those attributes as tracers instead of running the closed-over blackjax step. Also pass **`num_warmup=0`**: adaptation runs once inside `~_BlackjaxKernel.init()` (via `blackjax.window_adaptation`), so any NumPyro-driven warmup steps on top of that are simply discarded work, not additional tuning.


## Parameters


`model: ForecastModel | None = None`  
The NumPyro model to sample from.

`num_adaptation_steps: int = ``500`  
Number of window-adaptation steps.

`target_acceptance_rate: float = ``0.8`  
Target acceptance probability for dual-averaging step-size adaptation.
