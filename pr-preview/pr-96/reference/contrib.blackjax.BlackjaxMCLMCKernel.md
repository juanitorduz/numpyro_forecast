## contrib.blackjax.BlackjaxMCLMCKernel


BlackJAX Microcanonical Langevin Monte Carlo (MCLMC).


Usage

``` python
contrib.blackjax.BlackjaxMCLMCKernel(
    model=None,
    *,
    num_tuning_steps=500,
)
```


The step size, trajectory length `L`, and diagonal preconditioner (inverse mass matrix) are tuned once in `~_BlackjaxKernel.init()` via `blackjax.mclmc_find_L_and_step_size`; each MCMC step is then a single tuned MCLMC step.


## Run Configuration

Pass this kernel to `~numpyro.infer.MCMC` with **`chain_method="sequential"` only**: the instance holds its step/postprocess functions as plain attributes (`self._step_fn`, `self._postprocess_fn`), and `vmap`/`pmap` chain parallelism (`"vectorized"`/`"parallel"`) traces this instance, capturing those attributes as tracers instead of running the closed-over blackjax step. Also pass **`num_warmup=0`**: tuning runs once inside `~_BlackjaxKernel.init()` (via `blackjax.mclmc_find_L_and_step_size`), so any NumPyro-driven warmup steps on top of that are simply discarded work, not additional tuning.


## Parameters


`model: ForecastModel | None = None`  
The NumPyro model to sample from.

`num_tuning_steps: int = ``500`  
Number of tuning steps for `L` and the step size.
