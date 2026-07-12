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


## Parameters


`model: ForecastModel | None = None`  
The NumPyro model to sample from.

`num_tuning_steps: int = ``500`  
Number of tuning steps for `L` and the step size.
