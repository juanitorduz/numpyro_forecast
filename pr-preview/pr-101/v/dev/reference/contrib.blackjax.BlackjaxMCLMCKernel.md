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


The step size, trajectory length `L`, and diagonal preconditioner (inverse mass matrix) are tuned once in `_BlackjaxKernel.init()` via `blackjax.mclmc_find_L_and_step_size`; each MCMC step is then a single tuned MCLMC step.


## Run Configuration

Pass this kernel to `numpyro.infer.MCMC` with **`chain_method="sequential"` only**. With more than one chain, `chain_method="vectorized"` hands `_BlackjaxKernel.init()` a stacked `rng_key` of shape `(num_chains, 2)` rather than vmapping a per-chain call, and the `jax.random.split` inside `init` rejects a non-scalar key with a `ValueError` (pinned by the test suite). `chain_method="parallel"` is likewise unsupported: it is not exercised by this package and shares the same single-key assumption. Also pass **`num_warmup=0`**: tuning runs once inside `_BlackjaxKernel.init()` (via `blackjax.mclmc_find_L_and_step_size`), so any NumPyro-driven warmup steps on top of that are simply discarded work, not additional tuning.


## Parameters


`model: ForecastModel | None = None`  
The NumPyro model to sample from.

`num_tuning_steps: int = ``500`  
Number of tuning steps for `L` and the step size.
