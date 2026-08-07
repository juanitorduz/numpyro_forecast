## contrib.blackjax.BlackjaxNUTSKernel


BlackJAX NUTS with Stan-style window adaptation.


Usage

``` python
contrib.blackjax.BlackjaxNUTSKernel(
    model=None, *, num_adaptation_steps=500, target_acceptance_rate=0.8
)
```


Adaptation (step size and inverse mass matrix) runs once in `~_BlackjaxKernel.init()` via `blackjax.window_adaptation`; each MCMC step is then a single tuned NUTS step.


## Parameters


`model: ForecastModel | None = None`  
The NumPyro model to sample from.

`num_adaptation_steps: int = ``500`  
Number of window-adaptation steps.

`target_acceptance_rate: float = ``0.8`  
Target acceptance probability for dual-averaging step-size adaptation.
