## functional.resolve_optimizer()


Normalize an optimizer specification into a NumPyro optimizer.


Usage

``` python
functional.resolve_optimizer(optim)
```


Accepted forms: `None` (`Adam(0.01)`); a finite positive scalar learning rate (`float`/`int`/NumPy scalar/0-d array) giving `Adam(lr)`; an `optax.GradientTransformation` (wrapped via `numpyro.optim.optax_to_numpyro`, imported lazily so optax stays a soft dependency); a `_NumPyroOptim` (returned unchanged).


## Parameters


`optim: OptimizerLike`  
The optimizer specification (see `~numpyro_forecast.typing.OptimizerLike`).


## Returns


`_NumPyroOptim`  
The resolved NumPyro optimizer.


## Raises


`OptimizerResolutionError`  
For boolean inputs of any form, including 0-d boolean arrays (`bool` is an `int` subclass, so a bool would silently mean `Adam(1.0)`), and for any other unrecognized type; the message lists the accepted forms.

`ValueError`  
For a non-finite or non-positive learning rate.
