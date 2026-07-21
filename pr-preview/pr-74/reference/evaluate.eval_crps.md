## evaluate.eval_crps()


Empirical CRPS averaged over all data elements.


Usage

``` python
evaluate.eval_crps(pred: Float[Array, " sample *batch"] | Float[np.ndarray, " sample *batch"], truth: Float[Array, " *batch"] | Float[np.ndarray, " *batch"], batch_size: None = None) -> Array
 
evaluate.eval_crps(pred: Float[Array, " sample *batch"] | Float[np.ndarray, " sample *batch"], truth: Float[Array, " *batch"] | Float[np.ndarray, " *batch"], batch_size: int) -> Array | np.floating
```


A pure JAX scalar kernel (see `~numpyro_forecast.typing.Metric`).


## Parameters


`pred: Float[Array, ``" sample *batch"] | Float[np.ndarray, `<span class="st">`" sample *batch"``]`</span>  
Forecast samples with the sample axis first.

`truth: Float[Array, ``" *batch"] | Float[np.ndarray, `<span class="st">`" *batch"``]`</span>  
Ground-truth values (matching `pred` without the sample axis).

`batch_size: int | None = None`  
Optional number of flattened data cells (the product of the batch shape, e.g. time times series) evaluated on the accelerator per pass; the sample axis is never chunked. With host-resident (NumPy) inputs this bounds accelerator memory by `sample * batch_size` values plus the CRPS sort workspace instead of the full panel. Chunking only changes the summation order of the final mean (results are equal to float tolerance, not bitwise); at or above the cell count the single-pass path runs. `None` (default) evaluates in one pass.


## Returns


`Array | np.floating`  
The mean empirical CRPS as a scalar (a NumPy scalar when chunked).
