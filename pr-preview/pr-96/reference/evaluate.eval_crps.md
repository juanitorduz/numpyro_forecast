## evaluate.eval_crps()


Empirical CRPS averaged over all data elements.


Usage

``` python
evaluate.eval_crps(
    pred,
    truth,
    *,
    batch_size=None,
)
```


A pure JAX scalar kernel (see `~numpyro_forecast.typing.Metric`).


## Parameters


`pred: Float[ArrayLike, ``" sample *batch"]`  
Forecast samples with the sample axis first.

`truth: Float[ArrayLike, ``" *batch"]`  
Ground-truth values (matching `pred` without the sample axis).

`batch_size: int | None = None`  
Optional number of flattened data cells (the product of the batch shape, e.g. time times series) evaluated on the accelerator per pass; the sample axis is never chunked. With host-resident inputs (e.g. draws sampled with `device="host"`) this bounds accelerator memory by `sample * batch_size` values plus the CRPS sort workspace instead of the full panel, since both `pred` and `truth` are staged as NumPy views before chunking, whatever their own memory kind. Chunking only changes the summation order of the final mean (results are equal to float tolerance, not bitwise); at or above the cell count the single-pass path runs instead, which moves any host-committed operand back to device memory first (`~numpyro_forecast.functional._offload._device_view()`) so either `pred` or `truth` (or both) may be host-committed regardless of `batch_size`. `None` (default) evaluates in one pass.


## Returns


`Array`  
The mean empirical CRPS as a scalar array.
