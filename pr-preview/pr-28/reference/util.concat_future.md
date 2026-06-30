## util.concat_future()


Concatenate in-sample and forecast-horizon arrays along the time axis.


Usage

``` python
util.concat_future(
    prefix,
    suffix,
    *,
    axis=-2,
)
```


## Parameters


`prefix: Array`  
In-sample array.

`suffix: Array`  
Forecast-horizon array (same shape as `prefix` except along `axis`).

`axis: int = `<span class="dv">`-2`  
</span>  
Time axis to concatenate along (defaults to `-2`).


## Returns


`Array`  
The concatenation of `prefix` and `suffix` along `axis`.
