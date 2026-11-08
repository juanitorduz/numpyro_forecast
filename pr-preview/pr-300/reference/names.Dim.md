## names.Dim


xarray dimension names of the trees [to_datatree()](convert.to_datatree.md#numpyro_forecast.convert.to_datatree) writes.


Usage

``` python
names.Dim()
```


`TIME` has the value of `Plate.TIME` by convention; it is the tree's dimension, not NumPyro's plate, and only this one is compared against the stored dimension names.


## Attributes


`TIME: Final`  
`"time"`, the time axis (axis `-2` of every stored variable).

`OBS_DIM: Final`  
`"obs_dim"`, the default name of the observation axis of `obs`.

`COVARIATE_DIM: Final`  
`"covariate_dim"`, the default name of the covariate axis of `covariates`.

`CHAIN: Final`  
`"chain"`, the first ArviZ sample dimension.

`DRAW: Final`  
`"draw"`, the second ArviZ sample dimension.
