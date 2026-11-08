## datasets.USInflation


Quarterly US inflation for four price measures.


Usage

``` python
datasets.USInflation()
```


## Attributes


`rates: Float[Array, ``" quarters measures"]`  
Annualized quarter-on-quarter log inflation in percent, 400 \\ (\log P_t - \log P\_{t-1}), shape `(quarters, measures)` with time at axis `-2`.

`dates: np.ndarray`  
First day of each quarter of `rates` as `datetime64[D]`, starting at 1959-04-01.

`names: tuple[str, …]`  
Column names of `rates` in order: `("cpi", "core_cpi", "pce", "core_pce")`. Core PCE is last.
