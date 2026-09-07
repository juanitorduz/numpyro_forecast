## datasets.load_breakfast_at_the_frat()


Load the dunnhumby *Breakfast at the Frat* scanner panel as polars frames.


Usage

``` python
datasets.load_breakfast_at_the_frat(cache_dir=None)
```


The panel holds 156 weeks of weekly unit sales, prices and promotion mechanics for 55 products (58 lookup rows) in 77 stores (79 lookup rows). The loader downloads the [figshare copy](https://doi.org/10.6084/m9.figshare.30121060) of the workbook (about 31 MB) into `cache_dir` on first use, verifies its SHA-256 digest on every call, reads the transaction, product and store sheets with polars' calamine engine (every sheet carries a title row above the header), and lowercases every column name and every string value, so the examples work with one case. The figshare record declares a CC BY 4.0 license for the copy; dunnhumby's own terms govern the data.


## Parameters


`cache_dir: Path | None = None`  
Directory that holds the cached workbook. Defaults to `~/.cache/numpyro_forecast`.


## Returns


`BreakfastAtTheFrat`  
The three sheets as polars frames with lowercase column names and string values.


## Raises


`ValueError`  
If the digest of the cached workbook does not match the recorded one.

`ImportError`  
If `polars` or `fastexcel` is not installed (`pip install numpyro_forecast[dataframes]`).
