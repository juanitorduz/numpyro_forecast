## datasets.load_m5()


Load the M5 competition data (download and cache once, then read the files).


Usage

``` python
datasets.load_m5(cache_dir=None)
```


The files come from [Nixtla's mirror](https://github.com/Nixtla/m5-forecasts) of the competition data: the daily unit sales of the 30,490 series over the training period (days 1 to 1,941) and the 28 evaluation days released after the competition, the calendar with the events and the SNAP days, the weekly shelf prices, and the official evaluation weights. The 50 MB archive is downloaded once into `cache_dir` (default `~/.cache/numpyro_forecast/m5`) and checked against a pinned SHA-256 digest, also when an `m5.zip` is already in `cache_dir`. The files are read from the cache only after a full extraction of the verified archive, so an interrupted download or extraction is redone on the next call, as is a file that was deleted or truncated later. A cache folder written before this check existed has no completion marker: the first call extracts the files again from its `m5.zip` when that archive matches the digest, and downloads the archive again when it is missing or does not match, which fails without a network connection.

Reading needs polars (`pip install numpyro_forecast[dataframes]`).


## Parameters


`cache_dir: str | Path | None = None`  
Directory that holds the archive and the extracted files; created if missing.


## Returns


`M5Data`  
Dense `(days, series)` sales and price arrays, the identifier table, the calendar and the official weights.
