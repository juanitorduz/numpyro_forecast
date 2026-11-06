## datasets.load_m5()


Load the M5 competition data (download and cache once, then read the files).


Usage

``` python
datasets.load_m5(cache_dir=None)
```


The files come from [Nixtla's mirror](https://github.com/Nixtla/m5-forecasts) of the competition data: the daily unit sales of the 30,490 series over the training period (days 1 to 1,941) and the 28 evaluation days released after the competition, the calendar with the events and the SNAP days, the weekly shelf prices, and the official evaluation weights. The 50 MB archive is downloaded once into `cache_dir` (default `~/.cache/numpyro_forecast/m5`) and checked against a pinned SHA-256 digest.

Reading needs polars (`pip install numpyro_forecast[dataframes]`).


## Parameters


`cache_dir: str | Path | None = None`  
Directory that holds the archive and the extracted files; created if missing.


## Returns


`M5Data`  
Dense `(days, series)` sales and price arrays, the identifier table, the calendar and the official weights.
