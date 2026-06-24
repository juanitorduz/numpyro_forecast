## datasets.load_bart_weekly()


Load total weekly BART ridership (log scale) for the univariate example.


Usage

``` python
datasets.load_bart_weekly()
```


Hourly counts are summed over all origin-destination pairs, aggregated into non-overlapping weeks, and log-transformed.


## Returns


`Float[Array, ``" weeks 1"]`  
Log weekly totals with time at axis `-2` and a single observation dim.
