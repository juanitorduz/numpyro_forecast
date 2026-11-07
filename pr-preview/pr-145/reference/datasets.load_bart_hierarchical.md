## datasets.load_bart_hierarchical()


Load the windowed hierarchical BART panel for the hierarchical example.


Usage

``` python
datasets.load_bart_hierarchical(
    train_days=90,
    test_weeks=2,
)
```


The counts are `log1p`-transformed and transposed to the `(origin, time, destin)` convention, then restricted to a `train_days` training window followed by a `test_weeks` test window.


## Parameters


`train_days: int = ``90`  
Number of training days (24 hours each). Must be at least 1.

`test_weeks: int = ``2`  
Number of test weeks (`24 * 7` hours each). Must be non-negative; `0` returns the training window only, with `split == y.shape[1]`.


## Returns


`y: Float[Array, ``" origin time destin"]`  
Log counts over the train+test window with time at axis `-2`.

`split: int`  
Index along the time axis separating train from test.

`stations: list[str]`  
Station names.


## Raises


`ValueError`  
If `train_days` is less than 1 or `test_weeks` is negative (checked before the download), or if the requested `train_days` + `test_weeks` window exceeds the available history (which would otherwise wrap a negative slice index).
