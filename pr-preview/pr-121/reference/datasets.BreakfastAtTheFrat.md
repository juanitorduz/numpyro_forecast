## datasets.BreakfastAtTheFrat


The three sheets of the dunnhumby *Breakfast at the Frat* workbook as polars frames.


Usage

``` python
datasets.BreakfastAtTheFrat(
    transactions,
    products,
    stores,
)
```


Column names and string values are lowercase (see [load_breakfast_at_the_frat()](datasets.load_breakfast_at_the_frat.md#numpyro_forecast.datasets.load_breakfast_at_the_frat)).


## Attributes


`transactions: polars.DataFrame`  
One row per store, product and week: units, visits, households, spend, the shelf and base prices, and the promotion flags `feature`, `display` and `tpr_only`.

`products: polars.DataFrame`  
The product lookup keyed by `upc`: description, manufacturer, category, sub-category and size.

`stores: polars.DataFrame`  
The store lookup keyed by `store_id`, exactly as in the workbook: two store ids appear twice with different price-segment labels.
