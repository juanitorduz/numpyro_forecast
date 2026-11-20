## datasets.M5Data


The M5 competition data as dense arrays plus the identifier and calendar tables.


Usage

``` python
datasets.M5Data()
```


## Attributes


`sales: Float[np.ndarray, ``" days series"]`  
Daily unit sales `(days, series)` over the 1,941 training days followed by the 28 evaluation days, series in the order of the sales file.

`price: Float[np.ndarray, ``" days series"]`  
Weekly shelf price of every series repeated over its days, `NaN` on the days the item was not listed.

`keys: polars.DataFrame`  
One row per series: `id` (`item_id` and `store_id` joined by `_`) and the five identifier columns `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`.

`calendar: polars.DataFrame`  
One row per day with the `date` parsed, the Walmart week `wm_yr_wk`, the weekday columns, the event names and types (`null` when there is none) and the SNAP flags of the three states.

`weights: polars.DataFrame`  
The official evaluation weights of the 42,840 aggregates (`Level_id`, `Agg_Level_1`, `Agg_Level_2`, `Dollar_Sales`, `weight`).
