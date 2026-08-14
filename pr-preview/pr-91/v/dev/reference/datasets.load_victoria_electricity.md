## datasets.load_victoria_electricity()


Load hourly Victoria (Australia) electricity demand and temperature.


Usage

``` python
datasets.load_victoria_electricity()
```


The series covers the first eight weeks of 2014, sampled hourly, from the Victoria electricity demand dataset used in the TensorFlow Probability structural-time-series case study and in Hyndman and Athanasopoulos' *Forecasting: Principles and Practice*. The original half-hourly data is downsampled to hourly by taking every other step. The values are bundled as a small CSV next to this module.


## Returns


`demand: Float[Array, ``" time 1"]`  
Hourly electricity demand (GW) with time at axis `-2` and a single observation dimension.

`temperature: Float[Array, ``" time"]`  
Hourly temperature (degrees Celsius), aligned with `demand`.
