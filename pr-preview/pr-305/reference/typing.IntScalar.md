## typing.IntScalar


A concrete integer scalar: a Python `int` or a NumPy integer scalar (`np.int64`, …).


`typing.IntScalar=int | np.integer`


`np.int64` is not a subclass of `int`, so a count read from NumPy or a DataFrame (`df["day"].max()`, `np.sum(mask)`) fails a plain `int` hint. A package function with a parameter annotated [IntScalar](typing.IntScalar.md#numpyro_forecast.typing.IntScalar) accepts it and converts it with `operator.index` at the function boundary, so the function body works with a Python `int` (arbitrary precision, no NumPy overflow). A `float` such as `10.0` and a JAX array (also a tracer) are rejected. Two members of the union are wider than a count: a Python `bool` is an `int` and is accepted as `0` or `1`, and `np.timedelta64` is an `np.integer` (so it passes the hint) but has no `__index__`, so `operator.index` raises `TypeError` for it; convert a time difference to a count yourself (`delta // np.timedelta64(1, "D")`).
