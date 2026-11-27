## typing.RealScalar


A concrete real scalar: a Python `float` or `int`, or a NumPy integer or floating scalar.


`typing.RealScalar=IntScalar | float | np.floating`


A package function with a parameter annotated [RealScalar](typing.RealScalar.md#numpyro_forecast.typing.RealScalar) accepts an integer value as it is and converts a NumPy float to a Python `float` where it needs one. The alias spells `int`, so an integer passes it under any runtime configuration; it matters for NumPy scalars such as `np.int64(7)` or `np.float32(7.0)`, which a plain `float` hint rejects. A JAX array (also a tracer) is rejected; a Python `bool` passes as `0` or `1`.
