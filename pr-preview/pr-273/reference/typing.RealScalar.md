## typing.RealScalar


A concrete real scalar: a Python `float` or `int`, or a NumPy integer or floating scalar.


`typing.RealScalar=IntScalar | float | np.floating`


A package function with a parameter annotated [RealScalar](typing.RealScalar.md#numpyro_forecast.typing.RealScalar) converts the value with `float` at the function boundary. A Python `int` also satisfies a plain `float` hint (the PEP 484 numeric tower, which the runtime checks apply), so this alias matters for NumPy scalars such as `np.int64(7)` or `np.float32(7.0)`. A JAX array (also a tracer) is rejected.
