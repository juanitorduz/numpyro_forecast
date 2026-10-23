## typing.Array


A JAX array (alias of `jax.Array`).


Usage

``` python
typing.Array(
    shape, dtype=None, buffer=None, offset=0, strides=None, order=None
)
```


## Methods

| Name | Description |
|----|----|
| [__abs__()](#__abs__) | Alias of `jax.numpy.absolute()`. |
| [__array_namespace__()](#__array_namespace__) | Return the `Python array API`\_ namespace for JAX. |
| [__init__()](#__init__) | Initialize self. See help(type(self)) for accurate signature. |
| [__invert__()](#__invert__) | Compute the bitwise inversion of an input. |
| [__neg__()](#__neg__) | Return element-wise negative values of the input. |
| [__pos__()](#__pos__) | Return element-wise positive values of the input. |
| [addressable_data()](#addressable_data) | Return an array of the addressable data at a particular index. |
| [all()](#all) | Test whether all array elements along a given axis evaluate to True. |
| [any()](#any) | Test whether any array elements along a given axis evaluate to True. |
| [argmax()](#argmax) | Return the index of the maximum value. |
| [argmin()](#argmin) | Return the index of the minimum value. |
| [argpartition()](#argpartition) | Return the indices that partially sort the array. |
| [argsort()](#argsort) | Return the indices that sort the array. |
| [astype()](#astype) | Copy the array and cast to a specified dtype. |
| [byteswap()](#byteswap) | Swap the bytes of the array elements. |
| [choose()](#choose) | Construct an array choosing from elements of multiple arrays. |
| [clip()](#clip) | Return an array whose values are limited to a specified range. |
| [compress()](#compress) | Return selected slices of this array along given axis. |
| [conj()](#conj) | Return the complex conjugate of the array. |
| [conjugate()](#conjugate) | Return the complex conjugate of the array. |
| [copy()](#copy) | Return a copy of the array. |
| [copy_to_host_async()](#copy_to_host_async) | Copies an [Array](typing.Array.md#numpyro_forecast.typing.Array) to the host asynchronously. |
| [cumprod()](#cumprod) | Return the cumulative product of the array. |
| [cumsum()](#cumsum) | Return the cumulative sum of the array. |
| [diagonal()](#diagonal) | Return the specified diagonal from the array. |
| [dot()](#dot) | Compute the dot product of two arrays. |
| [flatten()](#flatten) | Flatten array into a 1-dimensional shape. |
| [item()](#item) | Copy an element of an array to a standard Python scalar and return it. |
| [max()](#max) | Return the maximum of array elements along a given axis. |
| [mean()](#mean) | Return the mean of array elements along a given axis. |
| [min()](#min) | Return the minimum of array elements along a given axis. |
| [nonzero()](#nonzero) | Return indices of nonzero elements of an array. |
| [prod()](#prod) | Return product of the array elements over a given axis. |
| [ptp()](#ptp) | Return the peak-to-peak range along a given axis. |
| [ravel()](#ravel) | Flatten array into a 1-dimensional shape. |
| [repeat()](#repeat) | Construct an array from repeated elements. |
| [reshape()](#reshape) | Returns an array containing the same data with a new shape. |
| [round()](#round) | Round array elements to a given decimal. |
| [searchsorted()](#searchsorted) | Perform a binary search within a sorted array. |
| [sort()](#sort) | Return a sorted copy of an array. |
| [squeeze()](#squeeze) | Remove one or more length-1 axes from array. |
| [std()](#std) | Compute the standard deviation along a given axis. |
| [sum()](#sum) | Sum of the elements of the array over a given axis. |
| [swapaxes()](#swapaxes) | Swap two axes of an array. |
| [take()](#take) | Take elements from an array. |
| [to_device()](#to_device) | Return a copy of the array on the specified device |
| [trace()](#trace) | Return the sum along the diagonal. |
| [transpose()](#transpose) | Returns a copy of the array with axes transposed. |
| [var()](#var) | Compute the variance along a given axis. |
| [view()](#view) | Return a bitwise copy of the array, viewed as a new dtype. |

------------------------------------------------------------------------


#### \_\_abs\_\_()


Alias of `jax.numpy.absolute()`.


Usage

``` python
__abs__(x)
```


------------------------------------------------------------------------


#### \_\_array_namespace\_\_()


Return the `Python array API`\_ namespace for JAX.


Usage

``` python
__array_namespace__(*, api_version=None)
```


.. \_Python array API: https://data-apis.org/array-api/


------------------------------------------------------------------------


#### \_\_init\_\_()


Initialize self. See help(type(self)) for accurate signature.


Usage

``` python
__init__(shape, dtype=None, buffer=None, offset=0, strides=None, order=None)
```


------------------------------------------------------------------------


#### \_\_invert\_\_()


Compute the bitwise inversion of an input.


Usage

``` python
__invert__(x)
```


JAX implementation of [numpy.invert()](typing.Array.md#numpyro_forecast.typing.Array.__invert__). This function provides the implementation of the `~` operator for JAX arrays.

## Parameters

- **x** -- input array, must be boolean or integer typed.

## Returns

An array of the same shape and dtype as \``x`, with the bits inverted.

See also: - `jax.numpy.bitwise_invert()`: Array API alias of this function. - `jax.numpy.logical_not()`: Invert after casting input to boolean.

## Examples

``` python
  >>> x = jnp.arange(5, dtype='uint8')
  >>> print(x)
```

\[0 1 2 3 4\]

``` python
  >>> print(jnp.invert(x))
```

\[255 254 253 252 251\]

This function implements the unary `~` operator for JAX arrays:

``` python
  >>> print(~x)
```

\[255 254 253 252 251\]

[invert()](typing.Array.md#numpyro_forecast.typing.Array.__invert__) operates bitwise on the input, and so the meaning of its output may be more clear by showing the bitwise representation:

``` python
  >>> with jnp.printoptions(formatter={'int': lambda x: format(x, '#010b')}):
  ...   print(f"{x  = }")
  ...   print(f"{~x = }")
```

x = Array(\[0b00000000, 0b00000001, 0b00000010, 0b00000011, 0b00000100\], dtype=uint8) ~x = Array(\[0b11111111, 0b11111110, 0b11111101, 0b11111100, 0b11111011\], dtype=uint8)

For boolean inputs, [invert()](typing.Array.md#numpyro_forecast.typing.Array.__invert__) is equivalent to `logical_not()`:

``` python
  >>> x = jnp.array([True, False, True, True, False])
  >>> jnp.invert(x)
```

Array(\[False, True, False, False, True\], dtype=bool)


------------------------------------------------------------------------


#### \_\_neg\_\_()


Return element-wise negative values of the input.


Usage

``` python
__neg__(x)
```


JAX implementation of `numpy.negative`.

## Parameters

- **x** -- input array or scalar.

## Returns

An array with same shape and dtype as `x` containing `-x`.

See also: - [jax.numpy.positive()](typing.Array.md#numpyro_forecast.typing.Array.__pos__): Returns element-wise positive values of the input. - `jax.numpy.sign()`: Returns element-wise indication of sign of the input.

## Notes

`jnp.negative`, when applied over `unsigned integer`, produces the result of their two's complement negation, which typically results in unexpected large positive values due to integer underflow.

## Examples

    For real-valued inputs:

``` python
  >>> x = jnp.array([0., -3., 7])
  >>> jnp.negative(x)
```

Array(\[-0., 3., -7.\], dtype=float32)

For complex inputs:

``` python
  >>> x1 = jnp.array([1-2j, -3+4j, 5-6j])
  >>> jnp.negative(x1)
```

Array(\[-1.+2.j, 3.-4.j, -5.+6.j\], dtype=complex64)

For unit32:

``` python
  >>> x2 = jnp.array([5, 0, -7]).astype(jnp.uint32)
  >>> x2
```

Array(\[ 5, 0, 4294967289\], dtype=uint32)

``` python
  >>> jnp.negative(x2)
```

Array(\[4294967291, 0, 7\], dtype=uint32)


------------------------------------------------------------------------


#### \_\_pos\_\_()


Return element-wise positive values of the input.


Usage

``` python
__pos__(x)
```


JAX implementation of [numpy.positive](typing.Array.md#numpyro_forecast.typing.Array.__pos__).

## Parameters

- **x** -- input array or scalar

## Returns

An array of same shape and dtype as `x` containing `+x`.

## Notes

`jnp.positive` is equivalent to `x.copy()` and is defined only for the types that support arithmetic operations.

See also: - `jax.numpy.negative()`: Returns element-wise negative values of the input. - `jax.numpy.sign()`: Returns element-wise indication of sign of the input.

## Examples

    For real-valued inputs:

``` python
  >>> x = jnp.array([-5, 4, 7., -9.5])
  >>> jnp.positive(x)
```

Array(\[-5. , 4. , 7. , -9.5\], dtype=float32)

``` python
  >>> x.copy()
```

Array(\[-5. , 4. , 7. , -9.5\], dtype=float32)

For complex inputs:

``` python
  >>> x1 = jnp.array([1-2j, -3+4j, 5-6j])
  >>> jnp.positive(x1)
```

Array(\[ 1.-2.j, -3.+4.j, 5.-6.j\], dtype=complex64)

``` python
  >>> x1.copy()
```

Array(\[ 1.-2.j, -3.+4.j, 5.-6.j\], dtype=complex64)

For uint32:

``` python
  >>> x2 = jnp.array([6, 0, -4]).astype(jnp.uint32)
  >>> x2
```

Array(\[ 6, 0, 4294967292\], dtype=uint32)

``` python
  >>> jnp.positive(x2)
```

Array(\[ 6, 0, 4294967292\], dtype=uint32)


------------------------------------------------------------------------


#### addressable_data()


Return an array of the addressable data at a particular index.


Usage

``` python
addressable_data(index)
```


------------------------------------------------------------------------


#### all()


Test whether all array elements along a given axis evaluate to True.


Usage

``` python
all(axis=None, out=None, keepdims=False, *, where=None)
```


Refer to `jax.numpy.all()` for the full documentation.


------------------------------------------------------------------------


#### any()


Test whether any array elements along a given axis evaluate to True.


Usage

``` python
any(axis=None, out=None, keepdims=False, *, where=None)
```


Refer to `jax.numpy.any()` for the full documentation.


------------------------------------------------------------------------


#### argmax()


Return the index of the maximum value.


Usage

``` python
argmax(axis=None, out=None, keepdims=None)
```


Refer to `jax.numpy.argmax()` for the full documentation.


------------------------------------------------------------------------


#### argmin()


Return the index of the minimum value.


Usage

``` python
argmin(axis=None, out=None, keepdims=None)
```


Refer to `jax.numpy.argmin()` for the full documentation.


------------------------------------------------------------------------


#### argpartition()


Return the indices that partially sort the array.


Usage

``` python
argpartition(kth, axis=-1)
```


Refer to `jax.numpy.argpartition()` for the full documentation.


------------------------------------------------------------------------


#### argsort()


Return the indices that sort the array.


Usage

``` python
argsort(axis=-1, *, kind=None, order=None, stable=True, descending=False)
```


Refer to `jax.numpy.argsort()` for the full documentation.


------------------------------------------------------------------------


#### astype()


Copy the array and cast to a specified dtype.


Usage

``` python
astype(dtype, copy=False, device=None)
```


This is implemented via `jax.lax.convert_element_type()`, which may have slightly different behavior than `numpy.ndarray.astype()` in some cases. In particular, the details of float-to-int and int-to-float casts are implementation dependent.


------------------------------------------------------------------------


#### byteswap()


Swap the bytes of the array elements.


Usage

``` python
byteswap()
```


This switches between a little-endian and big-endian data representation.

## Returns

An array with the same dtype as `self`, with underlying bytes of each entry reversed.

## Examples

``` python
  >>> import jax.numpy as jnp
  >>> x = jnp.arange(5, dtype='int32')
  >>> x
```

Array(\[0, 1, 2, 3, 4\], dtype=int32)

``` python
  >>> x.byteswap()
```

Array(\[ 0, 16777216, 33554432, 50331648, 67108864\], dtype=int32)

When the resulting bytes are viewed as a big-endian dtype (possible in NumPy, but not in JAX) they represent the original values:

``` python
  >>> import numpy as np
  >>> np.array(x.byteswap()).view('>i4')  # view as big-endian
```

array(\[0, 1, 2, 3, 4\], dtype='\>i4')

Calling byteswap twice will return the original array:

``` python
  >>> x.byteswap().byteswap()
```

Array(\[0, 1, 2, 3, 4\], dtype=int32)


------------------------------------------------------------------------


#### choose()


Construct an array choosing from elements of multiple arrays.


Usage

``` python
choose(choices, out=None, mode="raise")
```


Refer to `jax.numpy.choose()` for the full documentation.


------------------------------------------------------------------------


#### clip()


Return an array whose values are limited to a specified range.


Usage

``` python
clip(min=None, max=None)
```


Refer to `jax.numpy.clip()` for full documentation.


------------------------------------------------------------------------


#### compress()


Return selected slices of this array along given axis.


Usage

``` python
compress(condition, axis=None, *, out=None, size=None, fill_value=0)
```


Refer to `jax.numpy.compress()` for full documentation.


------------------------------------------------------------------------


#### conj()


Return the complex conjugate of the array.


Usage

``` python
conj()
```


Refer to `jax.numpy.conj()` for the full documentation.


------------------------------------------------------------------------


#### conjugate()


Return the complex conjugate of the array.


Usage

``` python
conjugate()
```


Refer to `jax.numpy.conjugate()` for the full documentation.


------------------------------------------------------------------------


#### copy()


Return a copy of the array.


Usage

``` python
copy()
```


Refer to `jax.numpy.copy()` for the full documentation.


------------------------------------------------------------------------


#### copy_to_host_async()


Copies an [Array](typing.Array.md#numpyro_forecast.typing.Array) to the host asynchronously.


Usage

``` python
copy_to_host_async()
```


For arrays that live an an accelerator, such as a GPU or a TPU, JAX may cache the value of the array on the host. Normally this happens behind the scenes when the value of an on-device array is requested by the user, but waiting to initiate a device-to-host copy until the value is requested requires that JAX block the caller while waiting for the copy to complete.

[copy_to_host_async](typing.Array.md#numpyro_forecast.typing.Array.copy_to_host_async) requests that JAX populate its on-host cache of an array, but does not wait for the copy to complete. This may speed up a future on-host access to the array's contents.


------------------------------------------------------------------------


#### cumprod()


Return the cumulative product of the array.


Usage

``` python
cumprod(axis=None, dtype=None, out=None)
```


Refer to `jax.numpy.cumprod()` for the full documentation.


------------------------------------------------------------------------


#### cumsum()


Return the cumulative sum of the array.


Usage

``` python
cumsum(axis=None, dtype=None, out=None)
```


Refer to `jax.numpy.cumsum()` for the full documentation.


------------------------------------------------------------------------


#### diagonal()


Return the specified diagonal from the array.


Usage

``` python
diagonal(offset=0, axis1=0, axis2=1)
```


Refer to `jax.numpy.diagonal()` for the full documentation.


------------------------------------------------------------------------


#### dot()


Compute the dot product of two arrays.


Usage

``` python
dot(b, *, precision=None, preferred_element_type=None)
```


Refer to `jax.numpy.dot()` for the full documentation.


------------------------------------------------------------------------


#### flatten()


Flatten array into a 1-dimensional shape.


Usage

``` python
flatten(order="C", *, out_sharding=None)
```


Refer to `jax.numpy.ravel()` for the full documentation.


------------------------------------------------------------------------


#### item()


Copy an element of an array to a standard Python scalar and return it.


Usage

``` python
item(*args)
```


------------------------------------------------------------------------


#### max()


Return the maximum of array elements along a given axis.


Usage

``` python
max(axis=None, out=None, keepdims=False, initial=None, where=None)
```


Refer to `jax.numpy.max()` for the full documentation.


------------------------------------------------------------------------


#### mean()


Return the mean of array elements along a given axis.


Usage

``` python
mean(axis=None, dtype=None, out=None, keepdims=False, *, where=None)
```


Refer to `jax.numpy.mean()` for the full documentation.


------------------------------------------------------------------------


#### min()


Return the minimum of array elements along a given axis.


Usage

``` python
min(axis=None, out=None, keepdims=False, initial=None, where=None)
```


Refer to `jax.numpy.min()` for the full documentation.


------------------------------------------------------------------------


#### nonzero()


Return indices of nonzero elements of an array.


Usage

``` python
nonzero(*, fill_value=None, size=None)
```


Refer to `jax.numpy.nonzero()` for the full documentation.


------------------------------------------------------------------------


#### prod()


Return product of the array elements over a given axis.


Usage

``` python
prod(
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=None,
    promote_integers=True
)
```


Refer to `jax.numpy.prod()` for the full documentation.


------------------------------------------------------------------------


#### ptp()


Return the peak-to-peak range along a given axis.


Usage

``` python
ptp(axis=None, out=None, keepdims=False)
```


Refer to `jax.numpy.ptp()` for the full documentation.


------------------------------------------------------------------------


#### ravel()


Flatten array into a 1-dimensional shape.


Usage

``` python
ravel(order="C", *, out_sharding=None)
```


Refer to `jax.numpy.ravel()` for the full documentation.


------------------------------------------------------------------------


#### repeat()


Construct an array from repeated elements.


Usage

``` python
repeat(repeats, axis=None, *, total_repeat_length=None, out_sharding=None)
```


Refer to `jax.numpy.repeat()` for the full documentation.


------------------------------------------------------------------------


#### reshape()


Returns an array containing the same data with a new shape.


Usage

``` python
reshape(*args, order="C", out_sharding=None)
```


Refer to `jax.numpy.reshape()` for full documentation.


------------------------------------------------------------------------


#### round()


Round array elements to a given decimal.


Usage

``` python
round(decimals=0, out=None)
```


Refer to `jax.numpy.round()` for full documentation.


------------------------------------------------------------------------


#### searchsorted()


Perform a binary search within a sorted array.


Usage

``` python
searchsorted(v, side="left", sorter=None, *, method="scan")
```


Refer to `jax.numpy.searchsorted()` for full documentation.


------------------------------------------------------------------------


#### sort()


Return a sorted copy of an array.


Usage

``` python
sort(axis=-1, *, kind=None, order=None, stable=True, descending=False)
```


Refer to `jax.numpy.sort()` for full documentation.


------------------------------------------------------------------------


#### squeeze()


Remove one or more length-1 axes from array.


Usage

``` python
squeeze(axis=None)
```


Refer to `jax.numpy.squeeze()` for full documentation.


------------------------------------------------------------------------


#### std()


Compute the standard deviation along a given axis.


Usage

``` python
std(
    axis=None,
    dtype=None,
    out=None,
    ddof=0,
    keepdims=False,
    *,
    where=None,
    correction=None
)
```


Refer to `jax.numpy.std()` for full documentation.


------------------------------------------------------------------------


#### sum()


Sum of the elements of the array over a given axis.


Usage

``` python
sum(
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=None,
    promote_integers=True
)
```


Refer to `jax.numpy.sum()` for full documentation.


------------------------------------------------------------------------


#### swapaxes()


Swap two axes of an array.


Usage

``` python
swapaxes(axis1, axis2)
```


Refer to `jax.numpy.swapaxes()` for full documentation.


------------------------------------------------------------------------


#### take()


Take elements from an array.


Usage

``` python
take(
    indices,
    axis=None,
    out=None,
    mode=None,
    unique_indices=False,
    indices_are_sorted=False,
    fill_value=None
)
```


Refer to `jax.numpy.take()` for full documentation.


------------------------------------------------------------------------


#### to_device()


Return a copy of the array on the specified device


Usage

``` python
to_device(device, *, stream=None)
```


## Parameters

- **device** -- `~jax.Device` or `~jax.sharding.Sharding` to which the created array will be committed.
- **stream** -- not implemented, passing a non-None value will lead to an error. Returns: copy of array placed on the specified device or devices.


------------------------------------------------------------------------


#### trace()


Return the sum along the diagonal.


Usage

``` python
trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)
```


Refer to `jax.numpy.trace()` for full documentation.


------------------------------------------------------------------------


#### transpose()


Returns a copy of the array with axes transposed.


Usage

``` python
transpose(*args)
```


Refer to `jax.numpy.transpose()` for full documentation.


------------------------------------------------------------------------


#### var()


Compute the variance along a given axis.


Usage

``` python
var(
    axis=None,
    dtype=None,
    out=None,
    ddof=0,
    keepdims=False,
    *,
    where=None,
    correction=None
)
```


Refer to `jax.numpy.var()` for full documentation.


------------------------------------------------------------------------


#### view()


Return a bitwise copy of the array, viewed as a new dtype.


Usage

``` python
view(dtype=None, type=None)
```


This is fuller-featured wrapper around `jax.lax.bitcast_convert_type()`.

If the source and target dtype have the same bitwidth, the result has the same shape as the input array. If the bitwidth of the target dtype is different from the source, the size of the last axis of the result is adjusted accordingly.

``` python
>>> jnp.zeros([1,2,3], dtype=jnp.int16).view(jnp.int8).shape
```

(1, 2, 6)

``` python
>>> jnp.zeros([1,2,4], dtype=jnp.int8).view(jnp.int16).shape
```

(1, 2, 2)

Conversions involving booleans are not well-defined in all situations. With regards to the shape of result as explained above, booleans are treated as having a bitwidth of 8. However, when converting to a boolean array, the input should only contain 0 or 1 bytes. Otherwise, results may be unpredictable or may change depending on how the result is used.

This conversion is guaranteed and safe::

``` python
  >>> jnp.array([1, 0, 1], dtype=jnp.int8).view(jnp.bool_)
```

Array(\[ True, False, True\], dtype=bool)

However, there are no guarantees about the results of any expression involving a view such as this: `jnp.array([1, 2, 3], dtype=jnp.int8).view(jnp.bool_)`. In particular, the results may change between JAX releases and depending on the platform. To safely convert such an array to a boolean array, compare it with `0`::

``` python
  >>> jnp.array([1, 2, 0], dtype=jnp.int8) != 0
```

Array(\[ True, True, False\], dtype=bool)

## Parameters

- **dtype** -- An optional output dtype. If not specified, the output dtype is the same as the input dtype.
- **type** -- Not implemented; accepted for NumPy compatibility. Returns: The array, viewed as the new dtype. Unlike NumPy, the array may or may not be a copy of the input array.
