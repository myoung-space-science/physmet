import numbers
import operator
import typing

import nonstring
import numpy
import numpy.typing

from .. import numeric
from .. import quantity
from . import _functions
from . import _types


T = typing.TypeVar('T')


class DimensionTypeError(Exception):
    """Invalid type for operation on dimensions."""


def dimensions_eq(a: _types.Dimensions, b):
    """Called for a == b."""
    if isinstance(b, _types.Dimensions):
        return a._names == b._names
    if isinstance(b, str) and len(a) == 1:
        return a._names[0] == b
    try:
        truth = a._names == tuple(b)
    except TypeError:
        return False
    return truth


def dimensions_lt(a: _types.Dimensions, b):
    """Called for a < b."""
    return dimensions_compare(a, b, operator.lt)


def dimensions_le(a: _types.Dimensions, b):
    """Called for a <= b."""
    return dimensions_compare(a, b, operator.le)


def dimensions_gt(a: _types.Dimensions, b):
    """Called for a > b."""
    return dimensions_compare(a, b, operator.gt)


def dimensions_ge(a: _types.Dimensions, b):
    """Called for a >= b."""
    return dimensions_compare(a, b, operator.ge)


def dimensions_compare(
    a: _types.Dimensions,
    b,
    f: typing.Callable[..., T],
) -> T:
    """Implement a binary comparison between `a` and `b`."""
    these = set(a._names)
    those = set(b._names if isinstance(b, _types.Dimensions) else b)
    return f(these, those)


def dimensions_replace(a: _types.Dimensions, old: str, new: str):
    """Replace a single dimension."""
    return dimensions_factory(*[new if n == old else n for n in a._names])


def dimensions_insert(
    a: _types.Dimensions,
    name: str,
    index: typing.SupportsIndex=None,
) -> _types.Dimensions:
    """Insert the named dimension at `index`."""
    names = list(a._names)
    if index is not None:
        names.insert(index, name)
    else:
        names.append(name)
    return dimensions_factory(*names)


def dimensions_and(a: _types.Dimensions, b):
    """Called for a & b."""
    try:
        those = _get_dimensions_operand(b)
    except DimensionTypeError:
        return NotImplemented
    full = list(a._names) + those
    common = set(a._names) & set(those)
    return dimensions_factory(*[n for n in full if n in common])


def dimensions_rand(a: _types.Dimensions, b):
    """Called for b & a."""
    try:
        those = _get_dimensions_operand(b)
    except DimensionTypeError:
        return NotImplemented
    full = those + list(a._names)
    common = set(a._names) & set(those)
    return dimensions_factory(*[n for n in full if n in common])


def dimensions_sub(a: _types.Dimensions, b):
    """Called for a - b."""
    try:
        those = _get_dimensions_operand(b)
    except DimensionTypeError:
        return NotImplemented
    return dimensions_factory(*[n for n in a if n not in those])


def dimensions_rsub(a: _types.Dimensions, b):
    """Called for b - a."""
    if isinstance(b, _types.Dimensions):
        return dimensions_factory(*[n for n in b if n not in a])
    raise NotImplemented


def dimensions_or(a: _types.Dimensions, b):
    """Called for a | b."""
    try:
        those = _get_dimensions_operand(b)
    except DimensionTypeError:
        return NotImplemented
    these = list(a._names)
    try:
        merged = nonstring.merge(these, those)
    except nonstring.MergeError as err:
        errmsg = str(err).replace('entries', 'dimensions')
        raise ValueError(errmsg) from err
    return dimensions_factory(merged)


def dimensions_ror(a: _types.Dimensions, b):
    """Called for b | a."""
    try:
        those = _get_dimensions_operand(b)
    except DimensionTypeError:
        return NotImplemented
    these = list(a._names)
    try:
        merged = nonstring.merge(those, these)
    except nonstring.MergeError as err:
        errmsg = str(err).replace('entries', 'dimensions')
        raise ValueError(errmsg) from err
    return dimensions_factory(merged)


def dimensions_xor(a: _types.Dimensions, b):
    """Called for a ^ b."""
    try:
        those = _get_dimensions_operand(b)
    except DimensionTypeError:
        return NotImplemented
    return (a - those) | (those - a)


def dimensions_rxor(a: _types.Dimensions, b):
    """Called for b ^ a."""
    try:
        those = _get_dimensions_operand(b)
    except DimensionTypeError:
        return NotImplemented
    return (those - a) | (a - those)


def _get_dimensions_operand(arg) -> typing.List[str]:
    """Return an appropriate operand for binary operations."""
    if isinstance(arg, _types.Dimensions):
        return list(arg._names)
    if isinstance(arg, str):
        return [arg]
    try:
        iter(arg)
    except TypeError:
        raise DimensionTypeError(arg)
    if all(isinstance(i, str) for i in arg):
        return list(arg)
    raise DimensionTypeError(arg)


def dimensions_copy(a: _types.Dimensions):
    """Create a copy of this instance."""
    return dimensions_factory(*a._names)


@typing.overload
def dimensions_permute(
    a: _types.Dimensions,
    order: typing.Iterable[typing.SupportsIndex],
) -> _types.Dimensions: ...

@typing.overload
def dimensions_permute(
    a: _types.Dimensions,
    *order: typing.SupportsIndex,
) -> _types.Dimensions: ...

def dimensions_permute(a: _types.Dimensions, *args, **kwargs):
    """Reorder these dimensions."""
    if args and 'order' in kwargs:
        raise TypeError("Too many arguments") from None
    if args:
        if len(args) == 1:
            return _permute(a, args[0])
        return _permute(a, args)
    if 'order' in kwargs:
        return _permute(a, kwargs['order'])
    raise TypeError("No arguments") from None


def _permute(
    a: _types.Dimensions,
    axes: typing.Iterable[typing.SupportsIndex],
) -> _types.Dimensions:
    """Helper for `~permute`."""
    try:
        names = [a[axis] for axis in axes]
    except (IndexError, TypeError) as err:
        raise ValueError(
            "Cannot interpret all values as indices"
        ) from err
    a.mismatch(names)
    return dimensions_factory(names)


def dimensions_mismatch(
    a: typing.Union[_types.Dimensions, typing.Iterable[str]],
    b: typing.Iterable[str],
) -> None:
    """Raise an exception if `a < b` or `a > b`.

    This is a convenience method to checking whether `a == b` in a set-wise
    sense, and issuing a descriptive message depending upon whether `a` is a
    subset of `b` or vice versa. Note that it can also be used as a static
    method to compare two iterables of strings.

    Parameters
    ----------
    a : dimensions or iterable of strings
        The reference set of named dimensions
    b : iterable of strings
        The object to compare to this collection (or to the first argument, when
        called as a static method).

    Returns
    -------
    `None`
        The two collections are equal in a set-wise sense.

    Raises
    ------
    `ValueError`
        The two collections are unequal in a set-wise sense.
    """
    x = set(a)
    y = set(b)
    if x < y:
        raise ValueError("Not enough dimensions") from None
    if x > y:
        diff = x - y
        dims = nonstring.join(diff, 'or', quoted=True)
        raise ValueError(
            f"Original dimensions do not include {dims}"
        ) from None


def dimensions_hash(a: _types.Dimensions):
    """Called for hash(a)."""
    return hash(a._names)


def dimensions_len(a: _types.Dimensions) -> int:
    """Called for len(a)."""
    return len(a._names)


def dimensions_getitem(a: _types.Dimensions, i: typing.SupportsIndex, /):
    """Called for a[i]."""
    return a._names[i]


def dimensions_str(a: _types.Dimensions) -> str:
    return f"{{{', '.join(repr(name) for name in a._names)}}}"


DIMENSIONS_OPERATORS = {
    '__eq__': dimensions_eq,
    '__lt__': dimensions_lt,
    '__le__': dimensions_le,
    '__gt__': dimensions_gt,
    '__ge__': dimensions_ge,
    'replace': dimensions_replace,
    'insert': dimensions_insert,
    '__and__': dimensions_and,
    '__rand__': dimensions_rand,
    '__sub__': dimensions_sub,
    '__rsub__': dimensions_rsub,
    '__or__': dimensions_or,
    '__ror__': dimensions_ror,
    '__xor__': dimensions_xor,
    '__rxor__': dimensions_rxor,
    'copy': dimensions_copy,
    'permute': dimensions_permute,
    'mismatch': dimensions_mismatch,
    '__hash__': dimensions_hash,
    '__len__': dimensions_len,
    '__getitem__': dimensions_getitem,
    '__str__': dimensions_str,
}


def dimensions_factory(*args):
    """Create a new set of data dimensions."""
    Type = type('Dimensions', (_types.Dimensions,), DIMENSIONS_OPERATORS)
    return Type(dimensions_args(*args))


def dimensions_args(*args):
    """Helper for `dimensions_factory`."""
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, numbers.Real):
            return tuple(f'x{i}' for i in range(arg))
    unwrapped = nonstring.unwrap(args, newtype=tuple)
    unique = nonstring.unique(unwrapped, separable=True)
    return _compute_names(unique)


def _compute_names(args: typing.Iterable[str]) -> typing.Tuple[str]:
    """Determine the instance names from arguments."""
    if isinstance(args, _types.Dimensions):
        return args._names
    if all(isinstance(arg, str) for arg in args):
        return tuple(args)
    raise TypeError(
        f"Unable to compute dimension names from {args!r}"
    ) from None


def array_dunder_eq(a: _types.Array, b):
    """Called for self == other."""
    if isinstance(b, _types.Array):
        return (
            numpy.array_equal(a.array, b.array)
            and
            a.dimensions == b.dimensions
        )
    return False


def array_dunder_ne(a, b):
    """Called for self != other."""
    return not (a == b)


def array_dunder_abs(a):
    """Compute abs(a)."""
    return array_unary(operator.abs, a)


def array_dunder_pos(a):
    """Compute +a."""
    return array_unary(operator.pos, a)


def array_dunder_neg(a):
    """Compute -a."""
    return array_unary(operator.neg, a)


def array_dunder_round(a, /, ndigits: int=None):
    """Compute round(a[, ndigits])."""
    return array_unary(round, a, ndigits=ndigits)


def array_dunder_lt(a, b):
    """Compute a < b."""
    return array_comparative(operator.lt, a, b)


def array_dunder_le(a, b):
    """Compute a <= b."""
    return array_comparative(operator.le, a, b)


def array_dunder_gt(a, b):
    """Compute a > b."""
    return array_comparative(operator.gt, a, b)


def array_dunder_ge(a, b):
    """Compute a >= b."""
    return array_comparative(operator.ge, a, b)


def array_dunder_add(a, b):
    """Compute a + b."""
    try:
        result = array_additive(operator.add, a, b)
    except ValueError as err:
        raise ValueError("Cannot add arrays") from err
    return result


def array_dunder_radd(a, b):
    """Compute b + a."""
    return array_dunder_add(b, a)


def array_dunder_sub(a, b):
    """Compute a - b."""
    try:
        result = array_additive(operator.sub, a, b)
    except ValueError as err:
        raise ValueError("Cannot subtract arrays") from err
    return result


def array_dunder_rsub(a, b):
    """Compute b - a."""
    return array_dunder_sub(b, a)


@_types.Array.implements(numpy.multiply)
def array_dunder_mul(a, b):
    """Compute a * b."""
    return array_multiplicative(operator.mul, a, b)


def array_dunder_rmul(a, b):
    """Compute b * a."""
    return array_dunder_mul(b, a)


@_types.Array.implements(numpy.true_divide)
def array_dunder_truediv(a, b):
    """Compute a / b."""
    return array_multiplicative(operator.truediv, a, b)


def array_dunder_rtruediv(a, b):
    """Compute b / a."""
    return array_dunder_truediv(b, a)


@_types.Array.implements(numpy.floor_divide)
def array_dunder_floordiv(a, b):
    """Compute a // b."""
    return array_multiplicative(operator.floordiv, a, b)


def array_dunder_rfloordiv(a, b):
    """Compute b // a."""
    return array_dunder_floordiv(b, a)


@_types.Array.implements(numpy.mod)
def array_dunder_mod(a, b):
    """Compute a % b."""
    return array_multiplicative(operator.mod, a, b)


def array_dunder_rmod(a, b):
    """Compute b % a."""
    return array_dunder_mod(b, a)


@_types.Array.implements(numpy.power)
def array_dunder_pow(a, b, mod: int=None):
    """Compute a ** b."""
    if isinstance(a, _types.Array) and isinstance(b, _types.Array):
        if a.dimensions != b.dimensions:
            raise ValueError(
                "Cannot compute a ** b between two arrays"
                " with different dimensions"
            ) from None
    f = pow
    if isinstance(a, (numbers.Number, numpy.ndarray)):
        return f(a, b.array)
    return array_multiplicative(f, a, b, mod=mod)


def array_dunder_rpow(a, b):
    """Compute b ** a."""
    return array_dunder_pow(b, a)


def array_unary(f, a, /):
    """Implement a unary arithmetic operation on array quantities."""
    if isinstance(a, _types.Array):
        return array_factory(f(a.array), dimensions=a.dimensions)
    raise TypeError(a)


def array_comparative(f, a, b, /):
    """Implement a comparative operation on array quantities."""
    if isinstance(a, _types.Array) and isinstance(b, _types.Array):
        if not a.dimensions == b.dimensions:
            raise ValueError(
                "Cannot compare arrays with different dimensions"
            ) from None
        return f(a.array, b.array)
    return f(a.array, b)


def array_additive(f, a, b, /):
    """Implement an additive operation on array quantities."""
    if isinstance(a, _types.Array) and isinstance(b, _types.Array):
        if not a.dimensions == b.dimensions:
            raise ValueError("_types.Arrays have different dimensions") from None
        return array_factory(f(a.array, b.array), dimensions=a.dimensions)
    if isinstance(a, _types.Array):
        return f(a.array, b)
    if isinstance(b, _types.Array):
        return f(a, b.array)
    raise TypeError(a, b)


def array_multiplicative(f, a, b, /, **kwargs):
    """Implement a multiplicative operation on array quantities."""
    try:
        operands = _get_array_operands(a, b)
    except TypeError as err:
        raise TypeError(a, b) from err
    r = f(*operands, **kwargs)
    if isinstance(a, _types.Array) and isinstance(b, _types.Array):
        return array_factory(r, a.dimensions | b.dimensions)
    if isinstance(a, _types.Array):
        return array_factory(r, a.dimensions)
    if isinstance(b, _types.Array):
        return array_factory(r, b.dimensions)
    raise TypeError(a, b)


def _get_array_operands(a, b):
    """Compute appropriate operands for a binary operation."""
    if isinstance(a, _types.Array) and isinstance(b, _types.Array):
        return _functions.remesh(a, b)
    return _get_array_operand(a), _get_array_operand(b)


def _get_array_operand(x):
    """Determine the appropriate operand for a mixed-type operation.
    
    This is a helper function for `_get_operands`. It assumes that
    `_get_operands` has already handled the case in which both arguments are
    data interfaces.
    """
    if isinstance(x, _types.Array):
        return x.array
    if isinstance(x, numeric.Object):
        return x.data
    if isinstance(x, (numbers.Real, numpy.ndarray)):
        return x
    raise TypeError(x)


@_types.Array.implements(numpy.array_equal)
def array_equal(
    a: _types.Array[numeric.RealValueType],
    b: _types.Array[numeric.RealValueType],
) -> bool:
    """Called for numpy.array_equal(a, b)."""
    return numpy.array_equal(numpy.array(a), numpy.array(b))


@_types.Array.implements(numpy.squeeze)
def squeeze(x: _types.Array[numeric.RealValueType], **kwargs):
    """Called for numpy.squeeze(x)."""
    data = numpy.squeeze(x.array, **kwargs)
    if data.ndim == 0:
        return float(data)
    dimensions = [d for d, n in x.shapemap.items() if n != 1]
    return array_factory(data, dimensions=dimensions)


@_types.Array.implements(numpy.mean)
def mean(x: _types.Array[numeric.RealValueType], **kwargs):
    """Compute the mean of the underlying array."""
    data = x.array.mean(**kwargs)
    axis = kwargs.get('axis')
    if axis is None:
        return data
    ax = axis % x.ndim
    dimensions = [d for d in x.dimensions if x.dimensions.index(d) != ax]
    return array_factory(data, dimensions=dimensions)


@_types.Array.implements(numpy.sum)
def sum(x: _types.Array[numeric.RealValueType], **kwargs):
    """Compute the sum of the underlying array."""
    data = x.array.sum(**kwargs)
    axis = kwargs.get('axis')
    if axis is None:
        return data
    if 'keepdims' in kwargs:
        return array_factory(data, dimensions=x.dimensions)
    ax = x.ndim + axis if axis < 0 else axis
    dimensions = [d for d in x.dimensions if x.dimensions.index(d) != ax]
    return array_factory(data, dimensions=dimensions)


@_types.Array.implements(numpy.cumsum)
def cumsum(x: _types.Array[numeric.RealValueType], **kwargs):
    """Compute the cumulative sum of the underlying array."""
    data = x.array.cumsum(**kwargs)
    if kwargs.get('axis') is None:
        return data
    return array_factory(data, dimensions=x.dimensions)


@_types.Array.implements(numpy.transpose)
def transpose(x: _types.Array[numeric.RealValueType], **kwargs):
    """Compute the transpose of an array and dimensions."""
    data = x.array.transpose()
    axes = kwargs.get('axes')
    if axes is None:
        dimensions = x.dimensions[::-1]
        return array_factory(data, dimensions=dimensions)
    dimensions = [x.dimensions[i] for i in tuple(axes)]
    return array_factory(data, dimensions=dimensions)


@_types.Array.implements(numpy.gradient)
def gradient(x: _types.Array[numeric.RealValueType], *args, **kwargs):
    """Compute the gradient of an array."""
    if not args:
        return _apply_gradient(x.dimensions, x.array, **kwargs)
    diffs = []
    for arg in args:
        if isinstance(arg, _types.Array):
            diffs.append(arg.array)
        elif isinstance(arg, numbers.Real):
            diffs.append(float(arg))
        else:
            diffs.append(numpy.array(arg))
    return _apply_gradient(x.dimensions, x.array, *diffs, **kwargs)


def _apply_gradient(
    dimensions,
    *args,
    **kwargs
) -> typing.List[_types.Array]:
    """Helper for `gradient`."""
    gradient = numpy.gradient(*args, **kwargs)
    if isinstance(gradient, list):
        return [
            array_factory(array, dimensions=dimensions)
            for array in gradient
        ]
    return array_factory(gradient, dimensions=dimensions)


@_types.Array.implements(numpy.trapz)
def trapz(x: _types.Array[numeric.RealValueType], *args, **kwargs):
    """Integrate the array via the composite trapezoidal rule."""
    data = numpy.trapz(x.array, *args, **kwargs)
    axis = kwargs.get('axis')
    if axis is None:
        dimensions = x.dimensions[:-1]
    else:
        ax = axis % x.ndim
        dimensions = [d for d in x.dimensions if x.dimensions.index(d) != ax]
    return array_factory(data, dimensions=dimensions)


def array_dunder_contains(a: _types.Array, __x) -> bool:
    """Called for x in self."""
    return __x in a._data or __x in a.array


def array_dunder_iter(a: _types.Array):
    """Called for iter(self)."""
    return iter(a._data)


def array_dunder_len(a: _types.Array) -> int:
    """Called for len(self)."""
    return len(a._data)


IndexLike = typing.Union[
    typing.SupportsIndex,
    typing.Tuple[typing.SupportsIndex],
    slice,
]


def array_dunder_getitem(a: _types.Array, args: IndexLike):
    """Called for subscription."""
    x = [args] if isinstance(args, typing.SupportsIndex) else args
    return array_factory(
        _functions.subarray(a, x),
        dimensions=a.dimensions,
    )


def array_dunder_str(a: _types.Array) -> str:
    """A simplified representation of this object."""
    return _as_string(a)


def array_dunder_repr(a: _types.Array) -> str:
    """An unambiguous representation of this object."""
    prefix = f'{a.__class__.__qualname__}('
    suffix = ')'
    return _as_string(a, prefix=prefix, suffix=suffix)


def _as_string(a: _types.Array, prefix: str='', suffix: str='') -> str:
    """Create a string representation of this object."""
    data = _format_data(a, prefix=prefix, suffix=suffix)
    dimensions = f"dimensions={a.dimensions}"
    return f"{prefix}{data}, {dimensions}{suffix}"


def _format_data(a: _types.Array, **kwargs):
    """Create an appropriate string to represent the data."""
    if not isinstance(a._data, numpy.ndarray):
        return str(type(a._data))
    try:
        signed = '+' if any(i < 0 for i in a.array.flat) else '-'
    except TypeError:
        # If we get here, it's probably because some values are strings
        signed = '-'
    return numpy.array2string(
        a.array,
        threshold=4,
        edgeitems=2,
        separator=', ',
        sign=signed,
        precision=3,
        floatmode='maxprec_equal',
        **kwargs
    )


def array_numpy_array(a: _types.Array, *args, **kwargs) -> numpy.ndarray:
    """Support casting to `numpy` array types.
    
    Notes
    -----
    This will retrieve the underlying array before applying `*args` and
    `**kwargs`, in order to avoid a `TypeError` when using
    `netCDF4.Dataset`. See
    https://github.com/mcgibbon/python-examples/blob/master/scripts/file-io/load_netCDF4_full.py
    """
    array = a.array
    return numpy.asarray(array, *args, **kwargs)


def array_remesh(a: _types.Array, b, invert: bool=False):
    """Construct array meshes for broadcasting.

    This method calls `~remesh` to reshape the arrays of both data
    interfaces in a way that is consistent with their combined dimensions.

    Parameters
    ----------
    a, b : `~Array`
        The data interfaces to mutually reshape.
    invert : bool, default=false
        If true, reverse the order of `self` and `other`.
    """
    if invert:
        return _functions.remesh(b, a)
    return _functions.remesh(a, b)


def array_has(a: _types.Array, value: numbers.Real) -> bool:
    """True if `value` is in this sequence's data array.

    This method uses `~data.isclose` to test whether `value` is
    equal, or very close, to a value in this object's real-valued data.

    Parameters
    ----------
    value : real
        The value for which to search.
    """
    return _functions.isclose(a, value)


@typing.overload
def array_transpose(
    a: _types.Array,
    axes: typing.Iterable[typing.Union[str, typing.SupportsIndex]],
    /,
) -> typing.Self: ...

@typing.overload
def array_transpose(
    a: _types.Array,
    *axes: typing.Union[str, typing.SupportsIndex],
) -> typing.Self: ...

def array_transpose(a: _types.Array, *args):
    """Transpose this array."""
    if not args:
        return a
    if all(isinstance(target, str) for target in args):
        return _transpose(a, args)
    if len(args) == 1:
        arg = args[0]
        if all(isinstance(target, str) for target in arg):
            return _transpose(a, arg)
    data = a.array.transpose(*args)
    new = a.dimensions.permute(*args)
    return array_factory(data, dimensions=new)


def _transpose(a: _types.Array, args: typing.Iterable[str]):
    """Helper for `~transpose`."""
    a.dimensions.mismatch(args)
    axes = [a.dimensions.index(arg) for arg in args]
    data = a.array.transpose(axes)
    return array_factory(data, dimensions=args)


def _load_array(
    a: _types.Array,
    index: typing.Optional[IndexLike]=None,
) -> numpy.typing.NDArray:
    """Load the array from disk or memory.

    This method is intended for internal use. Users should subscript
    instances via bracket notation (e.g., `array[:, 2]`).

    If `index` is `None` or an empty iterable, this method will produce the
    entire array. Otherwise, it will create the requested subarray from the
    internal `_data` attribute. It will always attempt to use a cached
    version of the full array before loading from disk. Therefore,
    repeatedly calling this method should not create a performance
    bottleneck.

    The specific algorithm is as follows:

    - If `index` is null (i.e., `None` or an empty iterable object, but not
        literal 0), the caller wants the full array.
    
        - If we already have the full array, return it.
        - Else, read it, save it, and return it.

    - Else, if `index` is not null, the caller wants a subarray.

        - If we already have the full array, subscript and return it.
        - Else, continue

    - Else, read and subscript the array, and return the subarray.

    The reasoning behind this algorithm is as follows: If we need to load
    the full array at any point, we may as well save it because subscripting
    an in-memory `numpy.ndarray` is much faster than re-reading from disk
    for large arrays. However, we should avoid reading in the full array if
    the caller only wants a small portion of it. We don't cache these
    subarrays because reusing a subarray is only meaningful if the indices
    haven't changed. Furthermore, accessing a subarray via bracket syntax
    creates a new object, at which point the subarray becomes the new
    object's full array.
    """
    if quantity.isnull(index):
        if a._array is not None:
            return a._array
        array = _read_array(a)
        a._array = array
        return array
    if a._array is not None:
        idx = numpy.index_exp[index]
        return a._array[idx]
    return _read_array(a, index)


def _read_array(
    a: _types.Array,
    index: typing.Optional[IndexLike]=None,
) -> numpy.typing.NDArray:
    """Read the array data from disk.
    
    If `index` is null in the sense defined by `~quantity.isnull` this method
    will load and return the full array. If `index` is not null, this method
    will first attempt to subscript the internal `_data` attribute before
    converting it to an array and returning it. If it catches either a
    `TypeError` or an `IndexError`, it will create the full array before
    subscripting and returning it. The former may occur if the internal `_data`
    attribute is a sequence type like `list`, `tuple`, or `range`; the latter
    may occur when attempting to subscript certain array-like objects (e.g.,
    `netCDF4._netCDF4.Variable`) despite passing a valid `numpy` index
    expression.
    """
    if not quantity.isnull(index):
        idx = numpy.index_exp[index]
        try:
            return numpy.asarray(a._data[idx])
        except (TypeError, IndexError):
            return numpy.asarray(a._data)[idx]
    return numpy.asarray(a._data)


ARRAY_OPERATORS = {
    '__eq__': array_dunder_eq,
    '__ne__': array_dunder_ne,
    '__lt__': array_dunder_lt,
    '__le__': array_dunder_le,
    '__gt__': array_dunder_gt,
    '__ge__': array_dunder_ge,
    '__abs__': array_dunder_abs,
    '__neg__': array_dunder_neg,
    '__pos__': array_dunder_pos,
    '__add__': array_dunder_add,
    '__radd__': array_dunder_radd,
    '__sub__': array_dunder_sub,
    '__rsub__': array_dunder_rsub,
    '__mul__': array_dunder_mul,
    '__rmul__': array_dunder_rmul,
    '__truediv__': array_dunder_truediv,
    '__rtruediv__': array_dunder_rtruediv,
    '__floordiv__': array_dunder_floordiv,
    '__rfloordiv__': array_dunder_rfloordiv,
    '__mod__': array_dunder_mod,
    '__rmod__': array_dunder_rmod,
    '__pow__': array_dunder_pow,
    '__rpow__': array_dunder_rpow,
    '__contains__': array_dunder_contains,
    '__iter__': array_dunder_iter,
    '__len__': array_dunder_len,
    '__getitem__': array_dunder_getitem,
    '__array__': array_numpy_array,
    'remesh': array_remesh,
    'has': array_has,
    'transpose': array_transpose,
    '_load_array': _load_array,
    '_read_array': _read_array,
    '__str__': array_dunder_str,
    '__repr__': array_dunder_repr,
}


def array_factory(*args, **kwargs):
    Type = type('Array', (_types.Array,), ARRAY_OPERATORS)
    a, d = array_args(*args, **kwargs)
    return Type(a, dimensions_factory(d))


def array_args(*args, **kwargs):
    if len(args) == 1:
        x = args[0]
        if isinstance(x, _types.Array):
            if kwargs:
                raise TypeError(
                    "Cannot change attributes when copying"
                    " an existing quantity"
                ) from None
            return x.array, x.dimensions
    array, dimensions = _parse_data_args(*args, **kwargs)
    if len(dimensions) != array.ndim:
        raise ValueError(
            f"Number of named dimensions ({len(dimensions)})"
            f" must equal number of array dimensions ({array.ndim})"
        ) from None
    return array, dimensions


def _parse_data_args(*args, dimensions=None):
    """Parse arguments for initializing a data interface."""
    array = _init_array(args[0])
    if len(args) > 1:
        if dimensions:
            raise TypeError(
                "Got positional and keyword arguments for dimensions"
            ) from None
        return array, dimensions_factory(args[1:])
    if dimensions is None:
        return array, dimensions_factory(array.ndim)
    return array, dimensions_factory(dimensions)


def _init_array(arg) -> numeric.Array:
    """Get an array-like data interface.
    
    This method ensures that calling methods get an object that has certain
    array-like properties, without requiring that `arg` be an instance of
    `numpy.ndarray` and without converting `arg` to a `numpy.ndarray` unless
    necessary.
    """
    data = _get_arraylike(arg)
    if all(hasattr(data, name) for name in ('shape', 'ndim', 'size')):
        return data
    return numpy.array(data)


def _get_arraylike(arg):
    """Extract an array-like interface from `arg`."""
    if isinstance(arg, numeric.Quantity):
        return arg.data
    if isinstance(arg, (list, tuple, set)):
        return numpy.array(arg)
    return arg


