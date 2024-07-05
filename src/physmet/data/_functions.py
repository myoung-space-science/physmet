import contextlib
import numbers
import typing

import nonstring
import numpy
import numpy.typing

from .. import indexer
from .. import numeric
from .. import quantity
from ._types import Array


def isequal(a, b):
    """True if `a` and `b` have equal numeric data.

    This is a convenience function that allows the caller to test whether two
    objects are numerically equivalent, even if they aren't strictly equal
    (perhaps because they have unequal metadata).
    """
    x = a.data if isinstance(a, numeric.Quantity) else a
    y = b.data if isinstance(b, numeric.Quantity) else b
    return numpy.array_equal(x, y)


def isclose(a: numeric.Quantity, b: numbers.Real) -> bool:
    """True if `b` is close to a value in `a`'s numeric data.

    This function is similar to (and, if fact, uses) `numpy.isclose`. The
    primary distinctions are that this function compares a single value to the
    real-valued data of a variable object and returns a single boolean value.
    
    Parameters
    ----------
    a : numeric quantity
        The object that may contain `b`.

    b : real number
        The value for which to search.

    Notes
    -----
    - This function exists to handle cases in which floating-point arithmetic
      has caused a numeric operation to return an imprecise result, especially
      for small numbers (e.g., certain unit conversions). It will first test for
      standard containment via `in` before attempting to determine if `b` is
      close enough, within a very strict tolerance, to any member of `a`.
    """
    data = a.data
    try:
        iter(data)
    except TypeError:
        return a == b or numpy.isclose(b, data, atol=0.0)
    if b in data:
        return True
    if b < numpy.min(data) or b > numpy.max(data):
        return False
    return numpy.any([numpy.isclose(b, data, atol=0.0)])


@typing.overload
def hasdtype(
    a: typing.Union[numeric.Quantity, numpy.typing.ArrayLike],
    t: numpy.typing.DTypeLike,
    /,
): ...

@typing.overload
def hasdtype(
    a: typing.Union[numeric.Quantity, numpy.typing.ArrayLike],
    t: typing.Tuple[numpy.typing.DTypeLike],
    /,
): ...

def hasdtype(a, t, /):
    """True if `a` has one of the given data types.
    
    This function wraps `numpy.issubdtype` to allow the caller to pass more than
    one data type at a time, similar to the behavior of the built-in functions
    `isinstance` and `issubclass`.

    Parameters
    ----------
    a : numeric quantity or array-like
        The object to check.
    t : dtype-like or tuple of dtype-like
        One or more objects that can be interpreted as `numpy` data types.

    Notes
    -----
    - If `a` is a numeric quantity, this function will operate on `a.data`.
    - If the array-like operand (either `a` or `a.data`, as appropriate) is not
      a `numpy.ndarray`, this function will first convert it to one.
    """
    x = a.data if isinstance(a, numeric.Quantity) else a
    y = x if isinstance(x, numpy.ndarray) else numpy.array(x)
    dtype = y.dtype
    if isinstance(t, tuple):
        return any(numpy.issubdtype(dtype, i) for i in t)
    return numpy.issubdtype(dtype, t)


_NT = typing.TypeVar('_NT', bound=numbers.Complex)


class Nearest(typing.NamedTuple):
    """The result of searching an array for a target value."""

    index: int
    value: _NT


def nearest(
    values: typing.Iterable[_NT],
    target: _NT,
    bound: str=None,
) -> Nearest:
    """Find the value in a collection nearest the target value.
    
    Parameters
    ----------
    values : iterable of numbers
        An iterable collection of numbers to compare to the target value. Must
        support conversion to a `numpy.ndarray`.

    target : number
        A single numerical value for which to search in `values`. Must be
        coercible to the type of `values`.

    bound : {None, 'lower', 'upper'}
        The constraint to apply when finding the nearest value:

        - None: no constraint
        - 'lower': ensure that the nearest value is equal to or greater than the
          target value (in other words, the target value is a lower bound for
          the nearest value)
        - 'upper': ensure that the nearest value is equal to or less than the
          target value (in other words, the target value is an upper bound for
          the nearest value)

    Returns
    -------
    Nearest
        A named tuple with `value` and `index` fields, respectively containing
        the value in `values` closest to `target` (given the constraint set by
        `bound`, if any) and the index of `value` in `values`. If the array
        corresponding to `values` is one-dimensional, `index` will be an
        integer; otherwise, it will be a tuple with one entry for each
        dimension.

    Notes
    -----
    This function is based on the top answer to this StackOverflow question:
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    However, a lower-voted answer (and the comments) has some suggestions for a
    bisection-based method.
    """

    array = numpy.asarray(values)
    index = numpy.abs(array - target).argmin()
    if bound == 'lower':
        try:
            while array[index] < target:
                index += 1
        except IndexError:
            index = -1
    elif bound == 'upper':
        try:
            while array[index] > target:
                index -= 1
        except IndexError:
            index = 0
    if array.ndim > 1:
        index = numpy.unravel_index(index, array.shape)
    return Nearest(index=index, value=array[index])


def isindexlike(x, /):
    """True if the input can index an axis."""
    if isinstance(x, numeric.Indexer):
        return True
    for f in (indexer.value, indexer.sequence):
        with contextlib.suppress(TypeError, IndexError):
            f(x)
            return True
    return False


def ismeasurable(x, /):
    """True if the user can expect to be able to measure the input.

    A measurable object may be:
    
    - an object that satisfies the `~base.Measurable` protocol
    - a real number
    - an iterable of real numbers
    - an iterable of real numbers followed by a unit-like object
    - an two-element iterable whose first element is an iterable of real numbers
      and whose second element is a unit-like object
    - an iterable of any of the previous objects.

    Parameters
    ----------
    x
        The candidate measurable object.

    Returns
    -------
    bool
        True if `x` is measurable; false otherwise.

    See Also
    --------
    `~measure`
        Create a `~Measurement` from measurable input.
    """
    args = nonstring.unwrap(x)
    if quantity.isnull(x):
        return False
    if isinstance(args, numeric.Measurable):
        return True
    if isinstance(args, numbers.Real):
        return True
    if not nonstring.isseparable(args):
        return False
    if all(isinstance(arg, numbers.Real) for arg in args):
        return True
    if quantity.isunitlike(args[-1]):
        arg0 = args[0]
        try:
            iter(arg0)
        except TypeError:
            values = args[:-1]
        else:
            values = arg0
        if all(isinstance(value, numbers.Real) for value in values):
            return True
    if all(isinstance(i, (list, tuple)) and ismeasurable(i) for i in args):
        return True
    return False


def subarray(a: Array, args):
    """Extract a subarray."""
    unwrapped = nonstring.unwrap(args)
    indices = indexer.resolve(a.shape, unwrapped)
    loaded = a._load_array(indices)
    array = numpy.array(loaded, ndmin=a.ndim)
    if isinstance(indices, numpy.ndarray) and indices.dtype == bool:
        return array
    if loaded.ndim != a.ndim:
        with contextlib.suppress(TypeError):
            shape = [
                _get_axis_size(a, i, v)
                for i, v in enumerate(indices)
            ]
            return array.reshape(shape)
    return array


def _get_axis_size(a: Array, i: int, v):
    """Helper for computing shape in `subarray`."""
    if isinstance(v, int):
        return 1
    if isinstance(v, slice):
        return (v.stop or a.shape[i]) - (v.start or 0)
    return a.shape[i]


def remesh(a: Array, b: Array):
    """Construct array meshes for broadcasting.
    
    This function mutually reshapes the internal `numpy` arrays of two data
    interfaces based on their combined dimensions. The result is a pair of
    `numpy` arrays that can broadcast together while preserving a shape
    consistent with the combined dimensions.

    Parameters
    ----------
    a, b : `~numeric.Array`
        The numeric arrays to reshape.

    Returns
    -------
    `tuple` of `numpy.ndarray`
        The reshaped `numpy` array of `a`, followed by that of `b`.

    Raises
    ------
    `TypeError`
        One or both of the arguments was not an instance of `~numeric.Array`.
    `ValueError`
        The original array shapes of `a` and `b` have incompatible shapes and
        dimensions. Note that this case acts like the `ValueError` raised when
        attempting to broadcast two `numpy` arrays with inconsistent shapes.
    """
    if not (isinstance(a, Array) and isinstance(b, Array)):
        raise TypeError(
            f"Both arguments must be instances of {Array!r}"
        ) from None
    dimensions = a.dimensions | b.dimensions
    for d in dimensions:
        adim = a.shapemap.get(d, 1)
        bdim = b.shapemap.get(d, 1)
        if adim > 1 and bdim > 1 and adim != bdim:
            # NOTE: This acts like the `ValueError` raised when attempting
            # to broadcast two `numpy` arrays with inconsistent shapes.
            raise ValueError(
                "Cannot create a mesh from arrays with"
                f" dimensions {a.dimensions} and {b.dimensions}"
                ", and corresponding shapes"
                f" shapes {a.shape} and {b.shape}"
            )
    ndim = len(dimensions)
    shape = tuple(
        # NOTE: This creates the maximal shape in order to handle cases in
        # which the input arrays have matching dimensions but one array is
        # singular. The previous implementation would always overwrite the
        # size of matching dimensions in `a` with those of `b`.
        max(a.shapemap.get(d, 1), b.shapemap.get(d, 1))
        for d in dimensions
    )
    idxmesh = numpy.ix_(*[range(i) for i in shape])
    idxa = tuple(idxmesh[dimensions.index(d)] for d in a.shapemap)
    idxb = tuple(idxmesh[dimensions.index(d)] for d in b.shapemap)
    x = numpy.array(a, ndmin=ndim) if a.size == 1 else a._load_array(idxa)
    y = numpy.array(b, ndmin=ndim) if b.size == 1 else b._load_array(idxb)
    return x, y


