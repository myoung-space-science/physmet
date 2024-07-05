import typing

import numpy
import numpy.typing

from .. import numeric
from .. import quantity

from ._implementations import (
    value_factory,
    sequence_factory,
)
from ._types import (
    Sequence,
    Value,
)

__all__ = [
    "Sequence",
    "Value",
    "sequence",
    "value",
]


@typing.overload
def value(
    x: typing.Union[
        typing.SupportsInt,
        typing.Iterable[typing.SupportsInt],
        numpy.typing.NDArray[numpy.integer],
    ],
    /,
) -> Value:
    """Create an index value from a numeric object.

    Parameters
    ----------
    x
        An object that supports conversion to `int`, an iterable object with a
        single member that supports conversion to `int`, or a one-element
        `numpy` array with integral data type.

    Returns
    -------
    `~Value`

    Raises
    ------
    `TypeError`
        The argument contains more than one element.
    """

@typing.overload
def value(x: numeric.Measurement, /) -> Value:
    """Create an index value from a unitless measurable object.

    Parameters
    ----------
    x
        An object with a `data` attribute and a unit equal to `'1'`.

    Returns
    -------
    `~Value`

    Raises
    -----
    `TypeError`
        The given object is not unitless.
    """

def value(x, /):
    return value_factory(x)


@typing.overload
def sequence(
    x: typing.Union[
        typing.SupportsInt,
        typing.Iterable[typing.SupportsInt],
        numpy.typing.NDArray[numpy.integer],
    ],
    /,
) -> Sequence:
    """Create an index sequence from an iterable with integer-like values.

    Parameters
    ----------
    x
        An object that supports conversion to `int`, an iterable object with
        members that all support conversion to `int`, or a `numpy` array with
        integral data type.

    Returns
    -------
    `~Value`

    Raises
    ------
    `TypeError`
        The given object cannot be coerced into a one-dimensional array.
    """

@typing.overload
def sequence(x: numeric.Measurement, /) -> Sequence:
    """Create an index sequence from a unitless measurable object.

    Parameters
    ----------
    x
        An object with a `data` attribute and a unit equal to `'1'`.

    Returns
    -------
    `~Value`

    Raises
    -----
    `TypeError`
        The given object is not unitless.
    """

def sequence(x, /):
    return sequence_factory(x)


_BUILTIN_TYPES = (int, slice, type(...), type(None))

_INDEX_DTYPES = (
    bool,
    numpy.int8,
    numpy.int16,
    numpy.int32,
    numpy.int64,
    numpy.uint8,
    numpy.uint16,
    numpy.uint32,
    numpy.uint64,
)

def resolve(shape: typing.Sequence[int], args):
    """Convert `args` into array indices based on `shape`.

    This function will first normalize `args` via `~normalize`. If the result
    contains an `Ellipsis` (i.e., `...`), this function will expand the
    `Ellipsis` via `~expand`. Otherwise, it will return the normalized indices.
    """
    normalized = normalize(shape, args)
    if _has_ellipsis(normalized):
        return expand(len(shape), normalized)
    return normalized


def _has_ellipsis(args):
    """Helper for `~resolve`."""
    try:
        iter(args)
    except TypeError:
        return False
    return any(arg is Ellipsis for arg in args)


def normalize(shape: typing.Sequence[int], args):
    """Compute appropriate array indices from `args`.

    If all indices in `args` have a standard form involving slices, an ellipsis,
    or integers, including 

    - `array[:]`
    - `array[...]`
    - `array[i, :]`
    - `array[:, j]`
    - `array[i, j]`

    (where i and j are integers) this function will immediately return them. If
    `args` is a `numpy.ndarray` with boolean type, this method will immediately
    return it. Otherwise, it will extract a sequence of indices that represents
    the original dimensions of the data.
    """
    if _normalized(args):
        return args
    if not isinstance(args, (tuple, list)):
        args = [args]
    expanded = expand_ellipsis(len(shape), *args)
    indices = []
    for i, arg in enumerate(expanded):
        if isinstance(arg, slice):
            indices.append(range(shape[i]))
        elif quantity.isintegral(arg):
            indices.append((arg,))
        else:
            indices.append(arg)
    return numpy.ix_(*list(indices))


def _normalized(args):
    """Helper for `~normalize`."""
    return (
        quantity.hastype(args, _BUILTIN_TYPES, tuple, strict=True)
        or
        isinstance(args, numpy.ndarray) and args.dtype in _INDEX_DTYPES
    )


def expand(ndim: int, indices):
    """Expand `indices` so that they will index `ndim` dimensions."""
    if isinstance(indices, (list, tuple)):
        return expand_ellipsis(ndim, *indices)
    if indices == slice(None):
        return expand_ellipsis(ndim, ...)
    if isinstance(indices, type(...)):
        return expand_ellipsis(ndim, indices)
    if ndim > 1:
        return expand_ellipsis(ndim, indices, ...)
    return (indices,)


def expand_ellipsis(ndim: int, *args: typing.SupportsIndex):
    """Expand an `Ellipsis` into one or more `slice` objects."""
    # If there are more arguments than dimensions, something is wrong. The one
    # exception is when the final argument is an Ellipsis. This allows upstream
    # code to programmatically use `...` to capture 0 or more remaining
    # dimensions. For example, `self.mfp[0, 0, ...]` should be equal to
    # `self.mfp[0, 0, :, :]` whereas `self.r[0, 0, ...]` should be equal to
    # `self.r[0, 0]`.
    nargs = len(args)
    if nargs > ndim and not isinstance(args[-1], type(...)):
        raise IndexError(
            f"Too many indices ({nargs}) for {ndim} dimensions"
        ) from None
    # Convert numpy arrays to lists to avoid ValueError in `count(...)`.
    norm = [
        list(arg) if isinstance(arg, numpy.ndarray) else arg
        for arg in args
    ]
    # Count the number of ellipses.
    count = norm.count(...)
    # If there is no ellipsis, there's nothing to do.
    if count == 0:
        return args
    # If there is more than one ellipsis, raise an exception.
    if count > 1:
        raise IndexError(
            f"Index arguments may only contain a single ellipsis ('...')"
        ) from None
    # Expand arguments into
    # 1) all arguments before the ellipsis
    # 2) a slice for every dimension represented by the ellipsis
    # 3) all arguments after the ellipsis
    nslice = ndim - nargs + 1
    ellpos = args.index(Ellipsis)
    return (
        *args[slice(0, ellpos)],
        *([slice(None)] * nslice),
        *args[slice(ellpos+1, nargs+1)],
    )

