import numbers
import typing

from . import axis
from .numeric import AxisLike
from ._axes import (
    Axes,
    axes_factory
)
from ._types import (
    Array,
    Object,
    Scalar,
    Tensor,
    Vector,
)
from ._implementations import (
    array_factory as array,
    scalar_factory as scalar,
    tensor_factory as tensor,
    vector_factory as vector,
)

@typing.overload
def axes(
    mapping: typing.Mapping[str, AxisLike],
    *lengths: numbers.Integral,
    shape: typing.Optional[typing.Iterable[numbers.Integral]]=None,
    dimensions: typing.Optional[typing.Iterable[str]]=None,
    axes: typing.Optional[typing.Iterable[AxisLike]]=None,
    **pairs: AxisLike
) -> Axes:
    """Create a axes from arguments. See Notes for precedence rules.

    Returns
    -------
    `~axes`

    Raises
    ------
    `ValueError`
        The given combination of arguments produced an empty axes.
    `TypeError`
        The caller passed `shape` or `axes` as a positional argument.

    Notes
    -----
    - A axes initialized from only a shape will contain axis pairs in which
      each name has the form `xi` and the corresponding sequence is a length-`i`
      zero-based integer array, where `i` ranges from 0 to `len(shape)-1`.
    - A axes initialized from a sequence of anonymous axes will contain
      key-value pairs with keys constructed as if initialized from a shape, each
      corresponding to an axis in the order given.
    - Explicit dimensions will replace default axis names in a axes initialized
      from either a shape or axes.
    - A axes initialized from key-value pairs or from a single mapping will use
      each key as a dimension and the corresponding value as the axis object.
    - In case of multiple initialization styles, key-value pairs take precedence
      over a mapping, which takes precedence over axes (with or without explicit
      dimensions), which take precedence over shape (with or without explicit
      dimensions).
    """

@typing.overload
def axes(*lengths: numbers.Integral) -> Axes:
    """Create a axes with the given axis lengths.

    Parameters
    ----------
    *lengths : integral numbers
        One or more integral axis lengths.

    Returns
    -------
    `~axes`
    """

@typing.overload
def axes(
    *,
    shape: typing.Iterable[numbers.Integral],
    dimensions: typing.Iterable[str],
) -> Axes:
    """Create a axes with the given shape.

    Parameters
    ----------
    shape : iterable of integral numbers
        An iterable object containing one or more integral axis lengths. Must be
        passed as a keyword argument.
    dimensions : iterable of strings
        The name of each axis. The length of `dimensions` must equal the length
        of `shape`. Must be passed as a keyword argument.

    Returns
    -------
    `~axes`

    Raises
    ------
    `TypeError`
        Caller passed dimensions without shape.
    `ValueError`
        Caller passed dimensions and shape with different lengths.
    """

@typing.overload
def axes(
    *,
    dimensions: typing.Iterable[str],
    axes: typing.Iterable[AxisLike],
) -> Axes:
    """Create a axes from anonymous axes.

    Parameters
    ----------
    dimensions : iterable of strings
        The name of each axis. The length of `dimensions` must equal the length
        of `axes`. Must be passed as a keyword argument.
    axes : iterable of axis-like
        An iterable object containing one or more axis-like objects. Each
        argument may be an iterable of integers or an instance of `~Axis`. Must
        be passed as a keyword argument.

    Returns
    -------
    `~axes`

    Raises
    ------
    `TypeError`
        Caller passed dimensions without shape.
    `ValueError`
        Caller passed dimensions and shape with different lengths.
    """

@typing.overload
def axes(**pairs: AxisLike) -> Axes:
    """Create a axes from individual dimension-axis pairs.

    Parameters
    ----------
    **pairs
        One or more key-value pairs. Each value may be an iterable of integers
        or an instance of `~Axis`. The corresponding key will become the string
        dimension corresponding to that axis.

    Returns
    -------
    `~axes`
    """

@typing.overload
def axes(mapping: typing.Mapping[str, AxisLike]) -> Axes:
    """Create a axes from a mapping of dimension-axis pairs.

    Parameters
    ----------
    mapping
        A mapping from dimension name to axis value. Each value may be an
        iterable of integers or an instance of `~Axis`. The corresponding key
        will become the string dimension corresponding to that axis.

    Returns
    -------
    `~axes`
    """

def axes(*args, **kwargs):
    return axes_factory(*args, **kwargs)


__all__ = [
    # Types
    Array,
    Axes,
    Object,
    Scalar,
    Vector,
    Tensor,
    # Functions
    array,
    scalar,
    tensor,
    vector,
    axes,
    # Modules
    axis,
]


