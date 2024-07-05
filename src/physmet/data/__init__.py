import typing

from ._types import (
    Array,
    Dimensions,
)
from ._implementations import (
    ARRAY_OPERATORS,
    DIMENSIONS_OPERATORS,
    array_factory,
    dimensions_factory,
)
from ._functions import (
    isclose,
    isequal,
    isindexlike,
    ismeasurable,
    hasdtype,
    nearest,
    remesh,
    subarray,
)

@typing.overload
def dimensions(names: typing.Iterable[str]) -> Dimensions:
    """Create array dimensions from an iterable of names."""

@typing.overload
def dimensions(*names: str) -> Dimensions:
    """Create array dimensions from one or more names."""

@typing.overload
def dimensions(n: int) -> Dimensions:
    """Create `n` array dimensions with default names."""

@typing.overload
def dimensions(*args) -> Dimensions:
    """Create array dimensions from the given arguments, if possible."""

def dimensions(*args):
    return dimensions_factory(*args)


ArrayType = typing.TypeVar('ArrayType', bound=Array)


@typing.overload
def array(x, /) -> Array:
    """Create an array-like quantity from `x`.

    This function will construct default dimension names based on the number of
    dimensions in `x`.
    """

@typing.overload
def array(x, /, dimensions: typing.Iterable[str]) -> Array:
    """Create an array-like quantity from `x`.

    This function will use the strings in `dimensions` as the dimension names.
    The length of `dimensions` must equal the number of dimensions in `x`.
    """

@typing.overload
def array(x, *dimensions: str) -> Array:
    """Create an array-like quantity from `x`.

    This function will use the string arguments following `x` as the dimension
    names. The number of strings must equal the number of dimensions in `x`.
    """

@typing.overload
def array(x: ArrayType, /) -> ArrayType:
    """Create an array-like quantity from `x`.

    This function will copy array data and dimensions from `x`.
    """

@typing.overload
def array(*args, **kwargs) -> Array:
    """Create an array-like quantity from the given arguments, if possible."""

def array(*args, **kwargs):
    return array_factory(*args, **kwargs)


__all__ = [
    ARRAY_OPERATORS,
    DIMENSIONS_OPERATORS,
    Array,
    Dimensions,
    array,
    dimensions,
    hasdtype,
    isclose,
    isequal,
    isindexlike,
    ismeasurable,
    nearest,
    remesh,
    subarray,
]


