import numbers
import typing

import numpy

from .. import metric
from .. import numeric

from ._functions import conversion
from . import _operators
from ._types import (
    Value,
    Sequence,
)


def object_withunit(
    a: numeric.MeasurementType,
    unit: metric.UnitLike,
) -> numeric.MeasurementType:
    """Convert this object to a new unit."""
    if converted := _object_convert(a, unit):
        return converted
    t = a.__class__.__qualname__
    raise TypeError(f"Cannot change unit on object of type {t!r}")


def _object_convert(a: numeric.MeasurementType, unit):
    """Helper for `object_withunit`."""
    if a.unit == unit:
        return a
    c = conversion(a, unit=unit)
    scale = float(c)
    if isinstance(a, Value):
        return value_factory(a.data * scale, unit=c.new)
    if isinstance(a, Sequence):
        return sequence_factory(numpy.array(a.data) * scale, unit=c.new)


VALUE_OPERATORS = {
    'withunit': object_withunit,
    '__repr__': _operators.__repr__,
    '__eq__': _operators.__eq__,
    '__int__': numeric.operators.__int__,
    '__float__': numeric.operators.__float__,
    '__complex__': numeric.operators.__complex__,
}


def value_factory(x, /, unit=None) -> Value:
    """Create a new measured value."""
    Type = type('Value', (Value,), VALUE_OPERATORS)
    d, u = value_args(x, unit)
    return Type(d, metric.unit(u or '1'))


def value_args(x, unit, /) -> typing.Tuple[typing.Any, metric.Unit]:
    """Parse arguments to initialize `~Value`."""
    if isinstance(x, Value):
        if unit is None:
            return x.data, x.unit
        raise ValueError(
            "Cannot change unit via object creation"
            ". Please use the 'withunit' function."
        ) from None
    if isinstance(x, (int, float, numbers.Real)):
        return x, unit
    if isinstance(x, (Sequence, numpy.ndarray)) and x.size == 1:
        if isinstance(x, Sequence):
            return x.data[0], x.unit
        if isinstance(x, numpy.ndarray):
            return value_args(x.ravel()[0], unit)
    if isinstance(x, numeric.Measurement):
        a = numpy.array(x.data)
        if a.ndim == 0:
            return a[0], x.unit
        if a.ndim == 1 and a.size == 1:
            return value_args(a.ravel()[0], x.unit)
        raise numeric.DataTypeError(
            f"Cannot create a single value from size-{a.size} data"
        ) from None
    if isinstance(x, typing.Sequence) and not isinstance(x, str):
        return value_args(numpy.array(x), unit)
    raise TypeError(
        f"Cannot create a value from {x!r} and {unit!r}"
    ) from None


def sequence_getitem(a: Sequence, i: typing.SupportsIndex):
    data = a.data[i]
    # NOTE: A measured sequence should return a single measured value when
    # the subscripted data represents a single value. This is consistent
    # with the behavior of built-in sequences and numpy arrays. We approach
    # this problem by naively trying to create a measured value from the
    # subscripted data and returning that value if we succeeded. If anything
    # goes wrong when trying to create the measured value, we assume that
    # the subscripted data does not represent a single value, and return a
    # new measured sequence.
    try:
        result = value_factory(data, unit=a.unit)
    except TypeError:
        result = sequence_factory(data, unit=a.unit)
    return result


def sequence_array(a: Sequence, *args, **kwargs) -> numpy.ndarray:
    """Convert this variable to a `numpy.ndarray`.
    
    Notes
    -----
    - This method converts an instance to a `numpy.ndarray` by calling
        `numpy.asarray` on the underlying data. Therefore, if the data is
        already a `numpy.ndarray`, calling `numpy.asarray` on this instance
        will return the original array.
    """
    return numpy.asarray(a.data, *args, **kwargs)


SEQUENCE_OPERATORS = {
    'withunit': object_withunit,
    '__repr__': _operators.__repr__,
    '__eq__': _operators.__eq__,
    '__len__': numeric.operators.__len__,
    '__iter__': numeric.operators.__iter__,
    '__contains__': numeric.operators.__contains__,
    '__getitem__': sequence_getitem,
    '__array__': sequence_array,
}


def sequence_factory(x, /, unit=None) -> Sequence:
    """Create a new measured sequence."""
    Type = type('Sequence', (Sequence,), SEQUENCE_OPERATORS)
    d, u = sequence_args(x, unit)
    return Type(d, metric.unit(u or '1'))


def sequence_args(x, unit, /) -> typing.Tuple[typing.Any, metric.Unit]:
    """Parse arguments to initialize `~Sequence`."""
    if isinstance(x, Sequence):
        if unit is None:
            return x.data, x.unit
        raise ValueError(
            "Cannot change unit via object creation"
            ". Please use the 'withunit' function."
        ) from None
    if isinstance(x, Value):
        return sequence_args([x.data], x.unit)
    if isinstance(x, numeric.Measurement):
        return numpy.array(x.data, ndmin=1), x.unit
    if isinstance(x, numbers.Real):
        return numpy.array([x]), unit
    a = numpy.asarray(x)
    if a.ndim > 0:
        return a.flatten(), unit
    raise TypeError(
        f"Cannot create a sequence from {x!r} and {unit!r}"
    ) from None



