import numbers
import typing

from ._types import (
    Array,
    Axis,
    Indexer,
    Index,
    Indices,
    Object,
    Measurable,
    Measurement,
    Quantity,
    Sequence,
    Value,
)
from . import mixins
from . import _operators as operators


AxisType = typing.TypeVar('AxisType', bound=Axis)
AxisLike = typing.Union[
    typing.Iterable[int],
    typing.Iterable[str],
    Axis,
]
IndexerType = typing.TypeVar('IndexerType', bound=Indexer)
MeasurableType = typing.TypeVar('MeasurableType', bound=Measurable)
MeasurementType = typing.TypeVar('MeasurementType', bound=Measurement)
RealValueType = typing.TypeVar('RealValueType', int, float, numbers.Real)


class DataTypeError(Exception):
    """Data has the wrong type."""


__all__ = [
    Array,
    Axis,
    DataTypeError,
    Indexer,
    Index,
    Indices,
    Object,
    Measurable,
    Measurement,
    Quantity,
    Sequence,
    Value,
    mixins,
    operators,
]