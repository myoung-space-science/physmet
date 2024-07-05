import typing

import nonstring

from .. import base
from .. import metric
from .. import numeric


T = typing.TypeVar('T')


class Value(numeric.Measurement[T], base.Value):
    """A numeric value with an associated metric unit."""

    def __init__(self, data: T, unit: metric.Unit) -> None:
        self._data = data
        self._unit = unit


class Sequence(numeric.Measurement[T], base.Sequence):
    """A sequence of numeric values with an associated metric unit."""

    def __init__(self, data: typing.Sequence[T], unit: metric.Unit) -> None:
        self._data = data
        self._unit = unit
        self._size = None

    @property
    def size(self) -> int:
        """The number of numerical values in this instance."""
        if self._size is None:
            self._size = nonstring.size(self.data)
        return self._size


