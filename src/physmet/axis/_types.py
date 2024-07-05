import typing

import numpy
import numpy.typing

from .. import metric
from .. import numeric


T = typing.TypeVar('T')


class Points(numeric.Axis[numpy.typing.NDArray[numpy.uint]]):
    """The interface to an integral axis."""

    def __init__(self, data: numpy.typing.NDArray[numpy.uint]) -> None:
        super().__init__(data)


class Symbols(numeric.Axis[typing.List[str]]):
    """The interface to a symbolic axis."""

    def __init__(self, data: typing.List[str]) -> None:
        super().__init__(data)


class _CoordinatesBase(
    numeric.Axis[numpy.typing.NDArray[numpy.floating]],
    numeric.Measurable,
    numeric.Measurement,
): ...


class Coordinates(_CoordinatesBase):
    """The interface to an axis with numeric data and an associated unit."""

    def __init__(
        self,
        data: numpy.typing.NDArray[numpy.floating],
        unit: metric.Unit,
    ) -> None:
        super().__init__(data)
        self._unit = unit

    @property
    def unit(self):
        """The metric unit of coordinate data."""
        return self._unit



