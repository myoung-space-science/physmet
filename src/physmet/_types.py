import abc
import typing

import numpy

from . import _axes
from . import base
from . import data
from . import metric
from . import numeric


T = typing.TypeVar('T')


class Object(numeric.Measurement[T], base.Real, numeric.mixins.Functions):
    """ABC for measurable numeric objects."""

    def __init__(
        self,
        data: T,
        unit: typing.Union[str, metric.Unit],
    ) -> None:
        self._data = data
        self._unit = unit

    def __repr__(self):
        """An unambiguous representation of this object."""
        s = f"data={self.data}, unit={str(self.unit)!r}"
        return f"{self.__class__.__qualname__}({s})"


# NOTE: This is necessary for `numpy` mixin support.
numeric.Object.register(Object)


ObjectType = typing.TypeVar('ObjectType', bound=Object)
"""Type variable bounded by `~Object`."""


class Scalar(Object[base.ScalarType], base.Scalar):
    """A real-valued, measurable object with 0-D data."""

    _FUNCTIONS = {}

    def __init__(
        self,
        data: base.ScalarType,
        unit: str | metric.Unit,
    ) -> None:
        super().__init__(data, unit)


class Vector(Object[typing.Sequence]):
    """A real-valued, measurable object with 1-D data."""

    _FUNCTIONS = {}

    def __init__(
        self,
        data: typing.Sequence,
        unit: str | metric.Unit,
    ) -> None:
        array = numpy.asanyarray(data)
        super().__init__(array, unit)
        self._size = array.size

    @property
    def size(self) -> int:
        """The number of data values in this vector."""
        return self._size


class Tensor(Object[base.ArrayType]):
    """A real-valued, measurable object with N-D data."""

    _FUNCTIONS = {}

    def __init__(
        self,
        data: base.ArrayType,
        unit: str | metric.Unit,
    ) -> None:
        if not isinstance(data, base.Array):
            raise TypeError(
                f"Data argument to {type(self)} must implement"
                f" the {base.Array} protocol"
            )
        super().__init__(data, unit)
        self._size = data.size
        self._ndim = data.ndim
        self._shape = data.shape

    @property
    def size(self) -> int:
        """The number of data values in this tensor."""
        return self._size

    @property
    def ndim(self) -> int:
        """The number of dimensions in this tensor's data."""
        return self._ndim

    @property
    def shape(self) -> int:
        """The length of each dimension in this tensor's data."""
        return self._shape


class Array(Tensor[data.Array]):
    """A `~Tensor` with labeled axes."""

    def __init__(
        self,
        data: data.Array,
        unit: str | metric.Unit,
        axes: _axes.Axes,
    ) -> None:
        super().__init__(data, unit)
        self._axes = axes
        self._data_interface = data
        self._axes = axes
        self._scale = 1.0
        self._squeezed = None
        self._scaled = None

    @abc.abstractmethod
    def __int__(self):
        """Called for int(self)."""

    @abc.abstractmethod
    def __float__(self):
        """Called for float(self)."""

    @abc.abstractmethod
    def __complex__(self):
        """Called for complex(self)."""

    @abc.abstractmethod
    def __getitem__(self, i, /):
        """Called for self[i]."""

    @abc.abstractmethod
    def transpose(self, *args):
        """Transpose this array."""

    @property
    def squeezed(self):
        """The equivalent `numpy` array, with singular dimensions removed."""
        if self._squeezed is None:
            self._squeezed = numpy.array(self).squeeze()
        return self._squeezed

    @property
    def data(self):
        if self._scaled is None:
            self._scaled = self._scale * self._data_interface
        return self._scaled

    @property
    def shapemap(self):
        """A mapping from dimension to axis length."""
        return self._data_interface.shapemap

    @property
    def dimensions(self):
        """This array's dimensions."""
        return self.axes.dimensions

    @property
    def axes(self):
        """This array's axes."""
        return self._axes



