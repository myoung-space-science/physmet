import abc
import numbers
import typing

import numpy
import numpy.typing

from .. import base
from .. import metric


T = typing.TypeVar('T')


@typing.runtime_checkable
class Quantity(typing.Protocol[T]):
    """Abstract protocol class for numeric quantities."""

    _data: T

    @abc.abstractmethod
    def __eq__(self, other) -> typing.Union[bool, typing.Iterable[bool]]: ...

    def __repr__(self):
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}(data={self.data})"

    @property
    def data(self) -> T:
        """The numeric data."""
        return self._data


# NOTE: Registering virtual subclasses of Object allow us to avoid raising
# "TypeError: Protocols with non-method members don't support issubclass()" when
# checking Quantity subclasses.
class Object(Quantity[T], abc.ABC):
    """Concrete implementation of a numerical quantity."""

    def __init__(self, data: T) -> None:
        self._data = data


@typing.runtime_checkable
class Value(Quantity[T], base.Value, typing.Protocol):
    """Abstract protocol class for singular numeric values.

    A numeric value is a numeric quantity that supports the value protocol.
    """

    def __round__(self) -> typing.Self: ...

    def __floor__(self) -> typing.Self: ...

    def __ceil__(self) -> typing.Self: ...

    def __trunc__(self) -> typing.Self: ...


@typing.runtime_checkable
class Sequence(Quantity[T], base.Sequence, typing.Protocol):
    """Abstract protocol class for numeric sequences.

    A numeric sequence is a numeric quantity that supports the sequence
    protocol.
    """


@typing.runtime_checkable
class Array(Quantity[T], base.Array, typing.Protocol):
    """Protocol for numeric arrays.

    A numeric array is a numeric quantity that supports the array protocol.
    """


@typing.runtime_checkable
class Measurement(Quantity[T], typing.Protocol):
    """Protocol for numeric measurements.

    A numeric measurement is a numeric quantity with an associated metric unit.
    """

    _unit: metric.Unit

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        s = f"data={self.data}, unit={str(self.unit)!r}"
        return f"{self.__class__.__qualname__}({s})"

    @property
    def isunitless(self) -> bool:
        """True if this object's unit is '1'."""
        return self.unit == '1'

    @property
    def unit(self) -> metric.Unit:
        """The metric unit of this object's data."""
        return self._unit

    @abc.abstractmethod
    def withunit(self, arg: typing.Union[str, metric.Unit]) -> typing.Self:
        raise NotImplementedError


@typing.runtime_checkable
class Measurable(typing.Protocol):
    """Protocol for explicitly measurable objects.

    Classes that implement this protocol must define the `__measure__` method,
    which should return an object that implements the `~Measurement` protocol.

    See Also
    --------
    `~measurable.measure`
        The function that calls `__measure__`.
    """

    __slots__ = ()

    @abc.abstractmethod
    def __measure__(self, **kwargs) -> Measurement: ...


I = typing.TypeVar('I', bound=typing.SupportsInt)


class Indexer(Quantity[T], base.Real):
    """A generalized axis-indexing type."""

    def __init__(self, data: T) -> None:
        self._data = data

    @abc.abstractmethod
    def shift(self, __x: I, floor: I=None, ceil: I=None) -> typing.Self: ...


class Index(Indexer[int], Value):
    """An integral index value."""

    @abc.abstractmethod
    def __index__(self) -> int: ...


class Indices(Indexer[numpy.typing.NDArray[numpy.integer]], Sequence):
    """A sequence of integral index values."""


class Axis(Quantity[T], base.Sequence):
    """A generalized array axis."""

    _indices: Indices=None

    def __init__(self, reference: T) -> None:
        self._data = reference
        self._length = None

    @abc.abstractmethod
    def __getitem__(self, i, /) -> typing.Self: ...

    @abc.abstractmethod
    def index(self, *targets: T): ...

    @property
    def indices(self):
        """The corresponding index sequence."""
        return self._indices

    @property
    def length(self) -> int:
        """The number of values in this axis."""
        if self._length is None:
            self._length = len(self.data)
        return self._length

