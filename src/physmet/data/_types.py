import abc
import collections.abc
import numbers
import typing

import numpy
import numpy.typing

from .. import base
from .. import numeric


IndexLike = typing.Union[
    typing.SupportsIndex,
    typing.Tuple[typing.SupportsIndex],
    slice,
]

DataType = typing.TypeVar('DataType', numeric.Array, numpy.ndarray)

T = typing.TypeVar('T')


class Dimensions(collections.abc.Sequence):
    """A representation of one or more array-axis names.
    
    This class is formally a sequence but it supports certain set-like
    operations in an order-preserving way.
    """

    def __init__(self, names: typing.Tuple[str]) -> None:
        self._names = names

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        """Called for self == other."""

    @abc.abstractmethod
    def __lt__(self, other) -> bool:
        """Called for self < other."""

    @abc.abstractmethod
    def __le__(self, other) -> bool:
        """Called for self <= other."""

    @abc.abstractmethod
    def __gt__(self, other) -> bool:
        """Called for self > other."""

    @abc.abstractmethod
    def __ge__(self, other) -> bool:
        """Called for self >= other."""

    @abc.abstractmethod
    def replace(self, old: str, new: str) -> typing.Self:
        """Replace a single dimension."""

    @abc.abstractmethod
    def insert(
        self,
        name: str,
        index: typing.SupportsIndex=None,
    ) -> typing.Self:
        """Insert the named dimension at `index`."""

    @abc.abstractmethod
    def __and__(self, other) -> typing.Self:
        """Called for self & other."""

    @abc.abstractmethod
    def __rand__(self, other) -> typing.Self:
        """Called for other & self."""

    @abc.abstractmethod
    def __sub__(self, other) -> typing.Self:
        """Called for self - other."""

    @abc.abstractmethod
    def __rsub__(self, other) -> typing.Self:
        """Called for other - self."""

    @abc.abstractmethod
    def __or__(self, other) -> typing.Self:
        """Called for self | other."""

    @abc.abstractmethod
    def __ror__(self, other) -> typing.Self:
        """Called for other | self."""

    @abc.abstractmethod
    def __xor__(self, other) -> typing.Self:
        """Called for self ^ other."""

    @abc.abstractmethod
    def __rxor__(self, other) -> typing.Self:
        """Called for other ^ self."""

    @abc.abstractmethod
    def copy(self) -> typing.Self:
        """Create a copy of this instance."""

    @abc.abstractmethod
    def permute(self, *args, **kwargs) -> typing.Self:
        """Reorder these dimensions."""

    @abc.abstractmethod
    def mismatch(self, other: typing.Iterable[str]):
        """Raise an exception if `self < other` or `self > other`."""

    @abc.abstractmethod
    def __hash__(self) -> int:
        """Support use as a mapping key."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Called for len(self)."""

    @abc.abstractmethod
    def __getitem__(self, i: typing.SupportsIndex, /) -> str:
        """Called for index-based access."""


class Array(numeric.mixins.Functions, numeric.Array[DataType], base.Real):
    """Base class for numeric arrays with named dimensions."""

    def __init__(
        self,
        data: DataType,
        dimensions: Dimensions,
    ) -> None:
        self._data = data
        self._shape = data.shape
        self._size = data.size
        self._ndim = data.ndim
        self._dimensions = dimensions
        self._array = data if isinstance(data, numpy.ndarray) else None

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        """Called for self == other."""

    @abc.abstractmethod
    def __ne__(self, other) -> bool:
        """Called for self != other."""

    @abc.abstractmethod
    def remesh(self, other, invert: bool=False):
        """Construct array meshes for broadcasting."""

    @abc.abstractmethod
    def has(self, value: numbers.Real) -> bool:
        """True if `value` is in this sequence's data array."""

    @abc.abstractmethod
    def transpose(self, *args):
        """Transpose this array."""

    @abc.abstractmethod
    def _load_array(
        self,
        index: typing.Optional[IndexLike]=None,
    ) -> numpy.typing.NDArray:
        """Load the array from disk or memory."""

    @abc.abstractmethod
    def _read_array(
        self,
        index: typing.Optional[IndexLike]=None,
    ) -> numpy.typing.NDArray:
        """Read the array data from disk."""

    @property
    def size(self) -> int:
        """The number of elements in this array."""
        return self._size

    @property
    def array(self):
        """The underlying array-like data interface."""
        if self._array is None:
            self._array = self._load_array()
        return self._array

    @property
    def shapemap(self) -> typing.Dict[str, int]:
        """The name and size of each axis."""
        return dict(zip(self.dimensions, self.shape))

    @property
    def dimensions(self) -> Dimensions:
        """The names of this array's indexable axes."""
        return self._dimensions

    @property
    def ndim(self) -> int:
        """The number of dimensions in this array."""
        return self._ndim

    @property
    def shape(self) -> typing.Tuple[int]:
        """The length of each dimension in this array."""
        return self._shape

    def _get_arg_data(self, arg):
        if isinstance(arg, Array):
            return self.array
        return super()._get_arg_data(arg)

    _FUNCTIONS = {}


Array._UFUNC_TYPES += [Array]
Array._FUNCTION_TYPES += [Array]


