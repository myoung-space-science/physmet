import abc
import typing

import numpy
import numpy.typing

from .. import numeric


T = typing.TypeVar('T')


class Value(numeric.Indexer[int], numeric.Value):
    """A single axis index."""

    @abc.abstractmethod
    def __index__(self) -> int: ...


NDArrayInt = numpy.typing.NDArray[numpy.integer]


class Sequence(numeric.Indexer[NDArrayInt], numeric.Sequence):
    """A sequence of axis indices."""

