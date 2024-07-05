import abc
import numbers
import typing

import nonstring
import numpy
import numpy.typing

from .. import metric


T = typing.TypeVar('T')


@typing.runtime_checkable
class Ordered(typing.Protocol):
    """Protocol for objects that support relative ordering.

    Classes that implement this protocol must define the following methods

    - `__lt__`
    - `__le__`
    - `__gt__`
    - `__ge__`

    The most appropriate return type will often be `bool` for each method, but
    exceptions exist (cf. `numpy.ndarray`).
    """

    __slots__ = ()

    @abc.abstractmethod
    def __lt__(self, other): ...

    @abc.abstractmethod
    def __le__(self, other): ...

    @abc.abstractmethod
    def __gt__(self, other): ...

    @abc.abstractmethod
    def __ge__(self, other): ...


@typing.runtime_checkable
class Comparable(Ordered, typing.Protocol):
    """Protocol for comparable objects.

    Classes that implement this protocol must implement the `~Ordered` protocol,
    and must define the following methods

    - `__eq__`
    - `__ne__`

    The most appropriate return type will often be `bool` for each method, but
    exceptions exist (cf. `numpy.ndarray`).
    """

    __slots__ = ()

    @abc.abstractmethod
    def __eq__(self, other): ...

    @abc.abstractmethod
    def __ne__(self, other): ...


@typing.runtime_checkable
class Additive(typing.Protocol):
    """Protocol for additive objects.

    Classes that implement this protocol must define the following methods

    - `__add__`
    - `__radd__`
    - `__sub__`
    - `__rsub__`

    Each method may return whatever type is appropriate for the class.
    """

    __slots__ = ()

    @abc.abstractmethod
    def __add__(self, other) -> typing.Self: ...

    @abc.abstractmethod
    def __radd__(self, other) -> typing.Self: ...

    @abc.abstractmethod
    def __sub__(self, other) -> typing.Self: ...

    @abc.abstractmethod
    def __rsub__(self, other) -> typing.Self: ...


@typing.runtime_checkable
class Multiplicative(typing.Protocol):
    """Protocol for multiplicative objects.

    Classes that implement this protocol must define the following methods

    - `__mul__`
    - `__rmul__`
    - `__truediv__`
    - `__rtruediv__`

    Each method may return whatever type is appropriate for the class.

    Notes
    -----
    - The floor- and modular-division operators do not appear in this protocol,
      because their distinction from "true" division really only makes sense in
      the context of real-valued numeric objects, whereas a more general set of
      objects may implement this protocol (e.g., symbolic expressions).
    """

    __slots__ = ()

    @abc.abstractmethod
    def __mul__(self, other) -> typing.Self: ...

    @abc.abstractmethod
    def __rmul__(self, other) -> typing.Self: ...

    @abc.abstractmethod
    def __truediv__(self, other) -> typing.Self: ...

    @abc.abstractmethod
    def __rtruediv__(self, other) -> typing.Self: ...


@typing.runtime_checkable
class Algebraic(Additive, Multiplicative, typing.Protocol):
    """Protocol for algebraic objects.

    Classes that implement this protocol must implement the `~Additive` and
    `~Multiplicative` protocols, and must define the `__pow__` method, which may
    return whatever type is appropriate for the class.

    Notes
    -----
    - The formal definition of an algebraic quantity requires that
      exponentiation involve only constant, rational exponents. However, this
      protocol does not place any restrictions on the type of exponent.
    - This protocol differs from the `~Additive` and `~Multiplicative`
      protocols, which require foward and reverse versions of their operators,
      by not requiring `__rpow__`. Doing so would imply that any algebraic
      object should be allowed as an exponent, which is not generally true. Of
      course, this doesn't prevent a concrete object from defining `__rpow__`.
    """

    __slots__ = ()

    @abc.abstractmethod
    def __pow__(self, other) -> typing.Self: ...


@typing.runtime_checkable
class Complex(Algebraic, typing.Protocol):
    """Protocol for complex-valued objects.

    Classes that implement this protocol must implement the `~Algebraic`
    protocol, and must define the following methods

    - `__abs__`
    - `__pos__`
    - `__neg__`

    Each method may return whatever type is appropriate for the class.
    """

    __slots__ = ()

    @abc.abstractmethod
    def __abs__(self) -> typing.Self: ...

    @abc.abstractmethod
    def __pos__(self) -> typing.Self: ...

    @abc.abstractmethod
    def __neg__(self) -> typing.Self: ...


@typing.runtime_checkable
class Real(Comparable, Complex, typing.Protocol):
    """Protocol for real-valued numeric objects.

    Classes that implement this protocol must implement the `~Comparable` and
    `~Complex` protocols, and must define the following methods

    - `__rpow__`
    - `__floordiv__`
    - `__rfloordiv__`
    - `__mod__`
    - `__rmod__`

    Each method may return whatever type is appropriate for the class.
    """

    @abc.abstractmethod
    def __rpow__(self, other) -> typing.Self: ...

    @abc.abstractmethod
    def __floordiv__(self, other) -> typing.Self: ...

    @abc.abstractmethod
    def __rfloordiv__(self, other) -> typing.Self: ...

    @abc.abstractmethod
    def __mod__(self, other) -> typing.Self: ...

    @abc.abstractmethod
    def __rmod__(self, other) -> typing.Self: ...


@typing.runtime_checkable
class Value(
    typing.SupportsInt,
    typing.SupportsFloat,
    typing.SupportsComplex,
    typing.Protocol): ...


@typing.runtime_checkable
class Scalar(Real, typing.Protocol):
    """Protocol for real-valued scalar objects.

    Classes that implement this protocol must implement the `~Real` protocol,
    and must define the following methods

    - `__int__`, which must return an `int`
    - `__float__`, which must return a `float`
    - `__round__`, which may return whatever type is appropriate to the class

    Notes
    -----
    - This protocol does not require `__trunc__` (called by `math.trunc`),
      `__floor__` (called by `math.floor`), or `__ceil__` (called by
      `math.ceil`) because doing so would exclude the built-in `float` type.
    """

    __slots__ = ()

    @abc.abstractmethod
    def __int__(self) -> int: ...

    @abc.abstractmethod
    def __float__(self) -> float: ...

    @abc.abstractmethod
    def __round__(self) -> typing.Self: ...


@typing.runtime_checkable
class Sequence(typing.Protocol):
    """Protocol for numeric sequences.

    Classes that implement this protocol must define the following methods

    - `__contains__`, which should return a `bool`
    - `__len__`, which should return an `int`
    - `__iter__`, which should return an iterator over the value type
    - `__getitem__`, which may return an appropriate value type
    - `__array__`, which should return a `numpy.ndarray`
    """

    # NOTE: This is more restrictive than `collections.abc.Sequence`, which only
    # requires `__getitem__` and `__len__` in order to define additional mixin
    # methods that include `__contains__` and `__iter__`.

    __slots__ = ()

    @abc.abstractmethod
    def __contains__(self, v, /) -> bool: ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __iter__(self): ...

    @abc.abstractmethod
    def __getitem__(self, i, /) -> typing.Self: ...

    @abc.abstractmethod
    def __array__(self, *args, **kwargs) -> numpy.typing.NDArray: ...


@typing.runtime_checkable
class Array(Sequence, typing.Protocol):
    """Protocol for array-like objects.

    Classes that support this protocol must define the following properties

    - `size`, which should return an `int`
    - `ndim`, which should return an `int`
    - `shape`, which should return a `tuple` of `int`s
    """

    __slots__ = ()

    @property
    @abc.abstractmethod
    def size(self) -> int: ...

    @property
    @abc.abstractmethod
    def ndim(self) -> int: ...

    @property
    @abc.abstractmethod
    def shape(self) -> typing.Tuple[int, ...]: ...


ArrayType = typing.TypeVar('ArrayType', bound=Array)
ScalarType = typing.TypeVar('ScalarType', bound=Scalar)

