import math
import operator
import typing

import numpy
import numpy.typing

from ._types import Quantity


def __abs__(a):
    """Called for abs(a)."""
    return unary(operator.abs, a)


def __neg__(a):
    """Called for -a."""
    return unary(operator.neg, a)


def __pos__(a):
    """Called for +a."""
    return unary(operator.pos, a)


def __round__(a, ndigits: typing.Optional[int]=None):
    """Called for round(a)."""
    return unary(round, a, ndigits=ndigits)


def __floor__(a):
    """Called for math.floor(a)."""
    return unary(math.floor, a)


def __ceil__(a):
    """Called for math.ceil(a)."""
    return unary(math.ceil, a)


def __trunc__(a):
    """Called for math.trunc(a)."""
    return unary(math.trunc, a)


def unary(f: typing.Callable, a, **kwargs):
    """Compute f(a)."""
    if isinstance(a, Quantity):
        return f(a.data, **kwargs)
    return f(a, **kwargs)


def __eq__(a, b):
    """Called for a == b."""
    return binary(operator.eq, a, b)


def __ne__(a, b):
    """Called for a != b."""
    return binary(operator.ne, a, b)


def __lt__(a, b):
    """Called for a < b."""
    return binary(operator.lt, a, b)


def __le__(a, b):
    """Called for a <= b."""
    return binary(operator.le, a, b)


def __gt__(a, b):
    """Called for a > b."""
    return binary(operator.gt, a, b)


def __ge__(a, b):
    """Called for a >= b."""
    return binary(operator.ge, a, b)


def __add__(a, b):
    """Called for a + b."""
    return binary(operator.add, a, b)


def __radd__(a, b):
    """Called for b + a."""
    return binary(operator.add, b, a)


def __sub__(a, b):
    """Called for a - b."""
    return binary(operator.sub, a, b)


def __rsub__(a, b):
    """Called for b - a."""
    return binary(operator.sub, b, a)


def __mul__(a, b):
    """Called for a * b."""
    return binary(operator.mul, a, b)


def __rmul__(a, b):
    """Called for b * a."""
    return binary(operator.mul, b, a)


def __truediv__(a, b):
    """Called for a / b."""
    return binary(operator.truediv, a, b)


def __rtruediv__(a, b):
    """Called for b / a."""
    return binary(operator.truediv, b, a)


def __floordiv__(a, b):
    """Called for a // b."""
    return binary(operator.floordiv, a, b)


def __rfloordiv__(a, b):
    """Called for b // a."""
    return binary(operator.floordiv, b, a)


def __mod__(a, b):
    """Called for a % b."""
    return binary(operator.mod, a, b)


def __rmod__(a, b):
    """Called for b % a."""
    return binary(operator.mod, b, a)


def __pow__(a, b):
    """Called for a ** b."""
    return binary(operator.pow, a, b)


def __rpow__(a, b):
    """Called for b ** a."""
    return binary(operator.pow, b, a)


def binary(f: typing.Callable, a, b, **kwargs):
    """Compute f(a, b)."""
    if isinstance(a, Quantity) and isinstance(b, Quantity):
        return f(a.data, b.data, **kwargs)
    if isinstance(a, Quantity):
        return f(a.data, b, **kwargs)
    if isinstance(b, Quantity):
        return f(a, b.data, **kwargs)
    return f(a, b, **kwargs)


def __contains__(a: Quantity, v, /) -> bool:
    """Called for v in a."""
    return v in a.data


def __len__(a: Quantity) -> int:
    """Called for len(a)."""
    return len(a.data)


T = typing.TypeVar('T')


def __iter__(a: Quantity[T]) -> typing.Iterator[T]:
    """Called for iter(a)."""
    return iter(a.data)


def iter_from_getitem(a: typing.Sequence):
    """Iterate over `a` via `a.__getitem__`."""
    i = 0
    try:
        while True:
            v = a[i]
            yield v
            i += 1
    except IndexError:
        return


def __array__(a: Quantity, *args, **kwargs) -> numpy.ndarray:
    """Called for conversion to a native `numpy` array."""
    return numpy.array(a.data, *args, **kwargs)


def __int__(a: Quantity):
    """Compute int(a)."""
    return int(a.data)


def __float__(a: Quantity):
    """Compute float(a)."""
    return float(a.data)


def __complex__(a: Quantity):
    """Compute complex(a)."""
    return complex(a.data)


