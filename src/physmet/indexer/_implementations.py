import numbers
import contextlib
import operator
import typing

import numpy
import numpy.typing

from .. import numeric
from .. import quantity
from . import _types


T = typing.TypeVar('T')
I = typing.TypeVar('I', bound=typing.SupportsInt)


class OperandTypeError(TypeError):
    """Wrong operand type for a numeric operation."""


def indexer_eq(a, b, /):
    """Compute a == b."""
    try:
        return comparative(operator.eq, a, b)
    except OperandTypeError:
        return False


def indexer_ne(a, b, /):
    """Compute a != b."""
    try:
        return comparative(operator.ne, a, b)
    except OperandTypeError:
        return False


def indexer_lt(a, b):
    """Compute a < b."""
    try:
        return comparative(operator.lt, a, b)
    except OperandTypeError:
        return NotImplemented


def indexer_le(a, b):
    """Compute a <= b."""
    try:
        return comparative(operator.le, a, b)
    except OperandTypeError:
        return NotImplemented


def indexer_gt(a, b):
    """Compute a > b."""
    try:
        return comparative(operator.gt, a, b)
    except OperandTypeError:
        return NotImplemented


def indexer_ge(a, b):
    """Compute a >= b."""
    try:
        return comparative(operator.ge, a, b)
    except OperandTypeError:
        return NotImplemented


def indexer_abs(a):
    """Compute abs(a)."""
    return unary(operator.abs, a)


def indexer_pos(a):
    """Compute +a."""
    return unary(operator.pos, a)


def indexer_neg(a):
    """Compute -a."""
    return unary(operator.neg, a)


def indexer_round(a, /, ndigits: int=None):
    """Compute round(a[, ndigits])."""
    return unary(round, a, ndigits=ndigits)


def indexer_add(a, b):
    """Compute a + b."""
    return arithmetic(operator.add, a, b)


def indexer_radd(a, b):
    """Compute b + a."""
    return indexer_add(b, a)


def indexer_sub(a, b):
    """Compute a - b."""
    return arithmetic(operator.sub, a, b)


def indexer_rsub(a, b):
    """Compute b - a."""
    return indexer_sub(b, a)


def indexer_mul(a, b):
    """Compute a * b."""
    return arithmetic(operator.mul, a, b)


def indexer_rmul(a, b):
    """Compute b * a."""
    return indexer_mul(b, a)


def indexer_truediv(a, b):
    """Compute a / b."""
    x = numpy.array(a.data) if isinstance(a, numeric.Quantity) else a
    y = numpy.array(b.data) if isinstance(b, numeric.Quantity) else b
    return x / y


def indexer_rtruediv(a, b):
    """Compute b / a."""
    return indexer_truediv(b, a)


def indexer_floordiv(a, b):
    """Compute a // b."""
    return arithmetic(operator.floordiv, a, b)


def indexer_rfloordiv(a, b):
    """Compute b // a."""
    return indexer_floordiv(a, b)


def indexer_mod(a, b):
    """Compute a % b."""
    return arithmetic(operator.mod, a, b)


def indexer_rmod(a, b):
    """Compute b % a."""
    return indexer_mod(b, a)


def indexer_pow(a, b, mod: int=None):
    """Compute a ** b."""
    negative_exponent = False
    try:
        negative_exponent = b < 0
    except ValueError:
        negative_exponent = all(b < 0)
    if negative_exponent:
        raise ValueError(
            "Cannot raise index to a negative integral power"
        ) from None
    return arithmetic(pow, a, b, mod=mod)


def indexer_rpow(a, b, mod: int=None):
    """Compute b ** a."""
    return indexer_pow(b, a, mod=mod)


A = typing.TypeVar('A', bound=numbers.Real)
R = typing.TypeVar('R', bound=numbers.Real)


def unary(
    f: typing.Callable[[A], R],
    a: numeric.IndexerType,
    **kwargs
) -> numeric.IndexerType:
    """Implement a unary operation on `_types.Value` or `_types.Sequence`."""
    if isinstance(a, _types.Sequence):
        return sequence_factory([f(v, **kwargs) for v in a])
    return value_factory(f(a.data, **kwargs))


B = typing.TypeVar('B', bound=numbers.Real)
BoolLike = typing.Union[bool, typing.Sequence[bool]]


def comparative(
    f: typing.Callable[[A, B], BoolLike],
    a: numeric.IndexerType,
    b: numeric.IndexerType,
) -> BoolLike:
    """Implement a binary comparative operation on `_types.Value` or `_types.Sequence`."""
    r = binary(f, a, b)
    if isinstance(a, _types.Value) and isinstance(b, _types.Value):
        return bool(r)
    return numpy.array(r, dtype=bool)


@typing.overload
def arithmetic(
    f: typing.Callable[[A, B], R],
    a: numeric.IndexerType,
    b: _types.Sequence,
    **kwargs
) -> _types.Sequence: ...

@typing.overload
def arithmetic(
    f: typing.Callable[[A, B], R],
    a: _types.Sequence,
    b: numeric.IndexerType,
    **kwargs
) -> _types.Sequence: ...

@typing.overload
def arithmetic(
    f: typing.Callable[[A, B], R],
    a: _types.Value,
    b: _types.Value,
    **kwargs
) -> _types.Value: ...

@typing.overload
def arithmetic(
    f: typing.Callable[[A, B], R],
    a: typing.Union[numeric.IndexerType, numbers.Real],
    b: typing.Union[numeric.IndexerType, numbers.Real],
    **kwargs
) -> numeric.IndexerType: ...

def arithmetic(f, a, b, **kwargs):
    """Implement a binary arithmetic operation on indexer types."""
    r = binary(f, a, b, **kwargs)
    if isinstance(a, numeric.Indexer) and isinstance(b, numeric.Indexer):
        if all(isinstance(i, _types.Value) for i in (a, b)):
            return _new_or_result(value_factory, r)
        return _new_or_result(sequence_factory, r)
    if isinstance(b, numbers.Integral):
        if isinstance(a, _types.Value):
            return _new_or_result(value_factory, r)
        if isinstance(a, _types.Sequence):
            return _new_or_result(sequence_factory, r)
    if isinstance(a, numbers.Integral):
        if isinstance(b, _types.Value):
            return _new_or_result(value_factory, r)
        if isinstance(b, _types.Sequence):
            return _new_or_result(sequence_factory, r)
    raise OperandTypeError(a, b)


def binary(f, a, b, **kwargs):
    """Compute f(a, b, **kwargs) on index data."""
    if isinstance(a, _types.Sequence) and isinstance(b, _types.Sequence):
        return [f(x, y, **kwargs) for (x, y) in zip(a.data, b.data)]
    if isinstance(a, _types.Sequence):
        if isinstance(b, _types.Value):
            return [f(x, b.data, **kwargs) for x in a.data]
        if quantity.isintegral(b):
            return [f(x, b, **kwargs) for x in a.data]
        raise OperandTypeError(a, b)
    if isinstance(b, _types.Sequence):
        if isinstance(a, _types.Value):
            return [f(a.data, y, **kwargs) for y in b.data]
        if quantity.isintegral(a):
            return [f(a, y, **kwargs) for y in b.data]
        raise OperandTypeError(a, b)
    if isinstance(a, _types.Value) and isinstance(b, _types.Value):
        return f(a.data, b.data, **kwargs)
    if isinstance(a, _types.Value) and quantity.isintegral(b):
        return f(a.data, b, **kwargs)
    if quantity.isintegral(a) and isinstance(b, _types.Value):
        return f(a, b.data, **kwargs)
    raise OperandTypeError(
        "Expected operands to be instances of "
        f"{_types.Value} or {_types.Sequence}"
    ) from None


def _new_or_result(factory, x):
    """Helper for computing correct return type.

    If applying `factory` to `x` succeeds, this function will return the result.
    Otherwise, it will return `x`.
    """
    try:
        result = factory(x)
    except Exception:
        result = x
    return result


def indexer_repr(a):
    """Called for repr(a)."""
    return f"{a.__class__.__qualname__}(data={a})"


INDEXER_OPERATORS = {
    '__eq__': indexer_eq,
    '__ne__': indexer_ne,
    '__lt__': indexer_lt,
    '__le__': indexer_le,
    '__gt__': indexer_gt,
    '__ge__': indexer_ge,
    '__abs__': indexer_abs,
    '__neg__': indexer_neg,
    '__pos__': indexer_pos,
    '__add__': indexer_add,
    '__radd__': indexer_radd,
    '__sub__': indexer_sub,
    '__rsub__': indexer_rsub,
    '__mul__': indexer_mul,
    '__rmul__': indexer_rmul,
    '__truediv__': indexer_truediv,
    '__rtruediv__': indexer_rtruediv,
    '__floordiv__': indexer_floordiv,
    '__rfloordiv__': indexer_rfloordiv,
    '__mod__': indexer_mod,
    '__rmod__': indexer_rmod,
    '__pow__': indexer_pow,
    '__rpow__': indexer_rpow,
    '__repr__': indexer_repr,
}


def index_shift(a: _types.Value, __x: I, floor: I=None, ceil: I=None):
    """Shift index by a constant value."""
    index = int(a.data) + int(__x)
    if floor is not None and index < floor:
        return value_factory(floor)
    if ceil is not None and index > ceil:
        return value_factory(ceil)
    return value_factory(index)


def __index__(a: _types.Value) -> int:
    """Called for use as an index or to convert to integer type."""
    return a.data


def indices_shift(a: _types.Sequence, __x: I, floor: I=None, ceil: I=None):
    """Shift indices by a constant value."""
    indices = numpy.array(a.data) + int(__x)
    if floor is not None:
        indices[indices < floor] = floor
    if ceil is not None:
        indices[indices > ceil] = ceil
    return sequence_factory(indices)


def __iter__(a: _types.Sequence):
    """Called for iter(a)."""
    # XXX: Should instances be iterators rather than iterables? See
    # https://stackoverflow.com/a/24377 for more information and suggestions.
    yield from a.data


def __len__(a: _types.Sequence):
    """Called for len(a)."""
    return len(a.data)


def __getitem__(a: _types.Sequence, i: typing.SupportsIndex, /):
    """Called for a[i]."""
    data = a.data[i]
    try:
        len(data)
    except TypeError:
        result = value_factory(data)
    else:
        result = sequence_factory(data)
    return result


def value_str(a: _types.Value):
    """Called for str(a)."""
    return str(a.data)


def value_factory(x, /):
    """Create a single numeric index value."""
    parsed = index_args(x)
    if isinstance(parsed, typing.SupportsInt):
        if isinstance(parsed, numpy.ndarray) and parsed.size == 1:
            return _value_factory(int(parsed[0]))
        return _value_factory(int(parsed))
    raise TypeError(
        f"Cannot convert {x} to a single integer"
    ) from None


VALUE_OPERATORS = {
    **INDEXER_OPERATORS,
    'shift': index_shift,
    '__index__': __index__,
    '__int__': numeric.operators.__int__,
    '__float__': numeric.operators.__float__,
    '__complex__': numeric.operators.__complex__,
    '__str__': value_str,
}


def _value_factory(data):
    """Create a new index value."""
    Type = type('Value', (_types.Value,), VALUE_OPERATORS)
    return Type(data)


def sequence_factory(x, /):
    """Create a sequence of numeric index values."""
    parsed = indices_args(x)
    if all(isinstance(i, typing.SupportsInt) for i in parsed):
        return _sequence_factory(numpy.array(parsed, dtype=int))
    raise TypeError(
        f"The numerical values of {x} do not all support"
        " conversion to integral type"
    ) from None


def sequence_str(a: _types.Sequence):
    """Called for str(a)."""
    if len(a.data) <= 4:
        return ', '.join(str(i) for i in a.data)
    indices = [*a.data[:2], '...', *a.data[-2:]]
    return f"[{', '.join(str(i) for i in indices)}]"


SEQUENCE_OPERATORS = {
    **INDEXER_OPERATORS,
    'shift': indices_shift,
    '__len__': numeric.operators.__len__,
    '__contains__': numeric.operators.__contains__,
    '__array__': numeric.operators.__array__,
    '__iter__': __iter__,
    '__getitem__': __getitem__,
    '__str__': sequence_str,
}


def _sequence_factory(data):
    """Create a new index sequence."""
    Type = type('Sequence', (_types.Sequence,), SEQUENCE_OPERATORS)
    return Type(data)


def index_args(x, /):
    """Parse arguments to initialize `~_types.Value`."""
    index = _index_args(x)
    if index is not None:
        return index
    indices = _indices_args(x)
    if indices is not None:
        return indices[0]
    raise TypeError(
        f"Cannot initialize {_types.Value} from {x}"
        f", which has type {type(x)}"
    ) from None


def indices_args(x, /):
    """Parse arguments to initialize `~_types.Sequence`."""
    indices = _indices_args(x)
    if indices is not None:
        return indices
    index = _index_args(x)
    if index is not None:
        return [index]
    raise TypeError(
        f"Cannot initialize {_types.Sequence} from {x}"
        f", which has type {type(x)}"
    ) from None


def _index_args(x, /):
    """Helper for `index_args`."""
    if isinstance(x, numpy.ndarray) and x.dtype == int:
        if x.size == 1:
            v = x[0] if x.ndim == 1 else x
            return int(v)
        raise TypeError(
            f"Can only initialize {_types.Value} from a"
            " zero- or one-dimensional array-like object"
        ) from None
    if isinstance(x, (numbers.Integral, numpy.integer)):
        return x
    if isinstance(x, (bytes, str)):
        try:
            index = int(x)
        except ValueError:
            return # Force a TypeError in factory
        return index
    if isinstance(x, numeric.Measurement):
        if x.isunitless:
            return x.data
        raise ValueError(
            f"Can only initialize {_types.Value} from a unitless object"
        ) from None


def _indices_args(x, /):
    """Helper for `indices_args`."""
    if isinstance(x, range):
        return x
    if isinstance(x, (tuple, list)):
        if all(isinstance(i, str) for i in x):
            with contextlib.suppress(ValueError):
                return [int(i) for i in x]
        if all(isinstance(i, (numbers.Integral, numpy.integer)) for i in x):
            return x
    if isinstance(x, numpy.ndarray) and x.dtype == int:
        if x.size == 1:
            v = x[0] if x.ndim == 1 else x
            return numpy.array(int(v), ndmin=1)
        y = x.squeeze()
        ndim = y.ndim
        if ndim == 1:
            return y
        raise TypeError(
            f"Cannot initialize {_types.Sequence} from a {ndim}-D array"
        ) from None
    if isinstance(x, (numbers.Integral, numpy.integer)):
        return [x]
    if isinstance(x, _types.Sequence):
        return x.data
    if isinstance(x, numeric.Measurement):
        if x.isunitless:
            return _indices_args(x.data)
        raise ValueError(
            f"The argument to {_types.Sequence} must be unitless."
        ) from None





