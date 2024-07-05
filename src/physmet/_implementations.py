import collections.abc
import functools
import math
import numbers
import typing

import numpy

from . import data
from . import indexer
from . import metric
from . import measurable
from . import measured
from . import numeric
from ._axes import (
    Axes,
    axes_factory,
)
from ._types import (
    Array,
    Object,
    ObjectType,
    Scalar,
    Tensor,
    Vector,
)


def require_object(f: typing.Callable):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not any(isinstance(arg, Object) for arg in args):
            types = ', '.join(str(type(arg)) for arg in args)
            raise TypeError(
                f"Cannot apply {f.__name__} to object(s) with type(s) {types}"
            ) from None
        return f(*args, **kwargs)
    return wrapper


@require_object
def object_dunder_eq(a: Object, b) -> bool:
    """Compute a == b."""
    if isinstance(b, Object):
        return numpy.array_equal(a.data, b.data) and a.unit == b.unit
    return False


@require_object
def object_dunder_ne(a: Object, b) -> bool:
    """Compute a != b."""
    return not (a == b)


@require_object
def object_dunder_lt(a: Object, b) -> typing.Union[bool, typing.Iterable[bool]]:
    """Compute a < b."""
    if isinstance(b, Object):
        if a.unit == b.unit:
            return a.data < b.data
        raise ValueError("Cannot compare objects with different units")
    return NotImplemented


@require_object
def object_dunder_le(a: Object, b) -> typing.Union[bool, typing.Iterable[bool]]:
    """Compute a <= b."""
    if isinstance(b, Object):
        if a.unit == b.unit:
            return a.data <= b.data
        raise ValueError("Cannot compare objects with different units")
    return NotImplemented


@require_object
def object_dunder_gt(a: Object, b) -> typing.Union[bool, typing.Iterable[bool]]:
    """Compute a > b."""
    if isinstance(b, Object):
        if a.unit == b.unit:
            return a.data > b.data
        raise ValueError("Cannot compare objects with different units")
    return NotImplemented


@require_object
def object_dunder_ge(a: Object, b) -> typing.Union[bool, typing.Iterable[bool]]:
    """Compute a >= b."""
    if isinstance(b, Object):
        if a.unit == b.unit:
            return a.data >= b.data
        raise ValueError("Cannot compare objects with different units")
    return NotImplemented


@require_object
def object_dunder_abs(a: ObjectType) -> ObjectType:
    """Compute abs(a)."""
    return _get_factory(a)(abs(a.data), a.unit)


@require_object
def object_dunder_pos(a: ObjectType) -> ObjectType:
    """Compute +a."""
    return _get_factory(a)(+a.data, a.unit)


@require_object
def object_dunder_neg(a: ObjectType) -> ObjectType:
    """Compute -a."""
    return _get_factory(a)(-a.data, a.unit)


NUMBER_TYPES = (numbers.Number, numpy.number)


@require_object
def object_dunder_add(
    a: typing.Union[ObjectType, typing.Any],
    b: typing.Union[ObjectType, typing.Any],
) -> ObjectType:
    """Compute a + b."""
    if not any(isinstance(i, Object) for i in (a, b)):
        raise TypeError(
            "Cannot this function to objects with types"
            f" {type(a)} and {type(b)}"
        ) from None
    _unit_type_error = TypeError("Cannot add a number to a unitful object")
    if isinstance(a, Object):
        if not a.isunitless and isinstance(b, NUMBER_TYPES):
            raise _unit_type_error from None
    if isinstance(b, Object):
        if not b.isunitless and isinstance(a, NUMBER_TYPES):
            raise _unit_type_error from None
    x = a if isinstance(a, Object) else measurable.measure(a)
    y = b if isinstance(b, Object) else measurable.measure(b)
    if not x.unit == y.unit:
        raise ValueError("Cannot add objects with different units")
    d = x.data + y.data
    unit = x.unit
    factory = _get_factory(x, y)
    if isinstance(x, Array):
        if isinstance(y, Vector):
            raise TypeError("Cannot add a vector to an array") from None
        axes = x.axes + y.axes if isinstance(y, Array) else x.axes
        return factory(d, unit, axes)
    return factory(d, unit)


@require_object
def object_dunder_radd(
    a: typing.Union[ObjectType, typing.Any],
    b: typing.Union[ObjectType, typing.Any],
) -> ObjectType:
    """Compute b + a."""
    return object_dunder_add(b, a)


@require_object
def object_dunder_sub(
    a: typing.Union[ObjectType, typing.Any],
    b: typing.Union[ObjectType, typing.Any],
) -> ObjectType:
    """Compute a - b."""
    _unit_type_error = TypeError(
        "Cannot subtract a number from a unitful object"
    )
    if isinstance(a, Object):
        if not a.isunitless and isinstance(b, NUMBER_TYPES):
            raise _unit_type_error from None
    if isinstance(b, Object):
        if not b.isunitless and isinstance(a, NUMBER_TYPES):
            raise _unit_type_error from None
    x = a if isinstance(a, Object) else measurable.measure(a)
    y = b if isinstance(b, Object) else measurable.measure(b)
    if not x.unit == y.unit:
        raise ValueError("Cannot subtract objects with different units")
    d = x.data - y.data
    unit = x.unit
    factory = _get_factory(x, y)
    if isinstance(x, Array):
        if isinstance(y, Vector):
            raise TypeError(
                "Cannot subtract a vector from an array"
            ) from None
        axes = x.axes + y.axes if isinstance(y, Array) else x.axes
        return factory(d, unit, axes)
    return factory(d, unit)


@require_object
def object_dunder_rsub(
    a: typing.Union[ObjectType, typing.Any],
    b: typing.Union[ObjectType, typing.Any],
) -> ObjectType:
    """Compute b - a."""
    return object_dunder_sub(b, a)


@require_object
def object_dunder_mul(
    a: typing.Union[ObjectType, typing.Any],
    b: typing.Union[ObjectType, typing.Any],
) -> ObjectType:
    """Compute a * b."""
    if all(isinstance(i, numeric.Measurement) for i in (a, b)):
        unit = a.unit * b.unit
        if isinstance(a, Array) and isinstance(b, Array):
            return array_factory(a.data * b.data, unit, a.axes | b.axes)
        if isinstance(a, Array):
            return array_factory(a.data.array * b.data, unit, a.axes)
        if isinstance(b, Array):
            return array_factory(a.data * b.data.array, unit, b.axes)
        return _get_factory(a, b)(a.data * b.data, unit)
    if isinstance(a, Object) and isinstance(b, numbers.Real):
        unit = a.unit
        if isinstance(a, Array):
            return array_factory(a.data.array * b, unit, a.axes)
        return _get_factory(a, b)(a.data * b, unit)
    if isinstance(a, numbers.Real) and isinstance(b, Object):
        unit = b.unit
        if isinstance(b, Array):
            return array_factory(a * b.data.array, unit, b.axes)
        return _get_factory(a, b)(a * b.data, unit)
    x = a if isinstance(a, Object) else measurable.measure(a)
    y = b if isinstance(b, Object) else measurable.measure(b)
    return object_dunder_mul(x, y)


@require_object
def object_dunder_rmul(
    a: typing.Union[ObjectType, typing.Any],
    b: typing.Union[ObjectType, typing.Any],
) -> ObjectType:
    """Compute b * a."""
    return object_dunder_mul(b, a)


@require_object
def object_dunder_truediv(
    a: typing.Union[ObjectType, typing.Any],
    b: typing.Union[ObjectType, typing.Any],
) -> ObjectType:
    """Compute a / b."""
    if all(isinstance(i, numeric.Measurement) for i in (a, b)):
        unit = a.unit / b.unit
        if isinstance(a, Array) and isinstance(b, Array):
            return array_factory(a.data / b.data, unit, a.axes | b.axes)
        if isinstance(a, Array):
            return array_factory(a.data.array / b.data, unit, a.axes)
        if isinstance(b, Array):
            return array_factory(a.data / b.data.array, unit, b.axes)
        return _get_factory(a, b)(a.data / b.data, unit)
    if isinstance(a, Object) and isinstance(b, numbers.Real):
        unit = a.unit
        if isinstance(a, Array):
            return array_factory(a.data.array / b, unit, a.axes)
        return _get_factory(a, b)(a.data / b, unit)
    if isinstance(a, numbers.Real) and isinstance(b, Object):
        unit = 1 / b.unit
        if isinstance(b, Array):
            return array_factory(a / b.data.array, unit, b.axes)
        return _get_factory(a, b)(a / b.data, unit)
    x = a if isinstance(a, Object) else measurable.measure(a)
    y = b if isinstance(b, Object) else measurable.measure(b)
    return object_dunder_truediv(x, y)


@require_object
def object_dunder_rtruediv(
    a: typing.Union[ObjectType, typing.Any],
    b: typing.Union[ObjectType, typing.Any],
) -> ObjectType:
    """Compute b / a."""
    return object_dunder_truediv(b, a)


@require_object
def object_dunder_floordiv(
    a: typing.Union[ObjectType, typing.Any],
    b: typing.Union[ObjectType, typing.Any],
) -> ObjectType:
    """Compute a // b."""
    if all(isinstance(i, numeric.Measurement) for i in (a, b)):
        unit = a.unit / b.unit
        if isinstance(a, Array) and isinstance(b, Array):
            return array_factory(a.data // b.data, unit, a.axes | b.axes)
        if isinstance(a, Array):
            return array_factory(a.data.array // b.data, unit, a.axes)
        if isinstance(b, Array):
            return array_factory(a.data // b.data.array, unit, b.axes)
        return _get_factory(a, b)(a.data // b.data, unit)
    if isinstance(a, Object) and isinstance(b, numbers.Real):
        unit = a.unit
        if isinstance(a, Array):
            return array_factory(a.data.array // b, unit, a.axes)
        return _get_factory(a, b)(a.data // b, unit)
    if isinstance(a, numbers.Real) and isinstance(b, Object):
        unit = 1 / b.unit
        if isinstance(b, Array):
            return array_factory(a // b.data.array, unit, b.axes)
        return _get_factory(a, b)(a // b.data, unit)
    x = a if isinstance(a, Object) else measurable.measure(a)
    y = b if isinstance(b, Object) else measurable.measure(b)
    return object_dunder_floordiv(x, y)


@require_object
def object_dunder_rfloordiv(
    a: typing.Union[ObjectType, typing.Any],
    b: typing.Union[ObjectType, typing.Any],
) -> ObjectType:
    """Compute b // a."""
    return object_dunder_floordiv(b, a)


@require_object
def object_dunder_mod(
    a: typing.Union[ObjectType, typing.Any],
    b: typing.Union[ObjectType, typing.Any],
) -> ObjectType:
    """Compute a % b."""
    if all(isinstance(i, numeric.Measurement) for i in (a, b)):
        unit = a.unit / b.unit
        if isinstance(a, Array) and isinstance(b, Array):
            return array_factory(a.data % b.data, unit, a.axes | b.axes)
        if isinstance(a, Array):
            return array_factory(a.data.array % b.data, unit, a.axes)
        if isinstance(b, Array):
            return array_factory(a.data % b.data.array, unit, b.axes)
        return _get_factory(a, b)(a.data % b.data, unit)
    if isinstance(a, Object) and isinstance(b, numbers.Real):
        unit = a.unit
        if isinstance(a, Array):
            return array_factory(a.data.array % b, unit, a.axes)
        return _get_factory(a, b)(a.data % b, unit)
    if isinstance(a, numbers.Real) and isinstance(b, Object):
        unit = 1 / b.unit
        if isinstance(b, Array):
            return array_factory(a % b.data.array, unit, b.axes)
        return _get_factory(a, b)(a % b.data, unit)
    x = a if isinstance(a, Object) else measurable.measure(a)
    y = b if isinstance(b, Object) else measurable.measure(b)
    return object_dunder_mod(x, y)


@require_object
def object_dunder_rmod(
    a: typing.Union[ObjectType, typing.Any],
    b: typing.Union[ObjectType, typing.Any],
) -> ObjectType:
    """Compute b % a."""
    return object_dunder_mod(b, a)


@require_object
def object_dunder_pow(
    a: typing.Union[ObjectType, typing.Any],
    b: typing.Union[ObjectType, typing.Any],
    mod: int=None,
) -> ObjectType:
    """Compute a ** b."""
    if isinstance(b, numeric.Measurement) and not b.isunitless:
        raise ValueError(
            "Cannot compute a ** b unless b is unitless"
        ) from None
    if isinstance(a, Object) and isinstance(b, Object):
        if isinstance(b, Scalar):
            p = b.data
            unit = pow(a.unit, p, mod=mod)
            if isinstance(a, Array):
                data = pow(a.data.array, p, mod=mod)
                return array_factory(data, unit, a.axes)
            return _get_factory(a)(pow(a.data, p, mod=mod), unit)
        if not a.isunitless:
            raise ValueError(
                "Cannot compute a ** b when b is array-like"
                " unless both a and b are unitless"
            )
        if isinstance(a, Array) and isinstance(b, Array):
            data = pow(a.data.array, b.data.array, mod=mod)
            return array_factory(data, '1', a.axes)
        if isinstance(a, Array):
            data = pow(a.data.array, b.data, mod=mod)
            return array_factory(data, '1', a.axes)
        if isinstance(b, Array):
            data = pow(a.data, b.data.array, mod=mod)
            return array_factory(data, '1', b.axes)
        return _get_factory(a, b)(pow(a.data, b.data, mod=mod), '1')
    if isinstance(a, Object) and isinstance(b, numbers.Real):
        unit = pow(a.unit, b)
        if isinstance(a, Array):
            data = pow(a.data.array, b, mod=mod)
            return array_factory(data, unit, a.axes)
        return _get_factory(a)(pow(a.data, b, mod=mod), unit)
    if isinstance(a, (numbers.Real, numpy.ndarray)) and isinstance(b, Object):
        if isinstance(b, Array):
            return pow(a, b.data.array, mod=mod)
        return pow(a, b.data, mod=mod)
    return NotImplemented


@require_object
def object_dunder_rpow(
    a: typing.Union[ObjectType, typing.Any],
    b: typing.Union[ObjectType, typing.Any],
) -> ObjectType:
    """Compute b ** a."""
    return object_dunder_pow(b, a)


def implement(operation: typing.Callable):
    """Register `operation` as the implementation for all object types."""
    @functools.wraps(operation)
    def wrapped(f: typing.Callable):
        Scalar.implements(operation)(f)
        Vector.implements(operation)(f)
        Tensor.implements(operation)(f)
        Array.implements(operation)(f)
        return f
    return wrapped


@Vector.implements(numpy.power)
@Tensor.implements(numpy.power)
@Array.implements(numpy.power)
def power(a, b, **kwargs):
    """Called for numpy.power(a, b)."""
    # NOTE: We implement this along with `object_dunder_pow` (for `__pow__`) and
    # `object_dunder_rpow` (for `__rpow__`) because we need to catch `__rpow__`
    # cases when the left operand is a `numpy.ndarray`.
    if isinstance(b, (Vector, Tensor, Array)) and not b.isunitless:
        raise ValueError(
            "Cannot compute a ** b unless b is unitless"
        ) from None
    x = a.data if isinstance(a, numeric.Quantity) else a
    y = b.data if isinstance(b, numeric.Quantity) else b
    return numpy.power(x, y, **kwargs)


@implement(numpy.sin)
def sin(x: Object[numeric.RealValueType]):
    """Compute the sin of `x`."""
    return _apply_trig(numpy.sin, x)


@implement(numpy.cos)
def cos(x: Object[numeric.RealValueType]):
    """Compute the cos of `x`."""
    return _apply_trig(numpy.cos, x)


@implement(numpy.tan)
def tan(x: Object[numeric.RealValueType]):
    """Compute the tan of `x`."""
    return _apply_trig(numpy.tan, x)


def _apply_trig(f, x: Object[numeric.RealValueType]):
    """Compute a trigonometric function of `x`."""
    if str(x.unit) in {'rad', 'deg'}:
        metadata = {'unit': '1'}
        if isinstance(x, Array):
            metadata['axes'] = x.axes
        return _get_factory(x)(f(x.data), **metadata)
    strf = f.__name__
    raise ValueError(
        f"Cannot compute {strf}(x)"
        f" when the unit of x is {str(x.unit)!r}"
    ) from None


@implement(numpy.sqrt)
def sqrt(x: Object[numeric.RealValueType]):
    """Compute the square root of `x`."""
    data = numpy.sqrt(x.data)
    metadata = {'unit': x.unit ** 0.5}
    if isinstance(x, Array):
        metadata['axes'] = x.axes
    return _get_factory(x)(data, **metadata)


@implement(numpy.log)
def log(x: Object[numeric.RealValueType]):
    """Compute the natural log of `x`."""
    return _apply_log(numpy.log, x)


@implement(numpy.log10)
def log10(x: Object[numeric.RealValueType]):
    """Compute the base-10 log of `x`."""
    return _apply_log(numpy.log10, x)


@implement(numpy.log2)
def log2(x: Object[numeric.RealValueType]):
    """Compute the base-2 log of `x`."""
    return _apply_log(numpy.log2, x)


@implement(numpy.log1p)
def log1p(x: Object[numeric.RealValueType]):
    """Compute the natural log of `x + 1`."""
    return _apply_log(numpy.log1p, x)


def _apply_log(f, x: Object[numeric.RealValueType]):
    """Compute a logarithmic function of `x`."""
    if str(x.unit) == '1':
        metadata = {'unit': '1'}
        if isinstance(x, Array):
            metadata['axes'] = x.axes
        return _get_factory(x)(f(x.data), **metadata)
    strf = f.__name__
    raise ValueError(
        f"Cannot compute {strf}(x) when x is not unitless"
    ) from None


@implement(numpy.squeeze)
def squeeze(x: Object[numeric.RealValueType], **kwargs):
    """Remove singular axes.

    Notes
    -----
    - Returning a scalar from a singular array is consistent with applying
      `numpy.squeeze` to a singular `numpy.ndarray`. For example,

      >>> numpy.squeeze(numpy.array([2])).ndim
      0
    """
    if isinstance(x, Scalar):
        return x
    data = numpy.squeeze(x.data, **kwargs)
    if isinstance(data, (numbers.Real)):
        return scalar_factory(data, unit=x.unit)
    if isinstance(data, numpy.ndarray) and data.ndim == 0:
        return scalar_factory(numpy.ravel(data)[0], unit=x.unit)
    metadata = {'unit': x.unit}
    if isinstance(x, Array):
        metadata['axes'] = {d: x.axes[d] for d in data.dimensions}
    return _get_factory(x)(data, **metadata)

@implement(numpy.mean)
def mean(x: Object[numeric.RealValueType], **kwargs):
    """Compute the mean of `x`."""
    data = numpy.mean(x.data, **kwargs)
    if isinstance(x, Scalar) or 'axis' not in kwargs:
        return scalar_factory(data, unit=x.unit)
    metadata = {'unit': x.unit}
    if isinstance(x, Array):
        metadata['axes'] = {d: x.axes[d] for d in data.dimensions}
    return _get_factory(x)(data, **metadata)


@implement(numpy.sum)
def sum(x: Object[numeric.RealValueType], **kwargs):
    """Compute the sum of `x`."""
    data = numpy.sum(x.data, **kwargs)
    if isinstance(x, Scalar) or 'axis' not in kwargs:
        return scalar_factory(data, unit=x.unit)
    metadata = {'unit': x.unit}
    if isinstance(x, Array):
        metadata['axes'] = {d: x.axes[d] for d in data.dimensions}
    return _get_factory(x)(data, **metadata)


@implement(numpy.cumsum)
def cumsum(x: Object[numeric.RealValueType], **kwargs):
    """Compute the cumulative sum of `x`.

    Notes
    -----
    - Calling plain `numpy.cumsum` on a number, on a 1-D array, and without
      passing an argument to `axis` all produces a 1-D array. Calling the same
      function on an N-D array (N >= 1) and including an argument to `axis`
      produces an N-D array. That is why calling this implementation on a
      scalar, on a vector, and without passing an argument to `axis` produces a
      vector whereas calling it on a vector, tensor, or array and including an
      argument to `axis` produces an instance of the same object. Note that
      calling this implementation with `axis=0` or `axis=-1` on a scalar will
      produce a vector and doing so with any argument to `axis` will raise an
      exception. This behavior is also consistent with the plain `numpy.cumsum`
      implementation.
    """
    data = numpy.cumsum(x.data, **kwargs)
    if isinstance(x, (Scalar, Vector)) or 'axis' not in kwargs:
        return vector_factory(data, unit=x.unit)
    metadata = {'unit': x.unit}
    if isinstance(x, Array):
        metadata['axes'] = {d: x.axes[d] for d in data.dimensions}
    return _get_factory(x)(data, **metadata)


@implement(numpy.transpose)
def array_transpose(x: Object[numeric.RealValueType], **kwargs):
    """Compute the transpose of `x`."""
    data = numpy.transpose(x.data, **kwargs)
    if isinstance(x, Scalar):
        return scalar_factory(numpy.ravel(data)[0], unit=x.unit)
    metadata = {'unit': x.unit}
    if isinstance(x, Array):
        metadata['axes'] = {d: x.axes[d] for d in data.dimensions}
    return _get_factory(x)(data, **metadata)


@implement(numpy.gradient)
def gradient(x: Object[numeric.RealValueType], *args, **kwargs):
    """Compute the gradient of an array."""
    if isinstance(x, Scalar):
        return []
    if not args:
        return _apply_gradient(x.unit, x, **kwargs)
    diffs = []
    units = []
    for arg in args:
        if isinstance(arg, Scalar):
            diffs.append(float(arg))
            units.append(arg.unit)
        elif isinstance(arg, (Vector, Tensor, Array)):
            diffs.append(numpy.array(arg))
            units.append(arg.unit)
        elif isinstance(arg, numbers.Real):
            diffs.append(float(arg))
            units.append('1')
        else:
            diffs.append(numpy.array(arg))
            units.append('1')
    u0 = units[0]
    if all(unit == u0 for unit in units[1:]):
        return _apply_gradient(x.unit / u0, x, *diffs, **kwargs)
    raise TypeError("Inconsistent units in gradient coordinates") from None


def _apply_gradient(unit, x, *args, **kwargs):
    """Helper for custom implementation of `numpy.gradient`."""
    gradient = numpy.gradient(numpy.array(x), *args, **kwargs)
    metadata = {'unit': unit}
    if isinstance(x, Array):
        metadata['axes'] = x.axes
    if isinstance(gradient, list):
        return [_get_factory(x)(array, **metadata) for array in gradient]
    return _get_factory(x)(gradient, **metadata)


@Array.implements(numpy.trapz)
def trapz(x: Object[numeric.RealValueType], *args, **kwargs):
    """Integrate `x` via the composite trapezoidal rule."""
    if isinstance(x, Scalar):
        raise TypeError(
            "Cannot apply numpy.trapz to a scalar object"
        ) from None
    data = numpy.trapz(x.data, *args, **kwargs)
    if isinstance(data, (numbers.Real, numpy.number)):
        return data
    metadata = {'unit': x.unit}
    if isinstance(x, Array):
        metadata['axes'] = {d: x.axes[d] for d in data.dimensions}
    return _get_factory(x)(data, **metadata)


def _get_factory(*args):
    """Get the appropriate object factory for `args`."""
    if any(isinstance(arg, Array) for arg in args):
        return array_factory
    if any(isinstance(arg, Tensor) for arg in args):
        return tensor_factory
    if any(isinstance(arg, Vector) for arg in args):
        return vector_factory
    if any(isinstance(arg, Scalar) for arg in args):
        return scalar_factory
    raise TypeError(
        "Expected at least one argument to be"
        " an Array, Tensory, Vector, or Scalar"
    ) from None


def object_dunder_array(a: Object, *args, **kwargs) -> numpy.ndarray:
    """Called for numpy.array(a)."""
    data = a.data
    return numpy.array(data, *args, **kwargs)


def object_withunit(a: ObjectType, unit: metric.UnitLike) -> ObjectType:
    """Convert this object to a new unit."""
    if converted := _object_convert(a, unit):
        return converted
    t = a.__class__.__qualname__
    raise TypeError(f"Cannot change unit on object of type {t!r}")


def _object_convert(a: ObjectType, unit) -> ObjectType:
    """Helper for `object_withunit`."""
    if a.unit == unit:
        return a
    c = measured.conversion(a, unit=unit)
    scale = float(c)
    if isinstance(a, Array):
        new = array_factory(a._data_interface, unit=c.new, axes=a.axes)
        new._scale = scale
        return new
    if isinstance(a, (Vector, Tensor)):
        return _get_factory(a)(numpy.array(a.data) * scale, unit=c.new)
    if isinstance(a, Scalar):
        return scalar_factory(float(a) * scale, unit=c.new)


def object_dunder_measure(a: Object) -> numeric.Measurement:
    """Called for measure(a)."""
    if isinstance(a, Tensor):
        return measurable.measurement(
            numpy.array(a.data).flatten(),
            unit=a.unit
        )
    return measurable.measurement(a.data, unit=a.unit)


def object_get_numpy_array(a: Object) -> numpy.ndarray:
    """Called to access an equivalent `numpy.ndarray`."""
    return numpy.array(a.data)


OBJECT_OPERATORS = {
    '__eq__': object_dunder_eq,
    '__ne__': object_dunder_ne,
    '__lt__': object_dunder_lt,
    '__le__': object_dunder_le,
    '__gt__': object_dunder_gt,
    '__ge__': object_dunder_ge,
    '__abs__': object_dunder_abs,
    '__pos__': object_dunder_pos,
    '__neg__': object_dunder_neg,
    '__add__': object_dunder_add,
    '__sub__': object_dunder_sub,
    '__mul__': object_dunder_mul,
    '__truediv__': object_dunder_truediv,
    '__floordiv__': object_dunder_floordiv,
    '__mod__': object_dunder_mod,
    '__pow__': object_dunder_pow,
    '__radd__': object_dunder_radd,
    '__rsub__': object_dunder_rsub,
    '__rmul__': object_dunder_rmul,
    '__rtruediv__': object_dunder_rtruediv,
    '__rfloordiv__': object_dunder_rfloordiv,
    '__rmod__': object_dunder_rmod,
    '__rpow__': object_dunder_rpow,
    'withunit': object_withunit,
    '__measure__': object_dunder_measure,
    '__array__': object_dunder_array,
}


@require_object
def scalar_dunder_round(a: Object, ndigits: int=None):
    """Compute round(a)."""
    return scalar_factory(round(a.data, ndigits=ndigits), unit=a.unit)


@require_object
def scalar_dunder_floor(a: Object):
    return scalar_factory(math.floor(a.data), unit=a.unit)


@require_object
def scalar_dunder_ceil(a: Object):
    return scalar_factory(math.ceil(a.data), unit=a.unit)


@require_object
def scalar_dunder_trunc(a: Object):
    return scalar_factory(math.trunc(a.data), unit=a.unit)


@require_object
def vector_dunder_iter(a: Object):
    """Called for iter(a)."""
    return _generate_scalars(a.data, a.unit)


@require_object
def vector_dunder_getitem(a: Object, i, /):
    """Called for a[i]."""
    d = a.data[i]
    try:
        return scalar_factory(d, unit=a.unit)
    except TypeError:
        return vector_factory(d, unit=a.unit)


@require_object
def tensor_dunder_iter(a: Object):
    """Called for iter(a)."""
    return _generate_scalars(numpy.array(a.data).flat, a.unit)


@require_object
def tensor_dunder_getitem(a: Object, i, /):
    """Called for a[i]."""
    d = a.data[i]
    try:
        return scalar_factory(d, unit=a.unit)
    except TypeError:
        return tensor_factory(d, unit=a.unit)


@require_object
def array_dunder_eq(a: Array, b):
    """Called for a == b when a is an instance of `~Array`."""
    object_result = object_dunder_eq(a, b)
    if isinstance(b, Array) and (a.axes != b.axes):
        return False
    return object_result


@require_object
def array_dunder_abs(a: Array) -> Array:
    """Compute abs(a) on an instance of `~Array`."""
    return array_factory(abs(a.data), a.unit, a.axes)


@require_object
def array_dunder_pos(a: Array) -> Array:
    """Compute +a on an instance of `~Array`."""
    return array_factory(+a.data, a.unit, a.axes)


@require_object
def array_dunder_neg(a: Array) -> Array:
    """Compute -a on an instance of `~Array`."""
    return array_factory(-a.data, a.unit, a.axes)


@require_object
def array_dunder_int(a: Array):
    """Called for int(a)."""
    return array_cast_to(int, a)


@require_object
def array_dunder_float(a: Array):
    """Called for float(a)."""
    return array_cast_to(float, a)


@require_object
def array_dunder_complex(a: Array):
    """Called for complex(a)."""
    return complex(float(a))


T = typing.TypeVar('T')


def array_cast_to(t: T, a: Array):
    """Convert this instance to a built-in numeric type."""
    if a.size == 1:
        return t(a.data.array.ravel()[0])
    raise TypeError(
        f"Cannot convert a size-{a.size} array to {t!r}"
    ) from None


@require_object
def array_dunder_getitem(a: Array, args, /):
    idx = indexer.expand(a.ndim, args)
    data = a.data[idx]
    axes = {k: v[i] for i, (k, v) in zip(idx, a.axes.items())}
    return array_factory(data, unit=a.unit, axes=axes)


@typing.overload
def array_transpose(
    a: Array,
    axes: typing.Iterable[typing.Union[str, typing.SupportsIndex]],
    /,
) -> typing.Self: ...

@typing.overload
def array_transpose(
    a: Array,
    *axes: typing.Union[str, typing.SupportsIndex],
) -> typing.Self: ...

@require_object
def array_transpose(a: Array, *args):
    """Transpose this array."""
    if not args:
        return a
    data = a._data_interface.transpose(*args)
    axes = a.axes.permute(*args)
    return array_factory(data, unit=a.unit, axes=axes)


@Array.implements(numpy.array_equal)
def array_equal(a: Array, b: Array) -> bool:
    """Called for numpy.array_equal(a, b)."""
    return numpy.array_equal(numpy.array(a), numpy.array(b))


def _generate_scalars(dataiter, unit):
    """Helper for iterator functions."""
    return (scalar_factory(x, unit=unit) for x in dataiter)


SCALAR_OPERATORS = {
    **OBJECT_OPERATORS,
    '__int__': numeric.operators.__int__,
    '__float__': numeric.operators.__float__,
    '__complex__': numeric.operators.__complex__,
    '__round__': scalar_dunder_round,
    '__floor__': scalar_dunder_floor,
    '__ceil__': scalar_dunder_ceil,
    '__trunc__': scalar_dunder_trunc,
    '__measure__': object_dunder_measure,
    '_get_numpy_array': object_get_numpy_array,
}


def scalar_factory(x, unit=None):
    """Create a new scalar."""
    Type = type('Scalar', (Scalar,), SCALAR_OPERATORS)
    d, u = scalar_args(x, unit)
    return Type(d, unit=metric.unit(u or '1'))


def scalar_args(x, unit, /):
    """Parse arguments to initialize `~Scalar`."""
    if isinstance(x, Object) and unit is not None:
        raise ValueError(
            "Cannot change unit via object creation"
            ". Please use the 'withunit' function."
        ) from None
    if isinstance(x, Scalar):
        return x.data, x.unit
    if isinstance(x, numbers.Real):
        return x, unit
    a = numpy.array(x)
    s = a.size
    if s != 1:
        raise TypeError(
            f"Cannot create a scalar from data with size {s}"
        ) from None
    data = a[0]
    if isinstance(x, (Vector, Tensor)):
        return data, x.unit
    return data, unit


VECTOR_OPERATORS = {
    **OBJECT_OPERATORS,
    '__contains__': numeric.operators.__contains__,
    '__len__': numeric.operators.__len__,
    '__array__': numeric.operators.__array__,
    '__iter__': vector_dunder_iter,
    '__getitem__': vector_dunder_getitem,
    '__measure__': object_dunder_measure,
    '_get_numpy_array': object_get_numpy_array,
}


def vector_factory(data, unit=None):
    """Create a new vector."""
    Type = type('Vector', (Vector, collections.abc.Sequence), VECTOR_OPERATORS)
    d, u = vector_args(data, unit)
    return Type(d, unit=metric.unit(u or '1'))


def vector_args(x, unit, /):
    """Parse arguments to initialize `~Vector`."""
    if isinstance(x, Object) and unit is not None:
        raise ValueError(
            "Cannot change unit via object creation"
            ". Please use the 'withunit' function."
        ) from None
    if isinstance(x, Vector):
        return x.data, x.unit
    if isinstance(x, (numeric.Measurement, Scalar)):
        y = [x.data] if isinstance(x, Scalar) else x
        return vector_args(numpy.asarray(y), x.unit)
    if isinstance(x, Tensor):
        return vector_args(numpy.array(x.data), x.unit)
    a = numpy.asarray(x)
    if a.ndim != 1:
        raise TypeError(
            "Vector data must be one-dimensional"
        ) from None
    return a, unit


TENSOR_OPERATORS = {
    **OBJECT_OPERATORS,
    '__contains__': numeric.operators.__contains__,
    '__len__': numeric.operators.__len__,
    '__array__': numeric.operators.__array__,
    '__iter__': tensor_dunder_iter,
    '__getitem__': tensor_dunder_getitem,
    '__measure__': object_dunder_measure,
    '_get_numpy_array': object_get_numpy_array,
}


def tensor_factory(data, unit=None):
    """Create a new tensor."""
    Type = type('Tensor', (Tensor, collections.abc.Sequence), TENSOR_OPERATORS)
    d, u = tensor_args(data, unit)
    return Type(d, unit=metric.unit(u or '1'))


def tensor_args(x, unit, /):
    """Parse arguments to initialize `~Tensor`."""
    if isinstance(x, Object) and unit is not None:
        raise ValueError(
            "Cannot change unit via object creation"
            ". Please use the 'withunit' function."
        ) from None
    if isinstance(x, Array): # NOTE: An Array is a Tensor.
        return x.data.array, x.unit
    if isinstance(x, Tensor):
        return x.data, x.unit
    if isinstance(x, (measured.Value, Scalar)):
        y = [x.data] if isinstance(x, Scalar) else x
        return tensor_args(numpy.asarray(y), x.unit)
    a = numpy.asarray(x)
    if a.ndim == 0:
        raise TypeError(
            "Tensor data must have at least one dimension"
        ) from None
    return a, unit


ARRAY_OPERATORS = {
    **OBJECT_OPERATORS,
    '__eq__': array_dunder_eq,
    '__abs__': array_dunder_abs,
    '__pos__': array_dunder_pos,
    '__neg__': array_dunder_neg,
    '__int__': array_dunder_int,
    '__float__': array_dunder_float,
    '__complex__': array_dunder_complex,
    '__getitem__': array_dunder_getitem,
    'transpose': array_transpose,
    '__measure__': object_dunder_measure,
    '_get_numpy_array': object_get_numpy_array,
}


def array_factory(x, unit=None, axes=None):
    """Create a new array."""
    Type = type('Array', (Array,), ARRAY_OPERATORS)
    d, u, a = array_args(x, unit, axes)
    return Type(d, unit=metric.unit(u or '1'), axes=a)


def array_args(x, unit, axes, /):
    """Parse arguments to initialize `~Array`."""
    if isinstance(x, Object):
        if unit is not None:
            raise ValueError(
                "Cannot change unit via object creation"
                ". Please use the 'withunit' function."
            ) from None
        if axes is not None:
            raise ValueError(
                "Cannot change axes via object creation."
            ) from None
    if isinstance(x, Array):
        return x.data, x.unit, x.axes
    dimensions = tuple(axes or ())
    d = data.array(array_data(x), *dimensions)
    axes = array_axes(d.shape, axes)
    if isinstance(x, numeric.Measurement):
        return d, x.unit, axes
    return d, unit, axes


def array_shape(x) -> typing.Tuple[int]:
    """Compute the shape of `x`."""
    if isinstance(x, (Tensor, data.Array, numpy.ndarray)):
        return x.shape
    if isinstance(x, Vector):
        return (x.size,)
    if isinstance(x, Scalar):
        return (1,)
    if shape := getattr(x, 'shape', None):
        return shape
    raise TypeError(
        f"Cannot determine shape of {x}"
    ) from None


def array_data(x):
    """Extract data for a physical array.

    Notes
    -----
    The behavior of this function depends on the type of argument.

    - If the argument is an instance of `~data.Array`, this function will
      extract the internal `_object` attribute rather than the `array` property,
      in order to avoid prematurely loading a large dataset into memory.
    - If the argument is an instance of a subclass of `~measured.Object`, this
      function will extract the `data` property (converting a scalar-like object
      to 1-D array, if necessary).
    - If the argument is an instance of `numpy.ndarray`, this function will
      immediately return it.
    - If the argument is a logically N-D sequence, this function will convert to
      an N-D `numpy.ndarray`.
    - If the argument is a complex number, this function will convert it to a
      1-D real-valued `numpy.ndarray`.
    - This function will immediately return all other types of objects. Notably,
      it will NOT attempt to convert an arbitrary object to a `numpy.ndarray` in
      case doing so would prematurely load a large dataset into memory.
    """
    if isinstance(x, data.Array):
        return x._data
    if isinstance(x, (Tensor, Vector, measured.Sequence)):
        return x.data
    if isinstance(x, (Scalar, measured.Value)):
        return numpy.array([x.data])
    if isinstance(x, numpy.ndarray):
        return x
    if isinstance(x, numbers.Complex):
        return numpy.array([float(x)])
    if isinstance(x, typing.Sequence):
        return numpy.array(x)
    return x # <- NetCDF, HDF, etc.


def array_axes(shape, axes) -> Axes:
    """Create axes from arguments."""
    if isinstance(axes, typing.Mapping):
        return axes_factory(axes)
    try:
        axes = axes_factory(shape=shape, dimensions=axes)
    except (TypeError, ValueError) as err:
        raise TypeError(
            "Cannot create array axes"
            f" from shape={shape} and axes={axes}"
        ) from err
    return axes

