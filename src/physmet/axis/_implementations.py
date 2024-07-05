import contextlib
import numbers
import typing

import nonstring
import numpy
import numpy.typing

from .. import base
from .. import data
from .. import indexer
from .. import measured
from .. import metric
from .. import measurable
from .. import numeric
from . import _types


T = typing.TypeVar('T')


def axis_index(a: numeric.Axis, *targets: T):
    """Called to index a generic axis."""
    indices = normalize(a, *targets)
    if not all(isinstance(arg, numbers.Integral) for arg in indices):
        raise TypeError(
            f"Not all target indices have integral type: {targets}"
        ) from None
    if index := _out_of_bounds(a, indices):
        raise ValueError(
            f"Index {index} is out of bounds for axis"
            f" with size {len(a)}"
        ) from None
    if len(indices) == 1:
        with contextlib.suppress(TypeError):
            return indexer.value(int(indices[0]))
    return indexer.sequence(indices)


def axis_contains(a: numeric.AxisType, x, /) -> bool:
    """Called for x in a.

    Notes
    -----
    - This function will return `True` if `a` can index `x` and the
      corresponding index is in `a.indices`. It will return `False` if the
      corresponding index is not in `a.indices` or if the attempt to index `x`
      fails.
    - If `x` is also a numeric axis, it must be the same type as `a`.
    """
    if isinstance(x, numeric.Axis) and not _same_types(a, x):
        return False
    try:
        iter(x)
    except TypeError:
        return _axis_contains(a, x)
    return all(_axis_contains(a, i) for i in x)


AXIS_TYPES = (
    _types.Points,
    _types.Symbols,
    _types.Coordinates,
)


def _same_types(a, b) -> bool:
    """Helper for `axis_contains`.

    This is somewhat of a hack because checking type(a) == type(b) doesn't can
    return a false negative.
    """
    for t in AXIS_TYPES:
        with contextlib.suppress(TypeError):
            if all(isinstance(i, t) for i in (a, b)):
                return True
    return False


def _axis_contains(a: numeric.AxisType, x, /) -> bool:
    """Helper for `axis_contains`.

    This function assumes that `x` is a single value.
    """
    try:
        return a.index(x) in a.indices
    except Exception:
        return False


def axis_getitem(a: numeric.AxisType, i, /) -> numeric.AxisType:
    """Called for a[i].

    This method will generate a new instance of the appropriate (sub)class from
    its reference array. If `i` is not `:` or `...`, this method will populate
    the internal `_indices` attribute with the indices corresponding to `i`.
    Otherwise, it will immediately return the new axis object, which will
    automatically populate `_indices` as necessary. Both cases are designed to
    preserve index information with regard to the original reference array.
    """
    axis = axis_factory(a.data[i])
    if i in (slice(None), ...):
        return axis
    index = normalize(a, nonstring.unwrap(i))
    axis._indices = indexer.sequence(index)
    return axis


def axis_len(a: numeric.Axis) -> int:
    """Called for len(self)."""
    return len(a.data)


def axis_iter(a: numeric.Axis) -> typing.Iterator[int]:
    """Called for iter(self)."""
    return iter(a.data)


def axis_eq(a: numeric.Axis, b) -> bool:
    """Called for self == other."""
    if isinstance(b, numeric.Axis):
        return numpy.array_equal(a.data, b.data)
    return NotImplemented


def axis_str(a: numeric.Axis) -> str:
    """A simplified representation of this object."""
    return _as_string(a)


def axis_repr(a: numeric.Axis) -> str:
    """An unambiguous representation of this object."""
    return _as_string(
        a,
        prefix=f'{a.__class__.__qualname__}(',
        suffix=')',
    )


def _format_data(a: numeric.Axis, **kwargs):
    """Create an appropriate string to represent the data."""
    array = numpy.array(a.data)
    try:
        signed = '+' if any(i < 0 for i in array.flat) else '-'
    except TypeError:
        # If we get here, it's probably because some values are strings
        signed = '-'
    return numpy.array2string(
        array,
        threshold=4,
        edgeitems=2,
        separator=', ',
        sign=signed,
        precision=3,
        floatmode='maxprec_equal',
        **kwargs
    )


def axis_get_indices(a: numeric.Axis) -> indexer.Sequence:
    """Helper for `Axis.indices`."""
    return indexer.sequence(range(a.length))


class AxisValueError(ValueError):
    """Error in axis argument value."""


class AxisTypeError(TypeError):
    """Error in axis argument type."""


def points_getitem(a: _types.Points, i, /) -> _types.Points:
    """Called for a[i].

    This method will generate a new instance from the reference array. If `i` is
    not `:` or `...`, this method will populate the internal `_indices`
    attribute with the indices corresponding to `i`. Otherwise, it will
    immediately return the new axis object, which will automatically populate
    `_indices` as necessary. Both cases are designed to preserve index
    information with regard to the original reference array.
    """
    axis = points_factory(a.data[i])
    if i in (slice(None), ...):
        return axis
    index = normalize(a, nonstring.unwrap(i))
    axis._indices = indexer.sequence(index)
    return axis


def points_get_indices(a: _types.Points) -> numeric.Sequence:
    """Helper for `Points.indices`."""
    return indexer.sequence(a.data)


def points_index(a: numeric.Axis, *targets: T):
    """Called to index an axis of integral points."""
    if not targets:
        return a.indices
    points = normalize(a, *targets)
    indices = []
    for point in points:
        if point not in a.data:
            raise AxisValueError(
                f"This axis does not contain the point {point}"
            ) from None
        index = numpy.where(a.data == point)
        indices.append(index[0][0])
    if len(indices) == 1:
        return indexer.value(indices[0])
    return indexer.sequence(indices)


def symbols_getitem(a: numeric.Axis, i, /):
    """Called for a[i]."""
    try:
        iter(i)
    except TypeError:
        axis = symbols_factory(a.data[i])
        if i in (slice(None), ...):
            return axis
        idx = normalize(a, nonstring.unwrap(i))
        axis._indices = indexer.sequence(idx)
        return axis
    else:
        return symbols_factory([a.data[x] for x in i])


def symbols_index(a: _types.Symbols, *targets: T):
    if not targets:
        return a.indices
    if not all(isinstance(target, str) for target in targets):
        raise AxisTypeError(
            "All index targets must be strings"
        ) from None
    indices = [
        a.data.index(str(target))
        for target in targets
    ]
    if len(indices) == 1:
        return indexer.value(indices[0])
    return indexer.sequence(indices)


def coordinates_index(
    a: _types.Coordinates,
    *targets: typing.Union[numbers.Real, str],
    closest: typing.Optional[str]=None,
) -> typing.Union[indexer.Value, indexer.Sequence]:
    if not targets:
        return a.indices
    this = _measure(targets)
    if not this:
        raise AxisTypeError(f"Cannot measure {targets}") from None
    # At this point, we can assume that `this` is a measurement that we can
    # use to compute the indices of the target values. First, we need to
    # convert the measured values into the unit of this axis. That
    # conversion is straightfoward unless `targets` comprises only numbers,
    # in which case `this` will be unitless. In that case, we assume that
    # the caller meant to imply the use of the current unit (which may
    # actually be '1') and we therefore need to assign the current unit to
    # the measured values rather than convert them to the current unit.
    # - Cases:
    #   - targets are all numbers
    #     - the implicit unit is '1': We need to assign the current unit.
    #   - targets include a unit
    #     - the explicit unit is equivalent to the current unit: No change
    #       is necessary.
    #     - the explicit unit is not equivalent to the current unit: We need
    #       to convert them to the current unit.
    #     - the explicit unit is inconsistent with the current unit: This
    #       will raise an exception during conversion.
    #
    # Rather than separately handling each case, this method explicitly
    # assigns the measured unit to the measured value unless the measured
    # unit is '1', in which case it explicitly assigns the current unit. It
    # then explicitly converts the result to the current unit.
    # - Thus
    #   - if the target unit is implicitly '1' and the current unit is also
    #     '1', we will end up re-assigning the correct unit to the measured
    #     values at the cost of creating an intermediate sequence object,
    #     but the subsequent conversion to the current unit will be a no-op
    #   - if the target unit is implicitly '1' and the current unit is not
    #     '1', we will correctly assign the current unit to the measured
    #     values and the subsequence conversion to the current unit will be
    #     a no-op
    #   - if the targets have an explicit unit, we will end up re-assigning
    #     that unit to the measured values at the cost of creating an
    #     intermediate sequence object before converting the measured values
    #     to the current unit
    #
    # - Determine the most appropriate unit.
    unit = this.unit if this.unit != '1' else a.unit
    # - Create a temporary sequence with the intermediate unit. The
    #   following line casts `this` to a `numpy.ndarray` in order to force
    #   the unit update.
    tmp = measured.sequence(numpy.array(this), unit=unit)
    # - Convert the temporary array to this coordinate's unit and extract a
    #   numpy array. If the unit of `targets` is inconsistent with our unit,
    #   this is when we'll find out.
    array = numpy.array(tmp.withunit(a.unit))
    # - Convert target values to indices in the reference array. This will
    #   first make sure each target is close enough to a value in the
    #   reference array to be considered "in" the array. That check is
    #   important because `data.nearest` will naively find the
    #   numerically closest value, which may not be useful. It also lets
    #   calling code determine how to handle values not contained in the
    #   reference array (e.g., by interpolation or extrapolation).
    indices = [_compute_index(a, target, closest) for target in array]
    if len(indices) == 1:
        return indexer.value(indices[0])
    return indexer.sequence(indices)


def _compute_index(a: numeric.Axis, target, closest):
    """Helper for `index`."""
    if data.isclose(a.data, target):
        return data.nearest(a.data, target).index
    if closest is None:
        raise AxisValueError(
            f"{a.data} does not contain {target}"
        ) from None
    if closest == 'lower':
        return data.nearest(a.data, target, 'upper').index
    if closest == 'upper':
        return data.nearest(a.data, target, 'lower').index
    raise ValueError(f"Unknown constraint {closest!r}") from None


def _measure(targets: typing.Sequence):
    """Create a measured sequence, if possible."""
    if len(targets) == 1:
        target = targets[0]
        if isinstance(target, numeric.Measurement):
            return measurable.measure(target)
        if data.ismeasurable(target):
            return measurable.measure(target)
    with contextlib.suppress(measurable.ParsingTypeError):
        return measurable.measure(targets)


def coordinates_getitem(a: _types.Coordinates, i, /) -> _types.Coordinates:
    """Called for a[i].

    This method will generate a new instance from the reference array. If `i` is
    not `:` or `...`, this method will populate the internal `_indices`
    attribute with the indices corresponding to `i`. Otherwise, it will
    immediately return the new axis object, which will automatically populate
    `_indices` as necessary. Both cases are designed to preserve index
    information with regard to the original reference array.
    """
    axis = coordinates_factory(a.data[i])
    if i in (slice(None), ...):
        return axis
    index = normalize(a, nonstring.unwrap(i))
    axis._indices = indexer.sequence(index)
    return axis


def coordinates_eq(a: _types.Coordinates, b):
    if isinstance(b, _types.Coordinates):
        return axis_eq(a, b) and a.unit == b.unit
    return axis_eq(a, b)


def coordinates_measure(a: numeric.Axis) -> numeric.Measurement:
    """Called for `~data.measure(self)`."""
    return measurable.measurement(a.data)


def coordinates_withunit(
    a: _types.Coordinates,
    new: metric.UnitLike,
    /,
) -> _types.Coordinates:
    """Convert this cooridinate to a new unit."""
    c = measured.conversion(a, new)
    return coordinates_factory(a.data * float(c), unit=c.new)


def normalize(a, *targets: T):
    if not targets:
        return range(len(a))
    if len(targets) == 1:
        target = targets[0]
        if isinstance(target, slice):
            return slice2range(target, stop=len(a))
        if isinstance(target, (list, tuple, numpy.ndarray)):
            return target
        with contextlib.suppress(TypeError):
            iter(target)
            return list(target)
        # If `targets` is a tuple with a single member at this point,
        # execution will fall through and the function will return a list
        # containing only that element.
    return list(targets)


def _as_string(a: numeric.Axis, prefix: str='', suffix: str='') -> str:
    """Create a string representation of this object."""
    data = _format_data(a, prefix=prefix, suffix=suffix)
    if isinstance(a, _types.Coordinates):
        return f"{prefix}{data}, unit={str(a.unit)!r}{suffix}"
    return f"{prefix}{data}{suffix}"


def _out_of_bounds(
    a: numeric.Axis,
    indices: typing.Iterable[typing.SupportsIndex],
) -> typing.Optional[typing.SupportsIndex]:
    """Return the first index outside the bounds of this axis."""
    for index in indices:
        if index < -len(a) or index > len(a)-1:
            return index


def slice2range(s: slice, /, stop: int=None) -> range:
    """Attempt to convert a `slice` to the equivalent `range`.
    
    Parameters
    ----------
    s : slice
        The `slice` object to convert.

    stop : int, optional
        If given, this function will use the value of `stop` as the (required)
        stop value of the resultant `range` object if the stop value of the
        given `slice` is `None`.

    Returns
    -------
    `range`
        A `range` object built from appropriate start, stop, and step values. If
        the given `slice` doesn't define a start value, this function will use
        0. If the given `slice` doesn't define a step value, this function will
        use 1. See Parameters and Raises for descriptions of behavior for
        different stop values.

    Raises
    ------
    `TypeError`
        Both the stop value of the given `slice` and the value of the `stop`
        keyword parameter are `None`. It is not possible to create a `range`
        object with non-integral stop value.
    """
    if stop is None and s.stop is None:
        raise TypeError(f"Cannot convert {s} to a range.")
    start = s.start or 0 # same effect if s.start == 0
    stop = stop if s.stop is None else s.stop
    step = s.step or 1 # handles invalid case of s.step == 0
    return range(start, stop, step)


AXIS_OPERATORS = {
    '__array__': numeric.operators.__array__,
    '__str__': axis_str,
    '__repr__': axis_repr,
    '__eq__': axis_eq,
    '__contains__': axis_contains,
    '__len__': axis_len,
    '__iter__': axis_iter,
    '__getitem__': axis_getitem,
    'index': axis_index,
}


def axis_factory(x, /) -> numeric.Axis:
    """Create a new general axis."""
    Type = type('Axis', (numeric.Axis,), AXIS_OPERATORS)
    return Type(x)


POINTS_OPERATORS = {
    **AXIS_OPERATORS,
    'index': points_index,
    '__getitem__': points_getitem,
}


def points_factory(x, /) -> _types.Points:
    """Create a new axis of integer points."""
    Type = type('Points', (_types.Points,), POINTS_OPERATORS)
    this = Type(points_args(x))
    this._indices = points_get_indices(this)
    return this


def points_args(x, /) -> numpy.typing.NDArray[numpy.uint]:
    """Parse arguments to initialize `~_types.Points`."""
    if isinstance(x, _types.Points):
        return x.data
    try:
        # XXX: Creating this intermediate array circumvents bugs that arise when
        # `x.__array__` does not accept arguments.
        tmp = numpy.array(x)
    except (ValueError, TypeError) as err:
        raise TypeError(
            f"Cannot initialize {_types.Points} from {x}"
        ) from err
    data = numpy.array(tmp, ndmin=1, dtype=int)
    if any(i < 0 for i in data):
        raise TypeError(
            f"Cannot initialize {_types.Points} from an object"
            " with negative values"
        ) from None
    return data


SYMBOLS_OPERATORS = {
    **AXIS_OPERATORS,
    '__getitem__': symbols_getitem,
    'index': symbols_index,
}


def symbols_factory(x, /) -> _types.Symbols:
    """Create a new axis of symbols."""
    Type = type('Symbols', (_types.Symbols,), SYMBOLS_OPERATORS)
    this = Type(symbols_args(x))
    this._indices = axis_get_indices(this)
    return this


def symbols_args(x, /) -> typing.List[str]:
    """Parse arguments to initialize `~_types.Symbols`."""
    if isinstance(x, _types.Symbols):
        return x.data
    if isinstance(x, str):
        return [x]
    if not isinstance(x, typing.Sequence):
        raise TypeError(
            f"Cannot initialize {_types.Symbols} from non-sequence"
            f" type {type(x)}"
        ) from None
    if not all(isinstance(i, str) for i in x):
        raise TypeError(
            f"Cannot initialize {_types.Symbols} from sequence"
            " with non-string members"
        ) from None
    return x


COORDINATES_OPERATORS = {
    **AXIS_OPERATORS,
    '__array__': numeric.operators.__array__,
    '__eq__': coordinates_eq,
    '__getitem__': coordinates_getitem,
    'index': coordinates_index,
    '__measure__': coordinates_measure,
    'withunit': coordinates_withunit,
}


def coordinates_factory(x, /, unit=None) -> _types.Coordinates:
    """Create a new axis of coordinates."""
    Type = type('Coordinates', (_types.Coordinates,), COORDINATES_OPERATORS)
    this = Type(*coordinates_args(x, unit))
    this._indices = axis_get_indices(this)
    return this


def coordinates_args(
    x,
    unit,
    /,
) -> typing.Tuple[numpy.typing.NDArray[numpy.floating], metric.Unit]:
    """Parse arguments to initialize `~_types.Coordinates`."""
    if isinstance(x, _types.Coordinates):
        if unit is None:
            return x.data, x.unit
        raise ValueError(
            "Cannot change unit via object creation"
            ". Please use the 'withunit' function."
        ) from None
    if isinstance(x, numeric.Measurement):
        s = measured.sequence(x)
        return numpy.array(s.data), s.unit
    if unit is None and data.ismeasurable(x):
        m = measurable.measure(x)
        return numpy.array(m.data), m.unit
    try:
        s = measured.sequence(x, unit=unit)
    except TypeError as err:
        raise TypeError(
            f"Cannot initialize {_types.Coordinates} from {x}"
        ) from err
    return numpy.array(s.data), s.unit




