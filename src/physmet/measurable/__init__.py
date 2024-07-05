import contextlib
import numbers
import typing
import typing_extensions

import nonstring
import numpy.typing

from .. import measured
from .. import metric
from .. import numeric
from .. import quantity
from . import _measurement


class ParsingTypeError(TypeError):
    """An argument to `~parse` has an invalid type."""


class ParsingValueError(ValueError):
    """An argument to `~parse` has an invalid value."""


class MeasurementTypeError(TypeError):
    """The `__measure__` method did not return a `~numeric.Measurement`."""


class MeasuringTypeError(TypeError):
    """An argument to `~measure` has an invalid type."""


T = typing.TypeVar('T')


@typing.overload
def measurement(
    data: typing.Union[
        numeric.RealValueType,
        typing.Sequence[numeric.RealValueType],
    ],
    /,
    unit: metric.UnitLike=typing.Literal['1']
) -> numeric.Measurement[numeric.RealValueType]:
    """Create a physical measurement.
    
    Parameters
    ----------
    data : real number or sequence of real numbers
        The numerical value(s) from which to create a new measurement.

    unit : `~metric.Unit`
        The metric unit associated with `data`.

    Returns
    -------
    `~Measurement`

    Raises
    ------
    `TypeError`
        The caller passed a single non-numeric object or a sequence that
        includes non-numeric members.
    """

@typing.overload
def measurement(
    data: numpy.typing.NDArray[numpy.floating],
    /,
    unit: metric.UnitLike=typing.Literal['1']
) -> numeric.Measurement[numeric.RealValueType]:
    """Create a physical measurement.
    
    Parameters
    ----------
    data : `numpy.ndarray`
        The numerical value(s) from which to create a new measurement.

    unit : `~metric.Unit`
        The metric unit associated with `data`.

    Returns
    -------
    `~Measurement`

    Raises
    ------
    `TypeError`
        The caller passed a single non-numeric object or a sequence that
        includes non-numeric members.
    """

@typing.overload
def measurement(
    data: numeric.Quantity[numeric.RealValueType],
    /,
    unit: metric.Unit=typing.Literal['1'],
) -> numeric.Measurement[numeric.RealValueType]:
    """Create a physical measurement.
    
    Parameters
    ----------
    data : numeric quantity
        An object with numeric data.

    unit : `~metric.Unit`
        The metric unit associated with `data`.

    Returns
    -------
    `~Measurement`
    """

@typing.overload
def measurement(
    data: numeric.Measurement[numeric.RealValueType],
    /,
) -> numeric.Measurement[numeric.RealValueType]:
    """Create a physical measurement.
    
    Parameters
    ----------
    data : measurement
        An existing measurement.

    Returns
    -------
    `~Measurement`
    """

def measurement(*args, **kwargs):
    return _measurement.factory(*args, **kwargs)


@typing.overload
def measure(*args) -> numeric.Measurement:
    """Measure the given arguments, if possible.

    Parameters
    ----------
    *args
        One or more objects from which to create a measurement. See
        `~ismeasurable` for a description of supported object types.

    Returns
    -------
    `~numeric.Measurement`

    Raises
    ------
    `~MeasuringTypeError`
        The input was empty or the parsing algorithm was not able to determine
        appropriate numeric values and a unique unit from the input arguments.

    See Also
    --------
    `~measurable.parse`
        The function that parses measurable input into numeric values and an
        associated unit, if possible.
    `~measurable.ismeasurable`
        The function that detects measurable input.
    """

@typing.overload
def measure(x: numeric.Measurement) -> numeric.Measurement:
    """Measure an existing measured object.
    
    Parameters
    ----------
    x : numeric measurement
        The measured object to measure. See Notes for additional information
        about how this function handles specific types of measured objects.

    Returns
    -------
    `~numeric.Measurement`

    Raises
    ------
    `~MeasurementTypeError`
        The type of object returned by `x.__measure__` was not an instance of
        `~numeric.Measurement`.

    Notes
    -----
    - If `x` is already an instance of `~numeric.Measurement`, this function will
      immediately return it.
    - If `x` implements the special method `__measure__`, this function will
      defer to that implementation.
    - If `x` is any other instance of `~measured.Object` or one of its
      subclasses, this function will create a new measurement from the
      real-valued data in `x.data` and the metric unit in `x.unit`.
    """


@typing.overload
def measure(x: numeric.MeasurableType, **kwargs) -> numeric.Measurement:
    """Measure an explicitly measurable object.
    
    Parameters
    ----------
    x : measurable
        An object that complies with the `~base.Measurable` protocol by
        implementing the special method `__measure__`.

    **kwargs
        Keyword arguments to pass to `x.__measure__`.

    Returns
    -------
    `~numeric.Measurement`

    Raises
    ------
    `~MeasurementTypeError`
        The type of object returned by `x.__measure__` was not an instance of
        `~numeric.Measurement`.
    """

def measure(*args, **kwargs):
    if not args:
        raise MeasuringTypeError("There is nothing to measure") from None
    this = args[0]
    if isinstance(this, numeric.Measurement):
        return this
    if isinstance(this, numeric.Measurable):
        return _measure_explicit(this, **kwargs)
    return _measure_implicit(args)


def _measure_explicit(x: numeric.Measurable, **kwargs):
    """Create a measurement by calling `x.__measure__`."""
    result = x.__measure__(**kwargs)
    if not isinstance(result, numeric.Measurement):
        raise MeasuringTypeError(
            f"{type(x)}.__measure__ returned"
            f" non-Measurement (type {type(result)})"
        )
    return result


def _measure_implicit(x):
    """Create a measurement by parsing `x`."""
    try:
        parsed = parse(x, distribute=False)
    except (ParsingValueError, ParsingTypeError) as err:
        raise MeasuringTypeError(f"Cannot measure {x}") from err
    data = parsed[:-1]
    unit = parsed[-1]
    return measured.sequence(data, unit=unit)


Ns = typing_extensions.TypeVarTuple('Ns')

@typing.overload
def parse(
    x: numeric.RealValueType,
    /,
) -> typing.Tuple[numeric.RealValueType, typing.Literal['1']]: ...

@typing.overload
def parse(
    x: typing.Tuple[numeric.RealValueType, str],
    /,
) -> typing.Tuple[numeric.RealValueType, str]: ...

@typing.overload
def parse(
    x: typing.Tuple[numeric.RealValueType, ...],
    /,
) -> typing.Tuple[typing_extensions.Unpack[Ns], typing.Literal['1']]: ...

@typing.overload
def parse(
    x: typing.Tuple[typing_extensions.Unpack[Ns], str],
    /,
) -> typing.Tuple[typing_extensions.Unpack[Ns], str]: ...

@typing.overload
def parse(
    x: typing.Tuple[typing_extensions.Unpack[Ns], str],
    /,
    distribute: typing.Literal[False],
) -> typing.Tuple[typing_extensions.Unpack[Ns], str]: ...

@typing.overload
def parse(
    x: typing.Tuple[typing_extensions.Unpack[Ns], str],
    /,
    distribute: typing.Literal[True],
) -> typing.Tuple[typing.Tuple[numbers.Real, str], ...]: ...

def parse(x, /, distribute: bool=False):
    """Parse an implicitly measurable object.
    
    This function will extract numeric values and an associated unit from `x`,
    if possible.

    Parameters
    ----------
    x
        The object or collection of objects to measure.

    distribute : bool, default=false
        If true, distribute the parsed unit among parsed numerical values.

    Returns
    -------
    `tuple`
        - If `distribute == True`, the returned tuple will contain one or more
          `tuple`s, each containing a single numeric value and a string that
          represents the associated unit common to all values. The length of the
          result will equal the number of parsed numeric values.
        - Otherwise, the returned `tuple` will contain parsed numeric values
          followed by a single unit string. The length of the result will be one
          more than the number of original numeric values.

    Raises
    ------
    `~ParsingTypeError`
        - The input is any non-iterable object but is not a single number.

    `~ParsingValueError`
        - The input is an empty iterable object.
        - The input contains multiple units.
        - The input contains multiple objects with individual units, but the
          units differ.

    Notes
    -----
    This function will parse the following types of arguments

    - a single number
    - a `list` or `tuple` of only numbers
    - a `list` or `tuple` of numbers followed by a single unit
    - a 2-element `list` or `tuple` in which the first element is a `list` or
      `tuple` of only numbers, and the second element is a single unit
    - an iterable collection of any of the above cases, as long as the units are
      consistent

    where the notion of numbers includes strings that can be converted to
    numeric values.
    """

    # Strip redundant lists and tuples.
    unwrapped = nonstring.unwrap(x)

    # Raise a type-based exception if input is `None`.
    if unwrapped is None:
        raise ParsingTypeError(f"Cannot measure {unwrapped!r}") from None

    # Raise a value-based exception for empty input.
    if quantity.isnull(unwrapped):
        raise ParsingValueError(
            f"Cannot measure empty input: {unwrapped!r}"
        ) from None

    # Handle a single numeric value.
    if isinstance(unwrapped, numbers.Real):
        result = (unwrapped, '1')
        return (result,) if distribute else result

    # Handle a single numerical string.
    if isinstance(unwrapped, str):
        try:
            result = (float(unwrapped), '1')
        except (ValueError, TypeError) as err:
            raise ParsingTypeError(
                f"Cannot measure non-numeric string {unwrapped!r}"
            ) from err
        return (result,) if distribute else result

    # Raise a type-based exception if input is not iterable.
    try:
        iter(unwrapped)
    except TypeError as err:
        raise ParsingTypeError(
            f"Cannot measure non-iterable input: {unwrapped!r}"
        ) from err

    # Recursively parse nested parsable objects.
    if all(isinstance(arg, (list, tuple)) for arg in unwrapped):
        return _recursive_parse(unwrapped, distribute)

    # Count the number of distinct unit-like objects.
    types = [type(arg) for arg in unwrapped]
    counted = {t: types.count(t) for t in (str, metric.Unit)}

    # Check for multiple units.
    errmsg = "You may only specify one unit."
    if counted[metric.Unit] > 1:
        # If the input contains more than one instance of the `Unit` class,
        # there is nothing we can do to salvage it.
        raise ParsingValueError(errmsg) from None
    if counted[str] > 1:
        # First, check for a single numeric string.
        if isinstance(unwrapped, str):
            with contextlib.suppress(ValueError):
                return parse([float(unwrapped)])
        # Next, check for all numeric strings.
        with contextlib.suppress(ValueError):
            return parse([float(arg) for arg in unwrapped])
        # Finally, check for numeric strings with a final unit.
        try:
            values = [float(arg) for arg in unwrapped[:-1]]
        except ValueError as err:
            raise ParsingValueError(errmsg) from err
        return parse([*values, unwrapped[-1]])

    # Handle flat numerical iterables, like (1.1,) or (1.1, 2.3).
    if all(isinstance(arg, numbers.Real) for arg in unwrapped):
        return _wrap_measurable(unwrapped, '1', distribute)

    # Ensure an explicit unit-like object. Note that, at this point, `unwrapped`
    # must have one of the following forms (where any tuple may be a list):
    # - (v0, v1, ..., unit)
    # - ((v0, v1, ...), unit)
    last = unwrapped[-1]
    unitless = all(
        not isinstance(arg, (str, metric.Unit)) for arg in unwrapped
    ) or last in ['1', metric.unit('1')]
    unit = '1' if unitless else str(last)

    if isinstance(last, (str, metric.Unit)):
        unit = str(unwrapped[-1])

        # Handle flat iterables with a unit, like (1.1, 'm') or (1.1, 2.3, 'm').
        if all(isinstance(arg, numbers.Real) for arg in unwrapped[:-1]):
            return _wrap_measurable(unwrapped[:-1], unit, distribute)

        # Handle iterable values with a unit, like [(1.1, 2.3), 'm'].
        if isinstance(unwrapped[0], (list, tuple, range)):
            return _wrap_measurable(unwrapped[0], unit, distribute)

    raise ParsingTypeError(x)


def _wrap_measurable(values, unit, distribute: bool):
    """Wrap a parsed measurable and return to caller."""
    if distribute:
        return tuple(nonstring.distribute(values, unit))
    return (*values, unit)


def _recursive_parse(unwrapped, distribute: bool):
    """Parse the measurable by calling back to `~parse`."""
    if distribute:
        parsed = [
            item
            for arg in unwrapped
            for item in parse(arg, distribute=True)
        ]
        return tuple(parsed)
    parsed = [
        parse(arg, distribute=False) for arg in unwrapped
    ]
    units = [item[-1] for item in parsed]
    if any(unit != units[0] for unit in units):
        errmsg = "Cannot combine measurements with different units."
        raise ParsingValueError(errmsg)
    values = [
        i for item in parsed for i in item[:-1]
    ]
    unit = units[0]
    return (*values, unit)
