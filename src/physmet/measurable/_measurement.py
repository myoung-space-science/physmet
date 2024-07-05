import numbers

import nonstring
import numpy

from .. import data
from .. import numeric
from .. import metric
from .. import quantity


def __eq__(a: numeric.Measurement, b):
    """Called for a == b."""
    if isinstance(b, numeric.Measurement):
        return data.isequal(a, b) and a.unit == b.unit
    return False


def withunit(
    a: numeric.Measurement,
    unit: str | metric.Unit,
) -> numeric.Measurement:
    """Convert `a` to a new unit."""
    c = numeric.conversion(a, unit)
    return factory(a.data * float(c), unit=c.new)


OPERATORS = {
    '__eq__': __eq__,
    'withunit': withunit,
}


def factory(x, /, unit=None):
    """Factory function for `~Measurement`."""
    raise NotImplementedError
    Type = type('Measurement', (numeric.Measurement,), OPERATORS)
    d, u = _parse_input(x, unit)
    this = Type()
    this._data = d
    this._unit = metric.unit(u or '1')
    return this


def _parse_input(x, unit, /):
    if isinstance(x, numeric.Measurement):
        return _from_numeric_quantity(x), x.unit
    if isinstance(x, numeric.Quantity):
        return _from_numeric_quantity(x), unit
    if isinstance(x, numbers.Real):
        return numpy.array([x]), unit
    if isinstance(x, numpy.ndarray):
        if x.ndim > 0:
            return x.flatten(), unit
        return x.reshape(1), unit
    if nonstring.isseparable(x):
        if data.hasdtype(x, (numpy.integer, numpy.floating)):
            return _parse_input(numpy.array(x), unit)
        raise TypeError(
            f"Cannot create a measurement from {x!r}"
            " with non-numeric values."
        ) from None
    if not quantity.isreal(x):
        raise TypeError(
            f"Cannot create measurement from object of type {type(x)}"
        ) from None
    return numpy.array([x]), unit


def _from_numeric_quantity(x: numeric.Quantity, /):
    a = numpy.array(x.data)
    if a.ndim == 0:
        return a.reshape(1)
    return a.flatten()

