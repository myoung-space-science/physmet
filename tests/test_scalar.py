import math
import numbers
import operator as standard
import typing

import numpy
import numpy.typing
import pytest

import physmet
from physmet import base
from physmet import data
from physmet import measured
from physmet import measurable
from physmet import metric
from physmet import numeric
import support


def test_factory():
    """Test various ways to create a scalar."""
    value = 2
    unit = 'km'
    scalar = physmet.scalar(value)
    copied = physmet.scalar(scalar)
    assert copied is not scalar
    assert copied == scalar
    assert scalar.data == value
    assert scalar.unit == '1'
    scalar = physmet.scalar(value, unit=unit)
    assert scalar.data == value
    assert scalar.unit == unit
    for arg in ([1], numpy.array([1])):
        scalar = physmet.scalar(arg, unit=unit)
        assert scalar.data == numpy.array(arg)
        assert scalar.unit == unit
    for arg in ([1, 2], numpy.array([1, 2])):
        with pytest.raises(TypeError):
            physmet.scalar(arg)
    tensor = physmet.tensor([value], unit=unit)
    for arg in (scalar, tensor):
        with pytest.raises(ValueError):
            physmet.scalar(arg, unit=unit)


def test_measure_scalar():
    """Test the use of `__measure__` on a scalar."""
    x = physmet.scalar(1.5, unit='J / s')
    assert data.ismeasurable(x)
    m = measurable.measure(x)
    assert isinstance(m, numeric.Measurement)
    assert m.unit == x.unit
    assert float(m) == float(x)
    assert int(m) == int(x)


def test_additive():
    """Test additive operations on scalars."""
    va = 2.0
    vb = 1.1
    meter = 'm'
    original = physmet.scalar(va, unit=meter)
    sameunit = physmet.scalar(vb, unit=meter)
    unitless = physmet.scalar(vb)
    valid = [
        # measurable tuple with same unit
        (original, (vb, original.unit), original.unit),
        # scalar with same unit
        (original, sameunit, original.unit),
        # plain number with unitless scalar
        (unitless, vb, '1'),
    ]
    operators = (standard.add, standard.sub)
    for f in operators:
        for a, b, u in valid:
            check_additive(f, a, b, u)
            check_additive(f, b, a, u)
    joule = 'J'
    diffunit = physmet.scalar(va, unit=joule)
    invalid = [
        # can't add or subtract objects with a different unit
        (original, (vb, joule), ValueError),
        (original, diffunit, ValueError),
        # can't add or subtract plain numbers from a unitful scalar
        (original, va, TypeError),
    ]
    for f in operators:
        for a, b, err in invalid:
            with pytest.raises(err):
                f(a, b)
            with pytest.raises(err):
                f(b, a)


def check_additive(f, a, b, unit: metric.UnitLike) -> None:
    """Helper for `test_additive`."""
    new = f(a, b)
    assert isinstance(new, physmet.Scalar)
    x = support.getdata(a)
    y = support.getdata(b)
    assert new.data == f(x, y)
    assert new.unit == unit


def test_multiplicative():
    """Test multiplicative operations on scalars."""
    va = 2.0
    vb = 1.1
    meter = 'm'
    joule = 'J'
    original = physmet.scalar(va, unit=meter)
    sameunit = physmet.scalar(vb, unit=meter)
    diffunit = physmet.scalar(va, unit=joule)
    valid = [
        # same unit
        (original, sameunit),
        (original, (vb, original.unit)),
        # different unit
        (original, diffunit),
        (original, (vb, diffunit.unit)),
        # scale by constant factor
        (original, va),
    ]
    operators = (
        (standard.mul, standard.mul),
        (standard.truediv, standard.truediv),
        (standard.floordiv, standard.truediv),
        (standard.mod, standard.truediv),
    )
    for f, g in operators:
        for a, b in valid:
            check_multiplicative(f, g, a, b)
            check_multiplicative(f, g, b, a)


def check_multiplicative(
    f: typing.Callable,
    g: typing.Callable,
    a: typing.Union[physmet.Scalar, numbers.Real, tuple],
    b: typing.Union[physmet.Scalar, numbers.Real, tuple],
) -> None:
    """Helper for `test_multiplicative`."""
    new = f(a, b)
    assert isinstance(new, physmet.Scalar)
    x = support.getdata(a)
    y = support.getdata(b)
    assert new.data == f(x, y)
    unit = support.compute_unit(g, a, b)
    assert new.unit == unit


def test_pow():
    """Test exponentiation on a scalar."""
    va = 2.0
    vb = 1.1
    meter = 'm'
    joule = 'J'
    original = physmet.scalar(va, unit=meter)
    sameunit = physmet.scalar(vb, unit=meter)
    diffunit = physmet.scalar(va, unit=joule)
    p = 3
    unitless = physmet.scalar(p)
    valid = [
        # raise a scalar by a number
        (original, p, physmet.Scalar),
        # raise a scalar by a unitless scalar
        (original, unitless, physmet.Scalar),
        # raise a number by a unitless scalar
        (p, unitless, type(p)),
    ]
    for a, b, t in valid:
        check_pow(standard.pow, a, b, t)
    invalid = [
        # a non-numeric exponent is meaningless
        (original, '1', TypeError),
        # can't raise anything by a unitful scalar
        (p, original, ValueError),
        (unitless, original, ValueError),
        (original, sameunit, ValueError),
        (sameunit, original, ValueError),
        (original, diffunit, ValueError),
        (diffunit, original, ValueError),
    ]
    for a, b, error in invalid:
        with pytest.raises(error):
            a ** b


def check_pow(
    f,
    a: typing.Union[physmet.Scalar, numbers.Real],
    b: typing.Union[physmet.Scalar, numbers.Real],
    t: typing.Type[numeric.MeasurementType]=physmet.Scalar,
) -> None:
    """Helper for `test_pow`."""
    new = f(a, b)
    assert isinstance(new, t)
    if t == physmet.Scalar:
        x = a.data if isinstance(a, physmet.Scalar) else a
        y = b.data if isinstance(b, physmet.Scalar) else b
        assert new.data == f(x, y)
        p = b.data if isinstance(b, physmet.Scalar) else b
        assert new.unit == f(a.unit, p)


def test_int():
    """Test the conversion to `int` for a scalar."""
    assert int(physmet.scalar(1.1, unit='m')) == 1


def test_float():
    """Test the conversion to `float` for a scalar."""
    assert float(physmet.scalar(1, unit='m')) == 1.0


def test_round():
    """Test the built-in `round` method on a scalar."""
    values = [
        -1.6,
        -1.1,
        +1.1,
        +1.6,
    ]
    unit = 'm'
    for value in values:
        a = physmet.scalar(value, unit=unit)
        r = round(a)
        assert isinstance(r, physmet.Scalar)
        assert r.data == round(value)
        assert r.unit == metric.unit(unit)


def test_floor():
    """Test the `math.floor` method on a scalar."""
    values = [
        -1.6,
        -1.1,
        +1.1,
        +1.6,
    ]
    unit = 'm'
    for value in values:
        a = physmet.scalar(value, unit=unit)
        r = math.floor(a)
        assert isinstance(r, physmet.Scalar)
        assert r.data == math.floor(value)
        assert r.unit == metric.unit(unit)


def test_ceil():
    """Test the `math.ceil` method on a scalar."""
    values = [
        -1.6,
        -1.1,
        +1.1,
        +1.6,
    ]
    unit = 'm'
    for value in values:
        a = physmet.scalar(value, unit=unit)
        r = math.ceil(a)
        assert isinstance(r, physmet.Scalar)
        assert r.data == math.ceil(value)
        assert r.unit == metric.unit(unit)


def test_trunc():
    """Test the `math.trunc` method on a scalar."""
    values = [
        -1.6,
        -1.1,
        +1.1,
        +1.6,
    ]
    unit = 'm'
    for value in values:
        a = physmet.scalar(value, unit=unit)
        r = math.trunc(a)
        assert isinstance(r, physmet.Scalar)
        assert r.data == math.trunc(value)
        assert r.unit == metric.unit(unit)


def test_trig():
    """Test `numpy` trigonometric ufuncs on a scalar."""
    value = 1.0
    for f in (numpy.sin, numpy.cos, numpy.tan):
        for unit in {'rad', 'deg'}:
            old = physmet.scalar(value, unit=unit)
            new = f(old)
            assert isinstance(new, physmet.Scalar)
            assert data.isequal(new, f(old.data))
            assert new.unit == '1'
        bad = physmet.scalar(value, unit='m')
        with pytest.raises(ValueError):
            f(bad)


def test_sqrt():
    """Test `numpy.sqrt` on a scalar."""
    scalar = physmet.scalar(2.0, unit='m')
    value = numpy.sqrt(scalar.data)
    unit = scalar.unit ** 0.5
    assert numpy.sqrt(scalar) == physmet.scalar(value, unit=unit)


def test_squeeze():
    """Test `numpy.squeeze` on a scalar."""
    value = 4
    old = physmet.scalar(value, unit='m')
    new = numpy.squeeze(old)
    assert new is old


def test_mean():
    """Test `numpy.mean` of a scalar."""
    value = 2
    old = physmet.scalar(value, unit='m')
    new = numpy.mean(old)
    assert new is not old
    assert isinstance(new, physmet.Scalar)
    assert numpy.array_equal(new.data, numpy.mean(value))
    assert new.unit == old.unit


def test_sum():
    """Test `numpy.sum` of a scalar."""
    value = 2
    old = physmet.scalar(value, unit='m')
    new = numpy.sum(old)
    assert new is not old
    assert isinstance(new, physmet.Scalar)
    assert numpy.array_equal(new.data, numpy.sum(value))
    assert new.unit == old.unit


def test_cumsum():
    """Test `numpy.cumsum` of a scalar."""
    value = 2
    old = physmet.scalar(value, unit='m')
    new = numpy.cumsum(old)
    assert isinstance(new, physmet.Vector)
    assert new.size == 1
    assert numpy.array_equal(new.data, numpy.cumsum(value))
    assert new.unit == old.unit


def test_transpose():
    """Test `numpy.transpose` on a scalar."""
    value = 2
    old = physmet.scalar(value, unit='cm')
    new = numpy.transpose(old)
    assert new is not old
    assert new == physmet.scalar(value, unit='cm')


def test_gradient():
    """Test `numpy.gradient` on a scalar."""
    value = 2
    scalar = physmet.scalar(value, unit='cm')
    assert numpy.gradient(scalar) == []


def test_unit():
    """Test the ability to update a scalar's unit."""
    old = physmet.scalar(2.0, 'm')
    new = old.withunit('km')
    assert isinstance(new, physmet.Scalar)
    assert new is not old
    assert new.unit == 'km'
    factor = old.unit >> new.unit
    assert float(new) == float(factor*old)
    with pytest.raises(ValueError):
        old.withunit('J')

