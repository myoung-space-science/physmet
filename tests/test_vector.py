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


@pytest.fixture
def values():
    return [1.0, 10.0, -5.0]


@pytest.fixture
def array(values: typing.List[float]):
    return numpy.array(values)


def test_factory():
    """Test various ways to create a vector."""
    value = 2
    unit = 'km'
    vector = physmet.vector([value])
    copied = physmet.vector(vector)
    assert copied is not vector
    assert copied == vector
    assert numpy.array_equal(vector.data, [value])
    assert vector.unit == '1'
    vector = physmet.vector([value], unit=unit)
    assert numpy.array_equal(vector.data, [value])
    assert vector.unit == unit
    scalar = physmet.scalar(value, unit=unit)
    vector = physmet.vector(scalar)
    assert numpy.array_equal(vector.data, [value])
    assert vector.unit == unit
    tensor = physmet.tensor([value, value], unit=unit)
    vector = physmet.vector(tensor)
    assert numpy.array_equal(vector.data, [value, value])
    assert vector.unit == unit
    with pytest.raises(TypeError):
        physmet.vector(value, unit=unit)
    with pytest.raises(TypeError):
        physmet.vector([[value]])
    with pytest.raises(TypeError):
        physmet.vector(physmet.tensor([[value], [value]]))
    for arg in (scalar, vector):
        with pytest.raises(ValueError):
            physmet.scalar(arg, unit=unit)


def test_measure_vector():
    """Test the use of `__measure__` on a vector."""
    x = physmet.vector([1.5, 3.0, -1.5, -3.0], unit='J / s')
    assert data.ismeasurable(x)
    m = measurable.measure(x)
    assert isinstance(m, numeric.Measurement)
    assert m.unit == x.unit
    assert numpy.array_equal(x, m)
    with pytest.raises(TypeError):
        float(m)
    with pytest.raises(TypeError):
        int(m)


def test_len():
    """A vector should have the same length as its data attribute."""
    data = numpy.array([1.5, 3.0, -1.5, -3.0])
    vectors = (
        physmet.vector(data),
        physmet.vector(data, unit='m / s'),
    )
    for vector in vectors:
        assert len(vector.data) == len(data)


def test_iter():
    """Test the behavior when iterating over a vector."""
    data = numpy.array([1.5, 3.0, -1.5, -3.0])
    unit = 'm / s'
    vector = physmet.vector(data, unit=unit)
    for this, x in zip(vector, data):
        assert isinstance(this, physmet.Scalar)
        assert this.data == x
        assert this.unit == unit


def test_subscription():
    """Test the behavior of a vector when subscripted."""
    data = numpy.array([1.5, 3.0, -1.5, -3.0])
    original = physmet.vector(data, unit='J / s')
    vector = original[1:3]
    assert isinstance(vector, physmet.Vector)
    assert numpy.array_equal(vector.data, data[1:3])
    scalar = original[0]
    assert isinstance(scalar, physmet.Scalar)
    assert scalar.data == original.data[0]
    assert scalar.unit == original.unit


def test_equality(array: numpy.ndarray):
    """Test equality-based comparative operations on vectors."""
    unit = 'm'
    # the reference vector
    a = physmet.vector(array, unit=unit)
    # directly test "equal" via __eq__
    assert a == physmet.vector(array, unit=unit)
    unequal = [
        # different values; equal shape; equal unit
        physmet.vector([5.0, 10.0, 1.0], unit=unit),
        # equal values; equal shape; different unit
        physmet.vector([5.0, 10.0, 1.0], unit='J'),
    ]
    for other in unequal:
        # directly test "not equal" via __ne__
        assert a != other
        # indirectly test "not equal" via __eq__
        assert not (a == other)


def test_ordering(array: numpy.ndarray):
    """Test ordering-based comparative operations on vectors."""
    vx = array
    vy = numpy.array([*array[:2], -array[2]])
    ux = 'm / s'
    uy = 'J'
    # the reference vector
    xx = physmet.vector(vx, unit=ux)
    # same unit; same shape
    yx = physmet.vector(vy, unit=ux)
    valid = [
        (standard.lt, xx, yx),
        (standard.le, xx, yx),
        (standard.gt, xx, yx),
        (standard.ge, xx, yx),
    ]
    for f, a, b in valid:
        r = f(a.data, b.data)
        assert numpy.array_equal(f(a, b), physmet.vector(r))
    # different unit
    yy = physmet.vector(vy, unit=uy)
    invalid = [
        (standard.lt, xx, yy),
        (standard.le, xx, yy),
        (standard.gt, xx, yy),
        (standard.ge, xx, yy),
    ]
    for f, a, b in invalid:
        with pytest.raises(ValueError):
            f(a, b)


def test_unary(array: numpy.ndarray):
    """Test unary numerical operations on vectors."""
    unit = 'm'
    vector = physmet.vector(array, unit=unit)
    operators = [
        abs,
        standard.pos,
        standard.neg,
    ]
    for f in operators:
        assert f(vector) == physmet.vector(f(array), unit=unit)


def test_additive(array: numpy.ndarray, values: typing.List[float]) -> None:
    """Test additive operations on vectors."""
    meter = 'm'
    joule = 'J'
    original = physmet.vector(array, unit=meter)
    sameunit = physmet.vector(array, unit=meter)
    valid = [
        # same unit
        (original, sameunit, original.unit),
        (original, (values[0], original.unit), original.unit),
        (original, (*values, original.unit), original.unit),
    ]
    operators = (standard.add, standard.sub)
    for f in operators:
        for a, b, u in valid:
            check_additive(f, a, b, u)
            check_additive(f, b, a, u)
    diffunit = physmet.vector(array, unit=joule)
    invalid = [
        # can't add or subtract vectors with different units
        (diffunit, original),
        (diffunit, (1.0, original.unit)),
        (diffunit, (1.0, 2.0, 3.0, original.unit)),
    ]
    for f in operators:
        for a, b in invalid:
            with pytest.raises(ValueError):
                f(a, b)
            with pytest.raises(ValueError):
                f(b, a)


def check_additive(f, a, b, unit: metric.UnitLike) -> None:
    """Helper for `test_additive`."""
    new = f(a, b)
    assert isinstance(new, physmet.Vector)
    x = support.getdata(a)
    y = support.getdata(b)
    assert numpy.array_equal(new.data, f(x, y))
    assert new.unit == unit


def test_multiplicative(array: numpy.ndarray, values: typing.List[float]):
    """Test multiplicative operations on vectors."""
    meter = 'm'
    joule = 'J'
    original = physmet.vector(array, unit=meter)
    sameunit = physmet.vector(array, unit=meter)
    diffunit = physmet.vector(array, unit=joule)
    value = 2.0
    singular = physmet.vector([value], unit=meter)
    operators = (
        (standard.mul, standard.mul),
        (standard.truediv, standard.truediv),
        (standard.floordiv, standard.truediv),
        (standard.mod, standard.truediv),
    )
    valid = [
        (original, sameunit),
        (original, diffunit),
        (original, singular),
        (original, value),
        (singular, singular),
        (original, (values[0], original.unit)),
        (original, (*values, original.unit)),
        (original, (values[0], diffunit.unit)),
        (original, (*values, diffunit.unit)),
    ]
    for f, g in operators:
        for a, b in valid:
            check_multiplicative(f, g, a, b)
            check_multiplicative(f, g, b, a)


def check_multiplicative(
    f: typing.Callable,
    g: typing.Callable,
    a: typing.Union[physmet.Vector, numbers.Real, tuple],
    b: typing.Union[physmet.Vector, numbers.Real, tuple],
) -> None:
    """Helper for `test_multiplicative`."""
    new = f(a, b)
    assert isinstance(new, physmet.Vector)
    x = support.getdata(a)
    y = support.getdata(b)
    assert numpy.array_equal(new.data, f(x, y))
    unit = support.compute_unit(g, a, b)
    assert new.unit == unit


def test_pow(array: numpy.ndarray) -> None:
    """Test exponentiation on a vector."""
    meter = 'm'
    joule = 'J'
    original = physmet.vector(array, unit=meter)
    p = 3
    unitless = physmet.vector(array)
    valid = [
        # can raise a vector by a number
        (original, p, physmet.Vector),
        # can raise a unitless vector by a unitless vector
        (unitless, unitless, physmet.Vector),
        # can raise a number by a unitless vector
        (p, unitless, numpy.ndarray),
        # can raise a numpy array by a unitless vector
        (original.data, unitless, numpy.ndarray)
    ]
    for a, b, t in valid:
        check_pow(standard.pow, a, b, t)
    diffunit = physmet.vector(array, unit=joule)
    invalid = [
        # a non-numeric exponent is meaningless
        (original, '1', TypeError),
        # cannot raise a unitful vector by even a unitless vector
        (original, unitless, ValueError),
        # cannot raise anything by a unitful vector
        (p, original, ValueError),
        (original.data, original, ValueError),
        (original, original, ValueError),
        (original, diffunit, ValueError),
        (diffunit, original, ValueError),
    ]
    for a, b, error in invalid:
        with pytest.raises(error):
            a ** b


def check_pow(
    f: typing.Callable,
    a: typing.Union[physmet.Vector, numbers.Real],
    b: typing.Union[physmet.Vector, numbers.Real],
    t: typing.Type[numeric.MeasurementType]=physmet.Vector,
) -> None:
    """Helper for `test_pow`."""
    new = f(a, b)
    assert isinstance(new, t)
    if isinstance(new, physmet.Vector):
        expected = compute(f, a, b)
        assert numpy.array_equal(new, expected)
        if a.isunitless:
            assert new.unit == '1'
        else:
            p = b.data if isinstance(b, physmet.Object) else b
            assert new.unit == f(a.unit, p)


def compute(
    f: typing.Callable,
    a: typing.Union[physmet.Vector, typing.SupportsFloat],
    b: typing.Union[physmet.Vector, typing.SupportsFloat],
) -> numpy.ndarray:
    """Compute the result of `f(a, b)`."""
    if all(not isinstance(i, physmet.Vector) for i in (a, b)):
        raise TypeError("Expected at least one of a or b to be a vector")
    if isinstance(b, typing.SupportsFloat):
        return f(a.data, float(b))
    if isinstance(a, typing.SupportsFloat):
        return f(float(a), b.data)
    return f(a.data, b.data)


def test_trig(array: numpy.ndarray):
    """Test `numpy` trigonometric ufuncs on a vector."""
    for f in (numpy.sin, numpy.cos, numpy.tan):
        for unit in {'rad', 'deg'}:
            old = physmet.vector(array, unit=unit)
            new = f(old)
            assert isinstance(new, physmet.Vector)
            assert numpy.array_equal(new, f(old.data))
            assert new.unit == '1'
        bad = physmet.vector(array, unit='m')
        with pytest.raises(ValueError):
            f(bad)


def test_sqrt(array: numpy.ndarray):
    """Test `numpy.sqrt` on a vector."""
    data = abs(array)
    old = physmet.vector(data, unit='m')
    new = numpy.sqrt(old)
    assert isinstance(new, physmet.Vector)
    assert numpy.array_equal(new, numpy.sqrt(data))
    assert new.unit == f"{old.unit}^1/2"


def test_squeeze(array: numpy.ndarray):
    """Test `numpy.squeeze` on a vector."""
    old = physmet.vector(array, unit='m')
    new = numpy.squeeze(old)
    assert isinstance(new, physmet.Vector)
    assert numpy.array_equal(new, numpy.squeeze(array))
    assert new.unit == old.unit
    singular = physmet.vector([2.0], unit='m')
    scalar = numpy.squeeze(singular)
    assert isinstance(scalar, physmet.Scalar)
    assert scalar.unit == singular.unit


def test_mean(array: numpy.ndarray):
    """Test `numpy.mean` of a vector."""
    old = physmet.vector(array, unit='m')
    new = numpy.mean(old)
    assert isinstance(new, physmet.Scalar)
    assert numpy.array_equal(new.data, numpy.mean(array))
    assert new.unit == old.unit


def test_sum(array: numpy.ndarray):
    """Test `numpy.sum` of a vector."""
    old = physmet.vector(array, unit='m')
    new = numpy.sum(old)
    assert isinstance(new, physmet.Scalar)
    assert numpy.array_equal(new.data, numpy.sum(array))
    assert new.unit == old.unit


def test_cumsum(array: numpy.ndarray):
    """Test `numpy.cumsum` of a vector."""
    old = physmet.vector(array, unit='m')
    new = numpy.cumsum(old)
    assert isinstance(new, physmet.Vector)
    assert numpy.array_equal(new.data, numpy.cumsum(array))
    assert new.unit == old.unit


def test_transpose(array: numpy.ndarray):
    """Test `numpy.transpose` on a vector."""
    old = physmet.vector(array, unit='cm')
    new = numpy.transpose(old)
    assert isinstance(new, physmet.Vector)
    assert numpy.array_equal(new.data, numpy.transpose(array))
    assert new.unit == old.unit


def test_gradient(array: numpy.ndarray):
    """Test `numpy.gradient` on a vector."""
    vector = physmet.vector(array, unit='m')
    cases = [
        {
            'dt': [],
            'unit': vector.unit,
            'reference': numpy.gradient(array),
        },
        {
            'dt': [0.5],
            'unit': vector.unit,
            'reference': numpy.gradient(array, 0.5),
        },
        {
            'dt': [physmet.scalar(0.5, unit='s')],
            'unit': vector.unit / 's',
            'reference': numpy.gradient(array, 0.5),
        },
    ]
    for case in cases:
        dt = case.get('dt', [])
        gradient = numpy.gradient(vector, *dt, axis=case.get('axis'))
        computed = gradient if isinstance(gradient, list) else [gradient]
        reference = case['reference']
        expected = reference if isinstance(reference, list) else [reference]
        unit = case['unit']
        for this, that in zip(computed, expected):
            assert isinstance(this, physmet.Vector)
            assert numpy.array_equal(this, that)
            assert this.unit == unit


def test_unit():
    """Test the ability to update a vector's unit."""
    old = physmet.vector([2.0, 4.0], 'm')
    new = old.withunit('km')
    assert isinstance(new, physmet.Vector)
    assert new is not old
    assert new.unit == 'km'
    factor = old.unit >> new.unit
    assert numpy.array_equal(new, factor*numpy.array(old))
    with pytest.raises(ValueError):
        old.withunit('J')


