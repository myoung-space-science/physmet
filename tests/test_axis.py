import pytest

import numpy

from physmet import axis
from physmet import indexer
from physmet import measurable
from physmet import measured


def test_points():
    """Test the representation of an integral axis."""
    trivial = [
        range(5),
        tuple(range(5)),
        list(range(5)),
    ]
    for arg in trivial:
        points = axis.points(arg)
        assert points.index(2) == indexer.value(2)
        with pytest.raises(ValueError):
            points.index(6)
    points = axis.points([1, 10, 100, 1000, 10000])
    assert len(points) == 5
    sliced = points[1:4]
    assert sliced == axis.points([10, 100, 1000])
    assert numpy.all(sliced.indices == indexer.sequence(range(1, 4)))
    singular = points[1]
    assert singular == axis.points([10])
    assert numpy.all(singular.indices == indexer.sequence(1))
    assert points.index(10) == indexer.value(1)
    with pytest.raises(ValueError):
        points.index(0)


def test_symbols():
    """Test the representation of a symbolic axis."""
    letters = ['a', 'b0', 'c', 'x', 'z']
    symbols = axis.symbols(letters)
    assert len(symbols) == 5
    sliced = symbols[1:4]
    assert sliced == axis.symbols(letters[1:4])
    assert numpy.all(sliced.indices == indexer.sequence(range(1, 4)))
    singular = symbols[1]
    assert singular == axis.symbols(['b0'])
    assert numpy.all(singular.indices == indexer.sequence(1))
    for index, letter in enumerate(letters):
        assert symbols.index(letter) == indexer.value(index)
        assert symbols.data[index] == letter
    errors = [
        ('A', ValueError),
        ('d', ValueError),
        (None, TypeError),
        (0, TypeError),
    ]
    for target, error in errors:
        with pytest.raises(error):
            symbols.index(target)


def test_coordinates():
    """Test the representation of a measured axis."""
    values = [-1.0, 1.0, 1.5, 2.0, 10.1]
    unit = 'm'
    array = numpy.array(values)
    coordinates = axis.coordinates(array, unit=unit)
    assert len(coordinates) == 5
    sliced = coordinates[1:4]
    assert sliced == axis.coordinates(array[1:4])
    assert numpy.all(sliced.indices == indexer.sequence(range(1, 4)))
    singular = coordinates[1]
    assert singular == axis.coordinates(array[1])
    assert numpy.all(singular.indices == indexer.sequence(1))
    assert coordinates.unit == unit
    for index, value in enumerate(values):
        assert coordinates.index(value) == indexer.value(index)
        assert coordinates.data[index] == value
    expected = indexer.sequence([0, 2])
    assert all(coordinates.index(-100.0, 150.0, 'cm') == expected)
    errors = [
        (-2.0, ValueError),
        (0.5, ValueError),
        (None, TypeError),
    ]
    for target, error in errors:
        with pytest.raises(error):
            coordinates.index(target)
    assert coordinates.index(1.2, closest='lower') == indexer.value(1)
    assert coordinates.index(1.2, closest='upper') == indexer.value(2)
    assert coordinates.withunit('cm').index(150.0) == indexer.value(2)
    measurement = measurable.measure([1.5], unit)
    assert coordinates.index(measurement) == indexer.value(2)

