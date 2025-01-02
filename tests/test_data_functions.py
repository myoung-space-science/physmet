import numpy

import support
from physmet import data
from physmet import indexer
from physmet import measured


def test_isindexlike():
    """Test the function that checks for index-like input.

    Any instance of `~numeric.index.Object` or anything that can initialize an
    instance of `~numeric.index.Object` should test true. Anything else should
    test false.
    """
    true = [
        1,
        '1',
        numpy.array(1, dtype=int),
        range(1, 2),
        [1],
        (1,),
        numpy.array([1], dtype=int),
        range(1, 3),
        [1, 2],
        (1, 2),
        ['1', '2'],
        ('1', '2'),
        numpy.array([1, 2], dtype=int),
        numpy.array([1, 2], ndmin=2),
        numpy.array([1, 2], ndmin=3),
        numpy.array([[1], [2]]),
        indexer.value(1),
        indexer.sequence([1, 2]),
    ]
    for case in true:
        assert data.isindexlike(case)
    false = [
        {1, 2},
        {1: 'a', 2: 'b'},
        slice(1, 3),
        numpy.array([[1, 2], [3, 4]]),
        1.0,
        '1.0',
        numpy.array(1, dtype=float),
        numpy.array([1], dtype=float),
        [1.0, 2],
        (1.0, 2),
        ['1.0', '2'],
        ('1.0', '2'),
        numpy.array([1, 2], dtype=float),
        measured.value(2, 'm'),
        measured.sequence([2, 3], 'm'),
    ]
    for case in false:
        assert not data.isindexlike(case)


def test_nearest():
    values = [0.1, 0.2, 0.3]
    basic = {
        0.11: (0, 0.1),
        0.15: (0, 0.1),
        0.20: (1, 0.2),
    }
    for target, (index, value) in basic.items():
        found = data.nearest(values, target)
        assert found.index == index
        assert found.value == value
    for target in [0.21, 0.25, 0.29]:
        found = data.nearest(values, target, bound='lower')
        assert found.index == 2
        assert found.value == 0.3
        found = data.nearest(values, target, bound='upper')
        assert found.index == 1
        assert found.value == 0.2
    values = numpy.arange(3.0 * 4.0 * 5.0).reshape(3, 4, 5)
    found = data.nearest(values, 32.9)
    assert found.index == (1, 2, 3)
    assert found.value == 33.0


def test_implicitly_measurable(measurables):
    """Test the function that determines if we can measure an object."""
    true = [
        *[case['test'] for case in measurables],
        0,
        1,
    ]
    for this in true:
        assert data.ismeasurable(this)
    false = [
        None,
        '1',
        (),
        [],
        {},
        set(),
        (1, (1, 'm')),
    ]
    for this in false:
        assert not data.ismeasurable(this)


def test_explicitly_measurable(measurables):
    """Test the function that determines if we can measure an object."""
    cases = [case['test'] for case in measurables]
    for case in cases:
        assert data.ismeasurable(case)
    assert data.ismeasurable(support.Measurable(1, unit='m / s'))


