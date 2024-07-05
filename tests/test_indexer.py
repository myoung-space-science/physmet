import numpy
import numpy.typing
import operator
import pytest

from physmet import indexer


@pytest.fixture
def index_args():
    """Arguments that can initialize an instance of `real.Index`."""
    return [
        # an integral number
        1,
        # an integral numeric string
        '1',
        # a 0-D array
        numpy.array(1, dtype=int),
        # a singular range
        range(1, 2),
        # a single-valued array-like object with integral value
        [1],
        (1,),
        numpy.array([1], dtype=int),
    ]


def test_value_factory(index_args):
    """Test various ways to create an index value."""
    for arg in index_args:
        x = indexer.value(arg)
        assert isinstance(x, indexer.Value)
        assert int(x) == 1
    errors = [
        # numbers must be integral
        (1.0, TypeError),
        # numeric strings must be integral
        ('1.0', TypeError),
        # arrays must have integral type
        (numpy.array([1], dtype=float), TypeError),
        # for consistency with `Indices`
        ({1}, TypeError),
        # a dict is neither numeric nor array-like
        ({1: 'a'}, TypeError),
        # for consistency with `Indices`
        (slice(1, 2), TypeError),
        # an index must have a single value
        (numpy.array([1, 2], dtype=int), TypeError),
        (numpy.array([1, 2], dtype=float), TypeError),
    ]
    for arg, error in errors:
        with pytest.raises(error):
            indexer.value(arg)


def test_value_operators():
    """Test numeric operations on an index value."""
    unary = (
        (operator.abs,  3, indexer.value( 3)),
        (operator.abs, -3, indexer.value( 3)),
        (operator.pos,  3, indexer.value( 3)),
        (operator.pos, -3, indexer.value(-3)),
        (operator.neg,  3, indexer.value(-3)),
        (operator.neg, -3, indexer.value( 3)),
    )
    for f, a, r in unary:
        assert f(indexer.value(a)) == r
    binary = (
        (operator.eq,       3, 3, numpy.array([True])),
        (operator.ne,       3, 2, numpy.array([True])),
        (operator.lt,       3, 4, numpy.array([True])),
        (operator.gt,       3, 2, numpy.array([True])),
        (operator.le,       3, 4, numpy.array([True])),
        (operator.le,       3, 3, numpy.array([True])),
        (operator.ge,       3, 2, numpy.array([True])),
        (operator.ge,       3, 3, numpy.array([True])),
        (operator.add,      3, 3, indexer.value(6)),
        (operator.sub,      3, 3, indexer.value(0)),
        (operator.mul,      3, 3, indexer.value(9)),
        (operator.truediv,  3, 3, 1.0),
        (operator.floordiv, 3, 3, indexer.value(1)),
        (operator.mod,      3, 3, indexer.value(0)),
        (operator.pow,      3, 2, indexer.value(9)),
    )
    for f, a, b, r in binary:
        assert f(a, indexer.value(b)) == r
        assert f(indexer.value(a), b) == r
        assert f(indexer.value(a), indexer.value(b)) == r
    errors = (
        (operator.pow,      3, -3, ValueError),
        (operator.pow,     -3, -3, ValueError),
    )
    for f, a, b, e in errors:
        with pytest.raises(e):
            f(a, indexer.value(b))
        with pytest.raises(e):
            f(indexer.value(a), b)
        with pytest.raises(e):
            f(indexer.value(a), indexer.value(b))


def test_sequence_factory(index_args):
    """Test various ways to create an index sequence."""
    # Anything that can initialize `Value` can initialize `Sequence`
    for arg in index_args:
        x = indexer.sequence(arg)
        assert isinstance(x, indexer.Sequence)
        assert list(x) == [1]
    valid = [
        # a non-singular range
        range(1, 3),
        # a sequence of numeric strings
        ['1', '2'],
        ('1', '2'),
        # a multi-valued array-like object
        [1, 2],
        (1, 2),
        numpy.array([1, 2], dtype=int),
        # ... array may have dimension > 1 if it is logically 1-D
        numpy.array([1, 2], ndmin=2),
        numpy.array([1, 2], ndmin=3),
        numpy.array([[1], [2]]),
    ]
    for arg in valid:
        x = indexer.sequence(arg)
        assert list(x) == [1, 2]
    errors = [
        # numeric sequences must have only integral elements
        ([1.0, 2], TypeError),
        ((1.0, 2), TypeError),
        (['1.0', '2'], TypeError),
        (('1.0', '2'), TypeError),
        # arrays must have integral type
        (numpy.array([1, 2], dtype=float), TypeError),
        # set elements are unordered
        ({1, 2}, TypeError),
        # a dict is neither numeric nor array-like
        ({1: 'a', 2: 'b'}, TypeError),
        # a slice is not iterable
        (slice(1, 3), TypeError),
        # indices must be logically one-dimensional
        (numpy.array([[1, 2], [3, 4]]), TypeError),
    ]
    for arg, error in errors:
        with pytest.raises(error):
            indexer.sequence(arg)


def test_sequence_operators():
    """Test numeric operations on an indices."""
    unary = (
        (operator.abs,  3, indexer.sequence( 3)),
        (operator.abs, -3, indexer.sequence( 3)),
        (operator.pos,  3, indexer.sequence( 3)),
        (operator.pos, -3, indexer.sequence(-3)),
        (operator.neg,  3, indexer.sequence(-3)),
        (operator.neg, -3, indexer.sequence( 3)),
    )
    for f, a, r in unary:
        assert f(indexer.sequence(a)) == r
    binary = (
        (operator.eq,       3, 3, numpy.array([True])),
        (operator.ne,       3, 2, numpy.array([True])),
        (operator.lt,       3, 4, numpy.array([True])),
        (operator.gt,       3, 2, numpy.array([True])),
        (operator.le,       3, 4, numpy.array([True])),
        (operator.le,       3, 3, numpy.array([True])),
        (operator.ge,       3, 2, numpy.array([True])),
        (operator.ge,       3, 3, numpy.array([True])),
        (operator.add,      3, 3, indexer.sequence(6)),
        (operator.sub,      3, 3, indexer.sequence(0)),
        (operator.mul,      3, 3, indexer.sequence(9)),
        (operator.truediv,  3, 3, numpy.array([1.0])),
        (operator.floordiv, 3, 3, indexer.sequence(1)),
        (operator.mod,      3, 3, indexer.sequence(0)),
        (operator.pow,      3, 2, indexer.sequence(9)),
    )
    for f, a, b, r in binary:
        assert f(a, indexer.sequence(b)) == r
        assert f(indexer.sequence(a), b) == r
        assert f(indexer.sequence(a), indexer.sequence(b)) == r
    errors = (
        (operator.pow,      3, -3, ValueError),
        (operator.pow,     -3, -3, ValueError),
    )
    for f, a, b, e in errors:
        with pytest.raises(e):
            f(a, indexer.sequence(b))
        with pytest.raises(e):
            f(indexer.sequence(a), b)
        with pytest.raises(e):
            f(indexer.sequence(a), indexer.sequence(b))


def test_sequence_subscription():
    """Test the behavior of indices when subscripted."""
    this = numpy.arange(10, dtype=int)
    original = indexer.sequence(this)
    sequence = original[1:4]
    assert isinstance(sequence, indexer.Sequence)
    assert numpy.array_equal(sequence.data, this[1:4])
    value = original[4]
    assert isinstance(value, indexer.Value)
    assert value.data == this[4]


def test_index_shift():
    """Test the linear-shift method on all index objects."""
    value = indexer.value(10)
    assert value.shift(+5) == indexer.value(15)
    assert value.shift(+5, ceil=14) == indexer.value(14)
    assert value.shift(+5, ceil=20) == indexer.value(15)
    assert value.shift(-5) == indexer.value(5)
    assert value.shift(-5, floor=6) == indexer.value(6)
    assert value.shift(-5, floor=0) == indexer.value(5)
    sequence = indexer.sequence([10, 11])
    assert all(sequence.shift(+5) == indexer.sequence([15, 16]))
    assert all(sequence.shift(+5, ceil=15) == indexer.sequence([15, 15]))
    assert all(sequence.shift(+5, ceil=20) == indexer.sequence([15, 16]))
    assert all(sequence.shift(-5) == indexer.sequence([5, 6]))
    assert all(sequence.shift(-5, floor=6) == indexer.sequence([6, 6]))
    assert all(sequence.shift(-5, floor=0) == indexer.sequence([5, 6]))


def test_normalize():
    """Test the function that computes array indices."""
    shape = (3, 4, 5)
    assert indexer.normalize(shape, ...) == ...
    assert indexer.normalize(shape, (...,)) == (...,)
    assert indexer.normalize(shape, slice(None)) == slice(None)
    assert indexer.normalize(shape, (slice(None),)) == (slice(None),)
    assert indexer.normalize(shape, (0, 1, 2)) == (0, 1, 2)
    assert numpy.array_equal(
        indexer.normalize(shape, numpy.array([0, 1, 2])),
        numpy.array([0, 1, 2]),
    )
    assert numpy.array_equal(
        indexer.normalize(shape, numpy.array([True, False, True])),
        numpy.array([True, False, True]),
    )
    result = indexer.normalize(
        shape,
        (slice(None), slice(None), (0, 1)),
    )
    expected = (
        numpy.array(range(3)).reshape(3, 1, 1),
        numpy.array(range(4)).reshape(1, 4, 1),
        numpy.array([0, 1]).reshape(1, 1, 2),
    )
    for x, y in zip(result, expected):
        assert numpy.array_equal(x, y)
    result = indexer.normalize(shape, (1, range(3), range(4)))
    expected = (
        numpy.array([1]).reshape(1, 1, 1),
        numpy.array(range(3)).reshape(1, 3, 1),
        numpy.array(range(4)).reshape(1, 1, 4),
    )
    for x, y in zip(result, expected):
        assert numpy.array_equal(x, y)


def test_expand():
    """Test the function that expands array indices."""
    cases = (
        (3, ...,          (slice(None), slice(None), slice(None))),
        (3, slice(None),  (slice(None), slice(None), slice(None))),
        (3, (..., -2, 4), (slice(None), -2, 4)),
        (4, (..., -2, 4), (slice(None), slice(None), -2, 4)),
        (3, (-2, ..., 4), (-2, slice(None), 4)),
        (4, (-2, ..., 4), (-2, slice(None), slice(None), 4)),
        (2, (-2, 4, ...), (-2, 4)),
        (3, (-2, 4, ...), (-2, 4, slice(None))),
        (4, (-2, 4, ...), (-2, 4, slice(None), slice(None))),
        (2, (-2, 4),      (-2, 4)),
        (3, (-2, 4),      (-2, 4)),
        (4, (-2, 4),      (-2, 4)),
        (1, 3,            (3,)),
        (2, 3,            (3, slice(None))),
        (3, 3,            (3, slice(None), slice(None))),
    )
    for ndim, arg, expected in cases:
        assert indexer.expand(ndim, arg) == expected
    errors = (
        (2, (..., -2, 4),     IndexError),
        (2, (-2, ..., 4),     IndexError),
        (4, (1, ..., 2, ...), IndexError),
    )
    for ndim, arg, error in errors:
        with pytest.raises(error):
            indexer.expand(ndim, arg)

