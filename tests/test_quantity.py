import numpy

import support
from physmet import metric
from physmet import quantity
from physmet import symbolic


def test_isnull():
    """Test the function that excludes 0 from truthiness evaluation."""
    assert quantity.isnull(None)
    assert quantity.isnull([])
    assert quantity.isnull(())
    assert quantity.isnull(numpy.array([]))
    assert not quantity.isnull(0)
    assert not quantity.isnull(numpy.zeros((2, 2)))


def test_unitlike():
    """Test the instance check for unit-like objects."""
    valid = [
        'm',
        'm / s',
        'I am not a unit',
        '',
        metric.unit('km^2 * J / erg'),
        symbolic.expression('I am not a unit'),
    ]
    for this in valid:
        assert quantity.isunitlike(this)
    invalid = [
        False,
        True,
        int(3),
        float(3),
        complex(3),
        metric.dimension('L'),
    ]
    for this in invalid:
        assert not quantity.isunitlike(this)


def test_hastype():
    """Test the function that checks for compound type matches."""
    # 1) all identical to `isinstance(...)`
    assert quantity.hastype(1, int)
    assert quantity.hastype('s', str)
    assert quantity.hastype([1, 2], list)
    # 2) targets have declared type and are wrapped in a `list`
    assert quantity.hastype([1, 2], int, list)
    # 3) same as case 2, but no declared wrapper
    assert not quantity.hastype([1, 2], int)
    # 4) same as case 2, but declared wrapper is not `list`
    assert not quantity.hastype([1, 2], int, tuple)
    # 5) similar to case 2, but one target has undeclared type
    assert not quantity.hastype([1, 2.0], int, list, strict=True)
    # 6) non-strict versions of 5
    assert quantity.hastype([1, 2.0], int, list, strict=False)
    assert quantity.hastype([1, '2.0'], int, list, strict=False)
    # 7) similar to cases 5 & 6, with consistent types
    assert quantity.hastype([1, 2.0], (int, float), list, strict=False)
    assert quantity.hastype([1, 2.0], (int, float), list, strict=True)
    # 8) variations on case 7 in which `float` is interpreted as a wrapper type
    #    (may lead to subtle bugs in user code)
    assert not quantity.hastype([1, 2.0], int, float, list, strict=True)
    assert quantity.hastype([1, 2.0], int, float, list)
    # *) indices tested in test_variable.py::test_variable_getitem
    indices = [
        slice(None),
        Ellipsis,
        (0, 0),
        (0, slice(None)),
        (slice(None), 0),
        (slice(None), slice(0, 1, None)),
    ]
    types = (int, slice, type(...))
    for index in indices:
        assert quantity.hastype(index, types, tuple, strict=True)
    assert not quantity.hastype('hello', types, strict=True)


