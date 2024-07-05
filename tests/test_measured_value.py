"""
Tests of the `measured.Value` class.
"""

import numpy
import numpy.typing
import pytest

from physmet import measured
from physmet import numeric


def test_factory():
    """Test the ability to create a measured value."""
    value = 1.5
    unit = 'm'
    original = measured.value(value, unit=unit)
    copied = measured.value(original)
    assert copied is not original
    assert copied == original
    valid = [
        [value, unit],
        [measured.sequence([value], unit)],
        [measured.sequence([[value]], unit)],
        [numpy.array([[value]]), unit],
        [[value], unit],
        [[[value]], unit],
    ]
    for args in valid:
        assert measured.value(*args) == original
    errors = (
        (measured.sequence([value, 2*value]), numeric.DataTypeError),
        (measured.sequence([[value, 2*value]]), numeric.DataTypeError),
        (numpy.array([[value, 2*value]]), TypeError),
        ([[value, 2*value]], TypeError),
        (str(value), TypeError),
        ([str(value)], TypeError),
        ([[str(value)]], TypeError),
    )
    for arg, error in errors:
        with pytest.raises(error):
            measured.value(arg)
    with pytest.raises(ValueError):
        measured.value(original, unit=unit)


